/**
 * @file pitz_daily.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Transient standard-k-epsilon counterpart to OpenFOAM pitzDaily.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "PitzDailyMesh.hh"
#include "PitzDailyReplay.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/MomentumSolveException.hh"
#include "equations/turbulence/TurbulenceModel.hh"
#include "parallel/CollectiveConstruction.hh"
#include "parallel/MeshPartitioner.hh"
#include "solvers/IncompressibleIsothermalSolver.hh"
#include "solvers/SolverDiagnosticsOutput.hh"
#include "solvers/SolverProgress.hh"
#include "solvers/SteadyStateSearch.hh"

#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using MeshData = SimpleFluid::Meshes::UnstructuredMeshData;
using Solver = SimpleFluid::IncompressibleIsothermalSolver<Pack>;

/**
 * @brief Read a positive integer from an environment variable.
 *
 * @param name Environment-variable name.
 * @param fallback Value used when the variable is unset.
 * @return Parsed positive value or @p fallback.
 * @throws std::invalid_argument if the configured value is invalid or non-positive.
 * @throws std::out_of_range if the configured value exceeds the integer range.
 */
int positive_environment_integer(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const int value = std::stoi(text);
    if (value <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive.");
    }
    return value;
}

/** @brief Read a non-negative integer from an environment variable. */
int non_negative_environment_integer(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const int value = std::stoi(text);
    if (value < 0)
    {
        throw std::invalid_argument(std::string(name) + " cannot be negative.");
    }
    return value;
}

/**
 * @brief Read a finite positive floating-point value from the environment.
 *
 * @param name Environment-variable name.
 * @param fallback Value used when the variable is unset.
 * @return Parsed positive value or @p fallback.
 * @throws std::invalid_argument if the configured value is invalid, non-finite, or non-positive.
 * @throws std::out_of_range if the configured value exceeds the floating-point range.
 */
double positive_environment_real(const char* name, double fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const double value = std::stod(text);
    if (!(value > 0.0) || !std::isfinite(value))
    {
        throw std::invalid_argument(std::string(name) + " must be finite and positive.");
    }
    return value;
}

/** @brief Read a strict boolean switch from an environment variable. */
bool environment_boolean(const char* name, bool fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
    {
        return fallback;
    }

    const std::string_view value{text};
    if (value == "1" || value == "true" || value == "yes" || value == "on")
    {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off")
    {
        return false;
    }
    throw std::invalid_argument(std::string(name) + " must be one of 0/1, false/true, no/yes, or off/on.");
}

/**
 * @brief Build a coarsened multi-block native mesh for the pitzDaily geometry.
 *
 * @param divisor Coarsening divisor applied to tutorial cell counts.
 * @return Assembled mesh with inlet, outlet, wall, and spanwise boundaries.
 * @throws std::runtime_error if node coordinates cannot be assigned.
 */
SimpleFluid::SP<Mesh> make_pitz_daily_mesh(int divisor, int multiplier)
{
    MeshData source;
    const auto comm = Tpetra::getDefaultComm();
    std::exception_ptr generation_error;
    if (comm->getRank() == 0)
    {
        try
        {
            source = SimpleFluid::detail::pitz_daily_mesh::generate(divisor, multiplier);
        }
        catch (...)
        {
            generation_error = std::current_exception();
        }
    }
    const int local_failed = generation_error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_failed, &any_failed);
    if (any_failed)
    {
        if (generation_error)
            std::rethrow_exception(generation_error);
        throw std::runtime_error("pitzDaily mesh generation failed on rank zero.");
    }
    auto partition = SimpleFluid::MeshPartitioner<Pack>::partition_native(std::move(source), comm, 1, true);
    return SimpleFluid::detail::collective_make_shared<Mesh>(*comm, partition.geometry, partition.indexer, comm);
}

/**
 * @brief Construct velocity, pressure, and turbulence boundary conditions.
 *
 * @return Boundary conditions matching the OpenFOAM pitzDaily case.
 */
SimpleFluid::BoundaryConditionSet pitz_daily_boundary_conditions()
{
    using Type = SimpleFluid::BoundaryConditionType;
    SimpleFluid::BoundaryConditionSet conditions;
    conditions.velocity["inlet"] = {Type::Dirichlet, {10.0, 0.0, 0.0}};
    conditions.velocity["outlet"] = {Type::Neumann, {}};
    conditions.pressure["inlet"] = {Type::Neumann, 0.0};
    conditions.pressure["outlet"] = {Type::Dirichlet, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["inlet"] = {Type::Dirichlet, 0.375};
    conditions.turbulence.dissipation_rate["inlet"] = {Type::Dirichlet, 14.855};

    for (const auto* boundary : {"upperWall", "lowerWall"})
    {
        conditions.velocity[boundary] = {Type::NoSlip, {}};
        conditions.pressure[boundary] = {Type::Neumann, 0.0};
        conditions.turbulence.turbulent_kinetic_energy[boundary] = {Type::Neumann, 0.0};
        conditions.turbulence.dissipation_rate[boundary] = {Type::Neumann, 0.0};
    }

    conditions.velocity["frontAndBack"] = {Type::Slip, {}};
    conditions.pressure["frontAndBack"] = {Type::Neumann, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["frontAndBack"] = {Type::Neumann, 0.0};
    conditions.turbulence.dissipation_rate["frontAndBack"] = {Type::Neumann, 0.0};
    conditions.turbulence.turbulent_kinetic_energy["outlet"] = {Type::Neumann, 0.0};
    conditions.turbulence.dissipation_rate["outlet"] = {Type::Neumann, 0.0};
    return conditions;
}

/**
 * @brief Write rank-local flow and turbulence cell data to CSV.
 *
 * @param mesh Mesh providing owned cell centers.
 * @param solver Solver providing velocity and pressure fields.
 * @param turbulence Turbulence model providing k, epsilon, and eddy viscosity.
 * @param rank MPI rank used in the output filename.
 * @throws std::runtime_error if output cannot be opened or epsilon is unavailable.
 */
void write_cells(const Mesh& mesh, const Solver& solver, const Solver::turbulence_model_type& turbulence, int rank)
{
    const char* configured_prefix = std::getenv("SIMPLEFLUID_PITZ_OUTPUT_PREFIX");
    const std::string prefix = configured_prefix == nullptr ? "simplefluid_cells" : configured_prefix;
    const std::string filename = prefix + "_rank" + std::to_string(rank) + ".csv";
    std::ofstream output(filename);
    if (!output)
    {
        throw std::runtime_error("Cannot open " + filename + " for writing.");
    }

    const auto* epsilon = turbulence.dissipation_rate();
    if (epsilon == nullptr)
    {
        throw std::runtime_error("pitzDaily requires an epsilon-family model.");
    }
    output << "x,y,z,ux,uy,uz,pressure,k,epsilon,nu_t\n";
    output << std::setprecision(17);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(owned);
        const auto center = mesh.cell_centroid(lid);
        const auto velocity = solver.velocity().value(lid);
        output << center.x << ',' << center.y << ',' << center.z << ',' << velocity.x << ',' << velocity.y << ','
               << velocity.z << ',' << solver.pressure().value(lid) << ','
               << turbulence.turbulent_kinetic_energy().value(lid) << ',' << epsilon->value(lid) << ','
               << turbulence.turbulent_kinematic_viscosity().value(lid) << '\n';
    }
}


/** Load a complete retained native snapshot; never interpolate or floor its values. */
void restore_cells(Solver& solver, Solver::turbulence_model_type& turbulence,
    const std::string& prefix, int input_ranks)
{
    const auto mesh = solver.velocity().mesh_ptr();
    SimpleFluid::pitz_replay::Rows rows;
    std::vector<const SimpleFluid::pitz_replay::Row*> matched;
    SimpleFluid::collective_detail::collective_local_validation(*mesh, "pitzDaily replay input", [&]
    {
        for (int rank = 0; rank < input_ranks; ++rank)
        {
            const auto filename = prefix + "_rank" + std::to_string(rank) + ".csv";
            std::ifstream input(filename);
            if (!input) throw std::runtime_error("Cannot open replay file " + filename);
            SimpleFluid::pitz_replay::read(input, rows, filename);
        }
        if (rows.size() != mesh->owned_cell_map()->getGlobalNumElements())
            throw std::invalid_argument("Replay cell count differs from the configured mesh.");
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto lid = static_cast<Pack::local_ordinal_type>(owned);
            const auto center = mesh->cell_centroid(lid);
            const auto it = rows.find(SimpleFluid::pitz_replay::key(center.x, center.y, center.z));
            if (it == rows.end()) throw std::invalid_argument("Replay does not cover a configured cell center.");
            const auto& row = it->second;
            if (std::abs(center.x - row[0]) > 1e-12 || std::abs(center.y - row[1]) > 1e-12 ||
                std::abs(center.z - row[2]) > 1e-12)
                throw std::invalid_argument("Replay cell geometry does not match.");
            matched.push_back(&row);
        }
    });
    Solver::field_type k(mesh, "replay_k"), epsilon(mesh, "replay_epsilon"), nut(mesh, "replay_nut");
    for (size_t owned = 0; owned < matched.size(); ++owned)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(owned);
        const auto& row = *matched[owned];
        solver.velocity().set_owned_value(lid, {row[3], row[4], row[5]});
        solver.pressure().set_owned_value(lid, row[6]);
        k.set_owned_value(lid, row[7]); epsilon.set_owned_value(lid, row[8]); nut.set_owned_value(lid, row[9]);
    }
    solver.velocity().sync_ghosts(); solver.pressure().sync_ghosts();
    turbulence.restore_transported_state(k, epsilon, nut, solver.velocity(), solver.material_properties(),
        solver.reference_density());
}

} // namespace

/**
 * @brief Run the configurable turbulent pitzDaily comparison case.
 *
 * @param argc Argument count passed to Tpetra.
 * @param argv Argument vector passed to Tpetra.
 * @return Process exit code, zero only when the requested run completes or
 *         reaches its configured steady-state criterion.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    const int mesh_divisor = positive_environment_integer("SIMPLEFLUID_PITZ_MESH_DIVISOR", 4);
    const int mesh_multiplier = positive_environment_integer("SIMPLEFLUID_PITZ_MESH_MULTIPLIER", 1);
    if (mesh_multiplier > 2)
    {
        throw std::invalid_argument("SIMPLEFLUID_PITZ_MESH_MULTIPLIER must be 1 or 2.");
    }
    const int steps = positive_environment_integer("SIMPLEFLUID_PITZ_STEPS", 200);
    const double time_step = positive_environment_real("SIMPLEFLUID_PITZ_DT", 1.0e-5);
    const bool search_for_steady_state = environment_boolean("SIMPLEFLUID_PITZ_STEADY_STATE", false);

    auto mesh = make_pitz_daily_mesh(mesh_divisor, mesh_multiplier);
    auto boundary_conditions = pitz_daily_boundary_conditions();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = time_step;
    time_options.steps = steps;
    time_options.kinematic_viscosity = 1.0e-5;
    time_options.velocity_neumann_zero_gradient_stress =
        environment_boolean("SIMPLEFLUID_PITZ_OPENFOAM_OUTLET_STRESS", false);
    time_options.isothermal_modified_pressure =
        environment_boolean("SIMPLEFLUID_PITZ_MODIFIED_PRESSURE", false);
    time_options.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::SIMPLE;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 500;
    const char* configured_preconditioner = std::getenv("SIMPLEFLUID_PITZ_TRANSPORT_PRECONDITIONER");
    const std::string_view transport_preconditioner =
        configured_preconditioner == nullptr ? "none" : configured_preconditioner;
    if (transport_preconditioner != "none" && transport_preconditioner != "ilut")
    {
        throw std::invalid_argument("SIMPLEFLUID_PITZ_TRANSPORT_PRECONDITIONER must be none or ilut.");
    }
    linear_options.preconditioner = SimpleFluid::parse_linear_preconditioner(transport_preconditioner);
    const char* transport_backend = std::getenv("SIMPLEFLUID_PITZ_TRANSPORT_BACKEND");
    if (transport_backend != nullptr)
    {
        const std::string_view value(transport_backend);
        if (value != "gmres" && value != "bicgstab")
            throw std::invalid_argument("SIMPLEFLUID_PITZ_TRANSPORT_BACKEND must be gmres or bicgstab.");
        linear_options.backend = SimpleFluid::parse_linear_solver_backend(value);
    }

    linear_options.tolerance =
        positive_environment_real("SIMPLEFLUID_PITZ_LINEAR_TOLERANCE", 1.0e-9);

    Solver solver(mesh, std::move(boundary_conditions), time_options, linear_options, 1.0);
    const char* pressure_mode = std::getenv("SIMPLEFLUID_PITZ_PRESSURE_MODE");
    const std::string_view selected_pressure_mode = pressure_mode == nullptr ? "gmres-muelu" : pressure_mode;
    if (selected_pressure_mode == "cg-symmetric-muelu")
    {
        auto pressure = solver.pressure_linear_solver_options();
        pressure.backend = SimpleFluid::LinearSolverBackend::Cg;
        pressure.preconditioner = SimpleFluid::LinearPreconditioner::MueLuSymmetric;
        solver.set_pressure_linear_solver_options(pressure);
    }
    else if (selected_pressure_mode != "gmres-muelu")
        throw std::invalid_argument("SIMPLEFLUID_PITZ_PRESSURE_MODE must be gmres-muelu or cg-symmetric-muelu.");


    SimpleFluid::TurbulenceModelOptions turbulence_options;
    turbulence_options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    turbulence_options.initial_turbulent_kinetic_energy = 0.375;
    turbulence_options.initial_dissipation_rate = 14.855;
    turbulence_options.wall_treatment = SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    turbulence_options.wall_options.boundary_names = {"upperWall", "lowerWall"};
    auto& turbulence = solver.configure_turbulence(turbulence_options);

    if (const char* replay_prefix = std::getenv("SIMPLEFLUID_PITZ_REPLAY_PREFIX"))
        restore_cells(solver, turbulence, replay_prefix,
            positive_environment_integer("SIMPLEFLUID_PITZ_REPLAY_RANKS", 3));

    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    if (rank == 0)
    {
        std::cout << "velocity_neumann: "
                  << (time_options.velocity_neumann_zero_gradient_stress ? "zeroGradient_stress" : "traction_free") << '\n';
        std::cout << "pressure_convention: "
                  << (time_options.isothermal_modified_pressure ? "modified" : "mechanical") << '\n';
        const auto pressure = solver.pressure_linear_solver_options();
        std::cout << "pressure_linear_solver: backend=" << SimpleFluid::to_string(pressure.backend)
                  << " preconditioner=" << SimpleFluid::to_string(pressure.preconditioner)
                  << " tolerance=" << pressure.tolerance << '\n';
        std::cout << "transport_linear_solver: backend=" << SimpleFluid::to_string(linear_options.backend)
                  << " preconditioner=" << SimpleFluid::to_string(linear_options.preconditioner) << '\n';
    }
    const char* diagnostics_prefix=std::getenv("SIMPLEFLUID_PITZ_DIAGNOSTICS_PREFIX");
    solver.enable_diagnostic_history(diagnostics_prefix != nullptr);
    const bool diagnostic_only=environment_boolean("SIMPLEFLUID_PITZ_DIAGNOSTIC_ONLY",false);
    const int diagnostics_interval=positive_environment_integer("SIMPLEFLUID_PITZ_DIAGNOSTICS_INTERVAL",10);
    SimpleFluid::SolverDiagnosticsOptions diagnostic_options;
    diagnostic_options.include_cell_data=environment_boolean("SIMPLEFLUID_PITZ_DIAGNOSTIC_FIELDS",false);
    diagnostic_options.include_face_data=diagnostic_options.include_cell_data;
    if(const char* text=std::getenv("SIMPLEFLUID_PITZ_DIAGNOSTIC_COEFFICIENT_DT"))
    {
        size_t used=0;const double value=std::stod(text,&used);
        if(used!=std::string(text).size() || !std::isfinite(value) || value<0)
            throw std::invalid_argument("Diagnostic coefficient dt must be finite and nonnegative");
        diagnostic_options.pressure_coefficient_time_step=value;
    }
    if(diagnostic_only && (!diagnostics_prefix || !std::getenv("SIMPLEFLUID_PITZ_REPLAY_PREFIX")))
        throw std::invalid_argument("Diagnostic-only mode requires a replay and diagnostics output prefix");
    auto write_diagnostics=[&]
    {
        if(!diagnostics_prefix)return;
        const auto result=solver.evaluate_diagnostics(diagnostic_options);
        SimpleFluid::diagnostic_output::write(solver.velocity().mesh(),result,
            std::string(diagnostics_prefix)+"_step"+std::to_string(solver.step_index()));
    };
    write_diagnostics();
    if(diagnostic_only)return EXIT_SUCCESS;
    bool steady_state_reached = false;
    std::optional<std::string> steady_state_failure;
    int rejected_steady_steps = 0;
    std::optional<SimpleFluid::SteadyStateStepStatistics<SimpleFluid::real_t>> final_steady_statistics;
    std::optional<SimpleFluid::real_t> last_requested_transport_linear_tolerance;
    if (!search_for_steady_state)
    {
        SimpleFluid::ProgressStream progress(std::cout);
        if(!diagnostics_prefix)solver.run(steps, progress);
        else
        {
            std::ofstream step_history;
            if(rank==0)
            {
                step_history.open(std::string(diagnostics_prefix)+"_step_history.csv");
                step_history<<std::setprecision(17)<<"step,time_since_replay_s,dt_s,k_clipped_cells,k_maximum_clip_m2_s2,k_integrated_clip_m5_s2,epsilon_clipped_cells,epsilon_maximum_clip_m2_s3,epsilon_integrated_clip_m5_s3\n";
            }
            for(int step=0;step<steps;++step)
            {
                solver.run(1,progress);
                const auto clipping=turbulence.clipping_statistics_by_field();
                if(rank==0)step_history<<solver.step_index()<<','<<solver.time()<<','<<solver.time_step()<<','
                    <<clipping.k.count<<','<<clipping.k.max_correction<<','<<clipping.k.volume_integral<<','
                    <<clipping.secondary.count<<','<<clipping.secondary.max_correction<<','<<clipping.secondary.volume_integral<<'\n';
                if(solver.step_index()%diagnostics_interval==0 || step+1==steps)write_diagnostics();
            }
        }
    }
    else
    {
        SimpleFluid::SteadyStateSearchOptions steady_options;
        steady_options.maximum_steps = steps;
        steady_options.required_consecutive_steps =
            positive_environment_integer("SIMPLEFLUID_PITZ_STEADY_CONSECUTIVE_STEPS", std::min(5, steps));
        steady_options.minimum_steps = positive_environment_integer("SIMPLEFLUID_PITZ_STEADY_MIN_STEPS",
            std::max(1, std::min(20, steps - steady_options.required_consecutive_steps + 1)));
        steady_options.maximum_retries_per_step =
            non_negative_environment_integer("SIMPLEFLUID_PITZ_STEADY_MAX_RETRIES", 4);
        steady_options.rejection_recovery_steps =
            non_negative_environment_integer("SIMPLEFLUID_PITZ_STEADY_REJECTION_RECOVERY_STEPS", 5);
        steady_options.relative_update_tolerance =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_TOLERANCE", 1.0e-4);
        steady_options.minimum_time_step =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_MIN_DT", time_step / 16.0);
        steady_options.maximum_time_step =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_MAX_DT", std::max(time_step, 5.0e-2));
        steady_options.target_courant_number = positive_environment_real("SIMPLEFLUID_PITZ_STEADY_TARGET_COURANT", 0.8);
        steady_options.time_step_growth_factor = positive_environment_real("SIMPLEFLUID_PITZ_STEADY_DT_GROWTH", 1.5);
        steady_options.time_step_reduction_factor =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_DT_REDUCTION", 0.5);
        steady_options.rejection_time_step_safety_factor =
            positive_environment_real("SIMPLEFLUID_PITZ_STEADY_REJECTION_SAFETY", 0.9);
        SimpleFluid::AdaptiveLinearToleranceOptions adaptive_linear_options;
        adaptive_linear_options.final_tolerance = linear_options.tolerance;
        adaptive_linear_options.relaxed_tolerance = positive_environment_real(
            "SIMPLEFLUID_PITZ_STEADY_RELAXED_LINEAR_TOLERANCE",
            std::max(1.0e-6, adaptive_linear_options.final_tolerance));
        adaptive_linear_options.full_accuracy_update_ratio = positive_environment_real(
            "SIMPLEFLUID_PITZ_STEADY_FULL_ACCURACY_UPDATE_RATIO", 10.0);
        const int progress_interval = positive_environment_integer("SIMPLEFLUID_PITZ_STEADY_PROGRESS_INTERVAL", 1);

        turbulence.validate_time_mode(false);
        SimpleFluid::AdaptiveSteadyStateController controller(steady_options, time_step);
        SimpleFluid::AdaptiveLinearToleranceController linear_tolerance_controller(
            adaptive_linear_options, steady_options.relative_update_tolerance);
        auto adaptive_transport_linear_options = solver.linear_solver_options();
        adaptive_transport_linear_options.tolerance =
            linear_tolerance_controller.current_linear_tolerance();
        solver.set_linear_solver_options(adaptive_transport_linear_options);
        SimpleFluid::SteadyStateFieldMonitor<Pack> monitor(
            solver.velocity().mesh_ptr(), 0.0, steady_options.update_scales);
        const auto* epsilon = turbulence.dissipation_rate();
        if (epsilon == nullptr)
        {
            throw std::logic_error("Steady pitzDaily search requires a dissipation-rate turbulence field.");
        }
        monitor.initialize(solver.velocity(), {&turbulence.turbulent_kinetic_energy(), epsilon});
        SimpleFluid::SteadyStateProgressStream progress(std::cout);

        if (rank == 0)
        {
            std::cout << "steady_state_search: enabled=yes max_steps=" << steady_options.maximum_steps
                      << " tolerance=" << steady_options.relative_update_tolerance
                      << " min_steps=" << steady_options.minimum_steps
                      << " consecutive_steps=" << steady_options.required_consecutive_steps
                      << " max_retries=" << steady_options.maximum_retries_per_step
                      << " rejection_recovery_steps=" << steady_options.rejection_recovery_steps
                      << " rejection_safety=" << steady_options.rejection_time_step_safety_factor
                      << " target_Co=" << steady_options.target_courant_number << " dt_range=["
                      << steady_options.minimum_time_step << ',' << steady_options.maximum_time_step << ']'
                      << " transport_linear_tolerance=" << adaptive_linear_options.relaxed_tolerance << "->"
                      << adaptive_linear_options.final_tolerance
                      << " full_accuracy_update_ratio=" << adaptive_linear_options.full_accuracy_update_ratio
                      << " pressure_linear_tolerance=" << solver.pressure_linear_solver_options().tolerance << '\n';
        }

        for (int iteration = 0; iteration < steady_options.maximum_steps; ++iteration)
        {
            const auto requested_transport_linear_tolerance =
                linear_tolerance_controller.current_linear_tolerance();
            const bool steady_sample_eligible =
                linear_tolerance_controller.full_accuracy_requested();
            SimpleFluid::real_t accepted_time_step{};
            bool accepted = false;
            int retries = 0;
            while (!accepted)
            {
                accepted_time_step = solver.time_step();
                try
                {
                    solver.step();
                    accepted = true;
                }
                catch (const SimpleFluid::RetryableMomentumNonconvergence& error)
                {
                    ++rejected_steady_steps;
                    const auto reduced_time_step = controller.rejected_time_step(accepted_time_step);
                    if (rank == 0)
                    {
                        progress.write_retry(iteration + 1,
                            std::min(retries + 1, steady_options.maximum_retries_per_step),
                            steady_options.maximum_retries_per_step, solver.time(), accepted_time_step,
                            reduced_time_step, error.what());
                    }
                    if (retries >= steady_options.maximum_retries_per_step || !(reduced_time_step < accepted_time_step))
                    {
                        steady_state_failure = error.what();
                        break;
                    }
                    ++retries;
                    solver.set_time_step(reduced_time_step);
                }
            }
            if (!accepted)
            {
                break;
            }

            const auto update_rates = monitor.observe(accepted_time_step);
            const bool solver_converged = solver.last_step_statistics().converged;
            const auto next_requested_transport_linear_tolerance =
                linear_tolerance_controller.observe(
                    static_cast<SimpleFluid::real_t>(update_rates.maximum()), solver_converged);
            const auto statistics =
                controller.observe(solver.time(), accepted_time_step, solver.maximum_courant_number(),
                    {static_cast<SimpleFluid::real_t>(update_rates.velocity),
                        static_cast<SimpleFluid::real_t>(update_rates.temperature),
                        static_cast<SimpleFluid::real_t>(update_rates.turbulence)},
                    solver_converged, steady_sample_eligible);
            final_steady_statistics = statistics;
            if(solver.step_index()%diagnostics_interval==0)write_diagnostics();
            last_requested_transport_linear_tolerance = requested_transport_linear_tolerance;
            adaptive_transport_linear_options.tolerance =
                next_requested_transport_linear_tolerance;
            solver.set_linear_solver_options(adaptive_transport_linear_options);
            if (rank == 0 &&
                (statistics.iteration == 1 || statistics.steady || statistics.iteration % progress_interval == 0))
            {
                progress.write(statistics, solver.last_step_statistics(),
                    requested_transport_linear_tolerance,
                    next_requested_transport_linear_tolerance);
            }
            if (statistics.steady)
            {
                steady_state_reached = true;
                break;
            }
            solver.set_time_step(statistics.next_time_step);
        }
    }

    if(search_for_steady_state)write_diagnostics();
    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_turbulence_fields = true;
    solver.write_parallel_solution_vtu("pitz_daily.vtu", output_options);
    write_cells(*mesh, solver, turbulence, rank);

    if (rank == 0)
    {
        std::cout << "pitzDaily: " << mesh->num_owned_cells() << " rank-zero owned cells, divisor=" << mesh_divisor
                  << ", steps=" << solver.step_index() << ", t=" << solver.time() << ", MPI ranks=" << comm->getSize() << ", mesh_multiplier=" << mesh_multiplier
                  << '\n';
        if (search_for_steady_state)
        {
            std::cout << "steady_state_search: reached=" << (steady_state_reached ? "yes" : "no")
                      << ", rejected_steps=" << rejected_steady_steps;
            if (final_steady_statistics)
            {
                const auto& statistics = *final_steady_statistics;
                std::cout << ", steps=" << statistics.iteration
                          << ", final_update_rate=" << statistics.update_rates.maximum()
                          << ", final_max_Co=" << statistics.maximum_courant_number
                          << ", final_dt=" << statistics.time_step;
                if (last_requested_transport_linear_tolerance)
                {
                    std::cout << ", last_transport_linear_tolerance="
                              << *last_requested_transport_linear_tolerance;
                }
            }
            else
            {
                std::cout << ", steps=0";
            }
            if (steady_state_failure)
            {
                std::cout << ", failure=\"" << *steady_state_failure << '\"';
            }
            std::cout << '\n';
        }
    }
    return search_for_steady_state && !steady_state_reached ? EXIT_FAILURE : EXIT_SUCCESS;
}
