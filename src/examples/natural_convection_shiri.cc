/**
 * @file natural_convection_shiri.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI-capable Boussinesq counterpart to NC_Tutorial_Shiri.pdf.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "dataclass/Database.hh"
#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/SolverProgress.hh"
#include "solvers/SteadyStateSearch.hh"

#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

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
        return fallback;

    const int value = std::stoi(text);
    if (value <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive.");
    return value;
}

/**
 * @brief Read a non-negative integer from an environment variable.
 */
int non_negative_environment_integer(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
        return fallback;

    const int value = std::stoi(text);
    if (value < 0)
        throw std::invalid_argument(
            std::string(name) + " cannot be negative.");
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
        return fallback;

    const double value = std::stod(text);
    if (!(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument(
            std::string(name) + " must be finite and positive.");
    return value;
}

/**
 * @brief Read a strict boolean switch from an environment variable.
 */
bool environment_boolean(const char* name, bool fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
        return fallback;

    const std::string_view value{text};
    if (value == "1" || value == "true" || value == "yes"
        || value == "on")
    {
        return true;
    }
    if (value == "0" || value == "false" || value == "no"
        || value == "off")
    {
        return false;
    }
    throw std::invalid_argument(
        std::string(name)
        + " must be one of 0/1, false/true, no/yes, or off/on.");
}

/**
 * @brief Identify a failed momentum predictor that is safe to retry.
 *
 * The momentum equation solves into a candidate and publishes velocity only
 * after convergence. A rejected predictor therefore leaves every accepted
 * primary and turbulence field unchanged.
 */
bool retryable_steady_state_failure(const std::runtime_error& error)
{
    const std::string_view message{error.what()};
    return message.find("IncompressibleMomentumEquation")
               != std::string_view::npos
        && message.find("did not converge")
               != std::string_view::npos;
}

/**
 * @brief Read a Belos backend name from an environment variable.
 */
SimpleFluid::LinearSolverBackend environment_linear_solver_backend(
    const char* name,
    SimpleFluid::LinearSolverBackend fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr
        ? fallback
        : SimpleFluid::parse_linear_solver_backend(text);
}

/**
 * @brief Read a linear preconditioner name from an environment variable.
 */
SimpleFluid::LinearPreconditioner environment_linear_preconditioner(
    const char* name,
    SimpleFluid::LinearPreconditioner fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr
        ? fallback
        : SimpleFluid::parse_linear_preconditioner(text);
}

/**
 * @brief Generate uniformly spaced coordinates over an interval.
 *
 * @param lower Lower interval endpoint.
 * @param upper Upper interval endpoint.
 * @param cells Number of cells.
 * @return Coordinate vector containing @p cells plus one edges.
 */
SimpleFluid::ArrReal uniform_edges(double lower, double upper, int cells)
{
    SimpleFluid::ArrReal edges;
    edges.reserve(static_cast<size_t>(cells) + 1);
    for (int edge = 0; edge <= cells; ++edge)
    {
        edges.push_back(
            lower + (upper - lower) * static_cast<double>(edge)
                  / static_cast<double>(cells));
    }
    return edges;
}

/**
 * @brief Generate geometrically graded coordinates over one mesh block.
 *
 * @param lower Lower interval endpoint.
 * @param upper Upper interval endpoint.
 * @param cells Number of cells.
 * @param expansion_ratio Last-cell width divided by first-cell width.
 * @return Coordinate vector containing @p cells plus one edges.
 */
SimpleFluid::ArrReal graded_edges(
    double lower, double upper, int cells, double expansion_ratio)
{
    SimpleFluid::ArrReal edges;
    edges.reserve(static_cast<size_t>(cells) + 1);
    edges.push_back(lower);
    if (cells == 1)
    {
        edges.push_back(upper);
        return edges;
    }

    if (expansion_ratio == 1.0)
        return uniform_edges(lower, upper, cells);

    const double ratio =
        std::pow(expansion_ratio, 1.0 / static_cast<double>(cells - 1));
    const double first_width =
        (upper - lower) * (ratio - 1.0)
        / (std::pow(ratio, cells) - 1.0);
    double width = first_width;
    for (int cell = 0; cell < cells; ++cell)
    {
        edges.push_back(edges.back() + width);
        width *= ratio;
    }
    edges.back() = upper;
    return edges;
}

/**
 * @brief Append a graded block without duplicating its shared first edge.
 */
void append_block(
    SimpleFluid::ArrReal& edges,
    double lower,
    double upper,
    int cells,
    double expansion_ratio)
{
    auto block = graded_edges(
        lower, upper, cells, expansion_ratio);
    edges.insert(edges.end(), block.begin() + 1, block.end());
}

/**
 * @brief Reproduce the two radial blocks in the OpenFOAM Shiri mesh.
 */
SimpleFluid::ArrReal shiri_radial_edges(int cells)
{
    if (cells == 1)
        return uniform_edges(0.075, 0.600, cells);

    const int inner_cells = cells / 2;
    const int outer_cells = cells - inner_cells;
    auto edges = graded_edges(
        0.075, 0.2625, inner_cells, 3.0);
    append_block(
        edges, 0.2625, 0.600, outer_cells, 1.0);
    return edges;
}

/**
 * @brief Reproduce the lower and upper axial blocks in the OpenFOAM mesh.
 */
SimpleFluid::ArrReal shiri_axial_edges(int cells)
{
    if (cells == 1)
        return uniform_edges(0.0, 1.5, cells);

    const int lower_cells = std::max(1, cells / 5);
    const int upper_cells = cells - lower_cells;
    auto edges = graded_edges(
        0.0, 0.120, lower_cells, 2.0);
    append_block(
        edges, 0.120, 1.5, upper_cells, 1.5);
    return edges;
}

/**
 * @brief Write rank-local cylindrical solution profiles for comparison.
 *
 * @tparam Pack Tpetra type pack used by the mesh and solver.
 * @param mesh Mesh providing owned cell centers.
 * @param solver Solver providing temperature, velocity, and pressure fields.
 * @param turbulence Standard k-epsilon state used by the comparison.
 * @param rank MPI rank used in the output filename.
 * @throws std::runtime_error if the output file cannot be opened.
 */
template<class Pack>
void write_profile_cells(
    const SimpleFluid::MeshHandle<Pack>& mesh,
    const SimpleFluid::BoussinesqSolver<Pack>& solver,
    const SimpleFluid::TurbulenceModel<Pack>& turbulence,
    int rank)
{
    const char* configured_prefix = std::getenv("SIMPLEFLUID_SHIRI_OUTPUT_PREFIX");
    const std::string prefix = configured_prefix == nullptr
                             ? "simplefluid_cells"
                             : configured_prefix;
    const std::string filename =
        prefix + "_rank" + std::to_string(rank) + ".csv";
    std::ofstream output(filename);
    if (!output)
        throw std::runtime_error("Cannot open " + filename + " for writing.");

    const auto* epsilon = turbulence.dissipation_rate();
    if (epsilon == nullptr)
        throw std::logic_error(
            "Shiri standard k-epsilon output requires an epsilon field.");
    const auto* buoyancy_production =
        turbulence.buoyancy_production();
    const auto* wall_y_plus =
        turbulence.wall_y_plus();
    if (buoyancy_production == nullptr || wall_y_plus == nullptr)
        throw std::logic_error(
            "Shiri turbulent output requires buoyancy and wall diagnostics.");
    const double turbulent_prandtl =
        turbulence.options().turbulent_prandtl_number;

    output << "r,theta,z,temperature,ur,utheta,uz,pressure,"
              "k,epsilon,nut,alphat,buoyancy_production,wall_y_plus\n";
    output << std::setprecision(17);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto center = mesh.cell_centroid(lid);
        const auto velocity = solver.velocity().value(lid);
        const double theta = std::atan2(center.y, center.x);
        const double cosine = std::cos(theta);
        const double sine = std::sin(theta);
        const double radial_velocity = velocity.x * cosine + velocity.y * sine;
        const double azimuthal_velocity = -velocity.x * sine + velocity.y * cosine;
        const double turbulent_viscosity =
            turbulence.turbulent_kinematic_viscosity().value(lid);
        output << std::hypot(center.x, center.y) << ',' << theta << ','
               << center.z << ',' << solver.temperature().value(lid) << ','
               << radial_velocity << ',' << azimuthal_velocity << ','
               << velocity.z << ',' << solver.pressure().value(lid) << ','
               << turbulence.turbulent_kinetic_energy().value(lid) << ','
               << epsilon->value(lid) << ',' << turbulent_viscosity << ','
               << turbulent_viscosity / turbulent_prandtl << ','
               << buoyancy_production->value(lid) << ','
               << wall_y_plus->value(lid) << '\n';
    }
}

} // namespace

/**
 * @brief Run the configurable MPI annulus natural-convection case.
 *
 * @param argc Argument count passed to Tpetra.
 * @param argv Argument vector passed to Tpetra.
 * @return Process exit code, zero on normal completion.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    using Pack = SimpleFluid::DefaultTpetraTypes;

    const int radial_cells =
        positive_environment_integer("SIMPLEFLUID_SHIRI_NR", 40);
    const int theta_cells =
        positive_environment_integer("SIMPLEFLUID_SHIRI_NTHETA", 20);
    const int axial_cells =
        positive_environment_integer("SIMPLEFLUID_SHIRI_NZ", 100);
    const int steps =
        positive_environment_integer("SIMPLEFLUID_SHIRI_STEPS", 200);
    const double time_step =
        positive_environment_real("SIMPLEFLUID_SHIRI_DT", 2.0e-3);
    const bool search_for_steady_state =
        environment_boolean(
            "SIMPLEFLUID_SHIRI_STEADY_STATE", false);

    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::ANNULUS));
    database->set("R", shiri_radial_edges(radial_cells));
    database->set(
        "Theta", uniform_edges(
            0.0, 0.25 * std::numbers::pi, theta_cells));
    database->set("Z", shiri_axial_edges(axial_cells));
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{
            "rmin", "rmax", "thetamin", "thetamax", "zmin", "zmax"});
    auto mesh = SimpleFluid::MeshFactory(database).build_handle<Pack>();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["rmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 360.0};
    for (const auto* boundary : {"rmax", "zmin", "zmax"})
    {
        bcs.temperature[boundary] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 290.0};
        bcs.velocity[boundary] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    bcs.velocity["rmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    for (const auto* boundary : {"thetamin", "thetamax"})
    {
        bcs.temperature[boundary] = {
            SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        bcs.velocity[boundary] = {
            SimpleFluid::BoundaryConditionType::Slip, {}};
    }

    constexpr double dynamic_viscosity = 1.8e-5;
    constexpr double reference_density = 1.198;
    constexpr double prandtl_number = 0.7;
    const double kinematic_viscosity =
        dynamic_viscosity / reference_density;
    const double thermal_diffusivity =
        kinematic_viscosity / prandtl_number;

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = time_step;
    time_options.steps = steps;
    time_options.thermal_diffusivity = thermal_diffusivity;
    time_options.kinematic_viscosity = kinematic_viscosity;
    time_options.thermal_expansion = 1.0 / 290.0;
    time_options.gravity_z = -9.81;
    time_options.reference_temperature = 290.0;
    time_options.n_pressure_correctors = 2;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 500;
    linear_options.tolerance = 1.0e-9;
    linear_options.backend = environment_linear_solver_backend(
        "SIMPLEFLUID_SHIRI_LINEAR_SOLVER_BACKEND",
        SimpleFluid::LinearSolverBackend::BiCGStab);
    linear_options.preconditioner = environment_linear_preconditioner(
        "SIMPLEFLUID_SHIRI_LINEAR_PRECONDITIONER",
        SimpleFluid::LinearPreconditioner::Jacobi);

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = reference_density;
    model_options.density = reference_density;
    model_options.specific_heat_capacity = 1000.0;
    model_options.dynamic_viscosity = dynamic_viscosity;
    model_options.thermal_conductivity =
        reference_density * 1000.0 * thermal_diffusivity;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, bcs, time_options, linear_options, model_options);
    auto pressure_linear_options =
        solver.pressure_linear_solver_options();
    pressure_linear_options.backend =
        environment_linear_solver_backend(
            "SIMPLEFLUID_SHIRI_PRESSURE_LINEAR_SOLVER_BACKEND",
            SimpleFluid::LinearSolverBackend::BiCGStab);
    pressure_linear_options.preconditioner =
        environment_linear_preconditioner(
            "SIMPLEFLUID_SHIRI_PRESSURE_LINEAR_PRECONDITIONER",
            SimpleFluid::LinearPreconditioner::MueLu);
    pressure_linear_options.reuse_preconditioner =
        pressure_linear_options.preconditioner
            != SimpleFluid::LinearPreconditioner::None;
    solver.set_pressure_linear_solver_options(
        pressure_linear_options);
    solver.initialize_heated_box(290.0, 290.0);

    SimpleFluid::TurbulenceModelOptions turbulence_options;
    turbulence_options.model =
        SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    turbulence_options.initial_turbulent_kinetic_energy = 0.00375;
    turbulence_options.initial_dissipation_rate = 0.00075;
    turbulence_options.turbulent_prandtl_number = 0.85;
    turbulence_options.buoyancy_model =
        SimpleFluid::TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
    turbulence_options.buoyancy_coefficient = 1.0;
    turbulence_options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    turbulence_options.wall_options.boundary_names = {
        "rmin", "rmax", "zmin", "zmax"};
    turbulence_options.wall_options.thermal_wall_law =
        SimpleFluid::TurbulenceThermalWallLaw::Jayatilleke;
    turbulence_options.wall_options
        .thermal_turbulent_prandtl_number = 0.85;
    auto& turbulence =
        solver.configure_turbulence(turbulence_options);

    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    if (rank == 0)
    {
        std::cout
            << "linear_solver: transport="
            << SimpleFluid::to_string(linear_options.backend) << '/'
            << SimpleFluid::to_string(linear_options.preconditioner)
            << ", pressure="
            << SimpleFluid::to_string(pressure_linear_options.backend)
            << '/'
            << SimpleFluid::to_string(
                   pressure_linear_options.preconditioner)
            << '\n';
    }
    bool steady_state_reached = false;
    std::optional<std::string> steady_state_failure;
    int rejected_steady_steps = 0;
    std::optional<
        SimpleFluid::SteadyStateStepStatistics<
            SimpleFluid::real_t>>
        final_steady_statistics;
    if (!search_for_steady_state)
    {
        SimpleFluid::ProgressStream progress(std::cout);
        solver.run(steps, progress);
    }
    else
    {
        SimpleFluid::SteadyStateSearchOptions steady_options;
        steady_options.maximum_steps = steps;
        steady_options.required_consecutive_steps =
            positive_environment_integer(
                "SIMPLEFLUID_SHIRI_STEADY_CONSECUTIVE_STEPS",
                std::min(5, steps));
        steady_options.minimum_steps =
            positive_environment_integer(
                "SIMPLEFLUID_SHIRI_STEADY_MIN_STEPS",
                std::max(
                    1,
                    std::min(
                        20,
                        steps
                            - steady_options
                                  .required_consecutive_steps
                            + 1)));
        steady_options.maximum_retries_per_step =
            non_negative_environment_integer(
                "SIMPLEFLUID_SHIRI_STEADY_MAX_RETRIES", 4);
        steady_options.relative_update_tolerance =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_TOLERANCE",
                1.0e-4);
        steady_options.minimum_time_step =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_MIN_DT",
                time_step / 16.0);
        steady_options.maximum_time_step =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_MAX_DT",
                std::max(time_step, 5.0e-2));
        steady_options.target_courant_number =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_TARGET_COURANT",
                0.8);
        steady_options.time_step_growth_factor =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_DT_GROWTH",
                1.5);
        steady_options.time_step_reduction_factor =
            positive_environment_real(
                "SIMPLEFLUID_SHIRI_STEADY_DT_REDUCTION",
                0.5);

        SimpleFluid::AdaptiveSteadyStateController controller(
            steady_options, time_step);
        SimpleFluid::SteadyStateFieldMonitor<Pack> monitor(
            solver.temperature().mesh_ptr(),
            time_options.reference_temperature,
            steady_options.update_scales);
        const auto* secondary =
            turbulence.dissipation_rate();
        if (secondary == nullptr)
        {
            secondary =
                turbulence.specific_dissipation_rate();
        }
        if (secondary == nullptr)
        {
            throw std::logic_error(
                "Steady turbulent Shiri search requires a secondary "
                "turbulence field.");
        }
        monitor.initialize(
            solver.velocity(),
            solver.temperature(),
            {&turbulence.turbulent_kinetic_energy(),
             secondary});
        SimpleFluid::SteadyStateProgressStream progress(std::cout);

        if (rank == 0)
        {
            std::cout
                << "steady_state_search: enabled=yes max_steps="
                << steady_options.maximum_steps
                << " tolerance="
                << steady_options.relative_update_tolerance
                << " min_steps=" << steady_options.minimum_steps
                << " consecutive_steps="
                << steady_options.required_consecutive_steps
                << " max_retries="
                << steady_options.maximum_retries_per_step
                << " target_Co="
                << steady_options.target_courant_number
                << " dt_range=["
                << steady_options.minimum_time_step << ','
                << steady_options.maximum_time_step << "]\n";
        }

        for (int iteration = 0;
             iteration < steady_options.maximum_steps;
             ++iteration)
        {
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
                catch (const std::runtime_error& error)
                {
                    if (!retryable_steady_state_failure(error))
                        throw;

                    ++rejected_steady_steps;
                    const auto reduced_time_step =
                        controller.rejected_time_step(
                            accepted_time_step);
                    if (rank == 0)
                    {
                        progress.write_retry(
                            iteration + 1,
                            std::min(
                                retries + 1,
                                steady_options
                                    .maximum_retries_per_step),
                            steady_options
                                .maximum_retries_per_step,
                            solver.time(),
                            accepted_time_step,
                            reduced_time_step,
                            error.what());
                    }
                    if (retries
                            >= steady_options
                                .maximum_retries_per_step
                        || !(reduced_time_step
                             < accepted_time_step))
                    {
                        steady_state_failure = error.what();
                        break;
                    }
                    ++retries;
                    solver.set_time_step(reduced_time_step);
                }
            }
            if (!accepted)
                break;

            const auto update_rates =
                monitor.observe(accepted_time_step);
            const auto statistics = controller.observe(
                solver.time(),
                accepted_time_step,
                solver.maximum_courant_number(),
                {static_cast<SimpleFluid::real_t>(
                     update_rates.velocity),
                 static_cast<SimpleFluid::real_t>(
                     update_rates.temperature),
                 static_cast<SimpleFluid::real_t>(
                     update_rates.turbulence)},
                solver.last_step_statistics().converged);
            final_steady_statistics = statistics;
            if (rank == 0)
            {
                progress.write(
                    statistics,
                    solver.last_step_statistics());
            }
            if (statistics.steady)
            {
                steady_state_reached = true;
                break;
            }
            solver.set_time_step(
                statistics.next_time_step);
        }
    }

    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_turbulence_fields = true;
    solver.write_parallel_solution_vtu(
        "natural_convection_shiri.vtu", output_options);
    write_profile_cells(*mesh, solver, turbulence, rank);

    if (rank == 0)
    {
        std::cout << "Shiri annulus: " << radial_cells << 'x' << theta_cells
                  << 'x' << axial_cells << " cells, "
                  << solver.step_index()
                  << " steps, t=" << solver.time() << ", MPI ranks="
                  << comm->getSize() << '\n';
        if (search_for_steady_state)
        {
            std::cout
                << "steady_state_search: reached="
                << (steady_state_reached ? "yes" : "no")
                << ", rejected_steps="
                << rejected_steady_steps;
            if (final_steady_statistics)
            {
                const auto& statistics =
                    *final_steady_statistics;
                std::cout
                    << ", steps=" << statistics.iteration
                    << ", final_update_rate="
                    << statistics.update_rates.maximum()
                    << ", final_max_Co="
                    << statistics.maximum_courant_number
                    << ", final_dt=" << statistics.time_step;
            }
            else
            {
                std::cout << ", steps=0";
            }
            if (steady_state_failure)
            {
                std::cout << ", failure=\""
                          << *steady_state_failure << '"';
            }
            std::cout << '\n';
        }
        for (const auto& statistics :
             turbulence.wall_y_plus_statistics())
        {
            std::cout << "wall_y_plus[" << statistics.boundary_name
                      << "]: min=" << statistics.minimum
                      << ", mean=" << statistics.area_weighted_mean
                      << ", max=" << statistics.maximum
                      << ", faces=" << statistics.global_face_count
                      << '\n';
        }
    }
    return search_for_steady_state && !steady_state_reached
        ? 2
        : 0;
}
