/** Solver-integrated uniform thermal expansion, compared with OpenFOAM FV. */
#include "IF97ReferenceWater.hh"
#include "VerificationLinearSolvers.hh"
#include "VerificationMesh.hh"
#include "VerificationParallel.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/BoussinesqSolver.hh"

#include <Tpetra_Core.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Solver = SimpleFluid::BoussinesqSolver<Pack>;
constexpr double dt = 1.0;
constexpr double power = 4.0e5;
// Per original cubic metre, water sensible energy is about 1.25 GJ. Scale
// this subtraction-roundoff bound with total volume, preserving its meaning.
constexpr double energy_tolerance = 5.0e-5;
constexpr int heated_steps = 20;
constexpr int quiet_steps = 5;

void check(double value, double tolerance, const char* what)
{
    if (!std::isfinite(value) || std::abs(value) > tolerance)
    {
        std::ostringstream message;
        message << what << " exceeds tolerance: " << std::scientific << std::setprecision(17)
                << value << " (limit " << tolerance << ')';
        throw std::runtime_error(message.str());
    }
}

int run(const std::string& mode, const std::filesystem::path& output, const std::filesystem::path& water_reference,
    const std::filesystem::path& mesh_file, const SimpleFluid::Verification::LinearSolverControls& linear_controls,
    int requested_steps)
{
    const auto reference = SimpleFluid::Verification::load_if97_reference_water(water_reference);
    const auto& water = reference.liquid;
    const double T0 = water.temperature;
    const double absolute_pressure = water.absolute_pressure;
    const double rho0 = water.density;
    const double cp = water.specific_heat_capacity;
    const double beta = reference.thermal_expansion;
    const auto grid = SimpleFluid::Verification::read_verification_mesh(mesh_file);
    const double initial_height = grid.z.back() - grid.z.front();
    const double area = (grid.x.back() - grid.x.front()) * (grid.y.back() - grid.y.front());
    const double initial_volume = area * initial_height;
    const double initial_mass = rho0 * initial_volume;
    const double volume_scale = initial_volume; // Relative to the original 1 m3 fixture.
    const double length_scale = initial_height;
    if (std::abs(grid.x.front()) > 1e-12 || std::abs(grid.y.front()) > 1e-12 || std::abs(grid.z.front()) > 1e-12)
        throw std::runtime_error("ALE reference box must start at the origin");
    auto geometry = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(grid.coordinates());
    auto mesh = std::make_shared<Mesh>(std::move(geometry));
    const auto communicator = mesh->owned_cell_map()->getComm();
    const SimpleFluid::Verification::ParallelContext parallel(communicator);
    SimpleFluid::BoundaryConditionSet bc;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bc.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        bc.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
        bc.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    bc.velocity["zmax"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    bc.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    SimpleFluid::TimeStepperOptions time;
    time.time_step = dt;
    time.steps = 1;
    time.thermal_diffusivity = water.thermal_diffusivity();
    time.kinematic_viscosity = water.kinematic_viscosity();
    time.thermal_expansion = beta;
    time.gravity_x = time.gravity_y = time.gravity_z = 0.0;
    time.reference_temperature = T0;
    time.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::PISO;
    time.n_pressure_correctors = 2;
    time.n_outer_correctors = 2;
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.0e-13;
    linear.max_iterations = 500;
    SimpleFluid::BoussinesqModelOptions model;
    model.reference_density = model.density = rho0;
    model.specific_heat_capacity = cp;
    model.dynamic_viscosity = water.dynamic_viscosity;
    model.thermal_conductivity = water.thermal_conductivity;
    Solver solver(mesh, bc, time, linear, model);
    // The enlarged 3-D pressure system reaches its explicit residual floor
    // near 1e-13. Keep temperature accuracy and physical conservation gates
    // independent of this pressure-only algebraic stopping criterion.
    auto pressure_linear = solver.pressure_linear_solver_options();
    pressure_linear.tolerance = 1.0e-12;
    solver.set_pressure_linear_solver_options(pressure_linear);
    linear_controls.apply_flow(solver);
    SimpleFluid::MaterialFeedbackOptions material;
    material.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    material.reference_density = material.liquid_density = rho0;
    material.gas_density = 1.0;
    material.reference_temperature = T0;
    material.thermal_expansion = beta;
    // ALE supports this built-in reference-water linearization. A nonlinear
    // IF97 callback would violate its material/energy rollback contract.
    material.reference_dynamic_viscosity = water.dynamic_viscosity;
    material.min_density = 1.0;
    solver.configure_material_feedback(material);
    auto& source = solver.add_temperature_source("uniform_heat", power);
    solver.initialize_linear_temperature({0.0, 0.0, 1.0}, T0, T0);
    SimpleFluid::FreeSurfaceOptions surface;
    surface.enabled = true;
    surface.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
    surface.gravity_axis = SimpleFluid::Dimension::Z;
    surface.range_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.initial_liquid_volume = initial_volume;
    surface.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    surface.vessel.bottom_elevation = 0.0;
    surface.vessel.top_elevation = 2 * initial_height;
    surface.vessel.cross_section_area = area;
    surface.vessel.total_internal_volume = 2 * initial_volume;
    surface.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    surface.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    surface.headspace.ambient_pressure = surface.headspace.initial_pressure = absolute_pressure;
    surface.headspace.initial_temperature = T0;
    surface.ale.top_boundary = "zmax";
    surface.ale.maximum_correctors = 30;
    // A level mismatch implies area*dL of global volume mismatch. Keep the
    // Picard level tighter on the enlarged area so the next pressure target
    // satisfies the unchanged fixed-flux compatibility guard.
    surface.ale.level_absolute_tolerance = 1.0e-13 / length_scale;
    surface.ale.level_relative_tolerance = 0.0;
    surface.ale.relaxation = 1.0;
    if (solver.configure_free_surface(surface) == nullptr)
    {
        throw std::runtime_error("Could not configure solver-integrated ALE.");
    }

    const auto rank_output = parallel.output_directory(output);
    std::filesystem::create_directories(rank_output);
    SimpleFluid::Verification::LinearSolverHistory linear_history(rank_output);
    std::ofstream csv(rank_output / "history.csv");
    std::ofstream spatial(rank_output / "fields.csv");
    spatial.exceptions(std::ios::badbit | std::ios::failbit);
    spatial << std::setprecision(17)
            << "time_s,sample,x_lower_m,x_upper_m,z_lower_m,z_upper_m,temperature_K,density_kg_m3,alpha_g,ux_m_s,uy_m_"
               "s,uz_m_s\n";
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(17)
        << "time_s,sample,temperature_K,level_m,volume_m3,liquid_mass_kg,energy_J,cumulative_heat_J,"
           "mass_residual_kg,energy_balance_residual_J,gcl_residual_m3_per_s,"
           "analytic_temperature_error_K,analytic_level_error_m,density_kg_m3,cp_J_kg_K,mu_Pa_s,k_W_m_K,"
           "nu_m2_s,thermal_diffusivity_m2_s,thermal_expansion_1_K,absolute_pressure_Pa,relative_flux_max_m3_s,relative_flux_per_volume_s\n";
    double cumulative_heat = 0.0;
    double exact_temperature = T0;
    double previous_temperature = T0;
    double previous_level = initial_height;
    int quiet_count = 0;
    const int configured_steps = heated_steps + (mode == "steady" ? quiet_steps : 0);
    const int steps = requested_steps ? requested_steps : configured_steps;
    if (steps < 1 || steps > configured_steps || (requested_steps && mode == "steady"))
        throw std::invalid_argument("--steps is a transient-only smoke limit within the configured run");
    const SimpleFluid::Verification::LoopTimer loop_timer;
    for (int step = 0; step <= steps; ++step)
    {
        const double q = step <= heated_steps ? power : 0.0;
        if (step > 0)
        {
            source.set_enabled(q > 0.0);
            solver.step();
            linear_history.write(step, solver.time(), solver.last_step_statistics());
            // Stable small root of BE: dT [1-beta(Told-T0)-beta*dT] = q*dt/(rho0*cp).
            const double a = 1.0 - beta * (exact_temperature - T0);
            const double b = q * dt / (rho0 * cp);
            exact_temperature += 2.0 * b / (a + std::sqrt(a * a - 4.0 * beta * b));
        }
        double volume = 0.0;
        double energy = 0.0;
        long double energy_accumulator = 0;
        double integrated_density = 0.0;
        double integrated_cp = 0.0;
        double integrated_mu = 0.0;
        double integrated_k = 0.0;
        double maximum_temperature_error = 0.0;
        const double level = solver.free_surface_diagnostics().pool_level;
        const auto& mass_density = solver.liquid_mass_inventory().cellMassInventory();
        const auto& fields = solver.material_properties();
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            const double cell_volume = mesh->cell_volume(cell);
            volume += cell_volume;
            energy_accumulator += static_cast<long double>(mass_density.value(cell)) * cell_volume * cp *
                                  solver.temperature().value(cell);
            integrated_density += cell_volume * fields.density.value(cell);
            integrated_cp += cell_volume * fields.specific_heat_capacity.value(cell);
            integrated_mu += cell_volume * fields.dynamic_viscosity.value(cell);
            integrated_k += cell_volume * fields.thermal_conductivity.value(cell);
            const double z = mesh->cell_centroid(cell).z;
            const auto velocity = solver.velocity().value(cell);
            // This fixture contains liquid only. Export the solved cell velocity,
            // not a velocity reconstructed from the imposed affine mesh motion.
            const auto iz = grid.interval(grid.z, z * initial_height / level);
            const auto center = mesh->cell_centroid(cell);
            const auto ix = grid.interval(grid.x, center.x), iy = grid.interval(grid.y, center.y);
            const auto sample = iz * grid.nx() + ix;
            const double cell_area = (grid.x[ix + 1] - grid.x[ix]) * (grid.y[iy + 1] - grid.y[iy]);
            if (iy == grid.ny() / 2)
                spatial << solver.time() << ',' << sample << ',' << grid.x[ix] << ',' << grid.x[ix + 1] << ','
                        << z - 0.5 * cell_volume / cell_area << ',' << z + 0.5 * cell_volume / cell_area << ','
                        << solver.temperature().value(cell) << ',' << fields.density.value(cell) << ",0," << velocity.x
                        << ',' << velocity.y << ',' << velocity.z << '\n';
            const auto temperature_error = solver.temperature().value(cell) - exact_temperature;
            maximum_temperature_error = std::max(
                maximum_temperature_error, std::isfinite(temperature_error) ? std::abs(temperature_error)
                                                                            : std::numeric_limits<double>::infinity());
        }
        energy = static_cast<double>(energy_accumulator);
        const std::array<double, 6> local_integrals{
            volume, energy, integrated_density, integrated_cp, integrated_mu, integrated_k};
        std::array<double, 6> global_integrals{};
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, static_cast<int>(local_integrals.size()),
            local_integrals.data(), global_integrals.data());
        volume = global_integrals[0];
        energy = global_integrals[1];
        integrated_density = global_integrals[2];
        integrated_cp = global_integrals[3];
        integrated_mu = global_integrals[4];
        integrated_k = global_integrals[5];
        check(parallel.max(maximum_temperature_error), 2.0e-7, "Cell temperature analytic error");
        const double mass = solver.liquid_mass_inventory().totalMass();
        const double temperature = energy / (mass * cp);
        const double exact_level = initial_height / (1.0 - beta * (exact_temperature - T0));
        const double density = integrated_density / volume;
        const double actual_cp = integrated_cp / volume;
        const double mu = integrated_mu / volume;
        const double k = integrated_k / volume;
        double maximum_relative_flux = 0.0;
        if (step > 0)
        {
            cumulative_heat += q * volume * dt;
            const auto& relative_flux = solver.mesh_relative_face_fluxes();
            for (const auto face : relative_flux.owned_face_ids())
            {
                const auto value = relative_flux.value(face);
                maximum_relative_flux = std::max(maximum_relative_flux,
                    std::isfinite(value) ? std::abs(value) : std::numeric_limits<double>::infinity());
            }
            maximum_relative_flux = parallel.max(maximum_relative_flux);
            // Resolved columns can retain weak internal circulation. Bound
            // its domain-volume-normalized transport independently of the
            // unchanged cell continuity, GCL and uniform-temperature gates.
            check(maximum_relative_flux / initial_volume, 2.0e-10, "Relative face flux per reference volume");
            check(parallel.max(solver.planar_ale_diagnostics().continuity.maximum), 2.0e-10,
                "Absolute volume continuity");
        }
        const double mass_residual = mass - initial_mass;
        const double energy_residual = energy - initial_mass * cp * T0 - cumulative_heat;
        const double gcl = step ? parallel.max(solver.planar_ale_diagnostics().maximum_gcl_residual) : 0.0;
        check(mass_residual, 2.0e-10 * volume_scale, "Liquid mass conservation");
        check(energy_residual, energy_tolerance * volume_scale, "Liquid energy conservation");
        check(gcl, 2.0e-11, "Mesh GCL");
        check(level - exact_level, 5.0e-10 * length_scale, "Level analytic error");
        check(volume - area * level, 2.0e-11 * volume_scale, "Mesh volume and pool level closure");
        if (step > heated_steps)
        {
            check(temperature - previous_temperature, 2.0e-8, "Steady temperature change");
            check(level - previous_level, 2.0e-11 * length_scale, "Steady level change");
            ++quiet_count;
        }
        check(solver.time() - step * dt, 1.0e-13, "Accepted physical time");
        csv << solver.time() << ",global," << temperature << ',' << level << ',' << volume << ',' << mass << ','
            << energy << ',' << cumulative_heat << ',' << mass_residual << ',' << energy_residual << ',' << gcl << ','
            << temperature - exact_temperature << ',' << level - exact_level << ',' << density << ',' << actual_cp
            << ',' << mu << ',' << k << ',' << mu / density << ',' << k / (density * actual_cp) << ',' << beta << ','
            << absolute_pressure << ',' << maximum_relative_flux << ',' << maximum_relative_flux / initial_volume << '\n';
        previous_temperature = temperature;
        previous_level = level;
    }
    csv.flush();
    spatial.flush();
    linear_history.flush();
    parallel.write_timing(rank_output, loop_timer.wall_seconds(), loop_timer.cpu_seconds());
    if (mode == "steady" && quiet_count != quiet_steps)
    {
        throw std::runtime_error("Steady state did not satisfy five consecutive source-off steps.");
    }
    std::cout << "planarALE " << (requested_steps ? "partial smoke" : mode) << ": " << steps << " accepted steps, "
              << quiet_count << " source-off convergence checks; wrote " << rank_output / "history.csv" << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    try
    {
        std::string mode = "transient";
        std::filesystem::path output = "planar_ale_comparison";
        std::filesystem::path water_reference = "verification/openfoam/reference_water.properties";
        std::filesystem::path mesh_file = "verification/openfoam/planarALE/mesh.dat";
        SimpleFluid::Verification::LinearSolverControls linear_controls;
        int requested_steps = 0;
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--mode" && i + 1 < argc)
                mode = argv[++i];
            else if (argument == "--output" && i + 1 < argc)
                output = argv[++i];
            else if (argument == "--water-properties" && i + 1 < argc)
                water_reference = argv[++i];
            else if (argument == "--steps" && i + 1 < argc)
                requested_steps = std::stoi(argv[++i]);
            else if (argument == "--mesh-file" && i + 1 < argc)
                mesh_file = argv[++i];
            else if (i + 1 < argc && linear_controls.parse(argument, argv[i + 1]))
                ++i;
            else
                throw std::invalid_argument("Usage: planar_ale_comparison --mode steady|transient --output DIR "
                                            "--water-properties FILE");
        }
        if (mode != "steady" && mode != "transient")
            throw std::invalid_argument("--mode must be steady or transient");
        return run(mode, output, water_reference, mesh_file, linear_controls, requested_steps);
    }
    catch (const std::exception& error)
    {
        std::cerr << "planarALE comparison failed: " << error.what() << '\n';
        return 1;
    }
}
