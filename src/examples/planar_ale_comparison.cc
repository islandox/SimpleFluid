/** Solver-integrated uniform thermal expansion, compared with OpenFOAM FV. */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/BoussinesqSolver.hh"
#include "IF97ReferenceWater.hh"

#include <Tpetra_Core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Solver = SimpleFluid::BoussinesqSolver<Pack>;
constexpr double dt = 1.0;
constexpr double power = 4.0e5;
// Water sensible energy is about 1.25 GJ. This bounds accumulated subtraction
// roundoff while resolving the accepted-volume heating correction (~2 kJ).
constexpr double energy_tolerance = 5.0e-5;
constexpr int heated_steps = 20;
constexpr int quiet_steps = 5;

void check(double value, double tolerance, const char* what)
{
    if (!std::isfinite(value) || std::abs(value) > tolerance)
    {
        throw std::runtime_error(std::string(what) + " exceeds tolerance: " + std::to_string(value));
    }
}

int run(const std::string& mode, const std::filesystem::path& output,
    const std::filesystem::path& water_reference)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
    {
        throw std::runtime_error("The OpenFOAM comparison fixture is serial; run without mpiexec.");
    }
    const auto reference = SimpleFluid::Verification::load_if97_reference_water(water_reference);
    const auto& water = reference.liquid;
    const double T0 = water.temperature;
    const double absolute_pressure = water.absolute_pressure;
    const double rho0 = water.density;
    const double cp = water.specific_heat_capacity;
    const double beta = reference.thermal_expansion;
    SimpleFluid::ArrReal z;
    for (int i = 0; i <= 8; ++i)
    {
        z.push_back(i / 8.0);
    }
    auto geometry = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, z}});
    auto mesh = std::make_shared<Mesh>(std::move(geometry));
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
    surface.initial_liquid_volume = 1.0;
    surface.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    surface.vessel.bottom_elevation = 0.0;
    surface.vessel.top_elevation = 2.0;
    surface.vessel.cross_section_area = 1.0;
    surface.vessel.total_internal_volume = 2.0;
    surface.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    surface.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    surface.headspace.ambient_pressure = surface.headspace.initial_pressure = absolute_pressure;
    surface.headspace.initial_temperature = T0;
    surface.ale.top_boundary = "zmax";
    surface.ale.maximum_correctors = 30;
    surface.ale.level_absolute_tolerance = 1.0e-13;
    surface.ale.level_relative_tolerance = 0.0;
    surface.ale.relaxation = 1.0;
    if (solver.configure_free_surface(surface) == nullptr)
    {
        throw std::runtime_error("Could not configure solver-integrated ALE.");
    }

    std::filesystem::create_directories(output);
    std::ofstream csv(output / "history.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(17)
        << "time_s,sample,temperature_K,level_m,volume_m3,liquid_mass_kg,energy_J,cumulative_heat_J,"
           "mass_residual_kg,energy_balance_residual_J,gcl_residual_m3_per_s,"
           "analytic_temperature_error_K,analytic_level_error_m,density_kg_m3,cp_J_kg_K,mu_Pa_s,k_W_m_K,"
           "nu_m2_s,thermal_diffusivity_m2_s,thermal_expansion_1_K,absolute_pressure_Pa\n";
    double cumulative_heat = 0.0;
    double exact_temperature = T0;
    double previous_temperature = T0;
    double previous_level = 1.0;
    int quiet_count = 0;
    const int steps = heated_steps + (mode == "steady" ? quiet_steps : 0);
    for (int step = 0; step <= steps; ++step)
    {
        const double q = step <= heated_steps ? power : 0.0;
        if (step > 0)
        {
            source.set_enabled(q > 0.0);
            solver.step();
            // Stable small root of BE: dT [1-beta(Told-T0)-beta*dT] = q*dt/(rho0*cp).
            const double a = 1.0 - beta * (exact_temperature - T0);
            const double b = q * dt / (rho0 * cp);
            exact_temperature += 2.0 * b / (a + std::sqrt(a * a - 4.0 * beta * b));
        }
        double volume = 0.0;
        double energy = 0.0;
        double integrated_density = 0.0;
        double integrated_cp = 0.0;
        double integrated_mu = 0.0;
        double integrated_k = 0.0;
        const auto& mass_density = solver.liquid_mass_inventory().cellMassInventory();
        const auto& fields = solver.material_properties();
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            const double cell_volume = mesh->cell_volume(cell);
            volume += cell_volume;
            energy += mass_density.value(cell) * cell_volume * cp * solver.temperature().value(cell);
            integrated_density += cell_volume * fields.density.value(cell);
            integrated_cp += cell_volume * fields.specific_heat_capacity.value(cell);
            integrated_mu += cell_volume * fields.dynamic_viscosity.value(cell);
            integrated_k += cell_volume * fields.thermal_conductivity.value(cell);
            check(solver.temperature().value(cell) - exact_temperature, 2.0e-7, "Cell temperature analytic error");
        }
        const double mass = solver.liquid_mass_inventory().totalMass();
        const double temperature = energy / (mass * cp);
        const double level = solver.free_surface_diagnostics().pool_level;
        const double exact_level = 1.0 / (1.0 - beta * (exact_temperature - T0));
        const double density = integrated_density / volume;
        const double actual_cp = integrated_cp / volume;
        const double mu = integrated_mu / volume;
        const double k = integrated_k / volume;
        if (step > 0)
        {
            cumulative_heat += q * volume * dt;
            const auto& relative_flux = solver.mesh_relative_face_fluxes();
            for (const auto face : relative_flux.owned_face_ids())
            {
                check(relative_flux.value(face), 2.0e-10, "Uniform expansion relative face flux");
            }
            check(solver.planar_ale_diagnostics().continuity.maximum, 2.0e-10, "Absolute volume continuity");
        }
        const double mass_residual = mass - rho0;
        const double energy_residual = energy - rho0 * cp * T0 - cumulative_heat;
        const double gcl = step ? solver.planar_ale_diagnostics().maximum_gcl_residual : 0.0;
        check(mass_residual, 2.0e-10, "Liquid mass conservation");
        check(energy_residual, energy_tolerance, "Liquid energy conservation");
        check(gcl, 2.0e-11, "Mesh GCL");
        check(level - exact_level, 5.0e-10, "Level analytic error");
        check(volume - level, 2.0e-11, "Mesh volume and pool level closure");
        if (step > heated_steps)
        {
            check(temperature - previous_temperature, 2.0e-8, "Steady temperature change");
            check(level - previous_level, 2.0e-11, "Steady level change");
            ++quiet_count;
        }
        check(solver.time() - step * dt, 1.0e-13, "Accepted physical time");
        csv << solver.time() << ",global," << temperature << ',' << level << ',' << volume << ',' << mass << ','
            << energy << ',' << cumulative_heat << ',' << mass_residual << ',' << energy_residual << ',' << gcl << ','
            << temperature - exact_temperature << ',' << level - exact_level << ',' << density << ',' << actual_cp
            << ',' << mu << ',' << k << ',' << mu / density << ',' << k / (density * actual_cp) << ',' << beta << ','
            << absolute_pressure << '\n';
        previous_temperature = temperature;
        previous_level = level;
    }
    if (mode == "steady" && quiet_count != quiet_steps)
    {
        throw std::runtime_error("Steady state did not satisfy five consecutive source-off steps.");
    }
    std::cout << "planarALE " << mode << ": " << steps << " accepted steps, " << quiet_count
              << " source-off convergence checks; wrote " << output / "history.csv" << '\n';
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
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--mode" && i + 1 < argc)
                mode = argv[++i];
            else if (argument == "--output" && i + 1 < argc)
                output = argv[++i];
            else if (argument == "--water-properties" && i + 1 < argc)
                water_reference = argv[++i];
            else
                throw std::invalid_argument("Usage: planar_ale_comparison --mode steady|transient --output DIR "
                                            "--water-properties FILE");
        }
        if (mode != "steady" && mode != "transient")
            throw std::invalid_argument("--mode must be steady or transient");
        return run(mode, output, water_reference);
    }
    catch (const std::exception& error)
    {
        std::cerr << "planarALE comparison failed: " << error.what() << '\n';
        return 1;
    }
}
