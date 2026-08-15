/**
 * @file constant_power_cylinder_vessel.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Cylinder vessel smoke example with uniform fission power,
 *        radiolytic gas, and boiling.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "examples/ExampleRunner.hh"

#include <Tpetra_Core.hpp>

#include <memory>

/**
 * @brief Run the constant-power cylindrical vessel multiphysics smoke case.
 *
 * @param argc Argument count passed through to Tpetra.
 * @param argv Argument vector passed through to Tpetra.
 * @return Process exit code, zero on normal completion.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);

    constexpr SimpleFluid::real_t liquid_density = 1000.0;
    constexpr SimpleFluid::real_t specific_heat_capacity = 4200.0;
    constexpr SimpleFluid::real_t thermal_diffusivity = 1.4e-7;
    constexpr SimpleFluid::real_t thermal_conductivity =
        liquid_density * specific_heat_capacity * thermal_diffusivity;
    constexpr SimpleFluid::real_t wall_heat_flux = 25.0;

    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.5});
    db->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    db->set("radius", SimpleFluid::real_t{0.5});
    db->set("height", SimpleFluid::real_t{1.0});
    db->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{"radial", "zmin", "zmax"});

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["zmin"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    // The transport equation adds k (dT/dn) A to the cell energy balance,
    // so q''/k supplies the same wall-to-fluid heat used by wall boiling.
    bcs.temperature["zmax"] =
        {SimpleFluid::BoundaryConditionType::Neumann,
         wall_heat_flux / thermal_conductivity};
    bcs.temperature["radial"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.velocity["zmin"] =
        {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    bcs.velocity["zmax"] =
        {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    bcs.velocity["radial"] =
        {SimpleFluid::BoundaryConditionType::NoSlip, {}};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-3;
    time_options.steps = 3;
    time_options.kinematic_viscosity = 1.0e-6;
    time_options.thermal_diffusivity = thermal_diffusivity;
    time_options.reference_temperature = 300.0;
    time_options.thermal_expansion = 2.1e-4;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 100;
    linear_options.tolerance = 1.0e-10;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = liquid_density;
    model_options.density = liquid_density;
    model_options.specific_heat_capacity = specific_heat_capacity;
    model_options.dynamic_viscosity = 1.0e-3;
    model_options.thermal_conductivity = thermal_conductivity;

    SimpleFluid::run_boussinesq_example<>(
        db,
        bcs,
        time_options,
        linear_options,
        model_options,
        [=](auto& solver)
        {
            solver.initialize_bottom_hot_top_cold(300.0, 300.0);

            SimpleFluid::FissionPowerSourceOptions fission;
            fission.profile = SimpleFluid::FissionPowerProfile::Constant;
            fission.power_density = 2.5e3;
            solver.configure_fission_power_source(fission);

            SimpleFluid::RadiolyticGasOptions radiolysis;
            radiolysis.mode =
                SimpleFluid::RadiolyticGasMode::IdealGasSource;
            radiolysis.hydrogen_yield_mol_per_j = 2.0e-7;
            radiolysis.max_source_alpha_rate = 10.0;
            radiolysis.reference_pressure = 101325.0;
            solver.configure_radiolytic_gas(radiolysis);

            SimpleFluid::ScalarVoidFractionOptions alpha_options;
            alpha_options.alpha_max = 0.5;
            solver.configure_scalar_void_fraction(alpha_options);

            SimpleFluid::BoilingSourceOptions boiling;
            boiling.enable_bulk_boiling = true;
            boiling.enable_wall_boiling = true;
            boiling.saturation_temperature = 299.99;
            boiling.boiling_time_scale = 5.0e-2;
            boiling.wall_evaporation_fraction = 0.2;
            boiling.wall_heat_flux = wall_heat_flux;
            boiling.wall_boiling_patches = {"zmax"};
            solver.configure_boiling_source(boiling);

            SimpleFluid::MaterialFeedbackOptions feedback;
            feedback.density_mode =
                SimpleFluid::DensityFeedbackMode::BoussinesqVoid;
            feedback.reference_density = 1000.0;
            feedback.liquid_density = 1000.0;
            feedback.gas_density = 1.0;
            feedback.reference_temperature = 300.0;
            feedback.thermal_expansion = 2.1e-4;
            feedback.reference_dynamic_viscosity = 1.0e-3;
            solver.configure_material_feedback(feedback);
        },
        "constant_power_cylinder_vessel.vtu",
        SimpleFluid::SolutionOutputOptions{
            .include_sources = true,
            .include_material_properties = true,
            .include_radiolytic_gas_fields = true});

    return 0;
}
