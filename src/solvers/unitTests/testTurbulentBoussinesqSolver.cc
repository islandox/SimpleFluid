/**
 * @file testTurbulentBoussinesqSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief End-to-end tests for Problem-owned turbulence in BoussinesqSolver.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/YPlusBoundaryLayerController.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Expose the pressure-projected transport flux for acceptance checks. */
class InspectableTurbulentBoussinesqSolver : public SimpleFluid::BoussinesqSolver<Pack>
{
public:
    using SimpleFluid::BoussinesqSolver<Pack>::BoussinesqSolver;

    const SimpleFluid::ScalarFaceFieldStored<Pack>& transport_face_fluxes()
    {
        return SimpleFluid::FluidSolver<Pack>::projected_face_fluxes();
    }
};

/** @brief Build slip velocity conditions on every box wall. @return Boundary set. */
SimpleFluid::BoundaryConditionSet slip_box_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    }
    return boundaries;
}

/** @brief Build a box with one no-slip wall and five slip walls. @return Boundary set. */
SimpleFluid::BoundaryConditionSet single_wall_box_boundaries()
{
    auto boundaries = slip_box_boundaries();
    boundaries.velocity["xmin"] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    return boundaries;
}

/** @brief Build a side-heated box with a linear-temperature boundary set. */
SimpleFluid::BoundaryConditionSet buoyant_box_boundaries()
{
    auto boundaries = slip_box_boundaries();
    boundaries.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 2.0};
    boundaries.temperature["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    return boundaries;
}

/**
 * @brief Build stable one-step options for a coupling mode.
 *
 * @param coupling Pressure-velocity coupling algorithm.
 * @return Configured time-step options.
 */
SimpleFluid::TimeStepperOptions stable_time_options(SimpleFluid::PressureVelocityCoupling coupling)
{
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-3;
    options.steps = 1;
    options.thermal_diffusivity = 1.0e-2;
    options.kinematic_viscosity = 1.0e-2;
    options.thermal_expansion = 0.0;
    options.gravity_x = 0.0;
    options.gravity_y = 0.0;
    options.gravity_z = 0.0;
    options.reference_temperature = 1.0;
    options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;
    options.pressure_velocity_coupling = coupling;
    options.n_pressure_correctors = 1;
    return options;
}

/** @brief Build baseline standard k-epsilon options. @return Turbulence options. */
SimpleFluid::TurbulenceModelOptions standard_k_epsilon_options()
{
    SimpleFluid::TurbulenceModelOptions options;
    options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    options.initial_turbulent_kinetic_energy = 0.1;
    options.initial_dissipation_rate = 0.009;
    options.min_turbulent_kinetic_energy = 1.0e-10;
    options.min_dissipation_rate = 1.0e-10;
    return options;
}

/**
 * @brief Initialize a checkerboard shear field in a Boussinesq solver.
 *
 * @param solver Solver whose velocity field is initialized.
 */
void initialize_shear(SimpleFluid::BoussinesqSolver<Pack>& solver)
{
    solver.initialize_heated_box(1.0, 1.0);
    const auto& mesh = solver.velocity().mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        const auto x = mesh.cell_centroid(cell_lid).x;
        const auto y = mesh.cell_centroid(cell_lid).y;
        solver.velocity().set_owned_value(
            cell_lid, {0.0, x < 1.0 ? (y < 1.0 ? 0.2 : -0.2) : (y < 1.0 ? -0.2 : 0.2), 0.0});
    }
    solver.velocity().sync_ghosts();
}

/** @brief Initialize a divergence-free recirculating shear field with nonzero face transport. */
void initialize_transporting_shear(SimpleFluid::BoussinesqSolver<Pack>& solver)
{
    solver.initialize_heated_box(1.0, 1.0);
    constexpr double domain_length = 3.0;
    constexpr double velocity_scale = 0.2;
    const auto wave_number = std::numbers::pi / domain_length;
    const auto& mesh = solver.velocity().mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        const auto centroid = mesh.cell_centroid(cell_lid);
        const auto velocity_x =
            velocity_scale * std::sin(wave_number * centroid.x) * std::cos(wave_number * centroid.y);
        const auto velocity_y =
            -velocity_scale * std::cos(wave_number * centroid.x) * std::sin(wave_number * centroid.y);
        solver.velocity().set_owned_value(cell_lid, {velocity_x, velocity_y, 0.0});
    }
    solver.velocity().sync_ghosts();
}

/**
 * @brief Assert positive finite turbulence and effective-transport fields.
 *
 * @param model Turbulence model to inspect.
 */
void expect_positive_turbulence_fields(const SimpleFluid::BoussinesqSolver<Pack>::turbulence_model_type& model)
{
    ASSERT_NE(model.dissipation_rate(), nullptr);
    ASSERT_EQ(model.specific_dissipation_rate(), nullptr);
    const auto& k = model.turbulent_kinetic_energy();
    const auto& epsilon = *model.dissipation_rate();
    const auto& nu_t = model.turbulent_kinematic_viscosity();
    const auto& mu_eff = model.effective_dynamic_viscosity();
    const auto& lambda_eff = model.effective_thermal_conductivity();
    for (size_t owned = 0; owned < k.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        EXPECT_TRUE(std::isfinite(k.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(epsilon.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(nu_t.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(mu_eff.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(lambda_eff.value(cell_lid)));
        EXPECT_GT(k.value(cell_lid), 0.0);
        EXPECT_GT(epsilon.value(cell_lid), 0.0);
        EXPECT_GT(nu_t.value(cell_lid), 0.0);
        EXPECT_GT(mu_eff.value(cell_lid), 0.0);
        EXPECT_GT(lambda_eff.value(cell_lid), 0.0);
    }
}

/** @brief Sum one rank-local scalar over the field communicator. */
template<class Mesh> double global_sum(const Mesh& mesh, double local_value)
{
    double global_value = 0.0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_value, &global_value);
    return global_value;
}

/** @brief Compute a distributed cell-volume integral. */
template<class Field> double global_integral(const Field& field)
{
    const auto& mesh = field.mesh();
    double local_integral = 0.0;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        local_integral += field.value(cell_lid) * mesh.cell_volume(cell_lid);
    }
    return global_sum(mesh, local_integral);
}

/** @brief Compute a distributed axial first moment of a cell field. */
template<class Field> double global_axial_moment(const Field& field)
{
    const auto& mesh = field.mesh();
    double local_moment = 0.0;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        local_moment += field.value(cell_lid) * mesh.cell_volume(cell_lid) * mesh.cell_centroid(cell_lid).z;
    }
    return global_sum(mesh, local_moment);
}

/** @brief Return the distributed minimum and maximum of a cell field. */
template<class Field> std::pair<double, double> global_range(const Field& field)
{
    double local_minimum = std::numeric_limits<double>::infinity();
    double local_maximum = -std::numeric_limits<double>::infinity();
    for (size_t owned = 0; owned < field.mesh().num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        local_minimum = std::min(local_minimum, static_cast<double>(field.value(cell_lid)));
        local_maximum = std::max(local_maximum, static_cast<double>(field.value(cell_lid)));
    }
    double minimum = 0.0;
    double maximum = 0.0;
    const auto& comm = *field.mesh().owned_cell_map()->getComm();
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local_minimum, &minimum);
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local_maximum, &maximum);
    return {minimum, maximum};
}

/** @brief Return the distributed maximum absolute pressure-projected face flux. */
double global_maximum_flux(const SimpleFluid::ScalarFaceFieldStored<Pack>& flux)
{
    double local_maximum = 0.0;
    for (const auto face_lid : flux.owned_face_ids())
    {
        local_maximum = std::max(local_maximum, std::abs(static_cast<double>(flux.value(face_lid))));
    }
    double maximum = 0.0;
    Teuchos::reduceAll(*flux.mesh().owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum, &maximum);
    return maximum;
}

/** @brief Compute global dissolved plus two-population hydrogen inventory. */
template<class Radiolysis> double global_hydrogen_inventory(const Radiolysis& radiolysis)
{
    return global_integral(radiolysis.dissolved_hydrogen_inventory()) + global_integral(radiolysis.micro_moles()) +
           global_integral(radiolysis.large_moles());
}

/** @brief Check one globally conservative radiolysis update. */
template<class Radiolysis>
void expect_global_hydrogen_balance(const Radiolysis& radiolysis, double hydrogen_before, double expected_produced)
{
    const auto& statistics = radiolysis.last_statistics();
    const auto hydrogen_after = global_hydrogen_inventory(radiolysis);
    const auto balance_tolerance =
        std::max(1.0e-13, 1.0e-9 * (std::abs(hydrogen_before) + std::abs(expected_produced)));
    const auto production_tolerance = std::max(1.0e-14, std::abs(expected_produced) * 1.0e-10);

    EXPECT_NEAR(statistics.hydrogen_before, hydrogen_before, balance_tolerance);
    EXPECT_NEAR(statistics.hydrogen_produced, expected_produced, production_tolerance);
    EXPECT_NEAR(statistics.hydrogen_escaped, 0.0, balance_tolerance);
    EXPECT_NEAR(statistics.hydrogen_after, hydrogen_after, balance_tolerance);
    EXPECT_NEAR(statistics.hydrogen_after + statistics.hydrogen_escaped,
        statistics.hydrogen_before + statistics.hydrogen_produced, balance_tolerance);
    EXPECT_NEAR(statistics.inventory_error, 0.0, balance_tolerance);
    EXPECT_NEAR(statistics.void_volume, global_integral(radiolysis.alpha_g()), balance_tolerance);

    double minimum = 0.0;
    double maximum = 0.0;
    const auto& comm = *radiolysis.alpha_g().mesh().owned_cell_map()->getComm();
    for (auto value : {statistics.hydrogen_before, statistics.hydrogen_produced, statistics.hydrogen_after,
             statistics.inventory_error, statistics.void_volume})
    {
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &value, &minimum);
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &value, &maximum);
        EXPECT_NEAR(maximum, minimum, std::max(1.0e-14, std::abs(maximum) * 1.0e-12));
    }
}

/** @brief Exercise the combined turbulent-radiolysis acceptance path. */
void exercise_combined_rans_radiolysis(const SimpleFluid::SP<MeshType>& mesh)
{
    constexpr double time_step = 1.0e-4;
    auto time_options = stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO);
    time_options.time_step = time_step;
    time_options.steps = 2;
    time_options.reference_temperature = 300.0;
    time_options.gravity_z = -9.81;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 1000.0;
    model_options.density = 1000.0;
    model_options.specific_heat_capacity = 4200.0;
    model_options.dynamic_viscosity = 1.0e-3;
    model_options.thermal_conductivity = 0.6;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-11;
    linear_options.max_iterations = 400;
    InspectableTurbulentBoussinesqSolver solver(
        mesh, slip_box_boundaries(), time_options, linear_options, model_options);
    initialize_transporting_shear(solver);
    solver.temperature().put_scalar(300.0);
    solver.temperature().sync_ghosts();

    auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
    std::vector<double> initial_k(mesh->num_owned_cells());
    std::vector<double> initial_epsilon(mesh->num_owned_cells());
    ASSERT_NE(turbulence.dissipation_rate(), nullptr);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        initial_k[owned] =
            turbulence.turbulent_kinetic_energy().value(cell_lid);
        initial_epsilon[owned] =
            turbulence.dissipation_rate()->value(cell_lid);
    }

    constexpr double total_fission_power = 8.0e7;
    SimpleFluid::FissionPowerSourceOptions fission;
    fission.profile = SimpleFluid::FissionPowerProfile::Gaussian;
    fission.total_power = total_fission_power;
    fission.center = SimpleFluid::vec3<>{0.65, 1.1, 0.65};
    fission.standard_deviation = SimpleFluid::vec3<>{0.55, 0.75, 0.45};
    solver.configure_fission_power_source(fission);

    SimpleFluid::RadiolyticGasOptions radiolysis_options;
    radiolysis_options.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    radiolysis_options.dissolved_transport = SimpleFluid::RadiolyticTransportMode::Advective;
    radiolysis_options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    radiolysis_options.constant_slip_velocity = 0.25;
    radiolysis_options.hydrogen_yield_mol_per_j = 2.0e-7;
    radiolysis_options.gas_release_efficiency = 1.0;
    radiolysis_options.max_source_alpha_rate = 1.0;
    radiolysis_options.reference_pressure = 1.0e5;
    radiolysis_options.henry_coefficient = 1.0e-5;
    radiolysis_options.surface_tension = 0.07;
    radiolysis_options.hydrogen_diffusivity = 1.0e-8;
    radiolysis_options.uranium_concentration_mol_per_m3 = 1000.0;
    radiolysis_options.hydrogen_yield_molecules_per_100_ev = 1.8;
    radiolysis_options.microbubble_lifetime = 1.0e30;
    radiolysis_options.large_bubble_dissolution_time = 1.0e30;
    radiolysis_options.micro_to_large_conversion_coefficient = 0.0;
    radiolysis_options.initial_dissolved_hydrogen = 1.0e-5;
    radiolysis_options.initial_micro_number_density = 1.0e10;
    radiolysis_options.initial_micro_moles = 1.0e-5;
    radiolysis_options.initial_large_number_density = 1.0e8;
    radiolysis_options.initial_large_moles = 5.0e-6;
    radiolysis_options.min_radius = 1.0e-12;
    radiolysis_options.max_radius = 1.0e-3;
    radiolysis_options.min_population = 1.0e-40;
    radiolysis_options.max_population = 1.0e40;
    auto& radiolysis = solver.configure_radiolytic_gas(radiolysis_options);

    SimpleFluid::MaterialFeedbackOptions feedback_options;
    feedback_options.density_mode = SimpleFluid::DensityFeedbackMode::Mixture;
    feedback_options.reference_density = 1000.0;
    feedback_options.liquid_density = 1000.0;
    feedback_options.gas_density = 1.0;
    feedback_options.reference_temperature = 300.0;
    feedback_options.reference_dynamic_viscosity = 1.0e-3;
    auto& feedback = solver.configure_material_feedback(feedback_options);

    InspectableTurbulentBoussinesqSolver zero_slip_solver(
        mesh, slip_box_boundaries(), time_options, linear_options, model_options);
    initialize_transporting_shear(zero_slip_solver);
    zero_slip_solver.temperature().put_scalar(300.0);
    zero_slip_solver.temperature().sync_ghosts();
    zero_slip_solver.configure_turbulence(standard_k_epsilon_options());
    zero_slip_solver.configure_fission_power_source(fission);
    auto zero_slip_options = radiolysis_options;
    zero_slip_options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ZeroSlip;
    zero_slip_options.constant_slip_velocity = 0.0;
    auto& zero_slip_radiolysis = zero_slip_solver.configure_radiolytic_gas(zero_slip_options);
    zero_slip_solver.configure_material_feedback(feedback_options);

    // Matched microbubble kinetics leave axial slip as the centroid discriminator.
    const auto expect_upward_slip_shift = [&]
    {
        EXPECT_GT(global_maximum_flux(zero_slip_solver.transport_face_fluxes()), 1.0e-12);
        const auto slip_inventory = global_integral(radiolysis.micro_moles());
        const auto zero_slip_inventory = global_integral(zero_slip_radiolysis.micro_moles());
        ASSERT_GT(slip_inventory, 0.0);
        ASSERT_GT(zero_slip_inventory, 0.0);
        EXPECT_NEAR(slip_inventory, zero_slip_inventory, std::max(slip_inventory, zero_slip_inventory) * 1.0e-10);
        const auto slip_centroid = global_axial_moment(radiolysis.micro_moles()) / slip_inventory;
        const auto zero_slip_centroid = global_axial_moment(zero_slip_radiolysis.micro_moles()) / zero_slip_inventory;
        EXPECT_GT(slip_centroid - zero_slip_centroid, 1.0e-7);
    };

    ASSERT_EQ(radiolysis.mode(), SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation);
    EXPECT_EQ(radiolysis.options().dissolved_transport, SimpleFluid::RadiolyticTransportMode::Advective);
    EXPECT_EQ(radiolysis.options().rise_velocity_mode, SimpleFluid::BubbleRiseVelocityMode::ConstantSlip);
    EXPECT_GT(radiolysis.options().constant_slip_velocity, 0.0);
    EXPECT_EQ(zero_slip_radiolysis.options().rise_velocity_mode, SimpleFluid::BubbleRiseVelocityMode::ZeroSlip);
    EXPECT_DOUBLE_EQ(zero_slip_radiolysis.options().constant_slip_velocity, 0.0);
    ASSERT_NE(solver.find_fission_power_source(), nullptr);
    EXPECT_NEAR(
        solver.find_fission_power_source()->integrated_power(),
        total_fission_power,
        total_fission_power * 1.0e-12);
    const auto [minimum_power, maximum_power] = global_range(solver.find_fission_power_source()->field());
    EXPECT_GT(maximum_power, minimum_power);

    const auto expect_coupled_state = [&]
    {
        expect_positive_turbulence_fields(turbulence);
        ASSERT_NE(solver.find_scalar_void_fraction_model(), nullptr);
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
            const auto alpha = radiolysis.alpha_g().value(cell_lid);
            EXPECT_TRUE(std::isfinite(alpha));
            EXPECT_GE(alpha, radiolysis_options.alpha_min);
            EXPECT_LE(alpha, radiolysis_options.alpha_max);
            EXPECT_NEAR(radiolysis.alpha_l().value(cell_lid), 1.0 - alpha, 1.0e-14);
            EXPECT_NEAR(solver.find_scalar_void_fraction_model()->alpha_g().value(cell_lid), alpha, 1.0e-14);

            for (const auto value : {radiolysis.dissolved_hydrogen_inventory().value(cell_lid),
                     radiolysis.micro_number_density().value(cell_lid), radiolysis.micro_moles().value(cell_lid),
                     radiolysis.large_number_density().value(cell_lid), radiolysis.large_moles().value(cell_lid)})
            {
                EXPECT_TRUE(std::isfinite(value));
                EXPECT_GE(value, 0.0);
            }

            const auto velocity = solver.velocity().value(cell_lid);
            EXPECT_TRUE(std::isfinite(velocity.x));
            EXPECT_TRUE(std::isfinite(velocity.y));
            EXPECT_TRUE(std::isfinite(velocity.z));
            const auto expected_density =
                feedback_options.liquid_density * (1.0 - alpha) + feedback_options.gas_density * alpha;
            EXPECT_NEAR(solver.material_properties().density.value(cell_lid), expected_density, 1.0e-10);
            EXPECT_NEAR(feedback.density_feedback().value(cell_lid), expected_density, 1.0e-10);
        }
        EXPECT_EQ(radiolysis.last_statistics().clipped_cells, 0);
        EXPECT_EQ(radiolysis.last_statistics().pressure_floor_cells, 0);
        EXPECT_EQ(radiolysis.last_statistics().radius_solver_failures, 0);
    };

    const auto hydrogen_before_fission = global_hydrogen_inventory(radiolysis);
    const auto expected_produced = radiolysis_options.gas_release_efficiency *
                                   radiolysis_options.hydrogen_yield_mol_per_j * total_fission_power * time_step;
    ASSERT_NO_THROW(solver.step());
    ASSERT_TRUE(solver.last_step_statistics().converged);
    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_GE(solver.last_step_statistics().linear_solves, 3);
    ASSERT_NO_THROW(zero_slip_solver.step());
    ASSERT_TRUE(zero_slip_solver.last_step_statistics().converged);
    EXPECT_EQ(zero_slip_solver.step_index(), 1);
    expect_upward_slip_shift();
    expect_global_hydrogen_balance(radiolysis, hydrogen_before_fission, expected_produced);
    expect_coupled_state();

    double local_k_delta = 0.0;
    double local_epsilon_delta = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        const auto volume = mesh->cell_volume(cell_lid);
        local_k_delta +=
            std::abs(
                turbulence.turbulent_kinetic_energy().value(cell_lid)
                - initial_k[owned])
          * volume;
        local_epsilon_delta +=
            std::abs(
                turbulence.dissipation_rate()->value(cell_lid)
                - initial_epsilon[owned])
          * volume;
    }
    EXPECT_GT(global_sum(*mesh, local_k_delta), 1.0e-12);
    EXPECT_GT(global_sum(*mesh, local_epsilon_delta), 1.0e-12);

    EXPECT_GT(global_maximum_flux(solver.transport_face_fluxes()), 1.0e-12);
    double local_minimum_velocity_y = std::numeric_limits<double>::infinity();
    double local_maximum_velocity_y = -std::numeric_limits<double>::infinity();
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        local_minimum_velocity_y = std::min(local_minimum_velocity_y, solver.velocity().value(cell_lid).y);
        local_maximum_velocity_y = std::max(local_maximum_velocity_y, solver.velocity().value(cell_lid).y);
    }
    double minimum_velocity_y = 0.0;
    double maximum_velocity_y = 0.0;
    const auto& comm = *mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local_minimum_velocity_y, &minimum_velocity_y);
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local_maximum_velocity_y, &maximum_velocity_y);
    EXPECT_GT(maximum_velocity_y - minimum_velocity_y, 0.1);
    const auto [minimum_alpha, maximum_alpha] = global_range(radiolysis.alpha_g());
    EXPECT_GT(maximum_alpha, minimum_alpha);
    const auto [minimum_density, maximum_density] = global_range(feedback.density_feedback());
    EXPECT_GT(maximum_density, minimum_density);

    std::vector<double> micro_moles_before_transport(mesh->num_owned_cells());
    std::vector<double> dissolved_before_transport(mesh->num_owned_cells());
    std::vector<double> velocity_z_before_feedback_step(
        mesh->num_owned_cells());
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        micro_moles_before_transport[owned] = radiolysis.micro_moles().value(cell_lid);
        dissolved_before_transport[owned] =
            radiolysis.dissolved_hydrogen_inventory().value(cell_lid);
        velocity_z_before_feedback_step[owned] =
            solver.velocity().value(cell_lid).z;
    }
    const auto global_micro_moles_before_transport = global_integral(radiolysis.micro_moles());
    const auto hydrogen_before_transport = global_hydrogen_inventory(radiolysis);

    solver.find_fission_power_source()->initialize_constant(0.0);
    EXPECT_DOUBLE_EQ(solver.find_fission_power_source()->integrated_power(), 0.0);
    zero_slip_solver.find_fission_power_source()->initialize_constant(0.0);
    EXPECT_DOUBLE_EQ(zero_slip_solver.find_fission_power_source()->integrated_power(), 0.0);
    ASSERT_NO_THROW(solver.step());
    ASSERT_TRUE(solver.last_step_statistics().converged);
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_GE(solver.last_step_statistics().linear_solves, 3);
    ASSERT_NO_THROW(zero_slip_solver.step());
    ASSERT_TRUE(zero_slip_solver.last_step_statistics().converged);
    EXPECT_EQ(zero_slip_solver.step_index(), 2);
    expect_upward_slip_shift();
    expect_global_hydrogen_balance(radiolysis, hydrogen_before_transport, 0.0);
    expect_coupled_state();

    double local_transport_delta = 0.0;
    double local_dissolved_transport_delta = 0.0;
    double local_feedback_velocity_delta = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        local_transport_delta +=
            std::abs(radiolysis.micro_moles().value(cell_lid) - micro_moles_before_transport[owned]) *
            mesh->cell_volume(cell_lid);
        local_dissolved_transport_delta +=
            std::abs(
                radiolysis.dissolved_hydrogen_inventory().value(cell_lid)
                - dissolved_before_transport[owned])
          * mesh->cell_volume(cell_lid);
        local_feedback_velocity_delta +=
            std::abs(
                solver.velocity().value(cell_lid).z
                - velocity_z_before_feedback_step[owned])
          * mesh->cell_volume(cell_lid);
    }
    const auto transport_delta = global_sum(*mesh, local_transport_delta);
    const auto dissolved_transport_delta =
        global_sum(*mesh, local_dissolved_transport_delta);
    const auto feedback_velocity_delta =
        global_sum(*mesh, local_feedback_velocity_delta);
    EXPECT_GT(transport_delta, 1.0e-14);
    EXPECT_GT(dissolved_transport_delta, 1.0e-18);
    EXPECT_GT(feedback_velocity_delta, 1.0e-12);
    EXPECT_NEAR(global_integral(radiolysis.micro_moles()), global_micro_moles_before_transport,
        std::max(1.0e-14, global_micro_moles_before_transport * 1.0e-9));

    InspectableTurbulentBoussinesqSolver no_advection_solver(
        mesh,
        slip_box_boundaries(),
        time_options,
        linear_options,
        model_options);
    initialize_transporting_shear(no_advection_solver);
    no_advection_solver.temperature().put_scalar(300.0);
    no_advection_solver.temperature().sync_ghosts();
    no_advection_solver.configure_turbulence(
        standard_k_epsilon_options());
    no_advection_solver.configure_fission_power_source(fission);
    auto no_advection_options = radiolysis_options;
    no_advection_options.dissolved_transport =
        SimpleFluid::RadiolyticTransportMode::NoAdvection;
    auto& no_advection_radiolysis =
        no_advection_solver.configure_radiolytic_gas(
            no_advection_options);
    no_advection_solver.configure_material_feedback(feedback_options);
    ASSERT_NO_THROW(no_advection_solver.step());
    no_advection_solver.find_fission_power_source()->initialize_constant(0.0);
    ASSERT_NO_THROW(no_advection_solver.step());

    double local_dissolved_advection_effect = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        local_dissolved_advection_effect +=
            std::abs(
                radiolysis.dissolved_hydrogen_inventory().value(cell_lid)
                - no_advection_radiolysis
                      .dissolved_hydrogen_inventory()
                      .value(cell_lid))
          * mesh->cell_volume(cell_lid);
    }
    EXPECT_GT(
        global_sum(*mesh, local_dissolved_advection_effect),
        1.0e-18);
}

} // namespace

/** @brief Verify turbulence is Problem-owned and laminar mode can be restored. */
TEST(TurbulentBoussinesqSolverTest, ConfiguresProblemOwnedModelAndRestoresLaminarMode)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, slip_box_boundaries(), stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO));

    EXPECT_EQ(solver.find_turbulence_model(), nullptr);
    auto& configured = solver.configure_turbulence(standard_k_epsilon_options());
    EXPECT_EQ(solver.find_turbulence_model(), &configured);
    EXPECT_EQ(std::as_const(solver).find_turbulence_model(), &configured);
    EXPECT_TRUE(configured.enabled());
    EXPECT_EQ(configured.type(), SimpleFluid::TurbulenceModelType::StandardKEpsilon);
    EXPECT_TRUE(solver.remove_turbulence_model());
    EXPECT_EQ(solver.find_turbulence_model(), nullptr);
    EXPECT_FALSE(solver.remove_turbulence_model());
}

/** @brief Advance turbulence through segregated and coupled pressure-velocity solves. */
TEST(TurbulentBoussinesqSolverTest, AdvancesTurbulenceThroughSegregatedAndCoupledPressureVelocitySolves)
{
    for (const auto coupling :
        {SimpleFluid::PressureVelocityCoupling::PISO, SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        SCOPED_TRACE("coupling=" + std::to_string(static_cast<int>(coupling)));
        auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-11;
        linear_options.max_iterations = 300;
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh, slip_box_boundaries(), stable_time_options(coupling), linear_options);
        initialize_shear(solver);

        auto& model = solver.configure_turbulence(standard_k_epsilon_options());
        const auto molecular_viscosity = std::as_const(solver).material_properties().dynamic_viscosity.value(0);
        solver.step();

        ASSERT_EQ(solver.step_index(), 1);
        ASSERT_EQ(solver.find_turbulence_model(), &model);
        EXPECT_TRUE(solver.last_step_statistics().converged);
        EXPECT_GE(solver.last_step_statistics().linear_solves, 3);
        EXPECT_DOUBLE_EQ(std::as_const(solver).material_properties().dynamic_viscosity.value(0), molecular_viscosity);
        expect_positive_turbulence_fields(model);
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
            EXPECT_NEAR(model.effective_dynamic_viscosity().value(cell_lid),
                molecular_viscosity + model.turbulent_kinematic_viscosity().value(cell_lid), 1.0e-12);
        }
    }
}

/**
 * @brief Verify that the solver supplies its Boussinesq state to turbulence.
 */
TEST(TurbulentBoussinesqSolverTest, PassesDirectBuoyancyContextThroughSegregatedAndCoupledSolvers)
{
    for (const auto coupling :
        {SimpleFluid::PressureVelocityCoupling::PISO, SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        SCOPED_TRACE("coupling=" + std::to_string(static_cast<int>(coupling)));
        auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
        auto time_options = stable_time_options(coupling);
        time_options.time_step = 1.0e-4;
        time_options.thermal_expansion = 1.0e-2;
        time_options.gravity_x = -1.0;
        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-11;
        linear_options.max_iterations = 400;
        SimpleFluid::BoussinesqSolver<Pack> solver(mesh, buoyant_box_boundaries(), time_options, linear_options);
        solver.initialize_heated_box(2.0, 0.0);

        auto options = standard_k_epsilon_options();
        options.buoyancy_model = SimpleFluid::TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
        auto& model = solver.configure_turbulence(options);
        ASSERT_NE(model.buoyancy_production(), nullptr);
        ASSERT_NO_THROW(solver.step());
        ASSERT_TRUE(solver.last_step_statistics().converged);

        const auto* production = model.buoyancy_production();
        ASSERT_NE(production, nullptr);
        EXPECT_EQ(model.output_fields().at("buoyancy_production"), production);
        double minimum_production = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
            const auto value = production->value(cell_lid);
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_LE(value, 0.0);
            minimum_production = std::min(minimum_production, value);
        }
        double global_minimum_production = 0.0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &minimum_production,
            &global_minimum_production);
        EXPECT_LT(global_minimum_production, 0.0);
    }
}

/** @brief Enable turbulence after a laminar implicit non-orthogonal step. */
TEST(TurbulentBoussinesqSolverTest, EnablesTurbulenceAfterLaminarStepWithImplicitNonOrthogonalAssembly)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
    auto time_options = stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO);
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;
    time_options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Implicit;
    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, slip_box_boundaries(), time_options);
    solver.initialize_heated_box(1.0, 1.0);

    solver.step();
    ASSERT_EQ(solver.step_index(), 1);
    ASSERT_EQ(solver.find_turbulence_model(), nullptr);

    auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
    EXPECT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_EQ(solver.find_turbulence_model(), &turbulence);
    expect_positive_turbulence_fields(turbulence);
}

/** @brief Verify turbulence expands zero-diffusivity physical matrix graphs. */
TEST(TurbulentBoussinesqSolverTest, ExpandsZeroDiffusivityPhysicalGraphsWhenTurbulenceIsEnabled)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
    auto time_options = stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO);
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;
    time_options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid;
    auto model_options = SimpleFluid::BoussinesqModelOptions::legacy_defaults(time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, slip_box_boundaries(), time_options, {}, model_options);
    solver.initialize_heated_box(1.0, 1.0);

    solver.step();
    ASSERT_EQ(solver.step_index(), 1);

    auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
    EXPECT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_EQ(solver.find_turbulence_model(), &turbulence);
    expect_positive_turbulence_fields(turbulence);
}

/** @brief Exercise all wall-treatment/closure pairings through both solvers. */
TEST(TurbulentBoussinesqSolverTest, AdvancesWallTreatmentClosurePairingsThroughSegregatedAndCoupledSolvers)
{
    /** @brief Turbulence model and compatible wall treatment under test. */
    struct WallCase
    {
        SimpleFluid::TurbulenceModelType model;
        SimpleFluid::TurbulenceWallTreatmentType treatment;
    };
    const WallCase wall_cases[] = {{SimpleFluid::TurbulenceModelType::StandardKEpsilon,
                                       SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon},
        {SimpleFluid::TurbulenceModelType::StandardKEpsilon,
            SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReKEpsilon},
        {SimpleFluid::TurbulenceModelType::RealizableKEpsilon,
            SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReKEpsilon},
        {SimpleFluid::TurbulenceModelType::SSTKOmega, SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReSST}};

    for (const auto coupling :
        {SimpleFluid::PressureVelocityCoupling::PISO, SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        for (const auto wall_case : wall_cases)
        {
            SCOPED_TRACE("coupling=" + std::to_string(static_cast<int>(coupling)) +
                         ", wall=" + std::string(SimpleFluid::to_string(wall_case.treatment)));
            auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
            auto time_options = stable_time_options(coupling);
            time_options.time_step = 1.0e-5;
            SimpleFluid::LinearSolverOptions linear_options;
            linear_options.tolerance = 1.0e-10;
            linear_options.max_iterations = 400;
            SimpleFluid::BoussinesqSolver<Pack> solver(
                mesh, single_wall_box_boundaries(), time_options, linear_options);
            initialize_shear(solver);

            SimpleFluid::TurbulenceModelOptions options;
            options.model = wall_case.model;
            options.initial_turbulent_kinetic_energy = 0.1;
            options.initial_dissipation_rate = 0.009;
            options.initial_specific_dissipation_rate = 2.0;
            options.min_turbulent_kinetic_energy = 1.0e-10;
            options.min_dissipation_rate = 1.0e-10;
            options.min_specific_dissipation_rate = 1.0e-10;
            options.wall_treatment = wall_case.treatment;
            options.wall_options.boundary_names = {"xmin"};
            if (wall_case.model == SimpleFluid::TurbulenceModelType::SSTKOmega)
                options.initial_wall_distance = 0.5;

            auto& model = solver.configure_turbulence(options);
            EXPECT_NO_THROW(solver.step());
            EXPECT_TRUE(solver.last_step_statistics().converged);
            EXPECT_EQ(solver.step_index(), 1);
            ASSERT_NE(model.wall_y_plus(), nullptr);
            ASSERT_NE(model.effective_dynamic_viscosity_boundary_cache(), nullptr);
            ASSERT_NE(model.effective_thermal_conductivity_boundary_cache(), nullptr);
            const auto& wall_statistics = model.wall_y_plus_statistics();
            ASSERT_EQ(wall_statistics.size(), 1U);
            EXPECT_EQ(wall_statistics.front().boundary_name, "xmin");
            EXPECT_EQ(wall_statistics.front().global_face_count, 4U);
            EXPECT_GT(wall_statistics.front().maximum, 0.0);
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto lid = static_cast<MeshType::local_ordinal_type>(owned);
                EXPECT_TRUE(std::isfinite(model.turbulent_kinetic_energy().value(lid)));
                EXPECT_GT(model.turbulent_kinetic_energy().value(lid), 0.0);
                const auto* secondary = wall_case.model == SimpleFluid::TurbulenceModelType::SSTKOmega
                                            ? model.specific_dissipation_rate()
                                            : model.dissipation_rate();
                ASSERT_NE(secondary, nullptr);
                EXPECT_TRUE(std::isfinite(secondary->value(lid)));
                EXPECT_GT(secondary->value(lid), 0.0);
            }
            if (wall_case.model == SimpleFluid::TurbulenceModelType::SSTKOmega)
            {
                SimpleFluid::YPlusBoundaryLayerControllerOptions controller_options;
                controller_options.target_y_plus = 0.5 * wall_statistics.front().maximum;
                controller_options.adaptation_exponent = 1.0;
                controller_options.minimum_height_ratio = 0.1;
                controller_options.relative_tolerance = 0.0;
                const SimpleFluid::YPlusBoundaryLayerController controller(controller_options);
                const auto update = controller.update_layer_specs({{"xmin", 4, 0.5, 1.2}}, wall_statistics);
                ASSERT_EQ(update.layer_specs.size(), 1U);
                EXPECT_NEAR(update.layer_specs.front().first_cell_height, 0.25, 1.0e-14);
            }
        }
    }
}

/** @brief Verify turbulence fields are emitted only when VTU output requests them. */
TEST(TurbulentBoussinesqSolverTest, TurbulenceFieldsAreOptInForSolutionOutput)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, slip_box_boundaries(), stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO));
    solver.initialize_heated_box(1.0, 1.0);
    solver.configure_turbulence(standard_k_epsilon_options());

    const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path();
    const auto default_file = directory / ("SimpleFluid_laminar_output_" + std::to_string(unique_id) + ".vtu");
    const auto turbulence_file = directory / ("SimpleFluid_turbulence_output_" + std::to_string(unique_id) + ".vtu");
    solver.write_solution_vtu(default_file.string());
    solver.write_solution_vtu(turbulence_file.string(), {.include_turbulence_fields = true});

    auto read = [](const std::filesystem::path& path)
    {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    const auto default_contents = read(default_file);
    const auto turbulence_contents = read(turbulence_file);
    EXPECT_EQ(default_contents.find("Name=\"nu_t\""), std::string::npos);
    EXPECT_EQ(default_contents.find("Name=\"epsilon\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"k\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"epsilon\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"nu_t\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"mu_eff\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"lambda_eff\""), std::string::npos);

    std::filesystem::remove(default_file);
    std::filesystem::remove(turbulence_file);
}

/** @brief Accept combined sheared RANS, Sheng transport, fission, and density feedback in serial. */
TEST(TurbulentBoussinesqSolverTest, CouplesShearedRansShengTransportFissionAndDensityFeedback)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(3, 3, 2));
    if (mesh->owned_cell_map()->getComm()->getSize() != 1)
    {
        GTEST_SKIP() << "This acceptance test is the serial counterpart.";
    }
    exercise_combined_rans_radiolysis(mesh);
}

/** @brief Accept the same combined path with globally conservative two-rank state. */
TEST(TurbulentBoussinesqSolverTest, CouplesShearedRansShengTransportFissionAndDensityFeedbackOnTwoRanks)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(3, 3, 2));
    if (mesh->owned_cell_map()->getComm()->getSize() != 2)
    {
        GTEST_SKIP() << "This acceptance test requires exactly two MPI ranks.";
    }
    exercise_combined_rans_radiolysis(mesh);
}
