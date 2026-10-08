/**
 * @file testRadiolyticGasModelMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI consistency tests for the radiolytic gas model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/RadiolyticGasModel.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <stdexcept>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;
using FaceFieldType = SimpleFluid::FaceField<Pack>;
using RadiolyticModelType = SimpleFluid::RadiolyticGasModel<Pack>;

thread_local const SimpleFluid::Meshes::MultiRegionMesh* gas_transport_guard_probe = nullptr;

double gas_diffusivity_with_transport_guard_probe(double)
{
    if (gas_transport_guard_probe) gas_transport_guard_probe->require_geometry_writable();
    return 1e-8;
}

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/**
 * @brief Sheng two-population options used by distributed inventory tests.
 *
 * @return Configured Sheng-model options.
 */
SimpleFluid::RadiolyticGasOptions sheng_options()
{
    SimpleFluid::RadiolyticGasOptions options;
    options.mode =
        SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.hydrogen_yield_mol_per_j = 2.0e-7;
    options.gas_release_efficiency = 1.0;
    options.max_source_alpha_rate = 1.0;
    options.reference_pressure = 1.0e5;
    options.henry_coefficient = 1.0e-5;
    options.surface_tension = 0.07;
    options.hydrogen_diffusivity = 1.0e-8;
    options.nucleation_radius = 6.6359547089482127e-8;

    options.min_radius = 1.0e-12;
    options.max_radius = 1.0e-3;
    options.min_population = 1.0e-40;
    options.max_population = 1.0e40;
    return options;
}

/**
 * @brief Create water-like material fields on the distributed test mesh.
 *
 * @param mesh Mesh owning the material fields.
 * @return Initialized water-property fields.
 */
SimpleFluid::MaterialPropertyFields<Pack> make_water_properties(
    const SimpleFluid::SP<MeshType>& mesh)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1000.0;
    options.density = 1000.0;
    options.specific_heat_capacity = 4200.0;
    options.dynamic_viscosity = 1.0e-3;
    options.thermal_conductivity = 0.6;
    return {mesh, options, time_options};
}

/**
 * @brief Sum a scalar diagnostic over all ranks in the mesh communicator.
 *
 * @param mesh Mesh providing the communicator.
 * @param local_value Local rank contribution.
 * @return Global sum across ranks.
 */
double global_sum(const MeshType& mesh, double local_value)
{
    double global_value = 0.0;
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_SUM,
        1,
        &local_value,
        &global_value);
    return global_value;
}

/**
 * @brief Compute a distributed cell-volume integral for a cell field.
 *
 * @param field Cell field to integrate.
 * @return Global volume integral.
 */
double global_integral(const FieldType& field)
{
    double local_integral = 0.0;
    const auto& mesh = field.mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        local_integral +=
            field.value(cell_lid) * mesh.cell_volume(cell_lid);
    }
    return global_sum(mesh, local_integral);
}

/**
 * @brief Verify that a replicated diagnostic has the same value on every rank.
 *
 * @param mesh Mesh providing the communicator.
 * @param value Replicated value to compare.
 */
void expect_same_on_all_ranks(const MeshType& mesh, double value)
{
    double minimum = 0.0;
    double maximum = 0.0;
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_MIN,
        1,
        &value,
        &minimum);
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_MAX,
        1,
        &value,
        &maximum);
    EXPECT_NEAR(
        maximum,
        minimum,
        std::max(1.0e-14, std::abs(maximum) * 1.0e-12));
}

} // namespace

/** @brief A rank-local mutation probe fails inside leased gas transport, then
 * all ranks release the lease and can roll back/replay real composite ALE.
 */
TEST(RadiolyticGasModelMultiRankTest, CompositeTransportLeaseFailureIsCollectiveAndReleasesForALE)
{
    using Handle = SimpleFluid::MeshHandle<Pack>;
    using Composite = SimpleFluid::Meshes::MultiRegionMesh;
    using Scalar = SimpleFluid::ScalarCellFieldStored<Pack, Handle>;
    using Vector = SimpleFluid::VectorCellFieldStored<Pack, Handle>;
    using Face = SimpleFluid::ScalarFaceFieldStored<Pack, Handle>;
    using Model = SimpleFluid::RadiolyticGasModel<Pack, Handle>;
    auto geometry = std::make_shared<Composite>(std::vector<Composite::Region>{
        SimpleFluid::Meshes::cartesian_region("left", {{{0, .5}, {0, 1}, {0, .5, 1}}}),
        SimpleFluid::Meshes::cartesian_region("right", {{{.5, 1}, {0, 1}, {0, .5, 1}}})},
        std::vector<Composite::Interface>{SimpleFluid::Meshes::StructuredPatchInterface{{0, 1}, {1, 0}}},
        SimpleFluid::Meshes::InterfaceTolerance{}, Composite::BoundaryNamePolicy::MergeMatchingNames);
    auto mesh = std::make_shared<Handle>(geometry);
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    const int has_cells = mesh->num_owned_cells() > 0;
    int all_have_cells = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &has_cells, &all_have_cells);
    ASSERT_EQ(all_have_cells, 1);
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = .2;
    options.initial_micro_number_density = 1e8;
    options.initial_micro_moles = 1e-6;
    options.microbubble_lifetime = 1e100;
    options.micro_to_large_conversion_coefficient = 0;
    options.diffusivity_mode = SimpleFluid::HydrogenDiffusivityMode::External;
    options.hydrogen_diffusivity_correlation = &gas_diffusivity_with_transport_guard_probe;
    Model model(mesh, options);
    model.enable_donor_hydrogen_deficit_tracking();
    Scalar temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"), power(mesh, 0., "power");
    Vector velocity(mesh, Handle::Vec3{}, "velocity");
    Face relative_flux(mesh, 0., "relative_flux");
    SimpleFluid::BoussinesqModelOptions water;
    water.reference_density = water.density = 1000.;
    water.specific_heat_capacity = 4200.;
    water.dynamic_viscosity = .001;
    water.thermal_conductivity = .6;
    SimpleFluid::MaterialPropertyFields<Pack, Handle> material(mesh, water, SimpleFluid::TimeStepperOptions{});
    model.initialize_state(0., temperature, pressure, velocity, material);
    const auto checkpoint = model.snapshot();
    const double before = model.global_submerged_hydrogen_moles();
    gas_transport_guard_probe = comm->getRank() == 0 ? geometry.get() : nullptr;
    int category = 0;
    std::string message;
    try { model.advance(.01, .01, temperature, pressure, velocity, relative_flux, material, &power); }
    catch (const std::logic_error& error) { category = 1; message = error.what(); }
    catch (const std::runtime_error& error) { category = 2; message = error.what(); }
    catch (const std::exception& error) { category = 3; message = error.what(); }
    gas_transport_guard_probe = nullptr;
    EXPECT_EQ(category, comm->getRank() == 0 ? 1 : 2);
    EXPECT_EQ(message, comm->getRank() == 0
        ? "Geometry is held by a region execution view."
        : "Radiolytic diffusivity evaluation failed on another rank.");
    EXPECT_NO_THROW(geometry->require_geometry_writable());
    model.restore(checkpoint);

    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    const auto advance = [&]
    {
        motion.begin_trial(1.1, .01);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
        model.refresh_geometry();
        model.advance(.01, .01, temperature, pressure, velocity, relative_flux, material, &power,
            &ale, SimpleFluid::Dimension::Z);
        EXPECT_GE(model.last_statistics().transport_linear.solves, 6);
        EXPECT_NEAR(model.global_submerged_hydrogen_moles(), before, 1e-12);
        EXPECT_NEAR(model.last_statistics().inventory_error, 0., 1e-12);
        EXPECT_NEAR(model.last_statistics().donor_inventory_error, 0., 1e-12);
        EXPECT_NO_THROW(geometry->require_geometry_writable());
        std::map<std::string, std::vector<double>> result;
        for (const auto& [name, field] : model.output_fields())
            for (size_t cell = 0; cell < mesh->num_local_cells(); ++cell)
                result[name].push_back(field->local_value(static_cast<Pack::local_ordinal_type>(cell)));
        return result;
    };
    const auto expected = advance();
    motion.rollback_trial();
    model.restore(checkpoint);
    model.refresh_geometry();
    const auto replay = advance();
    for (const auto& [name, values] : expected)
    {
        SCOPED_TRACE(name);
        for (size_t cell = 0; cell < values.size(); ++cell)
            EXPECT_NEAR(replay.at(name)[cell], values[cell], std::max(1e-17, std::abs(values[cell]) * 1e-11));
    }
    motion.rollback_trial();
    model.restore(checkpoint);
    model.refresh_geometry();
}

/** Check analytic source/decay, changing cell inputs and ghost state for either policy. */
void check_repeated_kinetics_with_heterogeneous_inputs(SimpleFluid::RadiolyticKineticsMode mode)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 2, 2, 0.25));
    auto options = sheng_options();
    options.kinetics_mode = mode;
    options.microbubble_lifetime = 1.e-5;
    options.large_bubble_dissolution_time = 2.e-5;
    options.micro_to_large_conversion_coefficient = 0.;
    options.max_subcycles = 9;
    RadiolyticModelType model(mesh, options);
    model.enable_donor_hydrogen_deficit_tracking();
    FieldType temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"),
        power(mesh, 0., "power");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0., "flux");
    auto material = make_water_properties(mesh);
    const std::array<double, 3> time_steps{8.e-6, 3.1e-5, 1.1e-5};
    const std::array<int, 3> expected_subcycles{4, 9, 6};
    std::vector<double> micro_moles(mesh->num_local_cells());
    std::vector<double> micro_number(mesh->num_local_cells());
    std::vector<double> donor(mesh->num_local_cells());
    double time = 0.;
    for (size_t step = 0; step < time_steps.size(); ++step)
    {
        SCOPED_TRACE(step);
        const auto power_at = [step](auto gid)
        {
            return (gid % 5 == 0 ? -1. : 1.) * 1000. * (1. + 0.1 * gid) * (step + 1.);
        };
        const auto temperature_at = [step](auto gid) { return 300. + 0.5 * gid + 2. * step; };
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto lid = static_cast<Pack::local_ordinal_type>(owned);
            const auto gid = mesh->cell_global_id(lid);
            power.set_owned_value(lid, power_at(gid));
            temperature.set_owned_value(lid, temperature_at(gid));
        }
        power.sync_ghosts();
        temperature.sync_ghosts();
        const auto dt = time_steps[step];
        const auto substep = dt / expected_subcycles[step];
        const auto retention = std::exp(-dt / options.microbubble_lifetime);
        // Closed-form geometric sum for source impulses preceding each decay.
        const auto source_retention = mode == SimpleFluid::RadiolyticKineticsMode::ExactInactive
            ? -options.microbubble_lifetime * std::expm1(-dt / options.microbubble_lifetime)
            : substep * std::exp(-substep / options.microbubble_lifetime)
                * std::expm1(-dt / options.microbubble_lifetime)
                / std::expm1(-substep / options.microbubble_lifetime);
        time += dt;
        model.advance(time, dt, temperature, pressure, velocity, flux, material, &power);
        EXPECT_EQ(model.last_statistics().maximum_subcycles,
            mode == SimpleFluid::RadiolyticKineticsMode::ExactInactive ? 1 : expected_subcycles[step]);
        double local_total = 0.;
        for (size_t local = 0; local < mesh->num_local_cells(); ++local)
        {
            const auto lid = static_cast<Pack::local_ordinal_type>(local);
            const auto gid = mesh->cell_global_id(lid);
            const auto production = options.gas_release_efficiency * options.hydrogen_yield_mol_per_j
                * std::max(power_at(gid), 0.);
            const auto cell_temperature = temperature_at(gid);
            const auto radius = SimpleFluid::RadiolyticGasPhysics::nucleation_radius(
                options, cell_temperature, options.reference_pressure);
            const auto nucleation_moles = 4. * std::numbers::pi / 3.
                * (options.reference_pressure * radius * radius * radius
                    + 2. * options.surface_tension * radius * radius)
                / (options.gas_constant * cell_temperature);
            micro_moles[local] = micro_moles[local] * retention + production * source_retention;
            micro_number[local] = micro_number[local] * retention
                + production * source_retention / nucleation_moles;
            donor[local] += production * dt;
            EXPECT_NEAR(model.micro_moles().local_value(lid), micro_moles[local],
                std::max(1.e-23, micro_moles[local] * 1.e-11));
            EXPECT_NEAR(model.micro_number_density().local_value(lid), micro_number[local],
                std::max(1.e-14, micro_number[local] * 1.e-11));
            EXPECT_NEAR(model.donor_hydrogen_deficit().local_value(lid), donor[local],
                std::max(1.e-23, donor[local] * 1.e-11));
            EXPECT_DOUBLE_EQ(model.output_fields().at("H2_production_rate")->local_value(lid), production);
            if (local < mesh->num_owned_cells())
                local_total += donor[local] * mesh->cell_volume(lid);
        }
        const auto total = global_sum(*mesh, local_total);
        EXPECT_NEAR(model.global_submerged_hydrogen_moles(), total, total * 1.e-11);
        EXPECT_NEAR(model.last_statistics().inventory_error, 0., total * 1.e-11);
        EXPECT_NEAR(model.last_statistics().donor_inventory_error, 0., total * 1.e-11);
    }
}

/** Subcycles preserve split-source decay, changing cell inputs and ghost state. */
TEST(RadiolyticGasModelMultiRankTest, RepeatedSubcyclesRefreshHeterogeneousInputsAndDonorInventory)
{
    check_repeated_kinetics_with_heterogeneous_inputs(SimpleFluid::RadiolyticKineticsMode::LegacySubcycled);
}

/** Exact inactive updates preserve continuous production, changing inputs, donors and ghosts. */
TEST(RadiolyticGasModelMultiRankTest, ExactInactiveRefreshesHeterogeneousInputsAndDonorInventory)
{
    check_repeated_kinetics_with_heterogeneous_inputs(SimpleFluid::RadiolyticKineticsMode::ExactInactive);
}

/** Direct model users cannot select inconsistent numerical methods across ranks. */
TEST(RadiolyticGasModelMultiRankTest, KineticsModeMustBeCollective)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 2, 2, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    auto options = sheng_options();
    if (comm->getRank() == 0)
        options.kinetics_mode = SimpleFluid::RadiolyticKineticsMode::ExactInactive;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"), power(mesh, 0., "power");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0., "flux");
    auto material = make_water_properties(mesh);
    EXPECT_THROW(model.advance(1.e-5, 1.e-5, temperature, pressure, velocity, flux, material, &power),
        std::invalid_argument);
}

/** @brief Collective policy selection rejects divergent ranks and supports GS. */
TEST(RadiolyticGasModelMultiRankTest, TransportSolverPolicyIsCollective)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "This test requires at least two MPI ranks.";

    auto options = sheng_options();
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-6;
    options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = 10.0;
    options.free_surface_patches = {"zmax"};
    options.microbubble_lifetime = 1.0e30;
    options.large_bubble_dissolution_time = 1.0e30;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);
    const auto original = model.transport_linear_solver_options();

    auto candidate = original;
    if (comm->getRank() == 0)
        candidate.tolerance = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(model.set_transport_linear_solver_options(candidate), std::exception);
    candidate = original;
    if (comm->getRank() == 0)
        candidate.backend = SimpleFluid::LinearSolverBackend::Cg;
    EXPECT_THROW(model.set_transport_linear_solver_options(candidate), std::exception);
    candidate = original;
    if (comm->getRank() == 0)
        candidate.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    EXPECT_THROW(model.set_transport_linear_solver_options(candidate), std::invalid_argument);
    candidate = original;
    if (comm->getRank() == 0)
        candidate.preconditioner = SimpleFluid::LinearPreconditioner::GaussSeidel;
    EXPECT_THROW(model.set_transport_linear_solver_options(candidate), std::invalid_argument);
    candidate = original;
    if (comm->getRank() == 0)
        candidate.tolerance *= 2.0;
    EXPECT_THROW(model.set_transport_linear_solver_options(candidate), std::invalid_argument);
    EXPECT_EQ(model.transport_linear_solver_options().backend, original.backend);
    EXPECT_EQ(model.transport_linear_solver_options().preconditioner, original.preconditioner);
    EXPECT_DOUBLE_EQ(model.options().transport_solver_tolerance, original.tolerance);

    candidate = original;
    candidate.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    candidate.preconditioner = SimpleFluid::LinearPreconditioner::GaussSeidel;
    model.set_transport_linear_solver_options(candidate);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    model.advance(0.025, 0.025, temperature, pressure, velocity, flux, material, &power);
    const auto& statistics = model.last_statistics();
    EXPECT_GT(statistics.hydrogen_escaped, 0.0);
    EXPECT_NEAR(statistics.inventory_error, 0.0, 1.0e-13);
    EXPECT_TRUE(statistics.transport_linear.converged);
    EXPECT_EQ(statistics.transport_linear.solves, 5);
    EXPECT_GT(statistics.transport_linear.iterations, 0);
    EXPECT_LE(statistics.transport_linear.achieved_tolerance, candidate.tolerance);
    expect_same_on_all_ranks(*mesh, statistics.transport_linear.solves);
    expect_same_on_all_ranks(*mesh, statistics.transport_linear.iterations);
    expect_same_on_all_ranks(*mesh, statistics.transport_linear.achieved_tolerance);
    // Every published column must retain the same halo values as an
    // independent scalar import, including after a state rollback.
    const auto check_publication = [&]
    {
        for (const auto& [name, field] : model.output_fields())
        {
            SCOPED_TRACE(name);
            Pack::vector_type expected(field->overlap_data().getMap(), false);
            Pack::import_type importer(field->owned_data().getMap(), expected.getMap());
            expected.doImport(field->owned_data(), importer, Tpetra::REPLACE);
            const auto values = expected.getData();
            const auto actual = field->local_read_view();
            for (size_t row = 0; row < values.size(); ++row)
                EXPECT_DOUBLE_EQ(actual(row, 0), values[row]);
        }
    };
    check_publication();
    const auto accepted = model.snapshot();
    model.advance(0.05, 0.025, temperature, pressure, velocity, flux, material, &power);
    check_publication();
    model.restore(accepted);
    check_publication();
}

TEST(RadiolyticGasModelMultiRankTest, ExternalMaterialFailureIsCollective)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 2, 2, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2) GTEST_SKIP() << "Requires two ranks.";
    for (const bool surface_failure : {true, false})
    {
        auto options = sheng_options();
        options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Inertial;
        options.surface_tension_mode = SimpleFluid::SurfaceTensionMode::External;
        options.surface_tension_correlation = [surface_failure](double temperature)
        {
            if (surface_failure && temperature > 325) throw std::runtime_error("external surface tension failure");
            return .07;
        };
        options.nucleation_radius_mode = SimpleFluid::NucleationRadiusMode::External;
        options.nucleation_radius_correlation = [surface_failure](double temperature, double)
        {
            if (!surface_failure && temperature > 325) throw std::runtime_error("external nucleation failure");
            return 6.6359547089482127e-8;
        };
        RadiolyticModelType model(mesh, options);
        FieldType temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"), power(mesh, 0., "power");
        VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
        FaceFieldType flux(mesh, 0., "flux");
        auto material = make_water_properties(mesh);
        model.initialize_state(0., temperature, pressure, velocity, material);
        ASSERT_GT(mesh->num_owned_cells(), 0U);
        if (comm->getRank() == 0) temperature.set_owned_value(0, 350.);
        EXPECT_THROW(model.advance(.001, .001, temperature, pressure, velocity, flux, material, &power), std::exception);
    }
}

/**
 * @brief Two-rank radiolysis update conserves global H2 and void diagnostics.
 */
TEST(RadiolyticGasModelMultiRankTest, ConservesGlobalHydrogenAndVoidInventory)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    auto options = sheng_options();
    options.microbubble_lifetime = 1.0e9;
    options.large_bubble_dissolution_time = 1.0e9;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    constexpr double power_density = 1.0e5;
    FieldType power(mesh, power_density, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    constexpr double time_step = 1.0e-6;
    model.advance(
        time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    double local_volume = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_volume += mesh->cell_volume(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    const auto total_volume = global_sum(*mesh, local_volume);
    const auto expected_produced =
        options.hydrogen_yield_mol_per_j
      * power_density
      * total_volume
      * time_step;
    const auto& statistics = model.last_statistics();

    EXPECT_NEAR(
        statistics.hydrogen_produced,
        expected_produced,
        expected_produced * 1.0e-10);
    EXPECT_DOUBLE_EQ(statistics.cumulative_hydrogen_produced, statistics.hydrogen_produced);
    EXPECT_DOUBLE_EQ(model.cumulative_hydrogen_produced(), statistics.cumulative_hydrogen_produced);
    EXPECT_DOUBLE_EQ(statistics.cumulative_dissolved_hydrogen_outflow, 0.0);
    EXPECT_DOUBLE_EQ(statistics.cumulative_submerged_bubble_hydrogen_escaped, 0.0);
    EXPECT_NEAR(
        statistics.inventory_error,
        0.0,
        std::max(1.0e-14, expected_produced * 1.0e-10));
    EXPECT_TRUE(std::isfinite(statistics.void_volume));
    EXPECT_GE(statistics.void_volume, 0.0);
    EXPECT_NEAR(
        statistics.void_volume,
        global_integral(model.alpha_g()),
        std::max(1.0e-14, statistics.void_volume * 1.0e-10));

    const auto dissolved_moles = model.global_dissolved_hydrogen_moles();
    const auto microbubble_moles = model.global_microbubble_hydrogen_moles();
    const auto large_bubble_moles = model.global_large_bubble_hydrogen_moles();
    const auto bubble_moles = model.global_submerged_bubble_hydrogen_moles();
    const auto total_submerged_moles = model.global_submerged_hydrogen_moles();
    const auto raw_bubble_volume = model.global_submerged_bubble_volume();
    EXPECT_NEAR(dissolved_moles, global_integral(model.dissolved_hydrogen_inventory()),
        std::max(1.0e-14, std::abs(dissolved_moles) * 1.0e-12));
    EXPECT_NEAR(microbubble_moles, global_integral(model.micro_moles()),
        std::max(1.0e-14, std::abs(microbubble_moles) * 1.0e-12));
    EXPECT_NEAR(large_bubble_moles, global_integral(model.large_moles()),
        std::max(1.0e-14, std::abs(large_bubble_moles) * 1.0e-12));
    EXPECT_DOUBLE_EQ(bubble_moles, microbubble_moles + large_bubble_moles);
    EXPECT_DOUBLE_EQ(total_submerged_moles, dissolved_moles + bubble_moles);
    EXPECT_NEAR(raw_bubble_volume, global_integral(*model.output_fields().at("alpha_g_raw")),
        std::max(1.0e-14, std::abs(raw_bubble_volume) * 1.0e-12));

    expect_same_on_all_ranks(*mesh, statistics.hydrogen_produced);
    expect_same_on_all_ranks(*mesh, statistics.cumulative_hydrogen_produced);
    expect_same_on_all_ranks(*mesh, statistics.cumulative_dissolved_hydrogen_outflow);
    expect_same_on_all_ranks(*mesh, statistics.cumulative_submerged_bubble_hydrogen_escaped);
    expect_same_on_all_ranks(*mesh, statistics.inventory_error);
    expect_same_on_all_ranks(*mesh, statistics.void_volume);
    expect_same_on_all_ranks(*mesh, dissolved_moles);
    expect_same_on_all_ranks(*mesh, microbubble_moles);
    expect_same_on_all_ranks(*mesh, large_bubble_moles);
    expect_same_on_all_ranks(*mesh, bubble_moles);
    expect_same_on_all_ranks(*mesh, total_submerged_moles);
    expect_same_on_all_ranks(*mesh, raw_bubble_volume);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_TRUE(std::isfinite(model.alpha_g().value(cell_lid)));
        EXPECT_GE(model.alpha_g().value(cell_lid), options.alpha_min);
        EXPECT_LE(model.alpha_g().value(cell_lid), options.alpha_max);
        EXPECT_TRUE(std::isfinite(
            model.dissolved_hydrogen_inventory().value(cell_lid)));
        EXPECT_GE(model.dissolved_hydrogen_inventory().value(cell_lid), 0.0);
        EXPECT_TRUE(std::isfinite(model.micro_moles().value(cell_lid)));
        EXPECT_GE(model.micro_moles().value(cell_lid), 0.0);
        EXPECT_TRUE(std::isfinite(model.large_moles().value(cell_lid)));
        EXPECT_GE(model.large_moles().value(cell_lid), 0.0);
    }
}

/** @brief Verify finite-Courant free-surface escape is globally conservative. */
TEST(RadiolyticGasModelMultiRankTest,
     ConservesFiniteCourantFreeSurfaceEscape)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    auto options = sheng_options();
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-6;
    options.rise_velocity_mode =
        SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = 10.0;
    options.free_surface_patches = {"zmax"};
    options.microbubble_lifetime = 1.0e30;
    options.large_bubble_dissolution_time = 1.0e30;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    const auto count_before =
        global_integral(model.micro_number_density())
      + global_integral(model.large_number_density());
    constexpr double time_step = 0.025;
    model.advance(
        time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    const auto count_after =
        global_integral(model.micro_number_density())
      + global_integral(model.large_number_density());
    const auto& statistics = model.last_statistics();
    const auto& molar_rate =
        *model.output_fields().at("H2_escape_molar_rate");
    const auto& number_rate =
        *model.output_fields().at("bubble_escape_number_rate");

    EXPECT_GT(statistics.hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(statistics.dissolved_hydrogen_outflow, 0.0);
    EXPECT_NEAR(statistics.submerged_bubble_hydrogen_escaped, statistics.hydrogen_escaped, 1.0e-13);
    EXPECT_NEAR(statistics.microbubble_hydrogen_escaped, statistics.submerged_bubble_hydrogen_escaped, 1.0e-13);
    EXPECT_DOUBLE_EQ(
        model.cumulative_submerged_bubble_hydrogen_escaped(), statistics.submerged_bubble_hydrogen_escaped);
    EXPECT_DOUBLE_EQ(statistics.large_bubble_hydrogen_escaped, 0.0);
    EXPECT_GT(statistics.escaped_bubble_count, 0.0);
    EXPECT_NEAR(
        statistics.hydrogen_after + statistics.hydrogen_escaped,
        statistics.hydrogen_before + statistics.hydrogen_produced,
        1.0e-13);
    EXPECT_NEAR(statistics.inventory_error, 0.0, 1.0e-13);
    EXPECT_NEAR(
        count_after + statistics.escaped_bubble_count,
        count_before,
        count_before * 1.0e-10);
    EXPECT_NEAR(
        time_step * global_integral(molar_rate),
        statistics.hydrogen_escaped,
        1.0e-13);
    EXPECT_NEAR(
        time_step * global_integral(number_rate),
        statistics.escaped_bubble_count,
        count_before * 1.0e-10);

    expect_same_on_all_ranks(*mesh, statistics.hydrogen_escaped);
    expect_same_on_all_ranks(*mesh, statistics.submerged_bubble_hydrogen_escaped);
    expect_same_on_all_ranks(*mesh, statistics.microbubble_hydrogen_escaped);
    expect_same_on_all_ranks(*mesh, statistics.large_bubble_hydrogen_escaped);
    expect_same_on_all_ranks(*mesh, statistics.escaped_bubble_count);
    expect_same_on_all_ranks(*mesh, statistics.inventory_error);
    expect_same_on_all_ranks(
        *mesh, statistics.cumulative_hydrogen_escaped);
    expect_same_on_all_ranks(*mesh, statistics.cumulative_submerged_bubble_hydrogen_escaped);
    expect_same_on_all_ranks(
        *mesh, statistics.cumulative_escaped_bubble_count);
}

/** @brief Verify clipped-cell diagnostics are reduced over every rank. */
TEST(RadiolyticGasModelMultiRankTest, ReducesClippedCellCountGlobally)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    auto options = sheng_options();
    options.alpha_max = 0.01;
    options.max_concentration = 1.0;
    options.initial_dissolved_hydrogen = 5.0;
    options.initial_large_number_density = 1.0e12;
    options.initial_large_moles = 1.0;
    options.microbubble_lifetime = 1.0e9;
    options.large_bubble_dissolution_time = 1.0e9;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        1.0e-9,
        1.0e-9,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    const auto expected_clipped = static_cast<int>(
        mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(
        model.last_statistics().clipped_cells, expected_clipped);
    const auto raw_bubble_volume = model.global_submerged_bubble_volume();
    const auto bounded_bubble_volume = global_integral(model.alpha_g());
    const auto unrepresented_bubble_volume = model.global_unrepresented_bubble_volume();
    EXPECT_GT(raw_bubble_volume, bounded_bubble_volume);
    EXPECT_NEAR(unrepresented_bubble_volume, raw_bubble_volume - bounded_bubble_volume, raw_bubble_volume * 1.0e-12);
    EXPECT_NEAR(
        raw_bubble_volume, global_integral(*model.output_fields().at("alpha_g_raw")), raw_bubble_volume * 1.0e-12);
    expect_same_on_all_ranks(
        *mesh,
        static_cast<double>(model.last_statistics().clipped_cells));
    expect_same_on_all_ranks(*mesh, raw_bubble_volume);
    expect_same_on_all_ranks(*mesh, unrepresented_bubble_volume);
}

/** @brief Reconstructed pressure domains and rejected candidates are global. */
TEST(RadiolyticGasModelMultiRankTest, CollectivelyValidatesReconstructedPressureOffset)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    auto options = sheng_options();
    options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Reconstructed;
    options.minimum_absolute_pressure = 5.0e4;
    options.initial_micro_number_density = 2.0e8;
    options.initial_micro_moles = 3.0e-6;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        pressure.set_owned_value(static_cast<Pack::local_ordinal_type>(owned), comm->getRank() == 0 ? -1000.0 : 1000.0);
    }
    pressure.sync_ghosts();
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    auto material = make_water_properties(mesh);
    model.initialize_state(0.0, temperature, pressure, velocity, material);

    const auto minimum_offset = model.minimum_valid_absolute_pressure_offset();
    EXPECT_NEAR(minimum_offset, options.minimum_absolute_pressure + 1000.0, 1.0e-10);
    expect_same_on_all_ranks(*mesh, minimum_offset);

    const auto rank_divergent_candidate = comm->getRank() == 0 ? minimum_offset - 1.0 : 2.0e5;
    EXPECT_THROW(model.evaluate_submerged_bubble_volume(rank_divergent_candidate), std::invalid_argument);
    const auto old_offset = model.absolute_pressure_offset();
    EXPECT_THROW(model.set_absolute_pressure_offset(comm->getRank() == 0 ? 1.5e5 : 2.0e5), std::invalid_argument);
    EXPECT_DOUBLE_EQ(model.absolute_pressure_offset(), old_offset);

    const auto expected_volume = model.evaluate_submerged_bubble_volume(2.0e5);
    model.set_absolute_pressure_offset(2.0e5);
    EXPECT_NEAR(model.global_submerged_bubble_volume(), expected_volume, std::max(1.0e-14, expected_volume * 1.0e-11));
    expect_same_on_all_ranks(*mesh, model.global_submerged_bubble_volume());
}

/** @brief A rank-local slip-property failure is reported before bubble transport. */
TEST(RadiolyticGasModelMultiRankTest,
     RankLocalBubbleSlipFailureThrowsCoherently)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    auto options = sheng_options();
    options.initial_large_number_density = 100.0;
    options.initial_large_moles = 1.0e-6;
    options.rise_velocity_mode =
        SimpleFluid::BubbleRiseVelocityMode::Celata2007;
    options.bubble_gas_density = 1.2;
    RadiolyticModelType model(mesh, options);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    model.initialize_state(
        0.0, temperature, pressure, velocity, material);

    ASSERT_GT(mesh->num_owned_cells(), 0U);
    if (comm->getRank() == 0)
    {
        // Deliberately leave overlap values untouched: only rank 0's owned
        // cell may fail the purely local bubble-slip validation stage.
        material.dynamic_viscosity.set_owned_value(
            0, std::numeric_limits<double>::quiet_NaN());
    }

    bool rejected = false;
    std::string message;
    try
    {
        model.advance(
            1.0e-6,
            1.0e-6,
            temperature,
            pressure,
            velocity,
            flux,
            material,
            &power);
    }
    catch (const std::exception& error)
    {
        rejected = true;
        message = error.what();
    }

    EXPECT_TRUE(rejected);
    if (comm->getRank() == 0)
    {
        EXPECT_NE(message.find("dynamic viscosity"), std::string::npos);
    }
    else
    {
        EXPECT_NE(
            message.find(
                "Radiolytic bubble-slip evaluation failed on another rank"),
            std::string::npos);
    }
}

/** One owned high-Sc population must fail before any rank enters a later collective. */
TEST(RadiolyticGasModelMultiRankTest, RankLocalLegacySchmidtLimitThrowsCoherently)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "This test requires at least two MPI ranks.";

    auto options = sheng_options();
    options.mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::LegacyHughmark;
    options.initial_large_number_density = 1.0e6;
    options.initial_large_moles = 1.0e-7;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    model.initialize_state(0.0, temperature, pressure, velocity, material);

    ASSERT_GT(mesh->num_owned_cells(), 0U);
    if (comm->getRank() == 0)
        material.dynamic_viscosity.set_owned_value(0, 3.0e-3);
    // D=1e-8, rho=1000: one owned cell has Sc=300; all other cells have Sc=100.
    bool rejected = false;
    std::string message;
    try
    {
        model.advance(1.0e-6, 1.0e-6, temperature, pressure, velocity,
                      flux, material, &power);
    }
    catch (const std::exception& error)
    {
        rejected = true;
        message = error.what();
    }
    EXPECT_TRUE(rejected);
    if (comm->getRank() == 0)
    {
        EXPECT_NE(message.find("Sc < 250"), std::string::npos) << message;
        EXPECT_NE(message.find("rank 0"), std::string::npos) << message;
        EXPECT_NE(message.find("cell"), std::string::npos) << message;
    }
    else
    {
        EXPECT_NE(message.find("another rank"), std::string::npos) << message;
    }
}

/** A rank-local finite-Pe state outside the Feng Stokes window fails coherently. */
TEST(RadiolyticGasModelMultiRankTest, RankLocalFengReynoldsLimitThrowsCoherently)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "This test requires at least two MPI ranks.";

    auto options = sheng_options();
    options.mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid;
    options.hydrogen_diffusivity = 4.5e-9;
    options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = 0.0045;
    options.initial_large_number_density = 1.0e6;
    options.initial_large_moles = 0.0002947014976484349; // R=1.2e-4 m at 298.85 K.
    options.microbubble_lifetime = 1.0e9;
    options.large_bubble_dissolution_time = 1.0e9;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 298.85, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        material.density.set_owned_value(cell, 1546.23563);
        material.dynamic_viscosity.set_owned_value(cell, 0.00243878721);
    }
    material.density.sync_ghosts();
    material.dynamic_viscosity.sync_ghosts();
    model.initialize_state(0.0, temperature, pressure, velocity, material);

    ASSERT_GT(mesh->num_owned_cells(), 0U);
    if (comm->getRank() == 0)
        material.dynamic_viscosity.set_owned_value(0, 1.0e-3);
    // Only rank 0's owned cell now has Re_d>1; Pe_a remains near 120.
    bool rejected = false;
    std::string message;
    try
    {
        model.advance(1.0e-6, 1.0e-6, temperature, pressure, velocity,
                      flux, material, &power);
    }
    catch (const std::exception& error)
    {
        rejected = true;
        message = error.what();
    }
    EXPECT_TRUE(rejected);
    if (comm->getRank() == 0)
    {
        EXPECT_NE(message.find("Re_d < 1"), std::string::npos) << message;
        EXPECT_NE(message.find("rank 0"), std::string::npos) << message;
        EXPECT_NE(message.find("cell"), std::string::npos) << message;
    }
    else
    {
        EXPECT_NE(message.find("another rank"), std::string::npos) << message;
    }
}

namespace
{
using ZeroALEHandle = SimpleFluid::MeshHandle<Pack>;
using ZeroALEModel = SimpleFluid::RadiolyticGasModel<Pack, ZeroALEHandle>;
using ZeroALEScalar = SimpleFluid::ScalarCellFieldStored<Pack, ZeroALEHandle>;

struct BatchedGhostProbe
{
    const ZeroALEModel* model = nullptr;
    bool donor = false;
    bool observed = false;
    bool clean = true;
};
thread_local BatchedGhostProbe batched_ghost_probe;
thread_local ZeroALEScalar* batched_mutation_target = nullptr;
thread_local double batched_mutation_value = 0;
thread_local bool batched_mutation_armed = false;

double inspect_batched_zero_ghosts(double)
{
    const auto* model = batched_ghost_probe.model;
    if (model)
    {
        const auto& work = model->last_statistics();
        if (work.transport_work[0].skipped && work.transport_work[3].skipped &&
            work.transport_work[4].skipped && (!batched_ghost_probe.donor || work.donor_transport_work.skipped))
        {
            // First reached from transported-volume reconstruction, before
            // kinetics and its final packed state import can repair ghosts.
            batched_ghost_probe.observed = true;
            const std::array<const ZeroALEScalar*, 4> fields{&model->dissolved_hydrogen_inventory(),
                batched_ghost_probe.donor ? &model->donor_hydrogen_deficit() : nullptr,
                &model->large_number_density(), &model->large_moles()};
            for (const auto* field : fields)
                if (field)
                {
                    const auto values = field->local_read_view();
                    for (size_t row = 0; row < values.extent(0); ++row)
                        batched_ghost_probe.clean = batched_ghost_probe.clean && values(row, 0) == 0;
                }
        }
    }
    return 1e-8;
}

double mutate_batched_zero_inventory(double)
{
    if (batched_mutation_armed)
    {
        batched_mutation_armed = false;
        batched_mutation_target->set_owned_value(0, batched_mutation_value);
    }
    return 1e-8;
}

struct ResetBatchedCallbackState
{
    ~ResetBatchedCallbackState()
    {
        batched_ghost_probe = {};
        batched_mutation_target = nullptr;
        batched_mutation_armed = false;
    }
};

SimpleFluid::SP<ZeroALEHandle> make_zero_ale_composite()
{
    using Composite = SimpleFluid::Meshes::MultiRegionMesh;
    return std::make_shared<ZeroALEHandle>(std::make_shared<Composite>(std::vector<Composite::Region>{
        SimpleFluid::Meshes::cartesian_region("left", {{{0, .5}, {0, 1}, {0, .5, 1}}}),
        SimpleFluid::Meshes::cartesian_region("right", {{{.5, 1}, {0, 1}, {0, .5, 1}}})},
        std::vector<Composite::Interface>{SimpleFluid::Meshes::StructuredPatchInterface{{0, 1}, {1, 0}}},
        SimpleFluid::Meshes::InterfaceTolerance{}, Composite::BoundaryNamePolicy::MergeMatchingNames));
}

struct DistributedALEZeroPair
{
    SimpleFluid::SP<ZeroALEHandle> mesh = make_zero_ale_composite();
    ZeroALEModel full, reduced;
    ZeroALEScalar temperature{mesh, 300., "temperature"}, pressure{mesh, 0., "pressure"}, power{mesh, 0., "power"};
    SimpleFluid::VectorCellFieldStored<Pack, ZeroALEHandle> velocity{mesh, ZeroALEHandle::Vec3{}, "velocity"};
    SimpleFluid::ScalarFaceFieldStored<Pack, ZeroALEHandle> flux{mesh, 0., "relative_flux"};
    SimpleFluid::MaterialPropertyFields<Pack, ZeroALEHandle> material;
    SimpleFluid::PlanarALEMeshMotion<Pack> motion{mesh};

    explicit DistributedALEZeroPair(SimpleFluid::RadiolyticGasOptions options, bool donor = true)
        : full(mesh, options), reduced(mesh, options), material(mesh, [] {
            SimpleFluid::BoussinesqModelOptions water;
            water.reference_density = water.density = 1000.;
            water.specific_heat_capacity = 4200.;
            water.dynamic_viscosity = .001;
            water.thermal_conductivity = .6;
            return water;
        }(), SimpleFluid::TimeStepperOptions{})
    {
        if (donor)
        {
            full.enable_donor_hydrogen_deficit_tracking();
            reduced.enable_donor_hydrogen_deficit_tracking();
        }
        reduced.set_skip_zero_auxiliary_transport(true);
        full.initialize_state(0., temperature, pressure, velocity, material);
        reduced.initialize_state(0., temperature, pressure, velocity, material);
    }

    void advance(double top, double time, double dt)
    {
        motion.begin_trial(top, dt);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
        for (auto* model : {&full, &reduced})
        {
            model->refresh_geometry();
            model->advance(time, dt, temperature, pressure, velocity, flux, material, &power, &ale);
        }
    }

    void compare() const
    {
        for (const auto& [name, field] : full.output_fields())
        {
            SCOPED_TRACE(name);
            const auto* actual = reduced.output_fields().at(name);
            for (size_t row = 0; row < mesh->num_local_cells(); ++row)
            {
                const auto lid = static_cast<Pack::local_ordinal_type>(row);
                const double expected = field->local_value(lid);
                EXPECT_NEAR(actual->local_value(lid), expected, std::max(1e-20, std::abs(expected) * 1e-12));
            }
        }
        EXPECT_NEAR(reduced.last_statistics().inventory_error, 0., 1e-12);
        EXPECT_NEAR(reduced.last_statistics().donor_inventory_error, 0., 1e-12);
    }
};
} // namespace

TEST(RadiolyticGasModelMultiRankTest, BatchedZeroAuxiliaryRepairsOverlapBeforePackedSync)
{
    for (const bool donor : {false, true})
    {
        SCOPED_TRACE(donor);
        auto options = sheng_options();
        options.diffusivity_mode = SimpleFluid::HydrogenDiffusivityMode::External;
        options.hydrogen_diffusivity_correlation = &inspect_batched_zero_ghosts;
        DistributedALEZeroPair state(options, donor);
        ResetBatchedCallbackState reset_callbacks;
        const auto comm = state.mesh->owned_cell_map()->getComm();
        if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
        const int local_ghosts = state.mesh->num_local_cells() > state.mesh->num_owned_cells();
        int any_ghosts = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_ghosts, &any_ghosts);
        ASSERT_EQ(any_ghosts, 1);
        const auto full_start = state.full.snapshot(), reduced_start = state.reduced.snapshot();
        for (const double top : {1.1, .9})
        {
            const std::array<const ZeroALEScalar*, 4> fields{&state.reduced.dissolved_hydrogen_inventory(),
                donor ? &state.reduced.donor_hydrogen_deficit() : nullptr,
                &state.reduced.large_number_density(), &state.reduced.large_moles()};
            for (const auto* field : fields)
                if (field)
                {
                    // Owned values remain exactly zero; poison every overlap
                    // row, including genuine remote ghosts and owned copies.
                    const_cast<ZeroALEScalar*>(field)->overlap_data().putScalar(37.);
                    const auto values = field->owned_read_view();
                    for (size_t row = 0; row < values.extent(0); ++row) EXPECT_DOUBLE_EQ(values(row, 0), 0.);
                }
            state.motion.begin_trial(top, .001);
            const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
            state.full.refresh_geometry();
            state.reduced.refresh_geometry();
            state.full.advance(.001, .001, state.temperature, state.pressure, state.velocity, state.flux,
                state.material, &state.power, &ale);
            batched_ghost_probe = {&state.reduced, donor, false, true};
            state.reduced.advance(.001, .001, state.temperature, state.pressure, state.velocity, state.flux,
                state.material, &state.power, &ale);
            EXPECT_TRUE(batched_ghost_probe.observed);
            EXPECT_TRUE(batched_ghost_probe.clean);
            batched_ghost_probe = {};
            state.compare();
            EXPECT_EQ(state.reduced.last_statistics().transport_linear.solves, 2);
            EXPECT_EQ(state.reduced.last_statistics().donor_transport_work.skipped, donor ? 1 : 0);
            state.motion.rollback_trial();
            state.full.restore(full_start);
            state.reduced.restore(reduced_start);
        }
    }
}

TEST(RadiolyticGasModelMultiRankTest, BatchedZeroAuxiliaryKeepsRankLocalMolesIndependent)
{
    auto options = sheng_options();
    options.microbubble_lifetime = options.large_bubble_dissolution_time = 1e100;
    options.micro_to_large_conversion_coefficient = 0;
    DistributedALEZeroPair state(options, false);
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    const auto full_start = state.full.snapshot(), reduced_start = state.reduced.snapshot();
    for (auto* model : {&state.full, &state.reduced})
    {
        auto& moles = const_cast<ZeroALEScalar&>(model->large_moles());
        if (comm->getRank() == 0) moles.set_owned_value(0, 1e-6);
        moles.sync_ghosts();
    }
    state.advance(1.1, .001, .001);
    state.compare();
    EXPECT_EQ(state.reduced.last_statistics().transport_work[3].skipped, 1);
    EXPECT_EQ(state.reduced.last_statistics().transport_work[4].skipped, 0);
    EXPECT_EQ(state.reduced.last_statistics().transport_work[4].assemblies, 1);
    EXPECT_EQ(state.reduced.last_statistics().donor_transport_work.solves, 0);
    EXPECT_GT(state.reduced.global_large_bubble_hydrogen_moles(), 0.);
    state.motion.rollback_trial();
    state.full.restore(full_start);
    state.reduced.restore(reduced_start);
    // Replaying the zero baseline must discard the previous nonzero mask.
    state.advance(.9, .001, .001);
    state.compare();
    EXPECT_EQ(state.reduced.last_statistics().transport_work[4].skipped, 1);
    EXPECT_EQ(state.reduced.last_statistics().transport_linear.solves, 2);
    EXPECT_DOUBLE_EQ(state.reduced.global_large_bubble_hydrogen_moles(), 0.);
    state.motion.rollback_trial();
}

TEST(RadiolyticGasModelMultiRankTest, BatchedZeroCallbackMutationIsCollectiveWithoutErasure)
{
    constexpr auto guard_message = "Radiolytic material callback modified an untransported zero auxiliary inventory.";
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        SCOPED_TRACE(scenario);
        auto options = sheng_options();
        options.diffusivity_mode = SimpleFluid::HydrogenDiffusivityMode::External;
        options.hydrogen_diffusivity_correlation = &mutate_batched_zero_inventory;
        DistributedALEZeroPair state(options);
        ResetBatchedCallbackState reset_callbacks;
        const auto comm = state.mesh->owned_cell_map()->getComm();
        if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
        const bool enabled = scenario != 3;
        if (!enabled) state.reduced.set_skip_zero_auxiliary_transport(false);
        const auto checkpoint = state.reduced.snapshot();
        auto& target = const_cast<ZeroALEScalar&>(scenario == 0
            ? state.reduced.dissolved_hydrogen_inventory() : state.reduced.large_moles());
        batched_mutation_target = &target;
        batched_mutation_value = scenario == 2 ? std::numeric_limits<double>::quiet_NaN() : 1e-6;
        batched_mutation_armed = comm->getRank() == 0;
        state.motion.begin_trial(1.1, .001);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
        state.reduced.refresh_geometry();
        int category = 0;
        std::string message;
        try
        {
            state.reduced.advance(.001, .001, state.temperature, state.pressure, state.velocity, state.flux,
                state.material, &state.power, &ale);
        }
        catch (const std::logic_error& error) { category = 1; message = error.what(); }
        catch (const std::runtime_error& error) { category = 2; message = error.what(); }
        batched_mutation_armed = false;
        batched_mutation_target = nullptr;
        if (enabled)
        {
            EXPECT_EQ(category, comm->getRank() == 0 ? 1 : 2);
            EXPECT_EQ(message, comm->getRank() == 0 ? guard_message
                : "Radiolytic diffusivity evaluation failed on another rank.");
            EXPECT_EQ(state.reduced.last_statistics().transport_work[0].skipped, 0);
            EXPECT_EQ(state.reduced.last_statistics().transport_linear.solves, 0);
            if (comm->getRank() == 0)
            {
                if (scenario == 2) EXPECT_TRUE(std::isnan(target.value(0)));
                else EXPECT_DOUBLE_EQ(target.value(0), batched_mutation_value);
            }
        }
        else
        {
            // The disabled policy retains its existing transport/physical
            // validation path, rather than adding the cached-zero guard.
            EXPECT_NE(message, guard_message);
            EXPECT_GT(state.reduced.last_statistics().transport_work[4].solves, 0);
            if (comm->getRank() == 0) EXPECT_GT(target.value(0), 0.);
        }
        state.motion.rollback_trial();
        state.reduced.restore(checkpoint);
    }
}

TEST(RadiolyticGasModelMultiRankTest, ALEZeroAuxiliaryResumesAfterRankLocalProduction)
{
    auto options = sheng_options();
    options.kinetics_mode = SimpleFluid::RadiolyticKineticsMode::ExactInactive;
    options.microbubble_lifetime = .01;
    options.micro_to_large_conversion_coefficient = 0;
    options.large_bubble_dissolution_time = 1e100;
    DistributedALEZeroPair state(options);
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    const int local_owned = state.mesh->num_owned_cells() > 0;
    int all_owned = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_owned, &all_owned);
    ASSERT_EQ(all_owned, 1);
    if (comm->getRank() == 0) state.power.set_owned_value(0, 1000.);
    state.power.sync_ghosts();
    for (const auto face : state.flux.owned_face_ids())
        if (!state.mesh->is_boundary_face(face))
            state.flux.set_owned_value(face, .05 * state.mesh->face_area_vector(face).x);
    state.flux.sync_ghosts();
    double time = 0;
    for (int step = 1; step <= 3; ++step)
    {
        const double dt = step * .001;
        time += dt;
        state.advance(step == 2 ? .9 : 1.1, time, dt);
        state.compare();
        const auto& work = state.reduced.last_statistics();
        EXPECT_EQ(work.transport_work[0].skipped, step == 1 ? 1 : 0);
        EXPECT_EQ(work.donor_transport_work.skipped, step == 1 ? 1 : 0);
        EXPECT_EQ(work.transport_work[3].skipped, 1);
        EXPECT_EQ(work.transport_work[4].skipped, 1);
        EXPECT_EQ(work.transport_work[1].solves, 1);
        EXPECT_EQ(work.transport_work[2].assemblies, 0);
        if (step == 1) EXPECT_EQ(work.transport_linear.solves, 2);
        else
        {
            EXPECT_GE(work.transport_work[0].solves, 1);
            EXPECT_GE(work.donor_transport_work.solves, 1);
        }
        EXPECT_GT(state.reduced.global_dissolved_hydrogen_moles(), 0.);
        state.motion.accept_trial();
        state.power.put_scalar(0.);
        state.power.sync_ghosts();
    }
}

TEST(RadiolyticGasModelMultiRankTest, ALEZeroAuxiliaryInvalidDiffusivityFailsBeforeSkipping)
{
    auto options = sheng_options();
    options.diffusivity_mode = SimpleFluid::HydrogenDiffusivityMode::External;
    options.hydrogen_diffusivity_correlation = +[](double temperature) { return temperature > 325 ? -1. : 1e-8; };
    DistributedALEZeroPair state(options);
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    const auto full_start = state.full.snapshot(), reduced_start = state.reduced.snapshot();
    if (comm->getRank() == 0 && state.mesh->num_owned_cells()) state.temperature.set_owned_value(0, 350.);
    state.temperature.sync_ghosts();
    state.motion.begin_trial(1.1, .001);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
    state.reduced.refresh_geometry();
    int category = 0;
    std::string message;
    try
    {
        state.reduced.advance(.001, .001, state.temperature, state.pressure, state.velocity, state.flux,
            state.material, &state.power, &ale);
    }
    catch (const std::invalid_argument& error) { category = 1; message = error.what(); }
    catch (const std::runtime_error& error) { category = 2; message = error.what(); }
    EXPECT_EQ(category, comm->getRank() == 0 ? 1 : 2);
    EXPECT_NE(message.find(comm->getRank() == 0 ? "hydrogen diffusivity" : "another rank"), std::string::npos);
    EXPECT_EQ(state.reduced.last_statistics().transport_work[0].skipped, 0);
    EXPECT_EQ(state.reduced.last_statistics().transport_work[1].solves, 0);
    state.motion.rollback_trial();
    state.temperature.put_scalar(300.);
    state.temperature.sync_ghosts();
    state.full.restore(full_start);
    state.reduced.restore(reduced_start);
    state.advance(.9, .001, .001);
    state.compare();
    EXPECT_EQ(state.reduced.last_statistics().transport_linear.solves, 2);
    state.motion.rollback_trial();
}

/** Auxiliary transport skips require global zero and resume after local kinetics. */
TEST(RadiolyticGasModelMultiRankTest, ZeroAuxiliaryTransportMatchesFullAndResumesCollectively)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    auto options = sheng_options();
    options.microbubble_lifetime = 0.01;
    options.large_bubble_dissolution_time = 1.e100;
    options.micro_to_large_conversion_coefficient = 0.;
    options.max_subcycles = 1;
    RadiolyticModelType full(mesh, options), reduced(mesh, options);
    reduced.set_skip_zero_auxiliary_transport(true);
    if (comm->getSize() > 1)
        EXPECT_THROW(reduced.set_skip_zero_auxiliary_transport(comm->getRank() == 0), std::invalid_argument);
    FieldType temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"), power(mesh, 0., "power");
    if (comm->getRank() == 0 && mesh->num_owned_cells())
        power.set_owned_value(0, 1000.);
    power.sync_ghosts();
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0., "flux");
    auto material = make_water_properties(mesh);
    for (int step = 1; step <= 3; ++step)
    {
        full.advance(step * .001, .001, temperature, pressure, velocity, flux, material, &power);
        reduced.advance(step * .001, .001, temperature, pressure, velocity, flux, material, &power);
        EXPECT_EQ(full.last_statistics().transport_linear.solves, 5);
        EXPECT_EQ(reduced.last_statistics().transport_linear.solves, step == 1 ? 2 : 3);
        EXPECT_EQ(reduced.last_statistics().transport_work[0].skipped, step == 1 ? 1 : 0);
        EXPECT_EQ(reduced.last_statistics().transport_work[3].skipped, 1);
        EXPECT_EQ(reduced.last_statistics().transport_work[4].skipped, 1);
        EXPECT_EQ(reduced.last_statistics().transport_work[1].solves, 1);
        EXPECT_EQ(reduced.last_statistics().transport_work[2].assemblies, 0);
        const auto expected = full.output_fields(), actual = reduced.output_fields();
        ASSERT_EQ(expected.size(), actual.size());
        for (const auto& [name, field] : expected)
        {
            SCOPED_TRACE(name);
            const auto a = field->owned_data().getData();
            const auto b = actual.at(name)->owned_data().getData();
            for (size_t cell = 0; cell < a.size(); ++cell)
                EXPECT_NEAR(a[cell], b[cell], std::max(1.e-20, std::abs(a[cell]) * 1.e-12));
        }
        EXPECT_NEAR(reduced.last_statistics().inventory_error, 0., 1.e-14);
    }
}

/** A populated large category and dissolved inventory must retain all five solves. */
TEST(RadiolyticGasModelMultiRankTest, ZeroAuxiliaryPolicyRetainsPopulatedInventories)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 1.e-5;
    options.initial_large_number_density = 1.e7;
    options.initial_large_moles = 1.e-6;
    options.microbubble_lifetime = options.large_bubble_dissolution_time = 1.e100;
    options.micro_to_large_conversion_coefficient = 0.;
    RadiolyticModelType model(mesh, options);
    model.set_skip_zero_auxiliary_transport(true);
    FieldType temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"), power(mesh, 0., "power");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0., "flux");
    auto material = make_water_properties(mesh);
    model.advance(.001, .001, temperature, pressure, velocity, flux, material, &power);
    EXPECT_EQ(model.last_statistics().transport_linear.solves, 5);
    for (const auto& work : model.last_statistics().transport_work)
        EXPECT_EQ(work.skipped, 0);
    EXPECT_NEAR(model.last_statistics().inventory_error, 0., 1.e-14);
}

/** A skipped number equation cannot leave its moles partner using a stale operator. */
TEST(RadiolyticGasModelMultiRankTest, ZeroNumberStillAssemblesPopulatedMoles)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    auto options = sheng_options();
    options.initial_large_moles = 1.e-6;
    options.microbubble_lifetime = options.large_bubble_dissolution_time = 1.e100;
    options.micro_to_large_conversion_coefficient = 0.;
    RadiolyticModelType full(mesh, options), reduced(mesh, options);
    reduced.set_skip_zero_auxiliary_transport(true);
    FieldType temperature(mesh, 300., "temperature"), pressure(mesh, 0., "pressure"), power(mesh, 0., "power");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0., "flux");
    auto material = make_water_properties(mesh);
    for (int step = 1; step <= 2; ++step)
    {
        full.advance(step * .001, .001, temperature, pressure, velocity, flux, material, &power);
        reduced.advance(step * .001, .001, temperature, pressure, velocity, flux, material, &power);
        EXPECT_EQ(reduced.last_statistics().transport_linear.solves, 3);
        EXPECT_EQ(reduced.last_statistics().transport_work[3].skipped, 1);
        EXPECT_EQ(reduced.last_statistics().transport_work[4].assemblies, 1);
        const auto a = full.large_moles().owned_data().getData();
        const auto b = reduced.large_moles().owned_data().getData();
        for (size_t i = 0; i < a.size(); ++i) EXPECT_DOUBLE_EQ(a[i], b[i]);
    }
}
