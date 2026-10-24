/**
 * @file testReactingGasModelMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI consistency tests for the reacting gas model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "GeneralGasModelTestSupport.hh"
#include "equations/ReactingGasModel.hh"
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
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
namespace GeneralGas = SimpleFluid::Testing::GeneralGasModel;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;
using FaceFieldType = SimpleFluid::FaceField<Pack>;
using ReactingModelType = SimpleFluid::ReactingGasModel<Pack>;

thread_local const SimpleFluid::Meshes::MultiRegionMesh* gas_transport_guard_probe = nullptr;

double gas_diffusivity_with_transport_guard_probe(double)
{
    if (gas_transport_guard_probe)
        gas_transport_guard_probe->require_geometry_writable();
    return 1e-8;
}

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/**
 * @brief Create water-like material fields on the distributed test mesh.
 *
 * @param mesh Mesh owning the material fields.
 * @return Initialized water-property fields.
 */
SimpleFluid::MaterialPropertyFields<Pack> make_water_properties(const SimpleFluid::SP<MeshType>& mesh)
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
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_value, &global_value);
    return global_value;
}

/**
 * @brief Verify that a replicated diagnostic has the same value on every rank.
 *
 * @param mesh Mesh providing the communicator.
 * @param value Replicated value to compare.
 */
template<class TestMesh> void expect_same_on_all_ranks(const TestMesh& mesh, double value)
{
    double minimum = 0.0;
    double maximum = 0.0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &value, &minimum);
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &value, &maximum);
    EXPECT_NEAR(maximum, minimum, std::max(1.0e-14, std::abs(maximum) * 1.0e-12));
}

} // namespace

/** @brief A rank-local mutation probe fails inside leased gas transport, then
 * all ranks release the lease and can roll back/replay real composite ALE.
 */
TEST(ReactingGasModelMultiRankTest, CompositeTransportLeaseFailureIsCollectiveAndReleasesForALE)
{
    using Handle = SimpleFluid::MeshHandle<Pack>;
    using Composite = SimpleFluid::Meshes::MultiRegionMesh;
    using Scalar = SimpleFluid::ScalarCellFieldStored<Pack, Handle>;
    using Vector = SimpleFluid::VectorCellFieldStored<Pack, Handle>;
    using Face = SimpleFluid::ScalarFaceFieldStored<Pack, Handle>;
    using Model = SimpleFluid::ReactingGasModel<Pack, Handle>;
    auto geometry = std::make_shared<Composite>(
        std::vector<Composite::Region>{SimpleFluid::Meshes::cartesian_region("left", {{{0, .5}, {0, 1}, {0, .5, 1}}}),
            SimpleFluid::Meshes::cartesian_region("right", {{{.5, 1}, {0, 1}, {0, .5, 1}}})},
        std::vector<Composite::Interface>{SimpleFluid::Meshes::StructuredPatchInterface{{0, 1}, {1, 0}}},
        SimpleFluid::Meshes::InterfaceTolerance{}, Composite::BoundaryNamePolicy::MergeMatchingNames);
    auto mesh = std::make_shared<Handle>(geometry);
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    const int has_cells = mesh->num_owned_cells() > 0;
    int all_have_cells = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &has_cells, &all_have_cells);
    ASSERT_EQ(all_have_cells, 1);
    GeneralGas::Parameters parameters;
    parameters.dissolved_one = .2;
    parameters.number_density[0] = 1e8;
    parameters.molar_one[0] = 1e-6;
    parameters.diffusivity_two = [](const auto& c)
    { return gas_diffusivity_with_transport_guard_probe(c.environment.temperature); };
    auto options = GeneralGas::options(parameters);
    Model model(mesh, options);
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
    const double before = model.global_species_moles(GeneralGas::one);
    gas_transport_guard_probe = comm->getRank() == 0 ? geometry.get() : nullptr;
    int category = 0;
    std::string message;
    try
    {
        model.advance(.01, .01, temperature, pressure, velocity, relative_flux, material);
    }
    catch (const std::logic_error& error)
    {
        category = 1;
        message = error.what();
    }
    catch (const std::runtime_error& error)
    {
        category = 2;
        message = error.what();
    }
    catch (const std::exception& error)
    {
        category = 3;
        message = error.what();
    }
    gas_transport_guard_probe = nullptr;
    EXPECT_EQ(category, comm->getRank() == 0 ? 1 : 2);
    if (comm->getRank() == 0)
        EXPECT_EQ(message, "Geometry is held by a region execution view.");
    else
        EXPECT_NE(message.find("another rank"), std::string::npos);
    EXPECT_NO_THROW(geometry->require_geometry_writable());
    model.restore(checkpoint);

    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    const auto advance = [&]
    {
        motion.begin_trial(1.1, .01);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
        model.refresh_geometry();
        model.advance(
            .01, .01, temperature, pressure, velocity, relative_flux, material, &ale, SimpleFluid::Dimension::Z);
        EXPECT_GE(model.last_statistics().transport_linear.solves, 4);
        EXPECT_NEAR(model.global_species_moles(GeneralGas::one), before, 1e-12);
        EXPECT_NEAR(GeneralGas::history(model).inventory_error, 0., 1e-12);
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

/** Actual source substeps refresh owned/overlap data; no specialized fallback or donor deficit. */
void check_repeated_kinetics_with_heterogeneous_inputs(bool continuous)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 2, 2, .25));
    FieldType external_power(mesh, 0., "declared_external_feed_driver");
    GeneralGas::Parameters p;
    p.source = [&external_power](const GeneralGas::R::ReactionContext& c, double dt)
    {
        GeneralGas::M::require(
            c.environment.location && c.environment.location->ownedCell, "Expected source location.");
        const double q =
            external_power.value(static_cast<Pack::local_ordinal_type>(*c.environment.location->ownedCell));
        GeneralGas::R::ReactionSourceInput input;
        if (q > 0)
        {
            const double n = 2e-7 * q * dt * c.environment.bulkVolume;
            input.externalTransfers = {{"test.heterogeneous_finite_interval_feed", GeneralGas::one,
                GeneralGas::Phase::Dissolved, n, .002 * n, 0, {}, {}}};
        }
        input.heat.rate = GeneralGas::R::VolumetricHeatRate{3, GeneralGas::R::SourceVolumeBasis::BulkVolume};
        return input;
    };
    p.population = [&external_power, continuous](
                       const SimpleFluid::BubbleGasPhysics::MultispeciesPopulationContext& c, double dt)
    {
        const auto& e = c.chemistry.environment;
        const auto row = static_cast<Pack::local_ordinal_type>(*e.location->ownedCell);
        const double q = external_power.value(row), production = q > 0 ? 2e-7 * q : 0;
        constexpr double lifetime = 1e-5, radius = 6.6359547089482127e-8;
        const double retention = std::exp(-dt / lifetime), loss = -std::expm1(-dt / lifetime);
        const double retained_source = production * e.bulkVolume * (continuous ? lifetime * loss : dt * retention);
        const double per_bubble = 4 * std::numbers::pi / 3 *
                                  (e.absolutePressure * radius * radius * radius + 2 * .07 * radius * radius) /
                                  (GeneralGas::gas_constant * e.temperature);
        const double old =
            c.chemistry.state.amount(c.chemistry.catalog, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro");
        SimpleFluid::BubbleGasPhysics::MultispeciesPopulationPacket packet;
        if (old * loss > 0)
            packet.transfers.push_back(
                {GeneralGas::one, GeneralGas::Phase::Dispersed, GeneralGas::Phase::Dissolved, "micro", {}, old * loss});
        if (retained_source > 0)
            packet.transfers.push_back({GeneralGas::one, GeneralGas::Phase::Dissolved, GeneralGas::Phase::Dispersed, {},
                "micro", retained_source});
        packet.numberEvents = {{"micro", retained_source / per_bubble, c.populations.front().number * loss}};
        return packet;
    };
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    const std::array<double, 3> time_steps{8e-6, 3.1e-5, 1.1e-5};
    const std::array<int, 3> subdivisions{4, 9, 6};
    std::vector<double> expected_moles(mesh->num_local_cells()), expected_count(mesh->num_local_cells()),
        supplied(mesh->num_local_cells());
    double time = 0;
    std::uint64_t revision = 0;
    for (std::size_t step = 0; step < time_steps.size(); ++step)
    {
        const auto q = [step](auto gid) { return (gid % 5 == 0 ? -1. : 1.) * 1000 * (1 + .1 * gid) * (step + 1.); };
        const auto T = [step](auto gid) { return 300 + .5 * gid + 2 * step; };
        for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(row);
            const auto gid = mesh->cell_global_id(cell);
            external_power.set_owned_value(cell, q(gid));
            f.temperature.set_owned_value(cell, T(gid));
        }
        external_power.sync_ghosts();
        f.temperature.sync_ghosts();
        const double dt = time_steps[step], h = dt / (continuous ? 1 : subdivisions[step]);
        for (int substep = 0; substep < (continuous ? 1 : subdivisions[step]); ++substep)
        {
            time += h;
            f.advance(time, h);
            ++revision;
            EXPECT_EQ(f.model.species_event_receipt().revision, revision);
            for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(row);
                const double cv = mesh->cell_volume(cell);
                const auto gid = mesh->cell_global_id(cell);
                double n = 0;
                for (const auto& e : f.model.source_ledgers()[row].externalTransfers)
                    if (e.species == GeneralGas::one)
                    {
                        n += e.moles;
                        EXPECT_DOUBLE_EQ(e.mass, .002 * e.moles);
                        EXPECT_DOUBLE_EQ(e.energy, 0);
                    }
                EXPECT_NEAR(n, std::max(q(gid), 0.) * 2e-7 * h * cv, std::max(1e-23, std::abs(n) * 1e-11));
                EXPECT_NEAR(f.model.thermal_source().value(cell), 3, 1e-13);
            }
        }
        const double retention = std::exp(-dt / 1e-5);
        const double source_retention = continuous
                                            ? -1e-5 * std::expm1(-dt / 1e-5)
                                            : h * std::exp(-h / 1e-5) * std::expm1(-dt / 1e-5) / std::expm1(-h / 1e-5);
        double local_total = 0;
        for (std::size_t local = 0; local < mesh->num_local_cells(); ++local)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(local);
            const auto gid = mesh->cell_global_id(cell);
            const double production = 2e-7 * std::max(q(gid), 0.), r = 6.6359547089482127e-8;
            const double mol_per_bubble =
                4 * std::numbers::pi / 3 * (1e5 * r * r * r + 2 * .07 * r * r) / (GeneralGas::gas_constant * T(gid));
            expected_moles[local] = expected_moles[local] * retention + production * source_retention;
            expected_count[local] = expected_count[local] * retention + production * source_retention / mol_per_bubble;
            supplied[local] += production * dt;
            EXPECT_NEAR(f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, "micro").local_value(cell),
                expected_moles[local], std::max(1e-23, expected_moles[local] * 1e-11));
            EXPECT_NEAR(f.model.population_number_field("micro").local_value(cell), expected_count[local],
                std::max(1e-14, expected_count[local] * 1e-11));
            if (local < mesh->num_owned_cells())
                local_total += supplied[local] * mesh->cell_volume(cell);
        }
        const double total = global_sum(*mesh, local_total);
        EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), total, total * 1e-11);
        EXPECT_NEAR(GeneralGas::history(f.model).external_supply, total, total * 1e-11);
        EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, total * 1e-11);
    }
}

/** Subcycles preserve split-source decay, changing cell inputs and ghost state. */
TEST(ReactingGasModelMultiRankTest, RepeatedSubcyclesRefreshHeterogeneousInputsAndDonorInventory)
{
    check_repeated_kinetics_with_heterogeneous_inputs(false);
}

/** Supplied continuous-production updates preserve changing inputs, donor budgets and ghosts. */
TEST(ReactingGasModelMultiRankTest, ExactInactiveRefreshesHeterogeneousInputsAndDonorInventory)
{
    check_repeated_kinetics_with_heterogeneous_inputs(true);
}

/** Direct model users cannot select inconsistent numerical methods across ranks. */
TEST(ReactingGasModelMultiRankTest, KineticsModeMustBeCollective)
{
    using Native = SimpleFluid::MeshHandle<Pack>;
    Native::DistributionOptions d;
    d.allow_empty_partitions = true;
    auto grid = std::make_shared<Native>(
        std::make_shared<Native::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0., 1.}, {0., 1.}, {0., 1.}}}),
        d);
    const auto comm = grid->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI metadata divergence.";
    auto options = GeneralGas::options();
    if (comm->getRank() == comm->getSize() - 1)
        options.population_process = [](const auto&, double)
        { return SimpleFluid::BubbleGasPhysics::MultispeciesPopulationPacket{}; };
    EXPECT_THROW((SimpleFluid::ReactingGasModel<Pack, Native>(grid, options)), std::invalid_argument);
    EXPECT_NO_THROW((SimpleFluid::ReactingGasModel<Pack, Native>(grid, GeneralGas::options())));
    int local_empty = grid->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, comm->getSize() - 1);
}

/** @brief Collective policy selection rejects divergent ranks and supports GS. */
TEST(ReactingGasModelMultiRankTest, TransportSolverPolicyIsCollective)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "This test requires at least two MPI ranks.";

    auto selected = GeneralGas::populated();
    selected.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    selected.controls.constant_slip_velocity = 10;
    selected.controls.free_surface_patches = {"zmax"};
    auto options = GeneralGas::options(selected);
    ReactingModelType model(mesh, options);
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
    FieldType power(mesh, 0.0, "qdot_external");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    model.initialize_state(0.0, temperature, pressure, velocity, material);
    model.advance(0.025, 0.025, temperature, pressure, velocity, flux, material);
    const auto& statistics = model.last_statistics();
    EXPECT_GT(GeneralGas::step_escape(model), 0.0);
    EXPECT_NEAR(GeneralGas::history(model).inventory_error, 0.0, 1.0e-13);
    EXPECT_TRUE(statistics.transport_linear.converged);
    EXPECT_EQ(statistics.transport_linear.solves, 9);
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
    model.advance(0.05, 0.025, temperature, pressure, velocity, flux, material);
    check_publication();
    model.restore(accepted);
    check_publication();
}

TEST(ReactingGasModelMultiRankTest, ExternalMaterialFailureIsCollective)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 2, 2, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    for (const bool surface : {true, false})
    {
        GeneralGas::Parameters p;
        if (surface)
        {
            p.controls.surface_tension_mode = SimpleFluid::SurfaceTensionMode::External;
            p.controls.surface_tension_correlation = [](double T)
            {
                if (T > 325)
                    throw std::runtime_error("external surface tension failure");
                return .07;
            };
        }
        else
        {
            p.reservoir_mol_per_joule[0] = 2e-7;
            p.nucleate_supplied_gas = true;
            p.nucleation_radius_law = [](const auto& c)
            {
                if (c.environment.temperature > 325)
                    throw std::runtime_error("external nucleation failure");
                return 6.6359547089482127e-8;
            };
        }
        GeneralGas::Fixture<MeshType> f(mesh, p);
        f.power.put_scalar(1000);
        f.initialize();
        const auto receipt = f.model.species_event_receipt();
        const auto histories = f.model.species_histories();
        if (comm->getRank() == 0)
            f.temperature.set_owned_value(0, 350);
        f.temperature.sync_ghosts();
        EXPECT_THROW(f.advance(.001, .001), std::exception);
        EXPECT_EQ(f.model.species_event_receipt(), receipt);
        EXPECT_EQ(f.model.species_histories(), histories);
        f.temperature.put_scalar(300);
        f.temperature.sync_ghosts();
        EXPECT_NO_THROW(f.advance(.001, .001));
        if (!surface)
            EXPECT_GT(GeneralGas::history(f.model).external_supply, 0);
    }
}

/**
 * @brief Two-rank reacting_gas update conserves global H2 and void diagnostics.
 */
TEST(ReactingGasModelMultiRankTest, ConservesGlobalHydrogenAndVoidInventory)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    GeneralGas::Parameters p;
    p.reservoir_mol_per_joule = {2e-7, 1e-7};
    p.nucleate_supplied_gas = true;
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.power.put_scalar(1e5);
    f.initialize();
    f.advance(1e-6, 1e-6);
    const double domain = global_sum(*mesh,
        [&]
        {
            double v = 0;
            for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
                v += mesh->cell_volume(static_cast<Pack::local_ordinal_type>(row));
            return v;
        }());
    for (const auto [id, coefficient] :
        std::array<std::pair<GeneralGas::R::SpeciesId, double>, 2>{{{GeneralGas::one, 2e-7}, {GeneralGas::two, 1e-7}}})
    {
        const double expected = coefficient * 1e5 * 1e-6 * domain;
        EXPECT_NEAR(GeneralGas::history(f.model, id).external_supply, expected, 1e-13);
        EXPECT_NEAR(f.model.global_species_moles(id), expected, 1e-13);
        EXPECT_NEAR(GeneralGas::history(f.model, id).inventory_error, 0, 1e-13);
        expect_same_on_all_ranks(*mesh, f.model.global_species_moles(id));
    }
    double local_volume = 0;
    const auto& catalog = f.model.options().reaction_mechanism->catalog();
    for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        const double cv = mesh->cell_volume(cell);
        const auto state = f.model.cell_species_state(cell, cv);
        double n = 0;
        for (const auto& species : catalog.species())
            n += state.amount(catalog, species.id, GeneralGas::Phase::Dispersed, "micro");
        const double count = f.model.population_number_field("micro").value(cell) * cv;
        local_volume += GeneralGas::volume(n, count, 300, 1e5, .07);
        EXPECT_GE(f.model.alpha_g().value(cell), 0);
        EXPECT_LE(f.model.alpha_g().value(cell), p.controls.alpha_max);
    }
    const double expected_volume = global_sum(*mesh, local_volume);
    EXPECT_NEAR(f.model.global_submerged_bubble_volume(), expected_volume, expected_volume * 1e-11);
    expect_same_on_all_ranks(*mesh, f.model.global_submerged_bubble_volume());
    // Same actual source/transport protocol with genuinely empty native owners.
    using Native = SimpleFluid::MeshHandle<Pack>;
    Native::DistributionOptions d;
    d.allow_empty_partitions = true;
    auto one_cell = std::make_shared<Native>(
        std::make_shared<Native::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0., 1.}, {0., 1.}, {0., 1.}}}),
        d);
    GeneralGas::Fixture<Native> empty(one_cell, p);
    empty.power.put_scalar(1e5);
    empty.initialize();
    empty.advance(1e-6, 1e-6);
    int local_empty = one_cell->num_owned_cells() == 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &total_empty);
    EXPECT_EQ(total_empty, comm->getSize() - 1);
    EXPECT_NEAR(empty.model.global_species_moles(GeneralGas::one), 2e-7 * 1e5 * 1e-6, 1e-13);
    EXPECT_EQ(&empty.model.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved).mesh(), one_cell.get());
}

/** @brief Verify finite-Courant free-surface escape is globally conservative. */
TEST(ReactingGasModelMultiRankTest, ConservesFiniteCourantFreeSurfaceEscape)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    GeneralGas::Parameters p;
    p.number_density[0] = 1e10;
    p.molar_one[0] = 1e-6;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = 10;
    p.controls.free_surface_patches = {"zmax"};
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    const double before = f.model.global_species_moles(GeneralGas::one);
    const double count = GeneralGas::integral(f.model.population_number_field("micro"));
    f.advance(.025, .025);
    const double escaped = GeneralGas::step_escape(f.model), number = GeneralGas::number_escape(f.model, "micro");
    EXPECT_GT(escaped, 0);
    EXPECT_GT(number, 0);
    EXPECT_DOUBLE_EQ(GeneralGas::number_escape(f.model, "large"), 0);
    EXPECT_NEAR(GeneralGas::step_escape(f.model, GeneralGas::one, true), escaped, 1e-13);
    EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one) + escaped, before, 1e-13);
    EXPECT_NEAR(GeneralGas::integral(f.model.population_number_field("micro")) + number, count, count * 1e-10);
    EXPECT_NEAR(GeneralGas::history(f.model).inventory_error, 0, 1e-13);
    double local_molar = 0, local_number = 0;
    const auto& catalog = f.model.options().reaction_mechanism->catalog();
    for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        local_molar -= f.model.escape_ledgers()[row].externalChange.amount(
            catalog, GeneralGas::one, GeneralGas::Phase::Dispersed, "micro");
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        if (mesh->cell_centroid(cell).z > .75)
            local_number += .025 * 10 * .25 * .25 * f.model.population_number_field("micro").value(cell);
    }
    EXPECT_NEAR(global_sum(*mesh, local_molar), escaped, 1e-13);
    EXPECT_NEAR(global_sum(*mesh, local_number), number, count * 1e-10);
    expect_same_on_all_ranks(*mesh, escaped);
    expect_same_on_all_ranks(*mesh, number);
    const auto accepted = f.model.population_escape_receipt();
    const auto snap = f.model.snapshot();
    f.advance(.05, .025);
    f.model.restore(snap);
    EXPECT_EQ(f.model.population_escape_receipt(), accepted);
}

/** @brief Verify clipped-cell diagnostics are reduced over every rank. */
TEST(ReactingGasModelMultiRankTest, ReducesClippedCellCountGlobally)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    constexpr double capacity = .01, number = 1e12, requested = 1;
    const double radius = std::cbrt(3 * capacity / (4 * std::numbers::pi * number));
    const double admitted = capacity * (1e5 + 2 * .07 / radius) / (GeneralGas::gas_constant * 300) * (1 - 8e-12);
    GeneralGas::Parameters p;
    p.controls.alpha_max = capacity;
    p.number_density[1] = number;
    p.molar_one[1] = admitted;
    p.dissolved_one = 1;
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    f.advance(1e-9, 1e-9);
    int local_admissions = 0;
    double local_reservoir = 0, local_volume = 0;
    for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        const double cv = mesh->cell_volume(cell);
        EXPECT_LE(f.model.alpha_g().value(cell), capacity);
        const double stored = f.model.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, "large").value(cell);
        EXPECT_NEAR(stored, admitted, admitted * 1e-12);
        if (stored < requested)
            ++local_admissions;
        local_reservoir += (requested - stored) * cv;
        local_volume += cv;
    }
    int admissions = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_admissions, &admissions);
    EXPECT_EQ(admissions, static_cast<int>(mesh->owned_cell_map()->getGlobalNumElements()));
    const double reservoir = global_sum(*mesh, local_reservoir), domain = global_sum(*mesh, local_volume);
    EXPECT_GT(reservoir, 0);
    EXPECT_NEAR(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large") + reservoir,
        requested * domain, requested * domain * 1e-12);
    EXPECT_NEAR(f.model.global_submerged_bubble_volume(), GeneralGas::integral(f.model.alpha_g()),
        f.model.global_submerged_bubble_volume() * 1e-12);
    expect_same_on_all_ranks(*mesh, static_cast<double>(admissions));
    expect_same_on_all_ranks(*mesh, reservoir);
}

/** @brief Reconstructed pressure domains and rejected candidates are global. */
TEST(ReactingGasModelMultiRankTest, CollectivelyValidatesReconstructedPressureOffset)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    GeneralGas::Parameters p;
    p.controls.pressure_mode = SimpleFluid::ReactingPressureMode::Reconstructed;
    p.controls.minimum_absolute_pressure = 5e4;
    p.number_density[0] = 2e8;
    p.molar_one[0] = 3e-6;
    auto options = GeneralGas::options(p);
    ReactingModelType model(mesh, options);
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
TEST(ReactingGasModelMultiRankTest, RankLocalBubbleSlipFailureThrowsCoherently)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    GeneralGas::Parameters p;
    p.number_density[1] = 100;
    p.molar_one[1] = 1e-6;
    const double r = GeneralGas::radius(1e-6, 100, 300, p.pressure, p.sigma);
    p.molar_mass_one = 1.2 * GeneralGas::gas_constant * 300 / (p.pressure + 2 * p.sigma / r);
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::Celata2007;
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    const auto receipt = f.model.species_event_receipt();
    if (comm->getRank() == 0)
        f.material.dynamic_viscosity.set_owned_value(0, std::numeric_limits<double>::quiet_NaN());
    std::string message;
    try
    {
        f.advance(1e-6, 1e-6);
    }
    catch (const std::exception& error)
    {
        message = error.what();
    }
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message.find(comm->getRank() == 0 ? "dynamic viscosity" : "another rank"), std::string::npos);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    f.material.dynamic_viscosity.put_scalar(.001);
    f.material.dynamic_viscosity.sync_ghosts();
    EXPECT_NO_THROW(f.advance(1e-6, 1e-6));
}

/** One owned high-Sc population must fail before any rank enters a later collective. */
TEST(ReactingGasModelMultiRankTest, RankLocalLegacySchmidtLimitThrowsCoherently)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    GeneralGas::Parameters p;
    p.number_density[1] = 1e6;
    p.molar_one[1] = 1e-7;
    p.diffusivity[0] = 1e-8;
    p.controls.mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::Hughmark;
    p.viscosity_temperature = [](double T) { return T > 325 ? .003 : .001; };
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    const auto receipt = f.model.species_event_receipt();
    if (comm->getRank() == 0)
        f.temperature.set_owned_value(0, 350);
    f.temperature.sync_ghosts();
    std::string message;
    try
    {
        f.advance(1e-6, 1e-6);
    }
    catch (const std::exception& error)
    {
        message = error.what();
    }
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message.find(comm->getRank() == 0 ? "Sc < 250" : "another rank"), std::string::npos);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    f.temperature.put_scalar(300);
    f.temperature.sync_ghosts();
    EXPECT_NO_THROW(f.advance(1e-6, 1e-6));
}

/** A rank-local finite-Pe state outside the Feng Stokes window fails coherently. */
TEST(ReactingGasModelMultiRankTest, RankLocalFengReynoldsLimitThrowsCoherently)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    GeneralGas::Parameters p;
    p.temperature = 298.85;
    p.density = 1546.23563;
    p.viscosity = .00243878721;
    p.number_density[1] = 1e6;
    p.molar_one[1] = .0002947014976484349;
    p.diffusivity[0] = 4.5e-9;
    p.controls.mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid;
    p.controls.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    p.controls.constant_slip_velocity = .0045;
    p.viscosity_temperature = [](double T) { return T >= 300 ? .001 : .00243878721; };
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    const auto receipt = f.model.species_event_receipt();
    if (comm->getRank() == 0)
        f.temperature.set_owned_value(0, 300);
    f.temperature.sync_ghosts();
    std::string message;
    try
    {
        f.advance(1e-6, 1e-6);
    }
    catch (const std::exception& error)
    {
        message = error.what();
    }
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message.find(comm->getRank() == 0 ? "Re_d < 1" : "another rank"), std::string::npos);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    f.temperature.put_scalar(298.85);
    f.temperature.sync_ghosts();
    EXPECT_NO_THROW(f.advance(1e-6, 1e-6));
}

namespace
{
using ZeroALEHandle = SimpleFluid::MeshHandle<Pack>;
using ZeroALEModel = SimpleFluid::ReactingGasModel<Pack, ZeroALEHandle>;
using ZeroALEScalar = SimpleFluid::ScalarCellFieldStored<Pack, ZeroALEHandle>;
thread_local ZeroALEScalar* batched_mutation_target = nullptr;
thread_local double batched_mutation_value = 0;
thread_local bool batched_mutation_armed = false;
struct ResetBatchedCallbackState
{
    ~ResetBatchedCallbackState()
    {
        batched_mutation_target = nullptr;
        batched_mutation_armed = false;
    }
};
SimpleFluid::SP<ZeroALEHandle> make_zero_ale_composite()
{
    using Composite = SimpleFluid::Meshes::MultiRegionMesh;
    return std::make_shared<ZeroALEHandle>(std::make_shared<Composite>(
        std::vector<Composite::Region>{SimpleFluid::Meshes::cartesian_region("left", {{{0, .5}, {0, 1}, {0, .5, 1}}}),
            SimpleFluid::Meshes::cartesian_region("right", {{{.5, 1}, {0, 1}, {0, .5, 1}}})},
        std::vector<Composite::Interface>{SimpleFluid::Meshes::StructuredPatchInterface{{0, 1}, {1, 0}}},
        SimpleFluid::Meshes::InterfaceTolerance{}, Composite::BoundaryNamePolicy::MergeMatchingNames));
}
struct DistributedALEZeroPair
{
    SimpleFluid::SP<ZeroALEHandle> mesh = make_zero_ale_composite();
    GeneralGas::Parameters parameters;
    ZeroALEScalar temperature{mesh, 300., "temperature"}, pressure{mesh, 0., "pressure"}, power{mesh, 0., "power"};
    SimpleFluid::VectorCellFieldStored<Pack, ZeroALEHandle> velocity{mesh, ZeroALEHandle::Vec3{}, "velocity"};
    SimpleFluid::ScalarFaceFieldStored<Pack, ZeroALEHandle> flux{mesh, 0., "relative_flux"};
    SimpleFluid::MaterialPropertyFields<Pack, ZeroALEHandle> material;
    ZeroALEModel full, reduced;
    SimpleFluid::PlanarALEMeshMotion<Pack> motion{mesh};
    explicit DistributedALEZeroPair(GeneralGas::Parameters p = {})
        : parameters(std::move(p)), material(
                                        mesh,
                                        []
                                        {
                                            SimpleFluid::BoussinesqModelOptions v;
                                            v.reference_density = v.density = 1000;
                                            v.specific_heat_capacity = 4200;
                                            v.dynamic_viscosity = .001;
                                            v.thermal_conductivity = .6;
                                            return v;
                                        }(),
                                        SimpleFluid::TimeStepperOptions{}),
          full(mesh, GeneralGas::options(parameters, &power)), reduced(mesh, GeneralGas::options(parameters, &power))
    {
        full.initialize_state(0, temperature, pressure, velocity, material);
        reduced.initialize_state(0, temperature, pressure, velocity, material);
    }
    void advance(double top, double end, double dt)
    {
        motion.begin_trial(top, dt);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
        full.refresh_geometry();
        GeneralGas::force_numeric_reassembly(reduced);
        for (auto* model : {&full, &reduced})
        {
            model->advance(end, dt, temperature, pressure, velocity, flux, material, &ale);
        }
    }
    void compare() const
    {
        for (const auto& [name, field] : full.output_fields())
        {
            const auto* actual = reduced.output_fields().at(name);
            for (std::size_t row = 0; row < mesh->num_local_cells(); ++row)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(row);
                const double expected = field->local_value(cell);
                EXPECT_NEAR(actual->local_value(cell), expected, std::max(1e-20, std::abs(expected) * 1e-12)) << name;
            }
        }
        for (const auto& history : reduced.species_histories())
        {
            SCOPED_TRACE(history.species);
            if (history.species == GeneralGas::one || history.species == GeneralGas::two)
                EXPECT_NEAR(history.inventory_error, 0, 1e-12);
            else
            {
                // Keep the historical absolute gas oracle distinct from the
                // large finite solution stock with no external or boundary events.
                EXPECT_DOUBLE_EQ(history.current_moles, history.initial_moles);
                EXPECT_DOUBLE_EQ(history.chemical_production, 0.0);
                EXPECT_DOUBLE_EQ(history.chemical_consumption, 0.0);
                EXPECT_DOUBLE_EQ(history.external_supply, 0.0);
                EXPECT_DOUBLE_EQ(history.external_withdrawal, 0.0);
                EXPECT_DOUBLE_EQ(history.physical_outflow, 0.0);
            }
        }
        EXPECT_EQ(reduced.population_escape_receipt().events, full.population_escape_receipt().events);
    }
};
template<class TestModel> void check_import(const TestModel& model)
{
    for (const auto& [name, field] : model.output_fields())
    {
        Pack::vector_type expected(field->overlap_data().getMap(), false);
        Pack::import_type importer(field->owned_data().getMap(), expected.getMap());
        expected.doImport(field->owned_data(), importer, Tpetra::REPLACE);
        const auto values = expected.getData();
        const auto actual = field->local_read_view();
        for (std::size_t row = 0; row < values.size(); ++row)
            EXPECT_DOUBLE_EQ(actual(row, 0), values[row]) << name;
    }
}
} // namespace

TEST(ReactingGasModelMultiRankTest, BatchedZeroAuxiliaryRepairsOverlapBeforePackedSync)
{
    DistributedALEZeroPair state;
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    int local = state.mesh->num_local_cells() > state.mesh->num_owned_cells(), any = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &any);
    ASSERT_EQ(any, 1);
    const auto a = state.full.snapshot(), b = state.reduced.snapshot();
    for (const double top : {1.1, .9})
    {
        for (const auto* field : std::array<const ZeroALEScalar*, 4>{
                 &state.reduced.species_field(GeneralGas::one, GeneralGas::Phase::Dissolved),
                 &state.reduced.species_field(GeneralGas::two, GeneralGas::Phase::Dissolved),
                 &state.reduced.population_number_field("large"),
                 &state.reduced.species_field(GeneralGas::one, GeneralGas::Phase::Dispersed, "large")})
        {
            const_cast<ZeroALEScalar*>(field)->overlap_data().putScalar(37);
            const auto owned = field->owned_read_view();
            for (std::size_t row = 0; row < owned.extent(0); ++row)
                EXPECT_DOUBLE_EQ(owned(row, 0), 0);
        }
        state.motion.begin_trial(top, .001);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
        for (auto* model : {&state.full, &state.reduced})
        {
            model->refresh_geometry();
            model->advance(
                .001, .001, state.temperature, state.pressure, state.velocity, state.flux, state.material, &ale);
        }
        state.compare();
        check_import(state.reduced);
        // Pointwise user callbacks have no promised access to intermediate mutable field ghosts.
        // Accepted publication is compared to an independent import, including all zero schema columns.
        state.motion.rollback_trial();
        state.full.restore(a);
        state.reduced.restore(b);
    }
}

TEST(ReactingGasModelMultiRankTest, BatchedZeroAuxiliaryKeepsRankLocalMolesIndependent)
{
    GeneralGas::Parameters p;
    p.number_density[1] = 1e7;
    p.molar_one[1] = 1e-6;
    p.molar_two[1] = 3e-6;
    DistributedALEZeroPair state(p);
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    const auto prior_full = state.full.snapshot(), prior_reduced = state.reduced.snapshot();
    for (auto* model : {&state.full, &state.reduced})
    {
        std::vector<GeneralGas::R::CellSpeciesState> values;
        std::vector<std::vector<double>> numbers;
        const auto& catalog = model->options().reaction_mechanism->catalog();
        for (std::size_t row = 0; row < state.mesh->num_owned_cells(); ++row)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(row);
            const double cv = state.mesh->cell_volume(cell);
            auto value = model->cell_species_state(cell, cv);
            value.set_amount(catalog, GeneralGas::one, GeneralGas::Phase::Dispersed,
                (comm->getRank() == 0 ? 2e-6 : 1e-6) * cv, "large");
            value.set_amount(catalog, GeneralGas::two, GeneralGas::Phase::Dispersed,
                (comm->getRank() == 0 ? 1e-6 : 4e-6) * cv, "large");
            values.push_back(std::move(value));
            numbers.push_back({0, 1e7 * cv});
        }
        model->initialize_species(values, numbers);
        model->initialize_state(0, state.temperature, state.pressure, state.velocity, state.material);
    }
    // Explicitly installing the rank-local physical stocks invalidates earlier
    // snapshots. Replay must start from this new initialization authority.
    EXPECT_THROW(state.full.restore(prior_full), std::invalid_argument);
    EXPECT_THROW(state.reduced.restore(prior_reduced), std::invalid_argument);
    const auto a = state.full.snapshot(), b = state.reduced.snapshot();
    state.advance(1.1, .001, .001);
    state.compare();
    check_import(state.reduced);
    EXPECT_GT(state.reduced.global_species_moles(GeneralGas::one), 0);
    EXPECT_GT(state.reduced.global_species_moles(GeneralGas::two), 0);
    state.motion.rollback_trial();
    state.full.restore(a);
    state.reduced.restore(b);
    state.advance(.9, .001, .001);
    state.compare();
    check_import(state.reduced);
    state.motion.rollback_trial();
}

TEST(ReactingGasModelMultiRankTest, BatchedZeroCallbackMutationIsCollectiveWithoutErasure)
{
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        GeneralGas::Parameters p;
        p.diffusivity_two = [](const auto&)
        {
            if (batched_mutation_armed)
            {
                batched_mutation_armed = false;
                batched_mutation_target->set_owned_value(0, batched_mutation_value);
            }
            return 0.;
        };
        DistributedALEZeroPair state(p);
        ResetBatchedCallbackState reset;
        const auto comm = state.mesh->owned_cell_map()->getComm();
        if (comm->getSize() < 2)
            GTEST_SKIP() << "Requires MPI.";
        auto& target = const_cast<ZeroALEScalar&>(state.reduced.species_field(GeneralGas::one,
            scenario == 0 ? GeneralGas::Phase::Dissolved : GeneralGas::Phase::Dispersed, scenario == 0 ? "" : "large"));
        batched_mutation_target = &target;
        batched_mutation_value = scenario == 2 ? std::numeric_limits<double>::quiet_NaN() : 1e-6;
        batched_mutation_armed = comm->getRank() == 0;
        const auto checkpoint = state.reduced.snapshot();
        const auto receipt = state.reduced.species_event_receipt();
        state.motion.begin_trial(1.1, .001);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
        state.reduced.refresh_geometry();
        EXPECT_THROW(state.reduced.advance(.001, .001, state.temperature, state.pressure, state.velocity, state.flux,
                         state.material, &ale),
            std::exception);
        batched_mutation_armed = false;
        batched_mutation_target = nullptr;
        // Unledgered callback writes are invalid. Atomic rejection restores accepted
        // material; legitimately supplied positive stocks are covered independently.
        EXPECT_EQ(state.reduced.species_event_receipt(), receipt);
        for (std::size_t row = 0; row < state.mesh->num_owned_cells(); ++row)
            EXPECT_DOUBLE_EQ(target.value(static_cast<Pack::local_ordinal_type>(row)), 0);
        state.motion.rollback_trial();
        state.reduced.restore(checkpoint);
        state.advance(.9, .001, .001);
        state.compare();
        state.motion.rollback_trial();
    }
}

TEST(ReactingGasModelMultiRankTest, ALEZeroAuxiliaryResumesAfterRankLocalProduction)
{
    GeneralGas::Parameters p;
    p.reservoir_mol_per_joule[0] = 2e-7;
    p.nucleate_supplied_gas = true;
    p.micro_decay_time = .01;
    DistributedALEZeroPair state(p);
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    if (comm->getRank() == 0)
        state.power.set_owned_value(0, 1000);
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
        check_import(state.reduced);
        EXPECT_GT(GeneralGas::history(state.reduced).external_supply, 0);
        EXPECT_GT(GeneralGas::compartment(state.reduced, GeneralGas::one, GeneralGas::Phase::Dissolved), 0);
        EXPECT_GT(GeneralGas::integral(state.reduced.population_number_field("micro")), 0);
        EXPECT_DOUBLE_EQ(GeneralGas::integral(state.reduced.population_number_field("large")), 0);
        EXPECT_GT(state.reduced.last_statistics().transport_linear.solves, 0);
        state.motion.accept_trial();
        state.power.put_scalar(0);
        state.power.sync_ghosts();
    }
}

TEST(ReactingGasModelMultiRankTest, ALEZeroAuxiliaryInvalidDiffusivityFailsBeforeSkipping)
{
    GeneralGas::Parameters p;
    p.diffusivity_two = [](const auto& c) { return c.environment.temperature > 325 ? -1. : 1e-8; };
    DistributedALEZeroPair state(p);
    const auto comm = state.mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires MPI.";
    const auto a = state.full.snapshot(), b = state.reduced.snapshot();
    const auto receipt = state.reduced.species_event_receipt();
    if (comm->getRank() == 0)
        state.temperature.set_owned_value(0, 350);
    state.temperature.sync_ghosts();
    state.motion.begin_trial(1.1, .001);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*state.mesh, state.motion);
    state.reduced.refresh_geometry();
    int category = 0;
    std::string message;
    try
    {
        state.reduced.advance(
            .001, .001, state.temperature, state.pressure, state.velocity, state.flux, state.material, &ale);
    }
    catch (const std::invalid_argument& e)
    {
        category = 1;
        message = e.what();
    }
    catch (const std::runtime_error& e)
    {
        category = 2;
        message = e.what();
    }
    EXPECT_EQ(category, comm->getRank() == 0 ? 1 : 2);
    EXPECT_NE(message.find(comm->getRank() == 0 ? "diffusivity" : "another rank"), std::string::npos);
    EXPECT_EQ(state.reduced.last_statistics().transport_linear.solves, 0);
    EXPECT_EQ(state.reduced.species_event_receipt(), receipt);
    state.motion.rollback_trial();
    state.temperature.put_scalar(300);
    state.temperature.sync_ghosts();
    state.full.restore(a);
    state.reduced.restore(b);
    state.advance(.9, .001, .001);
    state.compare();
    state.motion.rollback_trial();
}

/** Auxiliary transport skips require global zero and resume after local kinetics. */
TEST(ReactingGasModelMultiRankTest, ZeroAuxiliaryTransportMatchesFullAndResumesCollectively)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    const auto comm = mesh->owned_cell_map()->getComm();
    GeneralGas::Parameters p;
    p.reservoir_mol_per_joule[0] = 2e-7;
    p.nucleate_supplied_gas = true;
    p.micro_decay_time = .01;
    GeneralGas::Fixture<MeshType> full(mesh, p), fresh(mesh, p);
    if (comm->getRank() == 0)
    {
        full.power.set_owned_value(0, 1000);
        fresh.power.set_owned_value(0, 1000);
    }
    full.power.sync_ghosts();
    fresh.power.sync_ghosts();
    full.initialize();
    fresh.initialize();
    for (int step = 1; step <= 3; ++step)
    {
        GeneralGas::force_numeric_reassembly(fresh.model);
        full.advance(step * .001, .001);
        fresh.advance(step * .001, .001);
        for (const auto& [name, field] : full.model.output_fields())
        {
            const auto* actual = fresh.model.output_fields().at(name);
            for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(row);
                const double expected = field->value(cell);
                EXPECT_NEAR(actual->value(cell), expected, std::max(1e-20, std::abs(expected) * 1e-12));
            }
        }
        EXPECT_EQ(full.model.last_statistics().transport_linear.solves,
            fresh.model.last_statistics().transport_linear.solves);
        EXPECT_EQ(full.model.last_statistics().transport_linear.solves, step == 1 ? 1 : 4);
        EXPECT_NEAR(GeneralGas::history(fresh.model).inventory_error, 0, 1e-14);
        check_import(fresh.model);
    }
}

/** Every nonzero schema column retains its own solve (finite solvent, dissolved gas, large gas and N). */
TEST(ReactingGasModelMultiRankTest, ZeroAuxiliaryPolicyRetainsPopulatedInventories)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    GeneralGas::Parameters p;
    p.dissolved_one = 1e-5;
    p.number_density[1] = 1e7;
    p.molar_one[1] = 1e-6;
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    const double initial = f.model.global_species_moles(GeneralGas::one);
    f.advance(.001, .001);
    EXPECT_EQ(f.model.last_statistics().transport_linear.solves, 4);
    EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), initial, 1e-14);
    EXPECT_GT(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dissolved), 0);
    EXPECT_GT(GeneralGas::compartment(f.model, GeneralGas::one, GeneralGas::Phase::Dispersed, "large"), 0);
    EXPECT_GT(GeneralGas::integral(f.model.population_number_field("large")), 0);
    check_import(f.model);
}

/** A skipped number equation cannot leave its moles partner using a stale operator. */
TEST(ReactingGasModelMultiRankTest, ZeroNumberStillAssemblesPopulatedMoles)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, .25));
    GeneralGas::Parameters p;
    p.sigma = 0;
    p.number_density[1] = 1e7;
    p.molar_one[1] = 1e-6;
    p.population = [](const SimpleFluid::BubbleGasPhysics::MultispeciesPopulationContext& c, double)
    {
        const auto found = std::find_if(
            c.populations.begin(), c.populations.end(), [](const auto& value) { return value.id == "large"; });
        SimpleFluid::BubbleGasPhysics::MultispeciesPopulationPacket packet;
        packet.numberEvents = {{"large", 0, .25 * found->number}};
        return packet;
    };
    GeneralGas::Fixture<MeshType> f(mesh, p);
    f.initialize();
    const double material = f.model.global_species_moles(GeneralGas::one);
    const double number = GeneralGas::integral(f.model.population_number_field("large"));
    for (int step = 1; step <= 2; ++step)
    {
        f.advance(step * .001, .001);
        EXPECT_NEAR(f.model.global_species_moles(GeneralGas::one), material, 1e-14);
        EXPECT_NEAR(GeneralGas::integral(f.model.population_number_field("large")), number * std::pow(.75, step),
            number * 1e-12);
        EXPECT_GT(f.model.last_statistics().transport_linear.solves, 0);
        check_import(f.model);
    }
    auto invalid = p;
    invalid.number_density[1] = 0;
    GeneralGas::Fixture<MeshType> malformed(mesh, invalid);
    EXPECT_THROW(malformed.initialize(), std::invalid_argument);
    EXPECT_FALSE(malformed.model.initial_state_initialized());
}
