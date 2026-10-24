/** @file
 * @brief Prescribed-carrier source ordering and accepted population escape receipts.
 */
#include "ReactingMixtureTestSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace TM = Testing::ReactingMixture;
using Pack = DefaultTpetraTypes;
using Grid = MeshHandle<Pack>;
using Flow = BoussinesqSolver<Pack>;
using LO = Pack::local_ordinal_type;
constexpr R::SpeciesId a = 211, b = 907;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
enum class Failure
{
    None,
    ExternalHeat,
    ReactionHeat,
    TransferHeat,
    CancellingHeat
};

ReactingGasOptions options(Failure failure = Failure::None)
{
    using P = R::SpeciesPhase;
    auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {a, "ordinary reference gas A", .01, {{"X", 1}}, {P::Liquid, P::Dissolved, P::Dispersed, P::Resolved}},
        {b, "ordinary reference gas B", .01, {{"X", 1}}, {P::Dispersed, P::Resolved}}});
    ReactingGasOptions result;
    result.mode = ReactingGasMode::Species;
    result.source_schedule = ReactingSourceSchedule::PrescribedCarrierAfterTemperature;
    result.reference_pressure = 100000;
    result.surface_tension = 0;
    result.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
        std::vector<R::ReactionDefinition>{
            {73, "reference gas isomerization", {{a, P::Dispersed, -1, "micro"}, {b, P::Dispersed, 1, "micro"}},
                failure == Failure::ReactionHeat ? 1. : 0.}},
        R::EnergyConvention::SensibleEnergy);
    const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};
    result.gas_thermodynamics =
        R::make_ideal_gas_mixture(catalog, {{20, 300, 0}, {20, 300, 0}}, bounds, R::EnergyConvention::SensibleEnergy);
    // Required but unused provider. No liquid amounts or liquid caloric storage are claimed.
    result.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds,
        [](std::span<const double>) { return R::LinearCompressibleMixtureLiquid{300, 100000, 1, 2, 0, 1e-9, 0, 0, 0}; },
        R::EnergyConvention::SensibleEnergy);
    result.initial_species_state = [catalog](const R::ReactionEnvironment& context, std::int64_t)
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"micro", std::vector<double>(2)}, {"large", std::vector<double>(2)}};
        state.set_amount(*catalog, a, P::Dispersed, .1 * context.bulkVolume, "micro");
        return state;
    };
    result.initial_population_numbers = [](const R::ReactionContext& c)
    { return std::vector<double>{1e5 * c.environment.bulkVolume, 0}; };
    result.species_source = [failure](const R::ReactionContext& c, double dt)
    {
        R::ReactionSourceInput input;
        const double incoming = 1e-5 * c.environment.temperature * c.environment.bulkVolume * dt;
        input.externalTransfers.push_back(
            {"declared reference gas reservoir", a, P::Dispersed, incoming, .01 * incoming, 0, {}, "micro"});
        if (dt > .1)
        {
            if (failure == Failure::ExternalHeat || failure == Failure::CancellingHeat)
                input.heat.integratedEnergy = 2;
            if (failure == Failure::ReactionHeat)
                input.integratedExtents = {{73, .001 * dt * c.environment.bulkVolume}};
            if (failure == Failure::TransferHeat || failure == Failure::CancellingHeat)
                input.externalTransfers[0].energy = -2;
        }
        return input;
    };
    result.population_process = [](const BubbleGasPhysics::MultispeciesPopulationContext& c, double dt)
    {
        BubbleGasPhysics::MultispeciesPopulationPacket packet;
        packet.numberEvents = {
            {"micro", 1e-5 * c.chemistry.environment.temperature * c.chemistry.environment.bulkVolume * dt / 1e-6, 0}};
        return packet;
    };
    return result;
}
struct Fixture
{
    std::shared_ptr<Grid> grid;
    std::shared_ptr<Flow> flow;
    std::unique_ptr<MultiphysicsSolverCoupler<Pack>> coupled;
    Fixture()
    {
        Grid::DistributionOptions partition;
        partition.allow_empty_partitions = true;
        grid = std::make_shared<Grid>(
            std::make_shared<Grid::Cartesian>(Vec3D<ArrReal>{{{0, 1}, {0, 1}, {0, .5, 1}}}), partition);
        TimeStepperOptions time;
        time.time_step = .1;
        time.reference_temperature = 300;
        time.gravity_x = time.gravity_y = time.gravity_z = 0;
        BoussinesqModelOptions material;
        material.reference_density = material.density = 1;
        material.specific_heat_capacity = 2;
        material.dynamic_viscosity = material.thermal_conductivity = 0;
        flow = std::make_shared<Flow>(grid, BoundaryConditionSet{}, time, LinearSolverOptions{}, material);
        coupled = std::make_unique<MultiphysicsSolverCoupler<Pack>>(flow);
        coupled->add_prescribed_heat_source().initialize_constant(20); // Exact +1 K per .1 s.
    }
    auto& initialize(ReactingGasOptions selected)
    {
        auto& gas = coupled->configure_reacting_gas(selected);
        coupled->initialize_heated_box(300, 300);
        return gas;
    }
    template<class Field> auto values(const Field& field) const
    {
        using Value = std::remove_cvref_t<decltype(field.value(LO{}))>;
        std::vector<Value> result;
        for (std::size_t c = 0; c < grid->num_owned_cells(); ++c)
            result.push_back(field.value(static_cast<LO>(c)));
        return result;
    }
};

TEST(ReactingReferenceScheduleTest, UpdatedTemperatureDrivesExplicitGasAndNumberSourcesWithActualEmptyOwners)
{
    Fixture f;
    auto& gas = f.initialize(options());
    f.coupled->step();
    // q_n=T_new*1e-5, not T_old*1e-5; the difference is 1e-6 mol on this domain.
    EXPECT_NEAR(gas.global_species_moles(a), .1 + 301 * 1e-5 * .1, 1e-13);
    for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        EXPECT_NEAR(f.flow->temperature().value(cell), 301, 1e-11);
        EXPECT_NEAR(gas.population_number_field("micro").value(cell), 1e5 + 301 * 1e-5 * .1 / 1e-6, 1e-9);
        EXPECT_DOUBLE_EQ(gas.thermal_source().value(cell), 0);
        EXPECT_DOUBLE_EQ(gas.liquid_mass_density().value(cell), 0);
    }
    const auto comm = f.grid->owned_cell_map()->getComm();
    int empty = f.grid->num_owned_cells() == 0, empty_total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_total);
    EXPECT_EQ(empty_total, std::max(0, comm->getSize() - 2));
}

TEST(ReactingReferenceScheduleTest, EveryNonzeroRawHeatChannelRejectsAfterTemperatureAndRestoresWholeStep)
{
    for (const auto failure :
        {Failure::ExternalHeat, Failure::ReactionHeat, Failure::TransferHeat, Failure::CancellingHeat})
    {
        SCOPED_TRACE(static_cast<int>(failure));
        Fixture f;
        auto& gas = f.initialize(options(failure));
        f.coupled->step();
        const auto temperature = f.values(f.flow->temperature()), pressure = f.values(f.flow->pressure());
        const auto velocity = f.values(f.flow->velocity());
        const auto density = f.values(f.flow->material_properties().density);
        const auto histories = gas.species_histories();
        std::vector<double> flux;
        for (std::size_t face = 0; face < f.grid->num_owned_faces(); ++face)
            flux.push_back(f.flow->pressure_corrected_face_fluxes().value(static_cast<LO>(face)));
        const auto number_escape = gas.population_escape_receipt();
        const auto receipt = gas.species_event_receipt();
        TM::SavedReactingState saved(gas);
        f.flow->set_time_step(.2);
        EXPECT_ANY_THROW(f.coupled->step());
        EXPECT_DOUBLE_EQ(f.flow->time(), .1);
        EXPECT_EQ(f.values(f.flow->temperature()), temperature);
        EXPECT_EQ(f.values(f.flow->pressure()), pressure);
        EXPECT_EQ(f.values(f.flow->material_properties().density), density);
        const auto restored_velocity = f.values(f.flow->velocity());
        ASSERT_EQ(restored_velocity.size(), velocity.size());
        for (std::size_t c = 0; c < velocity.size(); ++c)
        {
            EXPECT_DOUBLE_EQ(restored_velocity[c].x, velocity[c].x);
            EXPECT_DOUBLE_EQ(restored_velocity[c].y, velocity[c].y);
            EXPECT_DOUBLE_EQ(restored_velocity[c].z, velocity[c].z);
        }
        EXPECT_EQ(gas.species_histories(), histories);
        EXPECT_EQ(gas.population_escape_receipt(), number_escape);
        for (std::size_t face = 0; face < flux.size(); ++face)
            EXPECT_DOUBLE_EQ(f.flow->pressure_corrected_face_fluxes().value(static_cast<LO>(face)), flux[face]);
        EXPECT_EQ(gas.species_event_receipt(), receipt);
        saved.expect_restored(gas);
        f.flow->set_time_step(.05);
        f.coupled->step();
        EXPECT_NEAR(gas.global_species_moles(a), .1 + 301 * 1e-5 * .1 + 301.5 * 1e-5 * .05, 1e-13);
    }
}

TEST(ReactingReferenceScheduleTest, DefaultTypedOrderingRemainsBeforeTemperature)
{
    Fixture f;
    auto selected = options();
    selected.source_schedule = ReactingSourceSchedule::BeforeTemperature;
    auto& gas = f.initialize(std::move(selected));
    f.coupled->step();
    EXPECT_NEAR(gas.global_species_moles(a), .1 + 300 * 1e-5 * .1, 1e-13);
}

TEST(ReactingReferenceScheduleTest, FiniteLiquidFilmAndRankDivergentSelectionFailClosed)
{
    Fixture f;
    auto selected = options();
    selected.species_properties = {{a, 1e-8, 0}};
    EXPECT_ANY_THROW(f.coupled->configure_reacting_gas(selected));
    selected = options();
    const auto initialize = selected.initial_species_state;
    const auto catalog = selected.reaction_mechanism;
    selected.initial_species_state = [initialize, catalog](const R::ReactionEnvironment& c, std::int64_t gid)
    {
        auto state = initialize(c, gid);
        state.set_amount(catalog->catalog(), a, R::SpeciesPhase::Liquid, .01 * c.bulkVolume);
        return state;
    };
    EXPECT_ANY_THROW((void) f.initialize(selected));
    selected = options();
    const auto comm = f.grid->owned_cell_map()->getComm();
    if (comm->getSize() > 1 && comm->getRank() == comm->getSize() - 1)
        selected.source_schedule = ReactingSourceSchedule::BeforeTemperature;
    if (comm->getSize() > 1)
        EXPECT_ANY_THROW(f.coupled->configure_reacting_gas(selected));
    selected = options();
    if (comm->getRank() == comm->getSize() - 1)
        selected.source_schedule = static_cast<ReactingSourceSchedule>(99);
    // At four ranks the only malformed selector is on an actual empty owner.
    EXPECT_ANY_THROW(f.coupled->configure_reacting_gas(selected));
    selected = options();
    selected.reaction_mechanism.reset();
    EXPECT_ANY_THROW(f.coupled->configure_reacting_gas(selected));
}

TEST(ReactingReferenceScheduleTest, PopulationEscapeUsesIndependentNumberFluxAndRestoresItsAcceptedIdentity)
{
    Fixture f;
    f.coupled->initialize_heated_box(300, 300);
    auto selected = options();
    const auto catalog = selected.reaction_mechanism;
    selected.constant_slip_velocity = .2;
    selected.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
    selected.free_surface_patches = {"zmax"};
    selected.species_properties = {{a, 0, 0}, {b, 0, 0}};
    for (auto& property : selected.species_properties)
        property.carried_molar_enthalpy = [](const R::ReactionContext&, R::SpeciesPhase, std::string_view)
        { return 0.; }; // Explicit carried trace, no owned phase-energy debit.
    selected.initial_species_state = [catalog](const R::ReactionEnvironment& c, std::int64_t)
    {
        if (!c.location)
            throw std::invalid_argument("The count oracle requires the actual physical source frame.");
        const bool lower = c.location->centroid[2] < .5;
        R::CellSpeciesState state(catalog->catalog());
        state.dispersedPopulations = {{"micro", std::vector<double>(2)}, {"large", std::vector<double>(2)}};
        state.set_amount(
            catalog->catalog(), a, R::SpeciesPhase::Dispersed, (lower ? 1e-6 : 2e-6) * c.bulkVolume, "micro");
        state.set_amount(
            catalog->catalog(), b, R::SpeciesPhase::Dispersed, (lower ? 8e-6 : 3e-7) * c.bulkVolume, "large");
        return state;
    };
    selected.initial_population_numbers = [](const R::ReactionContext& c)
    {
        const bool lower = c.environment.location->centroid[2] < .5;
        return std::vector<double>{
            (lower ? 3. : 7.) * c.environment.bulkVolume, (lower ? 11. : 2.) * c.environment.bulkVolume};
    };
    selected.species_source = [](const R::ReactionContext&, double dt)
    {
        R::ReactionSourceInput input;
        if (dt > .1)
            input.heat.integratedEnergy = 1.; // A deterministic later rejected source.
        return input;
    };
    selected.population_process = {};
    ReactingGasModel<Pack, Grid> gas(f.grid, selected);
    gas.initialize_state(
        0, f.flow->temperature(), f.flow->pressure(), f.flow->velocity(), f.flow->material_properties());
    ScalarFaceFieldStored<Pack, Grid> carrier(f.grid, 0.0, "zero_carrier");
    const auto initial = gas.snapshot();
    const auto initial_receipt = gas.population_escape_receipt();
    EXPECT_FALSE(initial_receipt.event.accepted);
    auto advance = [&](double time, double dt)
    {
        gas.advance(time, dt, f.flow->temperature(), f.flow->pressure(), f.flow->velocity(), carrier,
            f.flow->material_properties());
    };
    advance(.1, .1);
    const auto receipt = gas.population_escape_receipt();
    EXPECT_EQ(receipt.event, gas.species_event_receipt());
    EXPECT_TRUE(receipt.event.accepted);
    ASSERT_EQ(receipt.events.size(), 2);
    constexpr double beta = .2 * .1 / .5;
    const auto upper = [beta](double lower_old, double upper_old)
    { return (upper_old + beta * lower_old / (1 + beta)) / (1 + beta); };
    const std::array<double, 2> expected_upper{upper(3, 7), upper(11, 2)};
    for (std::size_t p = 0; p < 2; ++p)
    {
        EXPECT_EQ(receipt.events[p].population, selected.population_ids[p]);
        EXPECT_NEAR(receipt.events[p].number, .1 * .2 * expected_upper[p], 2e-11);
        double local_number = 0;
        const auto& count = gas.population_number_field(selected.population_ids[p]);
        for (std::size_t c = 0; c < f.grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const bool lower = f.grid->cell_centroid(cell).z < .5;
            const double lower_old = p == 0 ? 3 : 11;
            const double expected = lower ? lower_old / (1 + beta) : expected_upper[p];
            EXPECT_NEAR(count.value(cell), expected, 2e-10);
            local_number += count.value(cell) * f.grid->cell_volume(cell);
        }
        double global_number = 0;
        Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_number, &global_number);
        EXPECT_NEAR(global_number + receipt.events[p].number, p == 0 ? 5. : 6.5, 2e-10);
    }
    EXPECT_ANY_THROW(advance(.3, .2));
    EXPECT_EQ(gas.population_escape_receipt(), receipt);
    gas.restore(initial);
    EXPECT_EQ(gas.population_escape_receipt(), initial_receipt);
    advance(.1, .1);
    EXPECT_EQ(gas.population_escape_receipt().events, receipt.events);
    EXPECT_EQ(gas.population_escape_receipt().event.revision, receipt.event.revision);
    EXPECT_GT(gas.population_escape_receipt().event.publication_nonce, receipt.event.publication_nonce);
    const auto comm = f.grid->owned_cell_map()->getComm();
    int empty = f.grid->num_owned_cells() == 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, std::max(0, comm->getSize() - 2));
}

} // namespace
