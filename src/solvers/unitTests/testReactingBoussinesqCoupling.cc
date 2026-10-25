#include "geometry/MeshHandle.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <utility>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
using Pack = S::DefaultTpetraTypes;
using Mesh = S::MeshHandle<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

S::ReactingGasOptions reacting_options(R::EnergyConvention convention = R::EnergyConvention::SensibleEnergy)
{
    using P = R::SpeciesPhase;
    const auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {3, "donor", .010, {{"X", 5}}, {P::Liquid}}, {8, "product", .004, {{"X", 2}}, {P::Liquid}},
        {41, "first dissolved gas", .002, {{"X", 1}}, {P::Dissolved, P::Dispersed}},
        {97, "second dissolved gas", .004, {{"X", 2}}, {P::Dissolved, P::Dispersed}}});
    S::ReactingGasOptions result;
    result.mode = S::ReactingGasMode::Species;
    result.reference_pressure = 100000.;
    result.surface_tension = 0.;
    result.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
        std::vector<R::ReactionDefinition>{{1, "manufactured two-gas reaction",
            {{3, P::Liquid, -1}, {8, P::Liquid, 1}, {41, P::Dissolved, 1}, {97, P::Dissolved, 1}},
            convention == R::EnergyConvention::SensibleEnergy ? 100. : 0.}},
        convention);
    const R::ThermodynamicBounds bounds{250., 600., 10000., 2000000.};
    result.gas_thermodynamics = R::make_ideal_gas_mixture(
        catalog, {{20., 300., 0.}, {20., 300., 0.}, {20., 300., 0.}, {20., 300., 0.}}, bounds, convention);
    result.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300., 100000., 1000., 4200., 0., 1e-9, 0., .001, 0.}; },
        convention);
    result.initial_species_state = [catalog](const R::ReactionEnvironment& context, std::int64_t)
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"micro", std::vector<double>(4)}, {"large", std::vector<double>(4)}};
        state.set_amount(*catalog, 3, P::Liquid, 100000. * context.bulkVolume);
        return state;
    };
    result.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{1, .25 * context.environment.bulkVolume}};
        source.heat.integratedEnergy = -2. * context.environment.bulkVolume;
        return source;
    };
    return result;
}

std::shared_ptr<S::BoussinesqSolver<Pack>> solver()
{
    auto geometry = std::make_shared<Mesh::Cartesian>(S::Vec3D<S::ArrReal>{{{0., 1.}, {0., 1.}, {0., .5, 1.}}});
    auto mesh = std::make_shared<Mesh>(std::move(geometry));
    S::TimeStepperOptions time;
    time.time_step = .1;
    time.reference_temperature = 300.;
    time.gravity_x = time.gravity_y = time.gravity_z = 0.;
    S::BoussinesqModelOptions material;
    material.reference_density = material.density = 1000.;
    material.specific_heat_capacity = 4200.;
    material.dynamic_viscosity = .001;
    material.thermal_conductivity = 0.;
    return std::make_shared<S::BoussinesqSolver<Pack>>(
        mesh, S::BoundaryConditionSet{}, time, S::LinearSolverOptions{}, material);
}

TEST(ReactingBoussinesqCouplingTest, ChemicalAndIndependentSignedHeatEnterOnceWithoutAPrescribedHeatSource)
{
    auto flow = solver();
    S::MultiphysicsSolverCoupler<Pack> coupling(flow);
    auto& gas = coupling.configure_reacting_gas(reacting_options());
    coupling.initialize_heated_box(300., 300.);
    EXPECT_EQ(coupling.find_prescribed_heat_source(), nullptr);
    EXPECT_NEAR(gas.global_species_moles(3), 100000., 1e-9);
    constexpr double heatPerStep = 100. * .25 - 2.;
    for (int step = 1; step <= 2; ++step)
    {
        coupling.step();
        EXPECT_NEAR(gas.global_species_moles(3), 100000. - .25 * step, 1e-9);
        EXPECT_NEAR(gas.global_species_moles(41), .25 * step, 1e-12);
        EXPECT_NEAR(gas.global_species_moles(97), .25 * step, 1e-12);
        for (std::size_t owned = 0; owned < flow->temperature().mesh().num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            EXPECT_NEAR(flow->temperature().value(cell), 300. + step * heatPerStep / (1000. * 4200.), 2e-10);
            EXPECT_NEAR(gas.thermal_source().value(cell), heatPerStep / .1, 1e-10);
        }
    }
}

TEST(ReactingBoussinesqCouplingTest, TotalVoidSourceSpansProductionAndHeatingForBothSchedules)
{
    using P = R::SpeciesPhase;
    for (const auto schedule :
        {S::ReactingSourceSchedule::BeforeTemperature, S::ReactingSourceSchedule::PrescribedCarrierAfterTemperature})
        for (const auto changes : {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}})
        {
            const auto [production, heating] = changes;
            SCOPED_TRACE(static_cast<int>(schedule));
            SCOPED_TRACE(production);
            SCOPED_TRACE(heating);
            auto flow = solver();
            S::MultiphysicsSolverCoupler<Pack> coupling(flow);
            auto selected = reacting_options();
            selected.source_schedule = schedule;
            const auto schema = selected.reaction_mechanism;
            selected.initial_species_state = [schema](const R::ReactionEnvironment& context, std::int64_t)
            {
                R::CellSpeciesState state(schema->catalog());
                state.dispersedPopulations = {{"micro", std::vector<double>(4)}, {"large", std::vector<double>(4)}};
                state.set_amount(schema->catalog(), 41, P::Dispersed, .1 * context.bulkVolume, "micro");
                return state;
            };
            selected.initial_population_numbers = [](const R::ReactionContext& context)
            { return std::vector<double>{1e5 * context.environment.bulkVolume, 0.}; };
            selected.species_source = [production](const R::ReactionContext& context, double dt)
            {
                R::ReactionSourceInput source;
                if (production)
                {
                    const double amount = .001 * dt * context.environment.bulkVolume;
                    source.externalTransfers.push_back(
                        {"whole-step gas source", 41, P::Dispersed, amount, .002 * amount, 0., {}, "micro"});
                }
                return source;
            };
            selected.population_process = [production](
                                              const S::BubbleGasPhysics::MultispeciesPopulationContext& c, double dt)
            {
                S::BubbleGasPhysics::MultispeciesPopulationPacket packet;
                if (production)
                    packet.numberEvents.push_back({"micro", 1000. * dt * c.chemistry.environment.bulkVolume, 0.});
                return packet;
            };
            // The temperature owner supplies heat independently in both schedules; the gas-only
            // reference has no finite-liquid energy storage or post-temperature source heat.
            if (heating)
                coupling.add_prescribed_heat_source().initialize_constant(42000000.);
            auto& gas = coupling.configure_reacting_gas(selected);
            coupling.initialize_heated_box(300., 300.);
            const auto* scalar = coupling.find_scalar_void_fraction_model();
            ASSERT_NE(scalar, nullptr);
            std::vector<double> previous;
            for (std::size_t owned = 0; owned < flow->temperature().mesh().num_owned_cells(); ++owned)
                previous.push_back(scalar->alpha_g().value(static_cast<Pack::local_ordinal_type>(owned)));
            for (int step = 1; step <= 2; ++step)
            {
                coupling.step();
                EXPECT_NEAR(gas.global_species_moles(41), .1 + (production ? .0001 * step : 0.), 1e-13);
                for (std::size_t owned = 0; owned < previous.size(); ++owned)
                {
                    const auto cell = static_cast<Pack::local_ordinal_type>(owned);
                    const double alpha = scalar->alpha_g().value(cell);
                    const double expected = (.1 + (production ? .0001 * step : 0.)) * 8.31446261815324 *
                                            flow->temperature().value(cell) / 100000.;
                    EXPECT_NEAR(alpha, expected, 1e-13);
                    EXPECT_GT(alpha, previous[owned]);
                    EXPECT_NEAR(scalar->source_alpha_total().value(cell), (alpha - previous[owned]) / .1, 1e-13);
                    previous[owned] = alpha;
                }
            }
        }
}

TEST(ReactingBoussinesqCouplingTest, FormationEnergyStorageIsCollectivelyRejectedByTheSensibleTemperatureSolver)
{
    auto flow = solver();
    S::MultiphysicsSolverCoupler<Pack> coupling(flow);
    EXPECT_ANY_THROW(coupling.configure_reacting_gas(reacting_options(R::EnergyConvention::IncludesFormationEnergy)));
}
TEST(ReactingBoussinesqCouplingTest, RejectedOverbudgetSecondIntervalRestoresAcceptedFlowSpeciesAndThermalLedgers)
{
    auto flow = solver();
    S::MultiphysicsSolverCoupler<Pack> coupling(flow);
    auto selected = reacting_options();
    selected.species_source = [](const R::ReactionContext& context, double duration)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{1, 2.5 * duration * context.environment.bulkVolume}};
        source.heat.integratedEnergy = -20. * duration * context.environment.bulkVolume;
        return source;
    };
    auto& gas = coupling.configure_reacting_gas(selected);
    coupling.initialize_heated_box(300., 300.);
    coupling.step();
    const double acceptedTime = flow->time();
    std::vector<double> temperatures;
    std::vector<S::vec3<double>> velocities;
    std::vector<std::vector<double>> species;
    const auto acceptedLedgers = gas.source_ledgers();
    for (std::size_t owned = 0; owned < flow->temperature().mesh().num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        temperatures.push_back(flow->temperature().value(cell));
        velocities.push_back(flow->velocity().value(cell));
        species.push_back(gas.cell_species_state(cell).amounts);
    }
    flow->set_time_step(100000.); // Same declared rate now requests more donor than exists.
    EXPECT_ANY_THROW(coupling.step());
    EXPECT_DOUBLE_EQ(flow->time(), acceptedTime);
    EXPECT_NEAR(gas.global_species_moles(3), 99999.75, 1e-9);
    EXPECT_NEAR(gas.global_species_moles(41), .25, 1e-12);
    EXPECT_EQ(gas.source_ledgers().size(), acceptedLedgers.size());
    for (std::size_t owned = 0; owned < temperatures.size(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(flow->temperature().value(cell), temperatures[owned]);
        const auto actualVelocity = flow->velocity().value(cell);
        EXPECT_DOUBLE_EQ(actualVelocity.x, velocities[owned].x);
        EXPECT_DOUBLE_EQ(actualVelocity.y, velocities[owned].y);
        EXPECT_DOUBLE_EQ(actualVelocity.z, velocities[owned].z);
        EXPECT_EQ(gas.cell_species_state(cell).amounts, species[owned]);
        EXPECT_EQ(gas.source_ledgers()[owned].reactionExtents, acceptedLedgers[owned].reactionExtents);
        EXPECT_DOUBLE_EQ(gas.source_ledgers()[owned].reactionHeat, acceptedLedgers[owned].reactionHeat);
        EXPECT_DOUBLE_EQ(gas.source_ledgers()[owned].externalHeat, acceptedLedgers[owned].externalHeat);
        EXPECT_NEAR(gas.thermal_source().value(cell), 230., 1e-10);
    }
    flow->set_time_step(.1);
    coupling.step();
    EXPECT_NEAR(gas.global_species_moles(41), .5, 1e-12);
    EXPECT_NEAR(gas.global_species_moles(97), .5, 1e-12);
}
} // namespace
