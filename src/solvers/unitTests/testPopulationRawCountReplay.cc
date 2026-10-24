#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
namespace B = S::BubbleGasPhysics;
using Pack = S::DefaultTpetraTypes;
using Grid = S::MeshHandle<Pack>;
using Model = S::ReactingGasModel<Pack, Grid>;
using P = R::SpeciesPhase;
constexpr R::SpeciesId solvent = 44, gas = 101;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<Grid> grid()
{
    Grid::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Grid>(
        std::make_shared<Grid::Cartesian>(S::Vec3D<S::ArrReal>{{{0., 1.}, {0., 1.}, {0., .5, 1.}}}), distribution);
}
S::ReactingGasOptions options(bool fail_later)
{
    S::ReactingGasOptions result;
    result.mode = S::ReactingGasMode::TwoPopulation;
    result.reference_pressure = 100000;
    result.surface_tension = 0;
    result.min_radius = 1e-12;
    result.max_radius = .1;
    result.dissolved_transport = S::ReactingTransportMode::Advective;
    auto catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{solvent, "finite solvent", .018, {{"S", 1}}, {P::Liquid}},
            {gas, "ordinary gas", .002, {{"G", 1}}, {P::Dissolved, P::Dispersed}}});
    result.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(
        catalog, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::SensibleEnergy);
    const R::ThermodynamicBounds bounds{250, 600, 1000, 1e7};
    result.gas_thermodynamics =
        R::make_ideal_gas_mixture(catalog, {{20, 300, 0}, {20, 300, 0}}, bounds, R::EnergyConvention::SensibleEnergy);
    result.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4200, 0, 0, 0, .001, 0}; },
        R::EnergyConvention::SensibleEnergy);
    result.initial_species_state = [catalog](const R::ReactionEnvironment& e, std::int64_t)
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"micro", std::vector<double>(2)}, {"large", std::vector<double>(2)}};
        const double gas_moles = 1e-6 * e.bulkVolume;
        const double gas_volume = gas_moles * 8.31446261815324 * e.temperature / e.absolutePressure;
        state.set_amount(*catalog, solvent, P::Liquid, 1000 * (e.bulkVolume - gas_volume) / .018);
        state.set_amount(*catalog, gas, P::Dispersed, gas_moles, "micro");
        return state;
    };
    result.initial_population_numbers = [](const R::ReactionContext& c)
    { return std::vector<double>{1e20 * c.environment.bulkVolume, 0}; };
    result.population_process = [fail_later](const B::MultispeciesPopulationContext& c, double)
    {
        B::MultispeciesPopulationPacket packet;
        if (c.chemistry.environment.time == 0)
            packet.numberEvents = {{"micro", c.chemistry.environment.bulkVolume, c.populations.front().number}};
        else if (fail_later)
            packet.numberEvents = {{"micro", 0, c.populations.front().number}}; // Occupied gas cannot lose all N.
        return packet;
    };
    return result;
}
struct Fixture
{
    std::shared_ptr<Grid> mesh = grid();
    Model model;
    Model::field_type temperature{mesh, 300., "T"}, pressure{mesh, 0., "p"};
    Model::velocity_field_type velocity{mesh, Grid::Vec3{}, "U"};
    Model::face_flux_field_type flux{mesh, 0., "phi"};
    Model::material_type material;
    explicit Fixture(bool fail_later = false)
        : model(mesh, options(fail_later)), material(
                                                mesh,
                                                []
                                                {
                                                    S::BoussinesqModelOptions value;
                                                    value.reference_density = value.density = 1000;
                                                    value.specific_heat_capacity = 4200;
                                                    value.dynamic_viscosity = .001;
                                                    value.thermal_conductivity = 0;
                                                    return value;
                                                }(),
                                                S::TimeStepperOptions{})
    {
        model.initialize_state(0, temperature, pressure, velocity, material);
    }
    void advance(double end) { model.advance(end, .001, temperature, pressure, velocity, flux, material, nullptr); }
    void expect_density(double expected) const
    {
        for (std::size_t row = 0; row < mesh->num_local_cells(); ++row)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(row);
            EXPECT_DOUBLE_EQ(model.population_number_field("micro").local_value(cell), expected);
            EXPECT_NEAR(model.species_field(gas, P::Dispersed, "micro").local_value(cell), 1e-6, 1e-20);
        }
    }
};
TEST(PopulationRawCountReplayTest, StockSeededReceiptSurvivesPublicationAndCheckpointReplay)
{
    Fixture f;
    const auto initial = f.model.snapshot();
    f.expect_density(1e20);
    f.advance(.001);
    f.expect_density(1);
    const auto receipt = f.model.species_event_receipt();
    const auto histories = f.model.species_histories();
    f.advance(.002);
    f.expect_density(1);
    f.model.restore(initial);
    f.expect_density(1e20);
    f.advance(.001);
    f.expect_density(1);
    EXPECT_EQ(f.model.species_histories(), histories);
    EXPECT_EQ(f.model.species_event_receipt().revision, receipt.revision);
    EXPECT_GT(f.model.species_event_receipt().publication_nonce, receipt.publication_nonce);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    int local_empty = f.mesh->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (comm->getSize() == 4)
        EXPECT_EQ(empty, 2);
}
TEST(PopulationRawCountReplayTest, LaterUnrepresentableEmptyCountRejectsAndPreservesEarlierSnapshot)
{
    Fixture f(true);
    const auto initial = f.model.snapshot();
    f.advance(.001);
    f.expect_density(1);
    const auto accepted = f.model.snapshot();
    const auto receipt = f.model.species_event_receipt();
    const auto histories = f.model.species_histories();
    EXPECT_THROW(f.advance(.002), std::exception);
    f.expect_density(1);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    EXPECT_EQ(f.model.species_histories(), histories);
    f.model.restore(initial);
    f.advance(.001);
    f.expect_density(1);
    f.model.restore(accepted);
    f.expect_density(1);
}
} // namespace
