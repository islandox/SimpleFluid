#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
namespace B = S::BubbleGasPhysics;
using Pack = S::DefaultTpetraTypes;
using Grid = S::MeshHandle<Pack>;
using Model = S::ReactingGasModel<Pack, Grid>;
using P = R::SpeciesPhase;
constexpr R::SpeciesId solvent = 44, one = 101, two = 709;
constexpr double gas_constant = 8.31446261815324;
enum class Case
{
    Replacement,
    UnderflowLater,
    OverdrawLater,
    Dissolution,
    Film
};
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<Grid> grid()
{
    Grid::DistributionOptions d;
    d.allow_empty_partitions = true;
    // Each control volume is2m3: a positive minimum-subnormal mol amount cannot
    // be stored as its mol/m3 density, which must reject rather than disappear.
    return std::make_shared<Grid>(
        std::make_shared<Grid::Cartesian>(S::Vec3D<S::ArrReal>{{{0., 2.}, {0., 1.}, {0., 1., 2.}}}), d);
}
S::ReactingGasOptions options(Case selected)
{
    S::ReactingGasOptions o;
    o.mode = S::ReactingGasMode::TwoPopulation;
    o.reference_pressure = 100000;
    o.surface_tension = 0;
    o.min_radius = 1e-12;
    o.max_radius = .1;
    o.dissolved_transport = S::ReactingTransportMode::Advective;
    o.mass_transfer_mode = S::BubbleMassTransferMode::SphericalDiffusion;
    auto catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{solvent, "finite solvent", .018, {{"S", 1}}, {P::Liquid}},
            {one, "ordinary gas one", .002, {{"G", 1}}, {P::Dissolved, P::Dispersed}},
            {two, "ordinary gas two", .004, {{"G", 2}}, {P::Dissolved, P::Dispersed}}});
    o.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(
        catalog, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::SensibleEnergy);
    const R::ThermodynamicBounds bounds{250, 600, 1000, 1e7};
    o.gas_thermodynamics = R::make_ideal_gas_mixture(
        catalog, {{20, 300, 0}, {20, 300, 0}, {40, 300, 0}}, bounds, R::EnergyConvention::SensibleEnergy);
    o.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4200, 0, 0, 0, .001, 0}; },
        R::EnergyConvention::SensibleEnergy);
    for (const auto id : {one, two})
    {
        S::ReactingSpeciesProperties property;
        property.species = id;
        property.diffusivity = selected == Case::Film ? (id == one ? 4.5e-9 : 5.1886e-9) : 0;
        property.henry_coefficient = id == one ? 1e-5 : 2e-5;
        property.carried_molar_enthalpy = [id](const R::ReactionContext& c, P phase, std::string_view)
        {
            if (phase == P::Liquid || phase == P::Dissolved)
                return (id == one ? .002 : .004) *
                       (4200 * (c.environment.temperature - 300) + c.environment.absolutePressure / 1000);
            return (id == one ? 20 : 40) * (c.environment.temperature - 300) + gas_constant * c.environment.temperature;
        };
        o.species_properties.push_back(std::move(property));
    }
    o.initial_species_state = [catalog, selected](const R::ReactionEnvironment& e, std::int64_t)
    {
        R::CellSpeciesState state(*catalog);
        state.dispersedPopulations = {{"micro", std::vector<double>(3)}, {"large", std::vector<double>(3)}};
        const double n1 = (selected == Case::Film ? 1e-7 : 3e-6) * e.bulkVolume;
        const double n2 = (selected == Case::Film ? 2e-7 : 0) * e.bulkVolume;
        const double liquid = e.bulkVolume - (n1 + n2) * gas_constant * e.temperature / e.absolutePressure;
        const double d1 = (selected == Case::Film ? 10 : 1e-4) * liquid;
        const double d2 = (selected == Case::Film ? 20 : 0) * liquid;
        state.set_amount(*catalog, solvent, P::Liquid, (1000 * liquid - .002 * d1 - .004 * d2) / .018);
        state.set_amount(*catalog, one, P::Dissolved, d1);
        state.set_amount(*catalog, two, P::Dissolved, d2);
        state.set_amount(*catalog, one, P::Dispersed, n1, "micro");
        state.set_amount(*catalog, two, P::Dispersed, n2, "micro");
        return state;
    };
    o.initial_population_numbers = [selected](const R::ReactionContext& c)
    { return std::vector<double>{(selected == Case::Film ? 1e6 : 2e8) * c.environment.bulkVolume, 0}; };
    if (selected == Case::Dissolution)
        o.micro_dissolution_frequency = 2;
    else if (selected != Case::Film)
        o.population_process = [selected](const B::MultispeciesPopulationContext& c, double)
        {
            B::MultispeciesPopulationPacket packet;
            const auto& e = c.chemistry.environment;
            const double old = c.chemistry.state.amount(c.chemistry.catalog, one, P::Dispersed, "micro");
            const double old_count = c.populations.front().number;
            if (e.time == 0 || selected == Case::UnderflowLater)
            {
                const double incoming = e.time == 0 ? 2e-302 * e.bulkVolume : std::numeric_limits<double>::denorm_min();
                const double count = e.time == 0 ? 1e-284 * e.bulkVolume : 1e-306 * e.bulkVolume;
                packet.transfers = {{one, P::Dispersed, P::Dissolved, "micro", {}, old},
                    {one, P::Dissolved, P::Dispersed, {}, "micro", incoming}};
                packet.numberEvents = {{"micro", count, old_count}};
            }
            else if (selected == Case::OverdrawLater)
                packet.transfers = {{one, P::Dispersed, P::Dissolved, "micro", {}, .6 * old},
                    {one, P::Dissolved, P::Dispersed, {}, "micro", .3 * old},
                    {one, P::Dispersed, P::Dissolved, "micro", {}, .6 * old}};
            return packet;
        };
    return o;
}
struct Fixture
{
    std::shared_ptr<Grid> mesh = grid();
    Model model;
    Model::field_type temperature{mesh, 300., "T"}, pressure{mesh, 0., "p"};
    Model::velocity_field_type velocity{mesh, Grid::Vec3{}, "U"};
    Model::face_flux_field_type flux{mesh, 0., "phi"};
    Model::material_type material;
    explicit Fixture(Case selected = Case::Replacement)
        : model(mesh, options(selected)), material(
                                              mesh,
                                              []
                                              {
                                                  S::BoussinesqModelOptions p;
                                                  p.reference_density = p.density = 1000;
                                                  p.specific_heat_capacity = 4200;
                                                  p.dynamic_viscosity = .001;
                                                  p.thermal_conductivity = 0;
                                                  return p;
                                              }(),
                                              S::TimeStepperOptions{})
    {
        model.initialize_state(0, temperature, pressure, velocity, material);
    }
    void advance(double end, double dt = .001)
    {
        model.advance(end, dt, temperature, pressure, velocity, flux, material, nullptr);
    }
    void expect_tiny() const
    {
        for (std::size_t row = 0; row < mesh->num_local_cells(); ++row)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(row);
            EXPECT_DOUBLE_EQ(model.species_field(one, P::Dispersed, "micro").local_value(cell), 2e-302);
            EXPECT_DOUBLE_EQ(model.population_number_field("micro").local_value(cell), 1e-284);
        }
    }
};
TEST(PopulationRawMaterialReplayTest, TinyCompleteDonorReplacementPublishesBothMolesAndNumbersAndReplays)
{
    Fixture f;
    const auto initial = f.model.snapshot();
    f.advance(.001);
    f.expect_tiny();
    const auto histories = f.model.species_histories();
    const auto receipt = f.model.species_event_receipt();
    const auto ledgers = f.model.population_ledgers();
    f.model.restore(initial);
    f.advance(.001);
    f.expect_tiny();
    EXPECT_EQ(f.model.species_histories(), histories);
    EXPECT_EQ(f.model.species_event_receipt().revision, receipt.revision);
    EXPECT_GT(f.model.species_event_receipt().publication_nonce, receipt.publication_nonce);
    ASSERT_EQ(f.model.population_ledgers().size(), ledgers.size());
    for (std::size_t i = 0; i < ledgers.size(); ++i)
        EXPECT_EQ(f.model.population_ledgers()[i].chemicalChange.dispersedPopulations.front().amounts,
            ledgers[i].chemicalChange.dispersedPopulations.front().amounts);
    int local_empty = f.mesh->num_owned_cells() == 0, empty = 0;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (comm->getSize() == 4)
        EXPECT_EQ(empty, 2);
}
TEST(PopulationRawMaterialReplayTest, StoredDensityUnderflowRejectsWithoutErasingAnEarlierTinyPhase)
{
    Fixture f(Case::UnderflowLater);
    const auto initial = f.model.snapshot();
    f.advance(.001);
    f.expect_tiny();
    const auto accepted = f.model.snapshot();
    const auto histories = f.model.species_histories();
    const auto receipt = f.model.species_event_receipt();
    EXPECT_THROW(f.advance(.002), std::exception);
    f.expect_tiny();
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    EXPECT_EQ(f.model.species_histories(), histories);
    f.model.restore(initial);
    f.advance(.001);
    f.expect_tiny();
    f.model.restore(accepted);
    f.expect_tiny();
}
TEST(PopulationRawMaterialReplayTest, IncomingMaterialCannotFinanceCompetingGrossOutflow)
{
    Fixture f(Case::OverdrawLater);
    f.advance(.001);
    f.expect_tiny();
    const auto receipt = f.model.species_event_receipt();
    const auto histories = f.model.species_histories();
    EXPECT_THROW(f.advance(.002), std::exception);
    f.expect_tiny();
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
    EXPECT_EQ(f.model.species_histories(), histories);
}
TEST(PopulationRawMaterialReplayTest, BuiltInDeclaredDissolutionRetainsItsOriginalMaterialEvents)
{
    Fixture f(Case::Dissolution);
    f.advance(.001);
    const double retention = std::exp(-.002), fraction = -std::expm1(-.002);
    for (std::size_t row = 0; row < f.mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        EXPECT_NEAR(f.model.species_field(one, P::Dispersed, "micro").value(cell), 3e-6 * retention, 1e-16);
        EXPECT_NEAR(f.model.population_number_field("micro").value(cell), 2e8 * retention, 2e8 * 1e-12);
        const auto& catalog = f.model.options().reaction_mechanism->catalog();
        const double accepted = f.model.population_ledgers()[row].chemicalChange.amount(catalog, one, P::Dissolved);
        EXPECT_NEAR(accepted, 3e-6 * f.mesh->cell_volume(cell) * fraction, 1e-16);
    }
}
TEST(PopulationRawMaterialReplayTest, BuiltInTwoSpeciesFilmReceiptsSatisfyTheIndependentEOSAndSelectedDOverR)
{
    Fixture f(Case::Film);
    constexpr double dt = 1e-6;
    f.advance(dt, dt);
    for (std::size_t row = 0; row < f.mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        const double cv = f.mesh->cell_volume(cell);
        const double number = 1e6 * cv, initial_volume = 3e-7 * cv * gas_constant * 300 / 100000;
        std::array<double, 2> inferred{};
        for (std::size_t species = 0; species < 2; ++species)
        {
            const auto id = species == 0 ? one : two;
            const double D = species == 0 ? 4.5e-9 : 5.1886e-9;
            const double concentration = species == 0 ? 10 : 20, henry = species == 0 ? 1e-5 : 2e-5;
            const double fraction = species == 0 ? 1. / 3 : 2. / 3;
            const auto& catalog = f.model.options().reaction_mechanism->catalog();
            const double accepted =
                f.model.population_ledgers()[row].chemicalChange.amount(catalog, id, P::Dispersed, "micro");
            ASSERT_GT(accepted, 0);
            // D/r *4pi*r^2*N*(C-H*x*p) is the supplied ideal-mixture film law.
            inferred[species] =
                accepted / (dt * 4 * std::numbers::pi * D * number * (concentration - henry * fraction * 100000));
            const double geometric =
                number * 4 * std::numbers::pi * inferred[species] * inferred[species] * inferred[species] / 3;
            // Keep the production EOS inversion residual gate explicit, separate from arithmetic roundoff.
            EXPECT_LE(std::abs(geometric - initial_volume),
                (4e-12 + 128 * std::numeric_limits<double>::epsilon()) * initial_volume);
        }
        EXPECT_NEAR(inferred[0], inferred[1], 128 * std::numeric_limits<double>::epsilon() * inferred[0]);
    }
}
} // namespace
