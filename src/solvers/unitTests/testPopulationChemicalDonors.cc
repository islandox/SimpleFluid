/** @file
 * @brief Original donor authority across chemistry and later material stages.
 */
#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <string>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
namespace B = S::BubbleGasPhysics;
using Pack = S::DefaultTpetraTypes;
using Grid = S::MeshHandle<Pack>;
using Model = S::ReactingGasModel<Pack, Grid>;
using P = R::SpeciesPhase;
using LO = Pack::local_ordinal_type;
constexpr R::SpeciesId left = 0, right = 17;
constexpr double large_stock = 1e20, tiny_credit = 1e-280;
enum class Case
{
    ChemicalOverdraw,
    ExternalOverdraw,
    ProductFunding,
    SameStageBorrowing,
    TinyReplacement
};
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Grid> grid()
{
    Grid::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Grid>(
        std::make_shared<Grid::Cartesian>(S::Vec3D<S::ArrReal>{{{0., 1.}, {0., 1.}, {0., 1., 2.}}}), distribution);
}
S::ReactingGasOptions options(Case selected)
{
    const auto catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{left, "synthetic isomer left", 1e-20, {{"X", 1}}, {P::Liquid, P::Dissolved}},
            {right, "synthetic isomer right", 1e-20, {{"X", 1}}, {P::Liquid, P::Dissolved}}});
    S::ReactingGasOptions result;
    result.mode = S::ReactingGasMode::Species;
    result.population_ids.clear();
    result.reference_pressure = 100000;
    result.surface_tension = 0;
    result.solution_transport = S::ReactingTransportMode::Advective;
    result.reaction_advance.maximumRetries = 0;
    result.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
        std::vector<R::ReactionDefinition>{
            {41, "finite synthetic isomerization", {{left, P::Liquid, -1}, {right, P::Liquid, 1}}, 0}},
        R::EnergyConvention::SensibleEnergy);
    const R::ThermodynamicBounds bounds{250, 600, 1000, 1e7};
    result.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds,
        [](std::span<const double>) { return R::LinearCompressibleMixtureLiquid{300, 100000, 1, 2, 0, 0, 0, 0, 0}; },
        R::EnergyConvention::SensibleEnergy);
    // No gas is present. The supplied gas provider is a required field-schema property, not a gas donor.
    result.gas_thermodynamics =
        R::make_ideal_gas_mixture(catalog, {{20, 300, 0}, {20, 300, 0}}, bounds, R::EnergyConvention::SensibleEnergy);
    result.initial_species_state = [catalog](const R::ReactionEnvironment& e, std::int64_t)
    {
        R::CellSpeciesState state(*catalog);
        state.set_amount(*catalog, left, P::Liquid, large_stock * e.bulkVolume);
        return state;
    };
    result.initial_population_numbers = [](const R::ReactionContext&) { return std::vector<double>{}; };
    result.species_source = [selected](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput input;
        if (selected == Case::ExternalOverdraw)
            input.externalTransfers = {{"declared synthetic withdrawal", left, P::Liquid, -1, -1e-20, 0, {}, {}}};
        else
        {
            // Deplete the actual transported donor. A zero-flux linear solve can
            // differ from its initial stock by an arithmetic ulp at this scale.
            input.integratedExtents = {
                {41, selected == Case::TinyReplacement ? context.state.amount(context.catalog, left, P::Liquid) : 1}};
            if (selected == Case::TinyReplacement)
                input.externalTransfers = {
                    {"declared tiny synthetic feed", left, P::Liquid, tiny_credit, tiny_credit * 1e-20, 0, {}, {}}};
        }
        return input;
    };
    result.population_process = [selected](const B::MultispeciesPopulationContext& context, double)
    {
        B::MultispeciesPopulationPacket packet;
        if (selected == Case::ChemicalOverdraw || selected == Case::ExternalOverdraw)
            packet.transfers = {{left, P::Liquid, P::Dissolved, {}, {},
                context.chemistry.state.amount(context.chemistry.catalog, left, P::Liquid)}};
        else if (selected == Case::ProductFunding)
            packet.transfers = {{right, P::Liquid, P::Dissolved, {}, {}, 1}};
        else if (selected == Case::SameStageBorrowing)
            packet.transfers = {
                {right, P::Liquid, P::Dissolved, {}, {}, 1}, {right, P::Dissolved, P::Liquid, {}, {}, 1}};
        else
            packet.transfers = {{left, P::Liquid, P::Dissolved, {}, {}, tiny_credit}};
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

    explicit Fixture(Case selected)
        : model(mesh, options(selected)), material(
                                              mesh,
                                              []
                                              {
                                                  S::BoussinesqModelOptions p;
                                                  p.reference_density = p.density = 1;
                                                  p.specific_heat_capacity = 2;
                                                  p.dynamic_viscosity = p.thermal_conductivity = 0;
                                                  return p;
                                              }(),
                                              S::TimeStepperOptions{})
    {
        model.initialize_state(0, temperature, pressure, velocity, material);
        const auto comm = mesh->owned_cell_map()->getComm();
        int local_empty = mesh->num_owned_cells() == 0, actual_empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &actual_empty);
        EXPECT_EQ(actual_empty, std::max(0, comm->getSize() - 2));
        EXPECT_EQ(mesh->owned_cell_map()->getGlobalNumElements(), 2u);
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
            EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<LO>(c)), 1);
    }
    void advance() { model.advance(.01, .01, temperature, pressure, velocity, flux, material); }
    auto fields() const
    {
        std::map<std::string, std::vector<double>> result;
        for (const auto& [name, field] : model.output_fields())
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                result[name].push_back(field->value(static_cast<LO>(c)));
        return result;
    }
};
TEST(PopulationChemicalDonorTest, ChemicalAndExternalSubUlpDebitsCannotFundAFullLaterMaterialWithdrawal)
{
    for (const auto selected : {Case::ChemicalOverdraw, Case::ExternalOverdraw})
    {
        SCOPED_TRACE(static_cast<int>(selected));
        Fixture f(selected);
        const auto initial = f.model.snapshot();
        const auto fields = f.fields();
        const auto histories = f.model.species_histories();
        const auto receipt = f.model.species_event_receipt();
        EXPECT_FALSE(receipt.accepted);
        bool rejected = false;
        std::string failure;
        try
        {
            f.advance();
        }
        catch (const std::exception& error)
        {
            rejected = true;
            failure = error.what();
        }
        EXPECT_TRUE(rejected);
        if (f.mesh->num_owned_cells() > 0)
            EXPECT_NE(failure.find("overdraw an original material donor"), std::string::npos);
        EXPECT_EQ(f.fields(), fields);
        EXPECT_EQ(f.model.species_histories(), histories);
        EXPECT_EQ(f.model.species_event_receipt(), receipt);
        EXPECT_TRUE(f.model.source_ledgers().empty());
        EXPECT_TRUE(f.model.population_ledgers().empty());
        EXPECT_TRUE(f.model.escape_ledgers().empty());
        EXPECT_NO_THROW(f.model.restore(initial));
        EXPECT_EQ(f.fields(), fields);
    }
}
TEST(PopulationChemicalDonorTest, EarlierChemicalProductFundsALaterMaterialStageAndReplayKeepsProvenance)
{
    Fixture f(Case::ProductFunding);
    const auto initial = f.model.snapshot();
    f.advance();
    EXPECT_DOUBLE_EQ(f.model.global_species_moles(right), 2);
    for (std::size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto& catalog = f.model.options().reaction_mechanism->catalog();
        EXPECT_DOUBLE_EQ(f.model.species_field(right, P::Liquid).value(cell), 0);
        EXPECT_DOUBLE_EQ(f.model.species_field(right, P::Dissolved).value(cell), 1);
        EXPECT_DOUBLE_EQ(f.model.source_ledgers()[c].chemicalChange.amount(catalog, right, P::Liquid), 1);
        EXPECT_DOUBLE_EQ(f.model.population_ledgers()[c].chemicalChange.amount(catalog, right, P::Liquid), -1);
        EXPECT_DOUBLE_EQ(f.model.population_ledgers()[c].chemicalChange.amount(catalog, right, P::Dissolved), 1);
    }
    const auto accepted_fields = f.fields();
    const auto histories = f.model.species_histories();
    const auto receipt = f.model.species_event_receipt();
    f.model.restore(initial);
    f.advance();
    EXPECT_EQ(f.fields(), accepted_fields);
    EXPECT_EQ(f.model.species_histories(), histories);
    EXPECT_EQ(f.model.species_event_receipt().revision, receipt.revision);
    EXPECT_GT(f.model.species_event_receipt().publication_nonce, receipt.publication_nonce);
}
TEST(PopulationChemicalDonorTest, IncomingMaterialInTheSamePopulationPacketCannotFundItsOwnDonor)
{
    Fixture f(Case::SameStageBorrowing);
    const auto fields = f.fields();
    const auto histories = f.model.species_histories();
    const auto receipt = f.model.species_event_receipt();
    EXPECT_THROW(f.advance(), std::exception);
    EXPECT_EQ(f.fields(), fields);
    EXPECT_EQ(f.model.species_histories(), histories);
    EXPECT_EQ(f.model.species_event_receipt(), receipt);
}
TEST(PopulationChemicalDonorTest, TinySourceCreditSurvivesCompleteChemicalReplacementAndLaterPhaseTransfer)
{
    Fixture f(Case::TinyReplacement);
    const auto initial = f.model.snapshot();
    f.advance();
    for (std::size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto& catalog = f.model.options().reaction_mechanism->catalog();
        EXPECT_DOUBLE_EQ(f.model.species_field(left, P::Liquid).value(cell), 0);
        EXPECT_DOUBLE_EQ(f.model.species_field(left, P::Dissolved).value(cell), tiny_credit);
        EXPECT_DOUBLE_EQ(f.model.species_field(right, P::Liquid).value(cell), large_stock);
        EXPECT_DOUBLE_EQ(f.model.population_ledgers()[c].chemicalChange.amount(catalog, left, P::Liquid), -tiny_credit);
        EXPECT_DOUBLE_EQ(
            f.model.population_ledgers()[c].chemicalChange.amount(catalog, left, P::Dissolved), tiny_credit);
        EXPECT_EQ(f.model.source_ledgers()[c].externalTransfers.size(), 1u);
        if (f.model.source_ledgers()[c].externalTransfers.size() == 1)
            EXPECT_DOUBLE_EQ(f.model.source_ledgers()[c].externalTransfers[0].moles, tiny_credit);
    }
    EXPECT_DOUBLE_EQ(f.model.global_species_moles(left), 2 * tiny_credit);
    const auto accepted_fields = f.fields();
    const auto receipt = f.model.species_event_receipt();
    f.model.restore(initial);
    f.advance();
    EXPECT_EQ(f.fields(), accepted_fields);
    EXPECT_GT(f.model.species_event_receipt().publication_nonce, receipt.publication_nonce);
}
} // namespace
