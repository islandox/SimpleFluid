/** @file testSpeciesSurfaceAccounting.cc
 * @brief regressions for exact accepted event receipts and subset budgets.
 * Proposed serial + registered two/four-rank execution; not yet built or run.
 */
#include "solvers/ClosedMixtureHeadspaceModel.hh"
#include "solvers/details/SpeciesSurfaceAccounting.hh"
#include <gtest/gtest.h>
#include <limits>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
namespace D = S::detail::surface_species;
std::shared_ptr<const R::SpeciesCatalog> catalog(bool reordered = false)
{
    using P = R::SpeciesPhase;
    std::vector<R::SpeciesDefinition> species{{9, "G_A", .02, {{"X", 1}}, {P::Dissolved, P::Resolved, P::Dispersed}},
        {701, "G_B", .04, {{"Y", 1}}, {P::Dissolved, P::Resolved, P::Dispersed}}};
    if (reordered)
        std::reverse(species.begin(), species.end());
    return std::make_shared<const R::SpeciesCatalog>(std::move(species), 3);
}
R::SpeciesEventReceipt initial_receipt(const R::SpeciesCatalog& schema)
{
    return {false, 0, 1, schema.fingerprint(), 0, 0};
}
R::SpeciesEventReceipt next_receipt(const R::SpeciesEventReceipt& accepted, double end = .1, double dt = .1)
{
    return {true, accepted.revision + 1, accepted.initialization_generation, accepted.schema_fingerprint, end, dt,
        accepted.publication_nonce + 1};
}
TEST(SpeciesSurfaceAccountingTest, SignedSubsetNetSurvivesRoundedCancellingGrossReports)
{
    const auto schema = catalog();
    D::Accounting accepted;
    accepted.initialize(*schema, initial_receipt(*schema), {{"G_A", 1.}, {"G_B", 2.}});
    std::vector<R::SpeciesCompartmentEvents> events(1);
    events[0].species = 9;
    events[0].phase = R::SpeciesPhase::Dissolved;
    events[0].chemical_production = 1e20;
    events[0].chemical_consumption = 1e20;
    const std::vector<R::SpeciesSubsetSourceChange> signed_change{{9, 1e-12, 1e-12}, {701, 0, 0}};
    S::GasMolesBySpecies escaped;
    double source_mass = 0, outflow_mass = 0;
    const auto trial = accepted.preview(
        *schema, next_receipt(accepted.receipt), .1, .1, events, signed_change, {}, escaped, source_mass, outflow_mass);
    EXPECT_DOUBLE_EQ(D::Accounting::report(trial.net_source).at("G_A"), 1e-12);
    EXPECT_DOUBLE_EQ(source_mass, 1e-12 * .02);
    EXPECT_DOUBLE_EQ(D::Accounting::report(accepted.net_source).at("G_A"), 0.);

    S::HeadspaceOptions headspace;
    headspace.total_internal_volume = 1;
    headspace.initial_temperature = 300;
    auto vent = std::make_unique<S::VentedHeadspaceModel>(headspace);
    S::FreeSurfaceCouplingOptions gates;
    gates.gas_absolute_tolerance = 1e-15;
    gates.gas_relative_tolerance = 0;
    S::PlanarFreeSurfaceModel surface(
        std::make_shared<S::ConstantAreaVesselVolumeMap>(0., 1., 1.), std::move(vent), gates);
    S::FreeSurfaceUpdate update;
    update.liquid_volume_at_pressure = [](double) { return .5; };
    update.bubble_volume_at_pressure = [](double) { return 0.; };
    update.headspace_temperature = 300;
    update.gas.initial_moles = accepted.initial;
    update.gas.dissolved_moles = accepted.initial;
    surface.initialize(update);
    update.gas.generated_moles = D::Accounting::report(trial.production);
    update.gas.consumed_moles = D::Accounting::report(trial.consumption);
    update.gas.net_source_moles = D::Accounting::report(trial.net_source);
    update.gas.dissolved_moles["G_A"] = 1. + 1e-12;
    EXPECT_NO_THROW((void) surface.previewUpdate(update));
    update.gas.net_source_moles->at("G_A") = 0.;
    EXPECT_ANY_THROW((void) surface.previewUpdate(update));
}
TEST(SpeciesSurfaceAccountingTest, DissolvedOutflowDoesNotEnterHeadspaceAndGasEscapeRetainsBothSpecies)
{
    const auto schema = catalog();
    D::Accounting accepted;
    accepted.initialize(*schema, initial_receipt(*schema), {{"G_A", 2.}, {"G_B", 3.}});
    const std::vector<R::SpeciesSubsetSourceChange> source{{9, 0, 0}, {701, 0, 0}};
    const std::vector<R::SpeciesPhysicalEscapeEvent> events{{9, R::SpeciesPhase::Dissolved, {}, .125, .0025},
        {9, R::SpeciesPhase::Dispersed, "micro", .25, .005}, {701, R::SpeciesPhase::Dispersed, "large", .5, .02}};
    S::GasMolesBySpecies gas;
    double change = 0, outflow = 0;
    const auto trial =
        accepted.preview(*schema, next_receipt(accepted.receipt), .1, .1, {}, source, events, gas, change, outflow);
    EXPECT_DOUBLE_EQ(gas.at("G_A"), .25);
    EXPECT_DOUBLE_EQ(gas.at("G_B"), .5);
    EXPECT_DOUBLE_EQ(D::Accounting::report(trial.dissolved_outflow).at("G_A"), .125);
    EXPECT_DOUBLE_EQ(outflow, .0025);
    EXPECT_DOUBLE_EQ(change, 0.);
    EXPECT_FALSE(events[0].phase_energy_debited);
    EXPECT_FALSE(events[0].phase_momentum_debited);
}
TEST(SpeciesSurfaceAccountingTest, ReceiptRejectsDuplicateSkippedForeignAndWrongIntervalWithoutMutation)
{
    const auto schema = catalog();
    D::Accounting accepted;
    accepted.initialize(*schema, initial_receipt(*schema), {{"G_A", 1.}, {"G_B", 2.}});
    const std::vector<R::SpeciesSubsetSourceChange> source{{9, .5, 0}, {701, 0, 0}};
    S::GasMolesBySpecies escaped;
    double mass = 0, outflow = 0;
    const auto preview = [&](auto receipt)
    { return accepted.preview(*schema, receipt, .1, .1, {}, source, {}, escaped, mass, outflow); };
    EXPECT_ANY_THROW((void) preview(accepted.receipt));
    auto receipt = next_receipt(accepted.receipt);
    ++receipt.revision;
    EXPECT_ANY_THROW((void) preview(receipt));
    receipt = next_receipt(accepted.receipt);
    receipt.schema_fingerprint = catalog(true)->fingerprint();
    EXPECT_ANY_THROW((void) preview(receipt));
    receipt = next_receipt(accepted.receipt, .2, .2);
    EXPECT_ANY_THROW((void) preview(receipt));
    EXPECT_EQ(accepted.receipt, initial_receipt(*schema));
    EXPECT_DOUBLE_EQ(D::Accounting::report(accepted.net_source).at("G_A"), 0.);
    const auto saved = accepted.snapshot();
    accepted = preview(next_receipt(accepted.receipt));
    EXPECT_ANY_THROW(
        (void) accepted.preview(*schema, accepted.receipt, .1, .1, {}, source, {}, escaped, mass, outflow));
    accepted.restore(saved);
    const auto replay = preview(next_receipt(accepted.receipt));
    EXPECT_DOUBLE_EQ(D::Accounting::report(replay.net_source).at("G_A"), .5);
}
TEST(SpeciesSurfaceAccountingTest, RepeatedSubUlpVentedEventsSurviveSurfaceReplay)
{
    S::HeadspaceOptions options;
    options.total_internal_volume = 1;
    options.initial_temperature = 300;
    S::PlanarFreeSurfaceModel surface(std::make_shared<S::ConstantAreaVesselVolumeMap>(0., 1., 1.),
        std::make_unique<S::VentedHeadspaceModel>(options));
    S::FreeSurfaceUpdate update;
    update.liquid_volume_at_pressure = [](double) { return .5; };
    update.bubble_volume_at_pressure = [](double) { return 0.; };
    update.headspace_temperature = 300;
    update.gas.initial_moles = {{"G_A", .5}};
    update.gas.submerged_moles = {{"G_A", .5}};
    surface.initialize(update);
    const double event = std::numeric_limits<double>::epsilon() / 32.;
    // A pre-existing large vent amount makes later events smaller than its ULP.
    update.gas.submerged_moles["G_A"] = 0.;
    update.gas.escaped_moles_this_step["G_A"] = .5;
    surface.update(update);
    const auto saved = surface.snapshot();
    update.gas.generated_moles["G_A"] = 256 * event;
    update.gas.net_source_moles = S::GasMolesBySpecies{{"G_A", 256 * event}};
    update.gas.escaped_moles_this_step["G_A"] = event;
    for (int i = 0; i < 256; ++i)
    {
        update.gas.generated_moles["G_A"] = (i + 1) * event;
        update.gas.net_source_moles->at("G_A") = (i + 1) * event;
        surface.update(update);
    }
    const auto expected = static_cast<double>(.5L + 256.L * event);
    EXPECT_DOUBLE_EQ(surface.diagnostics().vented_gas_moles.at("G_A"), expected);
    EXPECT_DOUBLE_EQ(surface.committedEscapedMoles().at("G_A"), expected);
    surface.restore(saved);
    EXPECT_DOUBLE_EQ(surface.diagnostics().vented_gas_moles.at("G_A"), .5);
    EXPECT_DOUBLE_EQ(surface.committedEscapedMoles().at("G_A"), .5);
}
} // namespace
