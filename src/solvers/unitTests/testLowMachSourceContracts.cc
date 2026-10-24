/** @file testLowMachSourceContracts.cc
 * @brief Conservative posttransport donors, collective source stages and immutable receipt regressions.
 */
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/CompensatedSum.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
constexpr double gas_constant = 8.31446261815324;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> schema;
    std::shared_ptr<const R::MixtureThermodynamics> gas;
    std::shared_ptr<Flow> flow;
    explicit Fixture(unsigned count, double common_reference = 0, bool two_phases = false)
    {
        ArrReal z;
        for (unsigned k = 0; k <= count; ++k)
            z.push_back(double(k) / count);
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, z}),
            distribution);
        const std::vector roles{R::SpeciesPhase::Resolved};
        schema = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {103, "finite_A", .02, {{"X", 1}}, roles}, {709, "finite_B", .02, {{"X", 1}}, roles}});
        gas = R::make_ideal_gas_mixture(schema,
            {{20, 300, 6000 + .02 * common_reference}, {20, 300, 6000 + .02 * common_reference}},
            {250, 600, 10000, 2000000});
        Flow::Options options;
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        flow = std::make_shared<Flow>(mesh, two_phases ? std::vector{gas, gas} : std::vector{gas}, options);
        Flow::State initial;
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const double amount = 100000 * mesh->cell_volume(static_cast<LO>(c)) / (gas_constant * 300);
            initial.cells[c].phases = {
                {gas, {.8 * amount, .2 * amount}, 100000, (20 * 300 + .02 * common_reference) * amount, 300}};
            if (two_phases)
            {
                auto half = initial.cells[c].phases[0];
                for (auto& n : half.moles)
                    n *= .5;
                half.internal_energy *= .5;
                initial.cells[c].phases = {half, half};
            }
        }
        flow->initialize(initial);
    }
};
struct Receipt
{
    Flow::TransportSnapshot transport;
    std::vector<Flow::CellSources> packets;
};
void expect_exact_scalar(double actual, double expected)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(expected));
}
void expect_exact_receipt(const Flow::Trial& trial, const Receipt& receipt)
{
    EXPECT_EQ(trial.source_stage(), receipt.transport.stage);
    EXPECT_EQ(trial.transport_snapshot(), receipt.transport);
    EXPECT_EQ(trial.transport_snapshot().accepted_revision, receipt.transport.accepted_revision);
    EXPECT_EQ(trial.transport_snapshot().accepted_generation, receipt.transport.accepted_generation);
    EXPECT_EQ(trial.transport_snapshot().old_geometry_epoch, receipt.transport.old_geometry_epoch);
    EXPECT_EQ(trial.transport_snapshot().new_geometry_epoch, receipt.transport.new_geometry_epoch);
    EXPECT_EQ(trial.transport_snapshot().old_control_volumes, receipt.transport.old_control_volumes);
    EXPECT_EQ(trial.transport_snapshot().new_control_volumes, receipt.transport.new_control_volumes);
    ASSERT_EQ(trial.integrated_sources().size(), receipt.packets.size());
    ASSERT_EQ(trial.transport_snapshot().cells.size(), receipt.transport.cells.size());
    for (size_t c = 0; c < receipt.packets.size(); ++c)
    {
        EXPECT_EQ(trial.integrated_sources()[c].species_moles, receipt.packets[c].species_moles);
        EXPECT_EQ(trial.integrated_sources()[c].internal_energy, receipt.packets[c].internal_energy);
        for (size_t a = 0; a < receipt.packets[c].internal_energy.size(); ++a)
            expect_exact_scalar(
                trial.integrated_sources()[c].internal_energy[a], receipt.packets[c].internal_energy[a]);
        for (size_t a = 0; a < receipt.packets[c].species_moles.size(); ++a)
            for (size_t species = 0; species < receipt.packets[c].species_moles[a].size(); ++species)
                expect_exact_scalar(trial.integrated_sources()[c].species_moles[a][species],
                    receipt.packets[c].species_moles[a][species]);
        for (size_t axis = 0; axis < 3; ++axis)
            expect_exact_scalar(
                trial.integrated_sources()[c].momentum.component(axis), receipt.packets[c].momentum.component(axis));
        const auto& actual = trial.transport_snapshot().cells[c].phases;
        const auto& expected = receipt.transport.cells[c].phases;
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t a = 0; a < actual.size(); ++a)
        {
            EXPECT_EQ(actual[a].moles, expected[a].moles);
            for (size_t species = 0; species < expected[a].moles.size(); ++species)
                expect_exact_scalar(actual[a].moles[species], expected[a].moles[species]);
            expect_exact_scalar(actual[a].internal_energy, expected[a].internal_energy);
            expect_exact_scalar(actual[a].advected_material_volume, expected[a].advected_material_volume);
            EXPECT_EQ(actual[a].population_number, expected[a].population_number);
        }
    }
}
TEST(LowMachSourceContractsTest, FullFiniteConversionReservesActualAdvectedStockAndRetainsExactReceipt)
{
    Fixture fixture(8);
    Receipt receipt;
    int evaluations = 0;
    const auto accepted = fixture.flow->state();
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(
        Flow::SourceStageProvider{"manufactured-posttransport-conversion", 7, fixture.schema->fingerprint(),
            [&](const Flow::SourceStageContext& context)
            {
                ++evaluations; // Temporary instrumentation, never a physical-history input.
                receipt.transport = context.transport;
                receipt.packets.assign(context.transport.cells.size(), {});
                for (size_t c = 0; c < receipt.packets.size(); ++c)
                {
                    const auto& donor = context.transport.cells[c].phases[0];
                    const double finite_A = donor.moles[fixture.schema->index(103)];
                    receipt.packets[c].species_moles = {{-finite_A, finite_A}};
                    receipt.packets[c].internal_energy = {
                        fixture.mesh->cell_centroid(static_cast<LO>(c)).z < .125 ? 100.0 : 0.0};
                }
                return Flow::SourceStageResult{receipt.packets, {}};
            }});
    const auto trial = fixture.flow->preview_with_sources(.01, provider);
    EXPECT_GE(evaluations, 2);
    EXPECT_EQ(trial.diagnostics().source_evaluations, evaluations);
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
    EXPECT_DOUBLE_EQ(fixture.flow->state().time, accepted.time);
    expect_exact_receipt(trial, receipt);
    double local_change = 0, change = 0;
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const auto donor = receipt.transport.cells[c].phases[0].moles[0];
        local_change = std::max(local_change, std::abs(donor - accepted.cells[c].phases[0].moles[0]));
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[0].moles[0], 0);
        EXPECT_DOUBLE_EQ(trial.integrated_sources()[c].species_moles[0][0], -donor);
    }
    Teuchos::reduceAll(*fixture.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_change, &change);
    EXPECT_GT(change, 1e-8); // Actual flow changes donors; accepted inventory is not interchangeable.
    const int count_before_commit = evaluations;
    fixture.flow->commit(trial);
    EXPECT_EQ(evaluations, count_before_commit);
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision + 1);
}
TEST(LowMachSourceContractsTest, FailedLaterProviderDoesNotInvalidateRetainedTrialOrReevaluateOnRestore)
{
    Fixture fixture(2);
    Receipt receipt;
    int evaluations = 0;
    const auto saved = fixture.flow->checkpoint();
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "independent-heat", 1, fixture.schema->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++evaluations;
            receipt.transport = context.transport;
            receipt.packets.assign(context.transport.cells.size(), {});
            for (auto& packet : receipt.packets)
                packet.internal_energy = {10};
            return Flow::SourceStageResult{receipt.packets, {}};
        }});
    const auto retained = fixture.flow->preview_with_sources(.01, provider);
    expect_exact_receipt(retained, receipt);
    const auto bad = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{"rank-local-rejection",
        1, fixture.schema->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            if (fixture.mesh->owned_cell_map()->getComm()->getRank() == 0)
                throw std::runtime_error("Manufactured source failure.");
            return Flow::SourceStageResult{std::vector<Flow::CellSources>(context.transport.cells.size()), {}};
        }});
    EXPECT_THROW((void) fixture.flow->preview_with_sources(.01, bad), std::exception);
    const int before = evaluations;
    EXPECT_NO_THROW(fixture.flow->commit(retained));
    EXPECT_EQ(evaluations, before);
    EXPECT_NO_THROW(fixture.flow->restore(saved));
    EXPECT_EQ(evaluations, before);
}

TEST(LowMachSourceContractsTest, PhasePropertyFailureIsFencedBeforeHaloExchangeIncludingEmptyOwners)
{
    Fixture fixture(2);
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    const auto accepted = fixture.flow->state();
    const auto retained = fixture.flow->preview(.01);
    // The uniform field/import-plan layout is prepared before local user laws.
    Flow::Traits::scalar_cell_type phase_trace(fixture.mesh, 0.0, "private_phase_trace");
    int halo_entries = 0;
    const auto user_phase_property = [&](const R::PhaseThermalState& phase, size_t cell)
    {
        if (comm->getRank() == 0 && cell == 0)
            throw std::runtime_error("Manufactured MPI-free phase-property failure.");
        return phase.temperature;
    };
    const auto internal_provider = std::make_shared<const Flow::SourceStageProvider>(
        Flow::SourceStageProvider{"controller-internal-diffusion-stage", 1, fixture.schema->fingerprint(),
            [&](const Flow::SourceStageContext& context)
            {
                collective_detail::collective_local_validation(context.mesh, "Local phase properties before halo",
                    [&]
                    {
                        for (size_t c = 0; c < context.transport.cells.size(); ++c)
                            phase_trace.set_owned_value(static_cast<LO>(c),
                                user_phase_property(context.thermodynamic_trial.cells[c].phases[0], c));
                    });
                ++halo_entries;
                phase_trace.sync_ghosts(); // Must not be reached by ANY rank in this case.
                return Flow::SourceStageResult{std::vector<Flow::CellSources>(context.transport.cells.size()), {}};
            }});
    int rejected = 0;
    try
    {
        (void) fixture.flow->preview_with_sources(.01, internal_provider);
    }
    catch (const std::exception&)
    {
        rejected = 1;
    }
    int all_rejected = 0, total_halo_entries = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &all_rejected);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &halo_entries, &total_halo_entries);
    EXPECT_EQ(all_rejected, comm->getSize());
    EXPECT_EQ(total_halo_entries, 0);
    int empty = fixture.mesh->num_owned_cells() == 0 ? 1 : 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    if (comm->getSize() == 4)
        EXPECT_EQ(total_empty, 2);
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
    EXPECT_DOUBLE_EQ(fixture.flow->state().time, accepted.time);
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}

TEST(LowMachSourceContractsTest, GrossTurnoverAndTinyEventsConvergeIndependentlyOfZeroNetPackets)
{
    Fixture fixture(2);
    int evaluations = 0;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "cancelled-gross-turnover", 1, fixture.schema->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++evaluations;
            Flow::SourceStageResult result;
            const double factor = context.transport.stage.corrector == 0 ? 10.0 : 1.0;
            result.gross_events.assign(context.transport.cells.size(), {factor * 1e100, factor * 1e-250});
            return result;
        }});
    const auto accepted = fixture.flow->state();
    const auto trial = fixture.flow->preview_with_sources(.01, provider);
    EXPECT_EQ(evaluations, 3);
    EXPECT_EQ(trial.diagnostics().source_evaluations, 3);
    EXPECT_DOUBLE_EQ(trial.diagnostics().maximum_event_residual, 0);
    EXPECT_EQ(trial.integrated_gross_events().size(), accepted.cells.size());
    for (const auto& events : trial.integrated_gross_events())
    {
        ASSERT_EQ(events.size(), 2u);
        EXPECT_DOUBLE_EQ(events[0], 1e100);
        EXPECT_DOUBLE_EQ(events[1], 1e-250);
    }
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
}

TEST(LowMachSourceContractsTest, InvalidOrChangingGrossEventSchemasRejectAndKeepRetainedTrial)
{
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        Fixture fixture(2);
        const auto retained = fixture.flow->preview(.01);
        const auto accepted = fixture.flow->state();
        const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
            "invalid-event-schema", 1, fixture.schema->fingerprint(), [&](const Flow::SourceStageContext& context)
            {
                Flow::SourceStageResult result;
                result.gross_events.assign(context.transport.cells.size(), {1});
                for (auto& events : result.gross_events)
                {
                    if (scenario == 0 && context.transport.stage.corrector > 0)
                        events.push_back(0);
                    if (scenario == 1)
                        events[0] = -1;
                    if (scenario == 2)
                        events[0] = std::numeric_limits<double>::quiet_NaN();
                    if (scenario == 3)
                        events[0] = std::numeric_limits<double>::infinity();
                }
                if (scenario == 4 && context.transport.stage.corrector > 0)
                    result.gross_events.clear();
                return result;
            }});
        EXPECT_THROW((void) fixture.flow->preview_with_sources(.01, provider), std::exception);
        EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
        EXPECT_DOUBLE_EQ(fixture.flow->state().time, accepted.time);
        EXPECT_NO_THROW(fixture.flow->commit(retained));
    }
}

TEST(LowMachSourceContractsTest, EventLayoutFailureOnEmptyOwnerRejectsCollectively)
{
    Fixture fixture(2);
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    const auto retained = fixture.flow->preview(.01);
    const auto accepted = fixture.flow->state();
    int empty = fixture.mesh->num_owned_cells() == 0 ? 1 : 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    if (comm->getSize() == 4)
        EXPECT_EQ(total_empty, 2);
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "rank-local-event-layout", 1, fixture.schema->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            Flow::SourceStageResult result;
            const bool invalid = total_empty > 0 ? empty != 0 : comm->getRank() == comm->getSize() - 1;
            result.gross_events.assign(context.transport.cells.size() + (invalid ? 1 : 0), {1});
            return result;
        }});
    int rejected = 0;
    try
    {
        (void) fixture.flow->preview_with_sources(.01, provider);
    }
    catch (const std::exception&)
    {
        rejected = 1;
    }
    int total_rejected = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &total_rejected);
    EXPECT_EQ(total_rejected, comm->getSize());
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
    EXPECT_DOUBLE_EQ(fixture.flow->state().time, accepted.time);
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}

TEST(LowMachSourceContractsTest, DivergentProviderMetadataRejectsBeforeCallbackOnEveryRank)
{
    Fixture fixture(2);
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    const auto retained = fixture.flow->preview(.01);
    int evaluations = 0;
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        const bool divergent = comm->getRank() == comm->getSize() - 1;
        std::string identity = "uniform-provider-A";
        std::uint64_t revision = 1, fingerprint = fixture.schema->fingerprint();
        double tolerance = 0;
        int limit = 0;
        if (comm->getSize() == 1)
            revision = 0;
        else
        {
            if (scenario == 0 && divergent)
                identity = "uniform-provider-B";
            if (scenario == 1 && divergent)
                revision = 2;
            if (scenario == 2 && divergent)
                tolerance = 1e-7;
            if (scenario == 3 && divergent)
                limit = 2;
            if (scenario == 4 && divergent)
                fingerprint ^= 1;
        }
        const auto provider =
            std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{identity, revision, fingerprint,
                [&](const Flow::SourceStageContext&)
                {
                    ++evaluations;
                    return Flow::SourceStageResult{};
                },
                tolerance, limit});
        EXPECT_THROW((void) fixture.flow->preview_with_sources(.01, provider), std::exception);
        EXPECT_EQ(evaluations, 0);
    }
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}

TEST(LowMachSourceContractsTest, ProviderRelativeToleranceAndEvaluationLimitAreHonored)
{
    Fixture fixture(2);
    int evaluations = 0;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(
        Flow::SourceStageProvider{"explicit-event-gate", 1, fixture.schema->fingerprint(),
            [&](const Flow::SourceStageContext& context)
            {
                ++evaluations;
                Flow::SourceStageResult result;
                result.gross_events.assign(
                    context.transport.cells.size(), {context.transport.stage.corrector == 0 ? 1.0 : .9});
                return result;
            },
            .2, 2});
    const auto trial = fixture.flow->preview_with_sources(.01, provider);
    EXPECT_EQ(evaluations, 2);
    EXPECT_NEAR(trial.diagnostics().maximum_event_residual, .5, 1e-14);
    const auto retained = trial;
    evaluations = 0;
    const auto unsteady = std::make_shared<const Flow::SourceStageProvider>(
        Flow::SourceStageProvider{"limited-unsteady-events", 1, fixture.schema->fingerprint(),
            [&](const Flow::SourceStageContext& context)
            {
                ++evaluations;
                Flow::SourceStageResult result;
                result.gross_events.assign(
                    context.transport.cells.size(), {context.transport.stage.corrector % 2 ? 2.0 : 1.0});
                return result;
            },
            0, 2});
    EXPECT_THROW((void) fixture.flow->preview_with_sources(.01, unsteady), std::exception);
    EXPECT_EQ(evaluations, 2);
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}

TEST(LowMachSourceContractsTest, MultipleProviderPreviewsRetainIndependentReceiptsAndSharedLifetime)
{
    Fixture fixture(2);
    int evaluations = 0;
    auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "shared-independent-receipts", 3, fixture.schema->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++evaluations;
            Flow::SourceStageResult result;
            result.gross_events.assign(context.transport.cells.size(), {1});
            return result;
        }});
    const std::weak_ptr<const Flow::SourceStageProvider> lifetime = provider;
    const auto first = fixture.flow->preview_with_sources(.01, provider);
    const auto first_snapshot = first.transport_snapshot();
    const std::vector<Flow::PhaseFaceTransport> first_faces(
        first.owned_phase_transport().begin(), first.owned_phase_transport().end());
    const auto second = fixture.flow->preview_with_sources(.01, provider);
    EXPECT_NE(first.source_stage().invocation, second.source_stage().invocation);
    EXPECT_EQ(first.transport_snapshot(), first_snapshot);
    EXPECT_EQ(std::vector<Flow::PhaseFaceTransport>(
                  first.owned_phase_transport().begin(), first.owned_phase_transport().end()),
        first_faces);
    EXPECT_EQ(first.state().revision, second.state().revision);
    provider.reset();
    EXPECT_FALSE(lifetime.expired());
    const int before = evaluations;
    EXPECT_NO_THROW(fixture.flow->commit(first));
    EXPECT_EQ(evaluations, before);
    EXPECT_THROW(fixture.flow->commit(second), std::exception);
}

TEST(LowMachSourceContractsTest, InvalidProviderControlsRejectBeforeEvaluatingUserCode)
{
    Fixture fixture(2);
    const auto retained = fixture.flow->preview(.01);
    int evaluations = 0;
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        const double tolerance = scenario == 0   ? -1
                                 : scenario == 1 ? std::numeric_limits<double>::quiet_NaN()
                                 : scenario == 2 ? std::numeric_limits<double>::infinity()
                                                 : 0;
        const int limit = scenario == 3 ? 1 : scenario == 4 ? -1 : 0;
        const auto provider = std::make_shared<const Flow::SourceStageProvider>(
            Flow::SourceStageProvider{"invalid-source-controls", 1, fixture.schema->fingerprint(),
                [&](const Flow::SourceStageContext&)
                {
                    ++evaluations;
                    return Flow::SourceStageResult{};
                },
                tolerance, limit});
        EXPECT_THROW((void) fixture.flow->preview_with_sources(.01, provider), std::exception);
        EXPECT_EQ(evaluations, 0);
    }
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}

TEST(LowMachSourceContractsTest, ReferenceShiftPreservesSourceConvergenceWithPairedMassEnergyTransfer)
{
    constexpr double reference_shift = 1e6;
    Fixture base(8, 0, true), shifted(8, reference_shift, true);
    const auto provider = [](Fixture& fixture, double reference)
    {
        return std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{"paired-reference-transfer",
            1, fixture.schema->fingerprint(), [&fixture, reference](const Flow::SourceStageContext& context)
            {
                Flow::SourceStageResult result;
                result.sources.resize(context.transport.cells.size());
                result.gross_events.resize(context.transport.cells.size());
                for (size_t c = 0; c < result.sources.size(); ++c)
                {
                    const double dn = .1 * context.transport.cells[c].phases[0].moles[0];
                    const double temperature = context.thermodynamic_trial.cells[c].phases[0].temperature;
                    const double molar_enthalpy = (20 + gas_constant) * temperature + .02 * reference;
                    const double heat = fixture.mesh->cell_centroid(static_cast<LO>(c)).z < .125 ? 100 : 0;
                    result.sources[c].species_moles = {{-dn, 0}, {dn, 0}};
                    result.sources[c].internal_energy = {-dn * molar_enthalpy, dn * molar_enthalpy + heat};
                    result.gross_events[c] = {dn};
                }
                return result;
            }});
    };
    const auto original = base.flow->preview_with_sources(.01, provider(base, 0));
    const auto offset = shifted.flow->preview_with_sources(.01, provider(shifted, reference_shift));
    EXPECT_NEAR(original.state().thermodynamic_pressure, offset.state().thermodynamic_pressure, 1e-3);
    EXPECT_LE(original.diagnostics().maximum_source_residual, 1);
    EXPECT_LE(offset.diagnostics().maximum_source_residual, 1);
    ASSERT_EQ(original.state().cells.size(), offset.state().cells.size());
    for (size_t c = 0; c < original.state().cells.size(); ++c)
    {
        for (size_t a = 0; a < 2; ++a)
        {
            EXPECT_NEAR(
                original.state().cells[c].phases[a].temperature, offset.state().cells[c].phases[a].temperature, 1e-6);
            for (size_t i = 0; i < 2; ++i)
                EXPECT_NEAR(
                    original.state().cells[c].phases[a].moles[i], offset.state().cells[c].phases[a].moles[i], 1e-8);
        }
        EXPECT_NEAR(original.state().cells[c].mechanical_pressure, offset.state().cells[c].mechanical_pressure, 1e-4);
        for (size_t a = 0; a < 3; ++a)
            EXPECT_NEAR(
                original.state().cells[c].velocity.component(a), offset.state().cells[c].velocity.component(a), 1e-8);
    }
    ASSERT_EQ(original.owned_volume_flux().size(), offset.owned_volume_flux().size());
    for (size_t f = 0; f < original.owned_volume_flux().size(); ++f)
        EXPECT_NEAR(original.owned_volume_flux()[f], offset.owned_volume_flux()[f], 1e-8);
}

TEST(LowMachSourceContractsTest, EOSTangentProjectionMatchesIndependentTwoCellHelmholtzAndGaugeOracle)
{
    Fixture fixture(2);
    using Pressure = R::LowMachPressureEquation;
    Pressure equation(fixture.mesh);
    Pressure::ScalarField rho(fixture.mesh, 800, "helmholtz_density"), pressure(fixture.mesh, 7, "helmholtz_pressure");
    Pressure::VectorField velocity(fixture.mesh, "helmholtz_velocity");
    Pressure::FaceField predictor(fixture.mesh, 0.0, "helmholtz_predictor"),
        corrected(fixture.mesh, 0.0, "helmholtz_flux");
    velocity.owned_data().putScalar(0);
    rho.sync_ghosts();
    pressure.sync_ghosts();
    velocity.sync_ghosts();
    predictor.sync_ghosts();
    constexpr double dt = 1e-4, chi_lower = 2e-6, chi_upper = 3e-6, forcing = .02;
    const double mobility = dt / 800 / .5;
    const double delta_lower = forcing / (mobility * (1 + chi_lower / chi_upper) + chi_lower / dt);
    const double delta_upper = -chi_lower / chi_upper * delta_lower;
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    const auto gauge = fixture.mesh->owned_cell_map()->getMinAllGlobalIndex();
    std::vector<double> target(fixture.mesh->num_owned_cells()), chi(target.size()), expected_delta(target.size());
    double local_gauge = 0, expected_gauge = 0;
    for (size_t c = 0; c < target.size(); ++c)
    {
        const bool lower = fixture.mesh->cell_centroid(static_cast<LO>(c)).z < .5;
        target[c] = lower ? forcing : -forcing;
        chi[c] = lower ? chi_lower : chi_upper;
        expected_delta[c] = lower ? delta_lower : delta_upper;
        if (fixture.mesh->cell_global_id(static_cast<LO>(c)) == gauge)
            local_gauge = 7 + expected_delta[c];
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_gauge, &expected_gauge);
    LinearSolverOptions linear;
    linear.preconditioner = LinearPreconditioner::None;
    linear.tolerance = 1e-12;
    const auto result =
        equation.project_with_eos(rho, dt, target, chi, predictor, corrected, pressure, velocity, linear);
    EXPECT_NEAR(result.thermodynamic_pressure_increment, expected_gauge, 1e-12);
    EXPECT_LE(result.maximum_linearized_continuity_residual, 1e-12);
    EXPECT_GT(result.maximum_continuity_residual, .01); // Intermediate Newton step is not a physical acceptance.
    double local_weighted_delta = 0, weighted_delta = 0;
    for (size_t c = 0; c < target.size(); ++c)
    {
        const LO cell = static_cast<LO>(c);
        EXPECT_NEAR(pressure.value(cell), 7 + expected_delta[c] - expected_gauge, 1e-12);
        EXPECT_NEAR(100000 + result.thermodynamic_pressure_increment + pressure.value(cell),
            100000 + 7 + expected_delta[c], 1e-10);
        const double actual_delta = pressure.value(cell) + result.thermodynamic_pressure_increment - 7;
        local_weighted_delta += chi[c] * actual_delta;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_weighted_delta, &weighted_delta);
    EXPECT_NEAR(weighted_delta, 0, 1e-18);
    for (const auto face : corrected.owned_face_ids())
        if (!fixture.mesh->is_geometry_exterior_face(face))
        {
            const bool owner_lower = fixture.mesh->cell_centroid(fixture.mesh->owner_cell(face)).z < .5;
            EXPECT_NEAR(corrected.local_value(face),
                mobility * (owner_lower ? delta_lower - delta_upper : delta_upper - delta_lower), 1e-14);
        }
}

TEST(LowMachSourceContractsTest, ZeroEOSSusceptibilityRecoversTheOriginalConservativeProjection)
{
    Fixture fixture(2);
    using Pressure = R::LowMachPressureEquation;
    Pressure equation(fixture.mesh);
    Pressure::ScalarField rho(fixture.mesh, 800, "limit_density"), old_p(fixture.mesh, 0.0, "limit_old_p"),
        new_p(fixture.mesh, 0.0, "limit_new_p");
    Pressure::VectorField old_u(fixture.mesh, "limit_old_u"), new_u(fixture.mesh, "limit_new_u");
    Pressure::FaceField predictor(fixture.mesh, 0.0, "limit_predictor"), old_flux(fixture.mesh, 0.0, "limit_old_flux"),
        new_flux(fixture.mesh, 0.0, "limit_new_flux");
    old_u.owned_data().putScalar(0);
    new_u.owned_data().putScalar(0);
    rho.sync_ghosts();
    old_p.sync_ghosts();
    new_p.sync_ghosts();
    old_u.sync_ghosts();
    new_u.sync_ghosts();
    for (const auto face : predictor.owned_face_ids())
        if (!fixture.mesh->is_geometry_exterior_face(face))
            predictor.set_owned_value(face, .02);
    predictor.sync_ghosts();
    const std::vector<double> zero(fixture.mesh->num_owned_cells(), 0);
    LinearSolverOptions linear;
    linear.preconditioner = LinearPreconditioner::None;
    linear.tolerance = 1e-12;
    const auto original = equation.project(rho, 1e-4, zero, predictor, old_flux, old_p, old_u, linear);
    const auto limiting = equation.project_with_eos(rho, 1e-4, zero, zero, predictor, new_flux, new_p, new_u, linear);
    EXPECT_DOUBLE_EQ(limiting.thermodynamic_pressure_increment, 0);
    EXPECT_LE(original.maximum_continuity_residual, 1e-10);
    EXPECT_LE(limiting.maximum_continuity_residual, 1e-10);
    for (size_t c = 0; c < zero.size(); ++c)
    {
        const LO cell = static_cast<LO>(c);
        EXPECT_DOUBLE_EQ(old_p.value(cell), new_p.value(cell));
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_DOUBLE_EQ(old_u.value(cell).component(axis), new_u.value(cell).component(axis));
    }
    for (const auto face : old_flux.owned_face_ids())
        EXPECT_DOUBLE_EQ(old_flux.local_value(face), new_flux.local_value(face));
}

TEST(LowMachSourceContractsTest, InvalidEOSSusceptibilityRejectsCollectivelyBeforePublishing)
{
    Fixture fixture(2);
    using Pressure = R::LowMachPressureEquation;
    Pressure equation(fixture.mesh);
    Pressure::ScalarField rho(fixture.mesh, 800, "invalid_chi_density"),
        pressure(fixture.mesh, 3, "invalid_chi_pressure");
    Pressure::VectorField velocity(fixture.mesh, "invalid_chi_velocity");
    Pressure::FaceField predictor(fixture.mesh, 0.0, "invalid_chi_predictor"),
        corrected(fixture.mesh, 7, "invalid_chi_output");
    velocity.owned_data().putScalar(0);
    rho.sync_ghosts();
    pressure.sync_ghosts();
    velocity.sync_ghosts();
    predictor.sync_ghosts();
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    const std::vector<double> target(fixture.mesh->num_owned_cells(), 0);
    for (const double invalid :
        {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        auto chi = target;
        if (comm->getRank() == 0)
            chi.at(0) = invalid;
        EXPECT_THROW((void) equation.project_with_eos(rho, 1e-4, target, chi, predictor, corrected, pressure, velocity),
            std::exception);
        for (size_t c = 0; c < target.size(); ++c)
            EXPECT_DOUBLE_EQ(pressure.value(static_cast<LO>(c)), 3);
        for (const auto face : corrected.owned_face_ids())
            EXPECT_DOUBLE_EQ(corrected.local_value(face), 7);
    }
}

struct DenseMixtureFixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<Flow> flow;
    std::uint64_t schema_fingerprint = 0;
    explicit DenseMixtureFixture(bool arriving = false, double reference = 0)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
            distribution);
        const std::vector roles{R::SpeciesPhase::Liquid, R::SpeciesPhase::Resolved};
        const auto schema = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {103, "liquid_carrier", .02, {{"X", 1}}, roles}, {709, "gas_carrier", .02, {{"X", 1}}, roles}});
        schema_fingerprint = schema->fingerprint();
        const auto liquid = R::make_linear_compressible_mixture_liquid(schema, {250, 600, 10000, 2000000},
            [reference](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6 + reference, 0, 0}; });
        const auto gas = R::make_ideal_gas_mixture(
            schema, {{20, 300, 6000 + .02 * reference}, {20, 300, 6000 + .02 * reference}}, {250, 600, 10000, 2000000});
        Flow::Options options;
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.thermal_mode =
            arriving ? R::PhaseThermalMode::SeparatePhaseTemperatures : R::PhaseThermalMode::Equilibrium;
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas}, options);
        Flow::State state;
        state.thermodynamic_pressure = 100000;
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const bool empty_gas = arriving && mesh->cell_centroid(static_cast<LO>(c)).z > .5;
            const double volume = mesh->cell_volume(static_cast<LO>(c));
            const double liquid_mass = (empty_gas ? 1 : .8) * volume * 1000;
            const double temperature = arriving ? 350 : 300;
            const double gas_amount = empty_gas ? 0 : 100000 * .2 * volume / (gas_constant * 300);
            state.cells[c].phases = {{liquid, {liquid_mass / .02, 0}, 100000,
                                         liquid_mass * (1e6 + reference + 4000 * (temperature - 300)), temperature},
                {gas, {0, gas_amount}, 100000, gas_amount * (6000 + .02 * reference), gas_amount > 0 ? 300. : 0.}};
            state.cells[c].drift_velocity.assign(2, Flow::Vector{});
            if (arriving)
                state.cells[c].drift_velocity[1] = Flow::Vector{0, 0, .1};
        }
        flow->initialize(std::move(state));
    }
};

TEST(LowMachSourceContractsTest, DenseLiquidGasTinyStepHeatRemainsBoundedAndRetainsLaterThermalFailure)
{
    DenseMixtureFixture fixture;
    const auto accepted = fixture.flow->state();
    int calls = 0;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(
        Flow::SourceStageProvider{"dense-mixture-small-dt-thermal-domain", 1, fixture.schema_fingerprint,
            [&](const Flow::SourceStageContext& context)
            {
                ++calls;
                Flow::SourceStageResult result;
                result.sources.resize(context.transport.cells.size());
                result.gross_events.resize(context.transport.cells.size());
                for (size_t c = 0; c < result.sources.size(); ++c)
                {
                    if (context.thermodynamic_trial.cells[c].phases[0].temperature > 300.0001)
                        throw std::runtime_error("Manufactured later thermal failure.");
                    const double heat = 100000 * context.transport.time_step;
                    result.sources[c].internal_energy = {heat, 0};
                    result.gross_events[c] = {heat};
                }
                return result;
            },
            0, 20});
    const auto valid = fixture.flow->preview_with_sources(.0001, provider);
    EXPECT_GE(valid.diagnostics().source_evaluations, 2);
    EXPECT_LE(valid.diagnostics().source_evaluations, 20);
    EXPECT_LE(valid.diagnostics().maximum_continuity_residual, fixture.flow->options().continuity_tolerance);
    EXPECT_LE(valid.diagnostics().maximum_occupancy_error, fixture.flow->options().volume_relative_tolerance);
    for (const auto& cell : valid.state().cells)
    {
        const double n = std::accumulate(cell.phases[1].moles.begin(), cell.phases[1].moles.end(), 0.0);
        EXPECT_NEAR(cell.phases[0].temperature, 300 + 10 / (400 * 4000 + 20 * n), 1e-8);
        for (const auto& phase : cell.phases)
            EXPECT_LT(phase.absolute_pressure, 2000000);
    }
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
    EXPECT_THROW((void) fixture.flow->preview_with_sources(.01, provider), std::exception);
    const int before = calls;
    EXPECT_NO_THROW(fixture.flow->commit(valid));
    EXPECT_EQ(calls, before);
}

TEST(LowMachSourceContractsTest, DenseMixtureTinyStepHeatPreservesCommonEnergyReferenceInvariance)
{
    DenseMixtureFixture base, shifted(false, 1e6);
    const auto provider = [](DenseMixtureFixture& fixture)
    {
        const auto fingerprint = fixture.schema_fingerprint;
        return std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
            "dense-mixture-small-step-reference", 1, fingerprint, [](const Flow::SourceStageContext& context)
            {
                Flow::SourceStageResult result;
                result.sources.resize(context.transport.cells.size());
                for (auto& source : result.sources)
                    source.internal_energy = {10, 0};
                return result;
            }});
    };
    const auto a = base.flow->preview_with_sources(.0001, provider(base));
    const auto b = shifted.flow->preview_with_sources(.0001, provider(shifted));
    EXPECT_NEAR(a.state().thermodynamic_pressure, b.state().thermodynamic_pressure, 1e-3);
    for (size_t c = 0; c < a.state().cells.size(); ++c)
    {
        EXPECT_NEAR(a.state().cells[c].phases[0].temperature, b.state().cells[c].phases[0].temperature, 1e-7);
        EXPECT_NEAR(a.state().cells[c].mechanical_pressure, b.state().cells[c].mechanical_pressure, 1e-4);
    }
    for (size_t f = 0; f < a.owned_volume_flux().size(); ++f)
        EXPECT_NEAR(a.owned_volume_flux()[f], b.owned_volume_flux()[f], 1e-8);
}

void audit_arriving_phase_transport(
    const DenseMixtureFixture& fixture, const Flow::State& accepted, const Flow::Trial& trial)
{
    using Face = Flow::Traits::scalar_face_type;
    const auto& mesh = *fixture.mesh;
    const auto comm = mesh.owned_cell_map()->getComm();
    const auto packets = trial.owned_phase_transport();
    const auto corrected = trial.owned_volume_flux();
    int local_valid = packets.size() == mesh.num_owned_faces() && corrected.size() == packets.size(), valid = 0;
    for (const auto& packet : packets)
        local_valid = local_valid && packet.phase_volume_flux.size() == 2 && packet.material_fraction.size() == 2;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_valid, &valid);
    ASSERT_EQ(valid, 1); // All ranks take this assertion before any following halo exchanges.
    std::vector<std::unique_ptr<Face>> stage_flux(2);
    for (size_t a = 0; a < 2; ++a)
    {
        stage_flux[a] = std::make_unique<Face>(fixture.mesh, 0.0, "receipt_stage_phase_flux");
        for (size_t f = 0; f < packets.size(); ++f)
            stage_flux[a]->set_owned_value(
                static_cast<LO>(f), packets[f].material_fraction[a] * packets[f].phase_volume_flux[a]);
        stage_flux[a]->sync_ghosts();
    }
    Face material(fixture.mesh, 0.0, "independently_audited_material_flux");
    double local_theta = 0, global_theta = 0;
    for (size_t f = 0; f < packets.size(); ++f)
    {
        double theta = 0, slip = 0;
        for (size_t a = 0; a < 2; ++a)
        {
            theta += packets[f].material_fraction[a];
            slip +=
                packets[f].material_fraction[a] * (packets[f].phase_volume_flux[a] - packets[f].carrier_volume_flux);
        }
        material.set_owned_value(static_cast<LO>(f), theta * corrected[f] + slip);
        if (!mesh.is_geometry_exterior_face(static_cast<LO>(f)))
            local_theta = std::max(local_theta, theta);
    }
    material.sync_ghosts();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_theta, &global_theta);
    EXPECT_GT(global_theta, 1.1); // Counterflow donors produce a real theta>1; no normalization is admissible.
    const auto balance = [&](const Face& flux, LO cell)
    {
        double total = 0;
        for (size_t f = 0; f < mesh.num_faces(); ++f)
        {
            const LO face = static_cast<LO>(f);
            if (mesh.owner_cell(face) == cell)
                total += flux.local_value(face);
            else if (mesh.neighbor_cell(face) == cell)
                total -= flux.local_value(face);
        }
        return total;
    };
    const auto mass = [](const R::PhaseThermalState& phase)
    {
        double value = 0;
        for (size_t i = 0; i < phase.moles.size(); ++i)
            value += phase.moles[i] * phase.thermodynamics->catalog().species()[i].molarMass;
        return value;
    };
    const auto volume = [&](const R::PhaseThermalState& phase)
    {
        const double amount = std::accumulate(phase.moles.begin(), phase.moles.end(), 0.0);
        if (amount == 0)
            return 0.0;
        auto x = phase.moles;
        for (auto& value : x)
            value /= amount;
        return mass(phase) / phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, x).density;
    };
    const auto& snapshot = trial.transport_snapshot();
    const double dt = snapshot.time_step;
    std::vector<double> old_amount(2), new_amount(2);
    double local_old_mass = 0, local_new_mass = 0;
    detail::CompensatedSum<long double> local_energy_change, local_work, local_pressure_operands;
    Flow::Vector local_old_momentum{}, local_new_momentum{}, local_momentum_scale{};
    collective_detail::collective_local_validation(mesh, "Independent phase receipt and EOS budget audit",
        [&]
        {
            for (size_t c = 0; c < accepted.cells.size(); ++c)
            {
                const LO cell = static_cast<LO>(c);
                double final_volume = 0, advected_volume = 0;
                for (size_t a = 0; a < 2; ++a)
                {
                    const auto& old = accepted.cells[c].phases[a];
                    const auto& now = trial.state().cells[c].phases[a];
                    const double initial_volume = volume(old), actual_volume = volume(now);
                    const double phase_balance = balance(*stage_flux[a], cell);
                    const double expected_advected = initial_volume - dt * phase_balance;
                    const double captured_advected = snapshot.cells[c].phases[a].advected_material_volume;
                    const double rounding =
                        64 * std::numeric_limits<double>::epsilon() *
                        (std::abs(initial_volume) + std::abs(dt * phase_balance) + std::abs(captured_advected));
                    EXPECT_NEAR(captured_advected, expected_advected, rounding);
                    final_volume += actual_volume;
                    advected_volume += captured_advected;
                    local_old_mass += mass(old);
                    local_new_mass += mass(now);
                    local_energy_change += static_cast<long double>(now.internal_energy) - old.internal_energy;
                    local_work +=
                        -static_cast<long double>(now.absolute_pressure) * (actual_volume - captured_advected);
                    local_pressure_operands +=
                        std::abs(static_cast<long double>(now.absolute_pressure) * actual_volume) +
                        std::abs(static_cast<long double>(now.absolute_pressure) * captured_advected);
                    for (size_t i = 0; i < 2; ++i)
                    {
                        old_amount[i] += old.moles[i];
                        new_amount[i] += now.moles[i];
                    }
                    const auto old_velocity = accepted.cells[c].velocity + accepted.cells[c].drift_velocity[a];
                    const auto new_velocity =
                        trial.state().cells[c].velocity + trial.state().cells[c].drift_velocity[a];
                    local_old_momentum = local_old_momentum + old_velocity * mass(old);
                    local_new_momentum = local_new_momentum + new_velocity * mass(now);
                    for (size_t axis = 0; axis < 3; ++axis)
                        local_momentum_scale.component(axis) += mass(old) * std::abs(old_velocity.component(axis)) +
                                                                mass(now) * std::abs(new_velocity.component(axis));
                }
                EXPECT_NEAR(final_volume, mesh.cell_volume(cell),
                    fixture.flow->options().volume_relative_tolerance * mesh.cell_volume(cell));
                EXPECT_NEAR(balance(material, cell), (final_volume - advected_volume) / dt,
                    fixture.flow->options().continuity_tolerance);
            }
        });
    double old_mass = 0, new_mass = 0, energy_change = 0, work = 0, pressure_operands = 0;
    const double energy_local = static_cast<double>(local_energy_change.value()),
                 work_local = static_cast<double>(local_work.value()),
                 operands_local = static_cast<double>(local_pressure_operands.value());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_old_mass, &old_mass);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_new_mass, &new_mass);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &energy_local, &energy_change);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &work_local, &work);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &operands_local, &pressure_operands);
    const double tolerance = fixture.flow->options().conservation_relative_tolerance;
    EXPECT_NEAR(new_mass, old_mass, tolerance * std::max(old_mass, new_mass));
    EXPECT_NEAR(
        work, trial.diagnostics().pressure_work, 64 * std::numeric_limits<double>::epsilon() * pressure_operands);
    EXPECT_LE(std::abs(energy_change - work),
        trial.diagnostics().internal_energy_roundoff_bound +
            64 * std::numeric_limits<double>::epsilon() * trial.diagnostics().caloric_energy_scale);
    for (size_t i = 0; i < 2; ++i)
    {
        double before = 0, after = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &old_amount[i], &before);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &new_amount[i], &after);
        EXPECT_NEAR(before, after, tolerance * std::max(before, after));
    }
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const double local_before = local_old_momentum.component(axis),
                     local_after = local_new_momentum.component(axis),
                     local_scale = local_momentum_scale.component(axis);
        double before = 0, after = 0, scale = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_before, &before);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_after, &after);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_scale, &scale);
        const double pressure = trial.diagnostics().pressure_impulse.component(axis),
                     external = trial.diagnostics().external_impulse.component(axis);
        EXPECT_NEAR(after - before, pressure + external,
            tolerance * std::max({std::abs(before), std::abs(after), std::abs(pressure), std::abs(external), scale}));
    }
}

TEST(LowMachSourceContractsTest, IndependentGasArrivesBySlipWithoutMechanicalPressureRunaway)
{
    DenseMixtureFixture fixture(true);
    const auto accepted = fixture.flow->state();
    std::vector<bool> observed(accepted.cells.size(), false);
    const auto fingerprint = fixture.schema_fingerprint;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "separate-phase-slip-arrival", 1, fingerprint, [&](const Flow::SourceStageContext& context)
        {
            for (size_t c = 0; c < context.transport.cells.size(); ++c)
            {
                const auto& packet = context.transport.cells[c].phases[1];
                const double amount = std::accumulate(packet.moles.begin(), packet.moles.end(), 0.0);
                if (accepted.cells[c].phases[1].temperature == 0 && amount > 0)
                    observed[c] = true;
            }
            return Flow::SourceStageResult{};
        }});
    const auto trial = fixture.flow->preview_with_sources(.001, provider);
    int local = std::count(observed.begin(), observed.end(), true), global = 0;
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    EXPECT_EQ(global, 1);
    EXPECT_LE(trial.diagnostics().maximum_continuity_residual, fixture.flow->options().continuity_tolerance);
    EXPECT_LE(trial.diagnostics().maximum_occupancy_error, fixture.flow->options().volume_relative_tolerance);
    EXPECT_NEAR(trial.diagnostics().mass_error, 0, 1e-10);
    audit_arriving_phase_transport(fixture, accepted, trial);
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
    EXPECT_NO_THROW(fixture.flow->commit(trial));
}

TEST(LowMachSourceContractsTest, FinalUniformPressureGaugeMustStayInsideItsDeclaredBracketAndRollBack)
{
    Fixture fixture(2);
    auto options = fixture.flow->options();
    options.maximum_pressure = 100005;
    Flow solver(fixture.mesh, {fixture.gas}, options);
    auto state = fixture.flow->state();
    for (auto& cell : state.cells)
    {
        cell.mechanical_pressure = 7;
        for (auto& n : cell.phases[0].moles)
            n *= 100007. / 100000.;
        cell.phases[0].internal_energy *= 100007. / 100000.;
    }
    solver.initialize(state);
    const auto accepted = solver.state();
    const auto checkpoint = solver.checkpoint();
    bool rejected = false;
    try
    {
        (void) solver.preview(.01);
    }
    catch (const std::exception& error)
    {
        rejected = true;
        EXPECT_NE(std::string(error.what()).find("after mechanical gauge transfer"), std::string::npos);
    }
    EXPECT_TRUE(rejected);
    EXPECT_EQ(solver.state().revision, accepted.revision);
    EXPECT_DOUBLE_EQ(solver.state().time, accepted.time);
    EXPECT_DOUBLE_EQ(solver.state().thermodynamic_pressure, 100000);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(solver.state().cells[c].mechanical_pressure, 7);
        EXPECT_EQ(solver.state().cells[c].phases[0].moles, accepted.cells[c].phases[0].moles);
        EXPECT_DOUBLE_EQ(
            solver.state().cells[c].phases[0].internal_energy, accepted.cells[c].phases[0].internal_energy);
    }
    EXPECT_NO_THROW(solver.restore(checkpoint));
}
} // namespace
