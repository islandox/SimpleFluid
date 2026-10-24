/** Explicit population transport, source, finite-count and collective transaction regressions. */
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
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
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    std::shared_ptr<Flow> flow;
    Flow::Options options;
    Fixture(unsigned cells = 2, bool slip = false, bool empty_large = false, double number_scale = 1,
        bool nonuniform = false)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        ArrReal z;
        for (unsigned k = 0; k <= cells; ++k)
            z.push_back(double(k) / cells);
        if (nonuniform)
        {
            if (cells != 2)
                throw std::invalid_argument("Nonuniform audit requires exactly two physical cells.");
            z = {0, .3, 1};
        }
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, z}),
            distribution);
        const std::vector roles{R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed, R::SpeciesPhase::Resolved};
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {103, "supplied_A", .02, {{"X", 1}}, roles}, {709, "supplied_B", .04, {{"Y", 1}}, roles}});
        liquid =
            R::make_linear_compressible_mixture_liquid(catalog, {250, 600, 10000, 2000000}, [](std::span<const double>)
                { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; });
        gas = R::make_ideal_gas_mixture(catalog, {{20, 300, 6000}, {20, 300, 6000}}, {250, 600, 10000, 2000000});
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "micro"},
            {Flow::PhaseRole::DispersedGas, "large"}};
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas, gas}, options);
        Flow::State state;
        state.thermodynamic_pressure = 100000;
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const double volume = mesh->cell_volume(static_cast<LO>(c)), liquid_mass = .8 * volume * 1000;
            const double micro = 100000 * (empty_large ? .2 : .1) * volume / (gas_constant * 300),
                         large = empty_large ? 0 : 100000 * .1 * volume / (gas_constant * 300);
            const double fraction = mesh->cell_centroid(static_cast<LO>(c)).z < .5 ? .8 : .2;
            state.cells[c].phases = {{liquid, {liquid_mass / .02, 0}, 100000, liquid_mass * 1e6, 300},
                {gas, {fraction * micro, (1 - fraction) * micro}, 100000, 6000 * micro, 300},
                {gas, {(1 - fraction) * large, fraction * large}, 100000, 6000 * large, large > 0 ? 300. : 0.}};
            state.cells[c].phase_numbers = {std::nullopt, number_scale * 20 * micro, number_scale * 40 * large};
            state.cells[c].drift_velocity = {
                Flow::Vector{}, Flow::Vector{0, 0, slip ? .1 : 0}, Flow::Vector{0, 0, slip ? -.05 : 0}};
        }
        flow->initialize(std::move(state));
    }
};
void expect_exact_numbers(const Flow::State& a, const Flow::State& b)
{
    EXPECT_EQ(a.revision, b.revision);
    EXPECT_DOUBLE_EQ(a.time, b.time);
    EXPECT_DOUBLE_EQ(a.thermodynamic_pressure, b.thermodynamic_pressure);
    ASSERT_EQ(a.cells.size(), b.cells.size());
    for (size_t c = 0; c < a.cells.size(); ++c)
    {
        EXPECT_EQ(a.cells[c].phase_numbers, b.cells[c].phase_numbers);
        EXPECT_DOUBLE_EQ(a.cells[c].mechanical_pressure, b.cells[c].mechanical_pressure);
        ASSERT_EQ(a.cells[c].drift_velocity.size(), b.cells[c].drift_velocity.size());
        for (size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_DOUBLE_EQ(a.cells[c].velocity.component(axis), b.cells[c].velocity.component(axis));
            for (size_t p = 0; p < a.cells[c].drift_velocity.size(); ++p)
                EXPECT_DOUBLE_EQ(
                    a.cells[c].drift_velocity[p].component(axis), b.cells[c].drift_velocity[p].component(axis));
        }
        ASSERT_EQ(a.cells[c].phases.size(), b.cells[c].phases.size());
        for (size_t p = 0; p < a.cells[c].phases.size(); ++p)
        {
            const auto& x = a.cells[c].phases[p];
            const auto& y = b.cells[c].phases[p];
            EXPECT_EQ(x.thermodynamics.get(), y.thermodynamics.get());
            EXPECT_EQ(x.moles, y.moles);
            EXPECT_DOUBLE_EQ(x.internal_energy, y.internal_energy);
            EXPECT_DOUBLE_EQ(x.temperature, y.temperature);
            EXPECT_DOUBLE_EQ(x.absolute_pressure, y.absolute_pressure);
        }
    }
}
TEST(LowMachPopulationNumbersTest, ExactPhaseFluxAdvectionPreservesIndependentNumberPerMoleInvariant)
{
    for (const double scale : {1., 1e-30})
    {
        Fixture f(8, true, false, scale);
        const auto before = f.flow->state();
        const auto trial = f.flow->preview(.001);
        const auto& snapshot = trial.transport_snapshot();
        using Scalar = Flow::Traits::scalar_cell_type;
        using Face = Flow::Traits::scalar_face_type;
        for (size_t a = 1; a < 3; ++a)
        {
            collective_detail::collective_local_validation(*f.mesh, "Independent number receipt layout",
                [&]
                {
                    if (snapshot.cells.size() != before.cells.size() ||
                        snapshot.new_control_volumes.size() != before.cells.size() ||
                        trial.owned_phase_transport().size() != f.mesh->num_owned_faces())
                        throw std::runtime_error("Number receipt cell/face layout differs.");
                    for (const auto& cell : snapshot.cells)
                        if (cell.phases.size() != 3 || !cell.phases[a].population_number)
                            throw std::runtime_error("Number receipt lost the declared phase inventory.");
                    for (const auto& face : trial.owned_phase_transport())
                        if (face.phase_volume_flux.size() != 3)
                            throw std::runtime_error("Number receipt lost the phase-face flux.");
                });
            Scalar concentration(f.mesh, 0.0, "independent_population_concentration");
            for (size_t c = 0; c < before.cells.size(); ++c)
            {
                const auto& inventory = snapshot.cells[c].phases[a];
                const double amount = std::accumulate(inventory.moles.begin(), inventory.moles.end(), 0.0);
                EXPECT_NEAR(*inventory.population_number, scale * (a == 1 ? 20 : 40) * amount,
                    1e-10 * std::abs(*inventory.population_number));
                EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[a], *inventory.population_number);
                concentration.set_owned_value(
                    static_cast<LO>(c), *inventory.population_number / snapshot.new_control_volumes[c]);
            }
            concentration.sync_ghosts();
            Face number_flux(f.mesh, 0.0, "independently_audited_number_flux");
            const auto packets = trial.owned_phase_transport();
            for (size_t face = 0; face < packets.size(); ++face)
            {
                const LO fid = static_cast<LO>(face);
                if (f.mesh->is_geometry_exterior_face(fid))
                    continue;
                const double phi = packets[face].phase_volume_flux[a];
                const LO donor = phi >= 0 ? f.mesh->owner_cell(fid) : f.mesh->neighbor_cell(fid);
                number_flux.set_owned_value(fid, phi * concentration.local_value(donor)); // No extra alpha.
            }
            number_flux.sync_ghosts();
            double local_before = 0, local_after = 0, global_before = 0, global_after = 0;
            for (size_t c = 0; c < before.cells.size(); ++c)
            {
                double balance = 0;
                const LO cell = static_cast<LO>(c);
                for (size_t face = 0; face < f.mesh->num_faces(); ++face)
                {
                    const LO fid = static_cast<LO>(face);
                    if (f.mesh->owner_cell(fid) == cell)
                        balance += number_flux.local_value(fid);
                    else if (f.mesh->neighbor_cell(fid) == cell)
                        balance -= number_flux.local_value(fid);
                }
                const double expected = *before.cells[c].phase_numbers[a] - .001 * balance;
                EXPECT_NEAR(*snapshot.cells[c].phases[a].population_number, expected,
                    2e-10 * std::max(std::abs(expected), std::abs(*before.cells[c].phase_numbers[a])));
                local_before += *before.cells[c].phase_numbers[a];
                local_after += *trial.state().cells[c].phase_numbers[a];
            }
            const auto comm = f.mesh->owned_cell_map()->getComm();
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_before, &global_before);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_after, &global_after);
            EXPECT_NEAR(global_before, global_after, f.options.conservation_relative_tolerance * global_before);
        }
        expect_exact_numbers(f.flow->state(), before);
    }
}
TEST(LowMachPopulationNumbersTest, ExplicitNumberBirthDeathEntersOnceWithoutMassEnergyOrImplicitCreation)
{
    Fixture f;
    const auto accepted = f.flow->state();
    int calls = 0;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "explicit-number-moments", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++calls;
            Flow::SourceStageResult result;
            result.sources.resize(context.transport.cells.size());
            result.gross_events.resize(context.transport.cells.size());
            for (size_t c = 0; c < result.sources.size(); ++c)
            {
                const double born = .25 * *context.transport.cells[c].phases[1].population_number;
                const double died = .5 * *context.transport.cells[c].phases[2].population_number;
                result.sources[c].population_number_changes = {0, born, -died};
                result.gross_events[c] = {born, died};
            }
            return result;
        }});
    const auto checkpoint = f.flow->checkpoint();
    const auto first = f.flow->preview_with_sources(.01, provider);
    const auto retained = f.flow->preview_with_sources(.01, provider);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_NEAR(*first.state().cells[c].phase_numbers[1], 1.25 * *accepted.cells[c].phase_numbers[1],
            1e-10 * *accepted.cells[c].phase_numbers[1]);
        EXPECT_NEAR(*first.state().cells[c].phase_numbers[2], .5 * *accepted.cells[c].phase_numbers[2],
            1e-10 * *accepted.cells[c].phase_numbers[2]);
        EXPECT_EQ(first.state().cells[c].phases[0].moles, accepted.cells[c].phases[0].moles);
    }
    expect_exact_numbers(f.flow->state(), accepted);
    const int before = calls;
    f.flow->commit(first);
    EXPECT_EQ(calls, before);
    EXPECT_THROW(f.flow->commit(retained), std::exception);
    const auto expected = f.flow->state();
    f.flow->restore(checkpoint);
    const auto replay = f.flow->preview_with_sources(.01, provider);
    expect_exact_numbers(replay.state(), expected);
}
TEST(LowMachPopulationNumbersTest, NewOccupiedBinRequiresExplicitPairedNumberAndRetainsExactSnapshot)
{
    Fixture f(2, false, true);
    const auto accepted = f.flow->state();
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "declared-new-bin", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            Flow::SourceStageResult result;
            result.sources.resize(context.transport.cells.size());
            for (size_t c = 0; c < result.sources.size(); ++c)
            {
                const auto& donor = context.transport.cells[c].phases[1];
                auto transfer = donor.moles;
                double amount = 0;
                for (auto& n : transfer)
                {
                    n *= .1;
                    amount += n;
                }
                auto& source = result.sources[c];
                source.species_moles.assign(3, std::vector<double>(2));
                for (size_t i = 0; i < 2; ++i)
                {
                    source.species_moles[1][i] = -transfer[i];
                    source.species_moles[2][i] = transfer[i];
                }
                const double carried = (20 + gas_constant) * 300 * amount;
                source.internal_energy = {0, -carried, carried};
                source.population_number_changes = {0, 0, 10};
            }
            return result;
        }});
    const auto trial = f.flow->preview_with_sources(.01, provider);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(*trial.transport_snapshot().cells[c].phases[2].population_number, 0);
        EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[2], 10);
        EXPECT_GT(std::accumulate(
                      trial.state().cells[c].phases[2].moles.begin(), trial.state().cells[c].phases[2].moles.end(), 0.),
            0);
        EXPECT_DOUBLE_EQ(trial.integrated_sources()[c].population_number_changes[2], 10);
    }
    expect_exact_numbers(f.flow->state(), accepted);
}
TEST(LowMachPopulationNumbersTest, LocalDepletionBulkSourcesAndMissingBirthRejectBeforePublication)
{
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        Fixture f;
        const auto accepted = f.flow->state();
        const auto retained = f.flow->preview(.01);
        std::vector<Flow::CellSources> sources(accepted.cells.size());
        for (size_t c = 0; c < sources.size(); ++c)
        {
            auto& source = sources[c];
            source.population_number_changes = {0, 0, 0};
            if (scenario == 0)
                source.population_number_changes[1] = -2 * *accepted.cells[c].phase_numbers[1];
            if (scenario == 1)
                source.population_number_changes[1] = -*accepted.cells[c].phase_numbers[1]; // Gas still remains.
            if (scenario == 2)
                source.population_number_changes[0] = 1; // Bulk never owns N.
            if (scenario == 3)
                source.population_number_changes[1] = std::numeric_limits<double>::quiet_NaN();
            if (scenario == 4)
            {
                source.species_moles.assign(3, std::vector<double>(2));
                for (size_t i = 0; i < 2; ++i)
                    source.species_moles[1][i] = -accepted.cells[c].phases[1].moles[i]; // No automatic N death.
            }
        }
        EXPECT_THROW((void) f.flow->preview(.01, sources), std::exception);
        expect_exact_numbers(f.flow->state(), accepted);
        EXPECT_NO_THROW(f.flow->commit(retained));
    }
    Fixture empty(2, false, true);
    std::vector<Flow::CellSources> sources(empty.flow->state().cells.size());
    for (auto& source : sources)
        source.population_number_changes = {0, 0, 1};
    EXPECT_THROW((void) empty.flow->preview(.01, sources), std::exception); // Number without any gas is not a birth.
}
TEST(LowMachPopulationNumbersTest, InvalidInitialNumberOwnershipRejectsAndRetainsValidTrial)
{
    Fixture f;
    const auto accepted = f.flow->state();
    const auto retained = f.flow->preview(.01);
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        auto candidate = accepted;
        for (auto& cell : candidate.cells)
        {
            if (scenario == 0)
                cell.phase_numbers.clear();
            if (scenario == 1)
                cell.phase_numbers[1] = 0;
            if (scenario == 2)
                cell.phase_numbers[0] = 0;
            if (scenario == 3)
                cell.phase_numbers[1] = std::numeric_limits<double>::infinity();
            if (scenario == 4)
                cell.phase_numbers[1] = -1;
        }
        EXPECT_THROW(f.flow->initialize(candidate), std::exception);
        expect_exact_numbers(f.flow->state(), accepted);
    }
    EXPECT_NO_THROW(f.flow->commit(retained));
}
TEST(LowMachPopulationNumbersTest, RoleSchemaRejectsRankDivergenceIncludingActualEmptyOwners)
{
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Rank-divergent role metadata requires multiple MPI ranks.";
    int empty = f.mesh->num_owned_cells() == 0 ? 1 : 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    if (comm->getSize() == 4)
        EXPECT_EQ(total_empty, 2);
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        auto options = f.options;
        if (comm->getRank() == comm->getSize() - 1)
        {
            if (scenario == 0)
                options.phase_roles[1].role = Flow::PhaseRole::BulkGas;
            if (scenario == 1)
                options.phase_roles[1].identity = "other"; // Same length as micro.
            if (scenario == 2)
                options.phase_roles[1].identity.clear();
        }
        EXPECT_THROW((void) Flow(f.mesh, std::vector{f.liquid, f.gas, f.gas}, options), std::exception);
    }
}
TEST(LowMachPopulationNumbersTest, FailedNumberProviderOnOneRankPreservesEarlierImmutableTrial)
{
    Fixture f;
    const auto accepted = f.flow->state();
    const auto retained = f.flow->preview(.01);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "invalid-local-number-event", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            Flow::SourceStageResult result;
            result.sources.resize(context.transport.cells.size());
            if (comm->getRank() == 0 && !result.sources.empty())
                result.sources[0].population_number_changes = {0, -1e100, 0};
            return result;
        }});
    EXPECT_THROW((void) f.flow->preview_with_sources(.01, provider), std::exception);
    expect_exact_numbers(f.flow->state(), accepted);
    EXPECT_NO_THROW(f.flow->commit(retained));
}
TEST(LowMachPopulationNumbersTest, NumberSourceConvergenceCannotBeBypassedByUnchangedMassAndEnergy)
{
    Fixture f;
    int evaluations = 0;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "number-only-corrector", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++evaluations;
            Flow::SourceStageResult result;
            result.sources.resize(context.transport.cells.size());
            for (auto& source : result.sources)
                source.population_number_changes = {0, context.transport.stage.corrector == 0 ? 2. : 1., 0};
            return result;
        }});
    const auto trial = f.flow->preview_with_sources(.01, provider);
    EXPECT_EQ(evaluations, 3);
    EXPECT_LE(trial.diagnostics().maximum_source_residual, 1);
}

TEST(LowMachPopulationNumbersTest, CompleteDeclaredIsobaricRelocationCarriesAllSpeciesEnergyAndExplicitNumber)
{
    Fixture f(2, false, true);
    const auto accepted = f.flow->state();
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "complete-isobaric-bin-relocation", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            Flow::SourceStageResult result;
            result.sources.resize(context.transport.cells.size());
            for (size_t c = 0; c < result.sources.size(); ++c)
            {
                const auto& donor = context.transport.cells[c].phases[1];
                auto& source = result.sources[c];
                source.species_moles.assign(3, std::vector<double>(2));
                for (size_t i = 0; i < 2; ++i)
                {
                    source.species_moles[1][i] = -donor.moles[i];
                    source.species_moles[2][i] = donor.moles[i];
                }
                // Supplied total carried enthalpy for this explicitly isobaric synthetic event.
                // No extra latent term, compensating heat or solver p*dV source is introduced.
                const double carried = donor.internal_energy + 100000 * donor.advected_material_volume;
                source.internal_energy = {0, -carried, carried};
                source.population_number_changes = {0, -*donor.population_number, *donor.population_number};
            }
            return result;
        }});
    const auto trial = f.flow->preview_with_sources(.01, provider);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const auto& empty = trial.state().cells[c].phases[1];
        EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[1], 0);
        EXPECT_EQ(empty.moles, std::vector<double>({0, 0}));
        EXPECT_DOUBLE_EQ(empty.internal_energy, 0);
        EXPECT_DOUBLE_EQ(empty.temperature, 0);
        EXPECT_NEAR(*trial.state().cells[c].phase_numbers[2], *accepted.cells[c].phase_numbers[1],
            1e-10 * *accepted.cells[c].phase_numbers[1]);
    }
    expect_exact_numbers(f.flow->state(), accepted);
}
TEST(LowMachPopulationNumbersTest, SmallCellExactDeathRetainsGlobalBudgetBesideHugeUnchangedStock)
{
    Fixture f;
    auto state = f.flow->state();
    for (size_t c = 0; c < state.cells.size(); ++c)
        state.cells[c].phase_numbers[1] = f.mesh->cell_centroid(static_cast<LO>(c)).z < .5 ? 1e20 : 1.;
    f.flow->initialize(state);
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "small-cell-explicit-death", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            Flow::SourceStageResult result;
            result.sources.resize(context.transport.cells.size());
            for (size_t c = 0; c < result.sources.size(); ++c)
            {
                if (f.mesh->cell_centroid(static_cast<LO>(c)).z < .5)
                    continue;
                const auto& donor = context.transport.cells[c].phases[1];
                auto& source = result.sources[c];
                source.species_moles.assign(3, std::vector<double>(2));
                for (size_t i = 0; i < 2; ++i)
                {
                    source.species_moles[1][i] = -donor.moles[i];
                    source.species_moles[2][i] = donor.moles[i];
                }
                // Explicit synthetic isobaric material relocation: no orphan gas or inferred number death.
                const double carried = donor.internal_energy + 100000 * donor.advected_material_volume;
                source.internal_energy = {0, -carried, carried};
                source.population_number_changes = {0, -1, 0};
            }
            return result;
        }});
    const auto trial = f.flow->preview_with_sources(.01, provider);
    EXPECT_DOUBLE_EQ(trial.diagnostics().population_number_errors[1], 0);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        if (f.mesh->cell_centroid(static_cast<LO>(c)).z < .5)
            EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[1], 1e20);
        else
        {
            EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[1], 0);
            EXPECT_EQ(trial.state().cells[c].phases[1].moles, std::vector<double>({0, 0}));
        }
    }
}
TEST(LowMachPopulationNumbersTest, UnequalVolumesAndHeterogeneousNumberRatiosMatchClosedUpwindBalance)
{
    Fixture f(2, true, false, 1, true);
    auto state = f.flow->state();
    for (size_t c = 0; c < state.cells.size(); ++c)
        for (size_t a = 1; a < 3; ++a)
        {
            const bool left = f.mesh->cell_centroid(static_cast<LO>(c)).z < .3;
            const double ratio = a == 1 ? (left ? 20 : 70) : (left ? 35 : 11);
            state.cells[c].phase_numbers[a] = ratio * std::accumulate(state.cells[c].phases[a].moles.begin(),
                                                          state.cells[c].phases[a].moles.end(), 0.);
        }
    f.flow->initialize(state);
    const auto accepted = f.flow->state();
    const auto checkpoint = f.flow->checkpoint();
    const auto trial = f.flow->preview(.001);
    const auto& snapshot = trial.transport_snapshot();
    const auto packets = trial.owned_phase_transport();
    int empty = f.mesh->num_owned_cells() == 0, global_empty = 0;
    Teuchos::reduceAll(*f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &empty, &global_empty);
    if (f.mesh->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_EQ(global_empty, 2);
    double transported_flux = 0;
    for (size_t a = 1; a < 3; ++a)
    {
        std::array<double, 7> local{}, global{};
        collective_detail::collective_local_validation(*f.mesh, "Nonuniform number receipt preflight",
            [&]
            {
                if (snapshot.cells.size() != accepted.cells.size() || packets.size() != f.mesh->num_owned_faces())
                    throw std::runtime_error("Nonuniform number receipt layout differs.");
                for (size_t c = 0; c < accepted.cells.size(); ++c)
                {
                    const size_t side = f.mesh->cell_centroid(static_cast<LO>(c)).z < .3 ? 0 : 1;
                    if (snapshot.cells[c].phases.size() != 3 || !snapshot.cells[c].phases[a].population_number)
                        throw std::runtime_error("Nonuniform number receipt lost explicit Nadv.");
                    local[side] = *accepted.cells[c].phase_numbers[a];
                    local[2 + side] = *snapshot.cells[c].phases[a].population_number;
                    local[4 + side] = f.mesh->cell_volume(static_cast<LO>(c));
                }
                for (size_t face = 0; face < packets.size(); ++face)
                {
                    const LO fid = static_cast<LO>(face);
                    if (f.mesh->is_geometry_exterior_face(fid))
                        continue;
                    if (packets[face].phase_volume_flux.size() != 3)
                        throw std::runtime_error("Nonuniform phase flux receipt layout differs.");
                    const double owner_z = f.mesh->cell_centroid(f.mesh->owner_cell(fid)).z;
                    const double neighbor_z = f.mesh->cell_centroid(f.mesh->neighbor_cell(fid)).z;
                    local[6] += (owner_z < neighbor_z ? 1 : -1) * packets[face].phase_volume_flux[a];
                }
            });
        Teuchos::reduceAll(*f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 7, local.data(), global.data());
        EXPECT_DOUBLE_EQ(global[4], .3);
        EXPECT_DOUBLE_EQ(global[5], .7);
        // Independent exact 2x2 closed implicit upwind solve, N/CV and no phase-fraction factor.
        const double q = global[6];
        transported_flux += std::abs(q);
        double left = global[0], right = global[1];
        if (q >= 0)
        {
            left = global[0] / (1 + .001 * q / global[4]);
            right = global[1] + .001 * q * left / global[4];
        }
        else
        {
            right = global[1] / (1 - .001 * q / global[5]);
            left = global[0] - .001 * q * right / global[5];
        }
        EXPECT_NEAR(global[2], left, 2e-10 * std::max(global[0], left));
        EXPECT_NEAR(global[3], right, 2e-10 * std::max(global[1], right));
        EXPECT_NEAR(global[2] + global[3], global[0] + global[1],
            f.options.conservation_relative_tolerance * (global[0] + global[1]));
    }
    EXPECT_GT(transported_flux, 0);
    expect_exact_numbers(f.flow->state(), accepted);
    f.flow->commit(trial);
    const auto expected = f.flow->state();
    f.flow->restore(checkpoint);
    expect_exact_numbers(f.flow->state(), accepted);
    const auto replay = f.flow->preview(.001);
    expect_exact_numbers(replay.state(), expected);
}
} // namespace
