/** Native QBMM ownership, pre-transport capability gates, and transactional replay. */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
using Shape = R::QbmmShapeState;
using Family = std::shared_ptr<const R::QbmmPopulationSchema>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::vector<std::shared_ptr<const R::MixtureThermodynamics>> eos;
    Flow::Options options;
    Family family;
    std::shared_ptr<unsigned> liquid_evaluations = std::make_shared<unsigned>(0);

    explicit Fixture(unsigned cells = 1, bool allow_resolved = false)
    {
        ArrReal heights;
        for (unsigned i = 0; i <= cells; ++i)
            heights.push_back(static_cast<double>(i));
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, heights}),
            distribution);
        std::vector phases{R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed};
        if (allow_resolved)
            phases.push_back(R::SpeciesPhase::Resolved);
        catalog = std::make_shared<const R::SpeciesCatalog>(
            std::vector<R::SpeciesDefinition>{{11, "A", .02, {{"X", 1}}, phases}, {29, "B", .04, {{"X", 2}}, phases}});
        const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};
        auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds,
            [calls = liquid_evaluations](std::span<const double>)
            {
                ++*calls;
                return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 0, 0, 0};
            });
        auto gas = R::make_ideal_gas_mixture(catalog, {{20, 300, 0}, {20, 300, 0}}, bounds);
        eos = {liquid, gas, gas};
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "fine"},
            {Flow::PhaseRole::DispersedGas, "coarse"}};
        family = make_family(catalog, options.phase_roles[1], .04 / 14);
        options.qbmm_families = {nullptr, family, nullptr};
    }

    static Family make_family(
        const std::shared_ptr<const R::SpeciesCatalog>& catalog, const Flow::PhaseRoleDescriptor& role, double scale)
    {
        return std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), role.identity, role,
            scale, std::vector<R::QbmmMassFraction>{{11, .5}, {29, .5}});
    }

    Flow::State initial(bool vacuum = false) const
    {
        Flow::State state;
        state.thermodynamic_pressure = 100000;
        state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*family), std::nullopt};
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            // Weights 2 and 3 at scaled masses 1 and 4 give [5,14,50,194].
            // The separate count-only coarse phase contains one mole of B in either case.
            const double a = vacuum ? 0 : 1, b = .5 * a;
            const double gas_volume = (a + b + 1) * 8.31446261815324 * 300 / 100000;
            const double liquid_mass = 1000 * (mesh->cell_volume(static_cast<LO>(c)) - gas_volume);
            auto& cell = state.cells[c];
            cell.phases = {{eos[0], {liquid_mass / .02, 0}, 100000, 0, 300},
                {eos[1], {a, b}, 100000, 0, vacuum ? 0. : 300.}, {eos[2], {0, 1}, 100000, 0, 300}};
            cell.phase_numbers = {std::nullopt, vacuum ? 0. : 5., 17.};
            cell.qbmm_shape = {std::nullopt, vacuum ? Shape{} : Shape{50, 194}, std::nullopt};
        }
        return state;
    }

    std::shared_ptr<Flow> solver() const { return std::make_shared<Flow>(mesh, eos, options); }
};

void expect_state(const Flow::State& state, const Flow::State& expected)
{
    EXPECT_EQ(state.qbmm_families, expected.qbmm_families);
    EXPECT_EQ(state.time, expected.time);
    EXPECT_EQ(state.revision, expected.revision);
    EXPECT_EQ(state.thermodynamic_pressure, expected.thermodynamic_pressure);
    ASSERT_EQ(state.cells.size(), expected.cells.size());
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const auto& cell = state.cells[c];
        const auto& other = expected.cells[c];
        EXPECT_EQ(cell.qbmm_shape, other.qbmm_shape);
        EXPECT_EQ(cell.phase_numbers, other.phase_numbers);
        EXPECT_EQ(cell.mechanical_pressure, other.mechanical_pressure);
        ASSERT_EQ(cell.phases.size(), other.phases.size());
        ASSERT_EQ(cell.drift_velocity.size(), other.drift_velocity.size());
        for (size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_EQ(cell.velocity.component(axis), other.velocity.component(axis));
            for (size_t p = 0; p < cell.drift_velocity.size(); ++p)
                EXPECT_EQ(cell.drift_velocity[p].component(axis), other.drift_velocity[p].component(axis));
        }
        for (size_t p = 0; p < cell.phases.size(); ++p)
        {
            EXPECT_EQ(cell.phases[p].moles, other.phases[p].moles);
            EXPECT_EQ(cell.phases[p].internal_energy, other.phases[p].internal_energy);
            EXPECT_EQ(cell.phases[p].temperature, other.phases[p].temperature);
            EXPECT_EQ(cell.phases[p].absolute_pressure, other.phases[p].absolute_pressure);
        }
    }
}

void expect_identity(const Flow::Trial& trial, const Flow::State& accepted, R::InversionStatus status)
{
    const auto& snapshot = trial.transport_snapshot();
    EXPECT_EQ(snapshot.qbmm_families, accepted.qbmm_families);
    EXPECT_EQ(snapshot.schema_fingerprint, accepted.qbmm_families.at(1)->catalog_fingerprint);
    EXPECT_EQ(snapshot.accepted_revision, accepted.revision);
    EXPECT_EQ(snapshot.accepted_time, accepted.time);
    EXPECT_EQ(trial.state().qbmm_families, accepted.qbmm_families);
    ASSERT_EQ(snapshot.cells.size(), accepted.cells.size());
    ASSERT_EQ(trial.qbmm_inversion_status().size(), accepted.cells.size());
    ASSERT_EQ(trial.state().cells.size(), accepted.cells.size());
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const auto& before = accepted.cells[c];
        ASSERT_EQ(snapshot.cells[c].phases.size(), before.phases.size());
        EXPECT_EQ(trial.qbmm_inversion_status()[c],
            (std::vector<std::optional<R::InversionStatus>>{std::nullopt, status, std::nullopt}));
        EXPECT_EQ(trial.state().cells[c].qbmm_shape, before.qbmm_shape);
        for (size_t p = 0; p < before.phases.size(); ++p)
        {
            const auto& pre_source = snapshot.cells[c].phases[p];
            EXPECT_EQ(pre_source.qbmm_shape, before.qbmm_shape[p]);
            EXPECT_EQ(pre_source.population_number, before.phase_numbers[p]);
            EXPECT_EQ(pre_source.moles, before.phases[p].moles);
            EXPECT_EQ(pre_source.internal_energy, before.phases[p].internal_energy);
        }
    }
}

TEST(LowMachQbmmStateTest, FamilyBindingsRequireExactCatalogNamedRoleAndUniquePhase)
{
    Fixture f;
    EXPECT_EQ(Flow::qbmm_family_stamp(*f.family),
        (Flow::QbmmFamilyStamp{f.catalog->fingerprint(), f.family->fingerprint(), .04 / 14}));
    const auto rejects = [&](auto change)
    {
        auto options = f.options;
        change(options);
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    };
    rejects([](auto& o) { o.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures; });
    rejects([](auto& o) { o.qbmm_families.pop_back(); });
    rejects([](auto& o) { std::swap(o.qbmm_families[0], o.qbmm_families[1]); });
    rejects([](auto& o) { std::swap(o.qbmm_families[1], o.qbmm_families[2]); });
    rejects([](auto& o) { o.qbmm_families[2] = o.qbmm_families[1]; });
    rejects([](auto& o) { o.phase_roles[2].identity = o.phase_roles[1].identity; });
    auto definitions = f.catalog->species();
    definitions[0].name += "_foreign";
    const auto foreign = std::make_shared<const R::SpeciesCatalog>(definitions);
    rejects([&](auto& o) { o.qbmm_families[1] = f.make_family(foreign, o.phase_roles[1], .04 / 14); });
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto accepted = flow->state();
    const auto reject_stamp = [&](auto change)
    {
        auto bad = accepted;
        change(bad.qbmm_families);
        EXPECT_THROW(flow->initialize(bad), std::exception);
        expect_state(flow->state(), accepted);
    };
    reject_stamp([](auto& layout) { layout.clear(); });
    reject_stamp([](auto& layout) { layout[1].reset(); });
    reject_stamp([](auto& layout) { ++layout[1]->catalog_fingerprint; });
    reject_stamp([](auto& layout) { ++layout[1]->family_fingerprint; });
    reject_stamp([](auto& layout) { layout[1]->mass_scale_kg *= 2; });
}

TEST(LowMachQbmmStateTest, LocalShapeAndFixedCompositionFailuresPreserveAcceptedState)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto accepted = flow->state();
    const auto retained = flow->preview(.01);
    const auto rejects = [&](auto change)
    {
        auto bad = accepted;
        for (auto& cell : bad.cells)
            change(cell);
        EXPECT_THROW(flow->initialize(bad), std::exception);
        expect_state(flow->state(), accepted);
    };
    rejects([](auto& cell) { cell.qbmm_shape.clear(); });
    rejects([](auto& cell) { cell.qbmm_shape[1].reset(); });
    rejects([](auto& cell) { cell.qbmm_shape[2] = Shape{}; });
    rejects([](auto& cell) { cell.qbmm_shape[1]->H2 = 1; }); // Positive H2, but H0*H2 < H1^2.
    rejects([](auto& cell) { cell.qbmm_shape[1]->H3 = std::numeric_limits<double>::quiet_NaN(); });
    rejects([](auto& cell) { cell.phase_numbers[1] = 0; });
    // Positive weights at scaled masses zero and fourteen. Gas support is strictly positive.
    rejects(
        [](auto& cell)
        {
            cell.phase_numbers[1] = 2;
            cell.qbmm_shape[1] = Shape{196, 2744};
        });
    // Swap moles at constant total moles and zero reference energy. Ideal-gas occupancy still matches.
    rejects([](auto& cell) { std::swap(cell.phases[1].moles[0], cell.phases[1].moles[1]); });
    auto vacuum = f.initial(true);
    for (auto& cell : vacuum.cells)
        cell.qbmm_shape[1] = Shape{1, 1};
    EXPECT_THROW(flow->initialize(vacuum), std::exception);
    expect_state(flow->state(), accepted);
    flow->commit(retained);
    flow->initialize(f.initial(true));
    expect_identity(flow->preview(.01), flow->state(), R::InversionStatus::Absent);
}

TEST(LowMachQbmmStateTest, TwoFamiliesUseTheirOwnPhaseCountMassScaleAndCellDistribution)
{
    Fixture f(2);
    auto state = f.initial();
    auto coarse = std::make_shared<const R::QbmmPopulationSchema>(f.catalog, f.catalog->fingerprint(), "coarse",
        f.options.phase_roles[2], .02, std::vector<R::QbmmMassFraction>{{11, 0}, {29, 1}});
    f.options.qbmm_families[2] = coarse;
    state.qbmm_families[2] = Flow::qbmm_family_stamp(*coarse);
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        auto& cell = state.cells[c];
        if (f.mesh->cell_centroid(static_cast<LO>(c)).z > 1)
            cell.qbmm_shape[1] = Shape{196. / 5, 2744. / 25};
        cell.phase_numbers[2] = 1;
        cell.qbmm_shape[2] = Shape{4, 8}; // One mole of pure B gives H1 = 2 at this scale.
    }
    auto flow = f.solver();
    flow->initialize(state);
    const auto trial = flow->preview(.01);
    EXPECT_EQ(trial.transport_snapshot().qbmm_families, state.qbmm_families);
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const auto expected = f.mesh->cell_centroid(static_cast<LO>(c)).z < 1 ? R::InversionStatus::TwoNodes
                                                                              : R::InversionStatus::RankOne;
        EXPECT_EQ(trial.qbmm_inversion_status().at(c),
            (std::vector<std::optional<R::InversionStatus>>{std::nullopt, expected, R::InversionStatus::RankOne}));
        EXPECT_EQ(trial.state().cells.at(c).qbmm_shape, state.cells[c].qbmm_shape);
        for (size_t p = 1; p < 3; ++p)
        {
            const auto& phase = trial.transport_snapshot().cells.at(c).phases.at(p);
            EXPECT_EQ(phase.qbmm_shape, state.cells[c].qbmm_shape[p]);
            EXPECT_EQ(phase.moles, state.cells[c].phases[p].moles);
            EXPECT_EQ(phase.population_number, state.cells[c].phase_numbers[p]);
        }
    }
}

TEST(LowMachQbmmStateTest, LegacyNumberAndMolesRequireAnExplicitShapeDistribution)
{
    Fixture f;
    auto flow = f.solver();
    auto legacy = f.initial();
    legacy.qbmm_families.clear();
    for (auto& cell : legacy.cells)
        cell.qbmm_shape.clear();
    EXPECT_THROW(flow->initialize(legacy), std::exception);
    Flow::QbmmInitialization choices;
    choices.cells.resize(legacy.cells.size());
    for (auto& row : choices.cells)
        row.resize(f.eos.size());
    EXPECT_THROW(flow->initialize_qbmm(legacy, choices), std::exception);
    for (auto& row : choices.cells)
        row[1] = Flow::QbmmMonodisperse{};
    flow->initialize_qbmm(legacy, choices);
    const auto mono = flow->state();
    for (size_t c = 0; c < mono.cells.size(); ++c)
    {
        EXPECT_TRUE(mono.cells[c].qbmm_shape.at(1));
        if (!mono.cells[c].qbmm_shape[1])
            continue;
        const auto& shape = *mono.cells[c].qbmm_shape[1];
        EXPECT_NEAR(shape.H2, 196. / 5, 1e-12);
        EXPECT_NEAR(shape.H3, 2744. / 25, 1e-12);
        EXPECT_EQ(mono.cells[c].phase_numbers, legacy.cells[c].phase_numbers);
        EXPECT_EQ(mono.cells[c].phases[1].moles, legacy.cells[c].phases[1].moles);
    }
    expect_identity(flow->preview(.01), mono, R::InversionStatus::RankOne);
    EXPECT_THROW(flow->initialize_qbmm(mono, choices), std::exception);
    for (auto& row : choices.cells)
        row[1] = std::vector<R::QbmmAtom>{{2, 1}, {3, 5}}; // Correct count, wrong first moment.
    EXPECT_THROW(flow->initialize_qbmm(legacy, choices), std::exception);
    expect_state(flow->state(), mono);
    for (auto& row : choices.cells)
        row[1] = std::vector<R::QbmmAtom>{{2, 1}, {3, 4}};
    flow->initialize_qbmm(legacy, choices);
    for (const auto& cell : flow->state().cells)
        EXPECT_EQ(cell.qbmm_shape.at(1), (Shape{50, 194}));
    expect_identity(flow->preview(.01), flow->state(), R::InversionStatus::TwoNodes);
}

TEST(LowMachQbmmStateTest, MigratesOwnedCountOnlyCheckpointIncludingAcceptedHistory)
{
    Fixture f(2);
    auto count_options = f.options;
    count_options.qbmm_families.clear();
    Flow previous(f.mesh, f.eos, count_options);
    auto legacy = f.initial();
    legacy.qbmm_families.clear();
    for (auto& cell : legacy.cells)
        cell.qbmm_shape.clear();
    // Capture nonzero internal face history and exercise native transport after migration.
    auto moving = legacy;
    for (auto& cell : moving.cells)
        cell.velocity = Flow::Vector{0, 0, .125};
    previous.initialize(moving);
    const auto moving_checkpoint = previous.checkpoint();
    previous.initialize(legacy);
    previous.advance(.02);
    const auto saved = previous.checkpoint();
    const auto old_state = previous.state();
    std::vector<double> old_flux;
    for (size_t face = 0; face < f.mesh->num_owned_faces(); ++face)
        old_flux.push_back(previous.volume_flux().owned_value(static_cast<LO>(face)));
    const auto diagnostics = previous.diagnostics();
    previous.advance(.01);

    auto flow = f.solver();
    Flow::QbmmInitialization initialization;
    initialization.cells.resize(legacy.cells.size());
    for (auto& row : initialization.cells)
        row = {std::nullopt, Flow::QbmmMonodisperse{}, std::nullopt};
    flow->restore_qbmm_checkpoint(previous, saved, initialization);
    EXPECT_EQ(flow->state().time, old_state.time);
    EXPECT_EQ(flow->state().revision, old_state.revision);
    EXPECT_EQ(flow->diagnostics().correctors, diagnostics.correctors);
    EXPECT_EQ(flow->diagnostics().species_errors, diagnostics.species_errors);
    for (size_t face = 0; face < old_flux.size(); ++face)
        EXPECT_EQ(flow->volume_flux().owned_value(static_cast<LO>(face)), old_flux[face]);
    const auto after = flow->state();
    for (size_t c = 0; c < after.cells.size(); ++c)
    {
        EXPECT_EQ(after.cells[c].phase_numbers, old_state.cells[c].phase_numbers);
        EXPECT_EQ(after.cells[c].phases[1].moles, old_state.cells[c].phases[1].moles);
        EXPECT_NEAR(after.cells[c].qbmm_shape[1]->H2, 196. / 5, 1e-12);
    }
    const auto retained = flow->preview(.01);
    for (auto& row : initialization.cells)
        row[1] = std::vector<R::QbmmAtom>{{1, 1}};
    EXPECT_THROW(flow->restore_qbmm_checkpoint(previous, saved, initialization), std::exception);
    EXPECT_THROW(flow->restore_qbmm_checkpoint(*flow, saved, initialization), std::exception);
    expect_state(flow->state(), after);
    flow->commit(retained);
    for (auto& row : initialization.cells)
        row[1] = Flow::QbmmMonodisperse{};
    flow->restore_qbmm_checkpoint(previous, moving_checkpoint, initialization);
    double local_max = 0, global_max = 0;
    for (size_t face = 0; face < old_flux.size(); ++face)
        local_max = std::max(local_max, std::abs(flow->volume_flux().owned_value(static_cast<LO>(face))));
    Teuchos::reduceAll(*f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_max, &global_max);
    EXPECT_EQ(global_max, .125);
    EXPECT_NO_THROW((void) flow->preview(.01));
}

TEST(LowMachQbmmStateTest, CoexistingTrialsOwnSnapshotsSourcesAndInversionResultsAcrossRollback)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto origin = flow->state();
    auto controller = flow->acquire_controller();
    const auto checkpoint = flow->checkpoint();
    std::vector<Flow::CellSources> sources(origin.cells.size());
    for (auto& source : sources)
    {
        source.qbmm_shape_changes = {std::nullopt, Shape{}, std::nullopt};
        source.species_moles.assign(3, std::vector<double>(2));
        source.population_number_changes = {0, 0, 3}; // Count-only coarse phase remains independently owned.
    }
    const auto first = flow->preview(.01, sources, &controller);
    const auto snapshot = first.transport_snapshot();
    const auto statuses = first.qbmm_inversion_status();
    const auto endpoint = first.state();
    expect_identity(first, origin, R::InversionStatus::TwoNodes);
    const auto second = flow->preview(.02, {}, &controller);
    const auto rejects = [&](auto change)
    {
        auto bad = sources;
        for (auto& source : bad)
            change(source);
        EXPECT_THROW((void) flow->preview(.01, bad, &controller), std::exception);
        expect_state(flow->state(), origin);
    };
    rejects([](auto& source) { source.species_moles[1] = {.01, 0}; });
    rejects([](auto& source) { source.population_number_changes[1] = 1; });
    rejects([](auto& source) { source.qbmm_shape_changes[1] = Shape{1, 0}; });
    rejects([](auto& source) { source.qbmm_shape_changes[1] = Shape{0, -1}; });
    rejects([](auto& source) { source.qbmm_shape_changes.pop_back(); });
    rejects([](auto& source) { source.qbmm_shape_changes[0] = Shape{}; });
    for (auto& source : sources)
    {
        source.qbmm_shape_changes[1] = Shape{31, 37};
        source.population_number_changes[2] = 19;
    }
    EXPECT_EQ(first.transport_snapshot(), snapshot);
    EXPECT_EQ(first.qbmm_inversion_status(), statuses);
    EXPECT_EQ(first.integrated_sources().size(), origin.cells.size());
    for (const auto& source : first.integrated_sources())
    {
        EXPECT_EQ(source.qbmm_shape_changes.at(1), Shape{});
        EXPECT_EQ(source.population_number_changes.at(2), 3);
    }
    for (const auto& cell : first.state().cells)
        EXPECT_EQ(cell.phase_numbers.at(2), 20);
    expect_state(first.state(), endpoint);
    EXPECT_THROW(flow->commit(first), std::exception);
    const auto property_calls = *f.liquid_evaluations;
    flow->commit(first, &controller);
    EXPECT_EQ(*f.liquid_evaluations, property_calls);
    EXPECT_THROW(flow->commit(second, &controller), std::exception);
    expect_state(flow->state(), endpoint);
    auto changed_shape = endpoint;
    for (auto& cell : changed_shape.cells)
        cell.qbmm_shape[1] = Shape{196. / 5, 2744. / 25}; // Same number and material, now one size.
    flow->initialize(changed_shape, &controller);
    const auto calls_before_restore = *f.liquid_evaluations;
    flow->restore(checkpoint, &controller);
    EXPECT_EQ(*f.liquid_evaluations, calls_before_restore);
    expect_state(flow->state(), origin);
    EXPECT_THROW(flow->commit(first, &controller), std::exception);
    for (auto& source : sources)
    {
        source.qbmm_shape_changes[1] = Shape{};
        source.population_number_changes[2] = 3;
    }
    const auto replay = flow->preview(.01, sources, &controller);
    expect_state(replay.state(), endpoint);
    EXPECT_EQ(replay.transport_snapshot().cells, snapshot.cells);
    EXPECT_NE(replay.transport_snapshot().accepted_generation, snapshot.accepted_generation);
    EXPECT_EQ(replay.qbmm_inversion_status(), statuses);
}

TEST(LowMachQbmmStateTest, UnpairedProvidersMotionAndSlipRejectWhileCarrierTransportRemainsTransactional)
{
    Fixture f(2);
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto origin = flow->state();
    const auto retained = flow->preview(.01);
    int calls = 0;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "count-only-provider", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++calls;
            return Flow::SourceStageResult{std::vector<Flow::CellSources>(context.transport.cells.size()), {}};
        }});
    EXPECT_THROW((void) flow->preview_with_sources(.01, provider), std::exception);
    Flow::PlanarMotion motion(f.mesh);
    EXPECT_THROW((void) flow->preview_ale(.01, motion, 2.001), std::exception);
    EXPECT_THROW((void) flow->preview_ale_with_sources(.01, motion, 2.001, provider), std::exception);
    EXPECT_EQ(calls, 0);
    EXPECT_FALSE(motion.has_active_trial());
    auto slip = origin;
    for (auto& cell : slip.cells)
        cell.drift_velocity[1] = Flow::Vector{0, 0, .1};
    EXPECT_THROW(flow->initialize(slip), std::exception);
    expect_state(flow->state(), origin);
    flow->commit(retained);
    auto moving = origin;
    for (auto& cell : moving.cells)
        cell.velocity = Flow::Vector{0, 0, .125};
    flow->initialize(moving);
    double local_flux = 0, max_flux = 0;
    for (size_t i = 0; i < f.mesh->num_owned_faces(); ++i)
        local_flux = std::max(local_flux, std::abs(flow->volume_flux().local_value(static_cast<LO>(i))));
    Teuchos::reduceAll(*f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_flux, &max_flux);
    EXPECT_DOUBLE_EQ(max_flux, .125);
    const auto transported = flow->preview(.01);
    expect_state(flow->state(), moving);
    EXPECT_EQ(transported.state().time, .01);
    EXPECT_LE(transported.diagnostics().maximum_population_transport_residual, 1);
}

TEST(LowMachQbmmStateTest, ExistingReactionAndNumberCallbacksRequirePairedQbmmAllocation)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto origin = flow->state();
    const auto retained = flow->preview(.01);
    int chemistry_calls = 0, number_calls = 0;
    R::ReactingMixtureOptions options;
    options.mechanism = std::make_shared<const R::ReactionMechanism>(f.catalog,
        std::vector<R::ReactionDefinition>{
            {47, "2A_to_B", {{11, R::SpeciesPhase::Liquid, -2}, {29, R::SpeciesPhase::Liquid, 1}}, 0}},
        f.eos[0]->energy_convention());
    options.phases = {{R::SpeciesPhase::Liquid, {}, {}}, {R::SpeciesPhase::Dispersed, "fine", {}},
        {R::SpeciesPhase::Dispersed, "coarse", {}}};
    options.source_identity = "chemistry-count-only";
    options.source = [&](const auto&, double)
    {
        ++chemistry_calls;
        return R::ReactionSourceInput{};
    };
    options.population_events = std::make_shared<const R::MixturePopulationEventProvider>(
        R::MixturePopulationEventProvider{"count-only-tail", 1, f.catalog->fingerprint(), [&](const auto&)
            {
                ++number_calls;
                return std::vector<BubbleGasPhysics::PopulationNumberEvent>{};
            }});
    EXPECT_THROW((void) R::ReactingMixtureCoupler(flow, options), std::exception);
    expect_state(flow->state(), origin);
    EXPECT_EQ(chemistry_calls, 0);
    EXPECT_EQ(number_calls, 0);
    flow->commit(retained);
    expect_state(flow->state(), retained.state());
}

TEST(LowMachQbmmStateTest, UnrelatedMoleChangingChemistryAndUnboundDiffusionPreserveTheFamily)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    R::ReactingMixtureOptions options;
    options.mechanism = std::make_shared<const R::ReactionMechanism>(f.catalog,
        std::vector<R::ReactionDefinition>{
            {47, "2A_to_B", {{11, R::SpeciesPhase::Liquid, -2}, {29, R::SpeciesPhase::Liquid, 1}}, 0}},
        f.eos[0]->energy_convention());
    options.phases = {{R::SpeciesPhase::Liquid, {}, {}}, {R::SpeciesPhase::Dispersed, "fine", {}},
        {R::SpeciesPhase::Dispersed, "coarse", {}}};
    options.source_identity = "liquid-conversion";
    options.source = [](const auto&, double)
    {
        R::ReactionSourceInput input;
        input.integratedExtents = {{47, .25}};
        return input;
    };
    const auto diffusion = [&](size_t phase)
    {
        const auto& population = options.phases[phase].population;
        R::MixtureDiffusionLaw law;
        law.binding.transported = law.binding.counter = {R::SpeciesPhase::Dispersed, population};
        law.binding.counterSpecies = 29;
        law.binding.energyConvention = f.eos[0]->energy_convention();
        law.binding.identity = "same-carrier";
        law.binding.schemaFingerprint = f.catalog->fingerprint();
        law.primary_species = {11};
        law.molar_conductance = [](const auto&) { return std::vector<double>{1, 0}; };
        law.partial_molar_enthalpy = [](R::SpeciesId, const auto&, const R::MixtureFacePhaseSample& donor)
        { return 20 * (donor.temperature - 300) + 8.31446261815324 * donor.temperature; };
        return R::MixtureFaceTransportOptions{"diffusion", 1, f.catalog->fingerprint(), phase, law, std::nullopt};
    };
    options.face_transport = {diffusion(1)};
    EXPECT_THROW((void) R::ReactingMixtureCoupler(flow, options), std::exception);
    options.face_transport = {diffusion(2)};
    R::ReactingMixtureCoupler controller(flow, options);
    const auto trial = controller.preview(.01);
    EXPECT_EQ(trial.diagnostics().reaction_extents, (std::vector<double>{.25}));
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& before = flow->state().cells[c];
        const auto& after = trial.state().cells[c];
        EXPECT_EQ(after.phases[0].moles[0], before.phases[0].moles[0] - .5);
        EXPECT_EQ(after.phases[0].moles[1], .25);
        EXPECT_EQ(after.phases[1].moles, before.phases[1].moles);
        EXPECT_EQ(after.qbmm_shape, before.qbmm_shape);
        EXPECT_EQ(after.phase_numbers, before.phase_numbers);
    }
}

TEST(LowMachQbmmStateTest, PortableRestartRestoresFreshOwnersAndRejectsCorruptionOrIncompatibleSchemas)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    std::vector<Flow::CellSources> sources(f.mesh->num_owned_cells());
    for (auto& source : sources)
        source.population_number_changes = {0, 0, 3};
    flow->advance(.01, sources);
    const auto bytes = flow->serialize_qbmm_restart();
    const auto accepted = flow->state();
    Fixture fresh; // Independent EOS, family objects, mesh and solver.
    auto restored = fresh.solver();
    restored->restore_qbmm_restart(bytes);
    expect_state(restored->state(), accepted);
    EXPECT_EQ(restored->diagnostics().population_number_errors, flow->diagnostics().population_number_errors);
    EXPECT_EQ(restored->diagnostics().correctors, flow->diagnostics().correctors);
    for (size_t i = 0; i < f.mesh->num_owned_faces(); ++i)
        EXPECT_EQ(restored->volume_flux().local_value(static_cast<LO>(i)),
            flow->volume_flux().local_value(static_cast<LO>(i)));
    const auto next = flow->preview(.02);
    const auto retained = restored->preview(.02);
    expect_state(retained.state(), next.state());
    EXPECT_EQ(retained.qbmm_inversion_status(), next.qbmm_inversion_status());
    EXPECT_EQ(retained.transport_snapshot().cells, next.transport_snapshot().cells);
    const auto before_failure = restored->serialize_qbmm_restart();
    auto corrupt = bytes;
    EXPECT_FALSE(corrupt.empty());
    if (!corrupt.empty() && (f.mesh->owned_cell_map()->getComm()->getSize() == 1 || !f.mesh->num_owned_cells()))
        corrupt.pop_back(); // In MPI only the owner with zero cells supplies truncated bytes.
    EXPECT_THROW(restored->restore_qbmm_restart(corrupt), std::exception);
    EXPECT_EQ(restored->serialize_qbmm_restart(), before_failure);
    expect_state(restored->state(), accepted);
    restored->commit(retained);
    const auto stale = restored->preview(.01);
    restored->restore_qbmm_restart(bytes);
    EXPECT_THROW(restored->commit(stale), std::exception);
    expect_state(restored->state(), accepted);
    auto options = f.options;
    options.qbmm_families[1] = f.make_family(f.catalog, options.phase_roles[1], 2 * f.family->mass_scale_kg());
    Flow wrong_scale(f.mesh, f.eos, options);
    EXPECT_THROW(wrong_scale.restore_qbmm_restart(bytes), std::exception);
    EXPECT_FALSE(wrong_scale.initialized());
    options.qbmm_families[1] =
        std::make_shared<const R::QbmmPopulationSchema>(f.catalog, f.catalog->fingerprint(), "fine",
            options.phase_roles[1], f.family->mass_scale_kg(), std::vector<R::QbmmMassFraction>{{11, .25}, {29, .75}});
    Flow wrong_composition(f.mesh, f.eos, options);
    EXPECT_THROW(wrong_composition.restore_qbmm_restart(bytes), std::exception);
    EXPECT_FALSE(wrong_composition.initialized());
}

TEST(LowMachQbmmStateTest, EarlierRestartVersionsRestoreTheirCanonicalHistory)
{
    Fixture f;
    auto original = f.solver();
    original->initialize(f.initial());
    original->advance(.01);
    const auto current = original->serialize_qbmm_restart();
    ASSERT_GT(current.size(), 5 * sizeof(std::uint64_t));
    ASSERT_EQ(current[8], std::byte{4});
    for (unsigned version : {1u, 2u, 3u})
    {
        // Q3 and Q4 each appended one binary64 diagnostic before the two checksum words.
        auto bytes = current;
        bytes[8] = static_cast<std::byte>(version);
        constexpr auto word = sizeof(std::uint64_t);
        // V4 inserted controller presence + fingerprint after the 13 original header
        // words and appended the physical budget and shape diagnostics version.
        bytes.erase(bytes.begin() + 13 * word, bytes.begin() + 15 * word);
        constexpr unsigned v4_diagnostics_words = 4 + 2 + 1 + 1;
        bytes.resize(bytes.size() - (2 + v4_diagnostics_words + 3 - version) * word);
        std::uint64_t local = 14695981039346656037ULL, image = 0;
        for (auto b : bytes)
            local = (local ^ std::to_integer<std::uint64_t>(b)) * 1099511628211ULL;
        Teuchos::reduceAll(*f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &image);
        for (std::uint64_t word : {local, image})
            for (int i = 0; i < 8; ++i)
                bytes.push_back(static_cast<std::byte>((word >> (8 * i)) & 0xff));

        auto restored = f.solver();
        restored->restore_qbmm_restart(bytes);
        expect_state(restored->state(), original->state());
        EXPECT_EQ(restored->diagnostics().maximum_shape_transport_residual, 0);
        EXPECT_EQ(restored->diagnostics().maximum_shape_source_residual,
            version == 1 ? 0 : original->diagnostics().maximum_shape_source_residual);
        EXPECT_EQ(restored->diagnostics().correctors, original->diagnostics().correctors);
        const auto report = restored->qbmm_report();
        ASSERT_TRUE(report.convergence);
        EXPECT_EQ(report.convergence->maximum_shape_source_residual.has_value(), version >= 2);
        EXPECT_EQ(report.convergence->maximum_shape_transport_residual.has_value(), version >= 3);
        const auto rearchived = f.solver();
        rearchived->restore_qbmm_restart(restored->serialize_qbmm_restart());
        const auto reread = rearchived->qbmm_report();
        ASSERT_TRUE(reread.convergence);
        EXPECT_EQ(reread.convergence->maximum_shape_source_residual, report.convergence->maximum_shape_source_residual);
        EXPECT_EQ(
            reread.convergence->maximum_shape_transport_residual, report.convergence->maximum_shape_transport_residual);
    }
}

TEST(LowMachQbmmStateTest, DeferredModelRequestsRejectCollectivelyBeforeAdvance)
{
    Fixture f;
    using Model = R::QbmmModelOptions;
    struct Request
    {
        const char* name;
        void (*select)(Model&);
    };
    const Request requests[]{
        {"BDF2", [](Model& m) { m.transport.time = FVM::ScalarTimeScheme::BDF2; }},
        {"linear upwind", [](Model& m) { m.transport.convection = FVM::ScalarConvectionScheme::BoundedLinearUpwind; }},
        {"steady", [](Model& m) { m.transport.mode = FVM::AssemblyMode::Steady; }},
        {"phase slip", [](Model& m) { m.velocity = Model::Velocity::SuppliedPhaseSlip; }},
        {"node slip", [](Model& m) { m.velocity = Model::Velocity::NodeDependentSlip; }},
        {"independent momentum", [](Model& m) { m.velocity = Model::Velocity::IndependentPhaseMomentum; }},
        {"variable composition", [](Model& m) { m.variable_composition = true; }},
        {"merging unlike families", [](Model& m) { m.cross_family_coalescence = true; }},
        {"pool conversion", [](Model& m) { m.bulk_gas_conversion = true; }},
        {"capillarity", [](Model& m) { m.capillarity = true; }},
        {"node temperatures", [](Model& m) { m.independent_node_temperatures = true; }},
    };
    const auto comm = f.mesh->owned_cell_map()->getComm();
    for (const auto& request : requests)
    {
        SCOPED_TRACE(request.name);
        auto options = f.options;
        // At MPI2/4 only one empty owner requests the unsupported model.
        if (comm->getRank() == comm->getSize() - 1)
            request.select(options.qbmm_model);
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    }
    auto supported = f.solver();
    EXPECT_NO_THROW(supported->initialize(f.initial()));
}

TEST(LowMachQbmmStateTest, MaterialSlotsCannotConvertTheFamilyToOrFromAnUnresolvedGasPool)
{
    Fixture f(1, true);
    auto options = f.options;
    options.phase_roles[2] = {Flow::PhaseRole::BulkGas, "background"};
    auto flow = std::make_shared<Flow>(f.mesh, f.eos, options);
    auto input = f.initial();
    for (auto& cell : input.cells)
        cell.phase_numbers[2].reset();
    flow->initialize(input);
    const auto retained = flow->preview(.01);
    const auto origin = flow->serialize_qbmm_restart();
    R::ReactingMixtureOptions sources;
    sources.mechanism = std::make_shared<const R::ReactionMechanism>(
        f.catalog, std::vector<R::ReactionDefinition>{}, f.eos[0]->energy_convention());
    sources.phases = {{R::SpeciesPhase::Liquid, {}, {}}, {R::SpeciesPhase::Dispersed, "fine", {}},
        {R::SpeciesPhase::Resolved, {}, {}}};
    for (const bool into_family : {false, true})
    {
        auto transfers = std::make_shared<R::InterphaseConfiguration>();
        transfers->metadata = {"pool exchange", 1, f.catalog->fingerprint()};
        transfers->phase_enthalpy.resize(3);
        const size_t donor = into_family ? 2 : 1, receiver = into_family ? 1 : 2;
        transfers->stages = {{"pool exchange", R::InterphaseStageKind::Material, {{11, donor, receiver}}, {}}};
        int calls = 0;
        sources.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{{"constructive", 1, f.catalog->fingerprint()}, transfers, [&](const auto&)
                {
                    ++calls;
                    return std::vector<R::QbmmEventOperation>{};
                }});
        try
        {
            (void) R::ReactingMixtureCoupler(flow, sources);
            ADD_FAILURE() << "A bulk-gas exchange slot was admitted.";
        }
        catch (const std::exception& e)
        {
            EXPECT_NE(std::string(e.what()).find("bulk-gas conversion"), std::string::npos);
        }
        EXPECT_EQ(calls, 0);
        EXPECT_EQ(flow->serialize_qbmm_restart(), origin);
    }
    flow->commit(retained);
}

TEST(LowMachQbmmStateTest, LinearSolverOptionsMustAgreeOnEveryOwner)
{
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const bool last = comm->getRank() == comm->getSize() - 1;
    for (bool change_tolerance : {false, true})
    {
        auto options = f.options;
        if (last)
        {
            // A single-rank run checks the admissible range. With MPI the last rank
            // owns no cell and supplies a valid value that differs from its peers.
            if (change_tolerance)
                options.linear.tolerance = comm->getSize() == 1 ? 0. : options.linear.tolerance * .5;
            else
                options.linear.max_iterations = comm->getSize() == 1 ? 0 : options.linear.max_iterations + 1;
        }
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    }
    auto options = f.options;
    if (last)
        options.linear.preconditioner =
            comm->getSize() == 1 ? static_cast<LinearPreconditioner>(-1) : LinearPreconditioner::Jacobi;
    EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
}

TEST(LowMachQbmmStateTest, OneCellFailuresReachAllRanksAndPreserveAnEarlierTrial)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto origin = flow->state();
    const auto retained = flow->preview(.01);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    int empty = f.mesh->num_owned_cells() == 0, empty_total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_total);
    EXPECT_EQ(empty_total, comm->getSize() - 1);
    if (comm->getSize() > 1)
    {
        auto options = f.options;
        if (empty)
            options.qbmm_families[1] = f.make_family(f.catalog, options.phase_roles[1], 2 * f.family->mass_scale_kg());
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
        options = f.options;
        if (empty)
            options.qbmm_families.clear();
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    }
    auto bad = origin;
    if (empty || comm->getSize() == 1)
        bad.qbmm_families[1].reset();
    EXPECT_THROW(flow->initialize(bad), std::exception);
    std::vector<Flow::CellSources> sources(origin.cells.size());
    for (auto& source : sources)
        source.qbmm_shape_changes = {std::nullopt, Shape{0, 1}, std::nullopt};
    EXPECT_THROW((void) flow->preview(.01, sources), std::exception);
    expect_state(flow->state(), origin);
    if (comm->getSize() > 1)
    {
        const auto longer = flow->preview(.02);
        EXPECT_THROW(flow->commit(empty ? longer : retained), std::exception);
        expect_state(flow->state(), origin);
        for (auto& source : sources)
        {
            source.qbmm_shape_changes.clear();
            source.population_number_changes = {0, 0, 2}; // Replay differs despite identical base and dt.
        }
        const auto count_packet = flow->preview(.01, sources);
        EXPECT_THROW(flow->commit(empty ? retained : count_packet), std::exception);
        expect_state(flow->state(), origin);
    }
    const auto retry = flow->preview(.01);
    expect_state(retry.state(), retained.state());
    flow->commit(retained);
    expect_state(flow->state(), retry.state());

    if (comm->getSize() > 1)
    {
        const auto first = flow->checkpoint();
        flow->initialize(retry.state()); // Same clock, schema and physical stocks, a later owner generation.
        const auto later = flow->checkpoint();
        const auto next = flow->preview(.01);
        EXPECT_THROW(flow->restore(empty ? first : later), std::exception);
        expect_state(flow->state(), retry.state());
        flow->commit(next);
        expect_state(flow->state(), next.state());
    }
}
} // namespace
