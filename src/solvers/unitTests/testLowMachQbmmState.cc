/** Native QBMM ownership, pre-transport capability gates, and transactional replay. */
#include "QbmmRestartTestSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <bit>
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

    static Family make_family(const std::shared_ptr<const R::SpeciesCatalog>& catalog,
        const Flow::PhaseRoleDescriptor& role, double scale, R::QbmmClosureVersions versions = {})
    {
        return std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), role.identity, role,
            scale, std::vector<R::QbmmMassFraction>{{11, .5}, {29, .5}}, std::move(versions));
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

void expect_state(const Flow::State& state, const Flow::State& expected);

TEST(LowMachQbmmStateTest, LinearStageTwoRejectsOneOwnerAndPreservesEarlierTrial)
{
    Fixture f(3);
    f.options.qbmm_model.family_transport = R::QbmmModelOptions::FamilyTransport::RealizableLinearSSPRK2;
    f.options.qbmm_model.maximum_linear_substeps = 1;
    constexpr double base_pressure = 100000., marker = 1. / 64., marked_pressure = base_pressure + marker;
    const auto gas = f.eos[1];
    bool armed = false, injected = false;
    f.eos[1] = std::make_shared<const R::MixtureThermodynamics>(
        f.catalog, gas->bounds(), gas->kind(), gas->enthalpy_pressure_dependence(),
        [&](double temperature, double pressure, std::span<const double> fractions)
        {
            if (armed && pressure == marked_pressure)
            {
                injected = true;
                throw std::invalid_argument("Test stage-two donor EOS failure.");
            }
            return gas->evaluate(temperature, pressure, fractions);
        },
        gas->energy_convention());
    auto initial = f.initial(true);
    const auto occupied = f.initial();
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        auto& cell = initial.cells[c];
        const double z = f.mesh->cell_centroid(static_cast<LO>(c)).z;
        if (z < 1.)
            cell = occupied.cells[c];
        if (z > 1. && z < 2.)
        {
            // Stage one has only the coarse gas here. Hold both occupied
            // volumes fixed under the unique local pressure marker.
            cell.mechanical_pressure = marker;
            cell.phases[2].moles[1] *= marked_pressure / base_pressure;
            const std::array fractions{1., 0.};
            const auto liquid = f.eos[0]->evaluate(300., marked_pressure, fractions);
            auto& phase = cell.phases[0];
            phase.moles[0] *= liquid.density / 1000.;
            phase.internal_energy = phase.moles[0] * .02 * liquid.internal_energy;
        }
        cell.velocity.component(2) = .04;
    }
    auto flow = f.solver();
    flow->initialize(std::move(initial));
    const auto accepted = flow->state();
    std::vector<double> accepted_flux(f.mesh->num_owned_faces());
    for (size_t k = 0; k < accepted_flux.size(); ++k)
        accepted_flux[k] = flow->volume_flux().local_value(static_cast<LO>(k));
    const auto earlier = flow->preview(.001);

    const auto rejection = [&](double dt)
    {
        try
        {
            (void) flow->preview(dt);
        }
        catch (const std::exception& error)
        {
            return std::string(error.what());
        }
        return std::string{};
    };
    // Two interior faces carry .04 m3/s. A 15 s request needs more
    // than one substep under the .5 donor Courant cap in each 1 m3 cell.
    const auto cap_failure = rejection(15.);
    EXPECT_NE(cap_failure.find("subcycle cap"), std::string::npos) << cap_failure;
    expect_state(flow->state(), accepted);
    armed = true;
    const auto failure = rejection(.001);
    EXPECT_NE(failure.find("QBMM SSPRK2 stage 2 donor traces"), std::string::npos) << failure;
    const int local_injectors = injected;
    int injectors = 0;
    Teuchos::reduceAll(*f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_injectors, &injectors);
    EXPECT_EQ(injectors, 1);
    expect_state(flow->state(), accepted);
    for (size_t k = 0; k < accepted_flux.size(); ++k)
        EXPECT_EQ(flow->volume_flux().local_value(static_cast<LO>(k)), accepted_flux[k]);
    EXPECT_NO_THROW(flow->commit(earlier));
    expect_state(flow->state(), earlier.state());
}

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

using Testing::QbmmRestart::append_restart_word;

void append_restart_checksums(const Fixture& fixture, std::vector<std::byte>& body)
{
    Testing::QbmmRestart::append_restart_checksums(*fixture.mesh->owned_cell_map()->getComm(), body);
}

std::vector<std::byte> restart_with_fourier_uncertainty(
    const Fixture& fixture, std::vector<std::byte> bytes, std::optional<double> uncertainty, bool replacing = false)
{
    bytes.resize(bytes.size() - (replacing ? 3 : 2) * sizeof(std::uint64_t));
    if (uncertainty)
        append_restart_word(bytes, std::bit_cast<std::uint64_t>(*uncertainty));
    append_restart_checksums(fixture, bytes);
    return bytes;
}

std::vector<std::byte> earlier_restart(
    const Fixture& fixture, std::vector<std::byte> bytes, unsigned version, bool has_fourier_uncertainty = false)
{
    constexpr auto word = sizeof(std::uint64_t);
    if (version < 1 || version > 4 || bytes.size() < 19 * word || bytes[8] != std::byte{5})
        throw std::invalid_argument("Test conversion requires a version 5 QBMM archive and a version in 1-4.");
    bytes.resize(bytes.size() - (has_fourier_uncertainty ? 3 : 2) * word);
    bytes[8] = static_cast<std::byte>(version);
    // V5 inserted model + retained legacy restriction after V4's controller stamp.
    bytes.erase(bytes.begin() + 15 * word, bytes.begin() + 17 * word);
    if (version < 4)
    {
        bytes.erase(bytes.begin() + 13 * word, bytes.begin() + 15 * word);
        // This fixture's advanced native image has two species and one element:
        // presence, basis, mass, species-size + 2 values, element, shape-version.
        constexpr unsigned balance_words = 8;
        bytes.resize(bytes.size() - (balance_words + 3 - version) * word);
    }
    append_restart_checksums(fixture, bytes);
    return bytes;
}

R::ReactingMixtureOptions restart_controller_options(const Fixture& fixture, int& calls)
{
    R::ReactingMixtureOptions options;
    options.mechanism = std::make_shared<const R::ReactionMechanism>(
        fixture.catalog, std::vector<R::ReactionDefinition>{}, fixture.eos[0]->energy_convention());
    options.phases = {{R::SpeciesPhase::Liquid, {}, {}}, {R::SpeciesPhase::Dispersed, "fine", {}},
        {R::SpeciesPhase::Dispersed, "coarse", {}}};
    options.source_identity = "restart-source";
    options.source = [&calls](const auto&, double)
    {
        ++calls;
        return R::ReactionSourceInput{};
    };
    return options;
}

R::ReactingMixtureOptions fourier_restart_options(const Fixture& fixture, int& calls)
{
    auto options = restart_controller_options(fixture, calls);
    const auto fingerprint = fixture.catalog->fingerprint();
    options.face_transport = {{"restart Fourier face stage", 1, fingerprint, 0, std::nullopt,
        R::MixtureFourierLaw{"restart liquid contact", 1, fingerprint, [](const auto&) { return 1.; }}}};
    return options;
}

Flow::State fourier_restart_initial(const Fixture& fixture)
{
    auto state = fixture.initial();
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const double temperature = fixture.mesh->cell_centroid(cell).z < 1. ? 300. : 301.;
        const double pressure = state.thermodynamic_pressure;
        const double gas_volume = 2.5 * 8.31446261815324 * temperature / pressure;
        auto& phases = state.cells[c].phases;
        phases[0].moles[0] = 1000 * (fixture.mesh->cell_volume(cell) - gas_volume) / .02;
        constexpr std::array molar_masses{.02, .04};
        for (auto& phase : phases)
        {
            const double moles = phase.moles[0] + phase.moles[1];
            const std::array fractions{phase.moles[0] / moles, phase.moles[1] / moles};
            const double mass = phase.moles[0] * molar_masses[0] + phase.moles[1] * molar_masses[1];
            phase.temperature = temperature;
            phase.internal_energy =
                mass * phase.thermodynamics->evaluate(temperature, pressure, fractions).internal_energy;
        }
    }
    return state;
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

TEST(LowMachQbmmStateTest, UnpairedProvidersAndUndeclaredSlipRejectWithoutInvalidatingFixedTrial)
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
    EXPECT_NO_THROW((void) flow->preview_ale(.01, motion, 2.001));
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
    Fixture f(2);
    auto original = f.solver();
    original->initialize(fourier_restart_initial(f));
    int calls = 0;
    {
        R::ReactingMixtureCoupler controller(original, fourier_restart_options(f, calls));
        controller.advance(.01);
    }
    ASSERT_GT(original->diagnostics().maximum_fourier_gross_comparison_uncertainty_j.value_or(0.), 0.);
    const auto current = original->serialize_qbmm_restart();
    ASSERT_GT(current.size(), 5 * sizeof(std::uint64_t));
    ASSERT_EQ(current[8], std::byte{5});
    for (unsigned version : {1u, 2u, 3u, 4u})
    {
        const auto bytes = earlier_restart(f, current, version, true);

        auto restored = f.solver();
        restored->restore_qbmm_restart(bytes);
        expect_state(restored->state(), original->state());
        EXPECT_EQ(restored->diagnostics().maximum_shape_transport_residual,
            version < 3 ? 0 : original->diagnostics().maximum_shape_transport_residual);
        EXPECT_EQ(restored->diagnostics().maximum_shape_source_residual,
            version == 1 ? 0 : original->diagnostics().maximum_shape_source_residual);
        EXPECT_EQ(restored->diagnostics().correctors, original->diagnostics().correctors);
        const auto report = restored->qbmm_report();
        ASSERT_TRUE(report.convergence);
        EXPECT_FALSE(restored->diagnostics().maximum_fourier_gross_comparison_uncertainty_j);
        EXPECT_FALSE(report.convergence->maximum_fourier_gross_comparison_uncertainty_j);
        EXPECT_EQ(report.convergence->maximum_shape_source_residual.has_value(), version >= 2);
        EXPECT_EQ(report.convergence->maximum_shape_transport_residual.has_value(), version >= 3);
        const auto rearchived = f.solver();
        rearchived->restore_qbmm_restart(restored->serialize_qbmm_restart());
        const auto reread = rearchived->qbmm_report();
        ASSERT_TRUE(reread.convergence);
        EXPECT_FALSE(rearchived->diagnostics().maximum_fourier_gross_comparison_uncertainty_j);
        EXPECT_FALSE(reread.convergence->maximum_fourier_gross_comparison_uncertainty_j);
        EXPECT_EQ(reread.convergence->maximum_shape_source_residual, report.convergence->maximum_shape_source_residual);
        EXPECT_EQ(
            reread.convergence->maximum_shape_transport_residual, report.convergence->maximum_shape_transport_residual);
    }
}

TEST(LowMachQbmmStateTest, PortableFourierUncertaintyRoundTripsAnAcceptedContactStep)
{
    Fixture f(2);
    auto original = f.solver();
    original->initialize(fourier_restart_initial(f));
    int calls = 0;
    const auto options = fourier_restart_options(f, calls);
    {
        R::ReactingMixtureCoupler controller(original, options);
        controller.advance(.01);
    }
    const double uncertainty = original->diagnostics().maximum_fourier_gross_comparison_uncertainty_j.value_or(0.);
    ASSERT_GT(uncertainty, 0.);
    // Both temperatures stay in the same binary64 binade. The comparison
    // uses one operand-spacing bound per evaluation at dt * conductance = .01 J/K.
    EXPECT_DOUBLE_EQ(uncertainty, .02 * (std::nextafter(300., 301.) - 300.));
    ASSERT_GE(original->diagnostics().source_evaluations, 2);
    const auto archive = original->serialize_qbmm_restart();
    const auto old_v5 = restart_with_fourier_uncertainty(f, archive, std::nullopt, true);
    auto unknown = f.solver();
    unknown->restore_qbmm_restart(old_v5);
    EXPECT_FALSE(unknown->diagnostics().maximum_fourier_gross_comparison_uncertainty_j);
    const auto unknown_report = unknown->qbmm_report();
    ASSERT_TRUE(unknown_report.convergence);
    EXPECT_FALSE(unknown_report.convergence->maximum_fourier_gross_comparison_uncertainty_j);
    const auto reencoded_unknown = unknown->serialize_qbmm_restart();
    EXPECT_EQ(reencoded_unknown.size(), old_v5.size());

    auto restored = f.solver();
    const int before_restore = calls;
    restored->restore_qbmm_restart(archive);
    EXPECT_EQ(calls, before_restore);
    expect_state(restored->state(), unknown->state());
    EXPECT_EQ(restored->diagnostics().maximum_fourier_gross_comparison_uncertainty_j, uncertainty);
    const auto report = restored->qbmm_report();
    ASSERT_TRUE(report.convergence);
    EXPECT_EQ(report.convergence->maximum_fourier_gross_comparison_uncertainty_j, uncertainty);
    EXPECT_EQ(restored->serialize_qbmm_restart(), restart_with_fourier_uncertainty(f, reencoded_unknown, uncertainty));

    R::ReactingMixtureCoupler controller(restored, options);
    const auto checkpoint = controller.checkpoint();
    const auto trial = controller.preview(.01);
    const auto trial_report = controller.qbmm_report(trial);
    ASSERT_TRUE(trial_report.convergence);
    EXPECT_EQ(trial_report.convergence->maximum_fourier_gross_comparison_uncertainty_j,
        trial.diagnostics().flow.maximum_fourier_gross_comparison_uncertainty_j);
    controller.commit(trial);
    EXPECT_EQ(restored->diagnostics().maximum_fourier_gross_comparison_uncertainty_j,
        trial_report.convergence->maximum_fourier_gross_comparison_uncertainty_j);
    controller.restore(checkpoint);
    EXPECT_EQ(restored->diagnostics().maximum_fourier_gross_comparison_uncertainty_j, uncertainty);
}

TEST(LowMachQbmmStateTest, PortableFourierUncertaintyRetainsTheKnownZeroOfAnUnchangedContact)
{
    Fixture f(2);
    auto original = f.solver();
    original->initialize(f.initial());
    int calls = 0;
    {
        R::ReactingMixtureCoupler controller(original, fourier_restart_options(f, calls));
        controller.advance(.01);
    }
    ASSERT_GE(original->diagnostics().source_evaluations, 2);
    ASSERT_EQ(original->diagnostics().maximum_fourier_gross_comparison_uncertainty_j, 0.);
    auto restored = f.solver();
    restored->restore_qbmm_restart(original->serialize_qbmm_restart());
    EXPECT_EQ(restored->diagnostics().maximum_fourier_gross_comparison_uncertainty_j, 0.);
    const auto report = restored->qbmm_report();
    ASSERT_TRUE(report.convergence);
    EXPECT_EQ(report.convergence->maximum_fourier_gross_comparison_uncertainty_j, 0.);
}

TEST(LowMachQbmmStateTest, PortableFourierUncertaintyRejectsInvalidOrLegacyTrailingWords)
{
    Fixture f(2);
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto initial = flow->serialize_qbmm_restart();
    EXPECT_THROW(flow->restore_qbmm_restart(restart_with_fourier_uncertainty(f, initial, 0.)), std::exception);
    EXPECT_EQ(flow->serialize_qbmm_restart(), initial);
    flow->advance(.01);
    const auto accepted = flow->serialize_qbmm_restart();
    const auto retained = flow->preview(.01);
    EXPECT_THROW(flow->restore_qbmm_restart(restart_with_fourier_uncertainty(f, accepted, .125)), std::exception);
    EXPECT_EQ(flow->serialize_qbmm_restart(), accepted);

    auto fourier = f.solver();
    fourier->initialize(fourier_restart_initial(f));
    int calls = 0;
    {
        R::ReactingMixtureCoupler controller(fourier, fourier_restart_options(f, calls));
        controller.advance(.01);
    }
    ASSERT_GT(fourier->diagnostics().maximum_fourier_gross_comparison_uncertainty_j.value_or(0.), 0.);
    const auto valid = fourier->serialize_qbmm_restart();
    const auto comm = f.mesh->owned_cell_map()->getComm();
    for (const double invalid :
        {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        const auto corrupt = restart_with_fourier_uncertainty(f, valid, comm->getRank() == 0 ? invalid : .125, true);
        EXPECT_THROW(flow->restore_qbmm_restart(corrupt), std::exception);
        EXPECT_EQ(flow->serialize_qbmm_restart(), accepted);
    }
    if (comm->getSize() > 1)
    {
        const auto inconsistent = restart_with_fourier_uncertainty(f, valid, comm->getRank() == 0 ? 0. : .125, true);
        EXPECT_THROW(flow->restore_qbmm_restart(inconsistent), std::exception);
        EXPECT_EQ(flow->serialize_qbmm_restart(), accepted);
    }
    const auto legacy = earlier_restart(f, accepted, 4);
    EXPECT_THROW(flow->restore_qbmm_restart(restart_with_fourier_uncertainty(f, legacy, 0.)), std::exception);
    EXPECT_EQ(flow->serialize_qbmm_restart(), accepted);
    flow->commit(retained);
}

TEST(LowMachQbmmStateTest, RestartVelocityAndVersionMismatchPreserveAcceptedStateAndTrials)
{
    Fixture f;
    auto shared = f.solver();
    shared->initialize(f.initial());
    const auto shared_bytes = shared->serialize_qbmm_restart();
    auto slip_options = f.options;
    slip_options.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SuppliedPhaseSlip;
    Flow slip(f.mesh, f.eos, slip_options);
    slip.initialize(f.initial()); // Explicit slip mode differs even when every drift is zero.
    const auto zero_slip_bytes = slip.serialize_qbmm_restart();
    EXPECT_THROW(slip.restore_qbmm_restart(shared_bytes), std::exception);
    EXPECT_EQ(slip.serialize_qbmm_restart(), zero_slip_bytes);
    EXPECT_THROW(shared->restore_qbmm_restart(zero_slip_bytes), std::exception);
    EXPECT_EQ(shared->serialize_qbmm_restart(), shared_bytes);
    const auto old = earlier_restart(f, shared_bytes, 4);
    EXPECT_THROW(slip.restore_qbmm_restart(old), std::exception);
    if (f.mesh->owned_cell_map()->getComm()->getSize() > 1)
    {
        // Recompute a valid collective checksum for shards with different archive
        // versions. Rejection must come from uniform provenance, before publishing.
        auto mixed = f.mesh->owned_cell_map()->getComm()->getRank() == 0 ? old : shared_bytes;
        mixed.resize(mixed.size() - 2 * sizeof(std::uint64_t));
        append_restart_checksums(f, mixed);
        EXPECT_THROW(shared->restore_qbmm_restart(mixed), std::exception);
        EXPECT_EQ(shared->serialize_qbmm_restart(), shared_bytes);
    }

    auto drifting = f.initial();
    for (auto& cell : drifting.cells)
    {
        cell.drift_velocity.resize(f.eos.size());
        cell.drift_velocity[1] = Flow::Vector{.125, 0, -.25};
    }
    slip.initialize(drifting);
    Flow resumed(f.mesh, f.eos, slip_options);
    resumed.restore_qbmm_restart(slip.serialize_qbmm_restart());
    expect_state(resumed.state(), slip.state());

    const auto retained = shared->preview(.01);
    for (const unsigned version : {0u, 6u})
    {
        auto invalid = shared_bytes;
        invalid.resize(invalid.size() - 2 * sizeof(std::uint64_t));
        invalid[8] = static_cast<std::byte>(version);
        append_restart_checksums(f, invalid);
        EXPECT_THROW(shared->restore_qbmm_restart(invalid), std::exception);
        EXPECT_EQ(shared->serialize_qbmm_restart(), shared_bytes);
    }
    auto invalid_model = shared_bytes;
    invalid_model.resize(invalid_model.size() - 2 * sizeof(std::uint64_t));
    invalid_model[15 * sizeof(std::uint64_t)] = std::byte{2};
    append_restart_checksums(f, invalid_model);
    EXPECT_THROW(shared->restore_qbmm_restart(invalid_model), std::exception);
    EXPECT_EQ(shared->serialize_qbmm_restart(), shared_bytes);
    auto invalid_restriction = shared_bytes;
    invalid_restriction.resize(invalid_restriction.size() - 2 * sizeof(std::uint64_t));
    invalid_restriction[16 * sizeof(std::uint64_t)] = std::byte{2};
    append_restart_checksums(f, invalid_restriction);
    EXPECT_THROW(shared->restore_qbmm_restart(invalid_restriction), std::exception);
    EXPECT_EQ(shared->serialize_qbmm_restart(), shared_bytes);
    shared->commit(retained);
}

TEST(LowMachQbmmStateTest, LegacyDefaultOnlyRestrictionSurvivesRestoreAdvanceAndReencoding)
{
    Fixture f;
    auto original = f.solver();
    original->initialize(f.initial());
    original->advance(.01);
    const auto current = original->serialize_qbmm_restart();
    int calls = 0;
    auto forced = restart_controller_options(f, calls);
    forced.body_acceleration = Flow::Vector{}; // Engaged zero is an explicit force configuration too.
    auto substepped = restart_controller_options(f, calls);
    const auto& closures = f.family->closure_versions();
    substepped.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{
        {closures.source_identity, closures.source, f.catalog->fingerprint()}, nullptr,
        [](const R::MixtureQbmmContext&) { return std::vector<R::QbmmEventOperation>{}; }});
    substepped.qbmm_source_substep_relative_tolerance = .01;
    {
        auto modern = f.solver();
        modern->initialize(f.initial());
        EXPECT_NO_THROW((void) R::ReactingMixtureCoupler(modern, substepped));
    }
    for (const unsigned version : {1u, 2u, 3u, 4u})
    {
        auto restored = f.solver();
        restored->restore_qbmm_restart(earlier_restart(f, current, version));
        auto before = restored->serialize_qbmm_restart();
        EXPECT_THROW((void) R::ReactingMixtureCoupler(restored, forced), std::exception);
        try
        {
            (void) R::ReactingMixtureCoupler(restored, substepped);
            ADD_FAILURE() << "Legacy default-only provenance admitted QBMM substepping.";
        }
        catch (const std::exception& error)
        {
            EXPECT_NE(std::string_view(error.what()).find("Legacy QBMM restart requires default"), std::string::npos);
        }
        EXPECT_EQ(restored->serialize_qbmm_restart(), before);
        restored->advance(.01);
        const auto saved = restored->checkpoint();
        restored->advance(.01);
        const auto eos_calls = *f.liquid_evaluations;
        restored->restore(saved);
        EXPECT_EQ(*f.liquid_evaluations, eos_calls);
        before = restored->serialize_qbmm_restart();
        EXPECT_THROW((void) R::ReactingMixtureCoupler(restored, forced), std::exception);
        EXPECT_EQ(restored->serialize_qbmm_restart(), before);

        auto reencoded = f.solver();
        reencoded->restore_qbmm_restart(before);
        const auto reencoded_before = reencoded->serialize_qbmm_restart();
        EXPECT_THROW((void) R::ReactingMixtureCoupler(reencoded, forced), std::exception);
        EXPECT_EQ(reencoded->serialize_qbmm_restart(), reencoded_before);
        R::ReactingMixtureCoupler controller(reencoded, restart_controller_options(f, calls));
        const auto coupled_saved = controller.checkpoint();
        controller.restore(coupled_saved);
        EXPECT_EQ(calls, 0);
    }
}

TEST(LowMachQbmmStateTest, FreshInitializationClearsLegacyAndControllerProvenanceOnlyOnSuccess)
{
    Fixture f;
    auto original = f.solver();
    original->initialize(f.initial());
    int calls = 0;
    std::vector<std::byte> legacy;
    {
        R::ReactingMixtureCoupler controller(original, restart_controller_options(f, calls));
        legacy = earlier_restart(f, original->serialize_qbmm_restart(), 4);
    }
    auto restored = f.solver();
    restored->restore_qbmm_restart(legacy);
    const auto before = restored->serialize_qbmm_restart();
    EXPECT_TRUE(restored->qbmm_report().provenance.controller_configuration_signature);
    auto forced = restart_controller_options(f, calls);
    forced.body_acceleration = Flow::Vector{};
    auto invalid = f.initial();
    for (auto& cell : invalid.cells)
        cell.phases[0].moles[0] *= .9;
    EXPECT_THROW(restored->initialize(invalid), std::exception);
    EXPECT_EQ(restored->serialize_qbmm_restart(), before);
    EXPECT_THROW((void) restored->preview(.01), std::exception);
    EXPECT_THROW((void) R::ReactingMixtureCoupler(restored, forced), std::exception);

    restored->initialize(f.initial());
    const auto report = restored->qbmm_report();
    EXPECT_EQ(report.provenance.origin, Flow::QbmmStateOrigin::Initialized);
    EXPECT_FALSE(report.provenance.controller_configuration_signature);
    EXPECT_NO_THROW((void) restored->preview(.01));
    EXPECT_NO_THROW((void) R::ReactingMixtureCoupler(restored, forced));
    EXPECT_EQ(calls, 0);
}

TEST(LowMachQbmmStateTest, SignedRestartRequiresExactSourceAndForceMetadataWithoutCallbacks)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    int calls = 0;
    const auto options = restart_controller_options(f, calls);
    std::vector<std::byte> signed_bytes;
    {
        R::ReactingMixtureCoupler controller(flow, options);
        signed_bytes = flow->serialize_qbmm_restart();
    }
    for (const bool legacy : {false, true})
    {
        const auto bytes = legacy ? earlier_restart(f, signed_bytes, 4) : signed_bytes;
        auto restored = f.solver();
        restored->restore_qbmm_restart(bytes);
        const auto before = restored->serialize_qbmm_restart();
        auto changed = options;
        ++changed.source_revision;
        EXPECT_THROW((void) R::ReactingMixtureCoupler(restored, changed), std::exception);
        changed = options;
        changed.body_acceleration = Flow::Vector{0, 0, -1};
        EXPECT_THROW((void) R::ReactingMixtureCoupler(restored, changed), std::exception);
        EXPECT_EQ(restored->serialize_qbmm_restart(), before);
        EXPECT_EQ(calls, 0);
        R::ReactingMixtureCoupler matching(restored, options);
        const auto checkpoint = matching.checkpoint();
        matching.restore(checkpoint);
        EXPECT_EQ(calls, 0);
    }

    auto forced_options = options;
    forced_options.body_acceleration = Flow::Vector{0, 0, -1};
    std::vector<std::byte> forced_bytes;
    {
        R::ReactingMixtureCoupler controller(flow, forced_options);
        forced_bytes = flow->serialize_qbmm_restart();
    }
    auto restored = f.solver();
    restored->restore_qbmm_restart(forced_bytes);
    EXPECT_THROW((void) R::ReactingMixtureCoupler(restored, options), std::exception);
    R::ReactingMixtureCoupler matching(restored, forced_options);
    EXPECT_EQ(calls, 0);
}

TEST(LowMachQbmmStateTest, NativeCheckpointRejectsForeignSourceBindingWithoutInvalidatingTrial)
{
    Fixture f;
    auto flow = f.solver();
    flow->initialize(f.initial());
    int calls = 0;
    const auto signed_saved = [&]
    {
        R::ReactingMixtureCoupler controller(flow, restart_controller_options(f, calls));
        return flow->checkpoint();
    }();
    auto lease = flow->acquire_controller();
    const auto retained = flow->preview(.01, {}, &lease);
    const auto before = flow->serialize_qbmm_restart();
    const auto eos_calls = *f.liquid_evaluations;
    EXPECT_THROW(flow->restore(signed_saved, &lease), std::exception);
    EXPECT_EQ(flow->serialize_qbmm_restart(), before);
    EXPECT_EQ(*f.liquid_evaluations, eos_calls);
    EXPECT_EQ(calls, 0);
    flow->commit(retained, &lease);
}

TEST(LowMachQbmmStateTest, CountOnlyMigrationRetainsCompatibleEosReplacementAndControllerAuthority)
{
    Fixture f;
    auto count_options = f.options;
    count_options.qbmm_families.clear();
    Flow previous(f.mesh, f.eos, count_options);
    auto count_state = f.initial();
    count_state.qbmm_families.clear();
    for (auto& cell : count_state.cells)
        cell.qbmm_shape.clear();
    previous.initialize(count_state);
    int calls = 0;
    R::ReactingMixtureCoupler prior_controller(
        std::shared_ptr<Flow>(&previous, [](Flow*) {}), restart_controller_options(f, calls));
    const auto saved = previous.checkpoint();
    Flow::QbmmInitialization initialization;
    initialization.cells.resize(count_state.cells.size());
    for (auto& cell : initialization.cells)
        cell = {std::nullopt, Flow::QbmmMonodisperse{}, std::nullopt};

    auto replacement = f.eos;
    const auto& bounds = f.eos[0]->bounds();
    replacement[0] = R::make_linear_compressible_mixture_liquid(f.catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 0, 0, 0}; });
    Flow migrated(f.mesh, replacement, f.options);
    auto lease = migrated.acquire_controller();
    migrated.restore_qbmm_checkpoint(previous, saved, initialization, &lease);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(migrated.state().time, previous.state().time);
    EXPECT_EQ(migrated.state().revision, previous.state().revision);
    for (size_t c = 0; c < count_state.cells.size(); ++c)
    {
        EXPECT_EQ(migrated.state().cells[c].phases[0].moles, previous.state().cells[c].phases[0].moles);
        EXPECT_EQ(
            migrated.state().cells[c].phases[0].internal_energy, previous.state().cells[c].phases[0].internal_energy);
        EXPECT_EQ(migrated.state().cells[c].phases[0].thermodynamics, replacement[0]);
    }

    // Identical kind/bounds are insufficient: a different caloric reference must
    // reject, while the same live native mesh and all source callbacks remain intact.
    replacement[0] = R::make_linear_compressible_mixture_liquid(f.catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1, 0, 0}; });
    Flow incompatible(f.mesh, replacement, f.options);
    EXPECT_THROW(incompatible.restore_qbmm_checkpoint(previous, saved, initialization), std::exception);
    EXPECT_FALSE(incompatible.initialized());
    EXPECT_EQ(calls, 0);
}

TEST(LowMachQbmmStateTest, CountOnlyMigrationRequiresTheSameSuppliedAcceleration)
{
    Fixture f;
    auto count_options = f.options;
    count_options.qbmm_families.clear();
    auto previous = std::make_shared<Flow>(f.mesh, f.eos, count_options);
    auto initial = f.initial();
    initial.qbmm_families.clear();
    for (auto& cell : initial.cells)
        cell.qbmm_shape.clear();
    previous->initialize(initial);
    const auto unforced = previous->checkpoint();
    Flow::QbmmInitialization shape;
    shape.cells.resize(initial.cells.size(), {std::nullopt, Flow::QbmmMonodisperse{}, std::nullopt});
    int calls = 0;
    auto matching = restart_controller_options(f, calls);
    matching.body_acceleration = Flow::Vector{0, 0, -1};
    const auto live_checkpoint = [&]
    {
        R::ReactingMixtureCoupler controller(previous, matching);
        const auto fresh = previous->checkpoint();
        controller.advance(.01);
        return fresh;
    }();
    const auto stepped_checkpoint = previous->checkpoint();
    const auto saved_time = previous->state().time;

    auto receiving_options = f.options;
    const auto receiving_family = f.make_family(f.catalog, receiving_options.phase_roles[1], .04 / 28);
    receiving_options.qbmm_families[1] = receiving_family;
    auto receiving = std::make_shared<Flow>(f.mesh, f.eos, receiving_options);
    receiving->initialize_qbmm(initial, shape);
    const auto before = receiving->serialize_qbmm_restart();
    EXPECT_THROW(receiving->restore_qbmm_checkpoint(*previous, live_checkpoint, shape), std::exception);
    EXPECT_THROW(receiving->restore_qbmm_checkpoint(*previous, stepped_checkpoint, shape), std::exception);
    EXPECT_EQ(receiving->serialize_qbmm_restart(), before);

    auto changed = matching;
    changed.body_acceleration = Flow::Vector{0, 0, -2};
    {
        R::ReactingMixtureCoupler controller(receiving, changed);
        EXPECT_THROW(controller.restore_qbmm_checkpoint(*previous, stepped_checkpoint, shape), std::exception);
    }
    {
        R::ReactingMixtureCoupler controller(receiving, restart_controller_options(f, calls));
        const auto attached = receiving->serialize_qbmm_restart();
        const auto kept = controller.preview(.01);
        const int before_attempt = calls;
        EXPECT_THROW(controller.restore_qbmm_checkpoint(*previous, stepped_checkpoint, shape), std::exception);
        EXPECT_EQ(receiving->serialize_qbmm_restart(), attached);
        EXPECT_EQ(calls, before_attempt);
        controller.commit(kept);
    }

    previous->restore(unforced);
    const auto restored_unforced = [&]
    {
        R::ReactingMixtureCoupler controller(previous, matching);
        return previous->checkpoint();
    }();

    // The receiving QBMM family schema and source identity may differ from the count-only inputs.
    matching.source_identity = "new paired QBMM source configuration";
    {
        R::ReactingMixtureCoupler controller(receiving, matching);
        const int before_attempt = calls;
        EXPECT_THROW(controller.restore_qbmm_checkpoint(*previous, unforced, shape), std::exception);
        EXPECT_THROW(controller.restore_qbmm_checkpoint(*previous, restored_unforced, shape), std::exception);
        EXPECT_NO_THROW(controller.restore_qbmm_checkpoint(*previous, live_checkpoint, shape));
        EXPECT_EQ(receiving->state().time, 0);
        EXPECT_NO_THROW(controller.restore_qbmm_checkpoint(*previous, stepped_checkpoint, shape));
        EXPECT_EQ(receiving->state().time, saved_time);
        EXPECT_EQ(receiving->state().qbmm_families.at(1), Flow::qbmm_family_stamp(*receiving_family));
        EXPECT_EQ(calls, before_attempt);
    }

    previous->restore(stepped_checkpoint);
    auto invalid = initial;
    invalid.thermodynamic_pressure = 0;
    EXPECT_THROW(previous->initialize(invalid), std::exception);
    EXPECT_THROW(receiving->restore_qbmm_checkpoint(*previous, previous->checkpoint(), shape), std::exception);
    previous->initialize(initial);
    EXPECT_NO_THROW(receiving->restore_qbmm_checkpoint(*previous, previous->checkpoint(), shape));
    EXPECT_EQ(receiving->state().time, 0);
}

TEST(LowMachQbmmStateTest, RestartRejectsChangedLawMetadataAndLegacyLawConfiguration)
{
    Fixture f;
    auto options = f.options;
    R::QbmmClosureVersions versions;
    versions.laws = {{"declared-slip", 1, R::QbmmLawKind::PhaseSlip, "per physical phase", "m/s"}};
    const auto family = [&]
    {
        return std::make_shared<const R::QbmmPopulationSchema>(f.catalog, f.catalog->fingerprint(), "fine",
            options.phase_roles[1], f.family->mass_scale_kg(), std::vector<R::QbmmMassFraction>{{11, .5}, {29, .5}},
            versions);
    };
    options.qbmm_families[1] = family();
    Flow original(f.mesh, f.eos, options);
    auto state = f.initial();
    state.qbmm_families[1] = Flow::qbmm_family_stamp(*options.qbmm_families[1]);
    original.initialize(state);
    const auto bytes = original.serialize_qbmm_restart();
    const auto old = earlier_restart(f, bytes, 4);
    EXPECT_THROW(original.restore_qbmm_restart(old), std::exception);
    EXPECT_EQ(original.serialize_qbmm_restart(), bytes);

    ++versions.laws.front().revision;
    options.qbmm_families[1] = family();
    Flow changed(f.mesh, f.eos, options);
    state.qbmm_families[1] = Flow::qbmm_family_stamp(*options.qbmm_families[1]);
    changed.initialize(state);
    const auto before = changed.serialize_qbmm_restart();
    const auto eos_calls = *f.liquid_evaluations;
    EXPECT_THROW(changed.restore_qbmm_restart(bytes), std::exception);
    EXPECT_EQ(changed.serialize_qbmm_restart(), before);
    EXPECT_EQ(*f.liquid_evaluations, eos_calls);
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
        {"node slip with native implicit transport", [](Model& m) { m.velocity = Model::Velocity::NodeDependentSlip; }},
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

TEST(LowMachQbmmStateTest, PurePoolLawContractRejectsRuntimeAdmissionEvenWithConversionFlagDisabled)
{
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    for (const bool declare_pool_kind : {true, false})
    {
        auto options = f.options;
        if (comm->getRank() == comm->getSize() - 1)
        {
            R::QbmmClosureVersions closures;
            R::QbmmLawMetadata law{"Q9a contract", 1,
                declare_pool_kind ? R::QbmmLawKind::BulkGasConversion : R::QbmmLawKind::MassTransfer,
                "prescribed transfer", "mol/cell", R::QbmmApplicabilityGate::ReportOnly, true, {}};
            if (!declare_pool_kind)
                law.ranges.push_back({R::QbmmRangeVariable::BulkGasPoolVolumeFraction, 0, 1, {}});
            closures.laws.push_back(std::move(law));
            options.qbmm_families[1] =
                Fixture::make_family(f.catalog, options.phase_roles[1], f.family->mass_scale_kg(), closures);
        }
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    }
}

TEST(LowMachQbmmStateTest, RealizableLinearRequiresSharedCarrierAndUniformControls)
{
    Fixture f;
    using Model = R::QbmmModelOptions;
    f.options.qbmm_model.family_transport = Model::FamilyTransport::RealizableLinearSSPRK2;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const auto configures_only_on_last_rank = [&](auto change)
    {
        auto options = f.options;
        if (comm->getRank() == comm->getSize() - 1)
            change(options);
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    };
    const auto rejects = [&](auto change)
    { configures_only_on_last_rank([&](Flow::Options& options) { change(options.qbmm_model); }); };
    rejects([](Model& model) { model.transport.time = FVM::ScalarTimeScheme::BDF2; });
    rejects([](Model& model) { model.transport.convection = FVM::ScalarConvectionScheme::BoundedLinearUpwind; });
    rejects([](Model& model) { model.transport.mode = FVM::AssemblyMode::Steady; });
    rejects([](Model& model) { model.velocity = Model::Velocity::SuppliedPhaseSlip; });
    rejects([](Model& model) { model.velocity = Model::Velocity::NodeDependentSlip; });
    rejects([](Model& model) { model.velocity = Model::Velocity::IndependentPhaseMomentum; });
    rejects([](Model& model) { model.linear_cfl = std::nextafter(.5, 1.); });
    rejects([](Model& model) { model.linear_cfl = 0.; });
    rejects([](Model& model) { model.maximum_linear_substeps = 0; });
    configures_only_on_last_rank([](Flow::Options& options) { options.qbmm_families.clear(); });
    if (comm->getSize() > 1)
    {
        rejects([](Model& model) { model.linear_cfl = .25; });
        rejects([](Model& model) { --model.maximum_linear_substeps; });
        rejects([](Model& model) { model.family_transport = Model::FamilyTransport::NativeImplicit; });
    }
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto earlier = flow->preview(.01);
    EXPECT_EQ(earlier.diagnostics().family_transport, Model::FamilyTransport::RealizableLinearSSPRK2);
    EXPECT_EQ(earlier.diagnostics().linear_substeps, 1);
    EXPECT_EQ(earlier.diagnostics().linear_stages, 2u);
    EXPECT_EQ(earlier.diagnostics().maximum_linear_cfl, 0.);
    Flow::PlanarMotion motion(f.mesh);
    const auto epoch = f.mesh->geometry_epoch();
    EXPECT_NO_THROW((void) flow->preview_ale(.01, motion, 1.001));
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(f.mesh->geometry_epoch(), epoch);
    int calls = 0;
    auto forced = restart_controller_options(f, calls);
    // A forced controller requires an identity for the fixture's opaque liquid EOS.
    forced.body_acceleration = Flow::Vector{};
    EXPECT_THROW((void) R::ReactingMixtureCoupler(flow, std::move(forced)), std::exception);
    EXPECT_EQ(calls, 0);
    EXPECT_NO_THROW(flow->commit(earlier));

    Flow::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    const auto periodic = std::make_shared<Flow::Mesh>(
        std::make_shared<Flow::Mesh::Cartesian>(
            Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, 1, 2}}, Vec3D<bool>{false, false, true}),
        distribution);
    EXPECT_THROW((void) Flow(periodic, f.eos, f.options), std::exception);
    using Semi = Flow::Mesh::SemiStructured;
    const auto skew = std::make_shared<Flow::Mesh>(
        std::make_shared<Semi>(Arr<Semi::Vec3>{{0, 0, 0}, {1, 0, 0}, {2, .5, 0}, {2, 1.5, 0}, {1, 1, 0}, {0, 1, 0}},
            Arr<Arr<unsigned>>{{0, 1, 4, 5}, {1, 2, 3, 4}}, ArrReal{0, 1}),
        distribution);
    EXPECT_THROW((void) Flow(skew, f.eos, f.options), std::exception);
}

TEST(LowMachQbmmStateTest, ForcedLinearRequiresAClosedPolygonalDiskOrAnnulusBase)
{
    using Semi = Flow::Mesh::SemiStructured;
    Fixture f;
    f.options.qbmm_model.family_transport = R::QbmmModelOptions::FamilyTransport::RealizableLinearSSPRK2;
    f.options.qbmm_eos_law_identities = {
        R::ThermalLawIdentity{"polygon fixture liquid", 1, f.catalog->fingerprint()}, std::nullopt, std::nullopt};
    const Flow::Mesh::DistributionOptions distribution{.allow_empty_partitions = true};
    const auto square_base = [&](const std::vector<std::array<unsigned, 2>>& squares)
    {
        constexpr unsigned width = 6, height = 4;
        Arr<Semi::Vec3> nodes;
        Arr<Arr<unsigned>> cells;
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
                nodes.push_back({static_cast<double>(x), static_cast<double>(y), 0.});
        for (const auto& [x, y] : squares)
        {
            const unsigned corner = y * width + x;
            cells.push_back({corner, corner + 1, corner + width + 1, corner + width});
        }
        return std::make_shared<Flow::Mesh>(std::make_shared<Semi>(nodes, cells, ArrReal{0., 1.}), distribution);
    };
    std::vector<std::array<unsigned, 2>> two_holes;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 5; ++x)
            if (!(y == 1 && (x == 1 || x == 3)))
                two_holes.push_back({x, y});
    for (const auto& base :
        {square_base({{0, 0}, {1, 0}, {0, 1}}), square_base(two_holes), square_base({{0, 0}, {2, 0}})})
    {
        f.mesh = base;
        auto flow = f.solver();
        flow->initialize(f.initial());
        const auto unforced = flow->preview(.01);
        const auto previous = flow->serialize_qbmm_restart();
        int calls = 0;
        for (const auto force : {Flow::Vector{}, Flow::Vector{0., 0., -1.}})
        {
            auto options = restart_controller_options(f, calls);
            options.body_acceleration = force;
            EXPECT_THROW((void) R::ReactingMixtureCoupler(flow, std::move(options)), std::exception);
            EXPECT_EQ(flow->serialize_qbmm_restart(), previous);
        }
        EXPECT_EQ(calls, 0);
        EXPECT_NO_THROW(flow->commit(unforced));
    }

    // Irregular angular spacing, translated and rotated in the XY plane, with a split straight facet.
    Arr<Semi::Vec3> points;
    for (const double theta : {.23, .9, 2., 3.1, 4.2, 5.2})
        points.push_back({8. + std::cos(theta), -5. + std::sin(theta), 0.});
    points.push_back(points[1] * .75 + points[2] * .25);
    points.push_back(points[1] * .25 + points[2] * .75);
    f.mesh = std::make_shared<Flow::Mesh>(
        std::make_shared<Semi>(points, Arr<Arr<unsigned>>{{0, 1, 6, 7, 2, 3, 4, 5}}, ArrReal{0., 1.}), distribution);
    auto flow = f.solver();
    flow->initialize(f.initial());
    int calls = 0;
    auto options = restart_controller_options(f, calls);
    options.body_acceleration = Flow::Vector{};
    EXPECT_NO_THROW({
        R::ReactingMixtureCoupler controller(flow, std::move(options));
        EXPECT_EQ(flow->qbmm_report().body_acceleration_m_per_s2, Flow::Vector{});
        EXPECT_TRUE(flow->qbmm_report().balanced_linear_pressure_policy.has_value());
    });
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
    options = f.options;
    if (last)
        options.linear.absolute_tolerance = 1e-10;
    EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
}

void bind_native_laws(Fixture& fixture, std::vector<R::QbmmLawMetadata> laws)
{
    R::QbmmClosureVersions versions;
    versions.laws = std::move(laws);
    fixture.family = Fixture::make_family(
        fixture.catalog, fixture.options.phase_roles[1], fixture.family->mass_scale_kg(), std::move(versions));
    fixture.options.qbmm_families[1] = fixture.family;
    fixture.options.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SuppliedPhaseSlip;
}

R::QbmmLawMetadata native_slip_gate(double minimum = 0, double maximum = .125)
{
    return {"native-supplied-slip", 1, R::QbmmLawKind::PhaseSlip, "common family drift relative to carrier, m/s", "m/s",
        R::QbmmApplicabilityGate::RejectOutside, true,
        {{R::QbmmRangeVariable::SlipSpeedMPerS, minimum, maximum, std::nullopt},
            {R::QbmmRangeVariable::TimeStepS, .005, .02, std::nullopt}}};
}

TEST(LowMachQbmmStateTest, NativeHardTransportGatesAdmitProviderFreePreviewAndDeferDurationUntilPreview)
{
    for (const unsigned owned_cells : {1u, 4u})
    {
        Fixture f(owned_cells);
        const R::QbmmLawMetadata property{"native-property", 1, R::QbmmLawKind::TransportProperty,
            "frozen physical p,T,phase geometry and liquid concentration", "supplied SI transport property",
            R::QbmmApplicabilityGate::RejectOutside, true,
            {{R::QbmmRangeVariable::AbsolutePressurePa, 90000, 110000, std::nullopt},
                {R::QbmmRangeVariable::TemperatureK, 290, 310, std::nullopt},
                {R::QbmmRangeVariable::PhaseVolumeFraction, 0, 1, std::nullopt},
                {R::QbmmRangeVariable::GasVolumeFraction, 0, 1, std::nullopt},
                {R::QbmmRangeVariable::DiameterM, 0, 1, std::nullopt},
                {R::QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3, 0, 100000, 0}}};
        bind_native_laws(f, {native_slip_gate(), property});
        auto flow = f.solver(); // No controller and no population provider.
        auto initial = f.initial();
        for (auto& cell : initial.cells)
        {
            cell.drift_velocity.resize(3);
            cell.drift_velocity[1].x = .125; // Supplied field at the inclusive speed upper bound.
        }
        flow->initialize(initial);
        const auto initialized = flow->qbmm_report();
        for (const auto& cell : initialized.cells)
        {
            const auto& slip = cell.families[1]->applicability[0];
            EXPECT_EQ(slip.ranges[1].status, R::QbmmRangeStatus::MissingInput);
            EXPECT_FALSE(slip.gate_passed); // Reporting missing duration never rejects initialization.
        }
        const auto trial = flow->preview(.01);
        const auto trial_report = flow->qbmm_report(trial);
        for (const auto& cell : trial_report.cells)
            for (const auto& law : cell.families[1]->applicability)
                EXPECT_TRUE(law.gate_passed);
        const unsigned calls = *f.liquid_evaluations;
        flow->commit(trial);
        EXPECT_EQ(*f.liquid_evaluations, calls);
        const auto archive = flow->serialize_qbmm_restart();
        flow->restore_qbmm_restart(archive);
        // Portable restore revalidates the archived caloric state with the EOS.
        if (f.mesh->num_owned_cells() > 0)
            EXPECT_GT(*f.liquid_evaluations, calls);
        const auto restarted = flow->qbmm_report();
        for (const auto& cell : restarted.cells)
            EXPECT_EQ(cell.families[1]->applicability[0].ranges[1].status, R::QbmmRangeStatus::MissingInput);
        EXPECT_NO_THROW((void) flow->preview(.01));
    }
}

TEST(LowMachQbmmStateTest, NativeHardSlipGatesRejectZeroAndBadDurationWithoutPublishing)
{
    Fixture f;
    bind_native_laws(f, {native_slip_gate(.1, .2)});
    auto flow = f.solver();
    flow->initialize(f.initial()); // Exact zero slip is admissible state, outside this supplied law's range.
    const auto original = flow->serialize_qbmm_restart();
    EXPECT_THROW((void) flow->preview(.01), std::exception);
    EXPECT_EQ(flow->serialize_qbmm_restart(), original);

    bind_native_laws(f, {native_slip_gate()});
    flow = f.solver();
    flow->initialize(f.initial());
    const auto origin = flow->serialize_qbmm_restart();
    const auto retained = flow->preview(.01);
    EXPECT_THROW((void) flow->preview(.04), std::exception);
    EXPECT_EQ(flow->serialize_qbmm_restart(), origin);
    flow->commit(retained);
    EXPECT_EQ(flow->state().time, .01);
}

TEST(LowMachQbmmStateTest, NativeEndpointTransportGatesRejectSourceHeatedCandidateAndRetainEarlierTrial)
{
    // Independent admissibility control: the same heat packet has a valid numerical
    // endpoint when the identical declared physical range is diagnostic-only.
    for (const auto kind : {R::QbmmLawKind::TransportProperty, R::QbmmLawKind::PhaseSlip})
    {
        Fixture f;
        const R::QbmmLawMetadata gate{"temperature-conditioned-transport", 1, kind,
            "accepted and candidate physical temperature", kind == R::QbmmLawKind::PhaseSlip ? "m/s" : "Pa.s",
            R::QbmmApplicabilityGate::RejectOutside, true,
            {{R::QbmmRangeVariable::TemperatureK, 299, 300.00001, std::nullopt}}};
        bind_native_laws(f, {gate});
        auto flow = f.solver();
        flow->initialize(f.initial());
        const auto retained = flow->preview(.01);
        const auto origin = flow->serialize_qbmm_restart();
        std::vector<Flow::CellSources> heat(flow->state().cells.size());
        for (auto& source : heat)
            source.internal_energy = {4000, 0, 0};

        auto diagnostic = gate;
        diagnostic.gate = R::QbmmApplicabilityGate::ReportOnly;
        bind_native_laws(f, {diagnostic});
        auto reference = f.solver();
        reference->initialize(f.initial());
        const auto heated = reference->preview(.01, heat);
        for (const auto& cell : reference->qbmm_report(heated).cells)
        {
            EXPECT_GT(*cell.families[1]->temperature_K, 300.00001);
            const auto& applicability = cell.families[1]->applicability.front();
            EXPECT_EQ(applicability.ranges.front().status, R::QbmmRangeStatus::OutsideRange);
            EXPECT_TRUE(applicability.gate_passed);
        }
        EXPECT_THROW((void) flow->preview(.01, heat), std::exception);
        EXPECT_EQ(flow->serialize_qbmm_restart(), origin);
        flow->commit(retained);
        EXPECT_EQ(flow->state().time, .01);
    }
}

TEST(LowMachQbmmStateTest, NativeSlipGateFailureOnOneOwnerFencesBalancedAndEmptyPartitions)
{
    for (const unsigned owned_cells : {1u, 4u})
    {
        Fixture f(owned_cells);
        bind_native_laws(f, {native_slip_gate(0, .2)});
        auto flow = f.solver();
        auto initial = f.initial();
        const auto comm = f.mesh->owned_cell_map()->getComm();
        for (auto& cell : initial.cells)
        {
            cell.drift_velocity.resize(3);
            cell.drift_velocity[1].x = comm->getRank() == 0 ? .25 : .125;
        }
        flow->initialize(initial);
        const auto origin = flow->serialize_qbmm_restart();
        EXPECT_THROW((void) flow->preview(.01), std::exception);
        EXPECT_EQ(flow->serialize_qbmm_restart(), origin);
    }
}

TEST(LowMachQbmmStateTest, CountOnlyRankLocalPreviewDriftFailuresReachEveryOwnerWithoutAnyLawMetadata)
{
    for (const bool qbmm_enabled : {false, true})
    {
        Fixture f;
        auto initial = f.initial();
        if (!qbmm_enabled)
        {
            f.options.qbmm_families.clear();
            initial.qbmm_families.clear();
            for (auto& cell : initial.cells)
                cell.qbmm_shape.clear();
        }
        auto flow = f.solver();
        flow->initialize(initial);
        const auto retained = flow->preview(.01);
        const auto origin = flow->state();
        // Deliberate fault injection at the preview boundary, not a supported state
        // mutation API. The affected coarse phase has no family/shape metadata.
        auto& injected = const_cast<Flow::State&>(flow->state());
        const bool owner = f.mesh->owned_cell_map()->getComm()->getRank() == 0;
        if (owner)
            injected.cells[0].drift_velocity[2].z = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW((void) flow->preview(.01), std::exception);
        if (owner)
            injected.cells[0].drift_velocity[2].z = origin.cells[0].drift_velocity[2].z;
        expect_state(flow->state(), origin);
        flow->commit(retained);
        EXPECT_EQ(flow->state().time, .01);
    }
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

TEST(LowMachQbmmStateTest, NativeHardPropertyGateFencesCorrectorViscosityAndPreservesAcceptedState)
{
    Fixture f; // One cell leaves the other ranks empty in the registered MPI cases.
    constexpr double ceiling = 300.00001;
    const R::QbmmLawMetadata gate{"native-viscosity-spy", 1, R::QbmmLawKind::TransportProperty,
        "recovered family temperature at viscosity evaluation", "Pa.s", R::QbmmApplicabilityGate::RejectOutside, true,
        {{R::QbmmRangeVariable::TemperatureK, 299, ceiling, std::nullopt}}};
    bind_native_laws(f, {gate});
    unsigned calls = 0, outside = 0;
    f.options.mixture_dynamic_viscosity = [&](std::span<const R::PhaseThermalState> phases, std::span<const double>)
    {
        ++calls;
        const double temperature = phases[1].temperature;
        if (temperature < 299 || temperature > ceiling)
            ++outside;
        return 0.;
    };
    auto flow = f.solver();
    flow->initialize(f.initial());
    const auto retained = flow->preview(.01);
    const auto origin = flow->serialize_qbmm_restart();
    const auto in_range_calls = calls;
    std::vector<Flow::CellSources> heat(flow->state().cells.size());
    for (auto& source : heat)
        source.internal_energy = {4000, 0, 0};
    if (!heat.empty())
        EXPECT_GT(in_range_calls, 0u);
    EXPECT_EQ(outside, 0u);
    EXPECT_THROW((void) flow->preview(.01, heat), std::exception);
    EXPECT_EQ(calls, in_range_calls);
    EXPECT_EQ(outside, 0u);
    EXPECT_EQ(flow->serialize_qbmm_restart(), origin);
    flow->commit(retained);
    expect_state(flow->state(), retained.state());

    auto diagnostic = gate;
    diagnostic.gate = R::QbmmApplicabilityGate::ReportOnly;
    bind_native_laws(f, {diagnostic});
    auto reference = f.solver();
    reference->initialize(f.initial());
    EXPECT_NO_THROW((void) reference->preview(.01, heat));
    if (!heat.empty())
        EXPECT_GT(outside, 0u);
}

TEST(LowMachQbmmStateTest, BarycentricReportPreservesAbsoluteMomentumAcrossCancellationAndOverflow)
{
    Fixture f;
    f.family = Fixture::make_family(f.catalog, f.options.phase_roles[1], 1. / 14);
    f.options.qbmm_families[1] = f.family;
    f.options.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SuppliedPhaseSlip;
    // Fill one cubic metre with exactly 2 kg liquid and 1 kg bound gas at 300 K.
    const double gas_volume = 37.5 * 8.31446261815324 * 300 / 100000;
    f.eos[0] = R::make_linear_compressible_mixture_liquid(f.catalog, f.eos[0]->bounds(),
        [density = 2 / (1 - gas_volume)](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, density, 4000, 0, 0, 0, 0, 0}; });
    auto initial = f.initial();
    const double next = std::nextafter(1., 2.);
    const double largest = std::numeric_limits<double>::max();
    const double smallest = std::numeric_limits<double>::denorm_min();
    for (auto& cell : initial.cells)
    {
        cell.phases[0].moles = {100, 0};
        cell.phases[1].moles = {25, 12.5};
        cell.phases[2].moles = {0, 0};
        cell.phases[2].temperature = 0;
        cell.phase_numbers[2] = 0;
        cell.velocity = {-1, largest, 0};
        cell.drift_velocity = {{1, -largest, 0}, {next, largest, smallest}, {}};
    }
    auto flow = f.solver();
    flow->initialize(initial);
    const auto overflow_report = flow->qbmm_report();
    ASSERT_EQ(overflow_report.cells.size(), f.mesh->num_owned_cells());
    for (const auto& cell : overflow_report.cells)
    {
        // With masses 2 and 1, the signed absolute x momentum is exactly 2^-52.
        // The y means divide a finite operand, even though their momenta overflow double.
        EXPECT_EQ(cell.mass_weighted_drift.x, 1.);
        EXPECT_EQ(cell.barycentric_velocity.x, 0x1.5555555555555p-54);
        EXPECT_EQ(cell.mass_weighted_drift.y, -largest / 3.);
        EXPECT_EQ(cell.barycentric_velocity.y, largest / 1.5);
        EXPECT_EQ(cell.mass_weighted_drift.z, 0.);
        EXPECT_EQ(cell.barycentric_velocity.z, 0.);
    }

    // Both per-phase velocity sums round away the carrier in binary64. The
    // opposing drifts cancel exactly and the absolute velocity keeps it.
    for (auto& cell : initial.cells)
    {
        cell.velocity = {smallest, 0, 0};
        cell.drift_velocity = {{-largest / 2, 0, 0}, {largest, 0, 0}, {}};
    }
    flow->initialize(initial);
    const auto cancellation_report = flow->qbmm_report();
    ASSERT_EQ(cancellation_report.cells.size(), f.mesh->num_owned_cells());
    for (const auto& cell : cancellation_report.cells)
    {
        EXPECT_EQ(cell.mass_weighted_drift.x, 0.);
        EXPECT_EQ(cell.barycentric_velocity.x, smallest);
    }
}

TEST(LowMachQbmmStateTest, BarycentricReportPreservesUniformExtremeVelocitiesWithUnequalMasses)
{
    Fixture f;
    constexpr double gas_mass = 0x1p-53;
    const double largest = std::numeric_limits<double>::max();
    f.family = Fixture::make_family(f.catalog, f.options.phase_roles[1], gas_mass / 14);
    f.options.qbmm_families[1] = f.family;
    f.options.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SuppliedPhaseSlip;
    f.eos[0] = R::make_linear_compressible_mixture_liquid(f.catalog, f.eos[0]->bounds(), [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1, 4000, 0, 0, 0, 0, 0}; });
    auto initial = f.initial();
    for (auto& cell : initial.cells)
    {
        cell.phases[0].moles = {50, 0};
        cell.phases[1].moles = {25 * gas_mass, 12.5 * gas_mass};
        cell.phases[2].moles = {0, 0};
        cell.phases[2].temperature = 0;
        cell.phase_numbers[2] = 0;
        cell.velocity = {largest, 0, -largest};
        cell.drift_velocity = {{0, largest, 0}, {0, largest, 0}, {}};
    }

    auto flow = f.solver();
    flow->initialize(initial);
    const auto report = flow->qbmm_report();
    ASSERT_EQ(report.cells.size(), f.mesh->num_owned_cells());
    for (const auto& cell : report.cells)
    {
        ASSERT_TRUE(cell.families[1]);
        EXPECT_EQ(cell.families[1]->phase_mass_kg, gas_mass);
        // The physical mass is 1 + 2^-53 kg. Its binary64 density rounds to 1 kg/m^3.
        EXPECT_EQ(cell.mixture_density_kg_per_m3, 1.);
        EXPECT_EQ(cell.mass_weighted_drift.y, largest);
        EXPECT_EQ(cell.barycentric_velocity.x, largest);
        EXPECT_EQ(cell.barycentric_velocity.y, largest);
        EXPECT_EQ(cell.barycentric_velocity.z, -largest);
    }
}
} // namespace
