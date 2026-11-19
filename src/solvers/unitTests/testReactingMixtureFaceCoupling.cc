/** Public-controller conservation oracles for ordered mixture face sources. */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "solvers/details/LowMachFourierThermal.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <optional>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using Controller = R::ReactingMixtureCoupler;
using LO = Flow::Pack::local_ordinal_type;
constexpr R::SpeciesId solvent = 0, first = 19, second = 71;
constexpr double capacity = 256.; // Two 0.5 kg cells with cp=512 J/(kg K).
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Fixture
{
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<Flow> flow;
    explicit Fixture(bool thermal = false, bool reaction_everywhere = false, double heat = 0,
        double interface_height = .5, std::optional<std::array<double, 2>> initial_temperatures = std::nullopt)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{
                                                ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, interface_height, 1}}),
            distribution);
        catalog = std::make_shared<const R::SpeciesCatalog>(
            std::vector<R::SpeciesDefinition>{{first, "first solute", .1, {{"X", 1}}, {R::SpeciesPhase::Dissolved}},
                {second, "second solute", .1, {{"X", 1}}, {R::SpeciesPhase::Dissolved}},
                {solvent, "explicit finite counter", .1, {{"X", 1}}, {R::SpeciesPhase::Liquid}}});
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, {256, 512, 10000, 2000000}, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1, 512, 0, 0, 0, 0, 0}; },
            R::EnergyConvention::SensibleEnergy);
        mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
            std::vector<R::ReactionDefinition>{{7, "supplied isomer conversion",
                {{first, R::SpeciesPhase::Dissolved, -1, {}}, {second, R::SpeciesPhase::Dissolved, 1, {}}}, heat}},
            R::EnergyConvention::SensibleEnergy);
        Flow::Options controls;
        controls.minimum_pressure = 20000;
        controls.maximum_pressure = 1000000;
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        flow = std::make_shared<Flow>(mesh, std::vector{liquid}, controls);
        Flow::State state;
        state.thermodynamic_pressure = 100000;
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const bool lower = mesh->cell_centroid(cell).z < interface_height;
            const double n = lower || reaction_everywhere ? 2 : 0;
            const double baseline_temperature = thermal && !lower ? 301 : 300;
            const double t = initial_temperatures ? (*initial_temperatures)[lower ? 0 : 1] : baseline_temperature;
            const double mass = mesh->cell_volume(cell); // rho = 1 kg/m3.
            state.cells[c].phases = {{liquid, {n, 0, mass / .1 - n}, 100000, mass * 512 * (t - 300), t}};
        }
        flow->initialize(std::move(state));
    }
    R::ReactingMixtureOptions options() const
    {
        R::ReactingMixtureOptions result;
        result.mechanism = mechanism;
        result.phases = {{R::SpeciesPhase::Liquid, {},
            {R::SpeciesPhase::Dissolved, R::SpeciesPhase::Dissolved, R::SpeciesPhase::Liquid}}};
        result.source_identity = "independent public face oracle";
        return result;
    }
    R::MixtureFaceTransportOptions contacts(double conductance = 0, double fourier = 0) const
    {
        R::MixtureFaceTransportOptions result;
        result.identity = "supplied contact";
        result.schema_fingerprint = catalog->fingerprint();
        if (conductance > 0)
        {
            R::MixtureDiffusionLaw law;
            law.binding.identity = "finite explicit counter including ID zero";
            law.binding.schemaFingerprint = catalog->fingerprint();
            law.binding.counterSpecies = solvent;
            law.binding.energyConvention = R::EnergyConvention::SensibleEnergy;
            law.primary_species = {first};
            law.molar_conductance = [conductance](const auto&) { return std::vector<double>{conductance, 0, 0}; };
            law.partial_molar_enthalpy = [](R::SpeciesId, const auto&, const R::MixtureFacePhaseSample& donor)
            { return .1 * (512 * (donor.temperature - 300) + donor.absolute_pressure); };
            result.diffusion = std::move(law);
        }
        if (fourier > 0)
            result.fourier = R::MixtureFourierLaw{
                "supplied thermal contact", 1, catalog->fingerprint(), [fourier](const auto&) { return fourier; }};
        return result;
    }
    double sum(double local) const
    {
        double global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    }
    bool lower(size_t c) const { return mesh->cell_centroid(static_cast<LO>(c)).z < .5; }
};
void expect_same(const Flow::State& a, const Flow::State& b)
{
    EXPECT_EQ(a.revision, b.revision);
    EXPECT_DOUBLE_EQ(a.time, b.time);
    EXPECT_DOUBLE_EQ(a.thermodynamic_pressure, b.thermodynamic_pressure);
    ASSERT_EQ(a.cells.size(), b.cells.size());
    for (size_t c = 0; c < a.cells.size(); ++c)
    {
        ASSERT_EQ(a.cells[c].phases.size(), b.cells[c].phases.size());
        ASSERT_EQ(a.cells[c].drift_velocity.size(), b.cells[c].drift_velocity.size());
        for (size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_DOUBLE_EQ(a.cells[c].velocity.component(axis), b.cells[c].velocity.component(axis));
            for (size_t phase = 0; phase < a.cells[c].drift_velocity.size(); ++phase)
                EXPECT_DOUBLE_EQ(
                    a.cells[c].drift_velocity[phase].component(axis), b.cells[c].drift_velocity[phase].component(axis));
        }
        EXPECT_EQ(a.cells[c].phases[0].thermodynamics.get(), b.cells[c].phases[0].thermodynamics.get());
        EXPECT_DOUBLE_EQ(a.cells[c].phases[0].absolute_pressure, b.cells[c].phases[0].absolute_pressure);
        EXPECT_EQ(a.cells[c].phases[0].moles, b.cells[c].phases[0].moles);
        EXPECT_DOUBLE_EQ(a.cells[c].phases[0].temperature, b.cells[c].phases[0].temperature);
        EXPECT_DOUBLE_EQ(a.cells[c].phases[0].internal_energy, b.cells[c].phases[0].internal_energy);
        EXPECT_DOUBLE_EQ(a.cells[c].mechanical_pressure, b.cells[c].mechanical_pressure);
    }
}
TEST(ReactingMixtureFaceCouplingTest, DiffusionUsesActualPostChemistryDonorsAndExplicitZeroIdCounter)
{
    Fixture f;
    auto options = f.options();
    options.source = [](const R::CoupledReactionContext& context, double)
    {
        R::ReactionSourceInput result;
        if (context.reaction.state.amount(context.reaction.catalog, first, R::SpeciesPhase::Dissolved) > 0)
            result.integratedExtents = {{7, .5}};
        return result;
    };
    options.face_transport = {f.contacts(.125)};
    Controller controller(f.flow, options);
    const auto before = f.flow->state();
    const auto trial = controller.preview(.125);
    for (size_t c = 0; c < before.cells.size(); ++c)
    {
        const auto& n = trial.state().cells[c].phases[0].moles;
        EXPECT_NEAR(n[0], f.lower(c) ? 1.453125 : .046875, 2e-12);
        EXPECT_NEAR(n[1], f.lower(c) ? .5 : 0, 2e-12);
        EXPECT_NEAR(n[2], f.lower(c) ? 3.046875 : 4.953125, 2e-12);
        EXPECT_NEAR(trial.state().cells[c].phases[0].internal_energy, 0, 1e-8);
    }
    EXPECT_DOUBLE_EQ(trial.diagnostics().net_thermal_energy, 0);
    EXPECT_NEAR(trial.diagnostics().flow.mass_error, 0, 1e-12);
    expect_same(f.flow->state(), before);
    EXPECT_EQ(
        f.sum(f.mesh->num_owned_cells() == 0 ? 1 : 0), std::max(0, f.mesh->owned_cell_map()->getComm()->getSize() - 2));
    controller.commit(trial);
}
TEST(ReactingMixtureFaceCouplingTest, FourierClosesTheCoupledTwoCellEnergyBalanceAndReplays)
{
    Fixture f(true);
    auto options = f.options();
    options.face_transport = {f.contacts(0, 2)};
    Controller controller(f.flow, options);
    const auto checkpoint = controller.checkpoint();
    const auto trial = controller.preview(.125);
    const double difference = 1 / (1 + 2 * .125 * 2 / capacity);
    const double gain = .125 * 2 * difference;
    double energy = 0;
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& phase = trial.state().cells[c].phases[0];
        EXPECT_NEAR(phase.temperature, f.lower(c) ? 300 + gain / capacity : 301 - gain / capacity, 2e-8);
        energy += phase.internal_energy;
    }
    EXPECT_NEAR(f.sum(energy), capacity, 1e-7);
    EXPECT_NEAR(trial.diagnostics().net_thermal_energy, 0, 1e-12);
    controller.commit(trial);
    const auto accepted = f.flow->state();
    controller.restore(checkpoint);
    expect_same(controller.preview(.125).state(), accepted);
}
TEST(ReactingMixtureFaceCouplingTest, StiffFourierContactMatchesUnequalCapacityClosedForm)
{
    constexpr double interface_height = .25, dt = .125, conductance = 3072;
    constexpr std::array capacities{128., 384.};
    // The liquid is incompressible with cv = 512 J/(kg K), and rho = 1 kg/m3.
    // With no other source, C1*(T1-T1*)=-Q and C2*(T2-T2*)=Q. Substituting
    // into Q=dt*G*(T1-T2) gives the backward-Euler packet below.
    for (const auto& temperatures : {std::array{305., 300.}, std::array{300., 305.}, std::array{300., 300.}})
    {
        SCOPED_TRACE(temperatures[0] - temperatures[1]);
        Fixture f(false, false, 0, interface_height, temperatures);
        auto options = f.options();
        options.face_transport = {f.contacts(0, conductance)};
        Controller controller(f.flow, options);
        const auto before = f.flow->state();
        const auto trial = controller.preview(dt);
        const double integrated_conductance = dt * conductance;
        const double heat = integrated_conductance * (temperatures[0] - temperatures[1]) /
                            (1 + integrated_conductance * (1 / capacities[0] + 1 / capacities[1]));
        std::array<double, 2> energy_changes{};
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const bool upper = f.mesh->cell_centroid(static_cast<LO>(c)).z > interface_height;
            const double expected = upper ? heat : -heat;
            const auto& phase = trial.state().cells[c].phases[0];
            const auto& transported = trial.transport_snapshot().cells.at(c).phases.at(0);
            EXPECT_EQ(transported.internal_energy, before.cells[c].phases[0].internal_energy);
            EXPECT_EQ(transported.advected_material_volume, f.mesh->cell_volume(static_cast<LO>(c)));
            const double energy_change = phase.internal_energy - transported.internal_energy;
            energy_changes[upper] += energy_change;
            if (heat == 0)
            {
                EXPECT_EQ(phase.temperature, temperatures[upper]);
                EXPECT_EQ(energy_change, 0);
            }
            else
            {
                EXPECT_NEAR(phase.temperature, temperatures[upper] + expected / capacities[upper], 1e-8);
                EXPECT_NEAR(energy_change, expected, 1e-6);
            }
        }
        EXPECT_NEAR(f.sum(energy_changes[0]) + f.sum(energy_changes[1]), 0, 1e-7);
        EXPECT_EQ(trial.diagnostics().net_thermal_energy, 0);
        EXPECT_LE(trial.diagnostics().flow.source_evaluations, 80);
        EXPECT_LE(trial.diagnostics().flow.maximum_event_residual, 1);
        expect_same(f.flow->state(), before);
        controller.commit(trial);
        EXPECT_EQ(f.flow->state().time, dt);
    }
}
TEST(ReactingMixtureFaceCouplingTest, UnrepresentableFourierScalingKeepsRecoveredGuess)
{
    namespace thermal = R::fourier_thermal_detail;
    Fixture f(false, false, 0, .5, std::array{301., 300.});
    const auto recovered = f.flow->state();
    // The first block has a subnormal scaled diagonal. The second has normal diagonals
    // and its sole nonzero defect underflows in the squared scaled RHS norm.
    for (const auto& [small_capacity, large_capacity, conductance] :
        {std::array{1e-8, 1e302, 1e-9}, std::array{1e-200, 1., 1e-201}})
    {
        SCOPED_TRACE(small_capacity);
        auto stage = std::make_shared<R::low_mach_faces::Receipt>();
        auto& rows = stage->fourier_thermal.emplace(recovered.cells.size());
        for (std::size_t c = 0; c < rows.size(); ++c)
            rows[c] = {f.lower(c) ? small_capacity : large_capacity, 300.};
        const auto owned = [&](LO cell) -> std::optional<std::size_t>
        {
            if (cell != Flow::Mesh::invalid_local_id() && f.mesh->is_owned_cell(cell))
                return static_cast<std::size_t>(cell);
            return std::nullopt;
        };
        double owned_contacts = 0;
        for (std::size_t fidx = 0; fidx < f.mesh->num_faces(); ++fidx)
        {
            const auto face = static_cast<LO>(fidx);
            if (f.mesh->is_geometry_exterior_face(face))
                continue;
            const auto owner = f.mesh->owner_cell(face), neighbor = f.mesh->neighbor_cell(face);
            const auto a = owned(owner), b = owned(neighbor);
            if (a || b)
            {
                owned_contacts += a.has_value();
                stage->fourier_faces.push_back({f.mesh->face_geometry_global_id(face), a, b, 0, 0, 0, conductance,
                    f.mesh->cell_global_id(owner), f.mesh->cell_global_id(neighbor)});
            }
        }
        EXPECT_EQ(f.sum(owned_contacts), 1);
        const std::array<thermal::Stage, 1> stages{stage};
        auto guess = recovered;
        ASSERT_NO_THROW(thermal::correct(*f.mesh, 1., R::PhaseThermalMode::Equilibrium, stages, guess));
        expect_same(guess, recovered);
    }
}
TEST(ReactingMixtureFaceCouplingTest, LaterDonorFailurePreservesAcceptedStateAndEarlierValidTrial)
{
    Fixture f;
    auto options = f.options();
    options.face_transport = {f.contacts(.125)};
    Controller controller(f.flow, options);
    const auto before = f.flow->state();
    const auto retained = controller.preview(.125);
    EXPECT_THROW((void) controller.preview(100), std::exception);
    expect_same(f.flow->state(), before);
    EXPECT_NO_THROW(controller.commit(retained));
}
TEST(ReactingMixtureFaceCouplingTest, RawChemicalJouleSurvivesExternalAndFourierCancellation)
{
    Fixture f(true, true, 1);
    auto options = f.options();
    const auto mesh = f.mesh;
    const double huge = std::ldexp(1., 60);
    options.source = [mesh, huge](const R::CoupledReactionContext& context, double)
    {
        R::ReactionSourceInput result;
        result.integratedExtents = {{7, 1}};
        result.heat.integratedEnergy = mesh->cell_centroid(static_cast<LO>(context.owned_cell)).z < .5 ? -huge : huge;
        return result;
    };
    options.face_transport = {f.contacts(0, 8 * huge)};
    Controller controller(f.flow, options);
    const auto before = f.flow->state();
    const auto trial = controller.preview(.125);
    double increase = 0;
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& phase = trial.state().cells[c].phases[0];
        EXPECT_DOUBLE_EQ(phase.moles[0], 1);
        EXPECT_DOUBLE_EQ(phase.moles[1], 1);
        EXPECT_DOUBLE_EQ(phase.internal_energy - before.cells[c].phases[0].internal_energy, 1);
        EXPECT_DOUBLE_EQ(phase.temperature - before.cells[c].phases[0].temperature, 1 / capacity);
        increase += phase.internal_energy - before.cells[c].phases[0].internal_energy;
    }
    EXPECT_DOUBLE_EQ(f.sum(increase), 2);
    EXPECT_DOUBLE_EQ(trial.diagnostics().net_thermal_energy, 2);
    EXPECT_DOUBLE_EQ(trial.diagnostics().chemical_heat, 2);
    expect_same(f.flow->state(), before);
}
TEST(ReactingMixtureFaceCouplingTest, LossyCustomHeatAllocationRejectsAfterLaterFaceCancellation)
{
    Fixture f(true, true, 1);
    auto options = f.options();
    const auto mesh = f.mesh;
    const double huge = std::ldexp(1., 60);
    options.source = [mesh, huge](const R::CoupledReactionContext& context, double)
    {
        R::ReactionSourceInput result;
        result.integratedExtents = {{7, 1}};
        result.heat.integratedEnergy = mesh->cell_centroid(static_cast<LO>(context.owned_cell)).z < .5 ? -huge : huge;
        return result;
    };
    options.partition_heat = [](double rounded_heat, const auto&, const auto&)
    { return std::vector<double>{rounded_heat}; };
    options.face_transport = {f.contacts(0, 8 * huge)};
    Controller controller(f.flow, options);
    const auto before = f.flow->state();
    EXPECT_THROW((void) controller.preview(.125), std::exception);
    expect_same(f.flow->state(), before);
}

TEST(ReactingMixtureFaceCouplingTest, AcceptedRawReactionHeatSurvivesEqualRoundedExtentReports)
{
    Fixture f;
    const double huge = std::ldexp(1., 60), coefficient = std::ldexp(1., -60);
    f.mechanism = std::make_shared<const R::ReactionMechanism>(f.catalog,
        std::vector<R::ReactionDefinition>{{7, "normalized forward reaction",
                                               {{first, R::SpeciesPhase::Dissolved, -coefficient, {}},
                                                   {second, R::SpeciesPhase::Dissolved, coefficient, {}}},
                                               1},
            {8, "normalized opposing reaction",
                {{first, R::SpeciesPhase::Dissolved, coefficient, {}},
                    {second, R::SpeciesPhase::Dissolved, -coefficient, {}}},
                -1}},
        R::EnergyConvention::SensibleEnergy);
    auto initial = f.flow->state();
    for (auto& cell : initial.cells)
        cell.phases[0].moles = {2, 2, 1};
    f.flow->initialize(std::move(initial));
    auto options = f.options();
    options.integration.maximumSubstep = .5;
    options.source = [huge](const R::CoupledReactionContext& context, double)
    {
        R::ReactionSourceInput input;
        R::ReactionRateProvider provider;
        provider.identity = "independent normalized two-substep heat oracle";
        provider.schemaRevision = context.reaction.catalog.revision();
        provider.schemaFingerprint = context.reaction.catalog.fingerprint();
        provider.validityDescription = "Manufactured dyadic SSPRK2 stages with finite A/B donors.";
        provider.parameterUnits = "mol reaction/(m3 s)";
        provider.validate = [](const R::ReactionContext&) {};
        provider.evaluate = [huge](const R::ReactionContext& current)
        {
            const double time = current.environment.time, volume = current.environment.bulkVolume;
            const double forward = time == 0 ? 4 * huge : time == 1 ? 4 : 0;
            const double opposing = time == 0 ? 4 * huge : 0;
            return std::vector<R::ReactionRate>{{7, forward / volume, R::SourceVolumeBasis::BulkVolume},
                {8, opposing / volume, R::SourceVolumeBasis::BulkVolume}};
        };
        input.rates = std::move(provider);
        return input;
    };
    Controller controller(f.flow, options);
    const auto before = f.flow->state();
    const auto trial = controller.preview(1);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& ledger = trial.local_sources()[c];
        ASSERT_EQ(ledger.reactionExtents.size(), 2u);
        EXPECT_DOUBLE_EQ(ledger.reactionExtents[0], huge);
        EXPECT_DOUBLE_EQ(ledger.reactionExtents[1], huge);
        EXPECT_DOUBLE_EQ(ledger.reactionHeat, 1);
        const auto& phase = trial.state().cells[c].phases[0];
        EXPECT_DOUBLE_EQ(phase.internal_energy, 1);
        EXPECT_DOUBLE_EQ(phase.temperature, 300 + 1 / capacity);
    }
    EXPECT_DOUBLE_EQ(trial.diagnostics().chemical_heat, 2);
    EXPECT_DOUBLE_EQ(trial.diagnostics().net_thermal_energy, 2);
    expect_same(f.flow->state(), before);
}
} // namespace
