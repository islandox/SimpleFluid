/** Public-controller conservation oracles for ordered mixture face sources. */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>

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
    explicit Fixture(bool thermal = false, bool reaction_everywhere = false, double heat = 0)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
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
            const bool lower = mesh->cell_centroid(static_cast<LO>(c)).z < .5;
            const double n = lower || reaction_everywhere ? 2 : 0;
            const double t = thermal && !lower ? 301 : 300;
            state.cells[c].phases = {{liquid, {n, 0, 5 - n}, 100000, capacity * (t - 300), t}};
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
