/** Flow/controller ownership of configured local interphase thermal events. */
#include "ConditionedThermalTestSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace Support = Testing::ConditionedTransfer;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using LO = Flow::Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Fixture : Support::Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<Flow> flow;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    int kinetic_calls = 0, number_calls = 0, heat_calls = 0;
    explicit Fixture(double c_v = 0, double c_g = 0, bool single_cell = false) : Support::Fixture(c_v, c_g)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(
                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, single_cell ? ArrReal{0, 1} : ArrReal{0, .5, 1}}),
            distribution);
        mechanism = std::make_shared<const R::ReactionMechanism>(
            catalog, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::IncludesFormationEnergy);
        Flow::Options controls;
        controls.minimum_pressure = 50000;
        controls.maximum_pressure = 1000000;
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "vapor"},
            {Flow::PhaseRole::BulkGas, "noncondensable"}};
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas, gas}, controls);
        Flow::State initial;
        initial.thermodynamic_pressure = Support::p_old;
        initial.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const double volume = mesh->cell_volume(static_cast<LO>(c));
            initial.cells[c].phases = before;
            for (auto& phase : initial.cells[c].phases)
            {
                for (auto& n : phase.moles)
                    n *= volume;
                phase.internal_energy *= volume;
            }
            initial.cells[c].phase_numbers = {std::nullopt, 0, std::nullopt};
        }
        flow->initialize(std::move(initial));
    }
    R::ReactingMixtureOptions options(double heat_per_volume = 0)
    {
        R::ReactingMixtureOptions options;
        options.mechanism = mechanism;
        options.source_identity = "independent-phase-transfer";
        options.phases = {{R::SpeciesPhase::Liquid, {}, {R::SpeciesPhase::Liquid, R::SpeciesPhase::Dissolved}},
            {R::SpeciesPhase::Dispersed, "vapor", {}}, {R::SpeciesPhase::Resolved, {}, {}}};
        options.source = [this, heat_per_volume](const auto& context, double)
        {
            ++heat_calls;
            R::ReactionSourceInput input;
            input.heat.integratedEnergy = heat_per_volume * context.reaction.environment.bulkVolume;
            return input;
        };
        options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(
            R::MixtureInterphaseEventProvider{{"frozen-all-solvent-transfer", 1, catalog->fingerprint()}, configuration,
                [this](const R::MixtureInterphaseContext& context)
                {
                    ++kinetic_calls;
                    const auto local = static_cast<LO>(context.owned_cell);
                    const auto center = mesh->cell_centroid(local);
                    if (!context.location ||
                        context.location->geometryCellId !=
                            static_cast<std::uint64_t>(mesh->cell_geometry_global_id(local)) ||
                        context.location->geometryEpoch != mesh->geometry_epoch() ||
                        context.location->ownedCell != context.owned_cell ||
                        context.location->centroid != std::array<double, 3>{center.x, center.y, center.z} ||
                        context.accepted_time != flow->state().time)
                        throw std::logic_error(
                            "Interphase source received a stale or noncanonical spatial/time frame.");
                    return std::vector<std::vector<double>>{
                        {context.post_faces.amount(context.catalog, 44, R::SpeciesPhase::Liquid)}};
                }});
        options.population_events = std::make_shared<const R::MixturePopulationEventProvider>(
            R::MixturePopulationEventProvider{"explicit-new-vapor-count", 1, catalog->fingerprint(),
                [this](const R::MixturePopulationEventContext& context)
                {
                    ++number_calls;
                    std::vector<BubbleGasPhysics::PopulationNumberEvent> events;
                    if (context.post_material.amount(context.catalog, 44, R::SpeciesPhase::Dispersed, "vapor") > 0 &&
                        *context.transported.phases[1].population_number == 0)
                        events.push_back({"vapor", 1, 0});
                    return events;
                }});
        return options;
    }
};
void same(const Flow::State& a, const Flow::State& b)
{
    EXPECT_EQ(a.revision, b.revision);
    EXPECT_DOUBLE_EQ(a.time, b.time);
    EXPECT_DOUBLE_EQ(a.thermodynamic_pressure, b.thermodynamic_pressure);
    ASSERT_EQ(a.cells.size(), b.cells.size());
    for (size_t c = 0; c < a.cells.size(); ++c)
    {
        EXPECT_EQ(a.cells[c].phase_numbers, b.cells[c].phase_numbers);
        EXPECT_DOUBLE_EQ(a.cells[c].mechanical_pressure, b.cells[c].mechanical_pressure);
        EXPECT_EQ(a.cells[c].drift_velocity.size(), b.cells[c].drift_velocity.size());
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_DOUBLE_EQ(a.cells[c].velocity.component(i), b.cells[c].velocity.component(i));
            for (size_t p = 0; p < a.cells[c].drift_velocity.size(); ++p)
                EXPECT_DOUBLE_EQ(a.cells[c].drift_velocity[p].component(i), b.cells[c].drift_velocity[p].component(i));
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
TEST(ReactingMixtureInterphaseTransferTest, CompleteLiquidToEmptyVaporHasIndependentPressureAndThermalReceipt)
{
    Fixture f;
    const auto accepted = f.flow->state();
    Coupler controller(f.flow, f.options());
    const auto trial = controller.preview(.01);
    EXPECT_NEAR(trial.state().thermodynamic_pressure, 140000, 2e-9 * 140000);
    const double expected_t = 140000 / (200 + .4 * Support::gas_constant);
    EXPECT_EQ(trial.final_thermal_evaluations().size(), accepted.cells.size());
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[0].internal_energy, 0);
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[0].temperature, 0);
        EXPECT_NEAR(trial.state().cells[c].phases[1].temperature, expected_t, 2e-9 * expected_t);
        EXPECT_EQ(trial.state().cells[c].phases[2].moles, accepted.cells[c].phases[2].moles);
        EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[1], 1);
        EXPECT_DOUBLE_EQ(*trial.transport_snapshot().cells[c].phases[1].population_number, 0);
        const auto& evaluation = trial.final_thermal_evaluations()[c];
        EXPECT_NEAR(evaluation.total_volume, f.mesh->cell_volume(static_cast<LO>(c)),
            2e-10 * f.mesh->cell_volume(static_cast<LO>(c)));
        EXPECT_EQ(evaluation.phases[1].moles, trial.state().cells[c].phases[1].moles);
        EXPECT_DOUBLE_EQ(evaluation.phases[1].internal_energy, trial.state().cells[c].phases[1].internal_energy);
    }
    EXPECT_NEAR(trial.diagnostics().net_thermal_energy, 0, 1e-7);
    same(f.flow->state(), accepted);
    const int expected_calls = int(accepted.cells.size()) * trial.diagnostics().flow.source_evaluations;
    EXPECT_EQ(f.kinetic_calls, expected_calls);
    EXPECT_EQ(f.number_calls, expected_calls);
    EXPECT_EQ(f.heat_calls, expected_calls);
    if (!accepted.cells.empty())
        EXPECT_GT(f.constitutive_calls, f.kinetic_calls);
    const int kinetics = f.kinetic_calls, properties = f.constitutive_calls, numbers = f.number_calls;
    controller.commit(trial);
    EXPECT_EQ(f.kinetic_calls, kinetics);
    EXPECT_EQ(f.constitutive_calls, properties);
    EXPECT_EQ(f.number_calls, numbers);
}
TEST(ReactingMixtureInterphaseTransferTest, StrongIndependentHeatingAndPerSpeciesReferencesPreservePhysicalOracle)
{
    for (int shifted : {0, 1})
    {
        Fixture f(shifted ? 1e6 : 0, shifted ? -3e5 : 0);
        Coupler controller(f.flow, f.options(.5 * f.physical_U0));
        const auto trial = controller.preview(.01);
        EXPECT_NEAR(trial.state().thermodynamic_pressure, 210000, 2e-9 * 210000);
        for (const auto& cell : trial.state().cells)
            EXPECT_NEAR(cell.phases[1].temperature, 210000 / (200 + .4 * Support::gas_constant), 2e-9 * 1200);
    }
}
TEST(ReactingMixtureInterphaseTransferTest, FrozenReceiptReplayAndLaterHeatInfeasibilityPreserveEarlierTrial)
{
    Fixture f;
    auto options = f.options();
    options.source = [&](const auto& context, double dt)
    {
        ++f.heat_calls;
        R::ReactionSourceInput input;
        input.heat.integratedEnergy = (dt > .015 ? 4 * f.physical_U0 : 0) * context.reaction.environment.bulkVolume;
        return input;
    };
    Coupler controller(f.flow, options);
    const auto saved = controller.checkpoint();
    const auto accepted = f.flow->state();
    const auto first = controller.preview(.01);
    EXPECT_THROW((void) controller.preview(.02), std::exception);
    same(f.flow->state(), accepted);
    const int calls = f.kinetic_calls, properties = f.constitutive_calls, numbers = f.number_calls;
    controller.commit(first);
    EXPECT_EQ(f.kinetic_calls, calls);
    EXPECT_EQ(f.constitutive_calls, properties);
    EXPECT_EQ(f.number_calls, numbers);
    const auto expected = f.flow->state();
    controller.restore(saved);
    EXPECT_EQ(f.kinetic_calls, calls);
    EXPECT_EQ(f.constitutive_calls, properties);
    same(f.flow->state(), accepted);
    const auto replay = controller.preview(.01);
    same(replay.state(), expected);
}
TEST(ReactingMixtureInterphaseTransferTest, RankLocalKineticsFailureIsFencedBeforePressureAndRetainsEarlierTrial)
{
    Fixture f;
    auto options = f.options();
    const auto comm = f.mesh->owned_cell_map()->getComm();
    options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(R::MixtureInterphaseEventProvider{
        {"local-event-failure", 1, f.catalog->fingerprint()}, f.configuration, [&](const auto& context)
        {
            ++f.kinetic_calls;
            if (comm->getRank() == 0 && context.time_step > .015)
                throw std::runtime_error("Injected local kinetics failure.");
            return std::vector<std::vector<double>>{
                {context.post_faces.amount(context.catalog, 44, R::SpeciesPhase::Liquid)}};
        }});
    Coupler controller(f.flow, options);
    const auto accepted = f.flow->state();
    const auto valid = controller.preview(.01);
    EXPECT_THROW((void) controller.preview(.02), std::exception);
    same(f.flow->state(), accepted);
    EXPECT_NO_THROW(controller.commit(valid));
}
TEST(ReactingMixtureInterphaseTransferTest, MissingExplicitVaporBirthRejectsWithoutInferringCount)
{
    Fixture f;
    auto options = f.options();
    options.population_events.reset();
    Coupler controller(f.flow, options);
    const auto accepted = f.flow->state();
    EXPECT_THROW((void) controller.preview(.01), std::exception);
    same(f.flow->state(), accepted);
}
TEST(ReactingMixtureInterphaseTransferTest, ConfiguredProviderAndSlotsRejectRankDivergenceIncludingActualEmptyOwners)
{
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    int empty = f.mesh->num_owned_cells() == 0, total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total);
    if (comm->getSize() == 4)
        EXPECT_EQ(total, 2);
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        auto options = f.options();
        const bool change = comm->getRank() == comm->getSize() - 1;
        auto configuration = *f.configuration;
        auto metadata = options.interphase->metadata;
        if (change && scenario == 1)
            metadata.identity = "other-valid-event-law";
        if (change && scenario == 2)
            metadata.revision = 2;
        if (change && scenario == 3)
            configuration.stages[0].identity = "different-valid-stage";
        if (change && scenario == 4)
            metadata.schema_fingerprint++;
        if (comm->getSize() == 1)
            metadata.schema_fingerprint++;
        options.interphase =
            std::make_shared<const R::MixtureInterphaseEventProvider>(R::MixtureInterphaseEventProvider{metadata,
                std::make_shared<const R::InterphaseConfiguration>(configuration), options.interphase->integrate});
        if (change && scenario == 0 && comm->getSize() > 1)
            options.interphase.reset();
        EXPECT_THROW((void) Coupler(f.flow, options), std::exception);
    }
    EXPECT_EQ(f.kinetic_calls, 0);
    EXPECT_EQ(f.number_calls, 0);
}
TEST(ReactingMixtureInterphaseTransferTest,
    RawConditionedHeatPlanConvergesEvenWhenStaticPacketsAndGrossChannelsAreUnchanged)
{
    Fixture f;
    auto configuration = *f.configuration;
    configuration.stages = {{"fixed-internal-heat-slots", R::InterphaseStageKind::Heat, {}, {{0, 2}}}};
    const auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    int callbacks = 0;
    const auto provider = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "raw-heat-plan-convergence", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++callbacks;
            Flow::SourceStageResult result;
            result.sources.resize(context.transport.cells.size());
            auto batch = std::make_shared<R::PressureConditionedThermalBatch>();
            batch->configuration = shared;
            const auto& t = context.transport;
            batch->source = {t.stage.provider_identity, t.stage.provider_revision, t.stage.invocation,
                static_cast<std::uint64_t>(t.stage.corrector), t.schema_fingerprint, t.accepted_generation,
                t.old_geometry_epoch, t.new_geometry_epoch};
            for (size_t c = 0; c < t.cells.size(); ++c)
            {
                auto phases = context.thermodynamic_trial.cells[c].phases;
                for (size_t a = 0; a < phases.size(); ++a)
                    phases[a].moles = t.cells[c].phases[a].moles;
                result.sources[c].internal_energy.assign(3, 0);
                batch->cells.push_back(
                    R::compile_interphase_plan(shared, phases, {{t.stage.corrector == 0 ? 1. : 2.}}, {{}, {}, {}}, {}));
            }
            // Deliberately no generic gross channel: the conditioned plan itself must be observed.
            result.thermal_receipt = std::move(batch);
            return result;
        }});
    const auto accepted = f.flow->state();
    const auto trial = f.flow->preview_with_sources(.01, provider);
    EXPECT_EQ(callbacks, 3);
    EXPECT_EQ(trial.diagnostics().source_evaluations, 3);
    EXPECT_LE(trial.diagnostics().maximum_conditioned_thermal_residual, 1);
    same(f.flow->state(), accepted);
}
TEST(
    ReactingMixtureInterphaseTransferTest, RealConstitutiveFailureInsidePressureDoesNotReintegrateSourceAndRetainsTrial)
{
    Fixture f;
    auto configuration = *f.configuration;
    const auto original = configuration.phase_enthalpy[0];
    configuration.phase_enthalpy[0] = std::make_shared<const R::PartialMolarEnthalpyLaw>(R::PartialMolarEnthalpyLaw{
        original->metadata, original->bounds, [original](double t, double p, std::span<const double> x)
        {
            if (p > 120000)
                throw std::logic_error("Real constitutive failure during pressure root.");
            return original->evaluate(t, p, x);
        }});
    auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    auto options = f.options();
    options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(R::MixtureInterphaseEventProvider{
        {"no-rate-reentry", 1, f.catalog->fingerprint()}, shared, [&](const auto& context)
        {
            ++f.kinetic_calls;
            return std::vector<std::vector<double>>{
                {context.time_step > .015 ? context.post_faces.amount(context.catalog, 44, R::SpeciesPhase::Liquid)
                                          : 0}};
        }});
    Coupler controller(f.flow, options);
    const auto accepted = f.flow->state();
    const auto retained = controller.preview(.01);
    const int kinetics = f.kinetic_calls, numbers = f.number_calls;
    EXPECT_THROW((void) controller.preview(.02), std::exception);
    EXPECT_EQ(f.kinetic_calls, kinetics + int(accepted.cells.size()));
    EXPECT_EQ(f.number_calls, numbers + int(accepted.cells.size()));
    same(f.flow->state(), accepted);
    EXPECT_NO_THROW(controller.commit(retained));
}
TEST(ReactingMixtureInterphaseTransferTest, WholePlanCertificateCanProveEmptyWithoutReusingBareAffineEOSCapability)
{
    Fixture f;
    auto configuration = *f.configuration;
    int domain_calls = 0;
    configuration.caloric_domain = std::make_shared<const R::ConditionedCaloricDomainProvider>(
        R::ConditionedCaloricDomainProvider{{"analytic-full-final-gas-certificate", 1, f.catalog->fingerprint()},
            [&](const R::ConditionedCaloricQuery& query)
            {
                ++domain_calls;
                long double target = 0, amount = 0, volume = 0;
                for (size_t a = 0; a < query.plan.final_moles.size(); ++a)
                {
                    target += query.advected_energy[a];
                    volume += query.advected_volumes[a];
                    for (auto e : query.plan.raw_static_energy[a])
                        target += e;
                    for (auto n : query.plan.final_moles[a])
                        amount += n;
                }
                target += (query.reference_pressure.lower + query.mechanical_offset) * volume;
                const long double final_maximum = amount * Support::cp * 1200;
                if (target > final_maximum)
                    return R::CaloricPressureDomain{R::CaloricDomainKnowledge::CertifiedEmpty, std::nullopt};
                return R::CaloricPressureDomain{};
            }});
    auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    auto options = f.options(4 * f.physical_U0);
    options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(R::MixtureInterphaseEventProvider{
        {"certificate-amounts-frozen", 1, f.catalog->fingerprint()}, shared, [&](const auto& context)
        {
            ++f.kinetic_calls;
            return std::vector<std::vector<double>>{
                {context.post_faces.amount(context.catalog, 44, R::SpeciesPhase::Liquid)}};
        }});
    Coupler controller(f.flow, options);
    const auto accepted = f.flow->state();
    EXPECT_THROW((void) controller.preview(.01), std::exception);
    same(f.flow->state(), accepted);
    EXPECT_EQ(f.kinetic_calls, int(accepted.cells.size()));
    EXPECT_EQ(f.number_calls, int(accepted.cells.size()));
    EXPECT_EQ(domain_calls, int(accepted.cells.size()));
}

TEST(ReactingMixtureInterphaseTransferTest, MovingCompleteTransferClosesIndependentWorkAndReplaysWithoutCallbacks)
{
    Fixture f(0, 0, true);
    constexpr double target = 1.01, dt = .1;
    const double expected_pressure = (Support::gas_constant / Support::cv) * f.physical_U0 /
                                     (target + Support::gas_constant / Support::cv * (target - 1));
    const double expected_temperature = expected_pressure * target / ((f.nl + f.ng) * Support::gas_constant);
    const double expected_work = -expected_pressure * (target - 1);
    auto options = f.options();
    int target_calls = 0;
    options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(R::MixturePlanarMotionProvider{
        f.mesh, {}, "prescribed-expansion-with-complete-transfer", 1, [&target_calls](double, double)
        {
            ++target_calls;
            return target;
        }});
    auto fail = std::make_shared<bool>(false);
    const auto law = options.interphase;
    options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(R::MixtureInterphaseEventProvider{
        law->metadata, law->configuration, [law, fail](const R::MixtureInterphaseContext& context)
        {
            if (*fail)
                throw std::runtime_error("Manufactured interphase failure inside actual moving geometry.");
            return law->integrate(context);
        }});
    Coupler controller(f.flow, std::move(options));
    const auto accepted = f.flow->state();
    const auto checkpoint = controller.checkpoint();
    const auto trial = controller.preview(dt);
    EXPECT_EQ(target_calls, 1);
    EXPECT_NEAR(trial.state().thermodynamic_pressure, expected_pressure, 2e-9 * expected_pressure);
    EXPECT_NEAR(trial.diagnostics().flow.pressure_work, expected_work, 2e-9 * std::abs(expected_work));
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), 1.);
        EXPECT_DOUBLE_EQ(trial.transport_snapshot().old_control_volumes[c], 1.);
        EXPECT_DOUBLE_EQ(trial.transport_snapshot().new_control_volumes[c], target);
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[0].internal_energy, 0.);
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[0].temperature, 0.);
        EXPECT_NEAR(trial.state().cells[c].phases[1].temperature, expected_temperature, 2e-9 * expected_temperature);
        EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[1], 1.);
        double energy = 0;
        for (const auto& phase : trial.state().cells[c].phases)
            energy += phase.internal_energy;
        EXPECT_NEAR(energy, f.physical_U0 + expected_work, 2e-9 * f.physical_U0);
    }
    *fail = true;
    EXPECT_ANY_THROW((void) controller.preview(dt / 2));
    same(f.flow->state(), accepted);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
        EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), 1.);
    const std::array counts{target_calls, f.kinetic_calls, f.heat_calls, f.number_calls, f.constitutive_calls};
    controller.commit(trial); // The now-failing law is never called during commit.
    EXPECT_EQ((std::array{target_calls, f.kinetic_calls, f.heat_calls, f.number_calls, f.constitutive_calls}), counts);
    const auto committed = f.flow->state();
    for (size_t c = 0; c < accepted.cells.size(); ++c)
        EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), target);
    controller.restore(checkpoint);
    EXPECT_EQ((std::array{target_calls, f.kinetic_calls, f.heat_calls, f.number_calls, f.constitutive_calls}), counts);
    same(f.flow->state(), accepted);
    *fail = false;
    const auto replay = controller.preview(dt);
    controller.commit(replay);
    same(f.flow->state(), committed);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    int empty = f.mesh->num_owned_cells() == 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, comm->getSize() - 1);
}
} // namespace
