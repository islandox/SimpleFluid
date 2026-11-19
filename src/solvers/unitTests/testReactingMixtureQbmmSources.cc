/** Conservative fixed-mesh QBMM source integration with an analytic, closed-cell caloric oracle. */
#include "ConditionedThermalTestSupport.hh"
#include "QbmmRestartTestSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <numeric>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace Thermo = Testing::ConditionedTransfer;
namespace RestartIO = Testing::QbmmRestart;
using Coupler = R::ReactingMixtureCoupler;
using Flow = R::LowMachMixtureSolver;
using Ops = std::vector<R::QbmmEventOperation>;
using Law = std::function<Ops(const R::MixtureQbmmContext&)>;
using Kind = R::QbmmEventKind;
using Phase = R::SpeciesPhase;
using Shape = R::QbmmShapeState;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double scale = 1. / 128; // kg/bubble. The pure family species weighs 1 kg/mol.
constexpr double liquid_moles = .375, coarse_moles = .125;
constexpr double liquid_u0 = 2 * (Thermo::cv / Thermo::gas_constant) * Thermo::p_old;
R::QbmmMaterialTransfer from_liquid(double scaled_mass)
{
    return {R::QbmmPhaseExchange{Phase::Liquid}, {scale * scaled_mass, 0}};
}

struct Fixture : Thermo::Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<Flow> flow;
    std::shared_ptr<const R::QbmmPopulationSchema> family;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<const R::InterphaseConfiguration> slots;
    int law_calls = 0;

    explicit Fixture(std::vector<R::QbmmAtom> seed = {{4, 1}}, bool moving = false, bool bind_coarse = false,
        R::EnergyConvention convention = R::EnergyConvention::IncludesFormationEnergy,
        R::QbmmClosureVersions policies = {}, double liquid_stock = liquid_moles)
        : Thermo::Fixture(0, 0, convention)
    {
        Flow::Mesh::DistributionOptions partition;
        partition.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(
                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, moving ? ArrReal{0, 1, 2} : ArrReal{0, 1}}),
            partition);
        Flow::Options controls;
        controls.minimum_pressure = 50000;
        controls.maximum_pressure = 1000000;
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "vapor"},
            {Flow::PhaseRole::BulkGas, "inert"}, {Flow::PhaseRole::DispersedGas, "coarse"}};
        family = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "vapor",
            controls.phase_roles[1], scale, std::vector<R::QbmmMassFraction>{{44, 1}, {709, 0}}, std::move(policies));
        controls.qbmm_families = {nullptr, family, nullptr, nullptr};
        if (bind_coarse)
            controls.qbmm_families[3] = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(),
                "coarse", controls.phase_roles[3], scale, std::vector<R::QbmmMassFraction>{{44, 1}, {709, 0}});
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas, gas, gas}, controls);
        mechanism = std::make_shared<const R::ReactionMechanism>(
            catalog, std::vector<R::ReactionDefinition>{}, gas->energy_convention());

        // The family and coarse gases displace inert gas at the same initial p,T.
        double count = 0, vapor = 0;
        Shape shape{};
        for (const auto [weight, x] : seed)
        {
            count += weight;
            vapor += weight * x * scale;
            shape.H2 += weight * x * x;
            shape.H3 += weight * x * x * x;
        }
        const double inert =
            (1 - liquid_stock) * Thermo::p_old / (Thermo::gas_constant * Thermo::T0) - vapor - coarse_moles;
        Flow::State state;
        state.thermodynamic_pressure = Thermo::p_old;
        state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*family), std::nullopt, std::nullopt};
        if (bind_coarse)
            state.qbmm_families[3] = Flow::qbmm_family_stamp(*controls.qbmm_families[3]);
        state.cells.resize(mesh->num_owned_cells());
        for (auto& cell : state.cells)
        {
            cell.phases = {
                {liquid, {liquid_stock, 0}, Thermo::p_old, liquid_stock * liquid_u0, liquid_stock ? Thermo::T0 : 0},
                {gas, {vapor, 0}, Thermo::p_old, vapor * Thermo::cv * Thermo::T0, vapor ? Thermo::T0 : 0},
                {gas, {0, inert}, Thermo::p_old, inert * Thermo::cv * Thermo::T0, Thermo::T0},
                {gas, {coarse_moles, 0}, Thermo::p_old, coarse_moles * Thermo::cv * Thermo::T0, Thermo::T0}};
            cell.phase_numbers = {std::nullopt, count, std::nullopt, 2.};
            cell.qbmm_shape = {std::nullopt, shape, std::nullopt, std::nullopt};
            if (bind_coarse)
                cell.qbmm_shape[3] = Shape{128, 1024}; // Two bubbles at coordinate eight.
            if (moving)
                cell.velocity = Flow::Vector{0, 0, .125};
        }
        flow->initialize(state);
        auto config = *configuration;
        config.stages = {{"vapor-exchange", R::InterphaseStageKind::Material, {{44, 0, 1}, {44, 1, 0}}, {}}};
        if (bind_coarse)
            config.stages.front().material.push_back({44, 0, 3});
        config.phase_enthalpy.resize(4);
        config.phase_enthalpy[1] = std::make_shared<const R::PartialMolarEnthalpyLaw>(R::PartialMolarEnthalpyLaw{
            {"ideal-gas-partials", 1, catalog->fingerprint()}, gas->bounds(),
            [this](double t, double, std::span<const double>)
            {
                ++constitutive_calls;
                return R::PartialMolarEnthalpyPoint{{Thermo::cp * t, Thermo::cp * t}, {Thermo::cp, Thermo::cp}, {0, 0}};
            }});
        slots = std::make_shared<const R::InterphaseConfiguration>(std::move(config));
    }

    R::ReactingMixtureOptions options(Law law = {}, bool thermal = false)
    {
        R::ReactingMixtureOptions options;
        options.mechanism = mechanism;
        options.phases = {{Phase::Liquid, {}, {Phase::Liquid, Phase::Dissolved}}, {Phase::Dispersed, "vapor", {}},
            {Phase::Resolved, {}, {}}, {Phase::Dispersed, "coarse", {}}};
        if (law)
            options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{
                {"constructive", 1, catalog->fingerprint()}, thermal ? slots : nullptr,
                [this, law](const R::MixtureQbmmContext& input)
                {
                    ++law_calls;
                    EXPECT_EQ(input.family.fingerprint(), flow->options().qbmm_families.at(input.phase)->fingerprint());
                    EXPECT_EQ(input.accepted_time, flow->state().time);
                    const double n = input.material.amount(*catalog, 44, Phase::Dispersed, input.family.population());
                    const auto& phase = input.phases[input.phase];
                    const double volume = n * Thermo::gas_constant * phase.temperature / phase.absolute_pressure;
                    EXPECT_NEAR(input.phase_volumes[input.phase], volume, 2e-11 * std::max(1., volume));
                    return law(input);
                }});
        return options;
    }
};

TEST(ReactingMixtureQbmmSourcesTest, DeclaredRangesDistinguishMissingLiquidOutsidePressureAndActualDuration)
{
    using Variable = R::QbmmRangeVariable;
    using Status = R::QbmmRangeStatus;
    for (const auto gate : {R::QbmmApplicabilityGate::ReportOnly, R::QbmmApplicabilityGate::RejectOutside})
    {
        R::QbmmClosureVersions policies;
        policies.laws = {{"area-transfer", 2, R::QbmmLawKind::MassTransfer, "m2;mol/m3;s", "kg/cell", gate, true,
            {{Variable::CarrierLiquidConcentrationMolPerM3, 0, 1, 0},
                {Variable::AbsolutePressurePa, 110000, 200000, {}}, {Variable::TimeStepS, .025, .025, {}},
                {Variable::TemperatureK, 299, 301, {}}, {Variable::DiameterM, 0, 4, {}}}}};
        Fixture f({{2, 1}, {1, 3}}, false, false, R::EnergyConvention::IncludesFormationEnergy, policies, 0);
        Coupler controller(f.flow, f.options(
                                       [](const R::MixtureQbmmContext& ctx)
                                       {
                                           EXPECT_NE(ctx.law_inputs, nullptr);
                                           const auto& inputs = *ctx.law_inputs;
                                           EXPECT_FALSE(inputs.carrier_liquid_concentration_mol_per_m3);
                                           EXPECT_FALSE(inputs.carrier_liquid_temperature_K);
                                           EXPECT_EQ(inputs.time_step_s, .025);
                                           EXPECT_EQ(inputs.quadrature.active_atoms, 2u);
                                           EXPECT_GT(inputs.geometry->area_density, 0.0);
                                           EXPECT_NEAR(inputs.phase_concentration_mol_per_m3->at(0),
                                               Thermo::p_old / (Thermo::gas_constant * Thermo::T0), 1e-10);
                                           return Ops{};
                                       }));
        const auto saved = controller.checkpoint();
        const auto report = controller.qbmm_report();
        ASSERT_EQ(report.families[1]->applicability_counts.size(), 1u);
        EXPECT_EQ(report.families[1]->applicability_counts.front()[1], 1u);
        if (gate == R::QbmmApplicabilityGate::RejectOutside)
        {
            EXPECT_THROW((void) controller.preview(.025), std::exception);
            EXPECT_EQ(f.law_calls, 0);
            EXPECT_EQ(f.flow->state().time, 0.0);
            continue;
        }
        const auto trial = controller.preview(.025);
        const auto endpoint = controller.qbmm_report(trial);
        for (const auto& cell : endpoint.cells)
        {
            const auto& family = *cell.families.at(1);
            const auto& result = family.applicability.at(0);
            EXPECT_TRUE(family.closure_evaluated);
            EXPECT_FALSE(family.closure_validated);
            EXPECT_FALSE(result.applicable);
            EXPECT_TRUE(result.gate_passed);
            EXPECT_EQ(result.ranges[0].status, Status::MissingInput);
            EXPECT_EQ(result.ranges[1].status, Status::OutsideRange);
            EXPECT_EQ(result.ranges[2].status, Status::WithinRange);
            EXPECT_EQ(result.ranges[3].status, Status::WithinRange);
            EXPECT_EQ(result.ranges[4].status, Status::WithinRange);
        }
        const auto evaluated = f.law_calls;
        controller.commit(trial);
        controller.restore(saved);
        EXPECT_EQ(f.law_calls, evaluated);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, NativeTransportRejectsMissingLiquidAtEntryBeforePopulationCallbacks)
{
    using Variable = R::QbmmRangeVariable;
    R::QbmmClosureVersions policies;
    policies.laws = {{"liquid-property", 1, R::QbmmLawKind::TransportProperty, "physical liquid EOS", "Pa.s",
        R::QbmmApplicabilityGate::RejectOutside, true,
        {{Variable::CarrierLiquidConcentrationMolPerM3, 0, 1, 0},
            {Variable::CarrierLiquidTemperatureK, 299, 301, {}}}}};
    Fixture f({{4, 1}}, false, false, R::EnergyConvention::IncludesFormationEnergy, policies, 0);
    const auto report = f.flow->qbmm_report();
    for (const auto& cell : report.cells)
    {
        const auto& applicability = cell.families[1]->applicability.front();
        for (const auto& range : applicability.ranges)
        {
            EXPECT_EQ(range.status, R::QbmmRangeStatus::MissingInput);
            EXPECT_FALSE(range.observed_minimum);
        }
        EXPECT_FALSE(applicability.gate_passed);
    }
    const auto before = f.flow->serialize_qbmm_restart();
    EXPECT_THROW((void) f.flow->preview(.01), std::exception);
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), before);
    {
        Coupler controller(f.flow, f.options([](const auto&) { return Ops{}; }));
        const auto attached = f.flow->serialize_qbmm_restart();
        EXPECT_THROW((void) controller.preview(.01), std::exception);
        EXPECT_EQ(f.law_calls, 0);
        EXPECT_EQ(f.flow->serialize_qbmm_restart(), attached);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, TypedLawScopesCallbacksAndRestartToItsDeclaredFamily)
{
    constexpr double dt = .03125;
    for (const bool occupied : {true, false})
    {
        R::QbmmClosureVersions policies;
        const R::QbmmLawMetadata law{"occupied-population-law", 1, R::QbmmLawKind::Coalescence, "population and dt",
            "events/cell", R::QbmmApplicabilityGate::RejectOutside, true,
            {{R::QbmmRangeVariable::TimeStepS, dt, dt, {}}}};
        policies.laws = {law};
        const auto seed = occupied ? std::vector<R::QbmmAtom>{{4, 1}} : std::vector<R::QbmmAtom>{};
        Fixture f(seed, false, true, R::EnergyConvention::IncludesFormationEnergy, policies);
        int calls = 0;
        auto config = f.options();
        config.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{{"constructive", 1, f.catalog->fingerprint()}, nullptr, {},
                {std::make_shared<const R::MixtureQbmmLaw>(R::MixtureQbmmLaw{law, [&](const R::QbmmLawInputs& inputs)
                    {
                        ++calls;
                        EXPECT_EQ(inputs.population, "vapor");
                        EXPECT_EQ(inputs.time_step_s, dt);
                        EXPECT_EQ(inputs.quadrature.active_atoms, 1u);
                        return Ops{};
                    }})}});
        {
            Coupler controller(f.flow, config);
            const auto initialized = controller.checkpoint();
            EXPECT_EQ(calls, 0);
            if (occupied)
            {
                EXPECT_THROW((void) controller.preview(2 * dt), std::exception);
                EXPECT_EQ(calls, 0);
            }
            const auto trial = controller.preview(occupied ? dt : 2 * dt);
            const int evaluated =
                occupied ? int(f.mesh->num_owned_cells()) * trial.diagnostics().flow.source_evaluations : 0;
            EXPECT_EQ(calls, evaluated);
            controller.commit(trial);
            EXPECT_EQ(calls, evaluated);
            controller.restore(initialized);
            EXPECT_EQ(calls, evaluated);
            controller.commit(controller.preview(dt));
        }
        const auto saved_calls = calls;
        const auto archive = f.flow->serialize_qbmm_restart();
        auto restored = std::make_shared<Flow>(f.mesh, std::vector{f.liquid, f.gas, f.gas, f.gas}, f.flow->options());
        restored->restore_qbmm_restart(archive);
        EXPECT_THROW((void) Coupler(restored, f.options()), std::exception);
        Coupler matching(restored, config);
        EXPECT_EQ(calls, saved_calls);
        const auto continued = matching.preview(dt);
        EXPECT_EQ(continued.state().time, 2 * dt);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, BodyImpulseUsesEndMassAndDoesNotDuplicateReservoirImpulse)
{
    Fixture f;
    constexpr double feed_mass = 2 * scale, feed_impulse = .034, gravity = -2.7, dt = .03125;
    auto controls = f.options(
        [](const auto&)
        {
            return Ops{R::QbmmBirth{
                {{{1, 2}, {R::QbmmReservoirExchange{"feed", 1750, {feed_impulse, 0, 0}}, {feed_mass, 0}}}}}};
        });
    controls.body_acceleration = Flow::Vector{gravity, 0, 0};
    Coupler controller(f.flow, controls);
    const auto trial = controller.preview(dt);
    const double vapor = 4 * scale;
    const double inert =
        (1 - liquid_moles) * Thermo::p_old / (Thermo::gas_constant * Thermo::T0) - vapor - coarse_moles;
    const double mass = liquid_moles + vapor + coarse_moles + .028 * inert + feed_mass;
    EXPECT_NEAR(trial.diagnostics().flow.external_impulse.component(0), feed_impulse + mass * gravity * dt, 3e-15);
    EXPECT_EQ(trial.diagnostics().external_mass, feed_mass);
    EXPECT_EQ(trial.diagnostics().external_heat, 0);
}

TEST(ReactingMixtureQbmmSourcesTest, ClosedBirthGrowthShrinkageAndDisappearanceMatchAnalyticCaloricEndpoint)
{
    struct Case
    {
        double initial_count;
        R::QbmmEventOperation operation;
        Kind kind;
        double count, scaled_mass;
        Shape shape;
    };
    const std::vector<Case> cases{{0, R::QbmmBirth{{{{2, 1.5}, from_liquid(3)}}}, Kind::Birth, 2, 3, {4.5, 6.75}},
        {4, R::QbmmResize{0, 2, 2, from_liquid(2)}, Kind::Growth, 4, 6, {10, 18}},
        {4, R::QbmmResize{0, 2, .5, from_liquid(1)}, Kind::Shrinkage, 4, 3, {2.5, 2.25}},
        {4, R::QbmmDisappearance{0, 4, from_liquid(4)}, Kind::Disappearance, 0, 0, {0, 0}}};
    for (const auto& test : cases)
    {
        SCOPED_TRACE(static_cast<int>(test.kind));
        Fixture f(test.initial_count == 0 ? std::vector<R::QbmmAtom>{} : std::vector<R::QbmmAtom>{{4, 1}});
        Coupler controller(f.flow, f.options(
                                       [&](const auto& input)
                                       {
                                           EXPECT_EQ(input.quadrature.active_atoms, test.initial_count ? 1u : 0u);
                                           return Ops{test.operation};
                                       },
                                       true));
        const auto trial = controller.preview(.01);
        const double n0 = test.initial_count * scale, n1 = test.scaled_mass * scale;
        const double delta = n1 - n0, liquid = liquid_moles - delta;
        const double old_gas = (1 - liquid_moles) * Thermo::p_old / (Thermo::gas_constant * Thermo::T0);
        const double gas = old_gas + delta;
        const double U0 = liquid_moles * liquid_u0 + old_gas * Thermo::cv * Thermo::T0;
        const double t = (U0 - liquid * (liquid_u0 - 1000 * Thermo::T0)) / (1000 * liquid + Thermo::cv * gas);
        const double p = gas * Thermo::gas_constant * t / (1 - liquid);
        EXPECT_NEAR(trial.state().thermodynamic_pressure, p, 3e-9 * p);
        EXPECT_NEAR(trial.diagnostics().flow.pressure_work, 0, 2e-4);
        EXPECT_NEAR(trial.diagnostics().net_thermal_energy, 0, 1e-7);
        EXPECT_EQ(trial.diagnostics().external_mass, 0);
        EXPECT_EQ(trial.diagnostics().population_births.at(1), test.kind == Kind::Birth ? 2 : 0);
        EXPECT_EQ(trial.diagnostics().population_deaths.at(1), test.kind == Kind::Disappearance ? 4 : 0);
        ASSERT_TRUE(trial.interphase_receipt());
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& cell = trial.state().cells[c];
            ASSERT_TRUE(trial.local_qbmm_sources().at(c).at(1));
            const auto& receipt = *trial.local_qbmm_sources()[c][1];
            ASSERT_EQ(receipt.events.size(), 1u);
            EXPECT_EQ(receipt.events[0].kind, test.kind);
            EXPECT_EQ(receipt.events[0].duration, .01);
            EXPECT_EQ(receipt.family_fingerprint, f.family->fingerprint());
            EXPECT_EQ(receipt.final_moles, (std::vector<double>{n1, 0}));
            EXPECT_EQ(receipt.final_count, test.count);
            EXPECT_EQ(receipt.final_shape, test.shape);
            EXPECT_EQ(cell.phases[1].moles, receipt.final_moles);
            EXPECT_EQ(cell.qbmm_shape[1], test.shape);
            EXPECT_EQ(cell.phase_numbers[1], test.count);
            EXPECT_EQ(trial.transport_snapshot().cells[c].phases[1].population_number, test.initial_count);

            // S and G each have one atom per molecule. Check both inventories and their weighted total mass.
            EXPECT_EQ(cell.phases[0].moles, (std::vector<double>{liquid, 0}));
            EXPECT_EQ(cell.phases[3].moles, (std::vector<double>{coarse_moles, 0}));
            const double s = cell.phases[0].moles[0] + cell.phases[1].moles[0] + cell.phases[3].moles[0];
            const double g = cell.phases[2].moles[1], g0 = old_gas - n0 - coarse_moles;
            EXPECT_DOUBLE_EQ(s, liquid_moles + n0 + coarse_moles);
            EXPECT_DOUBLE_EQ(g, g0);
            EXPECT_NEAR(s + .028 * g, liquid_moles + n0 + coarse_moles + .028 * g0, 1e-14);
            const auto& plan = trial.interphase_receipt()->cells.at(c);
            EXPECT_EQ(plan.amounts.at(0), (std::vector<double>{std::max(delta, 0.), std::max(-delta, 0.)}));
            const auto& evaluation = trial.final_thermal_evaluations()[c];
            ASSERT_EQ(evaluation.material_events.size(), 1u);
            const size_t donor = delta > 0 ? 0 : 1;
            const double anchor = evaluation.anchors.front().temperature.at(donor);
            const double h = donor == 0 ? liquid_u0 + 1000 * (anchor - Thermo::T0) + p : Thermo::cp * anchor;
            EXPECT_NEAR(evaluation.material_events[0].carried_energy, std::abs(delta) * h, 2e-7 * std::abs(delta) * h);
            EXPECT_NEAR(evaluation.raw_material_enthalpy[1], delta * h, 2e-7 * std::abs(delta) * h);
            const std::array old_volumes{liquid_moles, n0 * Thermo::gas_constant * Thermo::T0 / Thermo::p_old,
                g0 * Thermo::gas_constant * Thermo::T0 / Thermo::p_old,
                coarse_moles * Thermo::gas_constant * Thermo::T0 / Thermo::p_old};
            const std::array final_volumes{liquid, n1 * Thermo::gas_constant * t / p, g0 * Thermo::gas_constant * t / p,
                coarse_moles * Thermo::gas_constant * t / p};
            double total_u = 0;
            for (size_t phase = 0; phase < 4; ++phase)
            {
                EXPECT_NEAR(cell.phases[phase].temperature, phase == 1 && !n1 ? 0 : t, 3e-9 * t);
                const double work = -p * (final_volumes[phase] - old_volumes[phase]);
                const double u0 = f.flow->state().cells[c].phases[phase].internal_energy;
                EXPECT_NEAR(
                    cell.phases[phase].internal_energy - u0, evaluation.applied_phase_energy[phase] + work, 2e-4);
                total_u += cell.phases[phase].internal_energy;
            }
            EXPECT_NEAR(total_u, U0, 2e-4);
        }
        EXPECT_EQ(f.law_calls, int(f.mesh->num_owned_cells()) * trial.diagnostics().flow.source_evaluations);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, TwoNodeDisappearanceSpendsExactMaterialAndCanonicalCount)
{
    Fixture f({{1. / 3., 1}, {2. / 3., 4}});
    auto options = f.options(
        [](const R::MixtureQbmmContext& ctx)
        {
            const double heavy = 8. / 3.;
            return Ops{R::QbmmDisappearance{0, ctx.quadrature.atoms[0].count, from_liquid(3 - heavy)},
                R::QbmmDisappearance{1, ctx.quadrature.atoms[1].count, from_liquid(heavy)}};
        },
        true);
    Coupler controller(f.flow, options);
    const auto trial = controller.preview(.01);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& cell = trial.state().cells[c];
        EXPECT_EQ(cell.phase_numbers[1], 0);
        EXPECT_EQ(cell.qbmm_shape[1], Shape{});
        EXPECT_EQ(cell.phases[1].moles, (std::vector<double>{0, 0}));
        EXPECT_EQ(cell.phases[1].internal_energy, 0);
        EXPECT_EQ(cell.phases[1].temperature, 0);
        EXPECT_EQ(cell.phases[0].moles[0], liquid_moles + 3 * scale);
        EXPECT_EQ(trial.local_qbmm_sources()[c][1]->events.size(), 2u);
    }
    controller.commit(trial);
}

TEST(ReactingMixtureQbmmSourcesTest, SameNodeCoalescenceAndBreakupRequireConvergedGrossTurnoverAtZeroNetSource)
{
    Fixture f({{4, 1}, {2, 2}}); // [6,8,12,20]. Each reaction is the other's exact inverse.
    const auto options = f.options(
        [](const auto& input)
        {
            const double pairs = input.stage.corrector == 0 ? .5 : 1;
            return Ops{R::QbmmCoalescence{0, 0, pairs}, R::QbmmBreakup{1, pairs, {{2, 1}}}};
        });
    {
        auto capped = options;
        capped.maximum_correctors = 2;
        Coupler controller(f.flow, capped);
        EXPECT_THROW((void) controller.preview(.01), std::exception);
    }
    Coupler controller(f.flow, options);
    const auto trial = controller.preview(.01);
    EXPECT_GE(trial.diagnostics().flow.source_evaluations, 3);
    EXPECT_LE(trial.diagnostics().flow.maximum_event_residual, 1);
    EXPECT_LE(trial.diagnostics().flow.maximum_shape_source_residual, 1);
    EXPECT_EQ(trial.diagnostics().population_number_changes.at(1), 0);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& entry = trial.local_qbmm_sources().at(c).at(1).value();
        EXPECT_EQ(entry.final_count, 6);
        EXPECT_EQ(entry.final_moles, (std::vector<double>{8 * scale, 0}));
        EXPECT_EQ(entry.final_shape, (Shape{12, 20}));
        ASSERT_EQ(entry.events.size(), 2u);
        EXPECT_EQ(entry.events[0].kind, Kind::Coalescence);
        EXPECT_EQ(entry.events[1].kind, Kind::Breakup);
        EXPECT_EQ(entry.events[0].gross_consumed, (R::QbmmEventMoments{2, 2, 2, 2}));
        EXPECT_EQ(entry.events[0].gross_created, (R::QbmmEventMoments{1, 2, 4, 8}));
        EXPECT_EQ(entry.events[1].gross_consumed, entry.events[0].gross_created);
        EXPECT_EQ(entry.events[1].gross_created, entry.events[0].gross_consumed);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, SignedEventChangeBelowStoredShapePrecisionStillControlsConvergence)
{
    Fixture f({{4, 1}, {2, 2}});
    constexpr double fraction = 0x1p-53;
    auto options = f.options(
        [](const auto& in) -> Ops
        {
            if (in.stage.corrector == 0)
                return {R::QbmmCoalescence{0, 0, 1}, R::QbmmBreakup{1, 1, {{2, 1}}}};
            return {R::QbmmCoalescence{0, 0, 1}, R::QbmmBreakup{1, 1 - fraction, {{2, 1}}},
                R::QbmmBreakup{1, fraction, {{1, .5}, {1, 1.5}}}};
        });
    {
        auto short_run = options;
        short_run.maximum_correctors = 2;
        Coupler controller(f.flow, short_run);
        EXPECT_THROW((void) controller.preview(.01), std::exception);
        EXPECT_EQ(f.law_calls, static_cast<int>(2 * f.mesh->num_owned_cells()));
    }
    Coupler controller(f.flow, options);
    const auto trial = controller.preview(.01);
    EXPECT_GE(trial.diagnostics().flow.source_evaluations, 3);
    for (const auto& row : trial.local_qbmm_sources())
    {
        const auto& receipt = row.at(1).value();
        EXPECT_EQ(receipt.final_shape, (Shape{12, 20}));
        EXPECT_EQ(receipt.event_shape_change[0], fraction / 2);
        EXPECT_EQ(receipt.event_shape_change[1], 1.5 * fraction);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, OpposingReservoirImpulseAndEnergyMustConvergeEvenAtFixedNetStock)
{
    Fixture f;
    for (const bool vary_energy : {true, false})
    {
        auto options = f.options(
            [&](const R::MixtureQbmmContext& input)
            {
                const double magnitude = input.stage.corrector == 0 ? 1. : 2.;
                const double energy = vary_energy ? magnitude : 0.;
                const double impulse = vary_energy ? 0. : magnitude;
                const R::QbmmMaterialTransfer debit{
                    R::QbmmReservoirExchange{"out", -energy, {-impulse, 0, 0}}, {scale, 0}};
                const R::QbmmMaterialTransfer credit{
                    R::QbmmReservoirExchange{"in", energy, {impulse, 0, 0}}, {scale, 0}};
                return Ops{R::QbmmDisappearance{0, 1, debit}, R::QbmmBirth{{{{1, 1}, credit}}}};
            });
        {
            auto capped = options;
            capped.maximum_correctors = 2;
            Coupler controller(f.flow, capped);
            EXPECT_THROW((void) controller.preview(.01), std::exception);
        }
        Coupler controller(f.flow, options);
        const auto trial = controller.preview(.01);
        EXPECT_GE(trial.diagnostics().flow.source_evaluations, 3);
        EXPECT_EQ(trial.diagnostics().external_mass, 0);
        EXPECT_EQ(trial.diagnostics().transfer_energy, 0);
        EXPECT_EQ(trial.diagnostics().flow.external_impulse.component(0), 0);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, ChangingDaughterSizesConvergeAtFixedMaterialCountAndTopology)
{
    Fixture f({{1, 2}});
    Coupler controller(f.flow, f.options(
                                   [](const auto& input)
                                   {
                                       std::vector<R::QbmmAtom> daughters =
                                           input.stage.corrector == 0 ? std::vector<R::QbmmAtom>{{2, 1}}
                                                                      : std::vector<R::QbmmAtom>{{1, .5}, {1, 1.5}};
                                       return Ops{R::QbmmBreakup{0, 1, std::move(daughters)}};
                                   }));
    const auto trial = controller.preview(.01);
    EXPECT_GE(trial.diagnostics().flow.source_evaluations, 3);
    for (const auto& cell : trial.state().cells)
    {
        EXPECT_EQ(cell.phase_numbers[1], 2);
        EXPECT_EQ(cell.phases[1].moles[0], 2 * scale);
        EXPECT_EQ(cell.qbmm_shape[1], (Shape{2.5, 3.5}));
    }
}

TEST(ReactingMixtureQbmmSourcesTest, ReactionHeatFollowsTheMechanismAndEosConventionThroughPairedTransfer)
{
    for (const auto convention : {R::EnergyConvention::SensibleEnergy, R::EnergyConvention::IncludesFormationEnergy})
    {
        Fixture f({{4, 1}}, false, false, convention);
        constexpr double extent = 1. / 128;
        R::ReactionDefinition reaction{
            19, "unbound phase conversion", {{44, Phase::Liquid, -1, {}}, {44, Phase::Dispersed, 1, "coarse"}}, 12800};
        if (convention == R::EnergyConvention::IncludesFormationEnergy)
        {
            EXPECT_THROW((void) R::ReactionMechanism(f.catalog, {reaction}, convention), std::invalid_argument);
            reaction.heatPerExtent = 0;
        }
        auto options = f.options([](const auto&) { return Ops{R::QbmmBirth{{{{1, 1}, from_liquid(1)}}}}; }, true);
        options.mechanism = std::make_shared<const R::ReactionMechanism>(
            f.catalog, std::vector<R::ReactionDefinition>{reaction}, convention);
        options.source_identity = "unbound-conversion";
        options.source = [](const auto&, double)
        {
            R::ReactionSourceInput input;
            input.integratedExtents = {{19, 1. / 128}};
            return input;
        };
        Coupler controller(f.flow, options);
        const auto trial = controller.preview(.01);
        const double expected_heat = reaction.heatPerExtent * extent;
        EXPECT_EQ(trial.diagnostics().chemical_heat, expected_heat);
        EXPECT_NEAR(trial.diagnostics().net_thermal_energy, expected_heat, 2e-7);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            long double energy_change = 0;
            for (size_t p = 0; p < 4; ++p)
                energy_change += static_cast<long double>(trial.state().cells[c].phases[p].internal_energy) -
                                 f.flow->state().cells[c].phases[p].internal_energy;
            EXPECT_NEAR(energy_change, expected_heat + trial.diagnostics().flow.pressure_work, 2e-4);
            EXPECT_EQ(trial.state().cells[c].phases[1].moles[0], 5 * scale);
        }
    }
}

TEST(ReactingMixtureQbmmSourcesTest, HeatOnlyChangesCaloricStateWithExactlyFixedCanonicalMoments)
{
    Fixture f;
    auto options = f.options();
    options.source_identity = "heat-only";
    options.source = [](const auto&, double)
    {
        R::ReactionSourceInput input;
        input.heat.integratedEnergy = 1000;
        return input;
    };
    Coupler controller(f.flow, options);
    const auto trial = controller.preview(.01);
    const double gas_moles = (1 - liquid_moles) * Thermo::p_old / (Thermo::gas_constant * Thermo::T0);
    const double t = Thermo::T0 + 1000 / (liquid_moles * 1000 + gas_moles * Thermo::cv);
    EXPECT_NEAR(trial.state().thermodynamic_pressure, Thermo::p_old * t / Thermo::T0, 3e-4);
    EXPECT_EQ(trial.diagnostics().external_heat, 1000);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& before = f.flow->state().cells[c];
        const auto& cell = trial.state().cells[c];
        EXPECT_NEAR(cell.phases[1].temperature, t, 3e-7);
        EXPECT_EQ(cell.phase_numbers, before.phase_numbers);
        EXPECT_EQ(cell.qbmm_shape, before.qbmm_shape);
        EXPECT_EQ(cell.phases[1].moles, before.phases[1].moles);
        EXPECT_TRUE(trial.local_qbmm_sources().at(c).at(1)->events.empty());
    }
}

TEST(ReactingMixtureQbmmSourcesTest, FailedSingleOwnerEventsPreserveCoexistingCandidatesAndCallbackFreeReplay)
{
    Fixture f;
    enum class Attempt
    {
        Valid,
        BadLaw,
        CountOverdraw,
        LiquidOverdraw,
        WrongComposition
    };
    Attempt attempt = Attempt::Valid;
    Coupler controller(f.flow,
        f.options(
            [&](const auto&)
            {
                switch (attempt)
                {
                    case Attempt::BadLaw:
                        throw std::domain_error("Injected owned-cell evaluation failure.");
                    case Attempt::CountOverdraw:
                        // Birth supplies four future bubbles but only four original bubbles can be merged.
                        return Ops{R::QbmmBirth{{{{4, 1}, from_liquid(4)}}}, R::QbmmCoalescence{0, 0, 3}};
                    case Attempt::LiquidOverdraw:
                        // Do not borrow the preceding disappearance's material to cover the original liquid donor.
                        return Ops{
                            R::QbmmDisappearance{0, 4, from_liquid(4)}, R::QbmmBirth{{{{1, 50}, from_liquid(50)}}}};
                    case Attempt::WrongComposition: {
                        auto transfer = from_liquid(1);
                        transfer.moles = {0, scale / .028}; // Exactly the intended mass carried by the wrong species.
                        return Ops{R::QbmmBirth{{{{1, 1}, transfer}}}};
                    }
                    default:
                        return Ops{R::QbmmResize{0, 2, 2, from_liquid(2)}};
                }
            },
            true));
    const auto checkpoint = controller.checkpoint();
    const auto before = f.flow->serialize_qbmm_restart();
    const auto first = controller.preview(.01);
    const auto other = controller.preview(.02);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    int is_empty = f.mesh->num_owned_cells() == 0, empty_total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &is_empty, &empty_total);
    EXPECT_EQ(empty_total, comm->getSize() - 1);
    for (auto bad : {Attempt::BadLaw, Attempt::CountOverdraw, Attempt::LiquidOverdraw, Attempt::WrongComposition})
    {
        attempt = bad;
        EXPECT_THROW((void) controller.preview(.01), std::exception);
        EXPECT_EQ(f.flow->serialize_qbmm_restart(), before);
    }
    const std::array calls{f.law_calls, f.constitutive_calls};
    controller.commit(first);
    EXPECT_THROW(controller.commit(other), std::exception);
    EXPECT_EQ(f.flow->state().time, .01);
    controller.restore(checkpoint);
    EXPECT_EQ(f.flow->state().time, 0);
    EXPECT_THROW(controller.commit(first), std::exception);
    EXPECT_EQ((std::array{f.law_calls, f.constitutive_calls}), calls);
    attempt = Attempt::Valid;
    const auto replay = controller.preview(.01);
    EXPECT_EQ(replay.state().thermodynamic_pressure, first.state().thermodynamic_pressure);
    EXPECT_EQ(first.transport_snapshot().cells, replay.transport_snapshot().cells);
    EXPECT_NE(first.transport_snapshot().accepted_generation, replay.transport_snapshot().accepted_generation);
    for (size_t c = 0; c < replay.state().cells.size(); ++c)
    {
        EXPECT_EQ(replay.state().cells[c].qbmm_shape, first.state().cells[c].qbmm_shape);
        EXPECT_EQ(replay.state().cells[c].phase_numbers, first.state().cells[c].phase_numbers);
        for (size_t p = 0; p < 4; ++p)
        {
            EXPECT_EQ(replay.state().cells[c].phases[p].moles, first.state().cells[c].phases[p].moles);
            EXPECT_EQ(
                replay.state().cells[c].phases[p].internal_energy, first.state().cells[c].phases[p].internal_energy);
        }
    }
}

TEST(ReactingMixtureQbmmSourcesTest, ScopedLegacyCountLawConsumesPostMaterialEndpointAndCannotWriteBoundFamily)
{
    Fixture f;
    bool misdirect = false;
    auto options = f.options([](const auto&) { return Ops{R::QbmmBirth{{{{1, 1}, from_liquid(1)}}}}; }, true);
    options.population_events = std::make_shared<const R::MixturePopulationEventProvider>(
        R::MixturePopulationEventProvider{"coarse-count", 1, f.catalog->fingerprint(),
            [&](const auto& context)
            {
                EXPECT_EQ(context.post_material.amount(*f.catalog, 44, Phase::Dispersed, "vapor"), 5 * scale);
                return std::vector<BubbleGasPhysics::PopulationNumberEvent>{{misdirect ? "vapor" : "coarse", 3, 0}};
            },
            {"coarse"}});
    auto dual = options;
    const auto& legacy = *options.population_events;
    dual.population_events =
        std::make_shared<const R::MixturePopulationEventProvider>(R::MixturePopulationEventProvider{
            legacy.identity, legacy.revision, legacy.schema_fingerprint, legacy.evaluate, {"vapor", "coarse"}});
    EXPECT_THROW((void) Coupler(f.flow, dual), std::exception);
    Coupler controller(f.flow, options);
    const auto trial = controller.preview(.01);
    for (const auto& cell : trial.state().cells)
    {
        EXPECT_EQ(cell.phase_numbers[1], 5);
        EXPECT_EQ(cell.phase_numbers[3], 5);
        EXPECT_FALSE(cell.qbmm_shape[3]);
    }
    EXPECT_EQ(trial.diagnostics().population_births[1], 1);
    EXPECT_EQ(trial.diagnostics().population_births[3], 3);
    for (const auto& number : trial.local_population_events())
    {
        EXPECT_EQ(number.provider_identity, "coarse-count");
        ASSERT_EQ(number.events.size(), 1u);
        EXPECT_EQ(number.events[0].population, "coarse");
        EXPECT_EQ(number.transported_numbers[1], 5);
        EXPECT_EQ(number.final_numbers[1], 5);
        EXPECT_EQ(number.changes[1], 0);
        EXPECT_EQ(number.births[1], 0);
        EXPECT_EQ(number.births[3], 3);
    }
    misdirect = true;
    EXPECT_THROW((void) controller.preview(.01), std::exception);
    controller.commit(trial);
}

TEST(ReactingMixtureQbmmSourcesTest, UnpairedBoundMaterialIsRejectedBeforeALaterLawCanObserveIt)
{
    Fixture f;
    const auto opts = f.options([](const auto&) { return Ops{}; });
    auto external = opts;
    external.source_identity = "unpaired-vapor";
    external.source = [](const auto&, double)
    {
        R::ReactionSourceInput input;
        // The final phase amount is unchanged, but both gross transfers are unallocated.
        input.externalTransfers = {{"in", 44, Phase::Dispersed, scale, scale, 0, {}, "vapor"},
            {"out", 44, Phase::Dispersed, -scale, -scale, 0, {}, "vapor"}};
        return input;
    };
    {
        Coupler controller(f.flow, external);
        EXPECT_THROW((void) controller.preview(.01), std::exception);
    }
    auto chemistry = opts;
    chemistry.mechanism = std::make_shared<const R::ReactionMechanism>(f.catalog,
        std::vector<R::ReactionDefinition>{
            {1, "unallocated-vaporization", {{44, Phase::Liquid, -1, {}}, {44, Phase::Dispersed, 1, "vapor"}}, 0}},
        f.gas->energy_convention());
    EXPECT_THROW((void) Coupler(f.flow, chemistry), std::exception);
    auto interphase = opts;
    interphase.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(
        R::MixtureInterphaseEventProvider{{"unpaired", 1, f.catalog->fingerprint()}, f.slots,
            [](const auto&) { return std::vector<std::vector<double>>{{0, 0}}; }});
    EXPECT_THROW((void) Coupler(f.flow, interphase), std::exception);
    EXPECT_EQ(f.law_calls, 0);
}

TEST(ReactingMixtureQbmmSourcesTest, SharedThermalSlotsPreserveTheOriginalDonorLawDomain)
{
    Fixture f;
    auto options = f.options([](const auto&) { return Ops{}; }, true);
    auto other = *f.slots;
    auto& donor = other.phase_enthalpy[0];
    auto bounds = donor->bounds;
    bounds.maximum_pressure = std::midpoint(bounds.minimum_pressure, bounds.maximum_pressure);
    donor = std::make_shared<const R::PartialMolarEnthalpyLaw>(
        R::PartialMolarEnthalpyLaw{donor->metadata, bounds, donor->evaluate});
    auto generic = [&](R::ConfiguredInterphaseStage stage)
    {
        other.stages = {std::move(stage)};
        return std::make_shared<const R::MixtureInterphaseEventProvider>(R::MixtureInterphaseEventProvider{
            {"generic", 1, f.catalog->fingerprint()}, std::make_shared<const R::InterphaseConfiguration>(other),
            [](const auto&) { return std::vector<std::vector<double>>{{0.}}; }});
    };
    options.interphase = generic({"material", R::InterphaseStageKind::Material, {{44, 0, 3}}, {}});
    EXPECT_THROW((void) Coupler(f.flow, options), std::exception);
    options.interphase = generic({"heat", R::InterphaseStageKind::Heat, {}, {{0, 3}}});
    EXPECT_NO_THROW((void) Coupler(f.flow, options));
}

TEST(ReactingMixtureQbmmSourcesTest, ReservoirEnergyAndMaterialFollowUnrelatedInterphaseStageExactlyOnce)
{
    for (const bool paired : {false, true})
    {
        SCOPED_TRACE(paired);
        Fixture f;
        constexpr double injected_energy = 1750, impulse = .002;
        auto options = f.options(
            [&](const R::MixtureQbmmContext& ctx)
            {
                EXPECT_EQ(ctx.material.amount(*f.catalog, 44, Phase::Liquid), liquid_moles - scale);
                EXPECT_EQ(ctx.material.amount(*f.catalog, 44, Phase::Dispersed, "coarse"), coarse_moles + scale);
                const R::QbmmMaterialTransfer feed{
                    R::QbmmReservoirExchange{"feed", injected_energy, {impulse, 0, 0}}, {2 * scale, 0}};
                Ops events{R::QbmmBirth{{{{1, 2}, feed}}}};
                if (paired)
                    events.emplace_back(R::QbmmResize{0, 1, 2, from_liquid(1)});
                return events;
            },
            paired);
        auto unrelated = *f.slots;
        unrelated.stages = {{"unbound-coarse-transfer", R::InterphaseStageKind::Material, {{44, 0, 3}}, {}}};
        options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(
            R::MixtureInterphaseEventProvider{{"generic-coarse-source", 1, f.catalog->fingerprint()},
                std::make_shared<const R::InterphaseConfiguration>(unrelated),
                [](const auto&) { return std::vector<std::vector<double>>{{scale}}; }});
        Coupler controller(f.flow, options);
        const auto trial = controller.preview(.01);
        EXPECT_EQ(trial.diagnostics().external_mass, 2 * scale);
        EXPECT_EQ(trial.diagnostics().transfer_energy, injected_energy);
        EXPECT_EQ(trial.diagnostics().external_heat, 0);
        EXPECT_NEAR(trial.diagnostics().net_thermal_energy, injected_energy, 1e-7);
        EXPECT_DOUBLE_EQ(trial.diagnostics().flow.external_impulse.component(0), impulse);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& cell = trial.state().cells[c];
            const auto& plan = trial.interphase_receipt()->cells.at(c);
            EXPECT_EQ(plan.amounts.size(), paired ? 2u : 1u);
            EXPECT_EQ(plan.amounts[0], (std::vector<double>{scale}));
            if (paired)
                EXPECT_EQ(plan.amounts[1], (std::vector<double>{scale, 0}));
            EXPECT_EQ(plan.material_tail.paired_stage, paired ? std::optional<size_t>(1) : std::nullopt);
            ASSERT_EQ(plan.material_tail.reservoirs.size(), 1u);
            EXPECT_EQ(plan.material_tail.reservoirs[0].phase, 1u);
            EXPECT_EQ(plan.material_tail.reservoirs[0].signed_moles, (std::vector<double>{2 * scale, 0}));
            EXPECT_EQ(plan.material_tail.reservoirs[0].energy, injected_energy);
            EXPECT_EQ(cell.phases[0].moles[0], liquid_moles - (paired ? 2 : 1) * scale);
            EXPECT_EQ(cell.phases[1].moles[0], (paired ? 7 : 6) * scale);
            EXPECT_EQ(cell.phases[3].moles[0], coarse_moles + scale);
            EXPECT_EQ(cell.phase_numbers[1], 5);
            EXPECT_EQ(cell.qbmm_shape[1], paired ? (Shape{11, 19}) : (Shape{8, 12}));
            double initial_energy = 0, final_energy = 0;
            for (size_t p = 0; p < 4; ++p)
            {
                EXPECT_EQ(plan.final_moles[p], cell.phases[p].moles);
                initial_energy += f.flow->state().cells[c].phases[p].internal_energy;
                final_energy += cell.phases[p].internal_energy;
            }
            EXPECT_NEAR(final_energy - initial_energy, injected_energy, 2e-4);
            const auto& raw = trial.local_qbmm_sources()[c][1]->events[0].transfers[0];
            EXPECT_TRUE(raw.into_family);
            EXPECT_EQ(std::get<R::QbmmReservoirExchange>(raw.transfer.counterpart).reservoir, "feed");
        }
    }
}

TEST(ReactingMixtureQbmmSourcesTest, TwoFamiliesReserveOneGrossLiquidDonorWithoutCrossFamilyCredit)
{
    Fixture f({{4, 1}}, false, true);
    bool overdraw = false;
    Coupler controller(f.flow, f.options(
                                   [&](const auto& ctx)
                                   {
                                       EXPECT_EQ(ctx.material.amount(*f.catalog, 44, Phase::Liquid), liquid_moles);
                                       if (ctx.phase == 1)
                                       {
                                           if (overdraw)
                                               return Ops{R::QbmmDisappearance{0, 4, from_liquid(4)},
                                                   R::QbmmBirth{{{{1, 24}, from_liquid(24)}}}};
                                           return Ops{R::QbmmBirth{{{{1, 1}, from_liquid(1)}}}};
                                       }
                                       const double mass = overdraw ? 26 : 2;
                                       return Ops{R::QbmmBirth{{{{1, mass}, from_liquid(mass)}}}};
                                   },
                                   true));
    const auto trial = controller.preview(.01);
    const auto before = f.flow->serialize_qbmm_restart();
    overdraw = true; // Combined gross consumption = 50 scales, original stock = 48, net = 46.
    EXPECT_THROW((void) controller.preview(.01), std::exception);
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), before);
    controller.commit(trial);
    for (const auto& cell : f.flow->state().cells)
    {
        EXPECT_EQ(cell.phases[0].moles[0], liquid_moles - 3 * scale);
        EXPECT_EQ(cell.phase_numbers[1], 5);
        EXPECT_EQ(cell.phase_numbers[3], 3);
        EXPECT_EQ(cell.qbmm_shape[1], (Shape{5, 5}));
        EXPECT_EQ(cell.qbmm_shape[3], (Shape{132, 1032}));
    }
}

TEST(ReactingMixtureQbmmSourcesTest, PairedEventsReserveCompensatedChemistryCreditAndDetectHiddenDeficit)
{
    constexpr double small = 0x1p-57; // Below half one binary64 ulp of the represented liquid stock.
    for (const double change : {small, -small})
    {
        SCOPED_TRACE(change);
        Fixture f;
        auto options = f.options(
            [&](const auto& context)
            {
                EXPECT_EQ(context.material.amount(*f.catalog, 44, Phase::Liquid), liquid_moles);
                R::QbmmBirth birth{{{{liquid_moles / scale, 1}, from_liquid(liquid_moles / scale)}}};
                if (change > 0)
                    birth.atoms.push_back({{small / scale, 1}, from_liquid(small / scale)});
                return Ops{birth};
            },
            true);
        options.source_identity = "sub-ulp-liquid-inventory";
        options.source = [change](const auto&, double)
        {
            R::ReactionSourceInput input;
            input.externalTransfers = {{"liquid-stock", 44, Phase::Liquid, change, change, change * liquid_u0, {}, {}}};
            return input;
        };
        Coupler controller(f.flow, options);
        if (change < 0)
        {
            const auto baseline = f.flow->serialize_qbmm_restart();
            EXPECT_THROW((void) controller.preview(.01), std::exception);
            EXPECT_EQ(f.flow->serialize_qbmm_restart(), baseline);
            continue;
        }

        const auto trial = controller.preview(.01);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& plan = trial.interphase_receipt()->cells[c];
            const auto& original = plan.raw_pre_moles[0][0];
            EXPECT_EQ(plan.pre_moles[0][0], liquid_moles);
            EXPECT_EQ((original[0] - liquid_moles) + original[1], small);
            EXPECT_EQ(plan.material_tail.paired_amounts[0], (std::vector<double>{liquid_moles, small}));
            EXPECT_EQ(plan.final_moles[0][0], 0);
            EXPECT_EQ(trial.state().cells[c].phases[0].moles[0], 0);
            const auto& receipt = *trial.local_qbmm_sources()[c][1];
            ASSERT_EQ(receipt.events.size(), 1u);
            ASSERT_EQ(receipt.events[0].transfers.size(), 2u);
            EXPECT_EQ(receipt.events[0].transfers[1].transfer.moles[0], small);
            EXPECT_EQ(receipt.final_moles[0], liquid_moles + 4 * scale);
            EXPECT_EQ(receipt.final_count, 52);
            EXPECT_EQ(receipt.final_shape, (Shape{52, 52}));
        }
        EXPECT_EQ(trial.diagnostics().external_mass, small);
        controller.commit(trial);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, MovingTransportFeedsCoherentEventsThermalsCommitAndRestart)
{
    using LO = Flow::Pack::local_ordinal_type;
    Fixture f({{4, 1}, {4, 2}}, true);
    const auto accepted = f.flow->state();
    std::vector<R::QbmmMomentBlock> law_moments(accepted.cells.size());
    std::vector<std::array<double, 2>> law_nodes(accepted.cells.size());
    auto options = f.options(
        [&](const R::MixtureQbmmContext& context)
        {
            EXPECT_EQ(context.quadrature.active_atoms, 2u);
            // Independent reconstruction from the law's atoms, before any event adds material.
            R::QbmmMomentBlock moments{};
            for (const auto& atom : context.quadrature.atoms)
            {
                double power = 1;
                for (size_t k = 0; k < moments.size(); ++k)
                {
                    moments[k] += atom.count * power;
                    power *= atom.mass_over_scale;
                }
            }
            EXPECT_NEAR(moments[1] * scale, context.material.amount(*f.catalog, 44, Phase::Dispersed, "vapor"), 2e-13);
            law_moments[context.owned_cell] = moments;
            const double x = context.quadrature.atoms[0].mass_over_scale;
            const double y = context.quadrature.atoms[1].mass_over_scale;
            law_nodes[context.owned_cell] = {x, y};
            const double birth = f.mesh->cell_centroid(static_cast<LO>(context.owned_cell)).z < 1 ? .25 : .5;
            return Ops{R::QbmmBirth{{{{birth, 1}, from_liquid(birth)}}}, R::QbmmResize{0, .5, x + 1, from_liquid(.5)},
                R::QbmmCoalescence{0, 0, .25}, R::QbmmBreakup{1, .25, {{2, y / 2}}}};
        },
        true);
    Coupler controller(f.flow, options);
    const auto before = f.flow->serialize_qbmm_restart();
    const auto trial = controller.preview(.01);
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), before);
    EXPECT_GE(trial.diagnostics().flow.source_evaluations, 2);
    EXPECT_LE(trial.diagnostics().flow.maximum_transport_residual, 1);
    EXPECT_LE(trial.diagnostics().flow.maximum_event_residual, 1);
    EXPECT_LE(trial.diagnostics().flow.maximum_shape_source_residual, 1);
    EXPECT_LE(trial.diagnostics().flow.maximum_conditioned_thermal_residual, 1);
    EXPECT_NE(trial.state().thermodynamic_pressure, accepted.thermodynamic_pressure);

    const auto comm = f.mesh->owned_cell_map()->getComm();
    std::array<double, 5> local_totals{};
    // The two equal-volume cells start with the same distribution. For the one interior
    // face, every moment transports in the same proportion as the donor number.
    EXPECT_EQ(trial.owned_phase_transport().size(), f.mesh->num_owned_faces());
    for (const auto& face : trial.owned_phase_transport())
    {
        const auto& rates = face.qbmm_flux.at(1).value();
        EXPECT_NEAR(rates.shape_flux.H2, 2.5 * rates.population_number_flux, 2e-13);
        EXPECT_NEAR(rates.shape_flux.H3, 4.5 * rates.population_number_flux, 2e-13);
    }
    double local_transport_change = 0;
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const auto& transported = trial.transport_snapshot().cells[c].phases[1];
        const auto& cell = trial.state().cells[c];
        const auto& receipt = *trial.local_qbmm_sources()[c][1];
        const double birth = f.mesh->cell_centroid(static_cast<LO>(c)).z < 1 ? .25 : .5;
        const auto [x, y] = law_nodes[c];
        const R::QbmmMomentBlock transport_moments{*transported.population_number, transported.moles[0] / scale,
            transported.qbmm_shape->H2, transported.qbmm_shape->H3};
        for (size_t k = 0; k < transport_moments.size(); ++k)
            EXPECT_NEAR(law_moments[c][k], transport_moments[k], 3e-12);
        local_transport_change = std::max(
            local_transport_change, std::abs(*transported.population_number - *accepted.cells[c].phase_numbers[1]));
        EXPECT_EQ(receipt.events.size(), 4u);
        if (receipt.events.size() == 4)
        {
            EXPECT_EQ(receipt.events[0].kind, Kind::Birth);
            EXPECT_EQ(receipt.events[1].kind, Kind::Growth);
            EXPECT_EQ(receipt.events[2].kind, Kind::Coalescence);
            EXPECT_EQ(receipt.events[3].kind, Kind::Breakup);
        }
        EXPECT_NEAR(*cell.phase_numbers[1], *transported.population_number + birth, 2e-13);
        EXPECT_NEAR(cell.phases[1].moles[0], transported.moles[0] + scale * (birth + .5), 2e-13);
        EXPECT_EQ(cell.phases[1].moles[1], 0);
        // Add/remove each original atom independently; neither event borrows another's products.
        for (const unsigned k : {2u, 3u})
        {
            const double delta = birth + .5 * (std::pow(x + 1, k) - std::pow(x, k)) +
                                 .25 * (std::pow(2 * x, k) - 2 * std::pow(x, k)) +
                                 .25 * (2 * std::pow(y / 2, k) - std::pow(y, k));
            const double shape = k == 2 ? cell.qbmm_shape[1]->H2 : cell.qbmm_shape[1]->H3;
            EXPECT_NEAR(shape, transport_moments[k] + delta, 5e-12);
        }
        EXPECT_EQ(cell.phases[1].moles, receipt.final_moles);
        EXPECT_EQ(cell.phase_numbers[1], receipt.final_count);
        EXPECT_EQ(cell.qbmm_shape[1], receipt.final_shape);
        const auto& evaluation = trial.final_thermal_evaluations()[c];
        EXPECT_EQ(evaluation.material_events.size(), 1u);
        const double t = evaluation.anchors.front().temperature[0];
        const double h = liquid_u0 + 1000 * (t - Thermo::T0) + evaluation.absolute_pressure;
        const double carried = scale * (birth + .5) * h;
        if (!evaluation.material_events.empty())
            EXPECT_NEAR(evaluation.material_events[0].carried_energy, carried, 2e-7 * carried);
        double volume = 0;
        for (size_t p = 0; p < cell.phases.size(); ++p)
        {
            EXPECT_NEAR(cell.phases[p].temperature, cell.phases[0].temperature, 2e-7);
            local_totals[0] += cell.phases[p].internal_energy - accepted.cells[c].phases[p].internal_energy;
            const auto& phase = cell.phases[p];
            const double phase_volume = p == 0 ? phase.moles[0]
                                               : std::accumulate(phase.moles.begin(), phase.moles.end(), 0.) *
                                                     Thermo::gas_constant * phase.temperature / phase.absolute_pressure;
            volume += phase_volume;
            local_totals[4] -= phase.absolute_pressure *
                               (phase_volume - trial.transport_snapshot().cells[c].phases[p].advected_material_volume);
        }
        EXPECT_NEAR(volume, f.mesh->cell_volume(static_cast<LO>(c)), 2e-9);
        local_totals[1] += cell.phases[0].moles[0] - accepted.cells[c].phases[0].moles[0];
        local_totals[2] += cell.phases[1].moles[0] - accepted.cells[c].phases[1].moles[0];
        local_totals[3] += *cell.phase_numbers[1] - *accepted.cells[c].phase_numbers[1];
    }
    std::array<double, 5> totals{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 5, local_totals.data(), totals.data());
    double transport_change = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_transport_change, &transport_change);
    EXPECT_GT(transport_change, 1e-8);
    EXPECT_NEAR(totals[0], trial.diagnostics().flow.pressure_work, 4e-4);
    EXPECT_NEAR(trial.diagnostics().flow.pressure_work, totals[4], 2e-8);
    EXPECT_NEAR(totals[1], -1.75 * scale, 2e-12);
    EXPECT_NEAR(totals[2], 1.75 * scale, 2e-12);
    EXPECT_NEAR(totals[3], .75, 2e-11);
    EXPECT_EQ(trial.diagnostics().external_mass, 0);
    EXPECT_EQ(trial.diagnostics().external_heat, 0);
    const std::array calls{f.law_calls, f.constitutive_calls};
    controller.commit(trial);
    EXPECT_EQ((std::array{f.law_calls, f.constitutive_calls}), calls);
    EXPECT_EQ(f.flow->state().time, .01);
    // Asymmetric births retain a nonzero accepted internal carrier flux.
    double local_flux = 0, global_flux = 0;
    for (size_t face = 0; face < f.mesh->num_owned_faces(); ++face)
        local_flux = std::max(local_flux, std::abs(f.flow->volume_flux().owned_value(static_cast<LO>(face))));
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_flux, &global_flux);
    EXPECT_GT(global_flux, 1e-8);
    const auto archive = f.flow->serialize_qbmm_restart();
    Flow restored(f.mesh, std::vector{f.liquid, f.gas, f.gas, f.gas}, f.flow->options());
    restored.restore_qbmm_restart(archive);
    EXPECT_EQ(restored.state().time, f.flow->state().time);
    EXPECT_EQ(restored.state().revision, f.flow->state().revision);
    EXPECT_EQ(restored.state().thermodynamic_pressure, f.flow->state().thermodynamic_pressure);
    EXPECT_EQ(restored.state().qbmm_families, f.flow->state().qbmm_families);
    for (size_t c = 0; c < restored.state().cells.size(); ++c)
    {
        const auto& cell = restored.state().cells[c];
        const auto& endpoint = trial.state().cells[c];
        EXPECT_EQ(cell.qbmm_shape, endpoint.qbmm_shape);
        EXPECT_EQ(cell.phase_numbers, endpoint.phase_numbers);
        for (size_t p = 0; p < cell.phases.size(); ++p)
        {
            EXPECT_EQ(cell.phases[p].moles, endpoint.phases[p].moles);
            EXPECT_EQ(cell.phases[p].internal_energy, endpoint.phases[p].internal_energy);
            EXPECT_EQ(cell.phases[p].temperature, endpoint.phases[p].temperature);
        }
    }
    for (size_t face = 0; face < f.mesh->num_owned_faces(); ++face)
        EXPECT_EQ(restored.volume_flux().owned_value(static_cast<LO>(face)),
            f.flow->volume_flux().owned_value(static_cast<LO>(face)));
}

TEST(ReactingMixtureQbmmSourcesTest, PlanarEventsUseLiveGeometryAndReplayAfterSingleOwnerFailure)
{
    using LO = Flow::Pack::local_ordinal_type;
    const char* writer = std::getenv("SIMPLEFLUID_Q7A_RESTART_WRITE");
    const char* reader = std::getenv("SIMPLEFLUID_Q7A_RESTART_READ");
    ASSERT_FALSE(writer && reader);
    struct Observations
    {
        double target = 0;
        int targets = 0, injected_failures = 0;
        bool fail = false;
    };
    constexpr double dt = .01;
    const auto make_options = [&](Fixture& f, Observations& calls, const char* target_identity = "Q7a coherent piston")
    {
        auto options = f.options(
            [&](const R::MixtureQbmmContext& context)
            {
                if (!context.law_inputs || !context.law_inputs->location)
                    throw std::runtime_error("The coherent source must provide its live location.");
                const auto& location = *context.law_inputs->location;
                const LO cell = static_cast<LO>(context.owned_cell);
                EXPECT_EQ(location.ownedCell, context.owned_cell);
                EXPECT_EQ(location.geometryCellId, f.mesh->cell_geometry_global_id(cell));
                EXPECT_EQ(location.geometryEpoch, f.mesh->geometry_epoch());
                EXPECT_NEAR(location.centroid[2], calls.target / 2, 1e-14);
                EXPECT_NEAR(context.law_inputs->bulk_volume_m3, calls.target, 1e-14);
                EXPECT_DOUBLE_EQ(context.law_inputs->bulk_volume_m3, f.mesh->cell_volume(cell));
                if (calls.fail)
                {
                    ++calls.injected_failures;
                    throw std::runtime_error("Manufactured source failure on the only owning rank.");
                }
                // The manufactured birth depends on both pieces of the live geometry.
                const double count = context.law_inputs->bulk_volume_m3 * location.centroid[2] * .5;
                return Ops{R::QbmmBirth{{{{count, 1}, from_liquid(count)}}}};
            },
            true);
        options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(
            R::MixturePlanarMotionProvider{f.mesh, {}, target_identity, 3, [&calls](double accepted_time, double step)
                {
                    ++calls.targets;
                    calls.target = 1. + .05 * (accepted_time + step);
                    return calls.target;
                }});
        return options;
    };
    const auto expect_same = [](const Fixture& a, const Fixture& b)
    {
        const auto& left = a.flow->state();
        const auto& right = b.flow->state();
        EXPECT_EQ(left.time, right.time);
        EXPECT_EQ(left.revision, right.revision);
        EXPECT_DOUBLE_EQ(left.thermodynamic_pressure, right.thermodynamic_pressure);
        ASSERT_EQ(left.cells.size(), right.cells.size());
        for (size_t c = 0; c < left.cells.size(); ++c)
        {
            const LO cell = static_cast<LO>(c);
            EXPECT_DOUBLE_EQ(a.mesh->cell_volume(cell), b.mesh->cell_volume(cell));
            EXPECT_EQ(a.mesh->cell_centroid(cell), b.mesh->cell_centroid(cell));
            EXPECT_EQ(left.cells[c].qbmm_shape, right.cells[c].qbmm_shape);
            EXPECT_EQ(left.cells[c].phase_numbers, right.cells[c].phase_numbers);
            EXPECT_EQ(left.cells[c].velocity, right.cells[c].velocity);
            for (size_t p = 0; p < left.cells[c].phases.size(); ++p)
            {
                const auto& x = left.cells[c].phases[p];
                const auto& y = right.cells[c].phases[p];
                EXPECT_EQ(x.moles, y.moles);
                EXPECT_DOUBLE_EQ(x.temperature, y.temperature);
                EXPECT_DOUBLE_EQ(x.internal_energy, y.internal_energy);
                EXPECT_DOUBLE_EQ(x.absolute_pressure, y.absolute_pressure);
            }
        }
        ASSERT_EQ(a.mesh->num_faces(), b.mesh->num_faces());
        for (size_t f = 0; f < a.mesh->num_faces(); ++f)
            EXPECT_DOUBLE_EQ(a.flow->volume_flux().local_value(static_cast<LO>(f)),
                b.flow->volume_flux().local_value(static_cast<LO>(f)));
    };

    Fixture f;
    Observations calls;
    Coupler controller(f.flow, make_options(f, calls));
    const auto comm = f.mesh->owned_cell_map()->getComm();
    if (writer)
    {
        controller.advance(dt);
        RestartIO::write_restart(*comm, writer, ".first", controller.serialize_qbmm_restart());
        controller.advance(dt);
        RestartIO::write_restart(*comm, writer, ".next", controller.serialize_qbmm_restart());
        return;
    }
    if (reader)
    {
        controller.restore_qbmm_restart(RestartIO::read_restart(*comm, reader, ".next"));
        Fixture fresh;
        Observations fresh_calls;
        Coupler restored(fresh.flow, make_options(fresh, fresh_calls));
        restored.restore_qbmm_restart(RestartIO::read_restart(*comm, reader, ".first"));
        EXPECT_EQ(fresh_calls.targets, 0);
        EXPECT_EQ(fresh.law_calls, 0);
        restored.advance(dt);
        EXPECT_EQ(fresh_calls.targets, 1);
        expect_same(f, fresh);
        return;
    }

    int empty = f.mesh->num_owned_cells() == 0, empty_total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_total);
    EXPECT_EQ(empty_total, comm->getSize() - 1);
    const auto before = controller.serialize_qbmm_restart();
    const auto first = controller.preview(dt);
    EXPECT_EQ(calls.targets, 1);
    EXPECT_GE(first.diagnostics().flow.source_evaluations, 2);
    const double first_target = 1. + .05 * dt;
    const double expected_birth = first_target * first_target / 4;
    for (size_t c = 0; c < first.state().cells.size(); ++c)
    {
        const auto& pre_source = first.transport_snapshot().cells[c].phases[1];
        const auto& receipt = *first.local_qbmm_sources()[c][1];
        ASSERT_EQ(receipt.events.size(), 1u);
        EXPECT_EQ(receipt.events[0].kind, Kind::Birth);
        EXPECT_NEAR(receipt.final_count - *pre_source.population_number, expected_birth, 1e-13);
        EXPECT_NEAR(receipt.final_moles[0] - pre_source.moles[0], scale * expected_birth, 1e-13);
        EXPECT_NEAR(receipt.final_shape.H2 - pre_source.qbmm_shape->H2, expected_birth, 1e-13);
        EXPECT_NEAR(receipt.final_shape.H3 - pre_source.qbmm_shape->H3, expected_birth, 1e-13);
    }
    const auto first_report = controller.qbmm_report(first);
    const auto epoch_before_failure = f.mesh->geometry_epoch();
    calls.fail = true;
    EXPECT_THROW((void) controller.preview(2 * dt), std::exception);
    calls.fail = false;
    int failures = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &calls.injected_failures, &failures);
    EXPECT_EQ(failures, 1);
    EXPECT_EQ(calls.targets, 2);
    EXPECT_GT(f.mesh->geometry_epoch(), epoch_before_failure);
    EXPECT_EQ(controller.serialize_qbmm_restart(), before);
    const auto retained = controller.qbmm_report(first);
    EXPECT_EQ(retained.provenance.preview_id, first_report.provenance.preview_id);
    EXPECT_EQ(retained.provenance.geometry_epoch, first.transport_snapshot().new_geometry_epoch);
    for (const auto& cell : retained.cells)
        EXPECT_NEAR(cell.bulk_volume_m3, first_target, 1e-14);
    const std::array callback_counts{calls.targets, f.law_calls, f.constitutive_calls};
    controller.commit(first);
    EXPECT_EQ((std::array{calls.targets, f.law_calls, f.constitutive_calls}), callback_counts);
    EXPECT_EQ(f.flow->state().time, dt);
    const auto archived = controller.serialize_qbmm_restart();
    Fixture fresh;
    Observations fresh_calls;
    Coupler restarted(fresh.flow, make_options(fresh, fresh_calls));
    restarted.restore_qbmm_restart(archived);
    EXPECT_EQ((std::array{fresh_calls.targets, fresh.law_calls, fresh.constitutive_calls}), (std::array{0, 0, 0}));
    expect_same(f, fresh);
    EXPECT_GE(restarted.diagnostics().correctors, 2);
    EXPECT_EQ(restarted.diagnostics().correctors, controller.diagnostics().correctors);
    EXPECT_EQ(restarted.diagnostics().flow.source_evaluations, controller.diagnostics().flow.source_evaluations);
    ASSERT_EQ(restarted.diagnostics().population_births.size(), f.flow->options().phase_roles.size());
    EXPECT_NEAR(restarted.diagnostics().population_births[1], expected_birth, 1e-13);
    EXPECT_EQ(restarted.diagnostics().population_births, controller.diagnostics().population_births);
    Fixture wrong;
    Observations wrong_calls;
    Coupler wrong_target(wrong.flow, make_options(wrong, wrong_calls, "different geometry target"));
    EXPECT_THROW(wrong_target.restore_qbmm_restart(archived), std::exception);
    EXPECT_EQ(wrong.flow->state().time, 0);
    EXPECT_EQ(wrong_calls.targets, 0);
    controller.advance(dt);
    restarted.advance(dt);
    expect_same(f, fresh);
}

TEST(ReactingMixtureQbmmSourcesTest, RankMixedControllerAndNativeAdvancementRejectBeforeCallbacks)
{
    constexpr double dt = .01, target = 1.0005;
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    Fixture unrelated;
    Flow::PlanarMotion native_motion(unrelated.mesh);
    const auto native_trial = unrelated.flow->preview_ale(dt, native_motion, target);

    int target_calls = 0, native_source_calls = 0;
    auto options = f.options([](const auto&) { return Ops{}; });
    options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(
        R::MixturePlanarMotionProvider{f.mesh, {}, "consistent controller entry", 1, [&](double, double)
            {
                ++target_calls;
                return target;
            }});
    Coupler controller(f.flow, options);
    auto native_source = std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{
        "native test source", 1, f.catalog->fingerprint(), [&](const Flow::SourceStageContext& context)
        {
            ++native_source_calls;
            return Flow::SourceStageResult{std::vector<Flow::CellSources>(context.transport.cells.size()), {}};
        }});
    const auto expect_mismatch = [](const auto& call)
    {
        try
        {
            call();
            ADD_FAILURE() << "Expected rank-mixed advancement to be rejected.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_STREQ(error.what(), "Low-Mach advancement operation must agree on every rank.");
        }
    };
    const auto accepted = controller.serialize_qbmm_restart();
    const auto before_preview = f.mesh->geometry_epoch();
    if (comm->getSize() > 1)
    {
        expect_mismatch(
            [&]
            {
                if (comm->getRank() == 0)
                    (void) controller.preview(dt);
                else
                    (void) f.flow->preview_ale_with_sources(dt, native_motion, target, native_source);
            });
        EXPECT_EQ(f.mesh->geometry_epoch(), before_preview);
        EXPECT_EQ(controller.serialize_qbmm_restart(), accepted);
    }
    EXPECT_EQ(target_calls, 0);
    EXPECT_EQ(f.law_calls, 0);
    EXPECT_EQ(native_source_calls, 0);

    const auto earlier = controller.preview(dt);
    EXPECT_EQ(target_calls, 1);
    const auto before_commit = f.mesh->geometry_epoch();
    const auto callback_counts = std::array{target_calls, f.law_calls, f.constitutive_calls, native_source_calls};
    if (comm->getSize() > 1)
    {
        expect_mismatch(
            [&]
            {
                if (comm->getRank() == 0)
                    controller.commit(earlier);
                else
                    f.flow->commit_ale(native_trial, native_motion);
            });
        EXPECT_EQ(f.mesh->geometry_epoch(), before_commit);
        EXPECT_EQ(controller.serialize_qbmm_restart(), accepted);
    }
    controller.commit(earlier);
    EXPECT_EQ((std::array{target_calls, f.law_calls, f.constitutive_calls, native_source_calls}), callback_counts);
    EXPECT_EQ(f.flow->state().time, dt);
}

TEST(ReactingMixtureQbmmSourcesTest, MovingExportRequiresItsOwnAcceptedIntervalOrFreshInitialization)
{
    using LO = Flow::Pack::local_ordinal_type;
    constexpr double dt = .01;
    const auto source = [](const auto&) { return Ops{R::QbmmBirth{{{{.25, 1}, from_liquid(.25)}}}}; };
    const auto moving_options = [&](Fixture& f, int& target_calls)
    {
        auto options = f.options(source, true);
        options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(R::MixturePlanarMotionProvider{
            f.mesh, {}, "fixed-to-moving provenance", 1, [&target_calls](double time, double step)
            {
                ++target_calls;
                return 1. + .05 * (time + step);
            }});
        return options;
    };
    const auto owned_flux = [](const Fixture& f)
    {
        std::vector<double> flux(f.mesh->num_owned_faces());
        for (size_t i = 0; i < flux.size(); ++i)
            flux[i] = f.flow->volume_flux().owned_value(static_cast<LO>(i));
        return flux;
    };

    Fixture f;
    {
        Coupler fixed(f.flow, f.options(source, true));
        const auto step = fixed.advance(dt);
        ASSERT_EQ(step.population_births.size(), f.flow->options().phase_roles.size());
        EXPECT_EQ(step.population_births[1], .25);
    }
    auto fixed_report = f.flow->qbmm_report();
    ASSERT_TRUE(fixed_report.provenance.controller_configuration_signature);
    int target_calls = 0;
    Coupler moving(f.flow, moving_options(f, target_calls));
    const auto epoch = f.mesh->geometry_epoch();
    const auto fixed_flux = owned_flux(f);
    const auto old_calls = std::array{target_calls, f.law_calls, f.constitutive_calls};
    EXPECT_THROW((void) moving.checkpoint(), std::exception);
    EXPECT_THROW((void) moving.serialize_qbmm_restart(), std::exception);
    EXPECT_EQ((std::array{target_calls, f.law_calls, f.constitutive_calls}), old_calls);
    EXPECT_EQ(f.mesh->geometry_epoch(), epoch);
    EXPECT_EQ(owned_flux(f), fixed_flux);
    EXPECT_EQ(f.flow->state().time, dt);
    EXPECT_EQ(moving.qbmm_report().provenance.controller_configuration_signature,
        fixed_report.provenance.controller_configuration_signature);

    moving.commit(moving.preview(dt));
    EXPECT_EQ(target_calls, 1);
    EXPECT_EQ(f.flow->state().time, 2 * dt);
    const auto moving_flux = owned_flux(f);
    const auto committed = moving.qbmm_report();
    ASSERT_TRUE(committed.interval);
    EXPECT_NE(committed.provenance.controller_configuration_signature,
        fixed_report.provenance.controller_configuration_signature);
    const auto checkpoint = moving.checkpoint();
    const auto archive = moving.serialize_qbmm_restart();
    moving.advance(dt);
    const auto before_restore = std::array{target_calls, f.law_calls, f.constitutive_calls};
    moving.restore(checkpoint);
    EXPECT_EQ((std::array{target_calls, f.law_calls, f.constitutive_calls}), before_restore);
    EXPECT_EQ(f.flow->state().time, 2 * dt);
    EXPECT_EQ(owned_flux(f), moving_flux);

    Fixture fresh;
    int fresh_targets = 0;
    Coupler restarted(fresh.flow, moving_options(fresh, fresh_targets));
    const auto initial_checkpoint = restarted.checkpoint();
    const auto initial_archive = restarted.serialize_qbmm_restart();
    restarted.restore(initial_checkpoint);
    restarted.restore_qbmm_restart(initial_archive);
    EXPECT_EQ(fresh.flow->state().time, 0.);
    EXPECT_EQ(fresh_targets, 0);
    restarted.restore_qbmm_restart(archive);
    EXPECT_EQ(fresh_targets, 0);
    EXPECT_EQ(fresh.law_calls, 0);
    EXPECT_EQ(fresh.flow->state().time, f.flow->state().time);
    EXPECT_EQ(owned_flux(fresh), moving_flux);
    EXPECT_EQ(restarted.diagnostics().population_births, moving.diagnostics().population_births);
    const auto restarted_report = restarted.qbmm_report();
    EXPECT_EQ(restarted_report.provenance.controller_configuration_signature,
        committed.provenance.controller_configuration_signature);
    ASSERT_EQ(restarted_report.cells.size(), committed.cells.size());
    for (size_t c = 0; c < restarted_report.cells.size(); ++c)
    {
        EXPECT_EQ(restarted_report.cells[c].bulk_volume_m3, committed.cells[c].bulk_volume_m3);
        EXPECT_EQ(restarted_report.cells[c].families[1]->moments, committed.cells[c].families[1]->moments);
    }
}

TEST(ReactingMixtureQbmmSourcesTest, PortableRestartBindsTheActualControllerAndThermalLawBeforeContinuation)
{
    Fixture f;
    auto config = f.options([](const auto&) { return Ops{}; }, true);
    std::vector<std::byte> saved;
    {
        Coupler original(f.flow, config);
        original.commit(original.preview(.01));
    }
    // Even after detaching, the archive identifies the last accepted controller.
    saved = f.flow->serialize_qbmm_restart();
    auto restored = std::make_shared<Flow>(f.mesh, std::vector{f.liquid, f.gas, f.gas, f.gas}, f.flow->options());
    restored->restore_qbmm_restart(saved);
    const auto restored_before = restored->serialize_qbmm_restart();
    EXPECT_THROW((void) restored->preview(.01), std::exception);
    auto no_law = config;
    no_law.qbmm_population.reset();
    EXPECT_THROW((void) Coupler(restored, no_law), std::exception);
    auto changed = config;
    const auto& provider = *config.qbmm_population;
    auto different_slots = std::make_shared<R::InterphaseConfiguration>(*provider.transfers);
    const auto& law = *provider.transfers->phase_enthalpy[0];
    auto metadata = law.metadata;
    ++metadata.revision;
    different_slots->phase_enthalpy[0] = std::make_shared<const R::PartialMolarEnthalpyLaw>(
        R::PartialMolarEnthalpyLaw{metadata, law.bounds, law.evaluate});
    changed.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
        R::MixtureQbmmProvider{provider.metadata, different_slots, provider.evaluate});
    EXPECT_THROW((void) Coupler(restored, changed), std::exception);
    EXPECT_EQ(restored->serialize_qbmm_restart(), restored_before);
    Coupler matching(restored, config);
    const auto trial = matching.preview(.01);
    matching.commit(trial);
    EXPECT_EQ(restored->state().revision, f.flow->state().revision + 1);
}

TEST(ReactingMixtureQbmmSourcesTest, MismatchedRankPolicyRejectsBeforeLawsAndNonzeroCarrierFluxIsAdmitted)
{
    Fixture f({{4, 1}}, true);
    const auto comm = f.mesh->owned_cell_map()->getComm();
    auto options = f.options([](const auto&) { return Ops{}; });
    auto divergent = options;
    if (comm->getRank() == comm->getSize() - 1)
    {
        const auto& provider = *options.qbmm_population;
        auto metadata = provider.metadata;
        ++metadata.revision;
        divergent.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{metadata, provider.transfers, provider.evaluate});
    }
    EXPECT_THROW((void) Coupler(f.flow, divergent), std::exception);
    Coupler controller(f.flow, options);
    const auto before = f.flow->serialize_qbmm_restart();
    EXPECT_EQ(f.law_calls, 0);
    const auto trial = controller.preview(.01);
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), before);
    EXPECT_EQ(f.law_calls, int(f.mesh->num_owned_cells()) * trial.diagnostics().flow.source_evaluations);
    controller.commit(trial);
}
} // namespace
