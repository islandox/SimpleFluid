/** @file testLowMachCaloricFeasibility.cc
 * @brief Certified caloric pressure domains, bounded seed discovery and immutable flow transactions.
 */
#include "equations/ReactingThermodynamics.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
constexpr double gas_constant = 8.31446261815324, cv = 20, molar_mass = .02;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
const R::ThermodynamicBounds bounds{290, 310, 10000, 2000000};
std::shared_ptr<const R::SpeciesCatalog> catalog()
{
    const std::vector roles{R::SpeciesPhase::Resolved};
    return std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {103, "supplied_A", molar_mass, {{"X", 1}}, roles}, {709, "supplied_B", molar_mass, {{"X", 1}}, roles}});
}
std::shared_ptr<const R::MixtureThermodynamics> ideal(double reference = 0)
{
    return R::make_ideal_gas_mixture(catalog(),
        {{cv, 300, cv * 300 + molar_mass * reference}, {cv, 300, cv * 300 + molar_mass * reference}}, bounds);
}
Flow::Options controls()
{
    Flow::Options value;
    value.minimum_pressure = 20000;
    value.maximum_pressure = 1000000;
    value.linear.preconditioner = LinearPreconditioner::None;
    value.linear.tolerance = 1e-12;
    return value;
}
struct Probe
{
    int off_reference_evaluations = 0;
    double fail_above = 0;
};
struct Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::MixtureThermodynamics> gas;
    std::shared_ptr<Flow> flow;
    std::shared_ptr<Probe> probe;
    double b = 0, reference = 0;
    Fixture(double covolume = 0, bool opaque = false, double energy_reference = 0, Flow::Options options = controls(),
        std::shared_ptr<Probe> diagnostic = std::make_shared<Probe>())
        : probe(std::move(diagnostic)), b(covolume), reference(energy_reference)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
            distribution);
        const auto evaluate = [b = b, reference = reference, probe = probe](
                                  double t, double p, std::span<const double> x)
        {
            if (p != 100000)
                ++probe->off_reference_evaluations;
            if (probe->fail_above > 0 && p > probe->fail_above)
                throw std::runtime_error("Manufactured real EOS failure during caloric discovery.");
            // Independent supplied nonideal oracle: p*(V-n*b)=nRT, e=cv*T.
            const double v = gas_constant * t / p + b;
            const double e = (cv * t + molar_mass * reference) / molar_mass;
            R::MixtureThermodynamicPoint point{molar_mass / v, e, e + (gas_constant * t + b * p) / molar_mass,
                cv / molar_mass, 0, -molar_mass * gas_constant / (p * v * v),
                molar_mass * gas_constant * t / (p * p * v * v), {}, {}};
            for (const auto fraction : x)
                point.fugacities.push_back(fraction * p * std::exp(b * p / (gas_constant * t)));
            return point;
        };
        gas = std::make_shared<const R::MixtureThermodynamics>(catalog(), bounds, R::EOSKind::Gas,
            b == 0 && !opaque ? R::EnthalpyPressureDependence::PressureIndependent
                              : R::EnthalpyPressureDependence::Unspecified,
            evaluate);
        flow = std::make_shared<Flow>(mesh, std::vector{gas}, std::move(options));
        Flow::State initial;
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const double n = 100000 * mesh->cell_volume(static_cast<LO>(c)) / (gas_constant * 300 + 100000 * b);
            initial.cells[c].phases = {{gas, {.7 * n, .3 * n}, 100000, (cv * 300 + molar_mass * reference) * n, 300}};
        }
        flow->initialize(std::move(initial));
    }
    std::vector<Flow::CellSources> injection(bool add_energy = true) const
    {
        std::vector<Flow::CellSources> result(flow->state().cells.size());
        for (size_t c = 0; c < result.size(); ++c)
        {
            const auto& old = flow->state().cells[c].phases[0];
            result[c].species_moles = {old.moles};
            result[c].internal_energy = {add_energy ? old.internal_energy : 0};
        }
        return result;
    }
    double expected_pressure() const { return 200000 * gas_constant * 300 / (gas_constant * 300 - 100000 * b); }
    void check_injection(const Flow::Trial& trial) const
    {
        EXPECT_NEAR(trial.state().thermodynamic_pressure, expected_pressure(), 1e-4);
        EXPECT_LE(trial.diagnostics().maximum_occupancy_error, flow->options().volume_relative_tolerance);
        EXPECT_LE(trial.diagnostics().maximum_continuity_residual, flow->options().continuity_tolerance);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            EXPECT_NEAR(trial.state().cells[c].phases[0].temperature, 300, 1e-7);
            for (size_t i = 0; i < 2; ++i)
                EXPECT_NEAR(
                    trial.state().cells[c].phases[0].moles[i], 2 * flow->state().cells[c].phases[0].moles[i], 1e-10);
        }
    }
};
R::CaloricPressureDomain supplied_covolume_domain(const R::CaloricPressureQuery& query, double b)
{
    // This supplied certificate handles this fixture's one physical phase only.
    if (query.phases.size() != 1)
        throw std::invalid_argument("Fixture certificate requires one physical phase.");
    const auto& phase = query.phases[0];
    const long double n = phase.moles[0] + phase.moles[1], v = query.advected_volumes[0],
                      offset = query.pressure_offsets[0];
    const auto& range = phase.thermodynamics->bounds();
    const long double coefficient = v - n * b;
    if (!(coefficient > 0))
        throw std::invalid_argument("Fixture certificate derivative must be positive.");
    const long double lower =
        (n * ((cv + gas_constant) * range.minimum_temperature + b * offset) - phase.internal_energy - offset * v) /
        coefficient;
    const long double upper =
        (n * ((cv + gas_constant) * range.maximum_temperature + b * offset) - phase.internal_energy - offset * v) /
        coefficient;
    double lo = std::max(query.pressure.lower, static_cast<double>(lower));
    double hi = std::min(query.pressure.upper, static_cast<double>(upper));
    if (static_cast<long double>(lo) < lower)
        lo = std::nextafter(lo, std::numeric_limits<double>::infinity());
    if (static_cast<long double>(hi) > upper)
        hi = std::nextafter(hi, -std::numeric_limits<double>::infinity());
    if (lo > hi)
        return {R::CaloricDomainKnowledge::CertifiedEmpty, std::nullopt};
    return {R::CaloricDomainKnowledge::CertifiedConnected, R::CaloricPressureInterval{lo, hi}};
}

TEST(LowMachCaloricFeasibilityTest, ExactAffineBoundsRetainMechanicalOffsetsAndEnergyReference)
{
    for (const double reference : {0., 1e6})
    {
        const auto gas = ideal(reference);
        const double old_volume = gas_constant * 300 / 100000, offset = 7000;
        const std::vector<R::PhaseThermalState> phases{
            {gas, {2, 0}, 100000, 2 * (cv * 300 + molar_mass * reference), 300}};
        const std::vector<double> volumes{old_volume}, offsets{offset};
        const R::CaloricPressureQuery query{
            phases, volumes, offsets, {20000, 1000000}, R::PhaseThermalMode::Equilibrium};
        const auto domain = R::exact_caloric_pressure_domain(query);
        ASSERT_EQ(domain.knowledge, R::CaloricDomainKnowledge::CertifiedConnected);
        ASSERT_TRUE(domain.interval);
        const double lower = (2 * (cv + gas_constant) * 290 - 2 * cv * 300) / old_volume - offset;
        const double upper = (2 * (cv + gas_constant) * 310 - 2 * cv * 300) / old_volume - offset;
        EXPECT_NEAR(domain.interval->lower, lower, 1e-6);
        EXPECT_NEAR(domain.interval->upper, upper, 1e-6);
        const auto rejected = R::evaluate_caloric_pressure_margins(query, 100000);
        ASSERT_EQ(rejected.lower.size(), 1u);
        EXPECT_LT(rejected.lower[0], 0);
        const auto feasible = R::evaluate_caloric_pressure_margins(query, 200000 - offset);
        EXPECT_GT(feasible.lower[0], 0);
        EXPECT_GT(feasible.upper[0], 0);
    }
}
TEST(LowMachCaloricFeasibilityTest, CommonTemperatureDomainDoesNotIntersectIndependentPhaseEnergies)
{
    const auto gas = ideal();
    const std::vector<R::PhaseThermalState> phases{
        {gas, {1, 0}, 100000, cv * 200, 300}, {gas, {1, 0}, 100000, cv * 400, 300}};
    const double volume = gas_constant * 300 / 100000;
    const std::vector<double> volumes{volume, volume}, offsets{0, 0};
    R::CaloricPressureQuery query{phases, volumes, offsets, {20000, 300000}, R::PhaseThermalMode::Equilibrium};
    const auto common = R::exact_caloric_pressure_domain(query);
    ASSERT_EQ(common.knowledge, R::CaloricDomainKnowledge::CertifiedConnected);
    ASSERT_TRUE(common.interval);
    EXPECT_LE(common.interval->lower, 100000);
    EXPECT_GE(common.interval->upper, 100000);
    query.mode = R::PhaseThermalMode::SeparatePhaseTemperatures;
    EXPECT_EQ(R::exact_caloric_pressure_domain(query).knowledge, R::CaloricDomainKnowledge::CertifiedEmpty);
}
TEST(LowMachCaloricFeasibilityTest, UnknownGasKindDoesNotImplyIdealEnthalpyAndFalseCapabilityRejects)
{
    Fixture opaque(1e-4, true);
    auto phases = opaque.flow->state().cells;
    for (const auto& cell : phases)
    {
        const std::vector<double> volumes{opaque.mesh->cell_volume(0)}, offsets{0};
        const R::CaloricPressureQuery query{
            cell.phases, volumes, offsets, {20000, 1000000}, R::PhaseThermalMode::Equilibrium};
        EXPECT_EQ(R::exact_caloric_pressure_domain(query).knowledge, R::CaloricDomainKnowledge::Unresolved);
    }
    const auto real = opaque.gas;
    const auto false_capability = std::make_shared<const R::MixtureThermodynamics>(catalog(), bounds, R::EOSKind::Gas,
        R::EnthalpyPressureDependence::PressureIndependent,
        [real](double t, double p, std::span<const double> x) { return real->evaluate(t, p, x); });
    EXPECT_THROW((void) false_capability->evaluate(300, 100000, std::vector<double>{1, 0}), std::invalid_argument);
}
TEST(LowMachCaloricFeasibilityTest, ExactlyEmptyPhaseNeverEvaluatesItsEOS)
{
    int calls = 0;
    const auto empty = std::make_shared<const R::MixtureThermodynamics>(catalog(), bounds, R::EOSKind::Liquid,
        [&](double, double, std::span<const double>) -> R::MixtureThermodynamicPoint
        {
            ++calls;
            throw std::runtime_error("Empty phase EOS must not run.");
        });
    const std::vector<R::PhaseThermalState> phases{{ideal(), {1, 0}, 100000, cv * 300, 300}, {empty, {0, 0}, 0, 0, 0}};
    const std::vector<double> volumes{gas_constant * 300 / 100000, 0}, offsets{0, -1e9};
    const auto result =
        R::exact_caloric_pressure_domain({phases, volumes, offsets, {20000, 300000}, R::PhaseThermalMode::Equilibrium});
    EXPECT_EQ(result.knowledge, R::CaloricDomainKnowledge::CertifiedConnected);
    EXPECT_EQ(calls, 0);
}
TEST(LowMachCaloricFeasibilityTest, InitiallyInfeasibleAcceptedPressureFindsAlternateAndRetainsReplay)
{
    for (const auto mode : {R::PhaseThermalMode::Equilibrium, R::PhaseThermalMode::SeparatePhaseTemperatures})
    {
        auto options = controls();
        options.thermal_mode = mode;
        Fixture fixture(0, false, 0, options);
        const auto accepted = fixture.flow->state();
        const auto checkpoint = fixture.flow->checkpoint();
        const auto source = fixture.injection();
        const auto trial = fixture.flow->preview(.01, source);
        fixture.check_injection(trial);
        EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
        EXPECT_DOUBLE_EQ(fixture.flow->state().time, accepted.time);
        fixture.flow->commit(trial);
        const auto expected = fixture.flow->state();
        fixture.flow->restore(checkpoint);
        const auto replay = fixture.flow->preview(.01, source);
        fixture.check_injection(replay);
        EXPECT_DOUBLE_EQ(replay.state().thermodynamic_pressure, expected.thermodynamic_pressure);
    }
}
TEST(LowMachCaloricFeasibilityTest, ReferenceShiftPreservesDiscoveredPressureTemperatureAndFlux)
{
    Fixture original, shifted(0, false, 1e6);
    const auto a = original.flow->preview(.01, original.injection());
    const auto b = shifted.flow->preview(.01, shifted.injection());
    original.check_injection(a);
    shifted.check_injection(b);
    EXPECT_NEAR(a.state().thermodynamic_pressure, b.state().thermodynamic_pressure, 1e-4);
    ASSERT_EQ(a.owned_volume_flux().size(), b.owned_volume_flux().size());
    for (size_t f = 0; f < a.owned_volume_flux().size(); ++f)
        EXPECT_NEAR(a.owned_volume_flux()[f], b.owned_volume_flux()[f], 1e-10);
}
TEST(LowMachCaloricFeasibilityTest, CertifiedEmptyKeepsOriginalInitialRangeErrorAndRetainedTrial)
{
    auto options = controls();
    options.maximum_pressure = 150000;
    Fixture fixture(0, false, 0, options);
    const auto retained = fixture.flow->preview(.01);
    const auto accepted = fixture.flow->state();
    bool rejected = false;
    try
    {
        (void) fixture.flow->preview(.01, fixture.injection(false));
    }
    catch (const std::exception& error)
    {
        rejected = true;
        if (fixture.mesh->num_owned_cells() > 0)
            EXPECT_NE(std::string(error.what()).find("phase enthalpy is outside supplied caloric validity bounds"),
                std::string::npos);
    }
    EXPECT_TRUE(rejected);
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}
TEST(LowMachCaloricFeasibilityTest, RealProviderFailureDuringSeedSearchIsAHardError)
{
    const auto probe = std::make_shared<Probe>();
    probe->fail_above = 150000;
    Fixture fixture(0, false, 0, controls(), probe);
    const auto retained = fixture.flow->preview(.01);
    bool rejected = false;
    try
    {
        (void) fixture.flow->preview(.01, fixture.injection());
    }
    catch (const std::exception& error)
    {
        rejected = true;
        if (fixture.mesh->num_owned_cells() > 0)
            EXPECT_NE(std::string(error.what()).find("Manufactured real EOS failure"), std::string::npos);
    }
    EXPECT_TRUE(rejected);
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}
TEST(LowMachCaloricFeasibilityTest, GenericSearchBudgetExhaustionIsUnresolvedAndNotProofOfNoState)
{
    auto options = controls();
    options.maximum_caloric_seed_evaluations = 1;
    Fixture fixture(1e-4, true, 0, options);
    const auto retained = fixture.flow->preview(.01);
    EXPECT_THROW((void) fixture.flow->preview(.01, fixture.injection()), R::CaloricDomainUnresolved);
    EXPECT_NO_THROW(fixture.flow->commit(retained));
    Fixture resolved(1e-4, true);
    resolved.check_injection(resolved.flow->preview(.01, resolved.injection()));
}
TEST(LowMachCaloricFeasibilityTest, SuppliedNonidealConnectedDomainFindsActualRoot)
{
    constexpr double b = 1e-4;
    auto options = controls();
    options.caloric_domain_provider = std::make_shared<const R::CaloricPressureDomainProvider>(
        R::CaloricPressureDomainProvider{"independent-covolume-certificate", 1,
            [b](const auto& query) { return supplied_covolume_domain(query, b); }});
    Fixture fixture(b, true, 0, options);
    fixture.check_injection(fixture.flow->preview(.01, fixture.injection()));
}
TEST(LowMachCaloricFeasibilityTest, RankLocalHookFailureIsFencedBeforeSeedEvaluationIncludingEmptyOwners)
{
    int hook_calls = 0;
    auto options = controls();
    const int rank = Teuchos::DefaultComm<int>::getComm()->getRank();
    // The communicator-derived rank is computed before entering this MPI-free user hook.
    options.caloric_domain_provider =
        std::make_shared<const R::CaloricPressureDomainProvider>(R::CaloricPressureDomainProvider{
            "fenced-local-domain-hook", 1, [&](const R::CaloricPressureQuery&) -> R::CaloricPressureDomain
            {
                ++hook_calls;
                if (rank == 0)
                    throw std::runtime_error("Manufactured local domain hook failure.");
                return {R::CaloricDomainKnowledge::Unresolved, std::nullopt};
            }});
    Fixture fixture(1e-4, true, 0, options);
    const auto retained = fixture.flow->preview(.01);
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    const auto accepted = fixture.flow->state();
    fixture.probe->off_reference_evaluations = 0;
    int rejected = 0;
    try
    {
        (void) fixture.flow->preview(.01, fixture.injection());
    }
    catch (const std::exception&)
    {
        rejected = 1;
    }
    int all_rejected = 0, global_seeds = 0, empty = fixture.mesh->num_owned_cells() == 0 ? 1 : 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &all_rejected);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &fixture.probe->off_reference_evaluations, &global_seeds);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(all_rejected, comm->getSize());
    EXPECT_EQ(global_seeds, 0);
    if (comm->getSize() == 4)
        EXPECT_EQ(total_empty, 2);
    if (empty)
        EXPECT_EQ(hook_calls, 0);
    EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}
TEST(LowMachCaloricFeasibilityTest, CertifiedHookContradictionIsHardFailure)
{
    auto options = controls();
    options.caloric_domain_provider = std::make_shared<const R::CaloricPressureDomainProvider>(
        R::CaloricPressureDomainProvider{"false-connected-certificate", 1, [](const auto&) -> R::CaloricPressureDomain
            { return {R::CaloricDomainKnowledge::CertifiedConnected, R::CaloricPressureInterval{500000, 600000}}; }});
    Fixture fixture(1e-4, true, 0, options);
    bool rejected = false;
    try
    {
        (void) fixture.flow->preview(.01, fixture.injection());
    }
    catch (const std::exception& error)
    {
        rejected = true;
        if (fixture.mesh->num_owned_cells() > 0)
            EXPECT_NE(std::string(error.what()).find("certificate contradicts actual EOS recovery"), std::string::npos);
    }
    EXPECT_TRUE(rejected);
}
TEST(LowMachCaloricFeasibilityTest, SourceCallbacksStayOutsideAllInnerCaloricAndPressureEvaluations)
{
    Fixture fixture;
    int source_calls = 0;
    const auto source = fixture.injection();
    const auto provider =
        std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{"fixed-source-inner-caloric-search",
            1, fixture.gas->catalog().fingerprint(), [&](const Flow::SourceStageContext& context)
            {
                ++source_calls;
                Flow::SourceStageResult result;
                result.sources = source;
                result.gross_events.assign(context.transport.cells.size(), {1});
                return result;
            }});
    const auto checkpoint = fixture.flow->checkpoint();
    const auto trial = fixture.flow->preview_with_sources(.01, provider);
    fixture.check_injection(trial);
    EXPECT_EQ(source_calls, trial.diagnostics().source_evaluations);
    EXPECT_GE(source_calls, 2);
    const int calls = source_calls;
    fixture.flow->commit(trial);
    fixture.flow->restore(checkpoint);
    EXPECT_EQ(source_calls, calls);
}

TEST(LowMachCaloricFeasibilityTest, ConstructionMetadataRejectsDivergenceIncludingActualEmptyOwners)
{
    Fixture fixture;
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    int empty = fixture.mesh->num_owned_cells() == 0 ? 1 : 0, total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    if (comm->getSize() == 4)
        EXPECT_EQ(total_empty, 2);
    const auto retained = fixture.flow->preview(.01);
    const auto accepted = fixture.flow->state();
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        SCOPED_TRACE(scenario);
        auto options = controls();
        const bool divergent = comm->getRank() == comm->getSize() - 1;
        if (scenario == 0)
        {
            if (divergent)
                options.maximum_caloric_seed_evaluations = 65;
        }
        else if (scenario <= 3)
        {
            const bool present = scenario != 1 || divergent;
            if (present)
                options.caloric_domain_provider = std::make_shared<const R::CaloricPressureDomainProvider>(
                    R::CaloricPressureDomainProvider{scenario == 2 && divergent ? "domain-B" : "domain-A",
                        std::uint64_t(scenario == 3 && divergent ? 2 : 1),
                        [](const R::CaloricPressureQuery&) { return R::CaloricPressureDomain{}; }});
        }
        auto gas = fixture.gas;
        if (scenario == 4 && divergent)
        {
            // Same physical evaluator/catalog; only the immutable capability differs.
            const auto base = fixture.gas;
            const std::shared_ptr<const R::SpeciesCatalog> schema(base, &base->catalog());
            gas = std::make_shared<const R::MixtureThermodynamics>(schema, base->bounds(), base->kind(),
                [base](double t, double p, std::span<const double> x) { return base->evaluate(t, p, x); });
        }
        if (comm->getSize() == 1)
            options.maximum_caloric_seed_evaluations = 0;
        EXPECT_THROW((void) Flow(fixture.mesh, std::vector{gas}, options), std::exception);
        EXPECT_EQ(fixture.flow->state().revision, accepted.revision);
        EXPECT_DOUBLE_EQ(fixture.flow->state().time, accepted.time);
    }
    EXPECT_NO_THROW(fixture.flow->commit(retained));
}
} // namespace
