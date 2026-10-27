#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/OrdinaryAggregate.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>

namespace
{
using SimpleFluid::ArithmeticMode;
using namespace SimpleFluid::PLIC;
using Geometry = ConvexCellGeometry<ArithmeticMode::Ordinary>;
using P = Point<ArithmeticMode::Ordinary>;
constexpr long double relative_allowance = 8e-15L;

GeometryLimits ordinary_limits()
{
    GeometryLimits limits;
    limits.arithmetic_mode = ArithmeticMode::Ordinary;
    return limits;
}
std::vector<OrdinaryAggregateTetra> cube_tetrahedra(GeometryPoint origin = {}, double scale = 1)
{
    std::array<unsigned, 3> axes{0, 1, 2};
    std::vector<OrdinaryAggregateTetra> result;
    do
    {
        OrdinaryAggregateTetra tetra;
        unsigned corner = 0;
        for (unsigned vertex = 0; vertex < 4; ++vertex)
        {
            std::array<long double, 3> point{};
            for (unsigned axis = 0; axis < 3; ++axis)
                point[axis] = static_cast<long double>(origin[axis]) + ((corner >> axis) & 1) * scale;
            tetra.vertices[vertex] = P(point);
            tetra.vertex_ids[vertex] = corner + 1;
            if (vertex < 3)
                corner |= 1u << axes[vertex];
        }
        result.push_back(std::move(tetra));
    } while (std::next_permutation(axes.begin(), axes.end()));
    return result;
}
OrdinaryAggregateDonor cut_donor(
    ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry, GeometryHalfspace plane, long double bulk, long double gas)
{
    OrdinaryAggregateDonor donor;
    donor.id = 37;
    donor.geometry = std::move(geometry);
    donor.gas = plane;
    donor.bulk_budget = bulk * relative_allowance;
    donor.gas_budget = gas * relative_allowance;
    donor.carrier_budget = (bulk - gas) * relative_allowance;
    return donor;
}
void expect_amount(long double observed, long double expected)
{
    EXPECT_LE(std::abs(observed - expected), relative_allowance * std::abs(expected));
}
void expect_split(const OrdinaryAggregateResult& result, long double bulk, long double gas)
{
    ASSERT_EQ(result.groups.size(), 1u);
    const auto& group = result.groups.front();
    EXPECT_EQ(group.donor, 37u);
    EXPECT_EQ(group.sign, 1);
    EXPECT_TRUE(group.bulk_positive);
    EXPECT_TRUE(group.gas_positive);
    EXPECT_TRUE(group.carrier_positive);
    expect_amount(result.signed_source_volume, bulk);
    expect_amount(group.nominal.bulk, bulk);
    expect_amount(group.nominal.gas, gas);
    expect_amount(group.nominal.carrier, bulk - gas);
    EXPECT_LE(std::abs(group.nominal.gas + group.nominal.carrier - group.nominal.bulk), bulk * relative_allowance);
    for (const auto bound :
        {group.approximation_bound.bulk, group.approximation_bound.gas, group.approximation_bound.carrier})
        EXPECT_EQ(bound, 0);
    EXPECT_LE(group.numerical_error_estimate.bulk, bulk * relative_allowance);
    EXPECT_LE(group.numerical_error_estimate.gas, gas * relative_allowance);
    EXPECT_LE(group.numerical_error_estimate.carrier, (bulk - gas) * relative_allowance);
    EXPECT_GT(result.statistics.witnesses, 0u);
    EXPECT_GT(result.statistics.operations, 0u);
}
void expect_no_exact_work(const GeometryStatistics& before)
{
    const auto after = geometry_statistics();
    EXPECT_EQ(after.exact_predicates, before.exact_predicates);
    EXPECT_EQ(after.rational_constructions, before.rational_constructions);
    EXPECT_EQ(after.rational_measures, before.rational_measures);
    EXPECT_EQ(after.rational_ledgers, before.rational_ledgers);
}
template<class F> void expect_error(F&& action, GeometryErrorCode code)
{
    try
    {
        action();
        ADD_FAILURE() << "Expected a typed aggregate rejection.";
    }
    catch (const GeometryError& error)
    {
        EXPECT_EQ(error.code(), code) << error.what();
    }
}

TEST(OrdinaryAggregateTest, SixTetraCubeRetainsAnalyticMinoritySlabsAcrossFrames)
{
    const auto before = geometry_statistics();
    const std::array<GeometryPoint, 3> origins{{{0, 0, 0}, {0x1p20, -0x1p18, 0x1p19}, {-8, 4, 2}}};
    const std::array<double, 3> scales{1, 4, 0x1p-8};
    for (unsigned frame = 0; frame < origins.size(); ++frame)
    {
        const auto origin = origins[frame];
        const auto scale = scales[frame];
        auto high = origin;
        for (unsigned axis = 0; axis < 3; ++axis)
            high[axis] += scale;
        const auto geometry = Geometry::box(origin, high);
        const auto tetrahedra = cube_tetrahedra(origin, scale);
        const OrdinaryAggregateSource source(tetrahedra, {}, ordinary_limits());
        const long double bulk = static_cast<long double>(scale) * scale * scale;
        for (const unsigned exponent : {1u, 12u, 30u, 50u})
        {
            SCOPED_TRACE(::testing::Message() << "frame=" << frame << ", exponent=" << exponent);
            const auto h = std::ldexp(1., -int(exponent));
            const std::array donors{cut_donor(geometry, {{1, 0, 0}, origin, h * scale}, bulk, bulk * h)};
            const auto result = source.measure(donors);
            expect_split(result, bulk, bulk * h);
            EXPECT_EQ(result.statistics.source_pieces, 6u);
            EXPECT_EQ(result.statistics.cancelled_boundaries, 6u);
            EXPECT_GE(result.statistics.storage_bytes, source.storage_bytes());
        }
    }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, CornerSimplexRetainsCubicMinorityWithoutMeasuringChildren)
{
    const auto before = geometry_statistics();
    const std::array<OrdinaryAggregateTetra, 1> tetrahedra{
        OrdinaryAggregateTetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {1, 2, 3, 4}, 1}};
    const OrdinaryAggregateSource source(tetrahedra, {}, ordinary_limits());
    const auto donor_geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    for (const unsigned exponent : {1u, 12u, 30u, 50u})
    {
        SCOPED_TRACE(exponent);
        const auto h = std::ldexp(1., -int(exponent));
        const auto gas = static_cast<long double>(h) * h * h / 6;
        const std::array donors{cut_donor(donor_geometry, {{1, 1, 1}, {}, h}, 1.L / 6, gas)};
        expect_split(source.measure(donors), 1.L / 6, gas);
    }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, ComplementaryDonorAndPhaseContactRetainsAnalyticTetraVolumes)
{
    const auto before = geometry_statistics();
    const std::array<OrdinaryAggregateTetra, 1> recipes{
        OrdinaryAggregateTetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {1, 2, 3, 4}, 1}};
    const OrdinaryAggregateSource source(recipes, {}, ordinary_limits());
    for (unsigned donor_axis : {0u, 1u})
    {
        // Swapping the two geometric roles exercises the reversed support
        // construction as well as reversing the public donor enumeration.
        GeometryPoint donor_normal{}, phase_normal{};
        donor_normal[donor_axis] = 1;
        phase_normal[1 - donor_axis] = 1;
        const auto cube = Geometry::box({0, 0, 0}, {1, 1, 1});
        const auto split = cube->clip({donor_normal, {}, .5}, false);
        ASSERT_TRUE(split.inside);
        ASSERT_TRUE(split.outside);
        for (const double h : {.5, .5 - 0x1p-20, .5 + 0x1p-20})
        {
            SCOPED_TRACE(::testing::Message() << "donor_axis=" << donor_axis << ", phase=" << h);
            const long double missing = 1 - static_cast<long double>(h);
            const long double overlap = std::max(.5L - h, 0.L);
            // Inclusion-exclusion of the two removed corner tetrahedra.
            const long double carrier = (missing * missing * missing - overlap * overlap * overlap) / 6;
            const long double bulk = 7.L / 48, gas = h == .5 ? 1.L / 8 : bulk - carrier;
            if (h == .5)
            {
                ASSERT_EQ(gas, 1.L / 8);
                ASSERT_EQ(carrier, 1.L / 48);
            }
            std::array donors{cut_donor(split.inside, {phase_normal, {}, h}, bulk, gas), OrdinaryAggregateDonor{}};
            donors[1].id = 38;
            donors[1].geometry = split.outside;
            donors[1].pure_carrier = true;
            donors[1].bulk_budget = donors[1].carrier_budget = relative_allowance / 48;
            for (unsigned order = 0; order < 2; ++order)
            {
                SCOPED_TRACE(order);
                const auto result = source.measure(donors); // Mandatory success, including exact contact.
                ASSERT_EQ(result.groups.size(), 2u);
                expect_amount(result.signed_source_volume, 1.L / 6);
                for (const auto& group : result.groups)
                {
                    const bool left = group.donor == 37;
                    ASSERT_TRUE(left || group.donor == 38);
                    expect_amount(group.nominal.bulk, left ? bulk : 1.L / 48);
                    expect_amount(group.nominal.gas, left ? gas : 0);
                    expect_amount(group.nominal.carrier, left ? carrier : 1.L / 48);
                    EXPECT_TRUE(group.bulk_positive);
                    EXPECT_EQ(group.gas_positive, left);
                    EXPECT_TRUE(group.carrier_positive);
                    EXPECT_LE(group.numerical_error_estimate.bulk, relative_allowance * (left ? bulk : 1.L / 48));
                    EXPECT_LE(group.numerical_error_estimate.gas, left ? relative_allowance * gas : 0);
                    EXPECT_LE(group.numerical_error_estimate.carrier, relative_allowance * (left ? carrier : 1.L / 48));
                    EXPECT_EQ(group.approximation_bound.bulk, 0);
                    EXPECT_EQ(group.approximation_bound.gas, 0);
                    EXPECT_EQ(group.approximation_bound.carrier, 0);
                }
                std::reverse(donors.begin(), donors.end());
            }
        }
    }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, NonbinaryCrossingReplaysSourceSupportToResolveExactContact)
{
    const auto before = geometry_statistics();
    const std::array<OrdinaryAggregateTetra, 1> recipes{
        OrdinaryAggregateTetra{{P({0, 0, 0}), P({3, 0, 0}), P({0, 3, 0}), P({0, 0, 3})}, {1, 2, 3, 4}, 1}};
    const OrdinaryAggregateSource source(recipes, {}, ordinary_limits());
    for (unsigned donor_axis : {0u, 1u})
    {
        GeometryPoint donor_normal{}, phase_normal{};
        donor_normal[donor_axis] = 1;
        phase_normal[1 - donor_axis] = 1;
        const double a = donor_axis == 0 ? 1 : 2, h = 3 - a;
        const auto split = Geometry::box({0, 0, 0}, {3, 3, 3})->clip({donor_normal, {}, a}, false);
        // At contact the source edge has weights 1/3 and 2/3. Rounded
        // barycentric placement alone cannot prove the second incidence.
        const long double removed = 3 - a;
        const long double right = removed * removed * removed / 6, bulk = 4.5L - right;
        for (const double delta : {0., -0x1p-20, 0x1p-20})
        {
            SCOPED_TRACE(::testing::Message() << "donor_axis=" << donor_axis << ", delta=" << delta);
            const long double missing = 3.L - (h + delta), overlap = std::max(-static_cast<long double>(delta), 0.L);
            const long double carrier = (missing * missing * missing - overlap * overlap * overlap) / 6;
            const long double gas = delta == 0 ? 3.L : bulk - carrier;
            std::array donors{
                cut_donor(split.inside, {phase_normal, {}, h + delta}, bulk, gas), OrdinaryAggregateDonor{}};
            donors[1].id = 38;
            donors[1].geometry = split.outside;
            donors[1].pure_carrier = true;
            donors[1].bulk_budget = donors[1].carrier_budget = right * relative_allowance;
            for (unsigned order = 0; order < 2; ++order)
            {
                const auto result = source.measure(donors);
                ASSERT_EQ(result.groups.size(), 2u);
                expect_amount(result.signed_source_volume, 4.5L);
                for (const auto& group : result.groups)
                {
                    const bool left = group.donor == 37;
                    ASSERT_TRUE(left || group.donor == 38);
                    expect_amount(group.nominal.bulk, left ? bulk : right);
                    expect_amount(group.nominal.gas, left ? gas : 0);
                    expect_amount(group.nominal.carrier, left ? carrier : right);
                    EXPECT_EQ(group.gas_positive, left);
                    EXPECT_TRUE(group.bulk_positive);
                    EXPECT_TRUE(group.carrier_positive);
                    EXPECT_EQ(group.approximation_bound.bulk, 0);
                    EXPECT_EQ(group.approximation_bound.gas, 0);
                    EXPECT_EQ(group.approximation_bound.carrier, 0);
                }
                std::reverse(donors.begin(), donors.end());
            }
        }
    }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, VertexPermutationsAndSourceReorderingRetainAnalyticSlab)
{
    const auto before = geometry_statistics();
    auto tetrahedra = cube_tetrahedra();
    constexpr long double gas = 0x1p-20L;
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, double(gas)}, 1, gas)};
    for (unsigned permutation = 0; permutation < 4; ++permutation)
    {
        SCOPED_TRACE(permutation);
        const OrdinaryAggregateSource source(tetrahedra, {}, ordinary_limits());
        expect_split(source.measure(donors), 1, gas);
        for (auto& tetra : tetrahedra)
        {
            std::rotate(tetra.vertices.begin(), tetra.vertices.begin() + 1, tetra.vertices.end());
            std::rotate(tetra.vertex_ids.begin(), tetra.vertex_ids.begin() + 1, tetra.vertex_ids.end());
        }
        std::reverse(tetrahedra.begin(), tetrahedra.end());
    }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, SharedSourceFaceRemainsExteriorWhenOtherSideBecomesContact)
{
    const auto before = geometry_statistics();
    // Two unit-base simplices meet on OBC. Their first vertices are the
    // opposite apices, so dropping OBC changes either one-sided integral.
    std::array<OrdinaryAggregateTetra, 2> tetrahedra{
        OrdinaryAggregateTetra{{P({0, 0, 1}), P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0})}, {2, 1, 3, 4}, 1},
        OrdinaryAggregateTetra{{P({0, 0, -1}), P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0})}, {5, 1, 3, 4}, 1}};
    const std::array phase_donors{
        cut_donor(Geometry::box({0, 0, -1}, {1, 1, 1}), {{0, 0, -1}, {}, 0}, 1.L / 3, 1.L / 6)};
    OrdinaryAggregateDonor upper;
    upper.id = 37;
    upper.geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    upper.pure_gas = true;
    upper.bulk_budget = upper.gas_budget = relative_allowance / 6;
    OrdinaryAggregateDonor lower;
    lower.id = 38;
    lower.geometry = Geometry::box({0, 0, -1}, {1, 1, 0});
    lower.pure_carrier = true;
    lower.bulk_budget = lower.carrier_budget = relative_allowance / 6;
    // Both donors authenticate complete source coverage; each one separately
    // reduces the other simplex to contact on the shared source boundary.
    std::array spatial_donors{upper, lower};
    for (unsigned permutation = 0; permutation < 4; ++permutation)
    {
        SCOPED_TRACE(permutation);
        const OrdinaryAggregateSource source(tetrahedra, {}, ordinary_limits());
        const auto phase_split = source.measure(phase_donors);
        expect_split(phase_split, 1.L / 3, 1.L / 6);
        EXPECT_EQ(phase_split.statistics.cancelled_boundaries, 1u);
        const auto spatial_split = source.measure(spatial_donors);
        ASSERT_EQ(spatial_split.groups.size(), 2u);
        expect_amount(spatial_split.signed_source_volume, 1.L / 3);
        for (const auto& group : spatial_split.groups)
        {
            ASSERT_TRUE(group.donor == 37 || group.donor == 38);
            const bool gas = group.donor == 37;
            EXPECT_EQ(group.sign, 1);
            EXPECT_TRUE(group.bulk_positive);
            EXPECT_EQ(group.gas_positive, gas);
            EXPECT_EQ(group.carrier_positive, !gas);
            expect_amount(group.nominal.bulk, 1.L / 6);
            expect_amount(group.nominal.gas, gas ? 1.L / 6 : 0);
            expect_amount(group.nominal.carrier, gas ? 0 : 1.L / 6);
            EXPECT_LE(group.numerical_error_estimate.bulk, relative_allowance / 6);
            EXPECT_LE(group.numerical_error_estimate.gas, gas ? relative_allowance / 6 : 0);
            EXPECT_LE(group.numerical_error_estimate.carrier, gas ? 0 : relative_allowance / 6);
            EXPECT_EQ(group.approximation_bound.bulk, 0);
            EXPECT_EQ(group.approximation_bound.gas, 0);
            EXPECT_EQ(group.approximation_bound.carrier, 0);
        }
        for (auto& tetra : tetrahedra)
        {
            std::rotate(tetra.vertices.begin() + 1, tetra.vertices.begin() + 2, tetra.vertices.end());
            std::rotate(tetra.vertex_ids.begin() + 1, tetra.vertex_ids.begin() + 2, tetra.vertex_ids.end());
        }
        std::reverse(tetrahedra.begin(), tetrahedra.end());
        std::reverse(spatial_donors.begin(), spatial_donors.end());
    }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, SourceIdentitiesAndRepeatedIncidenceRejectWithoutChangingRecipes)
{
    const auto good = cube_tetrahedra();
    auto changed = good;
    changed[0].vertices[0] = P({.125, 0, 0});
    OrdinaryAggregateContext context;
    expect_error([&] { (void) OrdinaryAggregateSource(changed, {}, ordinary_limits(), &context); },
        GeometryErrorCode::ProvenanceMismatch);
    EXPECT_EQ(context.stage, OrdinaryAggregateContext::Stage::Source);
    EXPECT_NE(context.source, OrdinaryAggregateContext::absent);
    EXPECT_EQ(context.donor, OrdinaryAggregateContext::absent);
    EXPECT_GT(context.bytes, 0u);
    auto repeated = good;
    repeated.push_back(good.front());
    expect_error(
        [&] { (void) OrdinaryAggregateSource(repeated, {}, ordinary_limits()); }, GeometryErrorCode::InvalidTopology);
    auto repeated_vertex = good;
    repeated_vertex[0].vertex_ids[1] = repeated_vertex[0].vertex_ids[0];
    expect_error([&] { (void) OrdinaryAggregateSource(repeated_vertex, {}, ordinary_limits()); },
        GeometryErrorCode::ProvenanceMismatch);
    const OrdinaryAggregateSource valid(good, {}, ordinary_limits());
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .5}, 1, .5)};
    expect_split(valid.measure(donors), 1, .5);
}

TEST(OrdinaryAggregateTest, ExhaustedSourceAndMeasurementBudgetsRejectExplicitly)
{
    const auto tetrahedra = cube_tetrahedra();
    auto sweep = SweepLimits{};
    sweep.byte_budget = 1;
    expect_error([&] { (void) OrdinaryAggregateSource(tetrahedra, sweep, ordinary_limits()); },
        GeometryErrorCode::ResourceLimit);
    sweep = {};
    sweep.maximum_pair_tests = 0;
    expect_error([&] { (void) OrdinaryAggregateSource(tetrahedra, sweep, ordinary_limits()); },
        GeometryErrorCode::ResourceLimit);
    sweep.maximum_pair_tests = 24; // Six tetrahedra contribute four authenticated source faces each.
    const OrdinaryAggregateSource source(tetrahedra, sweep, ordinary_limits());
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .5}, 1, .5)};
    OrdinaryAggregateContext context;
    expect_error([&] { (void) source.measure(donors, &context); }, GeometryErrorCode::ResourceLimit);
    EXPECT_GE(context.bytes, source.storage_bytes());
    EXPECT_EQ(context.pairs, 24u);
    const OrdinaryAggregateSource baseline(tetrahedra, {}, ordinary_limits());
    sweep = {};
    sweep.byte_budget = baseline.storage_bytes();
    const OrdinaryAggregateSource retained(tetrahedra, sweep, ordinary_limits());
    expect_error([&] { (void) retained.measure(donors); }, GeometryErrorCode::ResourceLimit);
}

TEST(OrdinaryAggregateTest, UnresolvedMinorityBudgetRejectsInsteadOfBecomingAbsent)
{
    const OrdinaryAggregateTetra tetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {0, 1, 2, 3}, 1};
    const OrdinaryAggregateSource source(std::span(&tetra, 1), {}, ordinary_limits());
    // A positive corner cannot be turned into an absent phase when its
    // independent numerical accuracy budget cannot be met.
    const double height = std::numeric_limits<double>::denorm_min();
    auto donor = cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 1, 1}, {}, height}, 1.L / 6, 1.L / 12);
    // Exhaust the independent phase budget without changing the source.
    donor.gas_budget = std::numeric_limits<long double>::denorm_min();
    const std::array limited{donor};
    expect_error([&] { (void) source.measure(limited); }, GeometryErrorCode::UnresolvedArithmetic);
    EXPECT_EQ(tetra.vertices[1].value[0], 1);
}

TEST(OrdinaryAggregateTest, UnrepresentedCutDistanceRemainsTypedUnresolved)
{
    const auto before = geometry_statistics();
    // The source determinant is exactly one, but the positive phase distance
    // at vertex 1 lies below the represented range. It cannot become contact.
    const auto delta = std::ldexp(1.L, std::numeric_limits<long double>::min_exponent + 4);
    ASSERT_EQ(std::fpclassify(delta), FP_NORMAL);
    const std::array<OrdinaryAggregateTetra, 1> recipes{
        OrdinaryAggregateTetra{{P({0, 0, 0}), P({1, 0, delta}), P({0, 1, -1}), P({0, 0, 1})}, {1, 2, 3, 4}, 1}};
    const OrdinaryAggregateSource source(recipes, {}, ordinary_limits());
    const std::array donors{
        cut_donor(Geometry::box({0, 0, -1}, {1, 1, 1}), {{0, 0, 0x1p-300}, {}, 0}, 1.L / 6, 1.L / 12)};
    expect_error([&] { (void) source.measure(donors); }, GeometryErrorCode::UnresolvedArithmetic);
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, RestoresCallerRoundingAndFlagsAfterSuccessAndRejection)
{
    const auto recipes = cube_tetrahedra();
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .25}, 1, .25)};
    struct Environment
    {
        std::fenv_t saved{};
        Environment() { std::fegetenv(&saved); }
        ~Environment() { std::fesetenv(&saved); }
    } restore;
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INEXACT);
    const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
    const OrdinaryAggregateSource source(recipes, {}, ordinary_limits());
    const auto result = source.measure(donors);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    ASSERT_EQ(result.groups.size(), 1u);
    EXPECT_EQ(result.groups[0].nominal.gas, .25L);
    auto limits = ordinary_limits();
    limits.byte_budget = 1;
    expect_error([&] { (void) OrdinaryAggregateSource(recipes, {}, limits); }, GeometryErrorCode::ResourceLimit);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
}

TEST(OrdinaryAggregateTest, PositiveSubnormalSimplexCannotClaimZeroErrorOrRelativeAllowance)
{
    using Limits = std::numeric_limits<long double>;
    if (Limits::radix != 2 || Limits::digits != 64 || Limits::min_exponent != -16381)
        GTEST_SKIP() << "The independently derived x87 subnormal counterexample requires this long-double format.";
    const auto edge = std::ldexp(1.L, -5480);
    ASSERT_EQ(std::fpclassify(edge), FP_NORMAL);
    // The exact determinant is 2^-16440 = 32 denorm_min. The exact
    // simplex volume is (16/3) denorm_min, while nearest storage gives
    // 5 denorm_min: a 1/16 relative error, far beyond the phase gate.
    const auto rounded_volume = std::ldexp(1.L, -16440) / 6;
    ASSERT_EQ(rounded_volume, 5 * Limits::denorm_min());
    ASSERT_GT(rounded_volume, 0);
    ASSERT_EQ(rounded_volume * relative_allowance, 0);
    const std::array<OrdinaryAggregateTetra, 1> recipes{
        OrdinaryAggregateTetra{{P({0, 0, 0}), P({edge, 0, 0}), P({0, edge, 0}), P({0, 0, edge})}, {1, 2, 3, 4}, 1}};
    const auto before = geometry_statistics();
    expect_error(
        [&]
        {
            const OrdinaryAggregateSource source(recipes, {}, ordinary_limits());
            OrdinaryAggregateDonor donor;
            donor.id = 51;
            donor.geometry = Geometry::box({-1, -1, -1}, {1, 1, 1});
            donor.pure_gas = true;
            donor.bulk_budget = donor.gas_budget = 1;
            (void) source.measure(std::span(&donor, 1));
        },
        GeometryErrorCode::UnresolvedArithmetic);
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, SourceDomainRetryRetainsAnalyticFlatTetraWithDistantUnrelatedSource)
{
    const auto before = geometry_statistics();
    constexpr long double height = 0x1p-60L;
    constexpr double width = .25;
    const std::array<OrdinaryAggregateTetra, 2> recipes{
        OrdinaryAggregateTetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({1, 1, height})}, {1, 2, 3, 4}, 1},
        OrdinaryAggregateTetra{
            {P({-2, -2, 1}), P({-1.5, -2, 1}), P({-2, -1.5, 1}), P({-2, -2, 1.5})}, {5, 6, 7, 8}, 1}};
    const auto original = recipes;
    const long double bulk = height / 6, gas = height * width * width * width / 12;
    // Under the source affine map x+y=lambda1+lambda2+2lambda3.
    // For width<=1 the reference cut is a tetra with fraction width^3/2.
    std::array<OrdinaryAggregateDonor, 2> donors;
    donors[0] = cut_donor(
        Geometry::box({0, 0, -double(height)}, {1, 1, 2 * double(height)}), {{1, 1, 0}, {}, width}, bulk, gas);
    donors[1].id = 91;
    donors[1].geometry = Geometry::box({-3, -3, .5}, {-1, -1, 2});
    donors[1].pure_carrier = true;
    donors[1].bulk_budget = donors[1].carrier_budget = 1e-14L;
    const OrdinaryAggregateSource source(recipes, {}, ordinary_limits());
    const auto result = source.measure(donors);
    const auto found =
        std::find_if(result.groups.begin(), result.groups.end(), [](const auto& group) { return group.donor == 37; });
    ASSERT_NE(found, result.groups.end());
    expect_amount(found->nominal.bulk, bulk);
    expect_amount(found->nominal.gas, gas);
    expect_amount(found->nominal.carrier, bulk - gas);
    EXPECT_GT(result.statistics.retries, 0u);
    EXPECT_TRUE(found->bulk_positive);
    EXPECT_TRUE(found->gas_positive);
    EXPECT_TRUE(found->carrier_positive);
    EXPECT_LE(found->numerical_error_estimate.bulk, donors[0].bulk_budget);
    EXPECT_LE(found->numerical_error_estimate.gas, donors[0].gas_budget);
    EXPECT_LE(found->numerical_error_estimate.carrier, donors[0].carrier_budget);
    EXPECT_EQ(found->approximation_bound.bulk, 0);
    EXPECT_EQ(found->approximation_bound.gas, 0);
    EXPECT_EQ(found->approximation_bound.carrier, 0);
    for (unsigned tetra = 0; tetra < recipes.size(); ++tetra)
        for (unsigned vertex = 0; vertex < 4; ++vertex)
        {
            EXPECT_EQ(recipes[tetra].vertices[vertex].value, original[tetra].vertices[vertex].value);
            EXPECT_EQ(recipes[tetra].vertex_ids[vertex], original[tetra].vertex_ids[vertex]);
        }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, CapturedPositiveThinEdgeRequiresMinorityAccuracyOrExplicitQualityRejection)
{
    const GeometryPoint origin{1.4500000000000002, 1.5, .5};
    const std::array<std::array<long double, 3>, 4> local{
        {{-1.11022302462515654042e-16L, 0, 0}, {-6.63531729561128713613e-17L, -1.45933612416548896817e-16L, 0},
            {-.150000000000000088666L, -.500000000000000145934L, -.5L},
            {-.150000000000000088666L, -.500000000000000145934L, .5L}}};
    OrdinaryAggregateTetra tetra;
    for (unsigned vertex = 0; vertex < 4; ++vertex)
    {
        auto point = local[vertex];
        for (unsigned axis = 0; axis < 3; ++axis)
            point[axis] += origin[axis];
        tetra.vertices[vertex] = P(point);
        tetra.vertex_ids[vertex] = vertex + 1;
    }
    // This oracle uses the represented global source, rather than assuming
    // adding a local construction origin preserves every input bit.
    const auto& p = tetra.vertices;
    const auto bulk = std::abs((p[1].value[0] - p[0].value[0]) * (p[2].value[1] - p[0].value[1]) -
                               (p[1].value[1] - p[0].value[1]) * (p[2].value[0] - p[0].value[0])) /
                      6;
    const long double a = 1 - p[2].value[1];
    const auto t0 = a / (a + p[0].value[1] - 1), t1 = a / (a + p[1].value[1] - 1);
    const auto gas = bulk * t0 * t1 * (3 - t0 - t1);
    ASSERT_GT(bulk, 0);
    ASSERT_GT(gas, 0);
    ASSERT_LT(gas, 0x1p-150L);
    const std::array tetrahedra{tetra};
    const OrdinaryAggregateSource source(tetrahedra, {}, ordinary_limits());
    const std::array donors{cut_donor(Geometry::box({0, 0, -1}, {2, 2, 2}), {{0, 1, 0}, {0, 1, 0}, 0}, bulk, gas)};
    const auto before = geometry_statistics();
    try
    {
        expect_split(source.measure(donors), bulk, gas);
        RecordProperty("quality_rejected", false);
    }
    catch (const GeometryError& error)
    {
        EXPECT_EQ(error.code(), GeometryErrorCode::UnresolvedArithmetic) << error.what();
        RecordProperty("quality_rejected", true);
    }
    expect_no_exact_work(before);
}
} // namespace
