#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numeric>
#include <type_traits>

namespace
{
using SimpleFluid::ArithmeticMode;
using namespace SimpleFluid::PLIC;
using Geometry = ConvexCellGeometry<ArithmeticMode::Ordinary>;
using P = Point<ArithmeticMode::Ordinary>;

// Historical analytic fixtures retain their explicit identities here only.
// Production consumes one indexed table; this builder checks fixture consistency
// and preserves the first occurrence and local tetrahedron order.
struct FixtureTetra
{
    std::array<P, 4> vertices;
    std::array<std::uint64_t, 4> vertex_ids{};
    int sign = 1;
};
SimpleFluid::geometry::swept_volume::OrdinaryAggregateSourceBatch make_batch(std::span<const FixtureTetra> fixtures)
{
    using namespace SimpleFluid::geometry::swept_volume;
    OrdinaryAggregateSourceBatch batch;
    std::map<std::uint64_t, OrdinaryAggregatePointIndex> indices;
    for (const auto& fixture : fixtures)
    {
        OrdinaryAggregateTetra tetra;
        tetra.sign = fixture.sign;
        for (unsigned k = 0; k < 4; ++k)
        {
            auto [position, fresh] =
                indices.try_emplace(fixture.vertex_ids[k], OrdinaryAggregatePointIndex{batch.points.size()});
            if (fresh)
                batch.points.push_back(fixture.vertices[k]);
            else if (batch.points[position->second.value].value != fixture.vertices[k].value ||
                     batch.points[position->second.value].supports() != fixture.vertices[k].supports())
                throw std::logic_error("Analytic fixture identity has inconsistent points.");
            tetra.vertices[k] = position->second;
        }
        batch.tetrahedra.push_back(tetra);
    }
    return batch;
}
constexpr long double relative_allowance = 8e-15L;

GeometryLimits ordinary_limits()
{
    GeometryLimits limits;
    limits.arithmetic_mode = ArithmeticMode::Ordinary;
    return limits;
}
std::vector<FixtureTetra> cube_tetrahedra(GeometryPoint origin = {}, double scale = 1)
{
    std::array<unsigned, 3> axes{0, 1, 2};
    std::vector<FixtureTetra> result;
    do
    {
        FixtureTetra tetra;
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
SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor cut_donor(
    ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry, GeometryHalfspace plane, long double bulk, long double gas)
{
    SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor donor;
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
void expect_split(
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateResult& result, long double bulk, long double gas)
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
void expect_identical_results(const SimpleFluid::geometry::swept_volume::OrdinaryAggregateResult& a,
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateResult& b)
{
    using Group = SimpleFluid::geometry::swept_volume::OrdinaryAggregateGroup;
    EXPECT_EQ(b.signed_source_volume, a.signed_source_volume);
    ASSERT_EQ(b.groups.size(), a.groups.size());
    for (std::size_t i = 0; i < a.groups.size(); ++i)
    {
        const auto& first = a.groups[i];
        const auto& again = b.groups[i];
        EXPECT_EQ(again.donor, first.donor);
        EXPECT_EQ(again.sign, first.sign);
        for (const auto member : {&Group::nominal, &Group::numerical_error_estimate, &Group::approximation_bound})
        {
            EXPECT_EQ((again.*member).bulk, (first.*member).bulk);
            EXPECT_EQ((again.*member).gas, (first.*member).gas);
            EXPECT_EQ((again.*member).carrier, (first.*member).carrier);
        }
        EXPECT_EQ(again.bulk_positive, first.bulk_positive);
        EXPECT_EQ(again.gas_positive, first.gas_positive);
        EXPECT_EQ(again.carrier_positive, first.carrier_positive);
    }
    EXPECT_EQ(b.statistics.storage_bytes, a.statistics.storage_bytes);
    EXPECT_EQ(b.statistics.pair_tests, a.statistics.pair_tests);
    EXPECT_EQ(b.statistics.operations, a.statistics.operations);
    EXPECT_EQ(b.statistics.cancelled_boundaries, a.statistics.cancelled_boundaries);
    EXPECT_EQ(b.statistics.failed_charge_bytes, a.statistics.failed_charge_bytes);
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

void expect_compact_accounting(const SimpleFluid::geometry::swept_volume::OrdinaryAggregateStatistics& statistics)
{
    EXPECT_EQ(
        std::accumulate(statistics.stage_charged_bytes.begin(), statistics.stage_charged_bytes.end(), std::size_t{}),
        statistics.storage_bytes);
    EXPECT_EQ(statistics.failed_charge_bytes, 0u);
    // Both counts are observed independently from actual face references.
    EXPECT_EQ(statistics.stored_vertices_seen, statistics.active_vertices_seen);
    EXPECT_EQ(statistics.peak_stored_vertices, statistics.peak_active_vertices);
    EXPECT_EQ(statistics.vertices_classified, statistics.active_vertices_seen);
    EXPECT_GT(statistics.peak_live_witnesses, 0u);
    EXPECT_GT(statistics.peak_live_witness_bytes, 0u);
    EXPECT_GT(statistics.peak_remainders, 0u);
}

using Index = SimpleFluid::geometry::swept_volume::OrdinaryAggregatePointIndex;
using Batch = SimpleFluid::geometry::swept_volume::OrdinaryAggregateSourceBatch;
static_assert(!std::is_convertible_v<std::size_t, Index>);
static_assert(!std::is_convertible_v<Index, std::size_t>);
static_assert(std::is_same_v<
    decltype(std::declval<const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource&>().points()),
    std::span<const P>>);

Batch indexed_corner()
{
    Batch batch;
    batch.points = {P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})};
    SimpleFluid::geometry::swept_volume::OrdinaryAggregateTetra tetra;
    for (unsigned k = 0; k < 4; ++k)
        tetra.vertices[k] = Index{k};
    batch.tetrahedra.push_back(tetra);
    return batch;
}
Batch indexed_cube()
{
    Batch batch;
    batch.points.reserve(8);
    batch.tetrahedra.reserve(6);
    for (std::size_t corner = 0; corner < 8; ++corner)
        batch.points.emplace_back(std::array<long double, 3>{static_cast<long double>(corner & 1),
            static_cast<long double>((corner >> 1) & 1), static_cast<long double>((corner >> 2) & 1)});
    std::array<unsigned, 3> axes{0, 1, 2};
    do
    {
        SimpleFluid::geometry::swept_volume::OrdinaryAggregateTetra tetra;
        std::size_t corner = 0;
        for (unsigned k = 0; k < 4; ++k)
        {
            tetra.vertices[k] = Index{corner};
            if (k < 3)
                corner |= std::size_t{1} << axes[k];
        }
        batch.tetrahedra.push_back(tetra);
    } while (std::next_permutation(axes.begin(), axes.end()));
    return batch;
}

Batch indexed_bipyramid()
{
    auto batch = indexed_corner();
    batch.points.emplace_back(std::array<long double, 3>{0, 0, -1});
    auto below = batch.tetrahedra.front();
    below.vertices[3] = Index{4};
    batch.tetrahedra.push_back(below);
    return batch;
}

struct BatchSnapshot
{
    Batch value;
    const P* points;
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateTetra* tetrahedra;
    std::size_t point_capacity, tetra_capacity;
    explicit BatchSnapshot(const Batch& batch)
        : value(batch), points(batch.points.data()), tetrahedra(batch.tetrahedra.data()),
          point_capacity(batch.points.capacity()), tetra_capacity(batch.tetrahedra.capacity())
    {
    }
    void expect_unchanged(const Batch& batch) const
    {
        EXPECT_EQ(batch.points.data(), points);
        EXPECT_EQ(batch.tetrahedra.data(), tetrahedra);
        EXPECT_EQ(batch.points.capacity(), point_capacity);
        EXPECT_EQ(batch.tetrahedra.capacity(), tetra_capacity);
        ASSERT_EQ(batch.points.size(), value.points.size());
        ASSERT_EQ(batch.tetrahedra.size(), value.tetrahedra.size());
        for (std::size_t i = 0; i < batch.points.size(); ++i)
        {
            for (unsigned axis = 0; axis < 3; ++axis)
                if (std::isnan(value.points[i].value[axis]))
                    EXPECT_TRUE(std::isnan(batch.points[i].value[axis]));
                else
                    EXPECT_EQ(batch.points[i].value[axis], value.points[i].value[axis]);
            EXPECT_EQ(batch.points[i].supports(), value.points[i].supports());
        }
        for (std::size_t i = 0; i < batch.tetrahedra.size(); ++i)
        {
            EXPECT_EQ(batch.tetrahedra[i].vertices, value.tetrahedra[i].vertices);
            EXPECT_EQ(batch.tetrahedra[i].sign, value.tetrahedra[i].sign);
        }
    }
};

TEST(OrdinaryAggregateTest, IndexedCubeUsesEightOwnedPointsAndPreservesAnalyticSlabsUnderPermutations)
{
    using namespace SimpleFluid::geometry::swept_volume;
    const auto before = geometry_statistics();
    auto recipe = indexed_cube();
    ASSERT_EQ(recipe.points.size(), 8u);
    ASSERT_EQ(recipe.tetrahedra.size(), 6u);
    const auto geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    for (unsigned permutation = 0; permutation < 4; ++permutation)
    {
        auto batch = recipe;
        const auto point_buffer = batch.points.data();
        const auto tetra_buffer = batch.tetrahedra.data();
        const auto original = batch;
        const OrdinaryAggregateSource source(std::move(batch), {}, ordinary_limits());
        ASSERT_EQ(source.points().size(), 8u);
        ASSERT_EQ(source.tetrahedra().size(), 6u);
        EXPECT_EQ(source.points().data(), point_buffer);
        EXPECT_EQ(source.tetrahedra().data(), tetra_buffer);
        for (std::size_t i = 0; i < source.points().size(); ++i)
            EXPECT_EQ(source.points()[i].value, original.points[i].value);
        for (std::size_t i = 0; i < source.tetrahedra().size(); ++i)
            EXPECT_EQ(source.tetrahedra()[i].vertices, original.tetrahedra[i].vertices);
        for (double a : {.125, .5, .875})
        {
            const std::array donors{cut_donor(geometry, {{1, 0, 0}, {}, a}, 1, a)};
            const auto result = source.measure(donors);
            expect_split(result, 1, a);
            EXPECT_EQ(result.statistics.cancelled_boundaries, 6u);
        }
        std::reverse(recipe.points.begin(), recipe.points.end());
        for (auto& tetra : recipe.tetrahedra)
        {
            for (auto& index : tetra.vertices)
                index.value = recipe.points.size() - 1 - index.value;
            std::rotate(tetra.vertices.begin(), tetra.vertices.begin() + 1, tetra.vertices.end());
        }
        std::reverse(recipe.tetrahedra.begin(), recipe.tetrahedra.end());
    }
    expect_no_exact_work(before);
}

TEST(OrdinaryAggregateTest, IndexedCornerHalfCutAndOwnedLifetimeRetainIndependentVolumes)
{
    using namespace SimpleFluid::geometry::swept_volume;
    const auto source = []
    {
        auto local = indexed_corner();
        return OrdinaryAggregateSource(std::move(local), {}, ordinary_limits());
    }();
    ASSERT_EQ(source.points().size(), 4u);
    const auto geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    for (const int direction : {1, -1})
    {
        const auto gas = direction > 0 ? 7.L / 48 : 1.L / 48;
        const std::array donors{
            cut_donor(geometry, {{static_cast<double>(direction), 0, 0}, {}, .5 * direction}, 1.L / 6, gas)};
        expect_split(source.measure(donors), 1.L / 6, gas);
    }
    const auto points = source.points();
    const auto tetrahedra = source.tetrahedra();
    OrdinaryAggregateDonor invalid;
    expect_error([&] { (void) source.measure(std::span(&invalid, 1)); }, GeometryErrorCode::InvalidInput);
    EXPECT_EQ(source.points().data(), points.data());
    EXPECT_EQ(source.tetrahedra().data(), tetrahedra.data());
    EXPECT_EQ(points[1].value, (std::array<long double, 3>{1, 0, 0}));
    EXPECT_EQ(tetrahedra[0].vertices[3], Index{3});
    const std::array donors{cut_donor(geometry, {{1, 0, 0}, {}, .5}, 1.L / 6, 7.L / 48)};
    expect_split(source.measure(donors), 1.L / 6, 7.L / 48);
}

TEST(OrdinaryAggregateTest, IndexedEqualCoordinatesRetainDistinctSlotIdentityAndSignChannels)
{
    using namespace SimpleFluid::geometry::swept_volume;
    auto distinct = indexed_corner();
    const auto extra_points = distinct.points;
    distinct.points.insert(distinct.points.end(), extra_points.begin(), extra_points.end());
    auto duplicate = distinct.tetrahedra.front();
    for (auto& index : duplicate.vertices)
        index.value += 4;
    distinct.tetrahedra.push_back(duplicate);
    const OrdinaryAggregateSource source(std::move(distinct), {}, ordinary_limits());
    const auto geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    const std::array donors{cut_donor(geometry, {{1, 0, 0}, {}, .5}, 1.L / 3, 7.L / 24)};
    const auto result = source.measure(donors);
    expect_split(result, 1.L / 3, 7.L / 24);
    EXPECT_EQ(result.statistics.cancelled_boundaries, 0u);
    ASSERT_EQ(source.points().size(), 8u);
    EXPECT_EQ(source.points()[0].value, source.points()[4].value);
    EXPECT_NE(source.tetrahedra()[0].vertices[0], source.tetrahedra()[1].vertices[0]);

    auto signed_batch = indexed_corner();
    auto negative = signed_batch.tetrahedra.front();
    negative.sign = -1;
    signed_batch.tetrahedra.push_back(negative);
    const OrdinaryAggregateSource signed_source(std::move(signed_batch), {}, ordinary_limits());
    const auto signed_result = signed_source.measure(donors);
    ASSERT_EQ(signed_result.groups.size(), 2u);
    EXPECT_EQ(signed_result.signed_source_volume, 0);
    EXPECT_EQ(signed_result.statistics.cancelled_boundaries, 0u);
    for (const auto& group : signed_result.groups)
    {
        EXPECT_TRUE(group.sign == -1 || group.sign == 1);
        expect_amount(group.nominal.bulk, 1.L / 6);
        expect_amount(group.nominal.gas, 7.L / 48);
        expect_amount(group.nominal.carrier, 1.L / 48);
    }
    EXPECT_NE(signed_result.groups[0].sign, signed_result.groups[1].sign);
}

TEST(OrdinaryAggregateTest, IndexedConstructorRejectionsPreserveCallerStorageAndEveryElement)
{
    using namespace SimpleFluid::geometry::swept_volume;
    for (unsigned example = 0; example < 10; ++example)
    {
        SCOPED_TRACE(example);
        auto batch = indexed_corner();
        auto code = GeometryErrorCode::InvalidInput;
        switch (example)
        {
            case 0:
                batch.tetrahedra[0].vertices[3] = Index{batch.points.size()};
                break;
            case 1:
                batch.tetrahedra[0].vertices[3] = Index{std::numeric_limits<std::size_t>::max()};
                break;
            case 2:
                batch.tetrahedra[0].vertices[3] = batch.tetrahedra[0].vertices[0];
                code = GeometryErrorCode::InvalidTopology;
                break;
            case 3:
                batch.tetrahedra[0].sign = 0;
                break;
            case 4:
                batch.tetrahedra[0].sign = 2;
                break;
            case 5:
                batch.points[1].value[0] = std::numeric_limits<long double>::infinity();
                break;
            case 6:
                batch.points[1].value[0] = std::numeric_limits<long double>::quiet_NaN();
                break;
            case 7:
                batch.points.emplace_back(
                    std::array<long double, 3>{0, 0, -std::numeric_limits<long double>::infinity()});
                break;
            case 8:
                batch.tetrahedra.push_back(batch.tetrahedra.front());
                code = GeometryErrorCode::InvalidTopology;
                break;
            case 9:
                batch.points[3] = batch.points[0];
                code = GeometryErrorCode::UnresolvedArithmetic;
                break;
        }
        const BatchSnapshot original(batch);
        OrdinaryAggregateContext context;
        expect_error([&] { (void) OrdinaryAggregateSource(std::move(batch), {}, ordinary_limits(), &context); }, code);
        EXPECT_EQ(context.stage, OrdinaryAggregateStage::Source);
        EXPECT_EQ(context.source, example == 7 ? OrdinaryAggregateContext::absent : (example == 8 ? 1u : 0u));
        original.expect_unchanged(batch);
    }
}

TEST(OrdinaryAggregateTest, IndexedIncidenceRetainsOriginalSemanticRejectionOrder)
{
    using namespace SimpleFluid::geometry::swept_volume;
    for (unsigned example = 0; example < 3; ++example)
    {
        SCOPED_TRACE(example);
        auto batch = indexed_bipyramid();
        std::size_t expected_source = 2, expected_pairs = 12, expected_cancelled = 1;
        if (example == 0)
        {
            // A valid opposite pair followed by a third shared base face.
            batch.points.emplace_back(std::array<long double, 3>{0, 0, 2});
            auto third = batch.tetrahedra.front();
            third.vertices[3] = Index{5};
            batch.tetrahedra.push_back(third);
        }
        else if (example == 1)
        {
            // The same-oriented second occurrence rejects before a later bad
            // sign or out-of-range index is allowed to change the diagnosis.
            batch.tetrahedra[1] = batch.tetrahedra.front();
            auto later = batch.tetrahedra.front();
            later.sign = 0;
            later.vertices[0] = Index{std::numeric_limits<std::size_t>::max()};
            batch.tetrahedra.push_back(later);
            expected_source = 1;
            expected_pairs = 5;
            expected_cancelled = 0;
        }
        else
        {
            // Two valid pairs acquire third occurrences. The smaller-key group
            // appears first when sorted but its third occurrence is later.
            const auto low = batch.tetrahedra;
            const auto points = batch.points;
            for (auto point : points)
            {
                point.value[0] += 2;
                batch.points.push_back(std::move(point));
            }
            auto high = low;
            for (auto& tetra : high)
                for (auto& index : tetra.vertices)
                    index.value += points.size();
            batch.tetrahedra = {high[0], high[1], low[0], low[1]};
            batch.points.emplace_back(std::array<long double, 3>{2, 0, 2});
            batch.points.emplace_back(std::array<long double, 3>{0, 0, 2});
            auto third_high = high[0], third_low = low[0];
            third_high.vertices[3] = Index{10};
            third_low.vertices[3] = Index{11};
            batch.tetrahedra.push_back(third_high);
            batch.tetrahedra.push_back(third_low);
            expected_source = 4;
            expected_pairs = 20;
            expected_cancelled = 2;
        }
        const BatchSnapshot original(batch);
        OrdinaryAggregateContext context;
        expect_error([&] { (void) OrdinaryAggregateSource(std::move(batch), {}, ordinary_limits(), &context); },
            GeometryErrorCode::InvalidTopology);
        original.expect_unchanged(batch);
        EXPECT_EQ(context.stage, OrdinaryAggregateStage::Source);
        EXPECT_EQ(context.source, expected_source);
        EXPECT_EQ(context.cut, OrdinaryAggregateContext::absent);
        EXPECT_EQ(context.pairs, expected_pairs);
        EXPECT_EQ(context.statistics.cancelled_boundaries, expected_cancelled);
    }
}

TEST(OrdinaryAggregateTest, IndexedSourcePairAllowancesPreservePrefixCountsAndCallerBatch)
{
    using namespace SimpleFluid::geometry::swept_volume;
    constexpr std::size_t facets = 24;
    for (bool geometry_controls : {false, true})
        for (const auto allowance : {std::size_t{0}, facets - 1, facets})
        {
            SCOPED_TRACE(::testing::Message() << "geometry=" << geometry_controls << " allowance=" << allowance);
            auto batch = indexed_cube();
            const BatchSnapshot original(batch);
            SweepLimits sweep;
            auto geometry = ordinary_limits();
            if (geometry_controls)
                geometry.maximum_pair_tests = allowance;
            else
                sweep.maximum_pair_tests = allowance;
            OrdinaryAggregateContext context;
            if (allowance == facets)
            {
                const OrdinaryAggregateSource source(std::move(batch), sweep, geometry, &context);
                EXPECT_EQ(source.points().data(), original.points);
                EXPECT_EQ(source.tetrahedra().data(), original.tetrahedra);
                EXPECT_EQ(context.pairs, facets);
                EXPECT_EQ(context.statistics.cancelled_boundaries, 6u);
                EXPECT_EQ(context.statistics.source_pieces, 6u);
            }
            else
            {
                expect_error([&] { (void) OrdinaryAggregateSource(std::move(batch), sweep, geometry, &context); },
                    GeometryErrorCode::ResourceLimit);
                original.expect_unchanged(batch);
                EXPECT_EQ(context.pairs, allowance);
                EXPECT_EQ(context.source, allowance / 4);
                EXPECT_EQ(context.statistics.source_pieces, 0u);
            }
            EXPECT_EQ(context.cut, OrdinaryAggregateContext::absent);
            EXPECT_EQ(context.statistics.pair_tests, context.pairs);
            EXPECT_EQ(context.statistics.storage_bytes, context.bytes);
        }
}

TEST(OrdinaryAggregateTest, IndexedPairAdmissionPrecedesSecondAndThirdIncidenceRejections)
{
    using namespace SimpleFluid::geometry::swept_volume;
    for (const bool third_occurrence : {false, true})
        for (const bool admit_rejecting_face : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "third=" << third_occurrence << " admit=" << admit_rejecting_face);
            auto batch = third_occurrence ? indexed_bipyramid() : indexed_corner();
            auto repeated = batch.tetrahedra.front();
            if (third_occurrence)
            {
                batch.points.emplace_back(std::array<long double, 3>{0, 0, 2});
                repeated.vertices[3] = Index{5};
            }
            batch.tetrahedra.push_back(repeated);
            const BatchSnapshot original(batch);
            const auto rejecting_pair = third_occurrence ? std::size_t{12} : std::size_t{5};
            SweepLimits sweep;
            sweep.maximum_pair_tests = rejecting_pair - (admit_rejecting_face ? 0 : 1);
            OrdinaryAggregateContext context;
            expect_error([&] { (void) OrdinaryAggregateSource(std::move(batch), sweep, ordinary_limits(), &context); },
                admit_rejecting_face ? GeometryErrorCode::InvalidTopology : GeometryErrorCode::ResourceLimit);
            original.expect_unchanged(batch);
            EXPECT_EQ(context.stage, OrdinaryAggregateStage::Source);
            EXPECT_EQ(context.source, third_occurrence ? 2u : 1u);
            EXPECT_EQ(context.cut, OrdinaryAggregateContext::absent);
            EXPECT_EQ(context.pairs, sweep.maximum_pair_tests);
            EXPECT_EQ(context.statistics.pair_tests, context.pairs);
            EXPECT_EQ(context.statistics.cancelled_boundaries, third_occurrence ? 1u : 0u);
        }
}

TEST(OrdinaryAggregateTest, IndexedIdenticalRecipesRepeatGroupOrderAmountsErrorsAndFlagsExactly)
{
    using namespace SimpleFluid::geometry::swept_volume;
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {.5, 1, 1}), {{0, 1, 0}, {}, .5}, .5, .25),
        cut_donor(Geometry::box({.5, 0, 0}, {1, 1, 1}), {{0, 1, 0}, {}, .5}, .5, .25)};
    auto labelled = donors;
    labelled[1].id = 11;
    const OrdinaryAggregateSource source(indexed_cube(), {}, ordinary_limits());
    const auto first = source.measure(labelled);
    ASSERT_EQ(first.groups.size(), 2u);
    expect_amount(first.signed_source_volume, 1);
    for (const auto& group : first.groups)
    {
        expect_amount(group.nominal.bulk, .5);
        expect_amount(group.nominal.gas, .25);
        expect_amount(group.nominal.carrier, .25);
        EXPECT_TRUE(group.bulk_positive && group.gas_positive && group.carrier_positive);
    }
    expect_identical_results(first, source.measure(labelled));
    const OrdinaryAggregateSource rebuilt(indexed_cube(), {}, ordinary_limits());
    expect_identical_results(first, rebuilt.measure(labelled));
}

TEST(OrdinaryAggregateTest, IndexedPartnersCancelOnlyWhenBothFacesSurviveTheSameChannel)
{
    using namespace SimpleFluid::geometry::swept_volume;
    const OrdinaryAggregateSource source(indexed_bipyramid(), {}, ordinary_limits());
    const auto box = Geometry::box({0, 0, -1}, {1, 1, 1});
    const std::array phases{cut_donor(box, {{0, 0, 1}, {}, 0}, 1.L / 3, 1.L / 6)};
    const auto full = source.measure(phases);
    expect_split(full, 1.L / 3, 1.L / 6);
    EXPECT_EQ(full.statistics.cancelled_boundaries, 1u);
    std::array sides{cut_donor(Geometry::box({0, 0, -1}, {1, 1, 0}), {{1, 0, 0}, {}, .5}, 1.L / 6, 7.L / 48),
        cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .5}, 1.L / 6, 7.L / 48)};
    sides[1].id = 38;
    for (unsigned order = 0; order < 2; ++order)
    {
        const auto result = source.measure(sides);
        ASSERT_EQ(result.groups.size(), 2u);
        EXPECT_EQ(result.statistics.cancelled_boundaries, 1u);
        expect_amount(result.signed_source_volume, 1.L / 3);
        for (const auto& group : result.groups)
        {
            expect_amount(group.nominal.bulk, 1.L / 6);
            expect_amount(group.nominal.gas, 7.L / 48);
            expect_amount(group.nominal.carrier, 1.L / 48);
            EXPECT_TRUE(group.bulk_positive && group.gas_positive && group.carrier_positive);
        }
        std::reverse(sides.begin(), sides.end());
    }
}

TEST(OrdinaryAggregateTest, IndexedCapacityAdmissionAcceptsExactAllowanceAndRejectsOneByteShort)
{
    using namespace SimpleFluid::geometry::swept_volume;
    auto basic = indexed_corner();
    const auto basic_points = basic.points.capacity();
    const auto basic_tetrahedra = basic.tetrahedra.capacity();
    const OrdinaryAggregateSource small(std::move(basic), {}, ordinary_limits());
    auto roomy = indexed_corner();
    roomy.points.reserve(17);
    roomy.tetrahedra.reserve(9);
    const auto extra = (roomy.points.capacity() - basic_points) * sizeof(P) +
                       (roomy.tetrahedra.capacity() - basic_tetrahedra) * sizeof(OrdinaryAggregateTetra);
    const OrdinaryAggregateSource reference(std::move(roomy), {}, ordinary_limits());
    EXPECT_EQ(reference.storage_bytes(), small.storage_bytes() + extra);
    for (const bool enough : {true, false})
    {
        auto batch = indexed_corner();
        batch.points.reserve(17);
        batch.tetrahedra.reserve(9);
        const BatchSnapshot original(batch);
        auto limits = ordinary_limits();
        limits.byte_budget = reference.storage_bytes() - (enough ? 0 : 1);
        OrdinaryAggregateContext context;
        if (enough)
        {
            const OrdinaryAggregateSource source(std::move(batch), {}, limits, &context);
            EXPECT_EQ(source.storage_bytes(), reference.storage_bytes());
            EXPECT_EQ(source.points().data(), original.points);
            EXPECT_EQ(source.tetrahedra().data(), original.tetrahedra);
            const auto points = source.points();
            expect_error([&] { (void) source.measure({}); }, GeometryErrorCode::ResourceLimit);
            EXPECT_EQ(source.points().data(), points.data());
        }
        else
        {
            expect_error([&] { (void) OrdinaryAggregateSource(std::move(batch), {}, limits, &context); },
                GeometryErrorCode::ResourceLimit);
            original.expect_unchanged(batch);
            EXPECT_GT(context.statistics.failed_charge_bytes, limits.byte_budget - context.bytes);
        }
    }
}

TEST(OrdinaryAggregateTest, IndexedEmptySourceRetainsUnusedFiniteTableEntries)
{
    using namespace SimpleFluid::geometry::swept_volume;
    Batch empty;
    const OrdinaryAggregateSource source(std::move(empty), {}, ordinary_limits());
    EXPECT_TRUE(source.points().empty());
    EXPECT_TRUE(source.tetrahedra().empty());
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .5}, 1, .5)};
    const auto result = source.measure(donors);
    EXPECT_TRUE(result.groups.empty());
    EXPECT_EQ(result.signed_source_volume, 0);
    EXPECT_EQ(result.statistics.source_pieces, 0u);
    Batch unused;
    unused.points.emplace_back(std::array<long double, 3>{2, 3, 4});
    const OrdinaryAggregateSource unused_source(std::move(unused), {}, ordinary_limits());
    ASSERT_EQ(unused_source.points().size(), 1u);
    EXPECT_TRUE(unused_source.tetrahedra().empty());
    EXPECT_EQ(unused_source.points()[0].value, (std::array<long double, 3>{2, 3, 4}));
}

TEST(OrdinaryAggregateTest, SharedSplitRetainsAnalyticEdgeVertexAndTangentContacts)
{
    using namespace SimpleFluid::geometry::swept_volume;
    FixtureTetra tetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {1, 2, 3, 4}, 1};
    const auto donor_geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    constexpr long double bulk = 1.L / 6;
    struct Case
    {
        GeometryPoint normal;
        long double gas;
    };
    // Symmetry across x=y halves the tetrahedron. The cone z>=x+y has
    // intercepts (1/2,0,1/2),(0,1/2,1/2),(0,0,1), so volume=1/24.
    const std::array<Case, 3> cases{{{{1, -1, 0}, 1.L / 12}, {{1, 1, -1}, 1.L / 24}, {{1, 0, 0}, 0}}};
    for (unsigned rotation = 0; rotation < 4; ++rotation)
    {
        const std::array recipes{tetra};
        const OrdinaryAggregateSource source(make_batch(recipes), {}, ordinary_limits());
        for (const auto& example : cases)
            for (const int direction : {1, -1})
            {
                auto normal = example.normal;
                for (auto& value : normal)
                    value *= direction;
                const auto gas = direction > 0 ? example.gas : bulk - example.gas;
                const std::array donors{cut_donor(donor_geometry, {normal, {}, 0}, bulk, gas)};
                const auto result = source.measure(donors);
                ASSERT_EQ(result.groups.size(), 1u);
                const auto& group = result.groups.front();
                expect_amount(group.nominal.bulk, bulk);
                expect_amount(group.nominal.gas, gas);
                expect_amount(group.nominal.carrier, bulk - gas);
                EXPECT_EQ(group.gas_positive, gas > 0);
                EXPECT_EQ(group.carrier_positive, gas < bulk);
                EXPECT_EQ(group.approximation_bound.bulk, 0);
                EXPECT_EQ(group.approximation_bound.gas, 0);
                EXPECT_EQ(group.approximation_bound.carrier, 0);
                EXPECT_LE(std::abs(group.nominal.gas + group.nominal.carrier - group.nominal.bulk),
                    bulk * relative_allowance);
                expect_compact_accounting(result.statistics);
                // Six box planes for coverage, six for direct metrics, and
                // one phase split: at most 13 classifications of four vertices.
                // Complementary children must not classify that input twice.
                EXPECT_LE(result.statistics.vertices_classified, 52u);
            }
        std::rotate(tetra.vertices.begin(), tetra.vertices.begin() + 1, tetra.vertices.end());
        std::rotate(tetra.vertex_ids.begin(), tetra.vertex_ids.begin() + 1, tetra.vertex_ids.end());
    }
}

TEST(OrdinaryAggregateTest, SharedCapIdDomainCanExceedBothCompactChildVertexCaps)
{
    using namespace SimpleFluid::geometry::swept_volume;
    auto batch = indexed_corner();
    const std::array<P, 4> points{batch.points[0], batch.points[1], batch.points[2], batch.points[3]};
    auto limits = ordinary_limits();
    limits.maximum_vertices = 6;
    // The x=1/2 cut has three fresh crossings: its shared ID domain has
    // seven slots, while the retained children have six and four vertices.
    const OrdinaryAggregateSource source(std::move(batch), {}, limits);
    const std::array donors{cut_donor(Geometry::tetrahedron(points, limits), {{1, 0, 0}, {}, .5}, 1.L / 6, 7.L / 48)};
    const auto first = source.measure(donors);
    expect_split(first, 1.L / 6, 7.L / 48);
    expect_compact_accounting(first.statistics);
    const auto again = source.measure(donors);
    expect_identical_results(first, again);
}

TEST(OrdinaryAggregateTest, CompactRemaindersPreserveFourSlabInventoriesAndSourceFaces)
{
    using namespace SimpleFluid::geometry::swept_volume;
    FixtureTetra tetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {1, 2, 3, 4}, 1};
    std::array<OrdinaryAggregateDonor, 4> donors;
    std::array<long double, 4> expected;
    for (unsigned i = 0; i < donors.size(); ++i)
    {
        const double a = i / 4., b = (i + 1) / 4.;
        expected[i] = ((1.L - a) * (1.L - a) * (1.L - a) - (1.L - b) * (1.L - b) * (1.L - b)) / 6;
        donors[i] = cut_donor(Geometry::box({a, 0, 0}, {b, 1, 1}), {{0, 1, -1}, {}, 0}, expected[i], expected[i] / 2);
        donors[i].id = i;
    }
    for (unsigned rotation = 0; rotation < 4; ++rotation)
    {
        const std::array recipes{tetra};
        const OrdinaryAggregateSource source(make_batch(recipes), {}, ordinary_limits());
        for (unsigned order = 0; order < 2; ++order)
        {
            const auto result = source.measure(donors);
            ASSERT_EQ(result.groups.size(), 4u);
            long double bulk = 0, gas = 0, carrier = 0;
            for (const auto& group : result.groups)
            {
                ASSERT_LT(group.donor, expected.size());
                expect_amount(group.nominal.bulk, expected[group.donor]);
                expect_amount(group.nominal.gas, expected[group.donor] / 2);
                expect_amount(group.nominal.carrier, expected[group.donor] / 2);
                EXPECT_TRUE(group.bulk_positive && group.gas_positive && group.carrier_positive);
                bulk += group.nominal.bulk;
                gas += group.nominal.gas;
                carrier += group.nominal.carrier;
            }
            expect_amount(bulk, 1.L / 6);
            expect_amount(gas, 1.L / 12);
            expect_amount(carrier, 1.L / 12);
            expect_compact_accounting(result.statistics);
            std::reverse(donors.begin(), donors.end());
        }
        std::rotate(tetra.vertices.begin(), tetra.vertices.begin() + 1, tetra.vertices.end());
        std::rotate(tetra.vertex_ids.begin(), tetra.vertex_ids.begin() + 1, tetra.vertex_ids.end());
    }
}

TEST(OrdinaryAggregateTest, FailedChargeReportsUnspentRequestAndStageAttribution)
{
    using namespace SimpleFluid::geometry::swept_volume;
    const auto recipes = cube_tetrahedra();
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .5}, 1, .5)};
    const OrdinaryAggregateSource reference(make_batch(recipes), {}, ordinary_limits());
    const auto required = reference.measure(donors).statistics.storage_bytes;
    auto limits = ordinary_limits();
    limits.byte_budget = required;
    const OrdinaryAggregateSource exact(make_batch(recipes), {}, limits);
    const auto admitted = exact.measure(donors);
    expect_split(admitted, 1, .5);
    EXPECT_EQ(admitted.statistics.storage_bytes, required);
    limits.byte_budget = required - 1;
    OrdinaryAggregateContext context;
    const OrdinaryAggregateSource limited(make_batch(recipes), {}, limits, &context);
    expect_error([&] { (void) limited.measure(donors, &context); }, GeometryErrorCode::ResourceLimit);
    EXPECT_EQ(context.bytes, context.statistics.storage_bytes);
    EXPECT_EQ(std::accumulate(context.statistics.stage_charged_bytes.begin(),
                  context.statistics.stage_charged_bytes.end(), std::size_t{}),
        context.bytes);
    ASSERT_LE(context.bytes, limits.byte_budget);
    EXPECT_GT(context.statistics.failed_charge_bytes, limits.byte_budget - context.bytes);
}

TEST(OrdinaryAggregateTest, DonorValidationFailureRetainsOneConsistentAccountingSnapshot)
{
    using namespace SimpleFluid::geometry::swept_volume;
    const auto recipes = cube_tetrahedra();
    OrdinaryAggregateContext initial;
    const OrdinaryAggregateSource reference(make_batch(recipes), {}, ordinary_limits(), &initial);
    auto limits = ordinary_limits();
    limits.maximum_operations = initial.operations + 1;
    const OrdinaryAggregateSource limited(make_batch(recipes), {}, limits);
    std::array donors{cut_donor(Geometry::box({0, 0, 0}, {.5, 1, 1}), {{0, 1, 0}, {}, .5}, .5, .25),
        cut_donor(Geometry::box({.5, 0, 0}, {1, 1, 1}), {{0, 1, 0}, {}, .5}, .5, .25)};
    donors[1].id = 38;
    OrdinaryAggregateContext context;
    // The provider must obey the aggregate's tighter remaining allowance even
    // while nested under a caller with ample operations and storage.
    SimpleFluid::PLIC::details::Operation caller(ordinary_limits());
    expect_error([&] { (void) limited.measure(donors, &context); }, GeometryErrorCode::ResourceLimit);
    EXPECT_EQ(context.stage, OrdinaryAggregateStage::DonorValidation);
    EXPECT_EQ(context.statistics.storage_bytes, context.bytes);
    EXPECT_EQ(context.statistics.operations, context.operations);
    EXPECT_EQ(context.statistics.pair_tests, context.pairs);
    EXPECT_LE(context.operations, limits.maximum_operations);
    // The ambient caller also counts the fixed FP-environment probe outside
    // the aggregate ledger. Its total is not the provider's remaining quota.
    EXPECT_GE(context.operations, initial.operations);
    EXPECT_GE(caller.work.operations, context.operations - initial.operations);
    EXPECT_EQ(std::accumulate(context.statistics.stage_charged_bytes.begin(),
                  context.statistics.stage_charged_bytes.end(), std::size_t{}),
        context.bytes);
    EXPECT_EQ(SimpleFluid::PLIC::details::active_work, &caller.work);
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
        const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
            make_batch(tetrahedra), {}, ordinary_limits());
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
    const std::array<FixtureTetra, 1> tetrahedra{
        FixtureTetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {1, 2, 3, 4}, 1}};
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(tetrahedra), {}, ordinary_limits());
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
    const std::array<FixtureTetra, 1> recipes{
        FixtureTetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {1, 2, 3, 4}, 1}};
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(recipes), {}, ordinary_limits());
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
            std::array donors{cut_donor(split.inside, {phase_normal, {}, h}, bulk, gas),
                SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor{}};
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
    const std::array<FixtureTetra, 1> recipes{
        FixtureTetra{{P({0, 0, 0}), P({3, 0, 0}), P({0, 3, 0}), P({0, 0, 3})}, {1, 2, 3, 4}, 1}};
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(recipes), {}, ordinary_limits());
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
            std::array donors{cut_donor(split.inside, {phase_normal, {}, h + delta}, bulk, gas),
                SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor{}};
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
        const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
            make_batch(tetrahedra), {}, ordinary_limits());
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
    std::array<FixtureTetra, 2> tetrahedra{
        FixtureTetra{{P({0, 0, 1}), P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0})}, {2, 1, 3, 4}, 1},
        FixtureTetra{{P({0, 0, -1}), P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0})}, {5, 1, 3, 4}, 1}};
    const std::array phase_donors{
        cut_donor(Geometry::box({0, 0, -1}, {1, 1, 1}), {{0, 0, -1}, {}, 0}, 1.L / 3, 1.L / 6)};
    SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor upper;
    upper.id = 37;
    upper.geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    upper.pure_gas = true;
    upper.bulk_budget = upper.gas_budget = relative_allowance / 6;
    SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor lower;
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
        const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
            make_batch(tetrahedra), {}, ordinary_limits());
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

TEST(OrdinaryAggregateTest, IndexedTopologyRejectsRepeatedVerticesAndMalformedFaceIncidence)
{
    using namespace SimpleFluid::geometry::swept_volume;
    const auto good = make_batch(cube_tetrahedra());
    auto repeated = good;
    repeated.tetrahedra.push_back(good.tetrahedra.front());
    expect_error([&] { (void) OrdinaryAggregateSource(std::move(repeated), {}, ordinary_limits()); },
        GeometryErrorCode::InvalidTopology);
    EXPECT_EQ(repeated.tetrahedra.size(), good.tetrahedra.size() + 1);
    auto repeated_vertex = good;
    repeated_vertex.tetrahedra[0].vertices[1] = repeated_vertex.tetrahedra[0].vertices[0];
    expect_error([&] { (void) OrdinaryAggregateSource(std::move(repeated_vertex), {}, ordinary_limits()); },
        GeometryErrorCode::InvalidTopology);
    EXPECT_EQ(repeated_vertex.tetrahedra[0].vertices[1], repeated_vertex.tetrahedra[0].vertices[0]);
    auto valid_batch = good;
    const OrdinaryAggregateSource valid(std::move(valid_batch), {}, ordinary_limits());
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .5}, 1, .5)};
    expect_split(valid.measure(donors), 1, .5);
}

TEST(OrdinaryAggregateTest, ExhaustedSourceAndMeasurementBudgetsRejectExplicitly)
{
    const auto tetrahedra = cube_tetrahedra();
    auto sweep = SimpleFluid::geometry::swept_volume::SweepLimits{};
    sweep.byte_budget = 1;
    expect_error(
        [&]
        {
            (void) SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource(
                make_batch(tetrahedra), sweep, ordinary_limits());
        },
        GeometryErrorCode::ResourceLimit);
    sweep = {};
    sweep.maximum_pair_tests = 0;
    expect_error(
        [&]
        {
            (void) SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource(
                make_batch(tetrahedra), sweep, ordinary_limits());
        },
        GeometryErrorCode::ResourceLimit);
    sweep.maximum_pair_tests = 24; // Six tetrahedra contribute four authenticated source faces each.
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(tetrahedra), sweep, ordinary_limits());
    const std::array donors{cut_donor(Geometry::box({0, 0, 0}, {1, 1, 1}), {{1, 0, 0}, {}, .5}, 1, .5)};
    SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext context;
    expect_error([&] { (void) source.measure(donors, &context); }, GeometryErrorCode::ResourceLimit);
    EXPECT_GE(context.bytes, source.storage_bytes());
    EXPECT_EQ(context.pairs, 24u);
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource baseline(
        make_batch(tetrahedra), {}, ordinary_limits());
    sweep = {};
    sweep.byte_budget = baseline.storage_bytes();
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource retained(
        make_batch(tetrahedra), sweep, ordinary_limits());
    expect_error([&] { (void) retained.measure(donors); }, GeometryErrorCode::ResourceLimit);
}

TEST(OrdinaryAggregateTest, UnresolvedMinorityBudgetRejectsInsteadOfBecomingAbsent)
{
    const FixtureTetra tetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({0, 0, 1})}, {0, 1, 2, 3}, 1};
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(std::span(&tetra, 1)), {}, ordinary_limits());
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
    const std::array<FixtureTetra, 1> recipes{
        FixtureTetra{{P({0, 0, 0}), P({1, 0, delta}), P({0, 1, -1}), P({0, 0, 1})}, {1, 2, 3, 4}, 1}};
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(recipes), {}, ordinary_limits());
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
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(recipes), {}, ordinary_limits());
    const auto result = source.measure(donors);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    ASSERT_EQ(result.groups.size(), 1u);
    EXPECT_EQ(result.groups[0].nominal.gas, .25L);
    auto limits = ordinary_limits();
    limits.byte_budget = 1;
    expect_error([&]
        { (void) SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource(make_batch(recipes), {}, limits); },
        GeometryErrorCode::ResourceLimit);
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
    const std::array<FixtureTetra, 1> recipes{
        FixtureTetra{{P({0, 0, 0}), P({edge, 0, 0}), P({0, edge, 0}), P({0, 0, edge})}, {1, 2, 3, 4}, 1}};
    const auto before = geometry_statistics();
    expect_error(
        [&]
        {
            const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
                make_batch(recipes), {}, ordinary_limits());
            SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor donor;
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
    const std::array<FixtureTetra, 2> recipes{
        FixtureTetra{{P({0, 0, 0}), P({1, 0, 0}), P({0, 1, 0}), P({1, 1, height})}, {1, 2, 3, 4}, 1},
        FixtureTetra{{P({-2, -2, 1}), P({-1.5, -2, 1}), P({-2, -1.5, 1}), P({-2, -2, 1.5})}, {5, 6, 7, 8}, 1}};
    const auto original = recipes;
    const long double bulk = height / 6, gas = height * width * width * width / 12;
    // Under the source affine map x+y=lambda1+lambda2+2lambda3.
    // For width<=1 the reference cut is a tetra with fraction width^3/2.
    std::array<SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor, 2> donors;
    donors[0] = cut_donor(
        Geometry::box({0, 0, -double(height)}, {1, 1, 2 * double(height)}), {{1, 1, 0}, {}, width}, bulk, gas);
    donors[1].id = 91;
    donors[1].geometry = Geometry::box({-3, -3, .5}, {-1, -1, 2});
    donors[1].pure_carrier = true;
    donors[1].bulk_budget = donors[1].carrier_budget = 1e-14L;
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(recipes), {}, ordinary_limits());
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
    FixtureTetra tetra;
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
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
        make_batch(tetrahedra), {}, ordinary_limits());
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
