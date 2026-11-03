#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include "geometry/swept_volume/details/OrdinaryAggregateRetention.hh"
#include "geometry/swept_volume/details/OrdinaryMetricRecords.hh"
#include "geometry/swept_volume/details/OrdinaryVertexArena.hh"
#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
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
    // One coverage tetra and one metric tetra each introduce four original
    // records. The x=1/2 phase plane crosses exactly three edges; both phase
    // children retain those same three constructions. Contact evidence may
    // add immutable versions, but must never duplicate a numerical crossing.
    EXPECT_EQ(first.statistics.retries, 0u);
    EXPECT_EQ(first.statistics.vertex_record_constructions - first.statistics.vertex_evidence_extensions, 11u);
    EXPECT_EQ(first.statistics.peak_live_vertex_records, 7u);
    EXPECT_EQ(first.statistics.peak_vertex_record_capacity, 8u);
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

namespace
{
struct ArenaTestWork
{
    SimpleFluid::geometry::swept_volume::OrdinaryAggregateStatistics stats;
    std::size_t live_arena_bytes = 0, live_slot_bytes = 0, live_incidence_bytes = 0;
    std::size_t live_compact_metric_bytes = 0, live_freeze_scratch_bytes = 0, live_vertex_index_bytes = 0;
    std::size_t external_connectivity_bytes = 0;
    std::size_t charges = 0, charged_bytes = 0, fail_at = std::numeric_limits<std::size_t>::max();
    std::size_t byte_budget = std::numeric_limits<std::size_t>::max();
    void charge(std::size_t bytes)
    {
        if (charges++ == fail_at || bytes > byte_budget || charged_bytes > byte_budget - bytes)
            throw std::bad_alloc();
        charged_bytes += bytes;
    }
    void observe_geometry(std::size_t transient_bytes = 0) noexcept
    {
        stats.live_incidence_bytes = live_incidence_bytes;
        stats.peak_incidence_bytes = std::max(stats.peak_incidence_bytes, live_incidence_bytes);
        stats.peak_combined_geometry_bytes = std::max(stats.peak_combined_geometry_bytes,
            live_arena_bytes + live_slot_bytes + live_incidence_bytes + live_compact_metric_bytes +
                live_freeze_scratch_bytes + live_vertex_index_bytes + external_connectivity_bytes + transient_bytes);
        stats.live_compact_metric_bytes = live_compact_metric_bytes;
        stats.live_freeze_scratch_bytes = live_freeze_scratch_bytes;
        stats.peak_compact_metric_bytes = std::max(stats.peak_compact_metric_bytes, live_compact_metric_bytes);
        stats.peak_freeze_scratch_bytes = std::max(stats.peak_freeze_scratch_bytes, live_freeze_scratch_bytes);
    }
};
using TestArena = SimpleFluid::geometry::swept_volume::details::OrdinaryVertexArena<int, ArenaTestWork>;
}

TEST(OrdinaryAggregateArena, ContactEvidenceUsesFiveTransientRecordsAndReleasesTheEmptyPage)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, cut = 0;
    std::array<TestArena::Handle, 4> points;
    for (unsigned i = 0; i < points.size(); ++i)
    {
        TestArena::Weights weights{}, errors{};
        weights[i] = 1;
        errors[i] = std::ldexp(1.L, -60 - static_cast<int>(i));
        points[i] = arena.create(weights, errors, {}, &source, &chart);
    }
    // Four retained vertices plus one new contact-evidence version require
    // five simultaneous records. At most one unused physical position remains.
    const auto construction = points[0].get().construction;
    points[0] = arena.extend(points[0], &cut);
    EXPECT_EQ(arena.live_records(), 4u);
    EXPECT_EQ(work.stats.peak_live_vertex_records, 5u);
    EXPECT_EQ(work.stats.peak_vertex_record_capacity, 6u);
    EXPECT_EQ(arena.record_capacity(), 4u);
    EXPECT_EQ(arena.slot_capacity(), 8u);
    EXPECT_EQ(points[0].get().construction, construction);
    EXPECT_EQ(points[0].get().supports, std::vector<const int*>{&cut});
    for (unsigned i = 1; i < points.size(); ++i)
    {
        EXPECT_EQ(points[i].get().weights[i], 1);
        EXPECT_EQ(points[i].get().errors[i], std::ldexp(1.L, -60 - static_cast<int>(i)));
    }
}

TEST(OrdinaryAggregateArena, SharedEvidenceIsImmutableAndSlotReuseKeepsSwapLastHandlesValid)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2;
    auto original = arena.create({1, 2, 3, 4}, {5, 6, 7, 8}, {&first}, &source, &chart);
    auto sibling = original;
    original = arena.extend(original, &second);
    EXPECT_EQ(sibling.get().supports, std::vector<const int*>{&first});
    EXPECT_EQ(original.get().supports, (std::vector<const int*>{&first, &second}));
    EXPECT_EQ(original.get().weights, sibling.get().weights);
    EXPECT_EQ(original.get().errors, sibling.get().errors);
    EXPECT_EQ(original.get().construction, sibling.get().construction);
    EXPECT_EQ(original.get().source, &source);
    EXPECT_EQ(original.get().chart, &chart);
    sibling = {};
    for (unsigned i = 0; i < 100; ++i)
    {
        auto temporary = arena.create({static_cast<long double>(i), 0, 0, 0}, {}, {}, &source, &chart);
        EXPECT_LT(temporary.slot_id().value, 4u);
        auto last = arena.create({0, static_cast<long double>(i), 0, 0}, {}, {}, &source, &chart);
        temporary = {}; // Moves last into the released physical position.
        EXPECT_EQ(last.get().weights[1], i);
        EXPECT_EQ(original.get().weights, (TestArena::Weights{1, 2, 3, 4}));
    }
    EXPECT_EQ(arena.slot_capacity(), 4u);
    EXPECT_GT(work.stats.vertex_relocation_bytes, 0u);
}

TEST(OrdinaryAggregateArena, EveryAdmissionExceptionRetainsExistingEvidenceAndAllowsReuse)
{
    for (std::size_t failure = 0; failure < 14; ++failure)
    {
        ArenaTestWork work;
        TestArena arena(work);
        int source = 0, chart = 0, first = 1, second = 2;
        auto original = arena.create({1, 2, 3, 4}, {4, 3, 2, 1}, {&first}, &source, &chart);
        std::array<TestArena::Handle, 3> other;
        for (auto& point : other)
            point = arena.create({}, {}, {}, &source, &chart);
        work.fail_at = work.charges + failure;
        try
        {
            auto extended = arena.extend(original, &second);
            EXPECT_EQ(extended.get().supports, (std::vector<const int*>{&first, &second}));
        }
        catch (const std::bad_alloc&)
        {
        }
        EXPECT_EQ(original.get().weights, (TestArena::Weights{1, 2, 3, 4}));
        EXPECT_EQ(original.get().supports, std::vector<const int*>{&first});
        work.fail_at = std::numeric_limits<std::size_t>::max();
        auto reused = arena.extend(original, &second);
        EXPECT_EQ(reused.get().errors, (TestArena::Weights{4, 3, 2, 1}));
        EXPECT_EQ(reused.get().supports, (std::vector<const int*>{&first, &second}));
    }
}


TEST(OrdinaryAggregateArena, PageFirstUseCreditsFollowPhysicalStorageRatherThanReusedMetadata)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0;
    auto first = arena.create({}, {}, {}, &source, &chart);
    const auto reused_slot = first.slot_id();
    first = {};
    ASSERT_EQ(arena.record_capacity(), 0u);
    const auto before_fresh_page = work.charged_bytes;
    first = arena.create({}, {}, {}, &source, &chart);
    EXPECT_EQ(first.slot_id(), reused_slot);
    // Reusing the identity slot does not add a construction charge to the new
    // two-record page reservation. The retained directories need no growth.
    EXPECT_EQ(work.charged_bytes - before_fresh_page, (sizeof(std::array<std::optional<TestArena::Record>, 2>)));
    const auto before_virgin_second = work.charged_bytes;
    auto second = arena.create({}, {}, {}, &source, &chart);
    EXPECT_EQ(work.charged_bytes, before_virgin_second);
    ASSERT_EQ(arena.record_capacity(), 2u);
    second = {};
    ASSERT_EQ(arena.record_capacity(), 2u);
    const auto before_recycled_second = work.charged_bytes;
    second = arena.create({}, {}, {}, &source, &chart);
    EXPECT_EQ(work.charged_bytes - before_recycled_second, sizeof(TestArena::Record));
}

TEST(OrdinaryAggregateArena, FailedAdmissionDoesNotConsumeTheVirginPhysicalPosition)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0;
    auto first = arena.create({}, {}, {}, &source, &chart);
    const auto admitted = work.charged_bytes;
    work.fail_at = work.charges;
    EXPECT_THROW((void) arena.create({}, {}, {}, &source, &chart), std::bad_alloc);
    EXPECT_EQ(work.charged_bytes, admitted);
    work.fail_at = std::numeric_limits<std::size_t>::max();
    auto second = arena.create({}, {}, {}, &source, &chart);
    EXPECT_EQ(work.charged_bytes, admitted);
    EXPECT_EQ(arena.live_records(), 2u);
}


TEST(OrdinaryAggregateArena, UniqueEvidenceReplacementRetainsFourRecordsAndObservesBothSupportBuffers)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2;
    std::array<TestArena::Handle, 4> points;
    points[0] = arena.create({1, 2, 3, 4}, {4, 3, 2, 1}, {&first}, &source, &chart);
    for (unsigned i = 1; i < points.size(); ++i)
        points[i] = arena.create({}, {}, {}, &source, &chart);
    const auto slot = points[0].slot_id();
    const auto construction = points[0].get().construction;
    const auto version = points[0].get().evidence_version;
    const auto copies = work.stats.vertex_handle_copies;
    arena.replace_with_extended_evidence(points[0], &second);
    EXPECT_EQ(points[0].slot_id(), slot);
    EXPECT_EQ(points[0].get().weights, (TestArena::Weights{1, 2, 3, 4}));
    EXPECT_EQ(points[0].get().errors, (TestArena::Weights{4, 3, 2, 1}));
    EXPECT_EQ(points[0].get().source, &source);
    EXPECT_EQ(points[0].get().chart, &chart);
    EXPECT_EQ(points[0].get().construction, construction);
    EXPECT_GT(points[0].get().evidence_version, version);
    EXPECT_EQ(points[0].get().supports, (std::vector<const int*>{&first, &second}));
    EXPECT_EQ(arena.live_records(), 4u);
    EXPECT_EQ(arena.record_capacity(), 4u);
    EXPECT_EQ(arena.slot_capacity(), 4u);
    EXPECT_EQ(work.stats.peak_live_vertex_records, 4u);
    EXPECT_EQ(work.stats.peak_vertex_record_capacity, 4u);
    EXPECT_EQ(work.stats.peak_vertex_slot_capacity, 4u);
    EXPECT_EQ(work.stats.vertex_relocation_bytes, 0u);
    EXPECT_EQ(work.stats.vertex_handle_copies, copies);
    EXPECT_EQ(work.stats.vertex_record_constructions, 5u);
    EXPECT_EQ(work.stats.vertex_evidence_extensions, 1u);
    // One original pointer plus two replacement pointers coexist at commit.
    EXPECT_EQ(work.stats.peak_incidence_bytes, 3 * sizeof(const int*));
    EXPECT_EQ(work.stats.peak_combined_geometry_bytes,
        work.live_arena_bytes + work.live_slot_bytes + 3 * sizeof(const int*));
    EXPECT_EQ(work.live_incidence_bytes, 2 * sizeof(const int*));
    RecordProperty("vertex_record_bytes", static_cast<int>(sizeof(TestArena::Record)));
}

TEST(OrdinaryAggregateArena, SharedEvidenceReplacementPreservesTheSiblingVersion)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2;
    auto target = arena.create({1, 2, 3, 4}, {4, 3, 2, 1}, {&first}, &source, &chart);
    auto sibling = target;
    const auto sibling_slot = sibling.slot_id();
    const auto version = sibling.get().evidence_version;
    arena.replace_with_extended_evidence(target, &second);
    EXPECT_EQ(sibling.slot_id(), sibling_slot);
    EXPECT_EQ(sibling.get().evidence_version, version);
    EXPECT_EQ(sibling.get().supports, std::vector<const int*>{&first});
    EXPECT_EQ(target.get().supports, (std::vector<const int*>{&first, &second}));
    EXPECT_GT(target.get().evidence_version, version);
    EXPECT_EQ(target.get().construction, sibling.get().construction);
    EXPECT_EQ(target.get().weights, sibling.get().weights);
    EXPECT_EQ(target.get().errors, sibling.get().errors);
    EXPECT_EQ(arena.live_records(), 2u);
}

TEST(OrdinaryAggregateArena, OneByteShortEvidenceReplacementPreservesEveryHandleAndAllowsRetry)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2;
    std::array<TestArena::Handle, 4> points;
    points[0] = arena.create({1, 2, 3, 4}, {4, 3, 2, 1}, {&first}, &source, &chart);
    for (unsigned i = 1; i < points.size(); ++i)
        points[i] = arena.create({static_cast<long double>(i), 0, 0, 0}, {}, {}, &source, &chart);
    const auto version = points[0].get().evidence_version;
    const auto construction = points[0].get().construction;
    const auto slot = points[0].slot_id();
    const auto admitted = work.charged_bytes;
    work.byte_budget = admitted + sizeof(TestArena::Record) + 2 * sizeof(const int*) - 1;
    EXPECT_THROW(arena.replace_with_extended_evidence(points[0], &second), std::bad_alloc);
    EXPECT_EQ(work.charged_bytes, admitted);
    EXPECT_EQ(points[0].get().evidence_version, version);
    EXPECT_EQ(points[0].get().construction, construction);
    EXPECT_EQ(points[0].slot_id(), slot);
    EXPECT_EQ(points[0].get().supports, std::vector<const int*>{&first});
    EXPECT_EQ(points[0].get().weights, (TestArena::Weights{1, 2, 3, 4}));
    EXPECT_EQ(points[0].get().errors, (TestArena::Weights{4, 3, 2, 1}));
    for (unsigned i = 1; i < points.size(); ++i)
        EXPECT_EQ(points[i].get().weights[0], i);
    EXPECT_EQ(arena.live_records(), 4u);
    EXPECT_EQ(arena.record_capacity(), 4u);
    EXPECT_EQ(arena.slot_capacity(), 4u);
    EXPECT_EQ(work.live_incidence_bytes, sizeof(const int*));
    work.byte_budget = std::numeric_limits<std::size_t>::max();
    arena.replace_with_extended_evidence(points[0], &second);
    EXPECT_GT(points[0].get().evidence_version, version);
    EXPECT_EQ(points[0].get().supports, (std::vector<const int*>{&first, &second}));
}

TEST(OrdinaryAggregateArena, SwapLastPreservesTheReplacedEvidenceVersion)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2;
    auto hole = arena.create({}, {}, {}, &source, &chart);
    auto last = arena.create({1, 2, 3, 4}, {4, 3, 2, 1}, {&first}, &source, &chart);
    arena.replace_with_extended_evidence(last, &second);
    const auto version = last.get().evidence_version;
    const auto construction = last.get().construction;
    const auto slot = last.slot_id();
    hole = {};
    EXPECT_EQ(last.get().evidence_version, version);
    EXPECT_EQ(last.get().construction, construction);
    EXPECT_EQ(last.slot_id(), slot);
    EXPECT_EQ(last.get().supports, (std::vector<const int*>{&first, &second}));
    EXPECT_EQ(last.get().weights, (TestArena::Weights{1, 2, 3, 4}));
    EXPECT_EQ(work.stats.vertex_relocation_bytes, sizeof(TestArena::Record));
}

TEST(OrdinaryAggregateArena, EvidenceCounterExhaustionRejectsBeforeReplacementCommit)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2;
    auto target = arena.create({}, {}, {&first}, &source, &chart);
    const auto version = target.get().evidence_version;
    const auto admitted = work.charged_bytes;
    work.stats.vertex_record_constructions = std::numeric_limits<std::size_t>::max();
    expect_error([&] { arena.replace_with_extended_evidence(target, &second); }, GeometryErrorCode::ResourceLimit);
    work.stats.vertex_record_constructions = 1;
    work.stats.vertex_evidence_extensions = std::numeric_limits<std::size_t>::max();
    expect_error([&] { arena.replace_with_extended_evidence(target, &second); }, GeometryErrorCode::ResourceLimit);
    work.stats.vertex_evidence_extensions = 0;
    EXPECT_EQ(work.charged_bytes, admitted);
    EXPECT_EQ(target.get().evidence_version, version);
    EXPECT_EQ(target.get().supports, std::vector<const int*>{&first});
    arena.replace_with_extended_evidence(target, &second);
    EXPECT_GT(target.get().evidence_version, version);
}

TEST(OrdinaryAggregateArena, SecondChargeFailureReleasesScratchAndRetainsSpentPrefixAndTargetVersion)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2;
    std::array<TestArena::Handle, 4> points;
    points[0] = arena.create({1, 2, 3, 4}, {4, 3, 2, 1}, {&first}, &source, &chart);
    for (unsigned i = 1; i < points.size(); ++i)
        points[i] = arena.create({static_cast<long double>(i), 0, 0, 0}, {}, {}, &source, &chart);
    const auto version = points[0].get().evidence_version;
    const auto construction = points[0].get().construction;
    const auto slot = points[0].slot_id();
    const auto admitted = work.charged_bytes;
    const auto replacement_charge = sizeof(TestArena::Record) + 2 * sizeof(const int*);
    // The complete replacement was admitted and allocated before this second
    // charge checks actual reserve capacity. Its spent prefix is never refunded.
    work.fail_at = work.charges + 1;
    EXPECT_THROW(arena.replace_with_extended_evidence(points[0], &second), std::bad_alloc);
    EXPECT_EQ(work.charged_bytes, admitted + replacement_charge);
    EXPECT_EQ(work.live_incidence_bytes, sizeof(const int*));
    EXPECT_GE(work.stats.peak_incidence_bytes, 3 * sizeof(const int*));
    EXPECT_EQ(points[0].get().evidence_version, version);
    EXPECT_EQ(points[0].get().construction, construction);
    EXPECT_EQ(points[0].slot_id(), slot);
    EXPECT_EQ(points[0].get().source, &source);
    EXPECT_EQ(points[0].get().chart, &chart);
    EXPECT_EQ(points[0].get().supports, std::vector<const int*>{&first});
    EXPECT_EQ(points[0].get().weights, (TestArena::Weights{1, 2, 3, 4}));
    EXPECT_EQ(points[0].get().errors, (TestArena::Weights{4, 3, 2, 1}));
    for (unsigned i = 1; i < points.size(); ++i)
        EXPECT_EQ(points[i].get().weights[0], i);
    EXPECT_EQ(arena.live_records(), 4u);
    EXPECT_EQ(arena.record_capacity(), 4u);
    EXPECT_EQ(arena.slot_capacity(), 4u);
    EXPECT_EQ(work.stats.vertex_record_constructions, 4u);
    EXPECT_EQ(work.stats.vertex_evidence_extensions, 0u);
    EXPECT_EQ(work.stats.vertex_relocation_bytes, 0u);
    work.fail_at = std::numeric_limits<std::size_t>::max();
    arena.replace_with_extended_evidence(points[0], &second);
    EXPECT_EQ(work.charged_bytes, admitted + 2 * replacement_charge);
    EXPECT_GT(points[0].get().evidence_version, version);
    EXPECT_EQ(points[0].get().supports, (std::vector<const int*>{&first, &second}));
    EXPECT_EQ(work.live_incidence_bytes, 2 * sizeof(const int*));
    EXPECT_EQ(work.stats.vertex_record_constructions, 5u);
    EXPECT_EQ(work.stats.vertex_evidence_extensions, 1u);
}

namespace
{
struct MetricTestPoly
{
    struct Face
    {
        std::vector<std::size_t> vertices;
        int source = -1;
    };
    std::vector<TestArena::Handle> vertices;
    std::vector<Face> faces;
    const TestArena::Weights& weights(std::size_t local) const { return vertices[local].get().weights; }
    const TestArena::Weights& errors(std::size_t local) const { return vertices[local].get().errors; }
};
using TestMetrics = SimpleFluid::geometry::swept_volume::details::OrdinaryMetricRecords<ArenaTestWork, long double>;

MetricTestPoly metric_triangle(TestArena& arena)
{
    MetricTestPoly poly;
    poly.vertices.push_back(arena.create({1, 0, 0, 0}, {0, 0x1p-40L, 0x1p-41L, 0x1p-42L}, {}, nullptr, nullptr));
    poly.vertices.push_back(arena.create({0, 1, 0, 0}, {0x1p-43L, 0, 0x1p-44L, 0x1p-45L}, {}, nullptr, nullptr));
    // Equal numerical rows with independently issued identity must remain distinct.
    poly.vertices.push_back(arena.create({1, 0, 0, 0}, {0, 0x1p-40L, 0x1p-41L, 0x1p-42L}, {}, nullptr, nullptr));
    poly.faces.push_back({{2, 0, 1}, 3});
    return poly;
}
} // namespace

TEST(OrdinaryAggregateMetric, FrozenRowsOutliveArenaAndPreserveEveryWeightErrorAndLocalIdentity)
{
    ArenaTestWork work;
    TestMetrics metrics(work);
    metrics.reserve_records(1, false);
    {
        TestArena arena(work);
        const auto bulk = metric_triangle(arena);
        MetricTestPoly gas;
        gas.vertices = {bulk.vertices[1], bulk.vertices[0]};
        gas.faces.push_back({{1, 0, 1}, -1});
        const std::array<const MetricTestPoly*, 3> channels{&bulk, &gas, nullptr};
        metrics.freeze(29, channels, nullptr);
        ASSERT_EQ(metrics.records().size(), 1u);
        const auto& frozen = metrics.records().front();
        ASSERT_EQ(frozen.rows.size(), 3u);
        ASSERT_EQ(frozen.local_to_numeric.size(), 5u);
        EXPECT_EQ(frozen.local_to_numeric[0], frozen.local_to_numeric[4]);
        EXPECT_EQ(frozen.local_to_numeric[1], frozen.local_to_numeric[3]);
        EXPECT_NE(frozen.local_to_numeric[0], frozen.local_to_numeric[2]);
    }
    EXPECT_EQ(work.stats.live_vertex_records, 0u);
    const auto& record = metrics.records().front();
    EXPECT_EQ(record.tetra, 29u);
    EXPECT_EQ(metrics.coefficient_count(), 0u);
    const auto bulk = TestMetrics::view(record, 0), gas = TestMetrics::view(record, 1);
    const std::array<TestArena::Weights, 3> expected_weights{{{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}}};
    const std::array<TestArena::Weights, 3> expected_errors{
        {{0, 0x1p-40L, 0x1p-41L, 0x1p-42L}, {0x1p-43L, 0, 0x1p-44L, 0x1p-45L}, {0, 0x1p-40L, 0x1p-41L, 0x1p-42L}}};
    for (unsigned local = 0; local < 3; ++local)
    {
        EXPECT_EQ(bulk.weights(local), expected_weights[local]);
        EXPECT_EQ(bulk.errors(local), expected_errors[local]);
    }
    EXPECT_EQ(gas.weights(0), expected_weights[1]);
    EXPECT_EQ(gas.errors(1), expected_errors[0]);
    EXPECT_EQ(bulk.faces.front().source, 3);
    EXPECT_EQ(gas.faces.front().source, -1);
    const std::vector<std::size_t> bulk_ring{2, 0, 1}, gas_ring{1, 0, 1};
    EXPECT_TRUE(std::equal(
        bulk.faces.front().vertices.begin(), bulk.faces.front().vertices.end(), bulk_ring.begin(), bulk_ring.end()));
    EXPECT_TRUE(std::equal(
        gas.faces.front().vertices.begin(), gas.faces.front().vertices.end(), gas_ring.begin(), gas_ring.end()));
    // Source-domain retry must still see its original local zero, even when
    // that point uses another channel's numeric row one.
    EXPECT_EQ(record.local_to_numeric[record.channels[1].local_first], record.local_to_numeric[1]);
    EXPECT_TRUE(TestMetrics::view(record, 2).faces.empty());
    EXPECT_EQ(work.stats.live_freeze_scratch_bytes, 0u);
    EXPECT_GT(work.stats.peak_freeze_scratch_bytes, 0u);
    EXPECT_GT(work.stats.live_compact_metric_bytes, 0u);
}

TEST(OrdinaryAggregateMetric, PureChannelsAliasBulkAndOnlyFirstAttemptStoresCoefficients)
{
    ArenaTestWork work;
    TestArena arena(work);
    const auto bulk = metric_triangle(arena);
    TestMetrics metrics(work);
    metrics.reserve_records(2, true);
    const std::array<long double, 4> coefficients{2, 3, 5, 7};
    metrics.freeze(0, std::array<const MetricTestPoly*, 3>{&bulk, &bulk, nullptr}, &coefficients);
    metrics.freeze(1, std::array<const MetricTestPoly*, 3>{&bulk, nullptr, &bulk}, &coefficients);
    ASSERT_EQ(metrics.records().size(), 2u);
    EXPECT_EQ(metrics.coefficient_count(), 2u);
    EXPECT_NE(metrics.records()[0].rows.data(), metrics.records()[1].rows.data());
    for (unsigned index = 0; index < 2; ++index)
    {
        const auto& record = metrics.records()[index];
        EXPECT_EQ(record.rows.size(), 3u);
        EXPECT_EQ(record.local_to_numeric.size(), 3u);
        EXPECT_EQ(record.faces.size(), 1u);
        const auto alias = index == 0 ? 1 : 2;
        EXPECT_EQ(TestMetrics::view(record, 0).faces.descriptors.data(),
            TestMetrics::view(record, alias).faces.descriptors.data());
        EXPECT_EQ(record.channels[0].local_first, record.channels[alias].local_first);
        EXPECT_TRUE(TestMetrics::view(record, index == 0 ? 2 : 1).faces.empty());
        EXPECT_EQ(metrics.coefficients(index), coefficients);
    }
}

TEST(OrdinaryAggregateMetric, FreezeAdmissionExactCapOneByteShortAndSpentPrefixAreMonotone)
{
    std::size_t exact_cap = 0, allocations = 0;
    {
        ArenaTestWork work;
        TestArena arena(work);
        const auto bulk = metric_triangle(arena);
        const auto charges_before = work.charges;
        TestMetrics metrics(work);
        metrics.reserve_records(1, false);
        metrics.freeze(0, std::array<const MetricTestPoly*, 3>{&bulk, &bulk, nullptr}, nullptr);
        exact_cap = work.charged_bytes;
        allocations = work.charges - charges_before;
    }
    for (bool short_by_one : {false, true})
    {
        ArenaTestWork work;
        work.byte_budget = exact_cap - short_by_one;
        TestArena arena(work);
        const auto bulk = metric_triangle(arena);
        const auto vertex = bulk.vertices.front().slot_id();
        const auto bytes_before = work.charged_bytes;
        {
            TestMetrics metrics(work);
            metrics.reserve_records(1, false);
            const auto freeze = [&]
            { metrics.freeze(0, std::array<const MetricTestPoly*, 3>{&bulk, &bulk, nullptr}, nullptr); };
            if (short_by_one)
                EXPECT_THROW(freeze(), std::bad_alloc);
            else
                EXPECT_NO_THROW(freeze());
        }
        EXPECT_EQ(work.stats.live_compact_metric_bytes, 0u);
        EXPECT_EQ(work.stats.live_freeze_scratch_bytes, 0u);
        EXPECT_GT(work.charged_bytes, bytes_before);
        EXPECT_EQ(bulk.vertices.front().slot_id(), vertex);
        EXPECT_EQ(bulk.weights(0), (TestArena::Weights{1, 0, 0, 0}));
        work.byte_budget = std::numeric_limits<std::size_t>::max();
        TestMetrics retry(work);
        retry.reserve_records(1, false);
        EXPECT_NO_THROW(retry.freeze(0, std::array<const MetricTestPoly*, 3>{&bulk, &bulk, nullptr}, nullptr));
        EXPECT_GT(work.charged_bytes, bytes_before);
    }
    // Every admitted allocation prefix is retained, while partial freeze
    // storage and scratch are released after each exceptional exit.
    for (std::size_t fail = 0; fail < allocations; ++fail)
    {
        ArenaTestWork work;
        TestArena arena(work);
        const auto bulk = metric_triangle(arena);
        const auto original = work.charged_bytes;
        work.fail_at = work.charges + fail;
        try
        {
            TestMetrics metrics(work);
            metrics.reserve_records(1, false);
            metrics.freeze(0, std::array<const MetricTestPoly*, 3>{&bulk, &bulk, nullptr}, nullptr);
        }
        catch (const std::bad_alloc&)
        {
            EXPECT_GE(work.charged_bytes, original);
        }
        EXPECT_EQ(work.stats.live_compact_metric_bytes, 0u);
        EXPECT_EQ(work.stats.live_freeze_scratch_bytes, 0u);
        EXPECT_EQ(arena.live_records(), 3u);
    }
}

TEST(OrdinaryAggregateMetric, OperationLimitAtIntegrationBoundaryRetainsCompletedConstructionAndSourceReuse)
{
    using namespace SimpleFluid::geometry::swept_volume;
    OrdinaryAggregateDonor donor;
    donor.id = 82;
    donor.geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    donor.pure_gas = true;
    donor.bulk_budget = donor.gas_budget = relative_allowance;
    const std::array donors{donor};
    const OrdinaryAggregateSource baseline(indexed_cube(), {}, ordinary_limits());
    const auto expected = baseline.measure(donors);
    ASSERT_EQ(expected.statistics.retries, 0u);
    const auto boundary = expected.statistics.metric_construction_operations;
    ASSERT_GT(boundary, 0u);
    ASSERT_LT(boundary, expected.statistics.operations);
    auto limits = ordinary_limits();
    limits.maximum_operations = boundary;
    const OrdinaryAggregateSource limited(indexed_cube(), {}, limits);
    for (unsigned attempt = 0; attempt < 2; ++attempt)
    {
        OrdinaryAggregateContext context;
        expect_error([&] { (void) limited.measure(donors, &context); }, GeometryErrorCode::ResourceLimit);
        EXPECT_EQ(context.stage, OrdinaryAggregateStage::Metric);
        EXPECT_EQ(context.retry, 0u);
        EXPECT_EQ(context.source, 0u);
        EXPECT_EQ(context.cut, OrdinaryAggregateContext::absent);
        EXPECT_EQ(context.operations, boundary);
        EXPECT_EQ(context.statistics.metric_construction_operations, boundary);
        EXPECT_EQ(context.statistics.vertex_record_constructions, expected.statistics.vertex_record_constructions);
        EXPECT_EQ(context.statistics.vertices_classified, expected.statistics.vertices_classified);
        EXPECT_EQ(context.statistics.witnesses, expected.statistics.witnesses);
        EXPECT_EQ(context.statistics.live_vertex_records, 0u);
        EXPECT_EQ(context.statistics.live_compact_metric_bytes, 0u);
        EXPECT_EQ(context.statistics.live_freeze_scratch_bytes, 0u);
        EXPECT_GT(context.statistics.peak_compact_metric_bytes, 0u);
        ASSERT_EQ(limited.points().size(), 8u);
        EXPECT_EQ(limited.points()[0].value, (std::array<long double, 3>{0, 0, 0}));
    }
    expect_amount(expected.groups.front().nominal.bulk, 1);
    expect_amount(expected.groups.front().nominal.gas, 1);
    EXPECT_EQ(expected.groups.front().nominal.carrier, 0);
    expect_identical_results(expected, baseline.measure(donors));
}

TEST(OrdinaryAggregateSlotBuffer, OwnsOneBindingAndPreservesBorrowedCopyMoveAndReverseLocalOrder)
{
    static_assert(sizeof(TestArena::Handle) == 2 * sizeof(std::size_t));
    static_assert(sizeof(TestArena::SlotBuffer::value_type) == sizeof(std::size_t));
    static_assert(sizeof(TestArena::SlotBuffer) == sizeof(void*) + sizeof(std::vector<TestArena::SlotId>));
    ArenaTestWork work;
    TestArena arena(work);
    {
        TestArena::SlotBuffer original(arena);
        original.reserve(3);
        for (unsigned local = 0; local < 3; ++local)
            original.push_back(arena.create({local + 1.L, 0, 0, 0}, {}, {}, nullptr, nullptr));
        const auto copies_before_borrow = work.stats.vertex_handle_copies;
        std::vector<TestArena::SlotId> order;
        for (auto borrowed : original)
            order.push_back(borrowed.slot_id());
        EXPECT_EQ(work.stats.vertex_handle_copies, copies_before_borrow);
        EXPECT_EQ(original[1].get().weights[0], 2);
        {
            TestArena::SlotBuffer copy(original);
            EXPECT_EQ(arena.live_records(), 3u);
            EXPECT_EQ(work.stats.vertex_handle_copies, copies_before_borrow + 3);
            const auto copies_before_move = work.stats.vertex_handle_copies;
            TestArena::SlotBuffer moved(std::move(copy));
            EXPECT_TRUE(copy.empty());
            EXPECT_EQ(copy.arena(), nullptr);
            EXPECT_EQ(work.stats.vertex_handle_copies, copies_before_move);
            for (unsigned local = 0; local < 3; ++local)
            {
                EXPECT_EQ(moved[local].slot_id(), order[local]);
                EXPECT_EQ(moved[local].get().weights[0], local + 1.L);
            }
        }
        EXPECT_EQ(arena.live_records(), 3u);
        EXPECT_EQ(work.live_vertex_index_bytes, original.capacity() * sizeof(std::size_t));
    }
    EXPECT_EQ(arena.live_records(), 0u);
    EXPECT_EQ(work.live_vertex_index_bytes, 0u);
    EXPECT_EQ(work.stats.vertex_relocation_bytes, 0u);
}

TEST(OrdinaryAggregateSlotBuffer, CopyAllocationFailuresPreserveSourceTargetAndSpentPrefix)
{
    for (std::size_t rejected_charge : {0u, 1u})
    {
        ArenaTestWork work;
        TestArena arena(work);
        TestArena::SlotBuffer source(arena), target(arena);
        source.reserve(2);
        target.reserve(1);
        source.push_back(arena.create({1, 0, 0, 0}, {}, {}, nullptr, nullptr));
        source.push_back(arena.create({0, 1, 0, 0}, {}, {}, nullptr, nullptr));
        target.push_back(arena.create({0, 0, 1, 0}, {}, {}, nullptr, nullptr));
        const auto source_first = source[0].slot_id(), target_first = target[0].slot_id();
        const auto admitted = work.charged_bytes, live_bytes = work.live_vertex_index_bytes;
        const auto copies = work.stats.vertex_handle_copies;
        work.fail_at = work.charges + rejected_charge;
        EXPECT_THROW(target = source, std::bad_alloc);
        EXPECT_EQ(source.size(), 2u);
        EXPECT_EQ(target.size(), 1u);
        EXPECT_EQ(source[0].slot_id(), source_first);
        EXPECT_EQ(target[0].slot_id(), target_first);
        EXPECT_EQ(arena.live_records(), 3u);
        EXPECT_EQ(work.live_vertex_index_bytes, live_bytes);
        EXPECT_EQ(work.stats.vertex_handle_copies, copies);
        EXPECT_EQ(work.charged_bytes, admitted + rejected_charge * 2 * sizeof(std::size_t));
        work.fail_at = std::numeric_limits<std::size_t>::max();
        target = source;
        ASSERT_EQ(target.size(), 2u);
        EXPECT_EQ(target[0].slot_id(), source_first);
        EXPECT_EQ(target[1].get().weights, (TestArena::Weights{0, 1, 0, 0}));
        EXPECT_EQ(arena.live_records(), 2u);
    }
}

TEST(OrdinaryAggregateSlotBuffer, EvidenceReplacementPreservesSharedVersionsAndFailureLocalIndices)
{
    for (bool shared : {false, true})
    {
        ArenaTestWork work;
        TestArena arena(work);
        int first = 1, second = 2;
        TestArena::SlotBuffer target(arena);
        target.reserve(4);
        target.push_back(arena.create({1, 2, 3, 4}, {4, 3, 2, 1}, {&first}, &first, &second));
        for (unsigned local = 1; local < 4; ++local)
            target.push_back(arena.create({}, {}, {}, nullptr, nullptr));
        std::optional<TestArena::SlotBuffer> sibling;
        if (shared)
            sibling.emplace(target);
        const auto old_slot = target[0].slot_id(), following = target[1].slot_id();
        const auto construction = target[0].get().construction, version = target[0].get().evidence_version;
        const auto incidence_bytes = work.live_incidence_bytes, slot_bytes = work.live_vertex_index_bytes;
        const auto copies = work.stats.vertex_handle_copies, admitted = work.charged_bytes;
        work.fail_at = work.charges + 1;
        EXPECT_THROW(arena.replace_with_extended_evidence(target, 0, &second), std::bad_alloc);
        EXPECT_EQ(target[0].slot_id(), old_slot);
        EXPECT_EQ(target[0].get().evidence_version, version);
        EXPECT_EQ(target[0].get().supports, (std::vector<const int*>{&first}));
        EXPECT_EQ(target[1].slot_id(), following);
        EXPECT_EQ(work.live_incidence_bytes, incidence_bytes);
        EXPECT_EQ(work.live_vertex_index_bytes, slot_bytes);
        EXPECT_EQ(arena.live_records(), 4u);
        EXPECT_GT(work.charged_bytes, admitted);
        work.fail_at = std::numeric_limits<std::size_t>::max();
        arena.replace_with_extended_evidence(target, 0, &second);
        EXPECT_EQ(target[0].get().weights, (TestArena::Weights{1, 2, 3, 4}));
        EXPECT_EQ(target[0].get().errors, (TestArena::Weights{4, 3, 2, 1}));
        EXPECT_EQ(target[0].get().construction, construction);
        EXPECT_GT(target[0].get().evidence_version, version);
        EXPECT_EQ(target[0].get().supports, (std::vector<const int*>{&first, &second}));
        EXPECT_EQ(target[1].slot_id(), following);
        EXPECT_EQ(work.stats.vertex_handle_copies, copies);
        if (shared)
        {
            EXPECT_NE(target[0].slot_id(), old_slot);
            EXPECT_EQ((*sibling)[0].slot_id(), old_slot);
            EXPECT_EQ((*sibling)[0].get().evidence_version, version);
            EXPECT_EQ((*sibling)[0].get().supports, (std::vector<const int*>{&first}));
            EXPECT_EQ(arena.live_records(), 5u);
        }
        else
        {
            EXPECT_EQ(target[0].slot_id(), old_slot);
            EXPECT_EQ(arena.live_records(), 4u);
        }
    }
}

TEST(OrdinaryAggregateMetric, CombinedPeakIncludesOldGeometryPreviousRecordNewBuffersAndFreezeScratch)
{
    ArenaTestWork work;
    TestArena arena(work);
    const auto bulk = metric_triangle(arena);
    // This test owns and counts the physical synthetic Poly, including its
    // inline vector descriptors, retained Handle buffer, and original face ring.
    const auto old_poly = sizeof(MetricTestPoly) + bulk.vertices.capacity() * sizeof(TestArena::Handle) +
                          bulk.faces.capacity() * sizeof(MetricTestPoly::Face) +
                          bulk.faces.front().vertices.capacity() * sizeof(std::size_t);
    work.external_connectivity_bytes = old_poly;
    work.observe_geometry();
    // Three issued records require two two-record pages, a two-pointer page
    // directory, and one four-slot block with its one-pointer directory.
    using Page = std::array<std::optional<TestArena::Record>, 2>;
    struct SlotLayout
    {
        TestArena::Record* record;
        std::size_t references, next_free;
    };
    using SlotBlock = std::array<SlotLayout, 4>;
    const auto old_arena = sizeof(TestArena) + 2 * sizeof(std::unique_ptr<Page>) + 2 * sizeof(Page);
    const auto old_slots = sizeof(std::unique_ptr<SlotBlock>) + sizeof(SlotBlock);
    ASSERT_EQ(arena.record_capacity(), 4u);
    ASSERT_EQ(arena.slot_capacity(), 4u);
    ASSERT_EQ(work.live_arena_bytes, old_arena);
    ASSERT_EQ(work.live_slot_bytes, old_slots);
    ASSERT_EQ(work.live_incidence_bytes, 0u);
    TestMetrics metrics(work);
    metrics.reserve_records(2, false);
    ASSERT_EQ(metrics.records().capacity(), 2u);
    const auto descriptors = sizeof(TestMetrics) + 2 * sizeof(TestMetrics::Record);
    const auto payload =
        3 * sizeof(TestMetrics::Row) + 3 * sizeof(std::size_t) + sizeof(TestMetrics::Face) + 3 * sizeof(std::size_t);
    const auto freeze_map = 3 * sizeof(std::size_t);
    const std::array<const MetricTestPoly*, 3> channels{&bulk, &bulk, nullptr};
    metrics.freeze(0, channels, nullptr);
    EXPECT_EQ(
        work.stats.peak_combined_geometry_bytes, old_poly + old_arena + old_slots + descriptors + payload + freeze_map);
    const auto first_peak = work.stats.peak_combined_geometry_bytes;
    metrics.freeze(1, channels, nullptr);
    for (const auto& record : metrics.records())
    {
        ASSERT_EQ(record.rows.capacity(), 3u);
        ASSERT_EQ(record.local_to_numeric.capacity(), 3u);
        ASSERT_EQ(record.faces.capacity(), 1u);
        ASSERT_EQ(record.rings.capacity(), 3u);
    }
    // The first tetra's completed payload remains live while the next complete
    // payload and its slot map coexist with every old construction allocation.
    const auto expected_peak = old_poly + old_arena + old_slots + descriptors + payload + payload + freeze_map;
    EXPECT_EQ(work.stats.peak_combined_geometry_bytes, expected_peak);
    EXPECT_GT(expected_peak, first_peak);
    EXPECT_EQ(work.stats.peak_compact_metric_bytes, descriptors + 2 * payload);
    EXPECT_EQ(work.stats.peak_freeze_scratch_bytes, freeze_map);
    EXPECT_EQ(work.stats.live_freeze_scratch_bytes, 0u);
    EXPECT_EQ(arena.live_records(), 3u);
}

TEST(OrdinaryAggregateSlotBuffer, MixedStandaloneHandleAndBufferPreserveAllSharedEvidenceVersions)
{
    ArenaTestWork work;
    TestArena arena(work);
    int source = 0, chart = 0, first = 1, second = 2, third = 3;
    const TestArena::Weights weights{.125L, .25L, .5L, .125L}, errors{0x1p-43L, 0x1p-44L, 0x1p-45L, 0x1p-46L};
    auto original = arena.create(weights, errors, {&first}, &source, &chart);
    const auto original_slot = original.slot_id();
    const auto original_version = original.get().evidence_version;
    const auto construction = original.get().construction;
    TestArena::SlotBuffer buffer(arena);
    buffer.reserve(1);
    buffer.push_back(original);
    const auto copies_before_replacement = work.stats.vertex_handle_copies;
    arena.replace_with_extended_evidence(buffer, 0, &second);
    EXPECT_EQ(work.stats.vertex_handle_copies, copies_before_replacement);
    EXPECT_EQ(original.slot_id(), original_slot);
    EXPECT_EQ(original.get().evidence_version, original_version);
    EXPECT_EQ(original.get().supports, (std::vector<const int*>{&first}));
    EXPECT_NE(buffer[0].slot_id(), original_slot);
    EXPECT_EQ(buffer[0].get().supports, (std::vector<const int*>{&first, &second}));
    const auto buffer_slot = buffer[0].slot_id();
    const auto buffer_version = buffer[0].get().evidence_version;
    auto standalone = buffer[0].retain();
    const auto copies_before_second_replacement = work.stats.vertex_handle_copies;
    arena.replace_with_extended_evidence(standalone, &third);
    EXPECT_EQ(work.stats.vertex_handle_copies, copies_before_second_replacement);
    EXPECT_EQ(buffer[0].slot_id(), buffer_slot);
    EXPECT_EQ(buffer[0].get().evidence_version, buffer_version);
    EXPECT_EQ(buffer[0].get().supports, (std::vector<const int*>{&first, &second}));
    EXPECT_NE(standalone.slot_id(), buffer_slot);
    EXPECT_EQ(standalone.get().supports, (std::vector<const int*>{&first, &second, &third}));
    for (const auto* record : {&original.get(), &buffer[0].get(), &standalone.get()})
    {
        EXPECT_EQ(record->weights, weights);
        EXPECT_EQ(record->errors, errors);
        EXPECT_EQ(record->construction, construction);
        EXPECT_EQ(record->source, &source);
        EXPECT_EQ(record->chart, &chart);
    }
    EXPECT_GT(buffer_version, original_version);
    EXPECT_GT(standalone.get().evidence_version, buffer_version);
    EXPECT_EQ(arena.live_records(), 3u);
}

namespace
{
auto geometry_counter_values()
{
    constexpr std::array members{&GeometryStatistics::source_imports, &GeometryStatistics::constructions,
        &GeometryStatistics::intersections, &GeometryStatistics::support_distances,
        &GeometryStatistics::volume_evaluations, &GeometryStatistics::moment_evaluations,
        &GeometryStatistics::partition_checks, &GeometryStatistics::structural_partition_reuses,
        &GeometryStatistics::floating_operations, &GeometryStatistics::interval_operations,
        &GeometryStatistics::ledger_operations, &GeometryStatistics::wider_routes, &GeometryStatistics::rejections,
        &GeometryStatistics::runtime_dispatches, &GeometryStatistics::exact_predicates,
        &GeometryStatistics::rational_constructions, &GeometryStatistics::rational_measures,
        &GeometryStatistics::rational_ledgers};
    const auto statistics = geometry_statistics();
    std::array<std::uint64_t, members.size()> values{};
    for (std::size_t i = 0; i < members.size(); ++i)
        values[i] = statistics.*members[i];
    return values;
}
} // namespace

TEST(OrdinaryAggregateMetric, BorrowedRetentionObserverPreservesEnvironmentCountersBudgetsAndFailure)
{
    using namespace SimpleFluid::geometry::swept_volume;
    namespace Retention = SimpleFluid::PLIC::details::retention;
    namespace AggregateDiagnostics = SimpleFluid::geometry::swept_volume::details;
    struct Environment
    {
        std::fenv_t saved{};
        Environment() { std::fegetenv(&saved); }
        ~Environment() { std::fesetenv(&saved); }
    } environment;
    OrdinaryAggregateDonor donor;
    donor.id = 132;
    donor.geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    donor.pure_carrier = true;
    donor.bulk_budget = donor.carrier_budget = relative_allowance;
    const std::array donors{donor};
    const OrdinaryAggregateSource source(indexed_cube(), {}, ordinary_limits());
    const auto warm = source.measure(donors);
    auto limits = ordinary_limits();
    limits.maximum_operations = warm.statistics.metric_construction_operations;
    const OrdinaryAggregateSource limited(indexed_cube(), {}, limits);
    std::vector<std::byte> scratch_bytes(2 * 1024 * 1024);
    Retention::ProbeScratch scratch(scratch_bytes);
    struct ObserverState
    {
        Retention::ProbeScratch& scratch;
        std::size_t calls = 0, exits = 0, owners = 0;
        bool complete = true, neutral = true;
    } observer{scratch};
    const auto callback =
        +[](const AggregateDiagnostics::OrdinaryAggregateRetentionObservation& observation, void* context)
    {
        auto& state = *static_cast<ObserverState*>(context);
        const auto before = geometry_counter_values();
        const auto round = std::fegetround(), flags = std::fetestexcept(FE_ALL_EXCEPT);
        std::array<Retention::BorrowedRoot, 9> roots;
        std::size_t count = 0;
        for (const auto& point : observation.source_points)
            roots[count++] = Retention::root(point, 7);
        for (const auto& declared : observation.donors)
            roots[count++] = Retention::root(*declared.geometry, 7);
        state.scratch.reset();
        const std::array ledgers{&observation.ledger};
        {
            const auto report = Retention::inspect_retention(std::span(roots).first(count), ledgers, state.scratch);
            state.complete &= report.complete() && report.root_count == 9 && report.ledger_count == 1;
            state.owners = std::max(state.owners, report.owner_union.owners);
        }
        state.neutral &= geometry_counter_values() == before && std::fegetround() == round &&
                         std::fetestexcept(FE_ALL_EXCEPT) == flags;
        ++state.calls;
        state.exits += observation.stage == "work-scope-exit";
    };
    struct Outcome
    {
        std::optional<OrdinaryAggregateResult> result;
        std::optional<GeometryErrorCode> error;
        std::string message;
        OrdinaryAggregateContext context;
        decltype(geometry_counter_values()) counters{};
        std::size_t caller_bytes = 0, caller_operations = 0, caller_pairs = 0;
        bool caller_restored = false;
        int rounding = 0, flags = 0;
    };
    const auto run = [&](const OrdinaryAggregateSource& input, bool observe)
    {
        Outcome outcome;
        std::fesetround(FE_UPWARD);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO | FE_INEXACT);
        SimpleFluid::PLIC::details::Operation caller(ordinary_limits());
        caller.work.bytes = 101;
        caller.work.operations = 17;
        caller.work.pairs = 7;
        const auto before = geometry_counter_values();
        const auto measure = [&]
        {
            try
            {
                outcome.result = input.measure(donors, &outcome.context);
            }
            catch (const GeometryError& error)
            {
                outcome.error = error.code();
                outcome.message = error.what();
            }
        };
        if (observe)
        {
            AggregateDiagnostics::ScopedOrdinaryAggregateRetentionObserver scope(callback, &observer);
            measure();
        }
        else
            measure();
        outcome.caller_restored = SimpleFluid::PLIC::details::active_work == &caller.work;
        outcome.caller_bytes = caller.work.bytes;
        outcome.caller_operations = caller.work.operations;
        outcome.caller_pairs = caller.work.pairs;
        outcome.rounding = std::fegetround();
        outcome.flags = std::fetestexcept(FE_ALL_EXCEPT);
        const auto after = geometry_counter_values();
        for (std::size_t i = 0; i < before.size(); ++i)
            outcome.counters[i] = after[i] - before[i];
        return outcome;
    };
    for (const auto* input : {&source, &limited})
    {
        const auto plain = run(*input, false), observed = run(*input, true);
        EXPECT_EQ(observed.rounding, FE_UPWARD);
        EXPECT_EQ(observed.flags, FE_DIVBYZERO | FE_INEXACT);
        EXPECT_EQ(observed.rounding, plain.rounding);
        EXPECT_EQ(observed.flags, plain.flags);
        EXPECT_EQ(observed.counters, plain.counters);
        EXPECT_TRUE(plain.caller_restored);
        EXPECT_TRUE(observed.caller_restored);
        EXPECT_EQ(observed.caller_bytes, plain.caller_bytes);
        EXPECT_EQ(observed.caller_operations, plain.caller_operations);
        EXPECT_EQ(observed.caller_pairs, plain.caller_pairs);
        EXPECT_EQ(observed.error, plain.error);
        EXPECT_EQ(observed.message, plain.message);
        EXPECT_EQ(observed.context.bytes, plain.context.bytes);
        EXPECT_EQ(observed.context.pairs, plain.context.pairs);
        EXPECT_EQ(observed.context.operations, plain.context.operations);
        EXPECT_EQ(observed.context.stage, plain.context.stage);
        EXPECT_EQ(observed.context.donor, plain.context.donor);
        EXPECT_EQ(observed.context.source, plain.context.source);
        EXPECT_EQ(observed.context.cut, plain.context.cut);
        EXPECT_EQ(observed.context.retry, plain.context.retry);
        if (plain.result)
        {
            ASSERT_TRUE(observed.result);
            expect_identical_results(*plain.result, *observed.result);
            expect_amount(observed.result->groups.front().nominal.bulk, 1);
            expect_amount(observed.result->groups.front().nominal.carrier, 1);
        }
        else
        {
            EXPECT_FALSE(observed.result);
            EXPECT_EQ(observed.error, GeometryErrorCode::ResourceLimit);
            EXPECT_EQ(observed.context.operations, limits.maximum_operations);
        }
    }
    EXPECT_TRUE(observer.complete);
    EXPECT_TRUE(observer.neutral);
    EXPECT_GT(observer.calls, 2u);
    EXPECT_EQ(observer.exits, 2u);
    EXPECT_GT(observer.owners, 0u);
}
