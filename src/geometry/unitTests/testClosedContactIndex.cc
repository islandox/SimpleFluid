/** Independent closed-range, provider-image and transactional admission oracles. */
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/ClosedContactIndex.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
using Cell = P::ConvexCellGeometry<ordinary>;
using Point = P::Point<ordinary>;
using Wide = std::array<long double, 3>;
constexpr std::size_t ample = 16 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

AMR::AdaptiveCellKey key(std::uint64_t id, std::uint64_t region = 0)
{
    return {region, AMR::ConvexCellAncestry{id, {1, 2}}};
}
AMR::AdaptiveContactImage image(Wide lower, Wide upper)
{
    AMR::AdaptiveContactImage result;
    result.bounds = {lower, upper};
    return result;
}
D::ClosedContactEntry entry(std::uint64_t id, Wide lower, Wide upper, char domain = 'd', int owner = 0)
{
    return {key(id), {domain}, 1, owner, {image(lower, upper)}};
}
std::vector<AMR::AdaptiveCellKey> keys(const std::vector<const D::ClosedContactEntry*>& values)
{
    std::vector<AMR::AdaptiveCellKey> result;
    for (const auto* value : values)
        result.push_back(value->key);
    return result;
}
std::vector<AMR::AdaptiveCellKey> query(
    D::ClosedContactIndex& index, const AMR::AdaptiveContactImage& value, char domain = 'd')
{
    const std::array images{value};
    const std::array domains{domain};
    return keys(index.query(domains, images));
}
AMR::AdaptiveLeaf cartesian_leaf(
    const AMR::AMRGrid& grid, unsigned level, std::array<std::uint64_t, 3> index, char domain = 'd')
{
    AMR::AdaptiveLeaf leaf;
    leaf.key = {grid.region_id, AMR::CartesianCellAncestry{level, index}};
    leaf.domain = {domain};
    leaf.provider = AMR::AdaptiveProviderKind::Cartesian;
    return leaf;
}
D::ClosedContactEntry contact_entry(const AMR::AdaptiveLeaf& leaf, const AMR::AdaptiveProviderKernel& kernel)
{
    return {leaf.key, leaf.domain, leaf.revision, 0, kernel.closed_contact_images(leaf)};
}
class UnqualifiedKernel : public AMR::AdaptiveProviderKernel
{
public:
    AMR::AdaptiveBytes descriptor() const override { return {'u'}; }
    P::ConvexGeometryPtr<ordinary> geometry(const AMR::AdaptiveLeaf& leaf) const override { return leaf.geometry; }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf& leaf) const override { return leaf.surfaces; }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf&, AMR::AdaptiveSplit) const override
    {
        throw std::invalid_argument("Fixture has no split implementation.");
    }
};

TEST(ClosedContactIndex, CartesianLogicalOracleIncludesFacesEdgesVerticesAndRootSeams)
{
    AMR::AMRGrid grid;
    grid.root_blocks = {2, 2, 2};
    grid.block_cells = {2, 2, 2};
    grid.origin = {7, 11, 17};
    grid.extent = {8, 16, 4};
    grid.max_level = 3;
    const auto kernel = AMR::cartesian_adaptive_kernel(grid);
    const auto target = cartesian_leaf(grid, 0, {1, 1, 1});
    // In coarse-cell units the target is [1,2]^3. These contacts are derived
    // from integer dyadic intervals, independently of the provider coordinates.
    const std::vector<AMR::AdaptiveLeaf> leaves{cartesian_leaf(grid, 2, {8, 4, 4}), // Face.
        cartesian_leaf(grid, 2, {8, 8, 4}),                                         // Edge.
        cartesian_leaf(grid, 2, {8, 8, 8}),                                         // Vertex.
        cartesian_leaf(grid, 1, {1, 2, 2}),                                         // Opposite face, finer level.
        cartesian_leaf(grid, 0, {2, 1, 1}),                                         // Adjacent root block.
        cartesian_leaf(grid, 2, {9, 9, 9}),                                         // Positive gap on every axis.
        cartesian_leaf(grid, 0, {0, 0, 0}), // Touches the opposite target vertex.
        cartesian_leaf(grid, 0, {3, 3, 3})};
    std::vector<AMR::AdaptiveCellKey> expected;
    for (const auto ordinal : {0u, 1u, 2u, 3u, 4u, 6u})
        expected.push_back(leaves[ordinal].key);
    std::sort(expected.begin(), expected.end());
    const auto target_images = kernel->closed_contact_images(target);
    ASSERT_EQ(target_images.size(), 1u);
    EXPECT_EQ(target_images[0].bounds.lower, (Wide{9, 15, 18}));
    EXPECT_EQ(target_images[0].bounds.upper, (Wide{11, 19, 19}));
    ASSERT_TRUE(target_images[0].logical);
    EXPECT_FALSE(target_images[0].logical->exact_embedding);
    const auto capabilities = kernel->contact_capabilities(target);
    EXPECT_EQ(capabilities.normalization, AMR::AdaptiveContactNormalization::PhysicalIdentity);
    EXPECT_FALSE(capabilities.split_union_preserving);
    EXPECT_FALSE(capabilities.coarsen_union_preserving);
    for (unsigned permutation = 0; permutation < 3; ++permutation)
    {
        D::ClosedContactIndex index(ample);
        std::vector<std::size_t> order(leaves.size());
        std::iota(order.begin(), order.end(), 0);
        if (permutation == 1)
            std::reverse(order.begin(), order.end());
        if (permutation == 2)
            std::rotate(order.begin(), order.begin() + 3, order.end());
        for (const auto ordinal : order)
            index.insert(contact_entry(leaves[ordinal], *kernel));
        EXPECT_EQ(keys(index.query(target.domain, target_images)), expected);
        for (auto it = leaves.rbegin(); it != leaves.rend(); ++it)
            (void) index.query(it->domain, kernel->closed_contact_images(*it));
        EXPECT_EQ(keys(index.query(target.domain, target_images)), expected);
    }
}

TEST(ClosedContactIndex, PeriodicFacesEdgesAndCornersRetainPhysicalImageDisplacements)
{
    AMR::AMRGrid grid;
    grid.root_blocks = {2, 2, 2};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 4, 4};
    grid.periodic = {true, true, true};
    const auto kernel = AMR::cartesian_adaptive_kernel(grid);
    const auto target = cartesian_leaf(grid, 0, {0, 0, 0});
    EXPECT_EQ(kernel->contact_capabilities(target).normalization, AMR::AdaptiveContactNormalization::PeriodicCartesian);
    const std::array leaves{
        cartesian_leaf(grid, 0, {3, 0, 0}), cartesian_leaf(grid, 0, {3, 3, 0}), cartesian_leaf(grid, 0, {3, 3, 3})};
    D::ClosedContactIndex index(ample);
    std::vector<AMR::AdaptiveCellKey> expected;
    for (std::size_t ordinal = 0; ordinal < leaves.size(); ++ordinal)
    {
        auto record = contact_entry(leaves[ordinal], *kernel);
        EXPECT_EQ(record.images.size(), std::size_t(1) << (ordinal + 1));
        for (const auto& value : record.images)
        {
            ASSERT_TRUE(value.logical);
            EXPECT_TRUE(value.logical->exact_embedding);
            Wide physical{}, canonical{};
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const bool seam = value.bounds.lower[axis] == 0 && value.bounds.upper[axis] == 0;
                canonical[axis] = seam ? 0 : (value.bounds.lower[axis] + value.bounds.upper[axis]) / 2;
                physical[axis] = seam ? 4 : canonical[axis];
            }
            const auto source = P::details::GeometryIdentityAccess::point(physical);
            const auto normalized = P::point_translated<ordinary>(source, value.physical_to_contact);
            EXPECT_EQ(normalized, P::details::GeometryIdentityAccess::point(canonical));
            EXPECT_EQ(kernel->contact_point(leaves[ordinal], source), normalized);
        }
        expected.push_back(record.key);
        index.insert(std::move(record));
    }
    index.insert(contact_entry(cartesian_leaf(grid, 0, {2, 2, 2}), *kernel));
    auto disconnected = contact_entry(leaves.back(), *kernel);
    disconnected.key.region = 99;
    disconnected.domain = {'x'};
    index.insert(disconnected);
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(keys(index.query(target.domain, kernel->closed_contact_images(target))), expected);
    EXPECT_EQ(query(index, image({0, 0, 0}, {0, 0, 0}), 'x'), (std::vector<AMR::AdaptiveCellKey>{disconnected.key}));
    index.erase(leaves.back().key);
    for (const auto& value : kernel->closed_contact_images(leaves.back()))
    {
        const auto remaining = query(index, value);
        EXPECT_EQ(std::count(remaining.begin(), remaining.end(), leaves.back().key), 0);
    }
}

TEST(ClosedContactIndex, MixedRoutesMeetByPhysicalCoordinatesWithinOneDomain)
{
    AMR::AMRGrid left, right;
    left.region_id = 11;
    right.region_id = 22;
    left.block_cells = right.block_cells = {2, 2, 2};
    left.extent = right.extent = {2, 2, 2};
    right.origin = {2, 0, 0};
    AMR::DistributedMixedRoute first{
        1, 11, 101, AMR::AdaptiveProviderKind::Cartesian, AMR::cartesian_adaptive_kernel(left), {}, false};
    AMR::DistributedMixedRoute second{
        2, 22, 202, AMR::AdaptiveProviderKind::Cartesian, AMR::cartesian_adaptive_kernel(right), {}, false};
    const auto mixed = AMR::mixed_adaptive_kernel({first, second}, ample);
    auto a = cartesian_leaf(left, 0, {1, 0, 0}), b = cartesian_leaf(right, 0, {0, 0, 0});
    const auto bind = [](AMR::AdaptiveLeaf& leaf, std::uint64_t route, std::uint64_t region)
    {
        leaf.key.region = region;
        leaf.join_recipe = std::make_shared<const AMR::AdaptiveBytes>(
            AMR::encode_join_capsule(AMR::AdaptiveJoinCapsule{route, {}, {}}, ample));
    };
    bind(a, 1, 101);
    bind(b, 2, 202);
    D::ClosedContactIndex index(ample);
    index.insert(contact_entry(b, *mixed));
    auto foreign_domain = b;
    foreign_domain.key.ancestry = AMR::CartesianCellAncestry{1, {0, 0, 0}};
    foreign_domain.domain = {'x'};
    index.insert(contact_entry(foreign_domain, *mixed));
    EXPECT_EQ(keys(index.query(a.domain, mixed->closed_contact_images(a))), (std::vector<AMR::AdaptiveCellKey>{b.key}));
    EXPECT_EQ(mixed->contact_capabilities(a).normalization, AMR::AdaptiveContactNormalization::PhysicalIdentity);
    second.periodic = true;
    EXPECT_THROW((void) AMR::mixed_adaptive_kernel({first, second}, ample), std::invalid_argument);
}

TEST(ClosedContactIndex, ConvexHullBoundsPreserveSkewTinySliverAndClosedTouch)
{
    const auto kernel = AMR::convex_adaptive_kernel(4, 0);
    for (const auto [scale, height] :
        std::array<std::pair<long double, long double>, 3>{{{1, 2}, {0x1p-300L, 0x1p-299L}, {1, 0x1p-40L}}})
    {
        AMR::AdaptiveLeaf a;
        a.key = key(1);
        a.domain = {'d'};
        a.geometry = Cell::tetrahedron(
            {Point({0, 0, 0}), Point({2 * scale, 0, 0}), Point({0, 2 * scale, 0}), Point({0, 0, height})});
        const auto values = kernel->closed_contact_images(a);
        ASSERT_EQ(values.size(), 1u);
        EXPECT_EQ(values[0].bounds.lower, (Wide{0, 0, 0}));
        EXPECT_EQ(values[0].bounds.upper, (Wide{2 * scale, 2 * scale, height}));
        D::ClosedContactIndex index(ample);
        index.insert(contact_entry(a, *kernel));
        EXPECT_EQ(query(index, image({2 * scale, 0, 0}, {3 * scale, scale, height})),
            (std::vector<AMR::AdaptiveCellKey>{a.key}));
        const auto separated = std::nextafter(2 * scale, std::numeric_limits<long double>::infinity());
        EXPECT_TRUE(query(index, image({separated, 0, 0}, {3 * scale, scale, height})).empty());
    }
    AMR::AdaptiveLeaf opposite;
    opposite.key = key(2);
    opposite.domain = {'d'};
    opposite.geometry = Cell::tetrahedron({Point({2, 2, 2}), Point({0, 2, 2}), Point({2, 0, 2}), Point({2, 2, 0})});
    D::ClosedContactIndex index(ample);
    index.insert(contact_entry(opposite, *kernel));
    // This tetrahedron satisfies x+y+z >= 4, whereas the first unit-scale
    // tetrahedron satisfies x+y+z <= 2. Equal hull boxes are only candidates.
    EXPECT_EQ(query(index, image({0, 0, 0}, {2, 2, 2})), (std::vector<AMR::AdaptiveCellKey>{opposite.key}));
    const auto capabilities = kernel->contact_capabilities(opposite);
    EXPECT_EQ(capabilities.normalization, AMR::AdaptiveContactNormalization::PhysicalIdentity);
    EXPECT_FALSE(capabilities.split_union_preserving);
    EXPECT_FALSE(capabilities.coarsen_union_preserving);
    UnqualifiedKernel unsupported;
    EXPECT_EQ(unsupported.contact_capabilities(opposite).normalization, AMR::AdaptiveContactNormalization::Unsupported);
    EXPECT_THROW((void) unsupported.closed_contact_images(opposite), std::invalid_argument);
}

TEST(ClosedContactIndex, CanonicalBoundsSurviveRebaseAndRestoreCallerEnvironment)
{
    struct Environment
    {
        std::fenv_t saved{};
        Environment() { (void) std::fegetenv(&saved); }
        ~Environment() { (void) std::fesetenv(&saved); }
    } restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    const auto source = Cell::box({0, 0, 0}, {1, 1, 1});
    const auto rebased = source->clip({{1, 0, 0}, {0x1p65, 0, 0}, 0}, false).inside;
    ASSERT_TRUE(rebased);
    const std::array<std::array<long double, 2>, 3> expected{{{0, 1}, {0, 1}, {0, 1}}};
    for (const auto rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        ASSERT_EQ(std::fesetround(rounding), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
        const auto exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        EXPECT_EQ(source->closed_contact_bounds(), expected);
        EXPECT_EQ(rebased->closed_contact_bounds(), expected);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    if (std::numeric_limits<long double>::digits > 60)
    {
        const long double endpoint = 1.L + 0x1p-60L;
        const auto wide =
            Cell::tetrahedron({Point({0, 0, 0}), Point({endpoint, 0, 0}), Point({0, 1, 0}), Point({0, 0, 1})});
        EXPECT_EQ(wide->closed_contact_bounds()[0][1], endpoint);
        EXPECT_GT(wide->closed_contact_bounds()[0][1], 1.L);
        EXPECT_EQ(double(endpoint), 1.);
    }
}

TEST(ClosedContactIndex, LogicalMetadataRequiresValidationAndAnExactEmbeddingPromise)
{
    auto source = entry(1, {0, 0, 0}, {4, 4, 4});
    source.images[0].logical = AMR::AdaptiveContactLogicalBox{{'f'}, {0, 0, 0}, {1, 1, 1}, {4, 4, 4}, true};
    D::ClosedContactIndex index(ample);
    index.insert(source);
    auto candidate = source.images[0];
    candidate.logical->lower = {2, 2, 2};
    candidate.logical->upper = {3, 3, 3};
    EXPECT_TRUE(query(index, candidate).empty());
    candidate.logical->exact_embedding = false;
    EXPECT_EQ(query(index, candidate), (std::vector<AMR::AdaptiveCellKey>{source.key}));
    candidate.logical->exact_embedding = true;
    candidate.logical->frame = {'g'};
    EXPECT_EQ(query(index, candidate), (std::vector<AMR::AdaptiveCellKey>{source.key}));
    candidate.logical->frame.clear();
    EXPECT_THROW((void) query(index, candidate), std::invalid_argument);
    candidate = source.images[0];
    candidate.logical->period[0] = 0;
    EXPECT_THROW((void) query(index, candidate), std::invalid_argument);
    candidate = source.images[0];
    candidate.logical->lower[0] = 2;
    EXPECT_THROW((void) query(index, candidate), std::invalid_argument);
    candidate = source.images[0];
    candidate.bounds.upper[0] = std::numeric_limits<long double>::quiet_NaN();
    EXPECT_THROW((void) query(index, candidate), std::invalid_argument);
    EXPECT_EQ(query(index, source.images[0]), (std::vector<AMR::AdaptiveCellKey>{source.key}));
}

TEST(ClosedContactIndex, ExtremeFiniteRangesPreserveCallerRoundingAndExceptionFlags)
{
    struct Environment
    {
        std::fenv_t saved{};
        Environment() { (void) std::fegetenv(&saved); }
        ~Environment() { (void) std::fesetenv(&saved); }
    } restore;
    const auto largest = std::numeric_limits<long double>::max();
    const auto first = entry(1, {-largest, 0, 0}, {0, 1, 1});
    const auto second = entry(2, {0, 0, 0}, {largest, 1, 1});
    const auto third = entry(3, {-largest, 0, 0}, {largest, 1, 1});
    auto replacement = entry(2, {-largest, -largest, -largest}, {largest, largest, largest});
    replacement.revision = 2;
    for (const auto rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        SCOPED_TRACE(rounding);
        D::ClosedContactIndex index(ample);
        ASSERT_EQ(std::fesetround(rounding), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
        const auto exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        const auto preserved = [&]
        {
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
        };
        index.insert(first);
        index.insert(second);
        // The third insertion evaluates a subtree heuristic spanning both
        // finite extrema. Its mathematical width exceeds long-double range.
        index.insert(third);
        preserved();
        index.replace(replacement);
        preserved();
        EXPECT_EQ(
            query(index, image({0, 0, 0}, {0, 0, 0})), (std::vector<AMR::AdaptiveCellKey>{key(1), key(2), key(3)}));
        preserved();
        EXPECT_EQ(index.ownership_covers(1).size(), 3u);
        preserved();
        index.erase(first.key);
        preserved();
    }
}

TEST(ClosedContactIndex, ExactLiveCapAndFailedReplacementPreserveExistingEntries)
{
    const auto original = entry(7, {0, 0, 0}, {1, 1, 1});
    auto replacement = original;
    replacement.revision = 2;
    replacement.images.clear();
    for (unsigned i = 0; i < 8; ++i)
        replacement.images.push_back(image({10.L + i, 0, 0}, {11.L + i, 1, 1}));
    D::ClosedContactIndex measured(ample);
    measured.insert(original);
    const auto insert_cap = measured.statistics().peak_bytes;
    D::ClosedContactIndex exact_insert(insert_cap);
    EXPECT_NO_THROW(exact_insert.insert(original));
    D::ClosedContactIndex short_insert(insert_cap - 1);
    EXPECT_THROW(short_insert.insert(original), std::invalid_argument);
    EXPECT_EQ(short_insert.find(original.key), nullptr);
    EXPECT_EQ(short_insert.statistics().entries, 0u);
    EXPECT_THROW((void) D::ClosedContactIndex(1), std::invalid_argument);
    measured.replace(replacement);
    const auto replacement_cap = measured.statistics().peak_bytes;
    D::ClosedContactIndex exact(replacement_cap), rejected(replacement_cap - 1);
    exact.insert(original);
    rejected.insert(original);
    const auto* retained = rejected.find(original.key);
    ASSERT_NE(retained, nullptr);
    EXPECT_NO_THROW(exact.replace(replacement));
    EXPECT_THROW(rejected.replace(replacement), std::invalid_argument);
    EXPECT_EQ(rejected.find(original.key), retained);
    EXPECT_EQ(retained->revision, 1u);
    EXPECT_EQ(retained->images.size(), 1u);
    EXPECT_EQ(retained->images[0].bounds.lower, (Wide{0, 0, 0}));
    EXPECT_EQ(query(rejected, original.images[0]), (std::vector<AMR::AdaptiveCellKey>{original.key}));
    const auto live = rejected.statistics().retained_bytes;
    const auto repeats = replacement_cap / sizeof(const D::ClosedContactEntry*) + 1;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat)
        ASSERT_EQ(query(rejected, original.images[0]), (std::vector<AMR::AdaptiveCellKey>{original.key}));
    EXPECT_EQ(rejected.statistics().retained_bytes, live);
    EXPECT_GT(rejected.statistics().admitted_bytes, replacement_cap);
    EXPECT_EQ(original.revision, 1u);
    EXPECT_EQ(original.images[0].bounds.upper, (Wide{1, 1, 1}));
    EXPECT_EQ(replacement.revision, 2u);
    EXPECT_EQ(replacement.images.size(), 8u);
    EXPECT_THROW(exact.insert(replacement), std::invalid_argument);
    EXPECT_THROW(exact.replace(original), std::invalid_argument);
}

TEST(ClosedContactIndex, IncrementalErasureReplacementAndOwnershipCoversKeepAllImages)
{
    D::ClosedContactIndex index(ample);
    for (unsigned step = 0; step < 32; ++step)
    {
        const auto id = (step * 17) % 32;
        index.insert(entry(id, {2.L * id, 0, 0}, {2.L * id + 1, 1, 1}, id % 3 ? 'd' : 'x', id % 2));
    }
    const auto covers = index.ownership_covers(1);
    ASSERT_EQ(covers.size(), 32u);
    for (unsigned id = 0; id < 32; ++id)
    {
        const char domain = id % 3 ? 'd' : 'x';
        const auto point = image({2.L * id, 0, 0}, {2.L * id, 0, 0});
        EXPECT_EQ(query(index, point, domain), (std::vector<AMR::AdaptiveCellKey>{key(id)}));
        EXPECT_EQ(std::count_if(covers.begin(), covers.end(),
                      [&](const auto& cover)
                      {
                          return cover.domain == AMR::AdaptiveBytes{domain} && cover.owner == int(id % 2) &&
                                 D::closed_contact_overlap(point.bounds, cover.bounds);
                      }),
            1);
        if (id % 2 == 0)
            index.erase(key(id));
        else
        {
            auto moved = entry(id, {1000.L + 2 * id, 0, 0}, {1001.L + 2 * id, 1, 1}, domain, id % 2);
            moved.revision = 2;
            index.replace(std::move(moved));
        }
        EXPECT_TRUE(query(index, point, domain).empty());
    }
    EXPECT_EQ(index.statistics().entries, 16u);
    for (unsigned id = 1; id < 32; id += 2)
        EXPECT_EQ(query(index, image({1000.L + 2 * id, 0, 0}, {1000.L + 2 * id, 0, 0}), id % 3 ? 'd' : 'x'),
            (std::vector<AMR::AdaptiveCellKey>{key(id)}));
    EXPECT_THROW((void) index.ownership_covers(0), std::invalid_argument);
}
TEST(ClosedContactIndex, ForkOwnsIndependentEntriesAndExactAggregateBounds)
{
    D::ClosedContactIndex source(ample);
    EXPECT_FALSE(source.aggregate_bounds().has_value());
    const auto first = entry(1, {-3, -2, 0}, {-1, 1, 2}, 'd', 0);
    const auto second = entry(2, {4, 3, -5}, {7, 8, -1}, 'x', 1);
    source.insert(first);
    source.insert(second);
    const auto wide = image({-10, -10, -10}, {10, 10, 10});
    const auto source_d = query(source, wide, 'd');
    const auto source_x = query(source, wide, 'x');
    EXPECT_EQ(source_d, (std::vector<AMR::AdaptiveCellKey>{first.key}));
    EXPECT_EQ(source_x, (std::vector<AMR::AdaptiveCellKey>{second.key}));
    const auto source_bounds = source.aggregate_bounds();
    ASSERT_TRUE(source_bounds);
    EXPECT_EQ(source_bounds->lower, (Wide{-3, -2, -5}));
    EXPECT_EQ(source_bounds->upper, (Wide{7, 8, 2}));

    auto candidate = source.fork(ample);
    const auto fork_peak = candidate.statistics().peak_bytes;
    const auto fork_retained = candidate.statistics().retained_bytes;
    EXPECT_EQ(candidate.statistics().entries, 2u);
    EXPECT_EQ(query(candidate, wide, 'd'), source_d);
    EXPECT_EQ(query(candidate, wide, 'x'), source_x);
    ASSERT_GT(fork_retained, 0u);
    EXPECT_THROW((void) source.fork(fork_retained - 1), std::invalid_argument);
    auto exact_peak = source.fork(fork_peak);
    EXPECT_EQ(exact_peak.statistics().entries, 2u);

    candidate.erase(first.key);
    auto replacement = entry(2, {20, 30, 40}, {21, 31, 41}, 'x', 1);
    replacement.revision = 2;
    candidate.replace(replacement);
    const auto candidate_bounds = candidate.aggregate_bounds();
    ASSERT_TRUE(candidate_bounds);
    EXPECT_EQ(candidate_bounds->lower, (Wide{20, 30, 40}));
    EXPECT_EQ(candidate_bounds->upper, (Wide{21, 31, 41}));
    EXPECT_TRUE(query(candidate, wide, 'd').empty());
    EXPECT_TRUE(query(candidate, wide, 'x').empty());
    EXPECT_EQ(query(candidate, image({20, 30, 40}, {20, 30, 40}), 'x'),
        (std::vector<AMR::AdaptiveCellKey>{second.key}));
    ASSERT_NE(candidate.find(second.key), nullptr);
    EXPECT_EQ(candidate.find(second.key)->revision, 2u);

    ASSERT_NE(source.find(first.key), nullptr);
    ASSERT_NE(source.find(second.key), nullptr);
    EXPECT_EQ(source.find(second.key)->revision, 1u);
    EXPECT_EQ(query(source, wide, 'd'), source_d);
    EXPECT_EQ(query(source, wide, 'x'), source_x);
    const auto unchanged = source.aggregate_bounds();
    ASSERT_TRUE(unchanged);
    EXPECT_EQ(unchanged->lower, (Wide{-3, -2, -5}));
    EXPECT_EQ(unchanged->upper, (Wide{7, 8, 2}));
}

TEST(ClosedContactIndex, RebindRejectsBelowRetentionAndLaterGrowthPreservesSource)
{
    D::ClosedContactIndex source(ample);
    const auto first = entry(1, {-2, -1, 0}, {0, 1, 2}, 'd');
    const auto second = entry(2, {2, 3, 4}, {4, 5, 6}, 'x');
    source.insert(first);
    source.insert(second);
    auto candidate = source.fork(ample);
    const auto retained = candidate.statistics().retained_bytes;
    ASSERT_GT(retained, 0u);
    const auto before = candidate.aggregate_bounds();
    ASSERT_TRUE(before);
    EXPECT_THROW(candidate.rebind_budget(retained - 1), std::invalid_argument);
    EXPECT_EQ(candidate.statistics().retained_bytes, retained);
    const auto after = candidate.aggregate_bounds();
    ASSERT_TRUE(after);
    EXPECT_EQ(after->lower, before->lower);
    EXPECT_EQ(after->upper, before->upper);
    EXPECT_EQ(query(candidate, image({-2, -1, 0}, {-2, -1, 0}), 'd'),
        (std::vector<AMR::AdaptiveCellKey>{first.key}));
    EXPECT_NO_THROW(candidate.rebind_budget(ample));
    const auto third = entry(3, {-9, 10, 11}, {-8, 12, 13}, 'd');
    EXPECT_NO_THROW(candidate.insert(third));
    const auto expanded = candidate.aggregate_bounds();
    ASSERT_TRUE(expanded);
    EXPECT_EQ(expanded->lower, (Wide{-9, -1, 0}));
    EXPECT_EQ(expanded->upper, (Wide{4, 12, 13}));
    EXPECT_EQ(query(candidate, image({-9, 10, 11}, {-9, 10, 11}), 'd'),
        (std::vector<AMR::AdaptiveCellKey>{third.key}));
    EXPECT_EQ(source.find(third.key), nullptr);
    EXPECT_EQ(query(source, image({-2, -1, 0}, {-2, -1, 0}), 'd'),
        (std::vector<AMR::AdaptiveCellKey>{first.key}));
    EXPECT_EQ(query(source, image({2, 3, 4}, {2, 3, 4}), 'x'),
        (std::vector<AMR::AdaptiveCellKey>{second.key}));
}

TEST(ClosedContactIndex, HomogeneousFrontierUsesBoundedSubtreesWithoutOmittingImages)
{
    D::ClosedContactIndex index(ample);
    EXPECT_TRUE(index.ownership_frontier_covers().empty());
    EXPECT_THROW((void) index.ownership_frontier_covers(0), std::invalid_argument);
    for (unsigned i = 0; i < 64; ++i)
        index.insert(entry(i, {3.L * i, 0, 0}, {3.L * i + 1, 1, 1}, 'd', 0));
    const auto covers = index.ownership_frontier_covers(8);
    EXPECT_EQ(covers.size(), 8u);
    std::size_t total = 0;
    for (const auto& cover : covers)
    {
        EXPECT_EQ(cover.domain, (AMR::AdaptiveBytes{'d'}));
        EXPECT_EQ(cover.owner, 0);
        total += cover.images;
    }
    EXPECT_EQ(total, 64u);
    for (unsigned i = 0; i < 64; ++i)
    {
        const auto expected = image({3.L * i, 0, 0}, {3.L * i + 1, 1, 1});
        EXPECT_TRUE(std::any_of(covers.begin(), covers.end(), [&](const auto& cover) {
            for (std::size_t axis = 0; axis < 3; ++axis)
                if (cover.bounds.lower[axis] > expected.bounds.lower[axis] ||
                    cover.bounds.upper[axis] < expected.bounds.upper[axis]) return false;
            return true;
        }));
    }
    EXPECT_THROW((void) index.ownership_frontier_covers(8, 1), std::invalid_argument);
    EXPECT_EQ(query(index, image({0, 0, 0}, {0, 0, 0})), (std::vector<AMR::AdaptiveCellKey>{key(0)}));
}

TEST(ClosedContactIndex, MixedFrontierKeepsOwnerDomainCountsAndWholeImageBounds)
{
    D::ClosedContactIndex index(ample);
    constexpr unsigned count = 24;
    for (unsigned i = 0; i < count; ++i)
        index.insert(entry(i, {2.L * i, -1, 0}, {2.L * i + 1, 1, 1}, i % 2 ? 'x' : 'd', i % 3));
    const auto covers = index.ownership_frontier_covers(1);
    std::array<std::array<std::size_t, 3>, 2> totals{};
    for (const auto& cover : covers)
    {
        ASSERT_TRUE(cover.domain == AMR::AdaptiveBytes{'d'} || cover.domain == AMR::AdaptiveBytes{'x'});
        ASSERT_GE(cover.owner, 0);
        ASSERT_LT(cover.owner, 3);
        totals[cover.domain == AMR::AdaptiveBytes{'x'}][std::size_t(cover.owner)] += cover.images;
    }
    for (const auto& by_owner : totals)
        for (const auto images : by_owner) EXPECT_EQ(images, 4u);
    for (unsigned i = 0; i < count; ++i)
    {
        const auto expected = image({2.L * i, -1, 0}, {2.L * i + 1, 1, 1});
        const AMR::AdaptiveBytes domain{i % 2 ? 'x' : 'd'};
        EXPECT_TRUE(std::any_of(covers.begin(), covers.end(), [&](const auto& cover) {
            if (cover.domain != domain || cover.owner != int(i % 3)) return false;
            for (std::size_t axis = 0; axis < 3; ++axis)
                if (cover.bounds.lower[axis] > expected.bounds.lower[axis] ||
                    cover.bounds.upper[axis] < expected.bounds.upper[axis]) return false;
            return true;
        }));
    }
    const auto total_bounds = index.aggregate_bounds();
    ASSERT_TRUE(total_bounds);
    EXPECT_EQ(total_bounds->lower, (Wide{0, -1, 0}));
    EXPECT_EQ(total_bounds->upper, (Wide{47, 1, 1}));
}

} // namespace
