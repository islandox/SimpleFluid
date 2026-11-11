/** Source-only sparse binding order and captured metric regressions. */
#include "adaptive_storage_test_helpers.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <bit>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <set>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace F = SimpleFluid::test::adaptive_storage;
namespace P = SimpleFluid::PLIC;
using SimpleFluid::ArithmeticMode;
using H = A::DistributedAdaptiveHierarchy;
using Access = P::details::NativeCellGeometrySourceAccess;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t ample = 64 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
auto make_hierarchy(std::vector<A::AdaptiveLeaf> owned, std::shared_ptr<const A::AdaptiveProviderKernel> kernel)
{
    return std::make_shared<const H>(
        std::move(owned), std::vector<A::AdaptiveFamily>{}, std::move(kernel), Tpetra::getDefaultComm());
}
auto rows(const H& source)
{
    std::vector<const A::AdaptiveLeaf*> result;
    for (const auto& leaf : source.owned_leaves())
        result.push_back(&leaf);
    return result;
}
auto acquire(const H& source, std::span<const A::AdaptiveLeaf* const> requested,
    A::GeometrySnapshotPurpose purpose = A::GeometrySnapshotPurpose::State)
{
    std::size_t used = 0;
    return source.geometry_snapshot(requested, ordinary, purpose, used, ample);
}
void same_metric(const A::AdaptiveCellMetrics& a, const A::AdaptiveCellMetrics& b)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(a.volume), std::bit_cast<std::uint64_t>(b.volume));
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(a.centroid[axis]), std::bit_cast<std::uint64_t>(b.centroid[axis]));
}
} // namespace

TEST(AdaptiveGeometryDemandSnapshot, AllProvidersKeepRequestedPrefixOrderAndCapturedNativeMetrics)
{
    for (const auto provider : F::providers)
        for (const auto components : {4u, 8u})
        {
            SCOPED_TRACE(F::name(provider));
            SCOPED_TRACE(components);
            auto fixture = F::make_disconnected(provider, components);
            const auto reference_source = make_hierarchy(fixture.owned, fixture.kernel);
            auto source = make_hierarchy(std::move(fixture.owned), fixture.kernel);
            const auto reference_rows = rows(*reference_source);
            const auto expected = acquire(*reference_source, reference_rows);
            const auto selected = rows(*source);
            std::map<A::AdaptiveCellKey, std::size_t> reference;
            for (std::size_t slot = 0; slot < reference_rows.size(); ++slot)
                reference.emplace(reference_rows[slot]->key, slot);
            std::vector<const A::AdaptiveLeaf*> woven;
            for (std::size_t begin = 0, end = selected.size(); begin < end;)
            {
                woven.push_back(selected[begin++]);
                if (begin < end)
                    woven.push_back(selected[--end]);
            }
            std::shared_ptr<const A::AdaptiveGeometrySnapshot> current, first;
            P::ConvexGeometryPtr<ordinary> first_geometry;
            std::vector<A::AdaptiveCellKey> order;
            std::size_t offset = 0, batch = 0;
            const std::array<std::size_t, 3> widths{1, 2, 4};
            while (offset < woven.size())
            {
                const auto count = std::min(widths[batch++ % widths.size()], woven.size() - offset);
                std::vector<const A::AdaptiveLeaf*> requested(woven.begin() + offset, woven.begin() + offset + count);
                if (current)
                    requested.insert(requested.begin(), woven.front());
                current = acquire(*source, requested);
                if (!first)
                {
                    first = current;
                    first_geometry =
                        Access::existing_geometry(P::CellGeometryView<ordinary>(first->ordinary_catalog(), 0));
                }
                for (const auto* leaf : requested)
                    std::erase(order, leaf->key);
                std::vector<A::AdaptiveCellKey> prefix;
                for (const auto* leaf : requested)
                    prefix.push_back(leaf->key);
                order.insert(order.begin(), prefix.begin(), prefix.end());
                ASSERT_EQ(current->ordinary_catalog()->size(), order.size());
                for (std::size_t slot = 0; slot < order.size(); ++slot)
                {
                    const auto* leaf = source->leaf_layout()->find(order[slot]);
                    ASSERT_NE(leaf, nullptr);
                    EXPECT_EQ(current->find(*leaf), slot);
                    same_metric(current->native_metrics()[slot], expected->native_metrics()[reference.at(order[slot])]);
                }
                offset += count;
            }
            const auto retained_first = current->find(*woven.front());
            ASSERT_TRUE(retained_first);
            EXPECT_EQ(
                Access::existing_geometry(P::CellGeometryView<ordinary>(current->ordinary_catalog(), *retained_first)),
                first_geometry);
            const auto first_key = woven.front()->key;
            std::weak_ptr<const A::AdaptiveLeafLayout> layout = source->leaf_layout();
            source.reset();
            EXPECT_TRUE(layout.expired());
            same_metric(first->native_metrics()[0], expected->native_metrics()[reference.at(first_key)]);
        }
}

TEST(AdaptiveGeometryDemandSnapshot, SamePurposeRawExpansionPreservesOldViewsAndOnlySelectedBindings)
{
    auto fixture = F::make_disconnected(F::Provider::Static, 8);
    const auto source = make_hierarchy(std::move(fixture.owned), std::move(fixture.kernel));
    const auto selected = rows(*source);
    const auto purpose = A::GeometrySnapshotPurpose::OriginalSources;
    const auto first = acquire(*source, std::span(selected).first(1), purpose);
    const auto old_geometry = Access::existing_geometry(P::CellGeometryView<ordinary>(first->ordinary_catalog(), 0));
    const auto second = acquire(*source, std::span(selected).subspan(1, 2), purpose);
    const auto third = acquire(*source, std::span(selected).subspan(3, 4), purpose);
    EXPECT_EQ(first->ordinary_catalog()->size(), 1u);
    EXPECT_EQ(second->ordinary_catalog()->size(), 3u);
    EXPECT_EQ(third->ordinary_catalog()->size(), 7u);
    const auto old = third->find(*selected.front());
    ASSERT_TRUE(old);
    EXPECT_EQ(Access::existing_geometry(P::CellGeometryView<ordinary>(third->ordinary_catalog(), *old)), old_geometry);
    const std::array duplicate{selected.front(), selected.front()};
    EXPECT_THROW((void) acquire(*source, duplicate, purpose), std::invalid_argument);
    EXPECT_EQ(first->ordinary_catalog()->size(), 1u);
}

TEST(AdaptiveGeometryDemandSnapshot, WovenBindingOrderVisitsEachAuthoritativeDataPageOnce)
{
    auto fixture = F::make_disconnected(F::Provider::Polyhedral, 66);
    const auto source = make_hierarchy(std::move(fixture.owned), std::move(fixture.kernel));
    auto selected = rows(*source);
    ASSERT_GT(selected.size(), 64u);
    std::vector<const A::AdaptiveLeaf*> woven;
    for (std::size_t begin = 0, end = selected.size(); begin < end;)
    {
        woven.push_back(selected[begin++]);
        if (begin < end)
            woven.push_back(selected[--end]);
    }
    const auto snapshot = acquire(*source, woven);
    const auto expected = std::count_if(selected.begin(), selected.end(), [](const auto* leaf) { return bool(leaf->geometry); });
    ASSERT_GT(expected, 64);
    std::size_t emitted = 0;
    snapshot->visit_authoritative_geometries([&](const auto&) { ++emitted; });
    EXPECT_EQ(emitted, expected);
}

TEST(AdaptiveGeometryDemandSnapshot, ColdSameSourceSharesOneDescriptorAcrossBindingPages)
{
    A::AMRGrid config;
    config.block_cells = {8, 4, 4};
    const A::CartesianAMR mesh(config);
    std::vector<std::uint64_t> ids(mesh.geometry()->num_cells());
    std::iota(ids.begin(), ids.end(), 0);
    auto fixture = mesh.detach_owned(ids, {'c', 'o', 'l', 'd'});
    const auto source = make_hierarchy(std::move(fixture.owned), std::move(fixture.kernel));
    const auto selected = rows(*source);
    ASSERT_GT(selected.size(), 64u);
    const auto snapshot = acquire(*source, selected);
    std::set<const void*> descriptors;
    std::size_t edges = 0;
    A::adaptive_detail::GeometrySnapshotAccess::visit_binding_sources(
        *snapshot, [&](const void* source) { ++edges; descriptors.insert(source); });
    EXPECT_GE(edges, 2u);
    EXPECT_EQ(descriptors.size(), 1u);
}

TEST(AdaptiveGeometryDemandSnapshot, SparseExpansionAfterCrossLayoutSuccessorKeepsPartiallyReusedPages)
{
    // A cross-layout successor reuses only the rows that still match its
    // predecessor, so a reused binding page holds physical rows outside the
    // successor's selection. A following same-layout sparse expansion must
    // repack only rows that page still binds, never a stale physical row.
    A::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    const A::CartesianAMR mesh(grid);
    std::vector<std::uint64_t> indices(mesh.geometry()->num_cells());
    std::iota(indices.begin(), indices.end(), 0);
    auto detached = mesh.detach_owned(indices, {'s', 'u', 'c', 'c'});
    const auto source = std::make_shared<const H>(std::move(detached.owned), std::move(detached.families),
        std::move(detached.kernel), Tpetra::getDefaultComm());
    const auto selected = rows(*source);
    ASSERT_EQ(selected.size(), 16u);
    const auto accepted = acquire(*source, selected);
    const A::AdaptiveMark mark{selected.front()->key, A::AdaptiveSplit::Isotropic};
    const auto candidate = source->apply(source->plan_adaptation(std::array{mark}));
    const auto candidate_rows = rows(*candidate);
    ASSERT_GT(candidate_rows.size(), selected.size());
    std::vector<const A::AdaptiveLeaf*> reused;
    for (const auto* leaf : candidate_rows)
        if (accepted->find(*leaf) && candidate->leaf_layout()->find(leaf->key))
            reused.push_back(leaf);
    ASSERT_EQ(reused.size(), 8u);
    const auto successor = acquire(*candidate, reused);
    ASSERT_EQ(successor->ordinary_catalog()->size(), reused.size());
    auto expanded = reused;
    for (const auto* leaf : candidate_rows)
        if (!successor->find(*leaf))
        {
            expanded.push_back(leaf);
            break;
        }
    ASSERT_EQ(expanded.size(), reused.size() + 1);
    const auto demand = acquire(*candidate, expanded);
    ASSERT_EQ(demand->ordinary_catalog()->size(), expanded.size());
    for (std::size_t slot = 0; slot < expanded.size(); ++slot)
        EXPECT_EQ(demand->find(*expanded[slot]), slot);
    EXPECT_TRUE(demand->matches_layout(candidate->leaf_layout()));
    EXPECT_TRUE(demand->find(*reused.front()));
}
