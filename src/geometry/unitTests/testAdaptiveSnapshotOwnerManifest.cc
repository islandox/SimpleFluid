/** Snapshot manifest and independent physical-owner regressions.
 * Expected ownership comes from full retain/export, never the incremental editor.
 */
#include "geometry/PLIC/details/CellCatalogDemandAccess.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityRetention.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/unitTests/adaptive_storage_test_helpers.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>
namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace P = SimpleFluid::PLIC;
namespace G = P::details::geometry_demand;
using S = A::AdaptiveGeometrySnapshot;
using H = A::DistributedAdaptiveHierarchy;
using SimpleFluid::ArithmeticMode;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t ample = 128 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Key = std::tuple<unsigned, unsigned, const void*>;
using Records = std::map<Key, G::OwnerRecord>;
G::Admission charge(std::size_t& used)
{
    return {&used, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; }};
}
Records records(const G::GeometryOwnerManifest& manifest)
{
    Records result;
    manifest.visit([&](const auto& row)
        { EXPECT_TRUE(result.emplace(Key{row.key.channel, row.key.kind, row.key.address}, row).second); });
    return result;
}
std::array<G::GeometryOwnerManifest, 2> capture(const S& snapshot)
{
    A::adaptive_detail::Budget budget{ample};
    std::array<G::GeometryOwnerManifest, 2> result;
    A::adaptive_detail::GeometrySnapshotAccess::owner_manifests(snapshot, budget, result[0], result[1]);
    EXPECT_EQ(budget.used, 0u);
    return result;
}
void agrees_with_independent_census(const std::shared_ptr<const S>& snapshot)
{
    const auto manifests = capture(*snapshot);
    P::GeometryAllocationLedger ordinary_ledger, bounded_ledger;
    std::size_t used = 0;
    snapshot->retain(ordinary_ledger, bounded_ledger, used, ample);
    const auto ordinary_full = P::details::GeometryOwnerImportAccess::export_manifest(ordinary_ledger, charge(used));
    const auto bounded_full =
        P::details::GeometryOwnerImportAccess::export_manifest(bounded_ledger, charge(used), nullptr, 1);
    std::array<Records, 2> expected{records(ordinary_full), records(bounded_full)};
    const auto external = P::details::GeometryOwnerGraphEditor::external_kind();
    EXPECT_EQ(expected[0].erase(Key{0, external, snapshot.get()}), 1u);
    struct Credit
    {
        std::shared_ptr<const void> owner;
        std::size_t payload, admitted;
    };
    std::array<std::map<const void*, Credit>, 2> credits;
    for (unsigned channel = 0; channel < 2; ++channel)
        P::details::GeometryAllocationOwnerAccess::visit_metadata_credits(channel ? bounded_ledger : ordinary_ledger,
            [&](const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted, bool imported)
            {
                EXPECT_FALSE(imported);
                EXPECT_TRUE(credits[channel].emplace(owner.get(), Credit{owner, payload, admitted}).second);
            });
    struct Context
    {
        std::array<Records, 2>& rows;
        decltype(credits)& metadata_credits;
        unsigned external;
    } context{expected, credits, external};
    A::adaptive_detail::GeometrySnapshotAccess::visit_owner_manifest_metadata(*snapshot,
        {&context, [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
            {
                auto& value = *static_cast<Context*>(raw);
                unsigned found_count = 0;
                for (unsigned channel = 0; channel < 2; ++channel)
                {
                    const auto found = value.rows[channel].find(Key{channel, value.external, owner.get()});
                    if (found == value.rows[channel].end())
                        continue;
                    ++found_count;
                    EXPECT_EQ(found->second.bytes, payload);
                    EXPECT_FALSE(found->second.owner.owner_before(owner));
                    EXPECT_FALSE(owner.owner_before(found->second.owner));
                    const auto credit = value.metadata_credits[channel].find(owner.get());
                    EXPECT_NE(credit, value.metadata_credits[channel].end());
                    if (credit != value.metadata_credits[channel].end())
                    {
                        EXPECT_EQ(credit->second.payload, payload);
                        EXPECT_EQ(credit->second.admitted, admitted);
                        EXPECT_EQ(admitted - payload, G::control_bytes);
                        EXPECT_FALSE(credit->second.owner.owner_before(owner));
                        EXPECT_FALSE(owner.owner_before(credit->second.owner));
                        value.metadata_credits[channel].erase(credit);
                    }
                    value.rows[channel].erase(found);
                }
                EXPECT_EQ(found_count, 1u);
                return true;
            }});
    EXPECT_TRUE(credits[0].empty());
    EXPECT_TRUE(credits[1].empty());
    const auto actual = std::array{records(manifests[0]), records(manifests[1])};
    for (unsigned channel = 0; channel < 2; ++channel)
    {
        ASSERT_EQ(actual[channel].size(), expected[channel].size());
        for (const auto& [key, row] : expected[channel])
        {
            const auto found = actual[channel].find(key);
            ASSERT_NE(found, actual[channel].end());
            EXPECT_EQ(found->second.bytes, row.bytes);
            EXPECT_FALSE(found->second.owner.owner_before(row.owner));
            EXPECT_FALSE(row.owner.owner_before(found->second.owner));
        }
        EXPECT_EQ(manifests[channel].payload_bytes(),
            std::accumulate(expected[channel].begin(), expected[channel].end(), std::size_t(0),
                [](auto bytes, const auto& row) { return bytes + row.second.bytes; }));
    }
}
std::set<const void*> index_buffers(const G::GeometryOwnerManifest& manifest)
{
    std::set<const void*> owners, payload;
    manifest.visit_metadata([&](const void* address, std::size_t) { owners.insert(address); });
    manifest.visit_payload([&](const void* address, std::size_t) { payload.insert(address); });
    for (const auto* owner : owners)
        payload.erase(owner);
    return payload;
}
std::shared_ptr<const H> cartesian_source(bool many_recipes = false)
{
    A::AMRGrid grid;
    grid.block_cells = {32, 8, 2};
    if (many_recipes)
    {
        // Distinct real block sources make the physical owner index span
        // several leaves. Cell counts alone do not establish owner coverage.
        grid.root_blocks = {256, 1, 1};
        grid.block_cells = {2, 2, 2};
    }
    A::CartesianAMR source(grid);
    std::vector<std::uint64_t> ids(source.geometry()->num_cells());
    std::iota(ids.begin(), ids.end(), 0);
    auto detached = source.detach_owned(ids, {'m', 'a', 'n', 'i', 'f', 'e', 's', 't'});
    return std::make_shared<const H>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), Tpetra::getDefaultComm());
}
struct PhysicalGraph
{
    std::map<const void*, std::size_t> bytes, incoming;
    std::map<const void*, A::AdaptiveCellKey> binding_last;
    std::map<const void*, std::vector<const void*>> children;
    std::vector<const void*> roots;
    std::map<const void*, std::map<const void*, std::size_t>> binding_buffers;
};
PhysicalGraph physical_graph(const S& snapshot)
{
    PhysicalGraph graph;
    A::adaptive_detail::GeometrySnapshotAccess::visit_cartesian_payload_edges(
        snapshot,
        [&](const void* parent, const void* owner, std::size_t bytes)
        {
            if (const auto found = graph.bytes.find(owner); found != graph.bytes.end())
                EXPECT_EQ(found->second, bytes);
            graph.bytes[owner] = bytes;
            ++graph.incoming[owner];
            if (parent)
                graph.children[parent].push_back(owner);
            else
                graph.roots.push_back(owner);
        },
        [&](const void* page, const void* buffer, std::size_t bytes)
        { graph.binding_buffers[page].emplace(buffer, bytes); },
        [&](const void* page, const A::AdaptiveCellKey& key)
        {
            graph.binding_last.emplace(page, key);
            graph.binding_buffers[page];
        });
    const auto manifest = capture(snapshot)[0];
    EXPECT_EQ(manifest.owner_count(), graph.bytes.size());
    manifest.visit(
        [&](const G::OwnerRecord& row)
        {
            ASSERT_TRUE(graph.bytes.contains(row.owner.get()));
            EXPECT_EQ(row.bytes, graph.bytes.at(row.owner.get()));
            EXPECT_EQ(row.references, graph.incoming.at(row.owner.get()));
        });
    return graph;
}
std::size_t index_height(const G::GeometryOwnerManifest& manifest)
{
    struct Context
    {
        std::map<const void*, std::size_t> depths;
        std::size_t height = 0;
    } context;
    manifest.visit_borrowed_dependencies({&context,
                                             [](void* raw, const void* owner, const void* parent, std::size_t)
                                             {
                                                 auto& value = *static_cast<Context*>(raw);
                                                 const auto depth = parent ? value.depths.at(parent) + 1 : 0;
                                                 value.depths.emplace(owner, depth);
                                                 value.height = std::max(value.height, depth + 1);
                                                 return true;
                                             }},
        {nullptr, [](void*, const void*, const G::OwnerRecord&) {}});
    return context.height;
}
void expect_exact_frontier(const PhysicalGraph& before, const PhysicalGraph& after,
    const A::adaptive_detail::GeometrySnapshotOwnerStatistics& actual, const G::GeometryOwnerManifest& old_manifest,
    const G::GeometryOwnerManifest& new_manifest)
{
    std::size_t first = 0, last = 0, changed = 0;
    std::vector<const void*> visited = before.roots;
    visited.insert(visited.end(), after.roots.begin(), after.roots.end());
    std::set<const void*> keys;
    for (const auto& [owner, bytes] : before.bytes)
    {
        keys.insert(owner);
        if (!after.bytes.contains(owner))
        {
            ++last;
            if (const auto edges = before.children.find(owner); edges != before.children.end())
                visited.insert(visited.end(), edges->second.begin(), edges->second.end());
        }
    }
    for (const auto& [owner, bytes] : after.bytes)
    {
        keys.insert(owner);
        if (!before.bytes.contains(owner))
        {
            ++first;
            if (const auto edges = after.children.find(owner); edges != after.children.end())
                visited.insert(visited.end(), edges->second.begin(), edges->second.end());
        }
    }
    for (const auto* owner : keys)
    {
        const auto old = before.incoming.find(owner), now = after.incoming.find(owner);
        changed += (old == before.incoming.end() ? 0 : old->second) != (now == after.incoming.end() ? 0 : now->second);
    }
    const std::set<const void*> destinations(visited.begin(), visited.end());
    EXPECT_EQ(actual.first_transitions, first);
    EXPECT_EQ(actual.last_transitions, last);
    EXPECT_EQ(actual.edge_visits, visited.size());
    EXPECT_EQ(actual.dependency_entries, destinations.size());
    const auto height = std::max(index_height(old_manifest), index_height(new_manifest));
    // Each changed key copies/splits its search path and at most one adjacent
    // rebalance path per level. Lookup calls are <=3 per touched key and <=2 per
    // changed key; the two constructors each authenticate the root channel once.
    const auto path_bound = 2 * (changed + 1) * (height + 1);
    EXPECT_LE(actual.index_copied_pages, path_bound);
    EXPECT_LE(actual.index_visited_pages, path_bound);
    EXPECT_LE(actual.index_lookup_pages, (3 * destinations.size() + 2 * changed + 2) * (height + 1));
}
} // namespace
TEST(AdaptiveSnapshotOwnerManifest, FixedDemandAtTwoSizesMatchesFullCensusAndPreservesControlsAndBuffers)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    for (const std::size_t base : {512, 1024})
    {
        SCOPED_TRACE(base);
        auto source = cartesian_source(true);
        std::vector<const A::AdaptiveLeaf*> rows;
        for (const auto& leaf : source->owned_leaves())
            rows.push_back(&leaf);
        ASSERT_GE(rows.size(), base + 8);
        const auto acquire = [&](std::span<const A::AdaptiveLeaf* const> wanted)
        {
            std::size_t used = 0;
            return source->geometry_snapshot(
                wanted, ordinary, A::GeometrySnapshotPurpose::OriginalSources, used, ample);
        };
        auto first = acquire(std::span(rows).first(base));
        agrees_with_independent_census(first);
        const auto first_manifest = capture(*first)[0];
        const auto first_graph = physical_graph(*first);
        using SourceAccess = P::details::NativeCellGeometrySourceAccess;
        struct Physical
        {
            const A::AdaptiveLeaf* row;
            P::details::NativeCartesianCellReference reference;
        };
        std::vector<Physical> physical;
        std::map<const void*, std::map<const void*, std::size_t>> source_buffers;
        std::set<const void*> recipe_controls;
        for (std::size_t row = 0; row < base; ++row)
        {
            const auto slot = first->find(*rows[row]);
            ASSERT_TRUE(slot);
            const auto ref = SourceAccess::source_cell(P::CellGeometryView<ordinary>(first->ordinary_catalog(), *slot));
            recipe_controls.insert(rows[row]->native_recipe.get());
            if (!source_buffers.contains(ref.source.get()))
            {
                auto& buffers = source_buffers[ref.source.get()];
                SourceAccess::visit_payload(
                    *ref.source, [&](const void* address, std::size_t bytes) { buffers.emplace(address, bytes); });
            }
            physical.push_back({rows[row], ref});
        }
        ASSERT_EQ(recipe_controls.size(), base / 8);
        ASSERT_EQ(source_buffers.size(), recipe_controls.size());
        ASSERT_GT(first_manifest.owner_count(), 2 * G::leaf_capacity);
        ASSERT_GT(index_buffers(first_manifest).size(), 2u);
        auto second = acquire(std::span(rows).subspan(base, 4));
        agrees_with_independent_census(second);
        const auto second_manifest = capture(*second)[0];
        const auto second_graph = physical_graph(*second);
        std::set<const void*> checked_sources;
        for (const auto& before : physical)
        {
            const auto slot = second->find(*before.row);
            ASSERT_TRUE(slot);
            const auto ref =
                SourceAccess::source_cell(P::CellGeometryView<ordinary>(second->ordinary_catalog(), *slot));
            EXPECT_EQ(ref.source.get(), before.reference.source.get());
            EXPECT_FALSE(ref.source.owner_before(before.reference.source));
            EXPECT_FALSE(before.reference.source.owner_before(ref.source));
            EXPECT_EQ(ref.cell_slot, before.reference.cell_slot);
            if (checked_sources.insert(ref.source.get()).second)
            {
                std::map<const void*, std::size_t> buffers;
                SourceAccess::visit_payload(
                    *ref.source, [&](const void* address, std::size_t bytes) { buffers.emplace(address, bytes); });
                EXPECT_EQ(buffers, source_buffers.at(before.reference.source.get()));
            }
        }
        auto third = acquire(std::span(rows).subspan(base + 4, 4));
        agrees_with_independent_census(third);
        ASSERT_EQ(third->ordinary_catalog()->size(), base + 8);
        const auto before = records(first_manifest), after = records(second_manifest);
        std::size_t reused = 0;
        for (const auto& [key, row] : before)
            if (const auto found = after.find(key); found != after.end())
            {
                EXPECT_EQ(found->second.bytes, row.bytes);
                EXPECT_FALSE(row.owner.owner_before(found->second.owner));
                EXPECT_FALSE(found->second.owner.owner_before(row.owner));
                ++reused;
            }
        EXPECT_GT(reused, 0u);
        const auto old_buffers = index_buffers(first_manifest), new_buffers = index_buffers(second_manifest);
        EXPECT_GT(std::count_if(old_buffers.begin(), old_buffers.end(),
                      [&](const void* address) { return new_buffers.contains(address); }),
            0);
        const auto work = A::adaptive_detail::GeometrySnapshotAccess::owner_statistics(*second);
        EXPECT_EQ(work.imported_entries, 0u);
        expect_exact_frontier(first_graph, second_graph, work, first_manifest, second_manifest);
        std::vector<const void*> old_pages;
        for (const auto& [page, ordinal] : first_graph.binding_last)
            old_pages.push_back(page);
        std::sort(old_pages.begin(), old_pages.end(), [&](const void* left, const void* right)
            { return first_graph.binding_last.at(left) < first_graph.binding_last.at(right); });
        // Exactly the two highest selected binding pages may remain fluid.
        // Every older page keeps its control and every actual backing buffer.
        const auto fluid = std::min<std::size_t>(2, old_pages.size());
        for (std::size_t page = 0; page < old_pages.size() - fluid; ++page)
        {
            ASSERT_TRUE(second_graph.binding_buffers.contains(old_pages[page]));
            EXPECT_EQ(
                first_graph.binding_buffers.at(old_pages[page]), second_graph.binding_buffers.at(old_pages[page]));
        }
        const auto retired_pages = std::count_if(first_graph.binding_last.begin(), first_graph.binding_last.end(),
            [&](const auto& page) { return !second_graph.binding_last.contains(page.first); });
        const auto added_pages = std::count_if(second_graph.binding_last.begin(), second_graph.binding_last.end(),
            [&](const auto& page) { return !first_graph.binding_last.contains(page.first); });
        EXPECT_LE(retired_pages, 2);
        const auto repacked_rows = 2 * G::leaf_capacity + 4;
        EXPECT_LE(added_pages, (repacked_rows + G::leaf_capacity - 1) / G::leaf_capacity);
        std::weak_ptr<const S> weak_first = first, weak_second = second;
        std::weak_ptr<const A::AdaptiveLeafLayout> layout = source->leaf_layout();
        first.reset();
        second.reset();
        rows.clear();
        source.reset();
        EXPECT_TRUE(weak_first.expired());
        EXPECT_TRUE(weak_second.expired());
        EXPECT_TRUE(layout.expired());
        agrees_with_independent_census(third);
    }
}

TEST(AdaptiveSnapshotOwnerManifest, PreparedSecondPageAndRawRebindingsMatchFullCensus)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    auto source = cartesian_source();
    const auto leaf = [&](std::uint64_t ordinal)
    {
        for (const auto& row : source->owned_leaves())
            if (row.native_cell == ordinal)
                return &row;
        return static_cast<const A::AdaptiveLeaf*>(nullptr);
    };
    const std::array selected{leaf(65), leaf(79), leaf(110), leaf(10)};
    for (const auto* row : selected)
        ASSERT_NE(row, nullptr);
    const auto acquire = [&](std::span<const A::AdaptiveLeaf* const> wanted, A::GeometrySnapshotPurpose purpose)
    {
        std::size_t used = 0;
        return source->geometry_snapshot(wanted, ordinary, purpose, used, ample);
    };
    for (unsigned row = 0; row < 3; ++row)
        agrees_with_independent_census(acquire(std::span(selected).subspan(row, 1), A::GeometrySnapshotPurpose::State));
    auto state = acquire(std::span(selected).first(3), A::GeometrySnapshotPurpose::State);
    std::size_t pages = 0;
    state->visit_prepared_cartesian_sources(
        [&](const auto& page)
        {
            EXPECT_EQ(page.size(), 64u);
            ++pages;
        });
    EXPECT_GT(pages, 0u);
    auto raw = acquire(selected, A::GeometrySnapshotPurpose::OriginalSources);
    agrees_with_independent_census(raw);
    const std::array reordered{selected[2], selected[0], selected[1]};
    raw = acquire(reordered, A::GeometrySnapshotPurpose::OriginalSources);
    agrees_with_independent_census(raw);
    state.reset();
    source.reset();
    agrees_with_independent_census(raw);
}

TEST(AdaptiveSnapshotOwnerManifest, ProductAndExplicitCertificatesKeepBothArithmeticChannelsExact)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    namespace F = SimpleFluid::test::adaptive_storage;
    for (const auto provider : {F::Provider::Product, F::Provider::Polyhedral})
        for (const auto mode : {ordinary, ArithmeticMode::BoundedCertification})
        {
            SCOPED_TRACE(F::name(provider));
            SCOPED_TRACE(static_cast<unsigned>(mode));
            auto fixture = F::make_disconnected(provider, 4);
            auto source = std::make_shared<const H>(std::move(fixture.owned), std::vector<A::AdaptiveFamily>{},
                std::move(fixture.kernel), Tpetra::getDefaultComm());
            std::vector<const A::AdaptiveLeaf*> rows;
            for (const auto& leaf : source->owned_leaves())
                rows.push_back(&leaf);
            ASSERT_GE(rows.size(), 4u);
            std::size_t used = 0;
            auto first = source->geometry_snapshot(
                std::span(rows).first(2), mode, A::GeometrySnapshotPurpose::State, used, ample);
            agrees_with_independent_census(first);
            used = 0;
            auto second = source->geometry_snapshot(
                std::span(rows).subspan(2, 2), mode, A::GeometrySnapshotPurpose::State, used, ample);
            agrees_with_independent_census(second);
            if (mode == ArithmeticMode::BoundedCertification)
                EXPECT_GT(capture(*second)[1].owner_count(), 0u);
            first.reset();
            rows.clear();
            source.reset();
            agrees_with_independent_census(second);
        }
}

TEST(AdaptiveSnapshotOwnerManifest, TwoDeferredSubsetsShareOnePhysicalBaseAndSupportFrontier)
{
    using SourceAccess = P::details::NativeCellGeometrySourceAccess;
    using Editor = P::details::GeometryOwnerGraphEditor;
    P::CartesianSourceDescription description;
    description.axes = {{{0, 1, 2}, {0, 1}, {0, 1}}};
    description.original_supports.push_back(P::details::GeometryIdentityAccess::support_axis(0, 0));
    for (std::uint32_t row = 0; row < 2; ++row)
    {
        P::CartesianSourceCellSlot slot;
        slot.lower = {row, 0, 0};
        slot.upper = {row + 1, 1, 1};
        if (!row)
            slot.support_slots[0] = 0;
        description.cells.push_back(slot);
    }
    P::GeometryLimits limits;
    limits.arithmetic_mode = ordinary;
    limits.byte_budget = ample;
    const auto original =
        SourceAccess::make(std::move(description), limits, P::details::NativeCellGeometryMoments::DeferredRaw);
    const auto first = P::CartesianCellSource::clone_deferred_subset(original, std::array<std::uint64_t, 1>{0}, limits);
    const auto second =
        P::CartesianCellSource::clone_deferred_subset(original, std::array<std::uint64_t, 1>{1}, limits);
    const auto base = SourceAccess::base_owner(*first);
    ASSERT_EQ(base, SourceAccess::base_owner(*second));
    ASSERT_FALSE(base.owner_before(SourceAccess::base_owner(*second)));
    ASSERT_FALSE(SourceAccess::base_owner(*second).owner_before(base));
    // Independent support ownership census: these identities are members of the
    // one shared base, so their incoming physical edges occur once per base.
    std::map<const void*, std::shared_ptr<const void>> support_owners;
    for (const auto& support : SourceAccess::retained_supports(*first))
        P::visit_identity_allocations(support,
            [&](const P::GeometryIdentityAllocation& row)
            {
                support_owners.emplace(row.owner.get(), row.owner);
                return true;
            });
    ASSERT_FALSE(support_owners.empty());
    const auto check_supports = [&](const G::GeometryOwnerManifest& manifest)
    {
        std::size_t found = 0;
        manifest.visit(
            [&](const G::OwnerRecord& row)
            {
                if (const auto expected = support_owners.find(row.owner.get()); expected != support_owners.end())
                {
                    EXPECT_FALSE(expected->second.owner_before(row.owner));
                    EXPECT_FALSE(row.owner.owner_before(expected->second));
                    EXPECT_EQ(row.references, 1u);
                    ++found;
                }
            });
        EXPECT_EQ(found, support_owners.size());
    };
    std::size_t used = 0;
    Editor initial({}, 0, used, ample);
    initial.cartesian(first, true);
    const auto accepted = initial.finish();
    check_supports(accepted);
    P::details::GeometryOwnerGraphWork add_work;
    Editor addition(accepted, 0, used, ample, nullptr, &add_work);
    addition.cartesian(second, true);
    const auto both = addition.finish();
    EXPECT_EQ(add_work.edge_visits, 2u); // New source -> already retained base; stop there.
    EXPECT_EQ(add_work.first_transitions, 1u);
    EXPECT_EQ(both.references(0, Editor::external_kind(), base), 2u);
    check_supports(both);
    P::details::GeometryOwnerGraphWork retire_work;
    Editor retirement(both, 0, used, ample, nullptr, &retire_work);
    retirement.cartesian(first, false);
    const auto surviving = retirement.finish();
    EXPECT_EQ(retire_work.edge_visits, 2u);
    EXPECT_EQ(retire_work.last_transitions, 1u);
    EXPECT_EQ(surviving.references(0, Editor::external_kind(), base), 1u);
    check_supports(surviving);
    Editor final(surviving, 0, used, ample);
    final.cartesian(second, false);
    EXPECT_EQ(final.finish().owner_count(), 0u);
}

TEST(AdaptiveSnapshotOwnerManifest, TwoChannelSnapshotExactBudgetAndOneByteShortPreserveAcceptedRootAndSeal)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    namespace F = SimpleFluid::test::adaptive_storage;
    struct Baseline
    {
        std::shared_ptr<const H> source;
        std::shared_ptr<const S> accepted;
        std::vector<const A::AdaptiveLeaf*> selected;
    };
    const auto baseline = [&]
    {
        auto fixture = F::make_disconnected(F::Provider::Polyhedral, 1);
        Baseline result;
        result.source = std::make_shared<const H>(std::move(fixture.owned), std::vector<A::AdaptiveFamily>{},
            std::move(fixture.kernel), Tpetra::getDefaultComm());
        for (const auto& leaf : result.source->owned_leaves())
            result.selected.push_back(&leaf);
        if (result.selected.size() > 2)
            result.selected.resize(2);
        std::size_t used = 0;
        result.accepted =
            result.source->geometry_snapshot(result.selected, ordinary, A::GeometrySnapshotPurpose::State, used, ample);
        return result;
    };
    const auto attempt = [&](const Baseline& base, std::size_t& used, std::size_t cap,
                             P::GeometryAllocationLedger& ordinary_ledger, P::GeometryAllocationLedger& bounded_ledger)
    {
        return base.source->geometry_snapshot(base.selected, ArithmeticMode::BoundedCertification,
            A::GeometrySnapshotPurpose::State, used, cap, &ordinary_ledger, &bounded_ledger);
    };
    constexpr std::size_t prefix = 37;
    auto probe = baseline();
    P::GeometryAllocationLedger probe_ordinary, probe_bounded;
    std::size_t final_used = prefix;
    auto measured = attempt(probe, final_used, ample, probe_ordinary, probe_bounded);
    EXPECT_GT(capture(*measured)[0].owner_count(), 0u);
    EXPECT_GT(capture(*measured)[1].owner_count(), 0u);
    agrees_with_independent_census(measured);

    // The outer final counter does not include the nested PLIC constructor's
    // temporary workspace envelope. Find the exact admission threshold without
    // a tolerance or an extra margin, keeping the unchanged ample ceiling.
    const auto succeeds = [&](std::size_t cap)
    {
        auto trial = baseline();
        P::GeometryAllocationLedger trial_ordinary, trial_bounded;
        std::size_t used = prefix;
        try
        {
            (void) attempt(trial, used, cap, trial_ordinary, trial_bounded);
            EXPECT_LE(used, cap);
            return true;
        }
        catch (const P::GeometryError& error)
        {
            if (error.code() != P::GeometryErrorCode::ResourceLimit)
                throw;
            return false;
        }
    };
    std::size_t required = final_used;
    if (!succeeds(required))
    {
        std::size_t lower = required, upper = ample;
        while (upper - lower > 1)
        {
            const auto middle = lower + (upper - lower) / 2;
            if (succeeds(middle))
                upper = middle;
            else
                lower = middle;
        }
        required = upper;
    }
    RecordProperty("MinimumAdmissionCap", std::to_string(required));
    RecordProperty("FinalOuterAdmission", std::to_string(final_used));
    auto exact_base = baseline();
    P::GeometryAllocationLedger exact_ordinary, exact_bounded;
    std::size_t exact = prefix;
    auto completed = attempt(exact_base, exact, required, exact_ordinary, exact_bounded);
    EXPECT_LE(exact, required);
    EXPECT_EQ(exact, final_used);
    agrees_with_independent_census(completed);

    auto rejected = baseline();
    const auto layout = rejected.source->leaf_layout();
    const auto old_manifests = capture(*rejected.accepted);
    const auto old_seal = A::adaptive_detail::GeometrySnapshotAccess::cache_admission_statistics(*rejected.source);
    P::GeometryAllocationLedger short_ordinary, short_bounded;
    std::size_t short_used = prefix;
    EXPECT_THROW((void) attempt(rejected, short_used, required - 1, short_ordinary, short_bounded), std::exception);
    EXPECT_GE(short_used, prefix);
    EXPECT_LE(short_used, required - 1);
    EXPECT_EQ(rejected.source->leaf_layout(), layout);
    EXPECT_FALSE(rejected.source->leaf_layout().owner_before(layout));
    EXPECT_FALSE(layout.owner_before(rejected.source->leaf_layout()));
    const auto now = A::adaptive_detail::GeometrySnapshotAccess::cache_admission_statistics(*rejected.source);
    EXPECT_EQ(now.resident_bytes, old_seal.resident_bytes);
    EXPECT_EQ(now.lease_bytes, old_seal.lease_bytes);
    std::size_t roots = 0;
    rejected.source->visit_geometry_snapshots(
        [&](const S& root)
        {
            EXPECT_EQ(&root, rejected.accepted.get());
            ++roots;
        });
    EXPECT_EQ(roots, 1u);
    const auto after = capture(*rejected.accepted);
    for (unsigned channel = 0; channel < 2; ++channel)
    {
        const auto expected_rows = records(old_manifests[channel]), actual_rows = records(after[channel]);
        ASSERT_EQ(expected_rows.size(), actual_rows.size());
        for (const auto& [key, row] : expected_rows)
        {
            const auto found = actual_rows.find(key);
            ASSERT_NE(found, actual_rows.end());
            EXPECT_EQ(found->second.bytes, row.bytes);
            EXPECT_EQ(found->second.references, row.references);
            EXPECT_FALSE(found->second.owner.owner_before(row.owner));
            EXPECT_FALSE(row.owner.owner_before(found->second.owner));
        }
    }
    EXPECT_THROW(
        short_ordinary.retain_external_owner(rejected.accepted, sizeof(S), short_used, ample), P::GeometryError);
    EXPECT_THROW(
        short_bounded.retain_external_owner(rejected.accepted, sizeof(S), short_used, ample), P::GeometryError);
    P::GeometryAllocationLedger retry_ordinary, retry_bounded;
    std::size_t retry = prefix;
    auto retried = attempt(rejected, retry, required, retry_ordinary, retry_bounded);
    EXPECT_LE(retry, required);
    EXPECT_EQ(retry, final_used);
    agrees_with_independent_census(retried);
}

TEST(AdaptiveSnapshotOwnerManifest, FlatLegacyBaseAdmissionStillRetainsEverySourceSupportChild)
{
    using SourceAccess = P::details::NativeCellGeometrySourceAccess;
    P::CartesianSourceDescription description;
    description.axes = {{{0, 1}, {0, 1}, {0, 1}}};
    description.original_supports.push_back(P::details::GeometryIdentityAccess::support_axis(0, 0));
    P::CartesianSourceCellSlot slot;
    slot.upper = {1, 1, 1};
    slot.support_slots[0] = 0;
    description.cells.push_back(slot);
    P::GeometryLimits limits;
    limits.arithmetic_mode = ordinary;
    limits.byte_budget = ample;
    const auto source =
        SourceAccess::make(std::move(description), limits, P::details::NativeCellGeometryMoments::DeferredRaw);
    ASSERT_FALSE(SourceAccess::retained_supports(*source).empty());
    const auto base = SourceAccess::base_owner(*source);
    const auto base_bytes = SourceAccess::base_bytes(*source);
    P::GeometryAllocationLedger fresh, prefixed;
    std::size_t fresh_used = 0, prefixed_used = 0;
    fresh.retain(source, fresh_used, ample);
    prefixed.retain_external_owner(base, base_bytes[0] + base_bytes[1] + base_bytes[2], prefixed_used, ample);
    prefixed.retain(source, prefixed_used, ample);
    const auto expected = records(P::details::GeometryOwnerImportAccess::export_manifest(fresh, charge(fresh_used)));
    const auto actual =
        records(P::details::GeometryOwnerImportAccess::export_manifest(prefixed, charge(prefixed_used)));
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(fresh.payload_bytes(), prefixed.payload_bytes());
    for (const auto& [key, row] : expected)
    {
        const auto found = actual.find(key);
        ASSERT_NE(found, actual.end());
        EXPECT_EQ(found->second.bytes, row.bytes);
        EXPECT_FALSE(found->second.owner.owner_before(row.owner));
        EXPECT_FALSE(row.owner.owner_before(found->second.owner));
    }
}

TEST(AdaptiveSnapshotOwnerManifest, TwoCatalogConstructionReceiptsIncludeControlAndScratchBeforeNextAdmission)
{
    using Catalog = P::CellGeometryCatalog<ordinary>;
    using Demand = P::details::CellCatalogDemandAccess;
    const auto cell = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto original = Catalog::materialized(std::vector<P::ConvexGeometryPtr<ordinary>>(65, cell));
    struct Counter
    {
        std::size_t used = 37, limit = ample, calls = 0;
    };
    const auto sink = [](Counter& counter) -> G::Admission
    {
        return {&counter, [](void* raw, std::size_t bytes)
            {
                auto& value = *static_cast<Counter*>(raw);
                if (value.used > value.limit || bytes > value.limit - value.used)
                    throw P::GeometryError(P::GeometryErrorCode::ResourceLimit, "Test construction cap exhausted.");
                value.used += bytes;
                ++value.calls;
            }};
    };
    const auto initialize = [&](Counter& counter)
    {
        P::GeometryLimits limits;
        limits.arithmetic_mode = ordinary;
        limits.byte_budget = counter.limit - counter.used;
        return Demand::initialize_precharged(original, limits, nullptr, sink(counter));
    };
    Counter measured;
    auto first = initialize(measured);
    const auto first_receipt = measured.used - 37;
    std::size_t retained_metadata = first->storage_bytes();
    Demand::visit_dependencies(*first,
        {&retained_metadata,
            [](void* raw, const std::shared_ptr<const void>&, std::size_t, std::size_t admitted)
            {
                *static_cast<std::size_t*>(raw) += admitted;
                return true;
            }},
        {nullptr, [](void*, G::SourceKind, const std::shared_ptr<const void>&) {}});
    EXPECT_GE(first_receipt, retained_metadata);
    EXPECT_GT(measured.calls, 1u);
    auto second = initialize(measured);
    const auto required = measured.used;
    EXPECT_EQ(second->size(), 65u);
    Counter exact{37, required};
    auto exact_first = initialize(exact);
    auto exact_second = initialize(exact);
    EXPECT_EQ(exact.used, required);
    EXPECT_EQ(exact_first->size(), 65u);
    EXPECT_EQ(exact_second->size(), 65u);
    Counter short_limit{37, required - 1};
    auto kept = initialize(short_limit);
    EXPECT_THROW((void) initialize(short_limit), P::GeometryError);
    EXPECT_EQ(kept->size(), 65u);
    EXPECT_EQ(original->size(), 65u);
    EXPECT_GE(short_limit.used, 37 + first_receipt);
    EXPECT_LE(short_limit.used, required - 1);
    Counter retry{37, required};
    EXPECT_NO_THROW((void) initialize(retry));
    EXPECT_NO_THROW((void) initialize(retry));
    EXPECT_EQ(retry.used, required);
}

TEST(AdaptiveSnapshotOwnerManifest, StoredIndexCreditsDeduplicateFreshRepeatedAndPreviouslyImportedChannels)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    namespace F = SimpleFluid::test::adaptive_storage;
    for (const auto mode : {ordinary, ArithmeticMode::BoundedCertification})
    {
        auto fixture = F::make_disconnected(F::Provider::Polyhedral, 1);
        auto source = std::make_shared<const H>(std::move(fixture.owned), std::vector<A::AdaptiveFamily>{},
            std::move(fixture.kernel), Tpetra::getDefaultComm());
        std::vector<const A::AdaptiveLeaf*> selected;
        for (const auto& leaf : source->owned_leaves())
            selected.push_back(&leaf);
        if (selected.size() > 2)
            selected.resize(2);
        P::GeometryAllocationLedger constructed_ordinary, constructed_bounded;
        std::size_t constructed_used = 37;
        const auto snapshot = source->geometry_snapshot(selected, mode, A::GeometrySnapshotPurpose::State,
            constructed_used, ample, &constructed_ordinary, &constructed_bounded);
        const auto after_construction = constructed_used;
        snapshot->retain(constructed_ordinary, constructed_bounded, constructed_used, ample);
        EXPECT_EQ(constructed_used, after_construction);
        const auto manifests = capture(*snapshot);
        P::GeometryAllocationLedger fresh_ordinary, fresh_bounded;
        std::size_t measured = 37;
        snapshot->retain(fresh_ordinary, fresh_bounded, measured, ample);
        const auto first_retain = measured;
        snapshot->retain(fresh_ordinary, fresh_bounded, measured, ample);
        EXPECT_EQ(measured, first_retain);
        agrees_with_independent_census(snapshot);
        P::GeometryAllocationLedger imported_ordinary, imported_bounded;
        std::size_t imported_used = 37;
        P::details::GeometryOwnerImportAccess::import(imported_ordinary, manifests[0], imported_used, ample);
        if (manifests[1].owner_count())
            P::details::GeometryOwnerImportAccess::import(imported_bounded, manifests[1], imported_used, ample);
        const auto imported_payload = imported_ordinary.payload_bytes() + imported_bounded.payload_bytes();
        snapshot->retain(imported_ordinary, imported_bounded, imported_used, ample);
        EXPECT_EQ(imported_ordinary.payload_bytes() + imported_bounded.payload_bytes(), imported_payload + sizeof(S));
        const auto first_imported_retain = imported_used;
        snapshot->retain(imported_ordinary, imported_bounded, imported_used, ample);
        EXPECT_EQ(imported_used, first_imported_retain);
        std::size_t imported_credits = 0;
        for (unsigned channel = 0; channel < 2; ++channel)
            P::details::GeometryAllocationOwnerAccess::visit_metadata_credits(
                channel ? imported_bounded : imported_ordinary,
                [&](const std::shared_ptr<const void>&, std::size_t payload, std::size_t admitted, bool prepaid_index)
                {
                    EXPECT_TRUE(prepaid_index);
                    EXPECT_EQ(admitted - payload, G::control_bytes);
                    ++imported_credits;
                });
        EXPECT_GT(imported_credits, 0u);
        P::GeometryAllocationLedger exact_ordinary, exact_bounded;
        std::size_t exact = 37;
        EXPECT_NO_THROW(snapshot->retain(exact_ordinary, exact_bounded, exact, first_retain));
        EXPECT_EQ(exact, first_retain);
        P::GeometryAllocationLedger short_ordinary, short_bounded;
        std::size_t short_used = 37;
        EXPECT_THROW(snapshot->retain(short_ordinary, short_bounded, short_used, first_retain - 1), P::GeometryError);
        EXPECT_THROW(snapshot->retain(short_ordinary, short_bounded, short_used, ample), P::GeometryError);
        P::GeometryAllocationLedger retry_ordinary, retry_bounded;
        std::size_t retry = 37;
        EXPECT_NO_THROW(snapshot->retain(retry_ordinary, retry_bounded, retry, first_retain));
        EXPECT_EQ(retry, first_retain);
    }
}

TEST(AdaptiveSnapshotOwnerManifest, UncachedNativeResultRecordsItsConstructedIndexCredits)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    A::CartesianAMR native(grid);
    std::vector<std::uint64_t> ids(native.geometry()->num_cells());
    std::iota(ids.begin(), ids.end(), 0);
    auto detached = native.detach_owned(ids, {'n', 'a', 't', 'i', 'v', 'e'});
    auto source = std::make_shared<const H>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), Tpetra::getDefaultComm());
    auto draft = source->native_draft();
    std::vector<const A::AdaptiveLeaf*> selected;
    for (const auto& leaf : draft.owned)
        selected.push_back(&leaf);
    ASSERT_FALSE(selected.empty());
    P::GeometryAllocationLedger ordinary_ledger, bounded_ledger;
    std::size_t used = 37;
    const auto result =
        A::adaptive_detail::GeometrySnapshotAccess::native_draft(*source, draft, selected, source->observation(),
            ordinary, A::GeometrySnapshotPurpose::State, used, ample, ordinary_ledger, bounded_ledger);
    const auto constructed = used;
    result->retain(ordinary_ledger, bounded_ledger, used, ample);
    EXPECT_EQ(used, constructed);
    agrees_with_independent_census(result);
}

TEST(AdaptiveSnapshotOwnerManifest, NewIndexReceiptsAfterFreshImportNeverEnumerateOldIndexMetadata)
{
    using Editor = P::details::GeometryOwnerGraphEditor;
    using Owners = P::details::GeometryAllocationOwnerAccess;
    struct Tree
    {
        std::map<const void*, std::shared_ptr<const void>> controls;
        std::map<const void*, const void*> parents;
        std::map<const void*, std::vector<const void*>> leaves;
        std::size_t height = 0;
    };
    const auto tree = [](const G::GeometryOwnerManifest& manifest)
    {
        Tree result;
        manifest.visit_dependencies(
            {&result,
                [](void* raw, const std::shared_ptr<const void>& owner, std::size_t, std::size_t)
                {
                    static_cast<Tree*>(raw)->controls.emplace(owner.get(), owner);
                    return true;
                }},
            {nullptr, [](void*, const G::OwnerRecord&) {}});
        manifest.visit_borrowed_dependencies({&result,
                                                 [](void* raw, const void* owner, const void* parent, std::size_t)
                                                 {
                                                     static_cast<Tree*>(raw)->parents.emplace(owner, parent);
                                                     return true;
                                                 }},
            {&result, [](void* raw, const void* parent, const G::OwnerRecord& row)
                { static_cast<Tree*>(raw)->leaves[parent].push_back(row.owner.get()); }});
        for (const auto& [owner, parent] : result.parents)
        {
            std::size_t depth = 0;
            for (auto at = parent; at; at = result.parents.at(at))
                ++depth;
            result.height = std::max(result.height, depth);
        }
        return result;
    };
    for (const std::size_t count : {2048u, 4096u})
    {
        SCOPED_TRACE(count);
        std::vector<std::shared_ptr<const std::uint64_t>> leaves;
        for (std::size_t row = 0; row < count; ++row)
            leaves.push_back(std::make_shared<const std::uint64_t>(row));
        std::size_t used = 0;
        Editor initial({}, 0, used, ample);
        for (const auto& leaf : leaves)
            (void) initial.external(leaf, sizeof(*leaf), true);
        const auto accepted = initial.finish();
        const auto before_tree = tree(accepted);
        const auto before_buffers = index_buffers(accepted);
        P::GeometryAllocationLedger ledger;
        used = 37;
        P::details::GeometryOwnerImportAccess::import(ledger, accepted, used, ample);
        Owners::reset_work_statistics();
        const auto replacement = std::make_shared<const std::uint64_t>(count);
        P::details::GeometryOwnerGraphWork graph_work;
        Editor next(accepted, 0, used, ample, &ledger, &graph_work);
        (void) next.external(replacement, sizeof(*replacement), true);
        (void) next.external(leaves.front(), sizeof(*leaves.front()), false);
        const auto result = next.finish();
        const auto work = Owners::work_statistics();
        const auto after_tree = tree(result);
        // Two independently specified changed keys. A single insert/delete pair
        // can affect only their search paths and local split/merge neighbors.
        constexpr std::size_t changed_keys = 2;
        const auto height = std::max(before_tree.height, after_tree.height);
        const auto copy_bound = 2 * (changed_keys + 1) * (height + 1);
        ASSERT_GT(before_tree.controls.size() - 1, copy_bound);
        EXPECT_LE(graph_work.index.copied_pages, copy_bound);
        std::vector<std::pair<const void*, const void*>> leaf_maxima;
        const auto less = std::less<const void*>{};
        std::set<const void*> affected;
        for (const auto& [leaf, keys] : before_tree.leaves)
        {
            ASSERT_FALSE(keys.empty());
            leaf_maxima.emplace_back(*std::max_element(keys.begin(), keys.end(), less), leaf);
            if (std::find(keys.begin(), keys.end(), leaves.front().get()) != keys.end())
                affected.insert(leaf);
        }
        std::sort(leaf_maxima.begin(), leaf_maxima.end(),
            [&](const auto& left, const auto& right) { return less(left.first, right.first); });
        auto target = std::lower_bound(leaf_maxima.begin(), leaf_maxima.end(), replacement.get(),
            [&](const auto& item, const void* key) { return less(item.first, key); });
        if (target == leaf_maxima.end())
            target = std::prev(leaf_maxima.end());
        const auto position = std::size_t(target - leaf_maxima.begin());
        // Conservatively allow the two adjacent leaves for any rebalance.
        for (const auto at :
            {position ? position - 1 : position, position, std::min(position + 1, leaf_maxima.size() - 1)})
            affected.insert(leaf_maxima[at].second);
        ASSERT_GT(before_tree.leaves.size(), affected.size());
        for (const auto& [leaf, keys] : before_tree.leaves)
            if (!affected.contains(leaf))
            {
                const auto found = after_tree.controls.find(leaf);
                ASSERT_NE(found, after_tree.controls.end());
                EXPECT_FALSE(before_tree.controls.at(leaf).owner_before(found->second));
                EXPECT_FALSE(found->second.owner_before(before_tree.controls.at(leaf)));
            }
        const auto after_buffers = index_buffers(result);
        const auto shared_buffers = std::count_if(before_buffers.begin(), before_buffers.end(),
            [&](const void* value) { return after_buffers.contains(value); });
        EXPECT_GE(std::size_t(shared_buffers), before_tree.leaves.size() - affected.size());
        EXPECT_EQ(work.metadata_credit_seed_rows, 0u);
        EXPECT_GT(work.metadata_credit_created, 0u);
        EXPECT_EQ(result.owner_count(), leaves.size());
        // The later explicit legacy retain resolves administrative credit once;
        // that compatibility census remains separate from qualified construction.
        struct Metadata
        {
            std::shared_ptr<const void> owner;
            std::size_t payload = 0, admitted = 0;
        } root;
        accepted.visit_dependencies(
            {&root,
                [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
                {
                    *static_cast<Metadata*>(raw) = {owner, payload, admitted};
                    return false;
                }},
            {nullptr, [](void*, const G::OwnerRecord&) {}});
        Owners::retain_manifest_metadata(ledger, 0, root.owner, root.payload, root.admitted, used, ample);
        EXPECT_GT(Owners::work_statistics().metadata_credit_seed_rows, 0u);
    }
}

TEST(AdaptiveSnapshotOwnerManifest, PrepaidFullCaptureChargesOnlyNewBookkeepingAndRetainsAdministrativeTotals)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    auto source = cartesian_source();
    std::vector<const A::AdaptiveLeaf*> rows;
    for (const auto& leaf : source->owned_leaves())
        if (rows.size() < 65)
            rows.push_back(&leaf);
    std::size_t acquired = 0;
    const auto snapshot =
        source->geometry_snapshot(rows, ordinary, A::GeometrySnapshotPurpose::OriginalSources, acquired, ample);
    P::GeometryAllocationLedger capture_ordinary, capture_bounded;
    std::size_t used = 37;
    P::details::GeometryAllocationOwnerAccess::begin_prepaid_capture(capture_ordinary, used, ample);
    snapshot->retain(capture_ordinary, capture_bounded, used, ample);
    std::size_t control = 0;
    P::details::GeometryAllocationOwnerAccess::visit_metadata_credits(capture_ordinary,
        [&](const std::shared_ptr<const void>&, std::size_t payload, std::size_t admitted, bool imported)
        {
            EXPECT_FALSE(imported);
            control += admitted - payload;
        });
    EXPECT_GT(control, 0u);
    EXPECT_EQ(used - 37, capture_ordinary.bookkeeping_bytes() - control);
    const auto before = used;
    snapshot->retain(capture_ordinary, capture_bounded, used, ample);
    EXPECT_EQ(used, before);
}

TEST(AdaptiveSnapshotOwnerManifest, IndexCreditControlAndSizeConflictsPoisonOnlyTrialLedger)
{
    using Owners = P::details::GeometryAllocationOwnerAccess;
    using Editor = P::details::GeometryOwnerGraphEditor;
    const auto leaf = std::make_shared<const std::uint64_t>(1);
    std::size_t construction = 0;
    Editor initial({}, 0, construction, ample);
    (void) initial.external(leaf, sizeof(*leaf), true);
    const auto accepted = initial.finish();
    struct Metadata
    {
        std::shared_ptr<const void> owner;
        std::size_t payload = 0, admitted = 0;
    } root;
    accepted.visit_dependencies(
        {&root,
            [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
            {
                *static_cast<Metadata*>(raw) = {owner, payload, admitted};
                return false;
            }},
        {nullptr, [](void*, const G::OwnerRecord&) {}});
    const auto exercise = [&](bool foreign_control)
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 37;
        Owners::retain_manifest_metadata(ledger, 0, root.owner, root.payload, root.admitted, used, ample);
        const auto foreign = std::shared_ptr<const void>(root.owner.get(), [](const void*) {});
        EXPECT_THROW(Owners::retain_manifest_metadata(ledger, 0, foreign_control ? foreign : root.owner,
                         root.payload + !foreign_control, root.admitted + !foreign_control, used, ample),
            P::GeometryError);
        EXPECT_THROW(Owners::retain_manifest_metadata(ledger, 0, root.owner, root.payload, root.admitted, used, ample),
            P::GeometryError);
        EXPECT_EQ(accepted.owner_count(), 1u);
        EXPECT_EQ(accepted.references(0, Editor::external_kind(), leaf), 1u);
    };
    exercise(false);
    exercise(true);
}

TEST(AdaptiveSnapshotOwnerManifest, RawSameSelectionSharesBindingDataAndKeepsIndependentEvaluationContexts)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    using Access = A::adaptive_detail::GeometrySnapshotAccess;
    using Sources = P::details::NativeCellGeometrySourceAccess;
    const auto small_source = []
    {
        A::AMRGrid grid;
        grid.block_cells = {2, 2, 2};
        const A::CartesianAMR mesh(grid);
        std::vector<std::uint64_t> ids(mesh.geometry()->num_cells());
        std::iota(ids.begin(), ids.end(), 0);
        auto input = mesh.detach_owned(ids, {'r', 'a', 'w'});
        return std::make_shared<const H>(
            std::move(input.owned), std::move(input.families), std::move(input.kernel), Tpetra::getDefaultComm());
    };
    for (const bool preserve_order : {true, false})
    {
        SCOPED_TRACE(preserve_order);
        auto source = small_source();
        std::vector<const A::AdaptiveLeaf*> rows;
        for (const auto& leaf : source->owned_leaves())
            rows.push_back(&leaf);
        std::shared_ptr<const S> state;
        for (std::size_t row = 0; row < 3; ++row)
        {
            std::size_t used = 0;
            state = source->geometry_snapshot(
                std::span(rows).subspan(row, 1), ordinary, A::GeometrySnapshotPurpose::State, used, ample);
        }
        const auto* requested = rows[preserve_order ? 2 : 0];
        const auto old_slot = state->find(*requested);
        ASSERT_TRUE(old_slot);
        if (preserve_order)
            ASSERT_EQ(*old_slot, 0u);
        std::size_t used = 0;
        const auto raw = source->geometry_snapshot(
            std::array{requested}, ordinary, A::GeometrySnapshotPurpose::OriginalSources, used, ample);
        const auto raw_slot = raw->find(*requested);
        ASSERT_TRUE(raw_slot);
        ASSERT_EQ(*raw_slot, 0u);
        const auto old_binding = Access::binding_owner(*state), raw_binding = Access::binding_owner(*raw);
        if (preserve_order)
        {
            EXPECT_EQ(old_binding.get(), raw_binding.get());
            EXPECT_FALSE(old_binding.owner_before(raw_binding));
            EXPECT_FALSE(raw_binding.owner_before(old_binding));
        }
        else
            EXPECT_NE(old_binding.get(), raw_binding.get());
        EXPECT_TRUE(raw->native_metrics().empty());
        const auto policy = raw->evaluation_policy(*raw_slot);
        EXPECT_FALSE(policy.original_integrals);
        EXPECT_FALSE(policy.ordinary_integrals);
        EXPECT_FALSE(policy.bounded_integrals);
        P::CellGeometryView<ordinary> raw_view(raw->ordinary_catalog(), *raw_slot);
        auto old_reference = Sources::source_cell(P::CellGeometryView<ordinary>(state->ordinary_catalog(), *old_slot));
        const auto raw_reference = Sources::source_cell(raw_view);
        EXPECT_NE(raw_reference.source.get(), old_reference.source.get());
        const auto old_base = Sources::base_owner(*old_reference.source);
        const auto raw_base = Sources::base_owner(*raw_reference.source);
        EXPECT_EQ(old_base.get(), raw_base.get());
        EXPECT_FALSE(old_base.owner_before(raw_base));
        EXPECT_FALSE(raw_base.owner_before(old_base));
        const auto index = std::get<A::CartesianCellAncestry>(requested->key.ancestry).index;
        const std::weak_ptr<const H> weak_source = source;
        const std::weak_ptr<const S> weak_state = state;
        const std::weak_ptr<const P::CartesianCellSource> weak_context = old_reference.source;
        const std::weak_ptr<const A::AdaptiveLeafLayout> weak_layout = source->leaf_layout();
        old_reference = {};
        rows.clear();
        state.reset();
        source.reset();
        EXPECT_TRUE(weak_source.expired());
        EXPECT_TRUE(weak_state.expired());
        EXPECT_TRUE(weak_context.expired());
        EXPECT_TRUE(weak_layout.expired());
        const auto before = P::geometry_statistics().moment_evaluations;
        const auto moments = raw_view.moments({0, 0, 0});
        EXPECT_EQ(P::geometry_statistics().moment_evaluations, before + 1);
        EXPECT_EQ(moments.volume.value, 0.125);
        for (unsigned axis = 0; axis < 3; ++axis)
            EXPECT_EQ(moments.first_moment[axis].value, 0.125 * (double(index[axis]) + 0.5) * 0.5);
        agrees_with_independent_census(raw);
    }
}
