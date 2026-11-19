/** Draft tests for testGeometryAllocationLedger.cc/new owner-graph executable.
 * Expectations follow the explicit pointer members in FixtureNode, not exports.
 */
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <vector>
namespace
{
namespace P = SimpleFluid::PLIC;
namespace G = P::details::geometry_demand;
using Editor = P::details::GeometryOwnerGraphEditor;
constexpr std::size_t graph_limit = 64 * 1024 * 1024;
struct FixtureNode
{
    std::shared_ptr<const FixtureNode> first, second;
};
void edge(Editor& editor, const std::shared_ptr<const FixtureNode>& node, bool retain)
{
    if (!node || !editor.external(node, sizeof(FixtureNode), retain))
        return;
    edge(editor, node->first, retain);
    edge(editor, node->second, retain);
}
std::size_t references(
    const G::GeometryOwnerManifest& manifest, const std::shared_ptr<const FixtureNode>& node, unsigned channel = 0)
{
    return manifest.references(channel, Editor::external_kind(), node);
}
} // namespace
TEST(GeometryOwnerGraph, DiamondCountsPhysicalIncomingEdgesAndPrunesSharedDescendants)
{
    auto leaf = std::make_shared<const FixtureNode>();
    auto shared = std::make_shared<const FixtureNode>(FixtureNode{leaf, {}});
    auto left = std::make_shared<const FixtureNode>(FixtureNode{shared, {}});
    auto right = std::make_shared<const FixtureNode>(FixtureNode{shared, {}});
    auto first = std::make_shared<const FixtureNode>(FixtureNode{left, right});
    std::size_t used = 0;
    P::details::GeometryOwnerGraphWork initial_work;
    Editor initial({}, 0, used, graph_limit, nullptr, &initial_work);
    edge(initial, first, true);
    const auto accepted = initial.finish();
    EXPECT_EQ(accepted.owner_count(), 5u);
    EXPECT_EQ(accepted.payload_bytes(), 5 * sizeof(FixtureNode));
    EXPECT_EQ(references(accepted, shared), 2u);
    EXPECT_EQ(references(accepted, leaf), 1u);
    EXPECT_EQ(initial_work.first_transitions, 5u);
    EXPECT_EQ(initial_work.edge_visits, 6u);

    auto replacement_left = std::make_shared<const FixtureNode>(FixtureNode{shared, {}});
    auto replacement = std::make_shared<const FixtureNode>(FixtureNode{replacement_left, right});
    P::details::GeometryOwnerGraphWork work;
    Editor changed(accepted, 0, used, graph_limit, nullptr, &work);
    edge(changed, replacement, true);
    edge(changed, first, false);
    const auto result = changed.finish();
    EXPECT_EQ(result.owner_count(), 5u);
    EXPECT_EQ(references(result, shared), 2u);
    EXPECT_EQ(references(result, leaf), 1u);
    EXPECT_EQ(work.first_transitions, 2u);
    EXPECT_EQ(work.last_transitions, 2u);
    EXPECT_EQ(work.edge_visits, 8u);
    // The old immutable graph and its exact reference counts remain intact.
    EXPECT_EQ(references(accepted, left), 1u);
    EXPECT_EQ(references(accepted, first), 1u);
}
TEST(GeometryOwnerGraph, ConflictingControlsAndSizesRejectBeforeAcceptedManifestChanges)
{
    auto owner = std::make_shared<const FixtureNode>();
    std::size_t used = 0;
    Editor initial({}, 0, used, graph_limit);
    edge(initial, owner, true);
    const auto accepted = initial.finish();
    auto foreign = std::shared_ptr<const FixtureNode>(owner.get(), [](const FixtureNode*) {});
    Editor wrong_control(accepted, 0, used, graph_limit);
    EXPECT_THROW((void) wrong_control.external(foreign, sizeof(FixtureNode), true), P::GeometryError);
    Editor wrong_size(accepted, 0, used, graph_limit);
    EXPECT_THROW((void) wrong_size.external(owner, sizeof(FixtureNode) + 1, true), P::GeometryError);
    EXPECT_EQ(accepted.owner_count(), 1u);
    EXPECT_EQ(accepted.payload_bytes(), sizeof(FixtureNode));
    EXPECT_EQ(references(accepted, owner), 1u);
}
TEST(GeometryOwnerGraph, ExactBudgetAndOneByteShortKeepAcceptedRootAndSpentPrefix)
{
    auto leaf = std::make_shared<const FixtureNode>();
    auto first = std::make_shared<const FixtureNode>(FixtureNode{leaf, {}});
    auto next = std::make_shared<const FixtureNode>(FixtureNode{leaf, leaf});
    std::size_t start = 0;
    Editor initial({}, 0, start, graph_limit);
    edge(initial, first, true);
    const auto accepted = initial.finish();
    const auto apply = [&](std::size_t& used, std::size_t cap)
    {
        Editor edit(accepted, 0, used, cap);
        edge(edit, next, true);
        edge(edit, first, false);
        return edit.finish();
    };
    std::size_t measured = 37;
    const auto expected = apply(measured, graph_limit);
    std::size_t exact = 37;
    const auto completed = apply(exact, measured);
    EXPECT_EQ(exact, measured);
    EXPECT_EQ(completed.payload_bytes(), expected.payload_bytes());
    std::size_t short_used = 37;
    EXPECT_THROW((void) apply(short_used, measured - 1), P::GeometryError);
    EXPECT_GE(short_used, 37u);
    EXPECT_LE(short_used, measured - 1);
    EXPECT_EQ(references(accepted, first), 1u);
    EXPECT_EQ(references(accepted, leaf), 1u);
    std::size_t retry = 37;
    EXPECT_EQ(apply(retry, measured).payload_bytes(), expected.payload_bytes());
}

TEST(GeometryOwnerGraph, ArithmeticChannelsKeepIndependentMetadataHolds)
{
    auto shared = std::make_shared<const FixtureNode>();
    std::size_t used = 0;
    Editor ordinary({}, 0, used, graph_limit);
    Editor bounded({}, 1, used, graph_limit);
    edge(ordinary, shared, true);
    edge(bounded, shared, true);
    const auto ordinary_manifest = ordinary.finish();
    const auto bounded_manifest = bounded.finish();
    EXPECT_EQ(references(ordinary_manifest, shared, 0), 1u);
    EXPECT_EQ(references(ordinary_manifest, shared, 1), 0u);
    EXPECT_EQ(references(bounded_manifest, shared, 1), 1u);
    EXPECT_EQ(references(bounded_manifest, shared, 0), 0u);
}

TEST(GeometryOwnerGraph, RejectedEditorAndLedgerCannotResumeAndAcceptedRootIsUnchanged)
{
    auto owner = std::make_shared<const FixtureNode>();
    std::size_t used = 0;
    Editor initial({}, 0, used, graph_limit);
    edge(initial, owner, true);
    const auto accepted = initial.finish();
    P::GeometryAllocationLedger ledger;
    Editor changed(accepted, 0, used, graph_limit, &ledger);
    EXPECT_THROW((void) changed.external(owner, sizeof(FixtureNode) + 1, true), P::GeometryError);
    EXPECT_THROW((void) changed.external(owner, sizeof(FixtureNode), true), P::GeometryError);
    EXPECT_THROW((void) changed.finish(), P::GeometryError);
    EXPECT_THROW(ledger.retain_external_owner(owner, sizeof(FixtureNode), used, graph_limit), P::GeometryError);
    EXPECT_EQ(references(accepted, owner), 1u);
}

TEST(GeometryOwnerGraph, ManifestRecordContributesOnePhysicalEdgeRegardlessOfPayloadMultiplicity)
{
    auto owner = std::make_shared<const FixtureNode>();
    std::size_t used = 0;
    const G::OwnerRecord row{{0, Editor::external_kind(), owner.get()}, owner, sizeof(FixtureNode), 99};
    Editor editor({}, 0, used, graph_limit);
    EXPECT_TRUE(editor.owner(row, true));
    EXPECT_FALSE(editor.owner(row, true));
    const auto result = editor.finish();
    EXPECT_EQ(references(result, owner), 2u);
    EXPECT_EQ(result.payload_bytes(), sizeof(FixtureNode));
    EXPECT_THROW((void) editor.finish(), P::GeometryError);
    Editor foreign(result, 0, used, graph_limit);
    auto mismatched = row;
    mismatched.key.channel = 1;
    EXPECT_THROW((void) foreign.owner(mismatched, true), P::GeometryError);
}

TEST(GeometryOwnerGraph, ImmutableManifestBranchAllocatesOnlyItsTwoChildSlots)
{
    std::size_t used = 0;
    Editor editor({}, 0, used, graph_limit);
    constexpr auto owners = G::leaf_capacity + 1;
    for (std::size_t i = 0; i != owners; ++i)
        editor.external(std::make_shared<const FixtureNode>(), sizeof(FixtureNode), true);
    const auto root = editor.finish();
    ASSERT_EQ(root.owner_count(), owners);
    std::size_t child_buffers = 0, payload = 0;
    root.visit_payload([&](const void*, std::size_t bytes)
    {
        // The head, node headers and both record buffers are larger than
        // the two-child buffer. No physical body owners enter this census.
        child_buffers += bytes == 2 * sizeof(std::shared_ptr<const void>);
        payload += bytes;
    });
    EXPECT_EQ(child_buffers, 1u);
    EXPECT_EQ(payload, root.metadata_payload_bytes());
}

TEST(GeometryOwnerGraph, PagedEditFoldsCancelledEdgesAndSharesDistantLeaves)
{
    constexpr auto changed = 2 * G::leaf_capacity;
    std::vector<std::shared_ptr<const FixtureNode>> owners(2 * changed), cancelled(changed);
    for (auto* rows : {&owners, &cancelled})
        for (auto& owner : *rows)
            owner = std::make_shared<const FixtureNode>();
    std::sort(owners.begin(), owners.end(), [](const auto& a, const auto& b)
        { return std::less<const void*>{}(a.get(), b.get()); });
    std::size_t startup = 0;
    Editor initial({}, 0, startup, graph_limit);
    for (const auto& owner : owners)
        edge(initial, owner, true);
    const auto accepted = initial.finish();
    const auto leaves = [](const auto& manifest)
    {
        std::map<const void*, const void*> result;
        manifest.visit_borrowed_dependencies({nullptr, [](void*, const void*, const void*, std::size_t) { return true; }},
            {&result, [](void* raw, const void* page, const G::OwnerRecord& row)
                { static_cast<decltype(result)*>(raw)->emplace(row.owner.get(), page); }});
        return result;
    };
    const auto before = leaves(accepted);
    const auto update = [&](std::size_t& used, std::size_t cap, P::details::GeometryOwnerGraphWork* work = nullptr)
    {
        Editor edit(accepted, 0, used, cap, nullptr, work);
        for (std::size_t i = 0; i != changed; ++i)
        {
            const auto slot = (37 * i) % changed;
            edge(edit, cancelled[i], true);
            edge(edit, owners[slot], slot % 7 != 0);
        }
        for (const auto& owner : cancelled)
            edge(edit, owner, false);
        return edit.finish();
    };
    constexpr std::size_t prefix = 37;
    std::size_t measured = prefix;
    P::details::GeometryOwnerGraphWork work;
    const auto second = update(measured, graph_limit, &work);
    EXPECT_EQ(work.dependency_entries, 2 * changed);
    EXPECT_EQ(work.edge_visits, 3 * changed);
    const auto after = leaves(second);
    std::size_t physical = 0, edges = 0;
    for (std::size_t i = 0; i != owners.size(); ++i)
    {
        const auto count = i >= changed ? 1u : i % 7 == 0 ? 0u : 2u;
        EXPECT_EQ(references(second, owners[i]), count);
        EXPECT_EQ(references(accepted, owners[i]), 1u);
        physical += count != 0;
        edges += count;
        if (i >= changed)
            EXPECT_EQ(before.at(owners[i].get()), after.at(owners[i].get()));
    }
    EXPECT_EQ(second.owner_count(), physical);
    EXPECT_EQ(second.physical_edge_count(), edges);
    EXPECT_EQ(second.payload_bytes(), physical * sizeof(FixtureNode));
    for (const auto& owner : cancelled)
        EXPECT_EQ(references(second, owner), 0u);

    std::size_t short_used = prefix;
    EXPECT_THROW((void) update(short_used, measured - 1), P::GeometryError);
    EXPECT_GT(short_used, prefix);
    EXPECT_EQ(leaves(accepted), before);
    std::size_t exact = prefix;
    EXPECT_EQ(update(exact, measured).physical_edge_count(), edges);
    EXPECT_EQ(exact, measured);

    // Wide cancelling edits still pay for their physical scratch while the
    // third generation keeps the accepted index and every retained control.
    Editor no_change(second, 0, exact, graph_limit);
    for (const auto& owner : cancelled)
    {
        edge(no_change, owner, true);
        edge(no_change, owner, false);
    }
    EXPECT_TRUE(no_change.finish().same_root(second));
    EXPECT_GT(exact, measured);
}

TEST(GeometryOwnerGraph, SortedBatchReservationCoversPageSplitsAndRedistribution)
{
    // Fill more than one branch, then leave roughly a quarter of each leaf.
    // This exercises intermediate merge allocations as well as the surviving pages.
    std::vector<G::OwnerRecord> changes(2 * G::branch_capacity * G::leaf_capacity + 1);
    for (auto& row : changes)
    {
        auto owner = std::make_shared<const char>('x');
        row = {{0, Editor::external_kind(), owner.get()}, owner, sizeof(char), 1};
    }
    std::sort(changes.begin(), changes.end(),
        [](const auto& a, const auto& b) { return G::OwnerKeyLess{}(a.key, b.key); });

    const auto apply = [&](const G::GeometryOwnerManifest& before)
    {
        struct Count
        {
            std::size_t bytes = 0, owners = 0;
        } count;
        const G::Admission admission{&count,
            [](void* raw, std::size_t bytes) { static_cast<Count*>(raw)->bytes += bytes; },
            [](void* raw, const auto&, std::size_t, std::size_t) { ++static_cast<Count*>(raw)->owners; }};
        const auto result = before.edited(changes, admission);
        EXPECT_GT(count.bytes, 0u);
        EXPECT_GT(count.owners, G::branch_capacity);
        EXPECT_LE(count.bytes, before.batch_edit_byte_bound(changes.size()));
        EXPECT_LE(count.owners, before.batch_edit_metadata_owner_bound(changes.size()));
        return result;
    };
    const auto full = apply({});
    ASSERT_EQ(full.owner_count(), changes.size());
    for (std::size_t i = 0; i != changes.size(); ++i)
        changes[i].references = i % 4 == 0;
    const auto sparse = apply(full);
    EXPECT_EQ(sparse.owner_count(), (changes.size() + 3) / 4);
    EXPECT_EQ(full.owner_count(), changes.size());
}
