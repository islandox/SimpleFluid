/** Independent physical ownership and transactional union/import regressions. */
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/unitTests/adaptive_storage_test_helpers.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
namespace
{
namespace P = SimpleFluid::PLIC;
namespace D = P::details;
namespace G = D::geometry_demand;
namespace A = SimpleFluid::Meshes::AMR;
using SimpleFluid::ArithmeticMode;
using Snapshot = A::AdaptiveGeometrySnapshot;
using Hierarchy = A::DistributedAdaptiveHierarchy;
constexpr std::size_t cap = 128 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
auto source()
{
    A::AMRGrid grid;
    grid.block_cells = {4, 4, 2};
    A::CartesianAMR native(grid);
    std::vector<std::uint64_t> ids(native.geometry()->num_cells());
    std::iota(ids.begin(), ids.end(), 0);
    auto input = native.detach_owned(ids, {'u', 'n', 'i', 'o', 'n'});
    return std::make_shared<const Hierarchy>(
        std::move(input.owned), std::move(input.families), std::move(input.kernel), Tpetra::getDefaultComm());
}
auto acquire(const Hierarchy& hierarchy, std::span<const A::AdaptiveLeaf* const> selected)
{
    std::size_t used = 37;
    return hierarchy.geometry_snapshot(
        selected, ArithmeticMode::Ordinary, A::GeometrySnapshotPurpose::State, used, cap);
}
struct Physical
{
    std::shared_ptr<const void> owner;
    std::size_t bytes;
};
using Key = std::pair<std::uint8_t, const void*>;
using Owners = std::map<Key, Physical>;
void append(Owners& rows, std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    auto [at, inserted] = rows.emplace(Key{kind, owner.get()}, Physical{owner, bytes});
    if (!inserted)
    {
        EXPECT_EQ(at->second.bytes, bytes);
        EXPECT_FALSE(owner.owner_before(at->second.owner));
        EXPECT_FALSE(at->second.owner.owner_before(owner));
    }
}
Owners census(std::span<const std::shared_ptr<const Snapshot>> roots)
{
    Owners rows;
    for (const auto& root : roots)
    {
        P::GeometryAllocationLedger ordinary, bounded;
        std::size_t used = 0;
        root->retain(ordinary, bounded, used, cap);
        // Export controls from a separate full retain, never the maintained union.
        const G::Admission allocation{
            &used, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; }};
        for (unsigned channel = 0; channel < 2; ++channel)
            D::GeometryOwnerImportAccess::export_manifest(channel ? bounded : ordinary, allocation, nullptr, channel)
                .visit([&](const G::OwnerRecord& row) { append(rows, row.key.kind, row.owner, row.bytes); });
    }
    return rows;
}
void matches(const D::GeometryOwnerUnion& value, std::span<const std::shared_ptr<const Snapshot>> roots)
{
    const auto expected = census(roots);
    Owners actual;
    for (const auto& root : value.manifests().owners)
        root.visit([&](const auto& row) { append(actual, row.key.kind, row.owner, row.bytes); });
    ASSERT_EQ(actual.size(), expected.size());
    std::size_t payload = 0;
    for (const auto& [key, row] : expected)
    {
        auto found = actual.find(key);
        ASSERT_NE(found, actual.end());
        EXPECT_EQ(found->second.bytes, row.bytes);
        EXPECT_FALSE(found->second.owner.owner_before(row.owner));
        EXPECT_FALSE(row.owner.owner_before(found->second.owner));
        payload += row.bytes;
    }
    EXPECT_EQ(value.manifests().payload_bytes(), payload);
    std::map<const void*, Physical> controls;
    for (const auto& root : roots)
        A::adaptive_detail::GeometrySnapshotAccess::visit_owner_manifest_metadata(*root,
            {&controls, [](void* raw, const std::shared_ptr<const void>& owner, std::size_t bytes, std::size_t admitted)
                {
                    auto& rows = *static_cast<decltype(controls)*>(raw);
                    auto [at, added] = rows.emplace(owner.get(), Physical{owner, admitted - bytes});
                    if (!added)
                        EXPECT_EQ(at->second.bytes, admitted - bytes);
                    return true;
                }});
    std::size_t held_controls = 0;
    for (const auto& [key, row] : controls)
        held_controls += row.bytes;
    EXPECT_EQ(value.stored_index_control_bytes(), held_controls);
    std::map<const void*, Physical> administration;
    value.visit_administration(
        {&administration, [](void* raw, const std::shared_ptr<const void>& owner, std::size_t, std::size_t admitted)
            {
                auto& rows = *static_cast<decltype(administration)*>(raw);
                EXPECT_TRUE(rows.emplace(owner.get(), Physical{owner, admitted}).second);
                return true;
            }});
    std::size_t index = 0;
    for (const auto& [key, row] : administration)
        index += row.bytes;
    EXPECT_EQ(value.resident_bytes(), payload + held_controls + index);
}
} // namespace
TEST(GeometryOwnerUnion, OverlappingRootsReplaceRemoveAndEmptyMatchIndependentPhysicalCensus)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP();
    auto hierarchy = source();
    std::vector<const A::AdaptiveLeaf*> rows;
    for (const auto& row : hierarchy->owned_leaves())
        rows.push_back(&row);
    auto first = acquire(*hierarchy, std::span(rows).first(2));
    auto second = acquire(*hierarchy, std::span(rows).subspan(2, 2));
    auto a = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(first),
         b = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(second);
    D::GeometryOwnerUnion value;
    P::GeometryAllocationLedger ordinary, bounded;
    std::size_t used = 37;
    value = value.staged(nullptr, &a, ordinary, bounded, used, cap);
    matches(value, std::array{first});
    D::GeometryAllocationOwnerAccess::reset_work_statistics();
    value = value.staged(nullptr, &b, ordinary, bounded, used, cap);
    matches(value, std::array{first, second});
    value = value.staged(&a, nullptr, ordinary, bounded, used, cap);
    matches(value, std::array{second});
    const auto work = D::GeometryAllocationOwnerAccess::work_statistics();
    EXPECT_EQ(work.import_fallbacks, 0u);
    EXPECT_EQ(work.metadata_credit_seed_rows, 0u);
    value = value.staged(&b, nullptr, ordinary, bounded, used, cap);
    EXPECT_EQ(value.resident_bytes(), 0u);
}
TEST(GeometryOwnerUnion, ExactCapOneByteShortPoisonRetryKeepAcceptedValueAndControls)
{
    auto hierarchy = source();
    std::vector<const A::AdaptiveLeaf*> rows;
    for (const auto& row : hierarchy->owned_leaves())
        rows.push_back(&row);
    auto first = acquire(*hierarchy, std::span(rows).first(2));
    auto second = acquire(*hierarchy, std::span(rows).subspan(2, 2));
    auto a = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(first),
         b = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(second);
    P::GeometryAllocationLedger setup0, setup1;
    std::size_t setup = 0;
    const auto accepted = D::GeometryOwnerUnion{}.staged(nullptr, &a, setup0, setup1, setup, cap);
    const auto apply = [&](P::GeometryAllocationLedger& zero, P::GeometryAllocationLedger& one, std::size_t& used,
                           std::size_t limit) { return accepted.staged(&a, &b, zero, one, used, limit); };
    P::GeometryAllocationLedger measure0, measure1;
    std::size_t needed = 37;
    const auto measured = apply(measure0, measure1, needed, cap);
    P::GeometryAllocationLedger exact0, exact1;
    std::size_t exact = 37;
    matches(apply(exact0, exact1, exact, needed), std::array{second});
    EXPECT_EQ(exact, needed);
    P::GeometryAllocationLedger short0, short1;
    std::size_t spent = 37;
    EXPECT_THROW((void) apply(short0, short1, spent, needed - 1), P::GeometryError);
    EXPECT_GE(spent, 37u);
    EXPECT_LE(spent, needed - 1);
    matches(accepted, std::array{first});
    const auto owner = std::make_shared<const int>(1);
    EXPECT_THROW(short0.retain_external_owner(owner, sizeof(int), spent, cap), P::GeometryError);
    P::GeometryAllocationLedger retry0, retry1;
    std::size_t retry = 37;
    matches(apply(retry0, retry1, retry, needed), std::array{second});
    EXPECT_EQ(retry, needed);
}
TEST(GeometryOwnerUnion, TwoRootReplacementMatchesIndependentCensusAndRejectsShortCapAtomically)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1);
    const auto hierarchy = source();
    std::vector<const A::AdaptiveLeaf*> rows;
    for (const auto& row : hierarchy->owned_leaves()) rows.push_back(&row);
    const std::array old_roots{acquire(*hierarchy, std::array{rows[0], rows[1]}),
        acquire(*hierarchy, std::array{rows[2], rows[3]})};
    const std::array new_roots{acquire(*hierarchy, std::array{rows[2], rows[4]}),
        acquire(*hierarchy, std::array{rows[0], rows[5]})};
    using Access = A::adaptive_detail::GeometrySnapshotAccess;
    const auto a = Access::owner_binding(old_roots[0]), b = Access::owner_binding(old_roots[1]);
    const auto c = Access::owner_binding(new_roots[0]), d = Access::owner_binding(new_roots[1]);
    const std::array install{D::GeometryOwnerUnion::Replacement{nullptr, &a},
        D::GeometryOwnerUnion::Replacement{nullptr, &b}};
    const std::array replace{D::GeometryOwnerUnion::Replacement{&a, &c},
        D::GeometryOwnerUnion::Replacement{&b, &d}};
    P::GeometryAllocationLedger setup0, setup1;
    std::size_t setup = 0;
    const auto accepted = D::GeometryOwnerUnion{}.staged(install, setup0, setup1, setup, cap);
    matches(accepted, old_roots);

    P::GeometryAllocationLedger measure0, measure1;
    std::size_t needed = 0;
    const auto next = accepted.staged(replace, measure0, measure1, needed, cap);
    matches(next, new_roots);
    EXPECT_EQ(next.generation(), accepted.generation() + 1);
    P::GeometryAllocationLedger short0, short1;
    std::size_t short_used = 0;
    ASSERT_GT(needed, 0u);
    EXPECT_THROW((void) accepted.staged(replace, short0, short1, short_used, needed - 1), P::GeometryError);
    matches(accepted, old_roots);
    P::GeometryAllocationLedger exact0, exact1;
    std::size_t exact = 0;
    matches(accepted.staged(replace, exact0, exact1, exact, needed), new_roots);
    EXPECT_EQ(exact, needed);
}
TEST(GeometryOwnerUnion, PersistentSubsetNeverPinsUnrelatedSourceRootOrIssuerUnion)
{
    auto one = source(), two = source();
    const auto* row1 = &*one->owned_leaves().begin();
    const auto* row2 = &*two->owned_leaves().begin();
    auto first = acquire(*one, std::array{row1}), second = acquire(*two, std::array{row2});
    auto a = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(first),
         b = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(second);
    std::weak_ptr<const Snapshot> unwanted = first;
    std::weak_ptr<const Hierarchy> source_owner = one;
    P::GeometryAllocationLedger op0, op1;
    std::size_t used = 0;
    auto value = D::GeometryOwnerUnion{}.staged(nullptr, &a, op0, op1, used, cap);
    value = value.staged(nullptr, &b, op0, op1, used, cap);
    P::GeometryAllocationLedger persistent0, persistent1;
    std::size_t persistent = 37;
    value.import_subset(b, persistent0, persistent1, persistent, cap);
    const auto before = persistent;
    value.import_subset(b, persistent0, persistent1, persistent, cap);
    EXPECT_EQ(persistent, before);
    value = {};
    op0 = {};
    op1 = {};
    a = {};
    first.reset();
    one.reset();
    EXPECT_TRUE(unwanted.expired());
    EXPECT_TRUE(source_owner.expired());
    EXPECT_GT(persistent0.owner_count(), 0u);
}
TEST(GeometryOwnerUnion, SingleChannelPairRejectsChangedRootOrWrongChannelBeforeAdmission)
{
    auto hierarchy = source();
    std::vector<const A::AdaptiveLeaf*> rows;
    for (const auto& row : hierarchy->owned_leaves())
        rows.push_back(&row);
    const auto first = acquire(*hierarchy, std::array{rows[0]}), second = acquire(*hierarchy, std::array{rows[1]});
    auto forged = A::adaptive_detail::GeometrySnapshotAccess::owner_pair(*first);
    forged.owners[0] = A::adaptive_detail::GeometrySnapshotAccess::owner_pair(*second).owners[0];
    P::GeometryAllocationLedger zero, one;
    std::size_t used = 37;
    EXPECT_THROW(D::GeometryOwnerImportAccess::import_pair(zero, one, forged, used, cap), P::GeometryError);
    EXPECT_EQ(used, 37u);
    auto wrong = A::adaptive_detail::GeometrySnapshotAccess::owner_pair(*first);
    wrong.owners[1] = wrong.owners[0];
    wrong.owners[0] = {};
    P::GeometryAllocationLedger other0, other1;
    used = 37;
    EXPECT_THROW(D::GeometryOwnerImportAccess::import_pair(other0, other1, wrong, used, cap), P::GeometryError);
    EXPECT_EQ(used, 37u);
}
TEST(GeometryOwnerUnion, PairedNeutralCreditChargesPayloadOnceAndSurvivesPeerExpiry)
{
    const auto neutral = std::make_shared<const std::array<char, 256>>();
    std::weak_ptr<const void> weak = neutral;
    P::GeometryAllocationLedger zero, one;
    std::size_t used = 37;
    {
        D::GeometryOwnerAdmissionPair pair(zero, one);
        zero.retain_external_owner(neutral, sizeof(*neutral), used, cap);
        one.retain_external_owner(neutral, sizeof(*neutral), used, cap);
    }
    EXPECT_EQ(zero.payload_bytes(), sizeof(*neutral));
    EXPECT_EQ(one.payload_bytes(), sizeof(*neutral));
    EXPECT_EQ(used - 37, sizeof(*neutral) + zero.bookkeeping_bytes() + one.bookkeeping_bytes());
    zero = {};
    EXPECT_FALSE(weak.expired());
    const auto before = used;
    one.retain_external_owner(neutral, sizeof(*neutral), used, cap);
    EXPECT_EQ(used, before);
}
TEST(GeometryOwnerUnion, FullInventoryReplacementStagesAddBeforeRetireWithoutLosingAcceptedOwners)
{
    auto hierarchy = source();
    std::vector<const A::AdaptiveLeaf*> rows;
    for (const auto& row : hierarchy->owned_leaves())
        rows.push_back(&row);
    std::vector<std::shared_ptr<const Snapshot>> roots;
    std::vector<D::GeometryOwnerRootBinding> bindings;
    D::GeometryOwnerUnion value;
    P::GeometryAllocationLedger zero, one;
    std::size_t used = 0;
    for (std::size_t i = 0; i < 20; ++i)
    {
        roots.push_back(acquire(*hierarchy, std::array{rows[i]}));
        bindings.push_back(A::adaptive_detail::GeometrySnapshotAccess::owner_binding(roots.back()));
        value = value.staged(nullptr, &bindings.back(), zero, one, used, cap);
    }
    const auto before = value;
    auto next = acquire(*hierarchy, std::array{rows[20]});
    auto replacement = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(next);
    value = value.staged(&bindings.front(), &replacement, zero, one, used, cap);
    roots.front() = next;
    matches(value, roots);
    EXPECT_GT(before.resident_bytes(), 0u);
    EXPECT_NE(value.generation(), before.generation());
}

TEST(GeometryOwnerUnion, PairedGraphOverlapCountsSharedPhysicalOwnerOnceAcrossChannels)
{
    using Editor = D::GeometryOwnerGraphEditor;
    const auto shared = std::make_shared<const std::array<char, 256>>();
    std::size_t used = 0, overlap = 0;
    Editor zero({}, 0, used, cap), one({}, 1, used, cap);
    zero.pair_with(one, overlap);
    EXPECT_TRUE(zero.external(shared, sizeof(*shared), true));
    EXPECT_EQ(overlap, 0u);
    EXPECT_FALSE(zero.external(shared, sizeof(*shared), true));
    EXPECT_EQ(overlap, 0u);
    EXPECT_TRUE(one.external(shared, sizeof(*shared), true));
    EXPECT_EQ(overlap, sizeof(*shared));
    EXPECT_FALSE(zero.external(shared, sizeof(*shared), false));
    EXPECT_EQ(overlap, sizeof(*shared));
    const auto first = zero.finish(), second = one.finish();
    EXPECT_EQ(first.payload_bytes() + second.payload_bytes() - overlap, sizeof(*shared));
    EXPECT_EQ(first.references(0, Editor::external_kind(), shared), 1u);
    EXPECT_EQ(second.references(1, Editor::external_kind(), shared), 1u);
    std::size_t later = overlap;
    Editor next0(first, 0, used, cap), next1(second, 1, used, cap);
    next0.pair_with(next1, later);
    EXPECT_TRUE(next1.external(shared, sizeof(*shared), false));
    EXPECT_EQ(later, 0u);
    EXPECT_EQ(next1.finish().owner_count(), 0u);
    EXPECT_EQ(next0.finish().owner_count(), 1u);
}

TEST(GeometryOwnerUnion, LookalikeDeltaRootRejectsBeforeSpendingAndPoisonsTrialLedger)
{
    auto hierarchy = source();
    const auto* leaf = &*hierarchy->owned_leaves().begin();
    auto snapshot = acquire(*hierarchy, std::array{leaf});
    const auto root = A::adaptive_detail::GeometrySnapshotAccess::owner_pair(*snapshot).owners[0];
    std::vector<G::OwnerRecord> rows;
    root.visit([&](const auto& row) { rows.push_back(row); });
    std::size_t scratch = 0;
    const G::Admission allocation{
        &scratch, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; }};
    const auto lookalike = G::GeometryOwnerManifest::built(rows, allocation);
    ASSERT_FALSE(root.same_root(lookalike));
    P::GeometryAllocationLedger ledger;
    std::size_t used = 37;
    D::GeometryOwnerImportAccess::import(ledger, root, used, cap);
    const auto before = used;
    EXPECT_THROW(D::GeometryOwnerImportAccess::import_derived(ledger, lookalike, root, used, cap), P::GeometryError);
    EXPECT_EQ(used, before);
    EXPECT_THROW(ledger.retain_external_owner(snapshot, sizeof(*snapshot), used, cap), P::GeometryError);
}
TEST(GeometryOwnerUnion, UnrelatedPopulatedPairUsesExplicitFallbackAndKeepsExactUnion)
{
    auto hierarchy = source();
    const auto* leaf = &*hierarchy->owned_leaves().begin();
    auto snapshot = acquire(*hierarchy, std::array{leaf});
    auto binding = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(snapshot);
    P::GeometryAllocationLedger setup0, setup1;
    std::size_t setup = 0;
    const auto value = D::GeometryOwnerUnion{}.staged(nullptr, &binding, setup0, setup1, setup, cap);
    P::GeometryAllocationLedger zero, one;
    std::size_t used = 37;
    const auto other = std::make_shared<const std::array<char, 19>>();
    zero.retain_external_owner(other, sizeof(*other), used, cap);
    D::GeometryAllocationOwnerAccess::reset_work_statistics();
    value.import_subset(binding, zero, one, used, cap);
    EXPECT_GT(D::GeometryAllocationOwnerAccess::work_statistics().import_fallbacks, 0u);
    const auto before = used;
    value.import_subset(binding, zero, one, used, cap);
    EXPECT_EQ(used, before);
    EXPECT_GT(zero.payload_bytes(), sizeof(*other));
}
TEST(GeometryOwnerUnion, DualSnapshotOverlapSealAgreesWithIndependentOwnerControlIntersection)
{
    namespace F = SimpleFluid::test::adaptive_storage;
    auto input = F::make_disconnected(F::Provider::Polyhedral, 1);
    auto hierarchy = std::make_shared<const Hierarchy>(
        std::move(input.owned), std::vector<A::AdaptiveFamily>{}, std::move(input.kernel), Tpetra::getDefaultComm());
    std::vector<const A::AdaptiveLeaf*> selected;
    for (const auto& leaf : hierarchy->owned_leaves())
        selected.push_back(&leaf);
    if (selected.size() > 2)
        selected.resize(2);
    std::size_t built = 37;
    auto snapshot = hierarchy->geometry_snapshot(
        selected, ArithmeticMode::BoundedCertification, A::GeometrySnapshotPurpose::State, built, cap);
    const auto pair = A::adaptive_detail::GeometrySnapshotAccess::owner_pair(*snapshot);
    Owners zero, one;
    pair.owners[0].visit([&](const auto& row) { append(zero, row.key.kind, row.owner, row.bytes); });
    pair.owners[1].visit([&](const auto& row) { append(one, row.key.kind, row.owner, row.bytes); });
    std::size_t common = 0;
    for (const auto& [key, row] : zero)
        if (const auto found = one.find(key); found != one.end())
        {
            EXPECT_EQ(found->second.bytes, row.bytes);
            EXPECT_FALSE(found->second.owner.owner_before(row.owner));
            EXPECT_FALSE(row.owner.owner_before(found->second.owner));
            common += row.bytes;
        }
    EXPECT_EQ(pair.overlap_bytes(), common);
    P::GeometryAllocationLedger imported0, imported1;
    std::size_t used = 37;
    D::GeometryOwnerImportAccess::import_pair(imported0, imported1, pair, used, cap);
    EXPECT_EQ(imported0.payload_bytes() + imported1.payload_bytes() - common, pair.payload_bytes());
    EXPECT_EQ(used - 37, pair.payload_bytes() + imported0.bookkeeping_bytes() + imported1.bookkeeping_bytes());
}

namespace union_work_oracle
{
struct NodeKey
{
    unsigned channel, kind;
    const void* address;
    friend bool operator==(const NodeKey&, const NodeKey&) = default;
    friend bool operator<(const NodeKey& a, const NodeKey& b) noexcept
    {
        if (a.channel != b.channel)
            return a.channel < b.channel;
        if (a.kind != b.kind)
            return a.kind < b.kind;
        return std::less<const void*>{}(a.address, b.address);
    }
};
NodeKey key(unsigned channel, unsigned kind, const void* address)
{
    return {channel, kind, address};
}
NodeKey metadata(unsigned channel, const void* address)
{
    return key(channel, D::GeometryOwnerGraphEditor::external_kind(), address);
}
struct Node
{
    std::shared_ptr<const void> control;
    std::size_t bytes = 0, incoming = 0;
    std::vector<NodeKey> children;
};
using Nodes = std::map<NodeKey, Node>;
struct Graph
{
    Nodes nodes;
    std::vector<NodeKey> roots;
};
void control(Nodes& nodes, NodeKey id, const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    ASSERT_EQ(id.address, owner.get());
    const auto [at, inserted] = nodes.emplace(id, Node{owner, bytes});
    if (!inserted)
    {
        EXPECT_EQ(at->second.bytes, bytes);
        EXPECT_FALSE(owner.owner_before(at->second.control));
        EXPECT_FALSE(at->second.control.owner_before(owner));
    }
}
/** The shared census authenticates controls; the separate borrowed census below
 * describes actual pointer slots and never calls an editor or its delta APIs. */
Graph graph(std::span<const D::GeometryOwnerRootBinding> bindings)
{
    Graph result;
    for (const auto& binding : bindings)
    {
        control(result.nodes, metadata(0, binding.header().get()), binding.header(), sizeof(Snapshot));
        for (unsigned channel = 0; channel < 2; ++channel)
        {
            struct Context
            {
                Nodes& nodes;
                unsigned channel;
            } context{result.nodes, channel};
            binding.payload().owners[channel].visit_dependencies(
                {&context,
                    [](void* raw, const std::shared_ptr<const void>& owner, std::size_t bytes, std::size_t)
                    {
                        auto& c = *static_cast<Context*>(raw);
                        control(c.nodes, metadata(c.channel, owner.get()), owner, bytes);
                        return true;
                    }},
                {&context, [](void* raw, const G::OwnerRecord& row)
                    {
                        auto& c = *static_cast<Context*>(raw);
                        EXPECT_EQ(row.key.channel, c.channel);
                        control(c.nodes, key(c.channel, row.key.kind, row.key.address), row.owner, row.bytes);
                    }});
        }
    }
    std::set<NodeKey> expanded;
    const auto edge = [&](std::optional<NodeKey> parent, NodeKey destination)
    {
        auto found = result.nodes.find(destination);
        EXPECT_NE(found, result.nodes.end());
        if (found == result.nodes.end())
            return false;
        ++found->second.incoming; // Every physical slot, including duplicate holds.
        if (parent)
            result.nodes.at(*parent).children.push_back(destination);
        else
            result.roots.push_back(destination);
        return expanded.insert(destination).second;
    };
    for (const auto& binding : bindings)
    {
        (void) edge(
            std::nullopt, metadata(0, binding.header().get())); // Header has no children in this union protocol.
        for (unsigned channel = 0; channel < 2; ++channel)
        {
            struct Context
            {
                const decltype(edge)& emit;
                unsigned channel;
            } context{edge, channel};
            binding.payload().owners[channel].visit_borrowed_dependencies(
                {&context,
                    [](void* raw, const void* owner, const void* parent, std::size_t)
                    {
                        auto& c = *static_cast<Context*>(raw);
                        const auto from = parent ? std::optional(metadata(c.channel, parent)) : std::nullopt;
                        return c.emit(from, metadata(c.channel, owner));
                    }},
                {&context, [](void* raw, const void* parent, const G::OwnerRecord& row)
                    {
                        auto& c = *static_cast<Context*>(raw);
                        EXPECT_EQ(row.key.channel, c.channel);
                        // The row stores one shared_ptr, irrespective of payload multiplicity.
                        (void) c.emit(metadata(c.channel, parent), key(c.channel, row.key.kind, row.owner.get()));
                    }});
        }
    }
    for (const auto& [id, node] : result.nodes)
        EXPECT_GT(node.incoming, 0u);
    return result;
}
struct Frontier
{
    std::array<std::size_t, 2> first{}, last{}, edges{}, entries{};
    std::array<std::set<NodeKey>, 2> changed;
};
Frontier frontier(const Graph& old, const Graph& next, const Graph& retired, const Graph& retained)
{
    Frontier result;
    std::array<std::set<NodeKey>, 2> destinations;
    const auto visit = [&](NodeKey id)
    {
        ++result.edges[id.channel];
        destinations[id.channel].insert(id);
    };
    for (const auto& id : retained.roots)
        visit(id);
    for (const auto& id : retired.roots)
        visit(id);
    for (const auto& [id, node] : next.nodes)
    {
        const auto previous = old.nodes.find(id);
        if (previous == old.nodes.end())
        {
            ++result.first[id.channel];
            for (const auto& child : node.children)
                visit(child);
        }
        if (previous == old.nodes.end() || previous->second.incoming != node.incoming)
            result.changed[id.channel].insert(id);
    }
    for (const auto& [id, node] : old.nodes)
    {
        if (!next.nodes.contains(id))
        {
            ++result.last[id.channel];
            result.changed[id.channel].insert(id);
            for (const auto& child : node.children)
                visit(child);
        }
    }
    for (unsigned channel = 0; channel < 2; ++channel)
        result.entries[channel] = destinations[channel].size();
    return result;
}
void exact_rows(const D::GeometryOwnerUnion& value, const Graph& expected)
{
    Nodes actual;
    for (const auto& manifest : value.manifests().owners)
        manifest.visit(
            [&](const G::OwnerRecord& row)
            {
                const auto id = key(row.key.channel, row.key.kind, row.key.address);
                EXPECT_TRUE(actual.emplace(id, Node{row.owner, row.bytes, row.references}).second);
            });
    ASSERT_EQ(actual.size(), expected.nodes.size());
    for (const auto& [id, node] : expected.nodes)
    {
        const auto found = actual.find(id);
        ASSERT_NE(found, actual.end());
        EXPECT_EQ(found->second.bytes, node.bytes);
        EXPECT_EQ(found->second.incoming, node.incoming);
        EXPECT_FALSE(found->second.control.owner_before(node.control));
        EXPECT_FALSE(node.control.owner_before(found->second.control));
    }
}
struct IndexNode
{
    std::shared_ptr<const void> control;
    std::size_t payload = 0, object = 0, depth = 0;
    const void* parent = nullptr;
    std::vector<const void*> children;
    std::vector<NodeKey> rows;
    std::map<const void*, std::size_t> buffers;
};
struct Index
{
    std::map<const void*, IndexNode, std::less<const void*>> nodes;
    const void* state = nullptr;
    std::size_t height = 0;
};
Index index(const G::GeometryOwnerManifest& manifest)
{
    Index result;
    manifest.visit_dependencies(
        {&result,
            [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t)
            {
                auto& out = *static_cast<Index*>(raw);
                EXPECT_TRUE(out.nodes.emplace(owner.get(), IndexNode{owner, payload}).second);
                return true;
            }},
        {nullptr, [](void*, const G::OwnerRecord&) {}});
    manifest.visit_borrowed_dependencies({&result,
                                             [](void* raw, const void* owner, const void* parent, std::size_t)
                                             {
                                                 auto& out = *static_cast<Index*>(raw);
                                                 auto& node = out.nodes.at(owner);
                                                 node.parent = parent;
                                                 if (parent)
                                                 {
                                                     node.depth = out.nodes.at(parent).depth + 1;
                                                     out.nodes.at(parent).children.push_back(owner);
                                                 }
                                                 else
                                                 {
                                                     EXPECT_EQ(out.state, nullptr);
                                                     out.state = owner;
                                                 }
                                                 out.height = std::max(out.height, node.depth + 1);
                                                 return true;
                                             }},
        {&result, [](void* raw, const void* parent, const G::OwnerRecord& row)
            {
                static_cast<Index*>(raw)->nodes.at(parent).rows.push_back(
                    key(row.key.channel, row.key.kind, row.key.address));
            }});
    const void* current = nullptr;
    manifest.visit_payload(
        [&](const void* address, std::size_t bytes)
        {
            if (const auto found = result.nodes.find(address); found != result.nodes.end())
            {
                current = address;
                found->second.object = bytes;
            }
            else
            {
                ASSERT_NE(current, nullptr);
                EXPECT_TRUE(result.nodes.at(current).buffers.emplace(address, bytes).second);
            }
        });
    for (const auto& [owner, node] : result.nodes)
    {
        std::size_t sum = node.object;
        for (const auto& [buffer, bytes] : node.buffers)
            sum += bytes;
        EXPECT_EQ(sum, node.payload);
    }
    return result;
}
Index index(const D::GeometryIndexControls& controls)
{
    Index result;
    controls.visit_metadata(
        {&result, [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t)
            {
                auto& out = *static_cast<Index*>(raw);
                EXPECT_TRUE(out.nodes.emplace(owner.get(), IndexNode{owner, payload}).second);
                return true;
            }});
    controls.visit_borrowed_dependencies({&result,
                                             [](void* raw, const void* owner, const void* parent, std::size_t)
                                             {
                                                 auto& out = *static_cast<Index*>(raw);
                                                 auto& node = out.nodes.at(owner);
                                                 node.parent = parent;
                                                 if (parent)
                                                 {
                                                     node.depth = out.nodes.at(parent).depth + 1;
                                                     out.nodes.at(parent).children.push_back(owner);
                                                 }
                                                 else
                                                 {
                                                     EXPECT_EQ(out.state, nullptr);
                                                     out.state = owner;
                                                 }
                                                 out.height = std::max(out.height, node.depth + 1);
                                                 return true;
                                             }},
        {&result, [](void* raw, const void* parent, const D::IndexControlRecord& row)
            { static_cast<Index*>(raw)->nodes.at(parent).rows.push_back(metadata(0, row.owner.get())); }});
    const void* current = nullptr;
    controls.visit_payload(
        [&](const void* address, std::size_t bytes)
        {
            if (const auto found = result.nodes.find(address); found != result.nodes.end())
            {
                current = address;
                found->second.object = bytes;
            }
            else
            {
                ASSERT_NE(current, nullptr);
                EXPECT_TRUE(result.nodes.at(current).buffers.emplace(address, bytes).second);
            }
        });
    for (const auto& [owner, node] : result.nodes)
    {
        std::size_t sum = node.object;
        for (const auto& [buffer, bytes] : node.buffers)
            sum += bytes;
        EXPECT_EQ(sum, node.payload);
    }
    return result;
}
using Controls = std::map<const void*, D::IndexControlRecord, std::less<const void*>>;
Controls input_controls(std::span<const D::GeometryOwnerRootBinding> bindings)
{
    Controls result;
    for (const auto& binding : bindings)
        for (const auto& manifest : binding.payload().owners)
            manifest.visit_dependencies(
                {&result,
                    [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
                    {
                        auto& rows = *static_cast<Controls*>(raw);
                        const auto [at, inserted] =
                            rows.emplace(owner.get(), D::IndexControlRecord{owner, payload, admitted - payload});
                        EXPECT_EQ(admitted - payload, G::control_bytes);
                        if (!inserted)
                        {
                            EXPECT_EQ(at->second.payload, payload);
                            EXPECT_EQ(at->second.control, admitted - payload);
                            EXPECT_FALSE(at->second.owner.owner_before(owner));
                            EXPECT_FALSE(owner.owner_before(at->second.owner));
                        }
                        return true;
                    }},
                {nullptr, [](void*, const G::OwnerRecord&) {}});
    return result;
}
Graph control_graph(const Controls& rows)
{
    Graph result;
    for (const auto& [address, row] : rows)
        result.nodes.emplace(metadata(0, address), Node{row.owner, row.payload, 1});
    return result;
}
std::set<NodeKey> control_changes(const Controls& old, const Controls& next)
{
    std::set<NodeKey> changes;
    for (const auto& [address, row] : old)
        if (const auto at = next.find(address); at == next.end())
            changes.insert(metadata(0, address));
        else
        {
            EXPECT_EQ(at->second.payload, row.payload);
            EXPECT_EQ(at->second.control, row.control);
            EXPECT_FALSE(row.owner.owner_before(at->second.owner));
            EXPECT_FALSE(at->second.owner.owner_before(row.owner));
        }
    for (const auto& [address, row] : next)
        if (!old.contains(address))
            changes.insert(metadata(0, address));
    return changes;
}
void exact_controls(const D::GeometryIndexControls& value, const Controls& expected)
{
    Controls actual;
    value.visit_records({&actual, [](void* raw, const D::IndexControlRecord& row)
        { EXPECT_TRUE(static_cast<Controls*>(raw)->emplace(row.owner.get(), row).second); }});
    ASSERT_EQ(actual.size(), expected.size());
    std::size_t control_bytes = 0;
    for (const auto& [address, row] : expected)
    {
        const auto found = actual.find(address);
        ASSERT_NE(found, actual.end());
        EXPECT_EQ(found->second.payload, row.payload);
        EXPECT_EQ(found->second.control, row.control);
        EXPECT_FALSE(found->second.owner.owner_before(row.owner));
        EXPECT_FALSE(row.owner.owner_before(found->second.owner));
        control_bytes += row.control;
    }
    EXPECT_EQ(value.control_bytes(), control_bytes);
}
// Maximum height from independent final cardinality and nonroot minimum
// occupancies. Include State and a spare lookup level for root exceptions.
std::size_t lookup_height(std::size_t rows)
{
    std::size_t height = 3;
    auto pages = std::max(std::size_t{1}, rows / (G::leaf_capacity / 2));
    while (pages > 1)
    {
        pages = std::max(std::size_t{1}, pages / (G::branch_capacity / 2));
        ++height;
    }
    return height;
}
NodeKey maximum(const Index& tree, const void* owner)
{
    const auto& node = tree.nodes.at(owner);
    if (!node.children.empty())
        return maximum(tree, node.children.back());
    EXPECT_FALSE(node.rows.empty());
    return node.rows.back();
}
struct IndexBound
{
    std::set<const void*, std::less<const void*>> paths, may_copy;
    std::size_t copies = 0, patch_visits = 0, max_page_bytes = 0;
    bool redistribution_closed = true;
};
IndexBound bound(const Index& old, const Graph& next, const std::set<NodeKey>& changes,
    std::size_t row_bytes = sizeof(G::OwnerRecord))
{
    IndexBound result;
    if (changes.empty())
        return result;
    const auto& state = old.nodes.at(old.state);
    EXPECT_EQ(state.children.size(), 1u);
    const auto root = state.children.front();
    for (const auto& [owner, node] : old.nodes)
        if (owner != old.state && owner != root)
            result.redistribution_closed = result.redistribution_closed &&
                                           (node.children.empty() ? node.rows.size() >= G::leaf_capacity / 2
                                                                  : node.children.size() >= G::branch_capacity / 2);
    std::map<const void*, std::vector<NodeKey>, std::less<const void*>> leaf_changes;
    for (const auto& change : changes)
    {
        auto owner = root;
        while (true)
        {
            result.paths.insert(owner);
            const auto& node = old.nodes.at(owner);
            if (node.children.empty())
            {
                leaf_changes[owner].push_back(change);
                break;
            }
            owner = node.children.back();
            for (const auto* child : node.children)
                if (!(maximum(old, child) < change))
                {
                    owner = child;
                    break;
                }
        }
    }
    result.patch_visits = result.paths.size();
    result.may_copy = result.paths;
    for (const auto* owner : result.paths)
    {
        const auto& node = old.nodes.at(owner);
        if (!node.parent || node.parent == old.state)
            continue;
        const auto& siblings = old.nodes.at(node.parent).children;
        const auto at = std::find(siblings.begin(), siblings.end(), owner);
        EXPECT_NE(at, siblings.end());
        if (at == siblings.end())
            continue;
        if (at != siblings.begin())
            result.may_copy.insert(*std::prev(at));
        if (std::next(at) != siblings.end())
            result.may_copy.insert(*std::next(at));
    }
    const auto ceiling = [](std::size_t count, std::size_t capacity)
    { return count / capacity + (count % capacity != 0); };
    std::map<const void*, std::size_t, std::less<const void*>> final_leaves;
    for (const auto& [owner, node] : old.nodes)
        if (owner != old.state && node.children.empty())
        {
            auto count = node.rows.size();
            if (const auto changed = leaf_changes.find(owner); changed != leaf_changes.end())
                for (const auto& change : changed->second)
                {
                    const bool was = std::binary_search(node.rows.begin(), node.rows.end(), change);
                    const bool now = next.nodes.contains(change);
                    count += now && !was;
                    count -= was && !now;
                }
            final_leaves.emplace(owner, count);
        }
    for (const auto& [owner, changes] : leaf_changes)
    {
        const auto count = final_leaves.at(owner);
        result.redistribution_closed = result.redistribution_closed && count != 0;
        if (count >= G::leaf_capacity / 2 || old.nodes.at(owner).parent == old.state)
            continue;
        const auto& siblings = old.nodes.at(old.nodes.at(owner).parent).children;
        const auto position = std::find(siblings.begin(), siblings.end(), owner);
        const auto healthy_pair = [&](const void* neighbor)
        {
            const auto rows = final_leaves.at(neighbor);
            const auto groups = ceiling(rows, G::leaf_capacity);
            // A split neighbor contributes its smaller packed page. Check both
            // possible adjacent pairs, independently of which side is chosen.
            return groups && count + rows / groups >= G::leaf_capacity / 2;
        };
        if (position != siblings.begin())
            result.redistribution_closed = result.redistribution_closed && healthy_pair(*std::prev(position));
        if (std::next(position) != siblings.end())
            result.redistribution_closed = result.redistribution_closed && healthy_pair(*std::next(position));
    }
    // Compute replacement packing bounds solely from accepted keys and the
    // independently counted final incoming references. Redistribution never
    // increases child count and makes at most two pages per underflow event.
    struct ReplacementBound
    {
        std::size_t minimum_groups, maximum_groups;
        bool may_underflow;
    };
    std::size_t underflow_paths = 0;
    const auto replacement = [&](const auto& self, const void* owner) -> ReplacementBound
    {
        const auto& node = old.nodes.at(owner);
        if (!result.paths.contains(owner))
            return {1, 1, false};
        std::size_t count = 0, minimum_count = 0, capacity = G::leaf_capacity;
        if (node.children.empty())
        {
            count = node.rows.size();
            for (const auto& change : leaf_changes.at(owner))
            {
                const bool was = std::binary_search(node.rows.begin(), node.rows.end(), change);
                const bool now = next.nodes.contains(change);
                count += now && !was;
                count -= was && !now;
            }
            minimum_count = count;
        }
        else
        {
            capacity = G::branch_capacity;
            for (const auto* child : node.children)
            {
                const auto packed = self(self, child);
                count += packed.maximum_groups;
                // Lower-level repairs can also cross a packing boundary, so
                // the parent must use a separate lower group count.
                minimum_count += packed.minimum_groups - std::size_t(packed.may_underflow);
            }
            if (count)
                minimum_count = std::max(minimum_count, std::size_t{1});
        }
        const auto groups = ceiling(count, capacity);
        result.copies += groups;
        const bool may_underflow = owner != root && minimum_count < capacity / 2;
        underflow_paths += may_underflow;
        return {ceiling(minimum_count, capacity), groups, may_underflow};
    };
    auto roots = replacement(replacement, root).maximum_groups;
    while (roots > 1)
    {
        roots = ceiling(roots, G::branch_capacity);
        result.copies += roots;
    }
    // Healthy replacement paths cannot redistribute. Charge the two possible
    // extra page copies only to paths whose input-derived occupancy may fall
    // below minimum; the checked adjacent-pair closure bounds each repair.
    result.copies += 2 * underflow_paths;
    for (const auto& [owner, node] : old.nodes)
        if (owner != old.state)
            result.max_page_bytes = std::max(result.max_page_bytes,
                node.object + G::control_bytes +
                    std::max(G::leaf_capacity * row_bytes, G::branch_capacity * sizeof(std::shared_ptr<const void>)));
    return result;
}
void untouched(const Index& old, const Index& next, const IndexBound& bounds)
{
    std::size_t checked = 0, buffers = 0;
    for (const auto& [owner, node] : old.nodes)
    {
        if (owner == old.state || bounds.may_copy.contains(owner))
            continue;
        const auto found = next.nodes.find(owner);
        ASSERT_NE(found, next.nodes.end());
        EXPECT_FALSE(node.control.owner_before(found->second.control));
        EXPECT_FALSE(found->second.control.owner_before(node.control));
        EXPECT_EQ(found->second.payload, node.payload);
        EXPECT_EQ(found->second.object, node.object);
        EXPECT_EQ(found->second.buffers, node.buffers);
        ++checked;
        buffers += node.buffers.size();
    }
    EXPECT_GT(checked, 0u);
    EXPECT_GT(buffers, 0u);
}
bool has_untouched(const Index& old, const IndexBound& bounds)
{
    return std::any_of(old.nodes.begin(), old.nodes.end(),
        [&](const auto& item) { return item.first != old.state && !bounds.may_copy.contains(item.first); });
}
auto many_recipe_source()
{
    A::AMRGrid grid;
    grid.root_blocks = {256, 1, 1};
    grid.block_cells = {2, 2, 2};
    A::CartesianAMR native(grid);
    std::vector<std::uint64_t> ids(native.geometry()->num_cells());
    std::iota(ids.begin(), ids.end(), 0);
    auto input = native.detach_owned(ids, {'u', 'n', 'i', 'o', 'n', '-', 'w', 'o', 'r', 'k'});
    return std::make_shared<const Hierarchy>(
        std::move(input.owned), std::move(input.families), std::move(input.kernel), Tpetra::getDefaultComm());
}
} // namespace union_work_oracle

TEST(GeometryOwnerUnion, FixedEditWorkMatchesIndependentPhysicalFrontierAtTwoDemandSizes)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !A::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    namespace W = union_work_oracle;
    bool nonvacuous_copy_bound = false;
    for (const std::size_t demand : {512, 1024})
    {
        SCOPED_TRACE(demand);
        auto hierarchy = W::many_recipe_source();
        std::vector<const A::AdaptiveLeaf*> rows;
        for (const auto& leaf : hierarchy->owned_leaves())
            rows.push_back(&leaf);
        ASSERT_GT(rows.size(), demand);
        const auto get = [&](std::span<const A::AdaptiveLeaf* const> wanted)
        {
            std::size_t used = 37;
            return hierarchy->geometry_snapshot(
                wanted, ArithmeticMode::Ordinary, A::GeometrySnapshotPurpose::OriginalSources, used, cap);
        };
        const auto first = get(std::span(rows).first(demand)), second = get(std::span(rows).subspan(demand, 1));
        ASSERT_EQ(first->ordinary_catalog()->size(), demand);
        ASSERT_EQ(second->ordinary_catalog()->size(), demand + 1);
        const auto old_binding = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(first);
        const auto new_binding = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(second);
        const std::array old_roots{old_binding}, new_roots{new_binding};
        const auto old_graph = W::graph(old_roots), new_graph = W::graph(new_roots);
        const auto expected = W::frontier(old_graph, new_graph, old_graph, new_graph);
        P::GeometryAllocationLedger setup0, setup1;
        std::size_t setup = 37;
        const auto accepted = D::GeometryOwnerUnion{}.staged(nullptr, &old_binding, setup0, setup1, setup, cap);
        W::exact_rows(accepted, old_graph);
        const auto old_index = W::index(accepted.manifests().owners[0]);
        const auto input_before = W::index(old_binding.payload().owners[0]),
                   input_after = W::index(new_binding.payload().owners[0]);
        const auto old_controls = W::input_controls(old_roots), new_controls = W::input_controls(new_roots);
        const auto changed_controls = W::control_changes(old_controls, new_controls);
        const auto control_graph = W::control_graph(new_controls);
        W::exact_controls(accepted.index_controls(), old_controls);
        const auto old_control_index = W::index(accepted.index_controls());
        const auto control_bounds =
            W::bound(old_control_index, control_graph, changed_controls, sizeof(D::IndexControlRecord));
        ASSERT_TRUE(control_bounds.redistribution_closed);
        ASSERT_GE(input_before.height, 3u);
        ASSERT_GT(input_before.nodes.size(), 3u);
        ASSERT_GE(old_index.height, 3u);
        ASSERT_GT(old_index.nodes.size(), 3u);
        const auto bounds = W::bound(old_index, new_graph, expected.changed[0]);
        // Nonempty leaves and healthy adjacent pairs bound redistribution to
        // the initially affected paths and their immediate siblings. Each
        // nonroot branch starts with at least 16 nonempty child replacements;
        // any sole merged branch child is healthy, so no underfull descendant
        // escapes into a neighboring branch during recursive redistribution.
        ASSERT_TRUE(bounds.redistribution_closed);
        nonvacuous_copy_bound = nonvacuous_copy_bound || (bounds.copies < old_index.nodes.size() - 1);
        for (const bool already_credited : {false, true})
        {
            SCOPED_TRACE(already_credited ? "credited" : "fresh");
            P::GeometryAllocationLedger ordinary, bounded;
            std::size_t used = 37;
            if (already_credited)
                accepted.import(ordinary, bounded, used, cap);
            D::GeometryAllocationOwnerAccess::reset_work_statistics();
            D::GeometryOwnerUnionWork work;
            const auto updated = accepted.staged(&old_binding, &new_binding, ordinary, bounded, used, cap, &work);
            const auto global = D::GeometryAllocationOwnerAccess::work_statistics();
            // Stop the qualified work window before any independent export or
            // full graph/control/buffer observation below.
            for (unsigned channel = 0; channel < 2; ++channel)
            {
                EXPECT_EQ(work.channels[channel].first_transitions, expected.first[channel]);
                EXPECT_EQ(work.channels[channel].last_transitions, expected.last[channel]);
                EXPECT_EQ(work.channels[channel].edge_visits, expected.edges[channel]);
                EXPECT_EQ(work.channels[channel].dependency_entries, expected.entries[channel]);
            }
            const auto introduced = expected.first[0] + expected.first[1];
            EXPECT_EQ(work.imported_rows, introduced);
            EXPECT_EQ(global.import_delta_rows, introduced);
            EXPECT_EQ(global.import_fallbacks, 0u);
            EXPECT_EQ(global.metadata_credit_seed_rows, 0u);
            EXPECT_EQ(global.import_rows, 0u);
            EXPECT_EQ(global.export_calls, 0u);
            EXPECT_EQ(global.export_rows, 0u);
            EXPECT_EQ(global.manifest_visit_rows, 0u);
            EXPECT_EQ(global.control_table_enumeration_rows, 0u);
            EXPECT_EQ(work.channels[0].index.visited_pages, bounds.patch_visits);
            EXPECT_LE(work.channels[0].index.copied_pages, bounds.copies);
            EXPECT_LE(work.channels[0].index.copied_bytes, bounds.copies * bounds.max_page_bytes);
            std::size_t new_metadata = 0;
            for (const auto& [id, node] : new_graph.nodes)
                new_metadata += !old_graph.nodes.contains(id) &&
                                id.kind == D::GeometryOwnerGraphEditor::external_kind() && !node.children.empty();
            const auto lookup_bound =
                (3 * expected.entries[0] + 2 * expected.changed[0].size() + new_metadata + 1) * (old_index.height + 1);
            EXPECT_LE(work.channels[0].lookup.visited_pages, lookup_bound);
            const auto delta_bound = (bounds.copies * (G::branch_capacity + 1) + 1) * (old_index.height + 1) +
                                     2 * bounds.copies * G::leaf_capacity * (old_index.height + 1);
            EXPECT_LE(global.import_delta_pages, delta_bound);
            EXPECT_EQ(work.controls.visited_pages, control_bounds.patch_visits);
            EXPECT_LE(work.controls.copied_pages, control_bounds.copies);
            EXPECT_LE(work.controls.copied_bytes, control_bounds.copies * control_bounds.max_page_bytes);
            // Source-audited call inventory; every term comes from independently
            // observed input graphs/cardinalities/paths, never measured copies.
            const auto E = expected.edges[0], K = expected.entries[0], Dkeys = expected.changed[0].size();
            const auto F = expected.first[0], Q = changed_controls.size();
            const auto R = bounds.copies + control_bounds.copies + 2; // two immutable States
            const auto H = old_index.height + 1, Hc = old_control_index.height + 1;
            const auto Hn = W::lookup_height(new_graph.nodes.size()), Hi = input_before.height + 1;
            ASSERT_EQ(old_binding.payload().owners[1].owner_count(), 0u);
            ASSERT_EQ(new_binding.payload().owners[1].owner_count(), 0u);
            const auto all_manifest_lookups = (7 + 4 * E + K + 2 * Dkeys + R + F) * H + 5 * Hn + 2 * Hi + delta_bound;
            const auto all_control_lookups = (3 * E + R + Q) * Hc;
            EXPECT_LE(global.manifest_lookup_pages, all_manifest_lookups);
            EXPECT_LE(global.index_control_lookup_pages, all_control_lookups);
            W::exact_controls(updated.index_controls(), new_controls);
            const auto new_control_index = W::index(updated.index_controls());
            if (W::has_untouched(old_control_index, control_bounds))
                W::untouched(old_control_index, new_control_index, control_bounds);
            EXPECT_GT(D::GeometryAllocationOwnerAccess::work_statistics().control_table_enumeration_rows, 0u);
            W::exact_rows(updated, new_graph);
            const auto new_index = W::index(updated.manifests().owners[0]);
            W::untouched(old_index, new_index, bounds);
            // Check every physically shared input index node's control and
            // backing buffers, not just a single reused buffer/address.
            std::size_t input_shared = 0;
            for (const auto& [owner, node] : input_before.nodes)
                if (const auto found = input_after.nodes.find(owner); found != input_after.nodes.end())
                {
                    EXPECT_FALSE(node.control.owner_before(found->second.control));
                    EXPECT_FALSE(found->second.control.owner_before(node.control));
                    EXPECT_EQ(node.buffers, found->second.buffers);
                    ++input_shared;
                }
            EXPECT_GT(input_shared, 0u);
            RecordProperty(
                std::string(already_credited ? "Credited" : "Fresh") + std::to_string(demand) + "VisitedEdges",
                std::to_string(expected.edges[0]));
            RecordProperty(
                std::string(already_credited ? "Credited" : "Fresh") + std::to_string(demand) + "AffectedIndexNodes",
                std::to_string(bounds.patch_visits));
        }
    }
    EXPECT_TRUE(nonvacuous_copy_bound);
}

TEST(GeometryOwnerUnion, FixedControlTableRemoveAddPreservesUntouchedPagesAtTwoCardinalities)
{
    namespace W = union_work_oracle;
    for (const std::size_t count : {2048, 4096})
    {
        SCOPED_TRACE(count);
        std::vector<D::IndexControlRecord> rows;
        rows.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            auto owner = std::make_shared<const std::array<std::byte, 19>>();
            rows.push_back({owner, sizeof(*owner), G::control_bytes});
        }
        const auto less = [](const auto& a, const auto& b)
        { return std::less<const void*>{}(a.owner.get(), b.owner.get()); };
        std::sort(rows.begin(), rows.end(), less);
        std::size_t used = 37;
        const G::Admission admission{
            &used, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; }};
        const auto accepted = D::GeometryIndexControls{}.edited(rows, admission);
        W::Controls old_rows;
        for (const auto& row : rows)
            ASSERT_TRUE(old_rows.emplace(row.owner.get(), row).second);
        W::exact_controls(accepted, old_rows);
        const auto before = W::index(accepted);
        ASSERT_GE(before.height, 3u);
        ASSERT_GT(before.nodes.size(), 3u);
        auto expected = old_rows;
        auto removed = rows[count / 2];
        ASSERT_EQ(expected.erase(removed.owner.get()), 1u);
        removed.control = 0;
        const auto new_owner = std::make_shared<const std::array<std::byte, 19>>();
        const D::IndexControlRecord added{new_owner, sizeof(*new_owner), G::control_bytes};
        ASSERT_TRUE(expected.emplace(new_owner.get(), added).second);
        std::array changes{removed, added};
        std::sort(changes.begin(), changes.end(), less);
        const auto changed = W::control_changes(old_rows, expected);
        ASSERT_EQ(changed.size(), 2u);
        const auto expected_graph = W::control_graph(expected);
        const auto bounds = W::bound(before, expected_graph, changed, sizeof(D::IndexControlRecord));
        ASSERT_TRUE(bounds.redistribution_closed);
        ASSERT_LT(bounds.copies, before.nodes.size() - 1);
        ASSERT_TRUE(W::has_untouched(before, bounds));
        D::GeometryAllocationOwnerAccess::reset_work_statistics();
        G::PageWork work;
        const auto updated = accepted.edited(changes, admission, &work);
        const auto global = D::GeometryAllocationOwnerAccess::work_statistics();
        EXPECT_EQ(work.visited_pages, bounds.patch_visits);
        EXPECT_LE(work.copied_pages, bounds.copies);
        EXPECT_LE(work.copied_bytes, bounds.copies * bounds.max_page_bytes);
        EXPECT_LE(global.index_control_lookup_pages, changed.size() * (before.height + 1));
        EXPECT_EQ(global.manifest_lookup_pages, 0u);
        EXPECT_EQ(global.control_table_enumeration_rows, 0u);
        EXPECT_EQ(global.import_fallbacks, 0u);
        EXPECT_EQ(global.metadata_credit_seed_rows, 0u);
        W::exact_controls(updated, expected);
        W::untouched(before, W::index(updated), bounds);
        // Holding the accepted value must preserve every old row/control too.
        W::exact_controls(accepted, old_rows);
        EXPECT_GT(D::GeometryAllocationOwnerAccess::work_statistics().control_table_enumeration_rows, 0u);
    }
}

TEST(GeometryOwnerUnion, UnrelatedPopulatedWholeUnionEnumeratesControlsAndPoisonsLateRejection)
{
    auto hierarchy = source();
    const auto* leaf = &*hierarchy->owned_leaves().begin();
    const auto snapshot = acquire(*hierarchy, std::array{leaf});
    const auto binding = A::adaptive_detail::GeometrySnapshotAccess::owner_binding(snapshot);
    P::GeometryAllocationLedger setup0, setup1;
    std::size_t setup = 37;
    const auto value = D::GeometryOwnerUnion{}.staged(nullptr, &binding, setup0, setup1, setup, cap);
    const auto accepted_control = value.manifests().owners[0].root_control();
    const auto accepted_table = value.index_controls().root_control();
    auto expected = census(std::array{snapshot});
    const auto other = std::make_shared<const std::array<char, 19>>();
    append(expected, D::GeometryOwnerGraphEditor::external_kind(), other, sizeof(*other));
    P::GeometryAllocationLedger ordinary, bounded;
    std::size_t used = 37;
    ordinary.retain_external_owner(other, sizeof(*other), used, cap);
    D::GeometryAllocationOwnerAccess::reset_work_statistics();
    value.import(ordinary, bounded, used, cap);
    const auto global = D::GeometryAllocationOwnerAccess::work_statistics();
    EXPECT_GT(global.import_fallbacks, 0u);
    EXPECT_GT(global.import_rows, 0u);
    EXPECT_GT(global.control_table_enumeration_rows, 0u);
    const auto exact = used;
    EXPECT_EQ(ordinary.payload_bytes(), value.manifests().payload_bytes() + sizeof(*other));
    EXPECT_EQ(bounded.payload_bytes(), 0u);
    Owners actual;
    std::size_t scratch = 0;
    const G::Admission allocation{
        &scratch, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; }};
    D::GeometryOwnerImportAccess::export_manifest(ordinary, allocation)
        .visit([&](const G::OwnerRecord& row) { append(actual, row.key.kind, row.owner, row.bytes); });
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [key, row] : expected)
    {
        const auto found = actual.find(key);
        ASSERT_NE(found, actual.end());
        EXPECT_EQ(found->second.bytes, row.bytes);
        EXPECT_FALSE(found->second.owner.owner_before(row.owner));
        EXPECT_FALSE(row.owner.owner_before(found->second.owner));
    }
    D::GeometryAllocationOwnerAccess::reset_work_statistics();
    value.import(ordinary, bounded, used, cap);
    EXPECT_EQ(used, exact);
    EXPECT_EQ(D::GeometryAllocationOwnerAccess::work_statistics().control_table_enumeration_rows, 0u);
    P::GeometryAllocationLedger rejected0, rejected1;
    std::size_t rejected = 37;
    rejected0.retain_external_owner(other, sizeof(*other), rejected, cap);
    D::GeometryAllocationOwnerAccess::reset_work_statistics();
    // Identical physical inputs and map row counts make the successful charge
    // reproducible. The final control-table administration charge is nonzero.
    EXPECT_THROW(value.import(rejected0, rejected1, rejected, exact - 1), P::GeometryError);
    EXPECT_GT(D::GeometryAllocationOwnerAccess::work_statistics().control_table_enumeration_rows, 0u);
    EXPECT_THROW(rejected0.retain_external_owner(other, sizeof(*other), rejected, cap), P::GeometryError);
    EXPECT_THROW(rejected1.retain_external_owner(other, sizeof(*other), rejected, cap), P::GeometryError);
    EXPECT_FALSE(accepted_control.owner_before(value.manifests().owners[0].root_control()));
    EXPECT_FALSE(value.manifests().owners[0].root_control().owner_before(accepted_control));
    EXPECT_FALSE(accepted_table.owner_before(value.index_controls().root_control()));
    EXPECT_FALSE(value.index_controls().root_control().owner_before(accepted_table));
    P::GeometryAllocationLedger retry0, retry1;
    std::size_t retry = 37;
    retry0.retain_external_owner(other, sizeof(*other), retry, cap);
    EXPECT_NO_THROW(value.import(retry0, retry1, retry, exact));
    EXPECT_EQ(retry, exact);
}
