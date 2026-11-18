#include "geometry/mesh/AMR/PersistentOrderedPages.hh"
#include "geometry/unitTests/pipeline_allocation_operators.inc"
#include <gtest/gtest.h>
#include <optional>

namespace
{
namespace G = SimpleFluid::geometry_detail;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
namespace P = D::persistent_ordered_detail;
struct Row
{
    std::uint64_t key, value;
};
struct Less
{
    bool operator()(const Row& a, const Row& b) const { return a.key < b.key; }
};
using Tree = P::OrderedPages<Row, Less>;

const Tree::Node* leaf(const Tree::Ptr& root, std::uint64_t key)
{
    auto* node = root.get();
    while (node && !node->leaf())
    {
        auto found = std::lower_bound(node->children.begin(), node->children.end(), key,
            [](const auto& child, std::uint64_t value) { return child->maximum->key < value; });
        if (found == node->children.end())
            return nullptr;
        node = found->get();
    }
    return node;
}
std::size_t height(const Tree::Ptr& root)
{
    std::size_t count = 0;
    auto* node = root.get();
    while (node)
    {
        ++count;
        node = node->leaf() ? nullptr : node->children.front().get();
    }
    return count;
}
TEST(NativePersistentMetadata, FixedReplacementCopiesItsDerivedPathAndMatchesActualRetainedCppAllocations)
{
    if (!SimpleFluid::PLIC::geometry_encoding_capacity_profile_supported())
    {
        auto owner = G::NativeConstructionCounterAccess::create<D::Budget>(8 * 1024 * 1024);
        G::NativeConstructionAdmission admission(owner);
        EXPECT_THROW((void) Tree(owner.owner(), nullptr, {}, admission.state()), std::invalid_argument);
        return;
    }
    std::vector<Row> rows;
    for (std::uint64_t key = 0; key < 4096; ++key)
        rows.push_back({key, key});
    auto arena = G::NativeConstructionCounterAccess::create<D::Budget>(8 * 1024 * 1024);
    G::NativeConstructionAdmission admission(arena);
    const auto prefix = arena.owner()->used;
    P::OrderedPageWork work;
    Tree builder(arena.owner(), &work, {}, admission.state());
    pipeline_trace::begin();
    auto a = builder.build(rows);
    const auto baseline_pages = work.copied_pages;
    const Row replacement{0, 99999};
    const auto expected_path = height(a);
    auto b = builder.edit(a, replacement, &replacement, true);
    pipeline_trace::recording.store(false);
    const auto trace = pipeline_trace::summary();
    ASSERT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(work.copied_pages - baseline_pages, expected_path);
    ASSERT_NE(leaf(a, 2048), nullptr);
    EXPECT_EQ(leaf(a, 2048), leaf(b, 2048));
    EXPECT_EQ(leaf(a, 2048)->rows.data(), leaf(b, 2048)->rows.data());
    const auto* found = Tree::lower_bound(b, [](const Row&) { return false; });
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->value, 99999u);
    EXPECT_EQ(arena.owner()->used - prefix, trace.all_live);
    EXPECT_EQ(arena.owner()->peak - prefix, trace.all_peak);
    a.reset();
    b.reset();
    EXPECT_EQ(arena.owner()->used, prefix);
}
TEST(NativePersistentMetadata, InheritedPagesKeepOnlyTheirDetachedCountersAcrossThreeGenerations)
{
    if (!SimpleFluid::PLIC::geometry_encoding_capacity_profile_supported())
        return; // the preceding case explicitly tests the unsupported-profile rejection
    std::vector<Row> rows;
    for (std::uint64_t key = 0; key < 4096; ++key)
        rows.push_back({key, key});
    Tree::Ptr c;
    std::weak_ptr<D::Budget> first_counter, second_counter;
    std::weak_ptr<const G::NativeAllocationState> first_state, second_state;
    {
        auto arena_a = G::NativeConstructionCounterAccess::create<D::Budget>(8 * 1024 * 1024);
        auto arena_b = G::NativeConstructionCounterAccess::create<D::Budget>(8 * 1024 * 1024);
        auto arena_c = G::NativeConstructionCounterAccess::create<D::Budget>(8 * 1024 * 1024);
        G::NativeConstructionAdmission admission_a(arena_a), admission_b(arena_b), admission_c(arena_c);
        first_counter = arena_a.owner();
        second_counter = arena_b.owner();
        first_state = admission_a.state();
        second_state = admission_b.state();
        auto a = Tree(arena_a.owner(), nullptr, {}, admission_a.state()).build(rows);
        const Row left{0, 99}, right{4095, 100};
        auto b = Tree(arena_b.owner(), nullptr, {}, admission_b.state()).edit(a, left, &left, true);
        c = Tree(arena_c.owner(), nullptr, {}, admission_c.state()).edit(b, right, &right, true);
        EXPECT_EQ(leaf(a, 2048), leaf(c, 2048));
        a.reset();
        b.reset();
    }
    EXPECT_FALSE(first_counter.expired());
    EXPECT_FALSE(second_counter.expired());
    EXPECT_FALSE(first_state.expired());
    EXPECT_FALSE(second_state.expired());
    c.reset();
    EXPECT_TRUE(first_state.expired());
    EXPECT_TRUE(second_state.expired());
    // The weak State controls keep their bootstrap allocators and Counters.
    EXPECT_FALSE(first_counter.expired());
    EXPECT_FALSE(second_counter.expired());
    first_state.reset();
    second_state.reset();
    EXPECT_TRUE(first_counter.expired());
    EXPECT_TRUE(second_counter.expired());
}
} // namespace
