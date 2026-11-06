/** Sparse candidate edits retain and merge the accepted ordered source. */
#include "geometry/mesh/AMR/AdaptiveLeafOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>
namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
A::AdaptiveCellKey key(std::uint64_t id) { return {0, A::ConvexCellAncestry{id, {}}}; }
A::AdaptiveLeaf leaf(std::uint64_t id, std::uint64_t revision = 0)
{
    A::AdaptiveLeaf value;
    value.key = key(id);
    value.revision = revision;
    return value;
}
TEST(AdaptiveLeafOverlay, SparseChangesMergeInKeyOrderAndKeepSource)
{
    std::vector<A::AdaptiveLeaf> input;
    for (unsigned i = 0; i < 4096; ++i)
        input.push_back(leaf(2 * i));
    auto source_budget = std::make_shared<D::Budget>(D::Budget{64 * 1024 * 1024});
    const auto source = D::AdaptiveLeafPages::build(input, source_budget);
    D::Budget budget{1024 * 1024};
    {
        D::AdaptiveLeafOverlay overlay(source, budget);
        EXPECT_EQ(overlay.find(key(0)), source.find(key(0)));
        EXPECT_EQ(overlay.edit_count(), 0u);
        EXPECT_TRUE(overlay.retire(key(10)));
        EXPECT_FALSE(overlay.retire(key(10)));
        EXPECT_TRUE(overlay.insert(leaf(10, 4)));
        EXPECT_FALSE(overlay.insert(leaf(10, 5)));
        EXPECT_TRUE(overlay.insert(leaf(11, 2)));
        EXPECT_TRUE(overlay.retire(key(12)));
        EXPECT_EQ(overlay.edit_count(), 3u);
        EXPECT_EQ(overlay.at(key(10)).revision, 4u);
        EXPECT_FALSE(overlay.contains(key(12)));
        EXPECT_EQ(source.at(key(10)).revision, 0u);
        EXPECT_TRUE(source.contains(key(12)));
        std::vector<std::uint64_t> ids;
        overlay.visit_owned([&](const auto& value) { ids.push_back(std::get<A::ConvexCellAncestry>(value.key.ancestry).root); });
        ASSERT_EQ(ids.size(), 4096u);
        EXPECT_TRUE(std::is_sorted(ids.begin(), ids.end()));
        EXPECT_EQ(ids[5], 10u);
        EXPECT_EQ(ids[6], 11u);
        std::vector<D::AdaptiveLeafPages::EditView> edits;
        overlay.visit_edits([&](auto edit) { edits.push_back(edit); });
        ASSERT_EQ(edits.size(), 3u);
        EXPECT_EQ(edits[0].expected_revision, 0u);
        EXPECT_EQ(edits[0].leaf->revision, 4u);
        EXPECT_FALSE(edits[1].expected_revision);
        EXPECT_EQ(edits[2].expected_revision, 0u);
        EXPECT_EQ(edits[2].leaf, nullptr);
        EXPECT_LT(budget.used, 64 * 1024u);
    }
    EXPECT_EQ(budget.used, 0u);
}
TEST(AdaptiveLeafOverlay, FailedAdmissionLeavesSourceAndCandidateUnchanged)
{
    std::vector<A::AdaptiveLeaf> input{leaf(0), leaf(2)};
    auto source_budget = std::make_shared<D::Budget>(D::Budget{1024 * 1024});
    const auto source = D::AdaptiveLeafPages::build(input, source_budget);
    D::Budget budget{0};
    D::AdaptiveLeafOverlay overlay(source, budget);
    EXPECT_THROW((void) overlay.insert(leaf(1)), std::invalid_argument);
    EXPECT_EQ(overlay.size(), 2u);
    EXPECT_EQ(overlay.edit_count(), 0u);
    EXPECT_EQ(overlay.find(key(0)), source.find(key(0)));
    EXPECT_FALSE(overlay.contains(key(1)));
    EXPECT_EQ(budget.used, 0u);
}
} // namespace
