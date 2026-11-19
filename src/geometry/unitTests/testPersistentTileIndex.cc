/** @file testPersistentTileIndex.cc
 * @brief Persistent tile rank, selective allocation and rollback oracles.
 */
#include "geometry/mesh/AMR/PersistentTileIndex.hh"

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
using Index = D::PersistentTileIndex<std::uint64_t>;
constexpr auto ample = std::size_t{32} * 1024 * 1024;

std::shared_ptr<D::Budget> arena(std::size_t cap = ample)
{
    return std::make_shared<D::Budget>(D::Budget{cap});
}

std::unique_ptr<Index> source(const std::shared_ptr<D::Budget>& memory, std::size_t supports)
{
    auto index = std::make_unique<Index>(*memory, memory);
    Index::Batch batch(*index);
    for (std::size_t support = 0; support < supports; ++support)
        for (std::uint64_t emission = 0; emission < 4; ++emission)
            batch.insert({support, emission}, support * 4 + emission);
    batch.publish();
    return index;
}

TEST(PersistentTileIndex, DoublingUnrelatedSupportsDoesNotCopyTheAcceptedIndex)
{
    std::size_t first = 0;
    for (const auto supports : {1024u, 2048u})
    {
        const auto old_memory = arena();
        auto accepted = source(old_memory, supports);
        const auto current_memory = arena();
        {
            Index candidate(*current_memory, current_memory);
            candidate.fork_from(*accepted);
            EXPECT_EQ(candidate.metadata_bytes(), accepted->metadata_bytes());
            EXPECT_EQ(candidate.inherited_metadata_bytes(), accepted->metadata_bytes());
            EXPECT_EQ(current_memory->used, 0u);
            const auto* unchanged = accepted->find({supports - 1, 3});
            candidate.update({10, 1}, [](auto& value) { value = 911; });
            EXPECT_EQ(candidate.at({10, 1}), 911u);
            EXPECT_EQ(accepted->at({10, 1}), 41u);
            EXPECT_EQ(candidate.find({supports - 1, 3}), unchanged);
            EXPECT_EQ(candidate.lower_bound({10, 1}), 41u);
            EXPECT_EQ(candidate.key_at(41), (Index::Key{10, 1}));
            EXPECT_EQ(candidate.value_at(41), 911u);
            EXPECT_EQ(candidate.lower_bound({supports, 0}), candidate.size());
            EXPECT_EQ(candidate.inherited_metadata_bytes() + current_memory->used, candidate.metadata_bytes());
            if (!first)
                first = current_memory->used;
            EXPECT_LE(current_memory->used, first + 4096);
            EXPECT_LT(current_memory->used, accepted->metadata_bytes() / 8);
            accepted.reset();
            EXPECT_EQ(old_memory->used, candidate.inherited_metadata_bytes());
            EXPECT_EQ(candidate.find({supports - 1, 3}), unchanged);
            EXPECT_EQ(*unchanged, supports * 4 - 1);
        }
        EXPECT_EQ(old_memory->used, 0u);
        EXPECT_EQ(current_memory->used, 0u);
    }
}

TEST(PersistentTileIndex, ExactBatchCapAndRejectedEditsPreserveAcceptedPages)
{
    const auto old_memory = arena();
    const auto accepted = source(old_memory, 72);
    const auto before = old_memory->used;
    const auto replace = [](Index& index, const D::Budget& memory)
    {
        Index::Batch batch(index);
        batch.replace_support(9);
        batch.insert({9, 11}, 71);
        batch.insert({9, 18}, 81);
        batch.publish(400);
        EXPECT_EQ(memory.used, index.metadata_bytes() - index.inherited_metadata_bytes());
    };
    const auto measured = arena();
    {
        Index candidate(*measured, measured);
        candidate.fork_from(*accepted);
        replace(candidate, *measured);
        EXPECT_EQ(candidate.size(), accepted->size() - 2);
        EXPECT_FALSE(candidate.find({9, 0}));
        EXPECT_EQ(candidate.at({9, 11}), 71u);
        EXPECT_EQ(candidate.at({9, 18}), 81u);
        EXPECT_EQ(candidate.key_at(candidate.lower_bound({9, 0})), (Index::Key{9, 11}));
        EXPECT_EQ(candidate.key_at(candidate.lower_bound({10, 0})), (Index::Key{10, 0}));
    }
    ASSERT_GT(measured->peak, 0u);
    EXPECT_EQ(measured->used, 0u);
    const auto short_memory = arena(measured->peak - 1);
    Index rejected(*short_memory, short_memory);
    rejected.fork_from(*accepted);
    EXPECT_THROW(replace(rejected, *short_memory), std::invalid_argument);
    EXPECT_EQ(short_memory->used, 0u);
    EXPECT_EQ(rejected.find({9, 0}), accepted->find({9, 0}));
    EXPECT_EQ(rejected.size(), accepted->size());
    EXPECT_EQ(old_memory->used, before);

    const auto exact_memory = arena(measured->peak);
    Index retry(*exact_memory, exact_memory);
    retry.fork_from(*accepted);
    EXPECT_NO_THROW(replace(retry, *exact_memory));
    EXPECT_EQ(retry.at({9, 18}), 81u);
    EXPECT_THROW(retry.update({9, 18},
                     [](auto& value)
                     {
                         value = 0;
                         throw std::invalid_argument("Reject test update.");
                     }),
        std::invalid_argument);
    EXPECT_EQ(retry.at({9, 18}), 81u);
    EXPECT_EQ(old_memory->used, before);
}
TEST(PersistentTileIndex, RetireSupportRangesCollapsesOnlyThePrivateRoot)
{
    const auto old_memory = arena();
    const auto accepted = source(old_memory, 80);
    const auto candidate_memory = arena();
    Index candidate(*candidate_memory, candidate_memory);
    candidate.fork_from(*accepted);
    {
        Index::Batch batch(candidate);
        for (std::size_t support = 0; support < 80; ++support)
            if (support != 31)
                batch.replace_support(support);
        batch.publish(4);
    }
    ASSERT_EQ(candidate.size(), 4u);
    EXPECT_EQ(accepted->size(), 320u);
    for (std::size_t i = 0; i < candidate.size(); ++i)
    {
        EXPECT_EQ(candidate.key_at(i), (Index::Key{31, i}));
        EXPECT_EQ(candidate.value_at(i), 124u + i);
    }
    EXPECT_EQ(candidate.inherited_metadata_bytes() + candidate_memory->used, candidate.metadata_bytes());
    {
        Index::Batch batch(candidate);
        batch.replace_support(31);
        batch.publish(0);
    }
    EXPECT_EQ(candidate.size(), 0u);
    EXPECT_EQ(candidate.metadata_bytes(), 0u);
    EXPECT_EQ(candidate_memory->used, 0u);
    EXPECT_EQ(accepted->at({31, 0}), 124u);
}

} // namespace
