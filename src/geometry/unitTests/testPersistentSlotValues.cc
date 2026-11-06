/** Sparse numeric metadata retains zero, full-width values and live overlap. */
#include "geometry/mesh/AMR/PersistentSlotValues.hh"
#include <gtest/gtest.h>
#include <limits>
#include <set>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
using Values = D::PersistentSlotValues<std::uint64_t>;
constexpr std::size_t ample = 64 * 1024 * 1024;
auto arena(std::size_t bytes = ample)
{
    return std::make_shared<D::Budget>(D::Budget{bytes});
}

TEST(PersistentSlotValues, ZeroIsPresentAndFullWidthKeysAndValuesRemainExact)
{
    const auto memory = arena();
    Values values(memory);
    EXPECT_FALSE(values.contains(0));
    EXPECT_EQ(values.find(0), nullptr);
    EXPECT_THROW((void) values.at(0), std::out_of_range);
    values.set(0, 0);
    values.set(std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_TRUE(values.contains(0));
    EXPECT_EQ(values.at(0), 0u);
    EXPECT_EQ(values.at(std::numeric_limits<std::size_t>::max()), std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(values.size(), 2u);
    std::vector<std::pair<std::size_t, std::uint64_t>> observed;
    values.visit([&](auto slot, auto value) { observed.emplace_back(slot, value); });
    ASSERT_EQ(observed.size(), 2u);
    EXPECT_EQ(observed[0], (std::pair<std::size_t, std::uint64_t>{0, 0}));
    EXPECT_EQ(observed[1].first, std::numeric_limits<std::size_t>::max());
    const auto before = memory->used;
    values.set(0, 0);
    EXPECT_EQ(memory->used, before);
    EXPECT_EQ(values.last_edit_work().copied_pages, 0u);
    EXPECT_THROW((void) Values(values, memory), std::invalid_argument);
    EXPECT_EQ(memory->used, before);
}

TEST(PersistentSlotValues, TwoSizesShareUnchangedPagesAndKeepRetiredOverlapUntilFinish)
{
    for (const std::size_t count : {4096u, 8192u})
    {
        SCOPED_TRACE(count);
        const auto source_memory = arena(), candidate_memory = arena();
        Values source(source_memory);
        for (std::size_t i = 0; i < count; ++i)
            source.set(i * 3, i);
        std::set<const void*> old_pages;
        source.visit_borrowed_payload(
            [&](const D::ContactPayloadAllocation& allocation)
            {
                if (allocation.kind == D::ContactPayloadKind::Object)
                    old_pages.insert(allocation.identity.address);
            },
            D::ContactPayloadRole::IncidenceAuthorityCountNodes);
        Values candidate(source, candidate_memory);
        EXPECT_EQ(candidate_memory->used, source.metadata_bytes());
        EXPECT_EQ(candidate.last_edit_work().copied_pages, 0u);
        candidate.set(count / 2 * 3, 0);
        EXPECT_GT(candidate.last_edit_work().copied_pages, 0u);
        EXPECT_LE(candidate.last_edit_work().copied_pages, 6u);
        EXPECT_GT(candidate.overlap_bytes(), 0u);
        EXPECT_EQ(candidate_memory->used, candidate.metadata_bytes() + candidate.overlap_bytes());
        std::size_t shared = 0;
        candidate.visit_borrowed_payload(
            [&](const D::ContactPayloadAllocation& allocation)
            {
                if (allocation.kind == D::ContactPayloadKind::Object)
                    shared += old_pages.contains(allocation.identity.address);
            },
            D::ContactPayloadRole::IncidenceAuthorityCountNodes);
        EXPECT_GE(shared, old_pages.size() - candidate.last_edit_work().copied_pages);
        const auto overlap = candidate.overlap_bytes(), before = candidate_memory->used;
        candidate.finish_overlap();
        EXPECT_EQ(candidate_memory->used, before - overlap);
        EXPECT_EQ(candidate_memory->used, candidate.metadata_bytes());
        EXPECT_EQ(source.at(count / 2 * 3), count / 2);
        EXPECT_EQ(candidate.at(count / 2 * 3), 0u);
        candidate.set(1, 19);
        EXPECT_EQ(candidate_memory->used, candidate.metadata_bytes());
        EXPECT_FALSE(source.contains(1));
        EXPECT_EQ(candidate.at(1), 19u);
    }
}

TEST(PersistentSlotValues, ExactPeakAndOneByteShortPreserveSourceAndCandidateValues)
{
    const auto source_memory = arena();
    Values source(source_memory);
    for (std::size_t i = 0; i < 257; ++i)
        source.set(i, i);
    const auto measured_memory = arena();
    {
        Values measured(source, measured_memory);
        measured.set(128, 1024);
        measured.finish_overlap();
    }
    EXPECT_EQ(measured_memory->used, 0u);
    ASSERT_GT(measured_memory->peak, source.metadata_bytes());
    const auto short_memory = arena(measured_memory->peak - 1);
    {
        Values rejected(source, short_memory);
        const auto before = short_memory->used;
        EXPECT_THROW(rejected.set(128, 1024), std::invalid_argument);
        EXPECT_EQ(rejected.at(128), 128u);
        EXPECT_EQ(short_memory->used, before);
        EXPECT_EQ(rejected.overlap_bytes(), 0u);
    }
    EXPECT_EQ(short_memory->used, 0u);
    const auto exact_memory = arena(measured_memory->peak);
    {
        Values exact(source, exact_memory);
        exact.set(128, 1024);
        EXPECT_EQ(exact.at(128), 1024u);
        EXPECT_EQ(source.at(128), 128u);
        exact.finish_overlap();
        EXPECT_EQ(exact_memory->used, exact.metadata_bytes());
    }
    EXPECT_EQ(exact_memory->used, 0u);
}

TEST(PersistentSlotValues, BorrowedBudgetFallbackDoesNotRetainTheSourceArena)
{
    D::Budget target{ample};
    std::unique_ptr<Values> candidate;
    {
        D::Budget borrowed{ample};
        auto source = std::make_unique<Values>(borrowed);
        source->set(0, 0);
        source->set(999, 31);
        candidate = std::make_unique<Values>(*source, target);
        source.reset();
        EXPECT_EQ(borrowed.used, 0u);
    }
    EXPECT_EQ(candidate->at(0), 0u);
    EXPECT_EQ(candidate->at(999), 31u);
    candidate->finish_overlap();
    EXPECT_EQ(target.used, candidate->metadata_bytes());
    candidate.reset();
    EXPECT_EQ(target.used, 0u);
}
} // namespace
