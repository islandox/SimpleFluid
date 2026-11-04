/** Immutable tile bodies: literal values, arena lifetimes and admission failures. */
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/PooledTile.hh"
#include "geometry/mesh/AMR/TileBodyPages.hh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
struct HeapActivity
{
    std::size_t calls = 0, bytes = 0;
};
thread_local HeapActivity* active_heap_probe = nullptr;
thread_local std::optional<std::size_t> allocations_before_failure;
class FailAllocation
{
public:
    explicit FailAllocation(std::size_t successful_allocations)
        : previous_(std::exchange(allocations_before_failure, successful_allocations))
    {
    }
    ~FailAllocation() { allocations_before_failure = previous_; }

private:
    std::optional<std::size_t> previous_;
};
class HeapProbe
{
public:
    explicit HeapProbe(HeapActivity& activity) : previous_(std::exchange(active_heap_probe, &activity)) {}
    ~HeapProbe() { active_heap_probe = previous_; }

private:
    HeapActivity* previous_;
};
} // namespace
void* operator new(std::size_t bytes)
{
    if (allocations_before_failure)
    {
        if (*allocations_before_failure == 0)
        {
            allocations_before_failure.reset();
            throw std::bad_alloc();
        }
        --*allocations_before_failure;
    }
    if (void* result = std::malloc(bytes ? bytes : 1))
    {
        if (active_heap_probe)
        {
            ++active_heap_probe->calls;
            active_heap_probe->bytes += bytes;
        }
        return result;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes)
{
    return ::operator new(bytes);
}
void operator delete(void* value) noexcept
{
    std::free(value);
}
void operator delete(void* value, std::size_t) noexcept
{
    std::free(value);
}
void operator delete[](void* value) noexcept
{
    std::free(value);
}
void operator delete[](void* value, std::size_t) noexcept
{
    std::free(value);
}

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
constexpr std::size_t ample = 1024 * 1024;
constexpr std::size_t caller_live = 113;
constexpr auto absent = std::numeric_limits<std::size_t>::max();
constexpr auto maximum_emission = std::numeric_limits<std::uint64_t>::max();
constexpr std::array<std::size_t, 5> literal_ring{19, 2, 31, 7, 11};
constexpr std::array<std::size_t, 5> reversed_ring{11, 7, 31, 2, 19};
D::PooledTile literal_tile()
{
    return {9, 4, 2, 7, 5, 0, absent, maximum_emission - 3, {19, 2, 31, 7, 11}};
}
void expect_scalars(const D::TileBodyScalars& actual, const D::TileBodyScalars& expected)
{
    EXPECT_EQ(actual.support, expected.support);
    EXPECT_EQ(actual.domain, expected.domain);
    EXPECT_EQ(actual.plane, expected.plane);
    EXPECT_EQ(actual.normal_slot, expected.normal_slot);
    EXPECT_EQ(actual.image_slot, expected.image_slot);
    EXPECT_EQ(actual.first_cell, expected.first_cell);
    EXPECT_EQ(actual.second_cell, expected.second_cell);
    EXPECT_EQ(actual.emission, expected.emission);
}
void expect_ring(const D::TileBodyLoopView& actual, std::span<const std::size_t> expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual.empty(), expected.empty());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(actual[i], expected[i]);
        EXPECT_EQ(actual.at(i), expected[i]);
    }
    EXPECT_EQ((std::vector<std::size_t>(actual.begin(), actual.end())),
        (std::vector<std::size_t>(expected.begin(), expected.end())));
    EXPECT_THROW((void) actual.at(expected.size()), std::out_of_range);
}
void expect_literal(const D::TileBodyHandle& handle)
{
    const auto view = handle.view();
    expect_scalars(view.scalars(), {9, 4, 2, 7, 5, 0, absent, maximum_emission - 3});
    expect_ring(view.intrinsic_loop(), literal_ring);
}

TEST(TileBodyPages, EmptyBuildersFreezeAndReapWithoutAllocationOrAdmission)
{
    D::Budget budget{ample};
    budget.add(caller_live);
    HeapActivity heap;
    D::TileBodyPageStatistics statistics;
    D::TileBodyReapStatistics reaped;
    std::size_t retained = 1;
    {
        HeapProbe probe(heap);
        D::TileBodyPages pages(budget);
        pages.freeze();
        reaped = pages.reap();
        statistics = pages.statistics();
        retained = pages.retained_bytes();
    }
    EXPECT_EQ(heap.calls, 0u);
    EXPECT_EQ(heap.bytes, 0u);
    EXPECT_EQ(budget.used, caller_live);
    EXPECT_EQ(budget.cumulative, caller_live);
    EXPECT_EQ(retained, 0u);
    EXPECT_EQ(statistics.pages, 0u);
    EXPECT_EQ(statistics.wide_pages, 0u);
    EXPECT_EQ(statistics.occupied_records, 0u);
    EXPECT_EQ(statistics.live_records, 0u);
    EXPECT_EQ(statistics.record_capacity, 0u);
    EXPECT_EQ(statistics.ring_size, 0u);
    EXPECT_EQ(statistics.ring_capacity, 0u);
    EXPECT_EQ(statistics.ring_bytes, 0u);
    EXPECT_EQ(statistics.metadata_bytes, 0u);
    EXPECT_EQ(statistics.page_credit_bytes, 0u);
    EXPECT_EQ(statistics.page_body_bytes, 0u);
    EXPECT_EQ(statistics.page_credit_overhead_bytes, 0u);
    EXPECT_EQ(statistics.deferred_pages, 0u);
    EXPECT_EQ(statistics.deferred_body_bytes, 0u);
    EXPECT_EQ(statistics.deferred_metadata_bytes, 0u);
    EXPECT_EQ(reaped.pages, 0u);
    EXPECT_EQ(reaped.occupied_records, 0u);
    EXPECT_EQ(reaped.released_bytes, 0u);
}

TEST(TileBodyPages, LiteralScalarsAndRingWindingSurviveArbitraryDegrees)
{
    D::Budget budget{ample};
    {
        D::TileBodyPages pages(budget);
        auto first_input = literal_tile();
        auto second_input = literal_tile();
        second_input.intrinsic_loop = {11, 7, 31, 2, 19};
        second_input.emission = maximum_emission;
        auto first = pages.append(first_input);
        auto second = pages.append(second_input);
        std::array<std::vector<std::size_t>, 2> large_rings;
        std::array<D::TileBodyHandle, 2> large;
        for (std::size_t i = 0; i < large.size(); ++i)
        {
            const auto degree = i ? 65u : 19u;
            for (std::size_t j = 0; j < degree; ++j)
                large_rings[i].push_back(1000 + 17 * j);
            auto input = literal_tile();
            input.intrinsic_loop = large_rings[i];
            large[i] = pages.append(input);
        }
        first_input = {};
        second_input = {};
        HeapActivity heap;
        {
            HeapProbe probe(heap);
            pages.freeze();
        }
        EXPECT_EQ(heap.calls, 0u);
        HeapActivity read_heap;
        bool exact = true;
        {
            HeapProbe probe(read_heap);
            for (unsigned repetition = 0; repetition < 10; ++repetition)
            {
                const auto view = first.view();
                const auto loop = view.intrinsic_loop();
                exact &= view.scalars().emission == maximum_emission - 3;
                for (std::size_t i = 0; i < literal_ring.size(); ++i)
                    exact &= loop[i] == literal_ring[i];
            }
        }
        EXPECT_TRUE(exact);
        EXPECT_EQ(read_heap.calls, 0u);
        expect_literal(first);
        expect_scalars(second.view().scalars(), {9, 4, 2, 7, 5, 0, absent, maximum_emission});
        expect_ring(second.view().intrinsic_loop(), reversed_ring);
        for (std::size_t i = 0; i < large.size(); ++i)
            expect_ring(large[i].view().intrinsic_loop(), large_rings[i]);
        const auto statistics = pages.statistics();
        EXPECT_EQ(statistics.pages, 1u);
        EXPECT_EQ(statistics.wide_pages, 0u); // Emission remains uint64 even on a compact page.
        EXPECT_EQ(statistics.occupied_records, 4u);
        EXPECT_EQ(statistics.ring_size, 94u);
        EXPECT_GE(statistics.ring_capacity, 94u);
        EXPECT_EQ(statistics.ring_bytes, statistics.ring_capacity * sizeof(std::uint32_t));
    }
    EXPECT_EQ(budget.used, 0u);
}

TEST(TileBodyPages, PartialPagesSealBeforeForkAndAdoptIntoIndependentArenaCredits)
{
    for (const std::size_t count : {1u, 15u, 16u, 17u})
    {
        SCOPED_TRACE(count);
        D::Budget source_budget{ample}, candidate_budget{ample};
        {
            D::TileBodyPages source(source_budget);
            std::vector<D::TileBodyHandle> originals;
            for (std::size_t i = 0; i < count; ++i)
                originals.push_back(source.append(literal_tile()));
            if (count % 16)
            {
                EXPECT_THROW((void) originals.back().view(), std::invalid_argument);
                const auto spent = candidate_budget.cumulative;
                EXPECT_THROW((void) D::TileBodyPages(source, candidate_budget), std::invalid_argument);
                EXPECT_EQ(candidate_budget.used, 0u);
                EXPECT_EQ(candidate_budget.cumulative, spent);
            }
            if (count >= 16)
                expect_literal(originals.front());
            source.freeze();
            const auto expected_pages = (count + 15) / 16;
            const auto source_statistics = source.statistics();
            EXPECT_EQ(source_statistics.pages, expected_pages);
            EXPECT_EQ(source_statistics.record_capacity, 16 * expected_pages);
            EXPECT_EQ(source_statistics.occupied_records, count);
            EXPECT_EQ(source_statistics.live_records, count);
            D::TileBodyPages candidate(source, candidate_budget);
            EXPECT_EQ(candidate.statistics().live_records, 0u);
            std::vector<D::TileBodyHandle> adopted;
            for (const auto& original : originals)
            {
                adopted.push_back(candidate.adopt(original));
                EXPECT_EQ(adopted.back().page_identity(), original.page_identity());
                EXPECT_NE(adopted.back().credit_owner(), original.credit_owner());
                expect_literal(adopted.back());
            }
            EXPECT_EQ(candidate.statistics().live_records, count);
            auto source_only_input = literal_tile();
            source_only_input.emission = 41;
            auto candidate_only_input = literal_tile();
            candidate_only_input.emission = 73;
            auto source_only = source.append(source_only_input);
            auto candidate_only = candidate.append(candidate_only_input);
            source.freeze();
            candidate.freeze();
            EXPECT_NE(source_only.page_identity(), candidate_only.page_identity());
            EXPECT_NE(source_only.page_identity(), originals.back().page_identity());
            EXPECT_NE(candidate_only.page_identity(), originals.back().page_identity());
            EXPECT_EQ(source_only.view().scalars().emission, 41u);
            EXPECT_EQ(candidate_only.view().scalars().emission, 73u);
            for (const auto& original : originals)
                expect_literal(original);
            for (const auto& copy : adopted)
                expect_literal(copy);
        }
        EXPECT_EQ(source_budget.used, 0u);
        EXPECT_EQ(candidate_budget.used, 0u);
    }
}

TEST(TileBodyPages, CopiesAndSurvivingLoopViewCountOneLiveRecordAndKeepTheArenaAlive)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    std::weak_ptr<D::Budget> weak = budget;
    D::TileBodyLoopView surviving;
    {
        D::TileBodyPages pages(*budget, budget);
        auto handle = pages.append(literal_tile());
        pages.freeze();
        auto copy = handle;
        std::optional<D::TileBodyView> view(handle.view());
        surviving = view->intrinsic_loop();
        auto loop_copy = surviving;
        EXPECT_EQ(pages.statistics().live_records, 1u);
        handle = {};
        copy = {};
        view.reset();
        loop_copy = {};
        EXPECT_EQ(pages.statistics().live_records, 1u);
        EXPECT_EQ(pages.reap().pages, 0u);
        expect_ring(surviving, literal_ring);
        budget.reset();
    }
    EXPECT_FALSE(weak.expired());
    expect_ring(surviving, literal_ring);
    surviving = {};
    EXPECT_TRUE(weak.expired());
}

TEST(TileBodyPages, LastLoopCopyReleasesTheRecordAfterAllOtherHandleKindsDisappear)
{
    D::Budget budget{ample};
    D::TileBodyPages pages(budget);
    auto first = pages.append(literal_tile());
    pages.freeze();
    auto copy = first;
    std::optional<D::TileBodyView> view(first.view());
    auto loop = view->intrinsic_loop();
    auto loop_copy = loop;
    auto moved = std::move(copy);
    EXPECT_FALSE(copy);
    EXPECT_EQ(pages.statistics().live_records, 1u);
    first = {};
    moved = {};
    view.reset();
    loop = {};
    EXPECT_EQ(pages.statistics().live_records, 1u);
    expect_ring(loop_copy, literal_ring);
    loop_copy = {};
    EXPECT_EQ(pages.statistics().live_records, 0u);
    EXPECT_EQ(pages.reap().occupied_records, 1u);
}

TEST(TileBodyPages, MaximumCompactCellIsPresentWhileSizeMaximumIsAbsent)
{
    if constexpr (std::numeric_limits<std::size_t>::digits <= 32)
        GTEST_SKIP() << "UINT32_MAX and the absent size_t sentinel differ only on wider platforms.";
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    D::Budget budget{ample};
    D::TileBodyPages pages(budget);
    auto input = literal_tile();
    input.first_cell = maximum;
    input.second_cell = absent;
    input.emission = maximum_emission;
    input.intrinsic_loop = {maximum, 0, 7};
    const auto first = pages.append(input);
    input.first_cell = absent;
    input.second_cell = maximum;
    const auto second = pages.append(input);
    pages.freeze();
    expect_scalars(first.view().scalars(), {9, 4, 2, 7, 5, maximum, absent, maximum_emission});
    expect_scalars(second.view().scalars(), {9, 4, 2, 7, 5, absent, maximum, maximum_emission});
    const std::array<std::size_t, 3> expected{maximum, 0, 7};
    expect_ring(first.view().intrinsic_loop(), expected);
    expect_ring(second.view().intrinsic_loop(), expected);
    EXPECT_EQ(pages.statistics().wide_pages, 0u);
}

TEST(TileBodyPages, EveryWideScalarAndWideRingRoundTripsWithoutLargeAllocations)
{
    if constexpr (std::numeric_limits<std::size_t>::digits <= 32)
        GTEST_SKIP() << "Wide pool slots require size_t wider than uint32_t.";
    constexpr auto wide = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
    constexpr std::array<std::size_t D::PooledTile::*, 7> members{&D::PooledTile::support, &D::PooledTile::domain,
        &D::PooledTile::plane, &D::PooledTile::normal_slot, &D::PooledTile::image_slot, &D::PooledTile::first_cell,
        &D::PooledTile::second_cell};
    constexpr std::array<D::TileBodyScalars, 7> expected{{
        {wide, 4, 2, 7, 5, 0, absent, maximum_emission - 3},
        {9, wide, 2, 7, 5, 0, absent, maximum_emission - 3},
        {9, 4, wide, 7, 5, 0, absent, maximum_emission - 3},
        {9, 4, 2, wide, 5, 0, absent, maximum_emission - 3},
        {9, 4, 2, 7, wide, 0, absent, maximum_emission - 3},
        {9, 4, 2, 7, 5, wide, absent, maximum_emission - 3},
        {9, 4, 2, 7, 5, 0, wide, maximum_emission - 3},
    }};
    for (std::size_t i = 0; i <= members.size(); ++i)
    {
        SCOPED_TRACE(i);
        D::Budget budget{ample};
        D::TileBodyPages pages(budget);
        auto input = literal_tile();
        if (i < members.size())
            input.*members[i] = wide;
        else
            input.intrinsic_loop = {19, wide, 7};
        HeapActivity heap;
        D::TileBodyHandle handle;
        {
            HeapProbe probe(heap);
            handle = pages.append(input);
            pages.freeze();
        }
        EXPECT_LT(heap.bytes, 64u * 1024);
        EXPECT_LT(budget.used, 64u * 1024);
        EXPECT_EQ(pages.statistics().wide_pages, 1u);
        if (i < members.size())
        {
            expect_scalars(handle.view().scalars(), expected[i]);
            expect_ring(handle.view().intrinsic_loop(), literal_ring);
        }
        else
        {
            expect_scalars(handle.view().scalars(), {9, 4, 2, 7, 5, 0, absent, maximum_emission - 3});
            const std::array<std::size_t, 3> ring{19, wide, 7};
            expect_ring(handle.view().intrinsic_loop(), ring);
        }
    }
}

TEST(TileBodyPages, FailedRingGrowthAndPromotionLeaveThePrefixIntactAndAllowRetry)
{
    if constexpr (std::numeric_limits<std::size_t>::digits <= 32)
        GTEST_SKIP() << "The promotion case requires wide size_t slots.";
    constexpr auto wide = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
    for (const unsigned mode : {0u, 1u, 2u})
    {
        SCOPED_TRACE(mode);
        const bool promote = mode != 0;
        D::Budget budget{ample};
        budget.add(caller_live);
        {
            D::TileBodyPages pages(budget);
            auto first = pages.append(literal_tile());
            auto next = literal_tile();
            next.emission = 101;
            if (mode == 2)
                next.intrinsic_loop = {11, wide, 31, 2, 19};
            else if (mode == 1)
            {
                next.plane = wide;
                next.intrinsic_loop = {11, 7, 31, 2, 19};
            }
            else
                next.intrinsic_loop = {11, 7, 31, 2, 19, 47, 53};
            const auto before = pages.statistics();
            const auto used = budget.used, spent = budget.cumulative;
            budget.limit = used;
            EXPECT_THROW((void) pages.append(next), std::invalid_argument);
            EXPECT_EQ(budget.used, used);
            EXPECT_EQ(budget.cumulative, spent);
            EXPECT_EQ(pages.statistics().occupied_records, 1u);
            EXPECT_EQ(pages.statistics().ring_size, before.ring_size);
            EXPECT_EQ(pages.statistics().ring_capacity, before.ring_capacity);
            EXPECT_EQ(pages.statistics().wide_pages, 0u);
            budget.limit = ample;
            auto retried = pages.append(next);
            pages.freeze();
            expect_literal(first);
            expect_scalars(retried.view().scalars(), {9, 4, mode == 1 ? wide : 2, 7, 5, 0, absent, 101});
            if (mode == 2)
            {
                const std::array<std::size_t, 5> expected{11, wide, 31, 2, 19};
                expect_ring(retried.view().intrinsic_loop(), expected);
            }
            else if (mode == 1)
                expect_ring(retried.view().intrinsic_loop(), reversed_ring);
            else
            {
                constexpr std::array<std::size_t, 7> expected{11, 7, 31, 2, 19, 47, 53};
                expect_ring(retried.view().intrinsic_loop(), expected);
            }
            EXPECT_EQ(pages.statistics().occupied_records, 2u);
            EXPECT_EQ(pages.statistics().wide_pages, promote ? 1u : 0u);
        }
        EXPECT_EQ(budget.used, caller_live);
    }
}

TEST(TileBodyPages, ObservedSimultaneousPeakAdmitsExactlyAndOneByteLessRejects)
{
    std::array<D::PooledTile, 17> inputs;
    for (auto& input : inputs)
        input = literal_tile();
    if constexpr (std::numeric_limits<std::size_t>::digits > 32)
        inputs[7].plane = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
    const auto append_all = [&](D::TileBodyPages& pages)
    {
        std::array<D::TileBodyHandle, 17> handles;
        for (std::size_t i = 0; i < inputs.size(); ++i)
            handles[i] = pages.append(inputs[i]);
        pages.freeze();
        return handles;
    };
    D::Budget probe{ample};
    probe.add(caller_live);
    {
        D::TileBodyPages pages(probe);
        const auto handles = append_all(pages);
        EXPECT_EQ(pages.statistics().occupied_records, 17u);
        expect_literal(handles.front());
        expect_literal(handles.back());
    }
    EXPECT_EQ(probe.used, caller_live);
    const auto peak = probe.peak;
    for (const bool admit : {true, false})
    {
        SCOPED_TRACE(admit);
        D::Budget budget{peak - (admit ? 0 : 1)};
        budget.add(caller_live);
        std::size_t spent = 0;
        {
            D::TileBodyPages pages(budget);
            if (admit)
            {
                EXPECT_NO_THROW((void) append_all(pages));
                EXPECT_EQ(budget.peak, peak);
            }
            else
                EXPECT_THROW((void) append_all(pages), std::invalid_argument);
            spent = budget.cumulative;
        }
        EXPECT_EQ(budget.used, caller_live);
        EXPECT_EQ(budget.cumulative, spent);
    }
}

TEST(TileBodyPages, RepeatedForeignAdoptionHasOneChargeAndSurvivesSourceArenaDestruction)
{
    auto source_budget = std::make_unique<D::Budget>(D::Budget{ample});
    D::Budget candidate_budget{ample};
    candidate_budget.add(caller_live);
    {
        auto source = std::make_unique<D::TileBodyPages>(*source_budget);
        auto original = source->append(literal_tile());
        source->freeze();
        D::TileBodyPages candidate(candidate_budget);
        auto adopted = candidate.adopt(original);
        const auto used = candidate_budget.used, spent = candidate_budget.cumulative;
        auto adopted_again = candidate.adopt(original);
        EXPECT_EQ(candidate_budget.used, used);
        EXPECT_EQ(candidate_budget.cumulative, spent);
        EXPECT_EQ(candidate.statistics().pages, 1u);
        EXPECT_EQ(candidate.statistics().live_records, 1u);
        EXPECT_EQ(adopted.credit_owner(), adopted_again.credit_owner());
        EXPECT_NE(adopted.credit_owner(), original.credit_owner());
        EXPECT_EQ(adopted.page_identity(), original.page_identity());
        original = {};
        source.reset();
        EXPECT_EQ(source_budget->used, 0u);
        source_budget.reset(); // Raw-budget source owners are gone; the candidate owns its own credit.
        expect_literal(adopted);
        expect_literal(adopted_again);
    }
    EXPECT_EQ(candidate_budget.used, caller_live);
}

TEST(TileBodyPages, ReapingPrivateBodiesReleasesCreditAndReusesTheDirectorySlot)
{
    D::Budget budget{ample};
    budget.add(caller_live);
    {
        D::TileBodyPages pages(budget);
        auto first = pages.append(literal_tile());
        pages.freeze();
        EXPECT_EQ(pages.reap().pages, 0u);
        const auto credit_bytes = first.credit_bytes();
        std::weak_ptr<const void> credit = first.credit_owner();
        auto body = first.body_owner();
        EXPECT_FALSE(body.expired());
        EXPECT_EQ(pages.statistics().page_body_bytes, first.body_bytes());
        EXPECT_EQ(pages.statistics().page_credit_overhead_bytes, credit_bytes - first.body_bytes());
        first = {};
        const auto before = budget.used, spent = budget.cumulative;
        const auto metadata = pages.metadata_bytes();
        const auto result = pages.reap();
        EXPECT_EQ(result.pages, 1u);
        EXPECT_EQ(result.occupied_records, 1u);
        EXPECT_EQ(result.released_bytes, credit_bytes);
        EXPECT_TRUE(credit.expired());
        EXPECT_TRUE(body.expired());
        EXPECT_EQ(budget.used, before - credit_bytes);
        EXPECT_EQ(budget.used, caller_live + metadata);
        EXPECT_EQ(budget.cumulative, spent);
        EXPECT_EQ(pages.statistics().pages, 0u);
        auto replacement_input = literal_tile();
        replacement_input.emission = 23;
        auto replacement = pages.append(replacement_input);
        pages.freeze();
        EXPECT_EQ(pages.metadata_bytes(), metadata);
        EXPECT_EQ(replacement.view().scalars().emission, 23u);
        EXPECT_EQ(pages.statistics().occupied_records, 1u);
    }
    EXPECT_EQ(budget.used, caller_live);
}

TEST(TileBodyPages, ReapedForeignOrdinalCannotAliasANewBodyWhenTheOriginalIsReadopted)
{
    D::Budget source_budget{ample}, candidate_budget{ample};
    {
        D::TileBodyPages source(source_budget);
        auto original = source.append(literal_tile());
        source.freeze();
        D::TileBodyPages candidate(source, candidate_budget);
        auto imported = candidate.adopt(original);
        const auto original_body_bytes = imported.body_bytes();
        imported = {};
        EXPECT_EQ(candidate.reap().pages, 1u);
        EXPECT_EQ(candidate.statistics().pages, 0u);
        const auto directory_bytes = candidate.metadata_bytes();
        auto different_input = literal_tile();
        different_input.emission = 57;
        different_input.intrinsic_loop = {11, 7, 31, 2, 19};
        auto different = candidate.append(different_input);
        candidate.freeze();
        EXPECT_EQ(candidate.metadata_bytes(), directory_bytes);
        ASSERT_NE(different.page_identity(), original.page_identity());
        auto readopted = candidate.adopt(original);
        EXPECT_EQ(readopted.page_identity(), original.page_identity());
        EXPECT_NE(readopted.credit_owner(), original.credit_owner());
        EXPECT_NE(readopted.page_identity(), different.page_identity());
        EXPECT_EQ(candidate.statistics().deferred_pages, 0u);
        EXPECT_EQ(candidate.statistics().deferred_body_bytes, 0u);
        EXPECT_EQ(candidate.statistics().deferred_metadata_bytes, 0u);
        EXPECT_EQ(candidate.statistics().page_body_bytes, original_body_bytes + different.body_bytes());
        EXPECT_EQ(candidate.statistics().page_credit_bytes,
            candidate.statistics().page_body_bytes + candidate.statistics().page_credit_overhead_bytes);
        EXPECT_EQ(candidate_budget.used, candidate.retained_bytes());
        expect_literal(readopted);
        expect_scalars(different.view().scalars(), {9, 4, 2, 7, 5, 0, absent, 57});
        expect_ring(different.view().intrinsic_loop(), reversed_ring);
        const auto used = candidate_budget.used, spent = candidate_budget.cumulative;
        auto second_readoption = candidate.adopt(original);
        EXPECT_EQ(second_readoption.credit_owner(), readopted.credit_owner());
        EXPECT_EQ(candidate_budget.used, used);
        EXPECT_EQ(candidate_budget.cumulative, spent);
    }
    EXPECT_EQ(source_budget.used, 0u);
    EXPECT_EQ(candidate_budget.used, 0u);
}

TEST(TileBodyPages, ReapDefersBodyAdmissionWhileTheAcceptedSourcePinsIt)
{
    auto source_budget = std::make_unique<D::Budget>(D::Budget{ample});
    D::Budget candidate_budget{ample};
    candidate_budget.add(caller_live);
    {
        auto source = std::make_unique<D::TileBodyPages>(*source_budget);
        auto original = source->append(literal_tile());
        source->freeze();
        D::TileBodyPages candidate(*source, candidate_budget);
        auto imported = candidate.adopt(original);
        std::weak_ptr<const void> imported_credit = imported.credit_owner();
        auto body = imported.body_owner();
        const auto body_bytes = imported.body_bytes();
        imported = {};
        const auto first_reap = candidate.reap();
        EXPECT_EQ(first_reap.pages, 1u);
        EXPECT_EQ(first_reap.occupied_records, 1u);
        EXPECT_TRUE(imported_credit.expired());
        EXPECT_EQ(candidate.statistics().pages, 0u);
        EXPECT_EQ(candidate.statistics().deferred_pages, 1u);
        EXPECT_EQ(candidate.statistics().deferred_body_bytes, body_bytes);
        EXPECT_GT(candidate.statistics().deferred_metadata_bytes, 0u);
        EXPECT_EQ(candidate.statistics().page_body_bytes, 0u);
        EXPECT_FALSE(body.expired());
        EXPECT_GT(candidate_budget.used, caller_live + candidate.metadata_bytes());
        EXPECT_EQ(candidate_budget.used, caller_live + candidate.retained_bytes());
        const auto pinned_used = candidate_budget.used, pinned_spent = candidate_budget.cumulative;
        const auto still_pinned = candidate.reap();
        EXPECT_EQ(still_pinned.pages, 0u);
        EXPECT_EQ(still_pinned.released_bytes, 0u);
        EXPECT_EQ(candidate_budget.used, pinned_used);
        EXPECT_EQ(candidate_budget.cumulative, pinned_spent);
        auto private_input = literal_tile();
        private_input.emission = 67;
        auto unrelated = candidate.append(private_input);
        candidate.freeze();
        const auto unrelated_credit = unrelated.credit_bytes();
        original = {};
        source.reset();
        EXPECT_EQ(source_budget->used, 0u);
        source_budget.reset();
        EXPECT_TRUE(body.expired());
        const auto before_release = candidate_budget.used, spent = candidate_budget.cumulative;
        const auto release = candidate.reap();
        EXPECT_GT(release.released_bytes, 0u);
        EXPECT_EQ(candidate_budget.used, before_release - release.released_bytes);
        EXPECT_EQ(candidate_budget.cumulative, spent);
        EXPECT_EQ(candidate.statistics().pages, 1u);
        EXPECT_EQ(candidate.statistics().live_records, 1u);
        EXPECT_EQ(candidate.statistics().page_credit_bytes, unrelated_credit);
        EXPECT_EQ(candidate.statistics().deferred_pages, 0u);
        EXPECT_EQ(candidate.statistics().deferred_body_bytes, 0u);
        EXPECT_EQ(candidate.statistics().deferred_metadata_bytes, 0u);
        EXPECT_EQ(candidate_budget.used, caller_live + candidate.metadata_bytes() + unrelated_credit);
        expect_scalars(unrelated.view().scalars(), {9, 4, 2, 7, 5, 0, absent, 67});
        expect_ring(unrelated.view().intrinsic_loop(), literal_ring);
    }
    EXPECT_EQ(candidate_budget.used, caller_live);
}

TEST(TileBodyPages, FailedDeferredMetadataAdmissionPreservesTheOriginalImportedPage)
{
    D::Budget source_budget{ample}, candidate_budget{ample};
    candidate_budget.add(caller_live);
    {
        D::TileBodyPages source(source_budget);
        auto original = source.append(literal_tile());
        source.freeze();
        D::TileBodyPages candidate(source, candidate_budget);
        auto imported = candidate.adopt(original);
        std::weak_ptr<const void> imported_credit = imported.credit_owner();
        imported = {};
        const auto before = candidate_budget.used, spent = candidate_budget.cumulative;
        candidate_budget.limit = before;
        EXPECT_THROW((void) candidate.reap(), std::invalid_argument);
        EXPECT_EQ(candidate_budget.used, before);
        EXPECT_EQ(candidate_budget.cumulative, spent);
        EXPECT_FALSE(imported_credit.expired());
        EXPECT_EQ(candidate.statistics().pages, 1u);
        EXPECT_EQ(candidate.statistics().occupied_records, 1u);
        EXPECT_EQ(candidate.statistics().live_records, 0u);
        EXPECT_EQ(candidate.statistics().deferred_pages, 0u);
        auto recovered = candidate.adopt(original);
        EXPECT_EQ(recovered.credit_owner(), imported_credit.lock());
        EXPECT_EQ(recovered.page_identity(), original.page_identity());
        EXPECT_EQ(candidate_budget.used, before);
        EXPECT_EQ(candidate_budget.cumulative, spent);
        expect_literal(recovered);
        recovered = {};
        candidate_budget.limit = ample;
        EXPECT_EQ(candidate.reap().pages, 1u);
        EXPECT_TRUE(imported_credit.expired());
        expect_literal(original);
    }
    EXPECT_EQ(source_budget.used, 0u);
    EXPECT_EQ(candidate_budget.used, caller_live);
}
TEST(TileBodyPages, SecondPromotionAllocationFailurePreservesTheCompactPrefix)
{
    if constexpr (std::numeric_limits<std::size_t>::digits <= 32)
        GTEST_SKIP() << "The wide-index fixture requires a wider size_t.";
    D::Budget budget{ample};
    budget.add(caller_live);
    {
        D::TileBodyPages pages(budget);
        auto prefix = pages.append(literal_tile());
        auto wide_input = literal_tile();
        wide_input.plane = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
        const auto before = budget.used, spent = budget.cumulative;
        bool rejected = false;
        {
            FailAllocation fail_second(1);
            try
            {
                (void) pages.append(wide_input);
            }
            catch (const std::bad_alloc&)
            {
                rejected = true;
            }
        }
        EXPECT_TRUE(rejected);
        EXPECT_EQ(budget.used, before);
        EXPECT_GT(budget.cumulative, spent);
        EXPECT_EQ(pages.statistics().occupied_records, 1u);
        EXPECT_EQ(pages.statistics().wide_pages, 0u);
        auto retry = pages.append(wide_input);
        pages.freeze();
        expect_literal(prefix);
        EXPECT_EQ(retry.view().scalars().plane, wide_input.plane);
        expect_ring(retry.intrinsic_loop(), literal_ring);
        EXPECT_EQ(pages.statistics().occupied_records, 2u);
    }
    EXPECT_EQ(budget.used, caller_live);
}

TEST(TileBodyPages, FailedReadoptionDirectoryGrowthKeepsTheDeferredBodyCredit)
{
    D::Budget source_budget{ample}, candidate_budget{ample};
    {
        D::TileBodyPages source(source_budget);
        auto original = source.append(literal_tile());
        source.freeze();
        D::TileBodyPages candidate(source, candidate_budget);
        EXPECT_EQ(candidate.reap().pages, 1u);
        auto replacement_input = literal_tile();
        replacement_input.emission = 73;
        auto replacement = candidate.append(replacement_input);
        candidate.freeze();
        const auto before = candidate_budget.used;
        const auto deferred_body = candidate.statistics().deferred_body_bytes;
        const auto overhead = replacement.credit_bytes() - replacement.body_bytes();
        candidate_budget.limit = before + overhead;
        EXPECT_THROW((void) candidate.adopt(original), std::invalid_argument);
        EXPECT_EQ(candidate_budget.used, before);
        EXPECT_EQ(candidate.statistics().deferred_pages, 1u);
        EXPECT_EQ(candidate.statistics().deferred_body_bytes, deferred_body);
        EXPECT_EQ(candidate.statistics().pages, 1u);
        EXPECT_EQ(replacement.view().scalars().emission, 73u);
        expect_literal(original);
        candidate_budget.limit = ample;
        auto retry = candidate.adopt(original);
        EXPECT_EQ(candidate.statistics().deferred_pages, 0u);
        EXPECT_EQ(candidate.statistics().page_body_bytes, original.body_bytes() + replacement.body_bytes());
        expect_literal(retry);
    }
    EXPECT_EQ(source_budget.used, 0u);
    EXPECT_EQ(candidate_budget.used, 0u);
}

TEST(TileBodyPages, DeferredOnlyForkRetainsIndependentCreditUntilEachArenaReaps)
{
    D::Budget source_budget{ample}, middle_budget{ample}, descendant_budget{ample};
    {
        auto source = std::make_unique<D::TileBodyPages>(source_budget);
        auto original = source->append(literal_tile());
        source->freeze();
        D::TileBodyPages middle(*source, middle_budget);
        EXPECT_EQ(middle.reap().pages, 1u);
        ASSERT_EQ(middle.statistics().pages, 0u);
        ASSERT_EQ(middle.statistics().deferred_pages, 1u);
        D::TileBodyPages descendant(middle, descendant_budget);
        EXPECT_EQ(descendant.statistics().pages, 0u);
        EXPECT_EQ(descendant.statistics().deferred_pages, 1u);
        EXPECT_EQ(descendant.statistics().deferred_body_bytes, original.body_bytes());
        const auto descendant_used = descendant_budget.used;
        auto weak_body = original.body_owner();
        original = {};
        source.reset();
        EXPECT_TRUE(weak_body.expired());
        EXPECT_EQ(source_budget.used, 0u);
        (void) middle.reap();
        EXPECT_EQ(middle.statistics().deferred_pages, 0u);
        EXPECT_EQ(descendant_budget.used, descendant_used);
        const auto spent = descendant_budget.cumulative;
        (void) descendant.reap();
        EXPECT_EQ(descendant.statistics().deferred_pages, 0u);
        EXPECT_LT(descendant_budget.used, descendant_used);
        EXPECT_EQ(descendant_budget.cumulative, spent);
    }
    EXPECT_EQ(middle_budget.used, 0u);
    EXPECT_EQ(descendant_budget.used, 0u);
}
} // namespace
