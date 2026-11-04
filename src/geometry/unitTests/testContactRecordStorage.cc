#include "geometry/mesh/AMR/ContactTracePool.hh"
#include "geometry/mesh/AMR/PagedRecords.hh"
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
constexpr std::size_t ample = 4 * 1024 * 1024;
D::ContactRecord record(unsigned depth = 0, A::AdaptiveSplit split = A::AdaptiveSplit::Isotropic, bool terminal = false)
{
    D::ContactRecord result;
    result.key = {0, A::CartesianCellAncestry{0, {0, 0, 0}}};
    result.domain = {'d'};
    result.spans = {1, 1, 1};
    result.factors = {0, 0, 0};
    result.traces.push_back({A::AdaptiveBytes(152, char(3)), depth, split, {}, terminal});
    return result;
}
std::size_t admit_buffers(D::Budget& budget, const D::ContactRecord& value)
{
    std::size_t bytes = value.traces.capacity() * sizeof(A::AdaptiveTraceFace);
    for (const auto& face : value.traces)
        bytes += face.stem.capacity();
    budget.add(bytes);
    return bytes;
}
TEST(ContactRecordStorage, EqualInventoriesShareWithoutChangingCanonicalPackets)
{
    D::Budget budget{ample}, encoding{ample};
    {
        D::ContactTracePool pool(budget);
        auto first = record();
        const auto bytes = admit_buffers(budget, first);
        const auto before = D::encode_contact(first, encoding);
        EXPECT_EQ(pool.adopt(first), bytes);
        ASSERT_TRUE(first.shared_traces);
        EXPECT_TRUE(first.traces.empty());
        EXPECT_EQ(D::encode_contact(first, encoding), before);
        const auto retained = budget.used;
        auto second = record();
        EXPECT_EQ(admit_buffers(budget, second), bytes);
        const auto cumulative = budget.cumulative;
        EXPECT_EQ(pool.adopt(second), bytes);
        EXPECT_EQ(first.shared_traces, second.shared_traces);
        EXPECT_EQ(budget.used, retained);
        EXPECT_EQ(budget.cumulative, cumulative);
        EXPECT_EQ(D::encode_contact(second, encoding), before);
        RecordProperty("trace_buffer_bytes_per_original_record", std::to_string(bytes));
        RecordProperty("shared_inventory_retained_bytes", std::to_string(pool.retained_bytes()));
    }
    EXPECT_EQ(budget.used, 0u);
}
TEST(ContactRecordStorage, EveryOrderedTraceFieldRemainsPartOfIdentity)
{
    D::Budget budget{ample};
    D::ContactTracePool pool(budget);
    std::array<D::ContactRecord, 6> values{record(), record(1), record(0, A::AdaptiveSplit::Base),
        record(0, A::AdaptiveSplit::Isotropic, true), record(), record()};
    values[4].traces[0].stem.back() = 4;
    values[5].traces.push_back(values[4].traces[0]);
    for (auto& value : values)
    {
        (void) admit_buffers(budget, value);
        (void) pool.adopt(value);
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            EXPECT_NE(values[i].shared_traces, values[j].shared_traces);
    auto reversed = record();
    reversed.traces.assign(values[5].trace_faces().rbegin(), values[5].trace_faces().rend());
    (void) admit_buffers(budget, reversed);
    (void) pool.adopt(reversed);
    EXPECT_NE(reversed.shared_traces, values[5].shared_traces);
    EXPECT_EQ(reversed.trace_faces()[0].stem.back(), 4);
}
TEST(ContactRecordStorage, CopiedPoolAndRecordsSurviveSourceDestruction)
{
    D::Budget source_budget{ample}, candidate_budget{ample};
    std::unique_ptr<D::ContactTracePool> copied;
    std::optional<D::ContactRecord> retained;
    {
        D::ContactTracePool source(source_budget);
        auto value = record(7);
        (void) admit_buffers(source_budget, value);
        (void) source.adopt(value);
        retained = value;
        copied = std::make_unique<D::ContactTracePool>(source, candidate_budget);
        auto equal = record(7);
        (void) admit_buffers(candidate_budget, equal);
        (void) copied->adopt(equal);
        EXPECT_EQ(retained->shared_traces, equal.shared_traces);
    }
    EXPECT_EQ(source_budget.used, 0u);
    ASSERT_TRUE(retained);
    EXPECT_EQ(retained->trace_faces()[0].depth, 7u);
    copied.reset();
    EXPECT_EQ(candidate_budget.used, 0u);
    EXPECT_EQ(retained->trace_faces()[0].stem, A::AdaptiveBytes(152, char(3)));
}
TEST(ContactRecordStorage, FailedPoolForkRetiresOnlyItsOwnLiveAdmission)
{
    D::Budget source_budget{ample};
    D::ContactTracePool source(source_budget);
    auto value = record();
    (void) admit_buffers(source_budget, value);
    (void) source.adopt(value);
    const auto before = source_budget.used;
    D::Budget short_budget{source.retained_bytes() - 1};
    EXPECT_THROW((void) D::ContactTracePool(source, short_budget), std::invalid_argument);
    EXPECT_EQ(short_budget.used, 0u);
    EXPECT_EQ(source_budget.used, before);
    EXPECT_EQ(value.trace_faces()[0].depth, 0u);
}
struct Metadata
{
    std::array<std::uint64_t, 3> key{};
    std::uint64_t stamp = 0;
};
TEST(ContactRecordStorage, MetadataPagesKeepAddressesAndBoundUnusedRecordSlots)
{
    D::Budget budget{ample};
    {
        D::PagedRecords<Metadata> records(budget);
        EXPECT_EQ(records.size(), 0u);
        EXPECT_EQ(records.storage_bytes(), 0u);
        records.reserve(0);
        EXPECT_EQ(budget.used, 0u);
        std::vector<const Metadata*> addresses;
        for (std::size_t i = 0; i < 257; ++i)
        {
            auto& value = records.emplace_back();
            value.key = {i, 2 * i, 3 * i};
            value.stamp = 10 + i;
            addresses.push_back(&value);
            EXPECT_LT(records.capacity() - records.size(), 16u);
            for (std::size_t j = 0; j <= i; ++j)
            {
                EXPECT_EQ(&records[j], addresses[j]);
                EXPECT_EQ(records[j].key, (std::array<std::uint64_t, 3>{j, 2 * j, 3 * j}));
            }
        }
        EXPECT_EQ(budget.used, records.storage_bytes());
    }
    EXPECT_EQ(budget.used, 0u);
}
TEST(ContactRecordStorage, FailedMetadataGrowthLeavesExistingRecordsAndPointersValid)
{
    D::Budget budget{ample};
    D::PagedRecords<Metadata> records(budget);
    for (unsigned i = 0; i < 16; ++i)
        records.emplace_back().stamp = i;
    const auto* first = &records[0];
    const auto before = budget.used;
    budget.limit = before;
    EXPECT_THROW((void) records.emplace_back(), std::invalid_argument);
    EXPECT_EQ(records.size(), 16u);
    EXPECT_EQ(&records[0], first);
    EXPECT_EQ(budget.used, before);
    budget.limit = ample;
    records.emplace_back().stamp = 16;
    EXPECT_EQ(records.size(), 17u);
    EXPECT_EQ(&records[0], first);
    EXPECT_EQ(records.back().stamp, 16u);
}
} // namespace
