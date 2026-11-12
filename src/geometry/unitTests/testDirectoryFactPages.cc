/** @file testDirectoryFactPages.cc
 * @brief Shared directory facts, literal projections and admission rollback.
 */
#include "geometry/mesh/AMR/DirectoryFactPages.hh"
#include <array>
#include <gtest/gtest.h>
#include <set>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
using Bytes = SimpleFluid::Meshes::AMR::AdaptiveBytes;
using Pages = D::DirectoryFactPages;
constexpr std::size_t ample = 64 * 1024 * 1024;
auto arena(std::size_t cap = ample)
{
    return std::make_shared<D::Budget>(D::Budget{cap});
}
struct Input
{
    std::vector<Bytes> keys;
    std::vector<D::DirectoryRegistrationView> rows;
    explicit Input(std::size_t count)
    {
        for (std::size_t i = 0; i < count; ++i)
            keys.push_back({char(i >> 8), char(i & 255), 'k'});
        std::sort(keys.begin(), keys.end());
        for (std::size_t i = 0; i < count; ++i)
            rows.push_back(
                {keys[i], int(i % 3), i % 5 ? std::numeric_limits<std::uint64_t>::max() - i : 7, 100 + i, i % 5 == 0});
    }
};

TEST(DirectoryFactPages, FixedFactEditSharesPagesAndReportsNumericOutputSeparately)
{
    const auto measure = [](std::size_t count)
    {
        Input input(count);
        auto source_memory = arena();
        auto source = Pages::reconcile(nullptr, input.rows, source_memory, 7);
        EXPECT_EQ(source_memory->used, source.storage_bytes());
        const auto* clean = source.page_identity(input.keys.back());
        auto changed = input.rows;
        for (auto& row : changed)
        {
            row.address += 3000;
            if (row.topology_revision)
                row.revision = 9;
        }
        changed[1].revision = 11;
        const auto target_memory = arena();
        auto target = Pages::reconcile(&source, changed, target_memory, 9);
        EXPECT_EQ(target.work().fact_edits, 1u);
        EXPECT_EQ(target.page_identity(input.keys.back()), clean);
        EXPECT_EQ(target.numeric_bytes(), count * sizeof(std::uint64_t));
        std::set<const void*> pages;
        std::size_t fact_union = 0;
        const auto retain = [&](const void* identity, std::size_t, std::size_t admission, bool numeric)
        {
            if (!numeric && admission && pages.insert(identity).second)
                fact_union += admission;
        };
        source.visit_storage(retain);
        target.visit_storage(retain);
        EXPECT_EQ(target_memory->used, fact_union + target.numeric_bytes());
        for (std::size_t i = 0; i < input.rows.size(); ++i)
        {
            const auto old = source.find(input.keys[i]), current = target.find(input.keys[i]);
            EXPECT_TRUE(old);
            EXPECT_TRUE(current);
            if (!old || !current)
                continue;
            EXPECT_EQ(old->revision, input.rows[i].revision);
            EXPECT_EQ(current->revision, changed[i].revision);
            EXPECT_EQ(current->address, changed[i].address);
            EXPECT_EQ(current->owner, changed[i].owner);
        }
        target.finish_overlap();
        EXPECT_EQ(target_memory->used, target.storage_bytes());
        return target.work();
    };
    const auto small = measure(4096), large = measure(8192);
    EXPECT_LT(large.created_pages, 2 * small.created_pages);
    EXPECT_LT(large.created_page_bytes, 2 * small.created_page_bytes);
}

TEST(DirectoryFactPages, FullWidthLiteralsAndUniformAddressesNeverBecomeEpochMarkers)
{
    const std::array<Bytes, 4> keys{{{'a'}, {'b'}, {'c'}, {'d'}}};
    const auto max = std::numeric_limits<std::uint64_t>::max();
    const std::array rows{D::DirectoryRegistrationView{keys[0], 0, 0, max, true},
        D::DirectoryRegistrationView{keys[1], 0, 0, max, false},
        D::DirectoryRegistrationView{keys[2], 0, max, max, false},
        D::DirectoryRegistrationView{keys[3], 0, 17, max, false}};
    auto source_memory = arena();
    auto source = Pages::reconcile(nullptr, rows, source_memory, 0);
    EXPECT_EQ(source.numeric_bytes(), 0u);
    auto changed = rows;
    changed[0].revision = max;
    changed[1].revision = 0; // Equal to the old epoch remains a literal.
    auto memory = arena();
    auto next = Pages::reconcile(&source, changed, memory, max);
    EXPECT_EQ(next.work().fact_edits, 0u);
    EXPECT_EQ(next.root_owner(), source.root_owner());
    EXPECT_EQ(next.find(keys[0])->revision, max);
    EXPECT_EQ(next.find(keys[1])->revision, 0u);
    EXPECT_EQ(next.find(keys[2])->revision, max);
    EXPECT_EQ(next.find(keys[3])->revision, 17u);
    EXPECT_EQ(next.find(keys[0])->address, max);

    auto transition = changed;
    transition[0].topology_revision = false;
    transition[1].topology_revision = true;
    transition[1].revision = max;
    auto last_memory = arena();
    auto last = Pages::reconcile(&next, transition, last_memory, max);
    EXPECT_EQ(last.work().fact_edits, 2u);
    EXPECT_EQ(last.find(keys[0])->revision, max);
    EXPECT_EQ(last.find(keys[1])->revision, max);
}

TEST(DirectoryFactPages, OneByteShortAfterPartialReconciliationPreservesSourceAndThirdGeneration)
{
    Input input(256);
    auto original_memory = arena();
    auto original = std::make_unique<Pages>(Pages::reconcile(nullptr, input.rows, original_memory, 7));
    const auto root = std::weak_ptr<const void>(original->root_owner());
    const auto source_bytes = original_memory->used;
    auto changed = input.rows;
    changed.front().owner = 4;
    changed[254].revision = 37;
    const auto probe_memory = arena();
    {
        const auto probe = Pages::reconcile(original.get(), changed, probe_memory, 7);
        EXPECT_EQ(probe.work().fact_edits, 2u);
    }
    EXPECT_EQ(probe_memory->used, 0u);
    auto tight = arena(probe_memory->peak - 1);
    std::size_t completed = 0;
    EXPECT_THROW((void) Pages::reconcile(original.get(), changed, tight, 7, &completed), std::invalid_argument);
    EXPECT_GT(completed, 0u);
    EXPECT_EQ(tight->used, 0u);
    EXPECT_EQ(original_memory->used, source_bytes);
    EXPECT_EQ(original->find(input.keys.front())->owner, input.rows.front().owner);
    tight->limit = probe_memory->peak;
    auto current = std::make_unique<Pages>(Pages::reconcile(original.get(), changed, tight, 7));
    current->finish_overlap();
    original.reset();
    EXPECT_TRUE(root.expired());
    EXPECT_EQ(current->find(input.keys.front())->owner, 4);
    changed[3].revision = 91;
    auto third_memory = arena();
    auto third = Pages::reconcile(current.get(), changed, third_memory, 7);
    third.finish_overlap();
    current.reset();
    EXPECT_EQ(third.find(input.keys[254])->revision, 37u);
    EXPECT_EQ(third.find(input.keys[3])->revision, 91u);
}
} // namespace
