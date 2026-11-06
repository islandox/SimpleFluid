/** Persistent owner-local leaf pages: independent ordered reference and admission oracles. */
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
using Pages = D::AdaptiveLeafPages;
using Reference = std::map<AMR::AdaptiveCellKey, std::uint64_t>;
constexpr std::size_t ample = 64 * 1024 * 1024;

AMR::AdaptiveCellKey key(std::uint64_t root)
{
    return {0, AMR::ConvexCellAncestry{root, {}}};
}
AMR::AdaptiveLeaf leaf(std::uint64_t root, std::uint64_t revision = 0)
{
    AMR::AdaptiveLeaf value;
    value.key = key(root);
    value.revision = revision;
    value.material = root + 3;
    return value;
}
std::vector<AMR::AdaptiveLeaf> leaves(std::size_t count)
{
    std::vector<AMR::AdaptiveLeaf> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        result.push_back(leaf(2 * i));
    return result;
}
Reference reference(const std::vector<AMR::AdaptiveLeaf>& values)
{
    Reference result;
    for (const auto& value : values)
        result.emplace(value.key, value.revision);
    return result;
}
void expect_matches(const Pages& pages, const Reference& expected)
{
    ASSERT_EQ(pages.size(), expected.size());
    EXPECT_EQ(pages.empty(), expected.empty());
    auto cursor = pages.begin();
    std::size_t ordinal = 0;
    for (const auto& [expected_key, revision] : expected)
    {
        ASSERT_NE(cursor, pages.end());
        EXPECT_EQ(cursor->key, expected_key);
        EXPECT_EQ(cursor->revision, revision);
        EXPECT_EQ(pages.find(expected_key), &*cursor);
        EXPECT_TRUE(pages.contains(expected_key));
        EXPECT_EQ(&pages.at(expected_key), &*cursor);
        EXPECT_EQ(pages.ordinal_of(expected_key), ordinal);
        EXPECT_EQ(&pages.at_ordinal(ordinal), &*cursor);
        ++cursor;
        ++ordinal;
    }
    EXPECT_EQ(cursor, pages.end());
    const auto absent = key(999999);
    EXPECT_FALSE(pages.contains(absent));
    EXPECT_EQ(pages.find(absent), nullptr);
    EXPECT_FALSE(pages.ordinal_of(absent));
    EXPECT_THROW((void) pages.at(absent), std::out_of_range);
    EXPECT_THROW((void) pages.at_ordinal(expected.size()), std::out_of_range);
}

TEST(AdaptiveLeafLayout, DeterministicInsertionReplacementAndDeletionMatchOrderedReference)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto pages = Pages::build({}, arena);
    Reference expected;
    expect_matches(pages, expected);
    std::vector<unsigned> order(256);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 generator(0x51a7);
    std::shuffle(order.begin(), order.end(), generator);
    for (const auto id : order)
    {
        const Pages::Edit edit{key(id), {}, leaf(id)};
        pages = Pages::edit(pages, std::span(&edit, 1), arena);
        expected.emplace(key(id), 0);
    }
    expect_matches(pages, expected);
    const auto original = pages;
    for (unsigned id = 0; id < 256; id += 7)
    {
        const Pages::Edit edit{key(id), 0, leaf(id, 1)};
        pages = Pages::edit(pages, std::span(&edit, 1), arena);
        expected.at(key(id)) = 1;
    }
    expect_matches(pages, expected);
    std::shuffle(order.begin(), order.end(), generator);
    for (const auto id : order)
    {
        const Pages::Edit edit{key(id), expected.at(key(id)), {}};
        pages = Pages::edit(pages, std::span(&edit, 1), arena);
        expected.erase(key(id));
        if (expected.size() % 32 == 0)
            expect_matches(pages, expected);
    }
    expect_matches(pages, expected);
    EXPECT_EQ(original.size(), 256u);
    for (unsigned id = 0; id < 256; ++id)
        EXPECT_EQ(original.at(key(id)).revision, 0u);
}

TEST(AdaptiveLeafLayout, InvalidChangesLeaveSourceAndPageChargesIntact)
{
    const auto input = leaves(129);
    auto base_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto original = Pages::build(input, base_arena);
    const auto expected = reference(input);
    const auto base_used = base_arena->used;
    auto edit_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto unchanged = Pages::edit(original, {}, edit_arena);
    EXPECT_EQ(unchanged.page_identity(key(0)), original.page_identity(key(0)));
    EXPECT_EQ(edit_arena->used, 0u);

    const Pages::Edit bad_revision{key(4), 99, leaf(4, 100)};
    EXPECT_THROW((void) Pages::edit(original, std::span(&bad_revision, 1), edit_arena), std::invalid_argument);
    const Pages::Edit duplicate_insert{key(4), {}, leaf(4, 1)};
    EXPECT_THROW((void) Pages::edit(original, std::span(&duplicate_insert, 1), edit_arena), std::invalid_argument);
    const Pages::Edit missing_erase{key(5), 0, {}};
    EXPECT_THROW((void) Pages::edit(original, std::span(&missing_erase, 1), edit_arena), std::invalid_argument);
    const std::array<Pages::Edit, 2> reversed{{{key(8), 0, {}}, {key(4), 0, {}}}};
    EXPECT_THROW((void) Pages::edit(original, reversed, edit_arena), std::invalid_argument);
    auto duplicate_input = input;
    duplicate_input[1] = duplicate_input[0];
    EXPECT_THROW((void) Pages::build(duplicate_input, edit_arena), std::invalid_argument);
    EXPECT_EQ(edit_arena->used, 0u);
    EXPECT_EQ(base_arena->used, base_used);
    expect_matches(original, expected);
    expect_matches(unchanged, expected);
}

TEST(AdaptiveLeafLayout, BatchDeletionCollapsesInteriorLevelsAndPreservesRanks)
{
    const auto input = leaves(4096);
    auto arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto original = Pages::build(input, arena);
    Reference expected = reference(input);
    std::vector<Pages::Edit> edits;
    for (std::size_t i = 0; i < 2048; ++i)
    {
        edits.push_back({key(2 * i), 0, {}});
        expected.erase(key(2 * i));
    }
    auto halfway = Pages::edit(original, edits, arena);
    expect_matches(halfway, expected);
    edits.clear();
    for (std::size_t i = 2048; i < 4050; ++i)
    {
        edits.push_back({key(2 * i), 0, {}});
        expected.erase(key(2 * i));
    }
    auto changed = Pages::edit(halfway, edits, arena);
    expect_matches(changed, expected);
    EXPECT_EQ(changed.storage().leaf_rows, 46u);
    EXPECT_LE(changed.storage().leaf_pages, 2u);
    EXPECT_EQ(original.size(), 4096u);
    EXPECT_EQ(original.at_ordinal(0).key, key(0));
}

TEST(AdaptiveLeafLayout, SparseEditSharesPagesAndHasBoundedCopiesAtTwoSizes)
{
    for (const std::size_t count : {4096u, 8192u})
    {
        SCOPED_TRACE(count);
        const auto input = leaves(count);
        auto base_arena = std::make_shared<D::Budget>(D::Budget{ample});
        auto original = Pages::build(input, base_arena);
        const auto old_storage = original.storage();
        EXPECT_EQ(old_storage.leaf_rows, count);
        EXPECT_LE(old_storage.maximum_leaf_rows, Pages::leaf_capacity);
        EXPECT_LE(old_storage.maximum_fanout, Pages::branch_capacity);
        auto edit_arena = std::make_shared<D::Budget>(D::Budget{ample});
        Pages::EditStats stats;
        const Pages::Edit edit{key(count), 0, leaf(count, 9)};
        auto changed = Pages::edit(original, std::span(&edit, 1), edit_arena, &stats);
        EXPECT_EQ(original.at(key(count)).revision, 0u);
        EXPECT_EQ(changed.at(key(count)).revision, 9u);
        EXPECT_NE(original.page_identity(key(count)), changed.page_identity(key(count)));
        EXPECT_EQ(original.page_identity(key(0)), changed.page_identity(key(0)));
        EXPECT_LE(stats.copied_rows, 2 * Pages::leaf_capacity);
        EXPECT_LE(stats.new_leaf_pages, 2u);
        EXPECT_LE(stats.new_nodes, 8u);
        std::map<const void*, std::size_t> old_pages, new_pages, owners;
        original.visit_pages([&](const void* owner, std::size_t bytes) { old_pages.emplace(owner, bytes); });
        changed.visit_pages([&](const void* owner, std::size_t bytes) { new_pages.emplace(owner, bytes); });
        owners = old_pages;
        for (const auto& [owner, bytes] : new_pages)
            owners.emplace(owner, bytes);
        std::size_t shared = 0, unique_bytes = 0;
        for (const auto& [owner, bytes] : old_pages)
            shared += new_pages.contains(owner);
        for (const auto& [owner, bytes] : owners)
            unique_bytes += bytes;
        EXPECT_GT(shared, 0u);
        EXPECT_LT(unique_bytes, original.storage().retained_bytes + changed.storage().retained_bytes);
        EXPECT_LE(unique_bytes, ample);
        original = {};
        EXPECT_EQ(changed.at(key(count)).revision, 9u);
    }
}

TEST(AdaptiveLeafLayout, DeepRecordCopyChargesAndMovingPagesRetainBuffers)
{
    auto input = leaves(256);
    for (auto& value : input)
    {
        std::get<AMR::ConvexCellAncestry>(value.key.ancestry).path = {1, 2, 3, 4};
        value.domain.assign(160, 'd');
        value.parent_family.assign(180, 'p');
        AMR::AdaptiveSurface surface;
        surface.key.assign(200, 's');
        surface.boundary.assign(220, 'b');
        value.surfaces.push_back(std::move(surface));
    }
    auto source_arena = std::make_shared<D::Budget>(D::Budget{ample});
    std::size_t bytes = 0;
    for (const auto& value : input)
        bytes += Pages::buffer_bytes(value);
    std::size_t next = 0;
    auto source = Pages::build_moving(input.size(), [&]() -> AMR::AdaptiveLeaf&& { return std::move(input[next++]); },
        source_arena, bytes);
    const auto first_key = source.at_ordinal(0).key;
    const auto edited_key = source.at_ordinal(128).key;
    auto replacement = source.at(edited_key);
    replacement.revision = 1;
    const Pages::Edit edit{edited_key, 0, replacement};
    auto measured_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto changed = Pages::edit(source, std::span(&edit, 1), measured_arena);
    const auto peak = measured_arena->peak;
    ASSERT_GT(peak, 0u);
    auto tight = std::make_shared<D::Budget>(D::Budget{peak - 1});
    EXPECT_THROW((void) Pages::edit(source, std::span(&edit, 1), tight), std::invalid_argument);
    EXPECT_EQ(tight->used, 0u);
    EXPECT_EQ(source.at(edited_key).revision, 0u);
    auto exact = std::make_shared<D::Budget>(D::Budget{peak});
    auto retry = Pages::edit(source, std::span(&edit, 1), exact);
    EXPECT_EQ(source.page_identity(first_key), changed.page_identity(first_key));
    source = {};
    EXPECT_GT(source_arena->used, 0u);
    const auto& retained = changed.at(first_key);
    EXPECT_EQ(retained.domain.size(), 160u);
    EXPECT_EQ(retained.parent_family.size(), 180u);
    ASSERT_EQ(retained.surfaces.size(), 1u);
    EXPECT_EQ(retained.surfaces[0].key.size(), 200u);
    EXPECT_EQ(retained.surfaces[0].boundary.size(), 220u);
    EXPECT_EQ(std::get<AMR::ConvexCellAncestry>(retained.key.ancestry).path.size(), 4u);
    changed = {};
    retry = {};
    EXPECT_EQ(source_arena->used, 0u);
}

TEST(AdaptiveLeafLayout, MovingBuildRejectsUnsortedAndDuplicatePageBoundary)
{
    for (bool duplicate : {false, true})
    {
        auto input = leaves(128);
        input[64].key = duplicate ? input[63].key : input[62].key;
        auto arena = std::make_shared<D::Budget>(D::Budget{ample});
        std::size_t bytes = 0;
        for (const auto& value : input)
            bytes += Pages::buffer_bytes(value);
        std::size_t next = 0;
        EXPECT_THROW((void) Pages::build_moving(input.size(),
                         [&]() -> AMR::AdaptiveLeaf&& { return std::move(input[next++]); }, arena, bytes),
            std::invalid_argument);
        EXPECT_EQ(arena->used, 0u);
    }
}

TEST(AdaptiveLeafLayout, BudgetBoundaryRejectsAtomicallyAndRetainedRootsKeepTheirCharge)
{
    const auto input = leaves(512);
    auto base_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto original = Pages::build(input, base_arena);
    const auto old_used = base_arena->used;
    const Pages::Edit edit{key(256), 0, leaf(256, 1)};
    auto measured_arena = std::make_shared<D::Budget>(D::Budget{ample});
    auto candidate = Pages::edit(original, std::span(&edit, 1), measured_arena);
    const auto required_peak = measured_arena->peak;
    ASSERT_GT(required_peak, 0u);
    auto tight_arena = std::make_shared<D::Budget>(D::Budget{required_peak - 1});
    EXPECT_THROW((void) Pages::edit(original, std::span(&edit, 1), tight_arena), std::invalid_argument);
    EXPECT_EQ(tight_arena->used, 0u);
    EXPECT_EQ(base_arena->used, old_used);
    EXPECT_EQ(original.at(key(256)).revision, 0u);
    auto exact_arena = std::make_shared<D::Budget>(D::Budget{required_peak});
    auto retry = Pages::edit(original, std::span(&edit, 1), exact_arena);
    EXPECT_EQ(retry.at(key(256)).revision, 1u);
    original = {};
    EXPECT_GT(base_arena->used, 0u); // candidate and retry pin unchanged base pages.
    EXPECT_EQ(candidate.at(key(0)).revision, 0u);
    candidate = {};
    retry = {};
    EXPECT_EQ(base_arena->used, 0u);
    EXPECT_EQ(measured_arena->used, 0u);
    EXPECT_EQ(exact_arena->used, 0u);
}
} // namespace
