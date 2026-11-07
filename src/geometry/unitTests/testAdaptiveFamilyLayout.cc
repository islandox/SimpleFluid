/** @file testAdaptiveFamilyLayout.cc
 * @brief Saved-family sharing, anchor selection and physical admission oracles.
 */
#include "geometry/mesh/AMR/AdaptiveFamilyLayout.hh"
#include "geometry/mesh/AMR/AdaptiveLeafBodyUsage.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <gtest/gtest.h>

#include <array>
#include <map>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
using Storage = D::AdaptiveFamilyStorage;
using Layout = AMR::AdaptiveFamilyLayout;
using Descriptor = SimpleFluid::Meshes::NativeFragmentDescriptor;
constexpr std::size_t ample = std::size_t{128} * 1024 * 1024;

std::shared_ptr<D::Budget> arena(std::size_t cap = ample)
{
    return std::make_shared<D::Budget>(D::Budget{cap});
}

AMR::AdaptiveFamily family(std::uint64_t root, bool product = false)
{
    AMR::AdaptiveFamily result;
    if (product)
    {
        result.parent.provider = AMR::AdaptiveProviderKind::Product;
        result.parent.key = {8, AMR::ProductCellAncestry{{root, 1, {1, 2, 3}}, {root, {0, 1}}}};
        result.split = AMR::AdaptiveSplit::Axial;
    }
    else
        result.parent.key = {7, AMR::ConvexCellAncestry{root, std::vector<std::uint64_t>(12, 1)}};
    result.parent.revision = 13;
    result.revision = 2;
    result.parent.domain.assign(89, 'd');
    result.parent.parent_family.assign(96, 'p');
    result.parent.surfaces.resize(3);
    for (auto& surface : result.parent.surfaces)
    {
        surface.key.assign(93, 's');
        surface.boundary.assign(63, 'b');
    }
    result.key = AMR::adaptive_family_key(result.parent.key, result.split);
    for (std::uint64_t side = 0; side < 2; ++side)
    {
        auto child = result.parent.key;
        if (product)
            std::get<AMR::ProductCellAncestry>(child.ancestry).axial.path.push_back(side);
        else
            std::get<AMR::ConvexCellAncestry>(child.ancestry).path.push_back(side);
        result.children.push_back(std::move(child));
    }
    result.anchor = AMR::encode_adaptive_key(result.children.front());
    return result;
}

auto allocation_union(const Layout& first, const Layout& second)
{
    std::map<const void*, std::size_t> owners;
    auto retain = [&](const void* identity, std::size_t bytes)
    {
        const auto [where, fresh] = owners.emplace(identity, bytes);
        if (!fresh)
            EXPECT_EQ(where->second, bytes);
    };
    Storage::visit_storage(first, retain);
    Storage::visit_storage(second, retain);
    std::size_t bytes = 0;
    for (const auto& [identity, size] : owners)
        bytes += size;
    return bytes;
}

TEST(AdaptiveFamilyLayout, FixedEditSharesBothSelectorsAndOnlyCopiesAffectedPages)
{
    auto measure = [](std::size_t count)
    {
        std::vector<AMR::AdaptiveFamily> input;
        input.reserve(count);
        for (std::size_t i = count; i > 0; --i)
            input.push_back(family(i, i % 2));
        auto source_arena = arena();
        auto source = Storage::build(input, source_arena, 41);
        EXPECT_EQ(source_arena->used, Storage::storage_bytes(source));
        std::vector<AMR::AdaptiveBytes> expected;
        for (const auto& value : input)
            expected.push_back(value.key);
        std::sort(expected.begin(), expected.end());
        std::size_t ordinal = 0;
        for (const auto row : source)
        {
            EXPECT_EQ(row.key, expected.at(ordinal));
            EXPECT_EQ(&row.parent, &source.at_ordinal(ordinal).parent);
            ++ordinal;
        }
        EXPECT_EQ(ordinal, count);
        EXPECT_FALSE(source.find({}));
        EXPECT_THROW((void) source.at_ordinal(count), std::out_of_range);

        auto replacement = input.back();
        const auto old_anchor = replacement.anchor;
        replacement.anchor = AMR::encode_adaptive_key(replacement.children.back());
        const auto clean_key = input.front().key;
        const auto clean_anchor = input.front().anchor;
        const auto* clean_parent = &source.at(clean_key).parent;
        const std::array edits{Storage::Edit{replacement.key, 41, &replacement}};
        Storage::Work work;
        auto target_arena = arena();
        std::size_t changes = 0;
        auto target = Storage::edited(source, edits, target_arena, 42, work,
            [&](const AMR::AdaptiveLeaf* before, const AMR::AdaptiveLeaf* after)
            {
                ++changes;
                EXPECT_EQ(before, &source.at(replacement.key).parent);
                ASSERT_NE(before, nullptr);
                ASSERT_NE(after, nullptr);
                EXPECT_NE(after, before);
                EXPECT_EQ(after->key, before->key);
            });
        EXPECT_EQ(changes, 1u);
        EXPECT_EQ(work.records, 1u);
        EXPECT_EQ(&target.at(clean_key).parent, clean_parent);
        EXPECT_EQ(target.at(clean_key).revision, 2u);
        EXPECT_EQ(target.at(replacement.key).revision, 42u);
        EXPECT_EQ(target.at(replacement.key).parent.revision, 13u);
        EXPECT_EQ(source.at(replacement.key).anchor, old_anchor);
        EXPECT_EQ(target.at(replacement.key).materialize().anchor, replacement.anchor);
        target.visit_anchored(old_anchor, [](auto) { ADD_FAILURE() << "Retired anchor still resolves"; });
        std::size_t matching = 0;
        target.visit_anchored(replacement.anchor,
            [&](auto row)
            {
                ++matching;
                EXPECT_EQ(row.key, replacement.key);
                EXPECT_EQ(&row.parent, &target.at(replacement.key).parent);
            });
        EXPECT_EQ(matching, 1u);
        target.visit_anchored(clean_anchor, [&](auto row) { EXPECT_EQ(&row.parent, clean_parent); });
        EXPECT_EQ(target_arena->used, allocation_union(source, target));
        EXPECT_GT(Storage::overlap_bytes(target), 0u);
        Storage::finish_overlap(target);
        EXPECT_EQ(target_arena->used, Storage::storage_bytes(target));
        EXPECT_EQ(Storage::overlap_bytes(target), 0u);
        return work;
    };
    const auto small = measure(4096);
    const auto large = measure(8192);
    EXPECT_EQ(large.records, small.records);
    EXPECT_EQ(large.record_bytes, small.record_bytes);
    EXPECT_LT(large.copied_pages, 2 * small.copied_pages);
    EXPECT_LT(large.copied_page_bytes, 2 * small.copied_page_bytes);
}

TEST(AdaptiveFamilyLayout, FailedAdmissionOrStaleRevisionLeavesBothRootsUntouched)
{
    const std::array source_rows{family(5, true), family(8)};
    const auto source_arena = arena();
    auto source = Storage::build(source_rows, source_arena, 17);
    auto replacement = source_rows.front();
    replacement.anchor = AMR::encode_adaptive_key(replacement.children.back());
    const auto old_storage = Storage::storage_bytes(source);
    const std::array edits{Storage::Edit{replacement.key, 17, &replacement}};
    const auto measured = arena();
    const auto ignore = [](const auto*, const auto*) {};
    Storage::Work work;
    auto target = Storage::edited(source, edits, measured, 18, work, ignore);
    EXPECT_EQ(target.at(replacement.key).anchor, replacement.anchor);
    const auto exact = measured->peak;
    ASSERT_GT(exact, old_storage);
    auto one_short = arena(exact - 1);
    Storage::Work failed;
    EXPECT_THROW((void) Storage::edited(source, edits, one_short, 18, failed,
                     [](const auto*, const auto*) { ADD_FAILURE() << "Rejected edit emitted a parent delta"; }),
        std::invalid_argument);
    EXPECT_EQ(one_short->used, 0u);
    EXPECT_EQ(failed.records, 0u);
    EXPECT_EQ(source_arena->used, old_storage);
    EXPECT_EQ(source.at(replacement.key).anchor, source_rows.front().anchor);
    EXPECT_EQ(source.at(replacement.key).revision, 17u);
    one_short->limit = exact;
    auto retry = Storage::edited(source, edits, one_short, 18, failed, ignore);
    EXPECT_EQ(retry.at(replacement.key).anchor, replacement.anchor);
    const std::array stale{Storage::Edit{replacement.key, 2, &replacement}};
    const auto bad = arena();
    EXPECT_THROW((void) Storage::edited(source, stale, bad, 18, failed, ignore), std::invalid_argument);
    EXPECT_EQ(bad->used, 0u);
}

TEST(AdaptiveFamilyLayout, SecondParentCallbackFailureRetiresPrivateCandidateAndDelta)
{
    std::array rows{family(7), family(3)};
    std::ranges::sort(rows, {}, &AMR::AdaptiveFamily::key);
    auto recipe = std::make_shared<Descriptor>();
    recipe->parts.push_back(std::make_shared<const AMR::AdaptiveBytes>(83, 'p'));
    for (auto& row : rows)
        row.parent.native_recipe = recipe;
    const auto source_arena = arena(), candidate_arena = arena();
    const auto source = Storage::build(rows, source_arena, 17);
    const auto source_bytes = source_arena->used;
    const std::array parents{&source.at(rows[0].key).parent, &source.at(rows[1].key).parent};
    auto replacements = rows;
    for (auto& row : replacements)
        row.anchor = AMR::encode_adaptive_key(row.children.back());
    const std::array edits{Storage::Edit{rows[0].key, rows[0].revision, &replacements[0]},
        Storage::Edit{rows[1].key, rows[1].revision, &replacements[1]}};
    Storage::Work work{7, 11, 13, 17};
    std::size_t calls = 0;
    {
        D::AdaptiveLeafBodyDelta delta(candidate_arena);
        EXPECT_THROW((void) Storage::edited(source, edits, candidate_arena, 18, work,
                         [&](const AMR::AdaptiveLeaf* before, const AMR::AdaptiveLeaf* after)
                         {
                             ASSERT_LT(calls, parents.size());
                             ASSERT_EQ(before, parents[calls]);
                             ASSERT_NE(after, nullptr);
                             EXPECT_EQ(after->key, parents[calls]->key);
                             delta.record(std::span(before, 1), false);
                             delta.record(std::span(after, 1), true);
                             if (++calls == 2)
                                 throw std::runtime_error("second parent callback rejected");
                         }),
            std::runtime_error);
        EXPECT_EQ(calls, 2u);
        EXPECT_GT(delta.metadata_bytes(), 0u);
        EXPECT_EQ(candidate_arena->used, delta.metadata_bytes());
    }
    EXPECT_EQ(candidate_arena->used, 0u);
    EXPECT_EQ(source_arena->used, source_bytes);
    EXPECT_EQ(work.copied_pages, 7u);
    EXPECT_EQ(work.copied_page_bytes, 11u);
    EXPECT_EQ(work.records, 13u);
    EXPECT_EQ(work.record_bytes, 17u);
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        EXPECT_EQ(&source.at(rows[i].key).parent, parents[i]);
        EXPECT_EQ(source.at(rows[i].key).anchor, rows[i].anchor);
        std::size_t matches = 0;
        source.visit_anchored(rows[i].anchor,
            [&](const auto& row)
            {
                ++matches;
                EXPECT_EQ(row.key, rows[i].key);
                EXPECT_EQ(&row.parent, parents[i]);
            });
        EXPECT_EQ(matches, 1u);
        source.visit_anchored(replacements[i].anchor, [](auto) { ADD_FAILURE() << "Private anchor was published"; });
    }
}

TEST(AdaptiveFamilyLayout, OwnerDeltaAndProductEpochKeepOnlyStillReferencedParentBodies)
{
    auto product = family(3, true);
    auto other = family(7);
    const auto product_key = product.key, other_key = other.key;
    auto recipe = std::make_shared<Descriptor>();
    recipe->parts.push_back(std::make_shared<const AMR::AdaptiveBytes>(173, 'r'));
    const std::weak_ptr<const Descriptor> weak_recipe = recipe;
    product.parent.native_recipe = recipe;
    recipe.reset();
    const auto first = arena(), second = arena(), third = arena();
    auto old = std::make_unique<Layout>(Storage::build(std::array{product, other}, first, 31));
    product.parent.native_recipe.reset();
    D::AdaptiveLeafBodyUsage accepted;
    {
        D::AdaptiveLeafBodyDelta census(first);
        for (const auto row : *old)
            census.record(std::span(&row.parent, 1), true);
        accepted = census.apply({}, D::AdaptiveLeafBodyRole::Family).usage;
    }
    ASSERT_GT(accepted.payload_bytes(), 0u);
    EXPECT_EQ(accepted.leaf_payload_bytes(), 0u);
    Storage::Work work;
    const std::array first_edits{Storage::Edit{other_key, other.revision, nullptr}};
    D::AdaptiveLeafBodyUsage after_retire;
    auto next = [&]
    {
        D::AdaptiveLeafBodyDelta delta(second);
        auto value = Storage::edited(*old, first_edits, second, 32, work,
            [&](const auto* before, const auto* after)
            {
                EXPECT_NE(before, nullptr);
                EXPECT_EQ(after, nullptr);
                delta.record(std::span(before, 1), false);
            });
        auto result = delta.apply(accepted, D::AdaptiveLeafBodyRole::Family, &accepted);
        EXPECT_EQ(result.new_payload_bytes, 0u);
        EXPECT_EQ(result.reused_payload_bytes, accepted.payload_bytes());
        after_retire = std::move(result.usage);
        Storage::finish_overlap(value);
        return std::make_unique<Layout>(std::move(value));
    }();
    EXPECT_EQ(&next->at(product_key).parent, &old->at(product_key).parent);
    EXPECT_EQ(old->at(product_key).revision, 31u);
    EXPECT_EQ(next->at(product_key).revision, 32u);
    EXPECT_EQ(next->at(product_key).parent.revision, 13u);
    EXPECT_FALSE(next->contains(other_key));
    old.reset();
    EXPECT_FALSE(weak_recipe.expired());

    const std::array second_edits{Storage::Edit{product_key, 32, nullptr}};
    D::AdaptiveLeafBodyDelta delta(third);
    auto empty = Storage::edited(*next, second_edits, third, 33, work,
        [&](const auto* before, const auto* after)
        {
            EXPECT_EQ(after, nullptr);
            delta.record(std::span(before, 1), false);
        });
    const auto result = delta.apply(after_retire, D::AdaptiveLeafBodyRole::Family, &after_retire);
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(result.usage.payload_bytes(), 0u);
    EXPECT_FALSE(weak_recipe.expired());
    Storage::finish_overlap(empty);
    next.reset();
    EXPECT_TRUE(weak_recipe.expired());
}
} // namespace
