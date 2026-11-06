/** @file testAdaptiveLeafBodyUsage.cc
 * @brief Physical owner, changed-page and rejection oracles for sparse leaf edits.
 */
#include "geometry/mesh/AMR/AdaptiveLeafBodyUsage.hh"
#include "geometry/mesh/AMR/AdaptiveLeafOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "utils/testing_environment.hh"

#include <Tpetra_Core.hpp>
#include <array>

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
using Descriptor = SimpleFluid::Meshes::NativeFragmentDescriptor;
using Pages = D::AdaptiveLeafPages;
constexpr auto ample = std::size_t{64} * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<D::Budget> arena(std::size_t bytes = ample)
{
    return std::make_shared<D::Budget>(D::Budget{bytes});
}

AMR::AdaptiveLeaf leaf(std::uint64_t root, std::shared_ptr<const Descriptor> recipe)
{
    AMR::AdaptiveLeaf row;
    row.key = {0, AMR::ConvexCellAncestry{root, {}}};
    row.native_recipe = std::move(recipe);
    return row;
}

std::size_t descriptor_bytes(const Descriptor& recipe)
{
    return sizeof(Descriptor) + recipe.view.capacity() + recipe.parts.capacity() * sizeof(decltype(recipe.parts)::value_type);
}

D::AdaptiveLeafBodyUsage initial_usage(const Pages& pages, const std::shared_ptr<D::Budget>& memory)
{
    D::AdaptiveLeafBodyDelta census(memory);
    pages.visit_leaf_pages([&](auto rows) { census.record(rows, true); });
    return census.apply({}).usage;
}

struct Change
{
    Pages pages;
    D::AdaptiveLeafBodyDelta::Result bodies;
};

Change edit(const Pages& pages, const D::AdaptiveLeafBodyUsage& usage, std::span<const Pages::EditView> edits,
    const std::shared_ptr<D::Budget>& memory)
{
    D::AdaptiveLeafBodyDelta census(memory);
    const Pages::PageChanges changes{
        [&](auto rows) { census.record(rows, false); }, [&](auto rows) { census.record(rows, true); }};
    auto edited = Pages::edit_borrowed(pages, edits, memory, nullptr, &changes);
    return {std::move(edited), census.apply(usage)};
}

TEST(AdaptiveLeafBodyUsage, SharedBodySurvivesOnePageRetirementAndDropsAfterLast)
{
    auto recipe = std::make_shared<Descriptor>();
    recipe->parts.push_back(std::make_shared<const AMR::AdaptiveBytes>(137, 'a'));
    const auto bytes = descriptor_bytes(*recipe) + sizeof(AMR::AdaptiveBytes) + recipe->parts.front()->capacity();
    std::vector<AMR::AdaptiveLeaf> rows;
    for (std::size_t i = 0; i < 128; ++i)
        rows.push_back(leaf(i, recipe));
    const auto first_arena = arena();
    const auto source = Pages::build(rows, first_arena);
    const auto usage = initial_usage(source, first_arena);
    ASSERT_EQ(usage.payload_bytes(), bytes);
    EXPECT_EQ(first_arena->used, source.retained_bytes() + usage.storage_bytes());

    std::vector<Pages::EditView> retiring;
    for (std::size_t i = 0; i < 64; ++i)
        retiring.push_back({rows[i].key, 0, nullptr});
    const auto second_arena = arena();
    const auto half = edit(source, usage, retiring, second_arena);
    ASSERT_EQ(half.pages.size(), 64);
    EXPECT_EQ(half.pages.page_identity(rows[64].key), source.page_identity(rows[64].key));
    EXPECT_EQ(half.bodies.new_payload_bytes, 0);
    EXPECT_EQ(half.bodies.reused_payload_bytes, bytes);
    EXPECT_EQ(half.bodies.usage.payload_bytes(), bytes);
    EXPECT_EQ(second_arena->used, half.bodies.usage.storage_bytes());

    retiring.clear();
    for (std::size_t i = 64; i < rows.size(); ++i)
        retiring.push_back({rows[i].key, 0, nullptr});
    const auto final_arena = arena();
    const auto empty = edit(half.pages, half.bodies.usage, retiring, final_arena);
    EXPECT_TRUE(empty.pages.empty());
    EXPECT_EQ(empty.bodies.usage.payload_bytes(), 0);
    EXPECT_EQ(empty.bodies.reused_payload_bytes, 0);
    EXPECT_EQ(final_arena->used, 0);
    EXPECT_EQ(usage.payload_bytes(), bytes);
}

TEST(AdaptiveLeafBodyUsage, EqualIndependentDescriptorsHaveSeparateChargesAndRollback)
{
    auto recipe = std::make_shared<Descriptor>();
    recipe->parts.push_back(std::make_shared<const AMR::AdaptiveBytes>(83, 'b'));
    auto independent = std::make_shared<const Descriptor>(*recipe);
    const auto original_bytes =
        descriptor_bytes(*recipe) + sizeof(AMR::AdaptiveBytes) + recipe->parts.front()->capacity();
    std::vector<AMR::AdaptiveLeaf> rows;
    for (std::size_t i = 0; i < 128; ++i)
        rows.push_back(leaf(i, recipe));
    const auto accepted_arena = arena();
    const auto source = Pages::build(rows, accepted_arena);
    const auto usage = initial_usage(source, accepted_arena);
    const auto original_charge = accepted_arena->used;

    D::Budget overlay_memory{ample};
    D::AdaptiveLeafOverlay overlay(source, overlay_memory);
    ASSERT_TRUE(overlay.retire(rows.front().key));
    auto replacement = leaf(0, independent);
    replacement.revision = 1;
    ASSERT_TRUE(overlay.insert(std::move(replacement)));
    ASSERT_EQ(overlay.edit_count(), 1);
    EXPECT_EQ(overlay.size(), source.size());
    EXPECT_EQ(overlay.find(rows[64].key), source.find(rows[64].key));
    EXPECT_EQ(overlay.find(rows[0].key)->revision, 1);
    std::vector<Pages::EditView> edits;
    overlay.visit_edits([&](auto change) { edits.push_back(change); });

    EXPECT_THROW((void) edit(source, usage, edits, accepted_arena), std::invalid_argument);
    EXPECT_EQ(accepted_arena->used, original_charge);

    const auto measured = arena();
    {
        const auto next = edit(source, usage, edits, measured);
        EXPECT_EQ(next.bodies.reused_payload_bytes, original_bytes);
        EXPECT_EQ(next.bodies.new_payload_bytes, descriptor_bytes(*independent));
        EXPECT_EQ(next.bodies.usage.payload_bytes(), original_bytes + descriptor_bytes(*independent));
        EXPECT_EQ(next.pages.page_identity(rows[64].key), source.page_identity(rows[64].key));
    }
    ASSERT_GT(measured->peak, 0);
    EXPECT_EQ(measured->used, 0);
    const auto exact = arena(measured->peak);
    EXPECT_NO_THROW((void) edit(source, usage, edits, exact));
    EXPECT_EQ(exact->used, 0);
    const auto shortfall = arena(measured->peak - 1);
    EXPECT_THROW((void) edit(source, usage, edits, shortfall), std::exception);
    EXPECT_EQ(shortfall->used, 0);
    EXPECT_EQ(accepted_arena->used, original_charge);
    EXPECT_EQ(source.at(rows[0].key).revision, 0);
}

TEST(AdaptiveLeafBodyUsage, SavedParentReusesBodyRetiredByLeafSeal)
{
    auto recipe = std::make_shared<Descriptor>();
    recipe->parts.push_back(std::make_shared<const AMR::AdaptiveBytes>(67, 'r'));
    const std::array rows{leaf(0, recipe)};
    const auto source_arena = arena();
    const auto source = Pages::build(rows, source_arena);
    const auto accepted = initial_usage(source, source_arena);
    const std::array<Pages::EditView, 1> retired{{{rows.front().key, 0, nullptr}}};
    const auto next_arena = arena();
    const auto sealed = edit(source, accepted, retired, next_arena);
    ASSERT_TRUE(sealed.pages.empty());
    ASSERT_EQ(sealed.bodies.usage.payload_bytes(), 0);

    D::AdaptiveLeafBodyDelta saved(next_arena);
    saved.record(rows, true);
    const auto family = saved.apply(sealed.bodies.usage, D::AdaptiveLeafBodyRole::Family, &accepted);
    EXPECT_EQ(family.usage.leaf_payload_bytes(), 0);
    EXPECT_EQ(family.usage.payload_bytes(), accepted.payload_bytes());
    EXPECT_EQ(family.new_payload_bytes, 0);
    EXPECT_EQ(family.reused_payload_bytes, accepted.payload_bytes());
}

TEST(AdaptiveLeafBodyUsage, ReplacedLeafAndSavedParentKeepSeparatePhysicalCredits)
{
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    AMR::CartesianAMR mesh(grid);
    const std::array<std::uint64_t, 1> selection{0};
    auto detached = mesh.detach_owned(selection, {'a'});
    auto parent = detached.owned.front();
    parent.geometry = detached.kernel->geometry(parent);
    auto replacement = parent;
    replacement.geometry = detached.kernel->geometry(parent);
    replacement.native_recipe = std::make_shared<const Descriptor>(*parent.native_recipe);
    replacement.revision = 1;

    const auto accepted_arena = arena();
    const std::array<AMR::AdaptiveLeaf, 1> accepted_rows{parent};
    const auto source = Pages::build(accepted_rows, accepted_arena);
    auto usage = initial_usage(source, accepted_arena);
    const auto original_payload = usage.payload_bytes();
    {
        D::AdaptiveLeafBodyDelta family(accepted_arena);
        family.record(accepted_rows, true);
        auto joint = family.apply(usage, D::AdaptiveLeafBodyRole::Family);
        EXPECT_EQ(joint.new_payload_bytes, 0);
        EXPECT_EQ(joint.usage.payload_bytes(), original_payload);
        usage = std::move(joint.usage);
    }
    const auto source_bytes = accepted_arena->used;

    const auto reproduce = [&](const std::shared_ptr<D::Budget>& memory)
    {
        const std::array<Pages::EditView, 1> edits{{{parent.key, 0, &replacement}}};
        const auto next = edit(source, usage, edits, memory);
        EXPECT_EQ(next.pages.at(parent.key).revision, 1);
        EXPECT_GT(next.bodies.new_payload_bytes, 0);
        EXPECT_GT(next.bodies.usage.payload_bytes(), next.bodies.usage.leaf_payload_bytes());
        // Simulate a packet whose dirty leaf was normalized but whose parent
        // still owns the original decoded recipe and certified body.
        D::AdaptiveLeafBodyDelta saved_parent(memory);
        saved_parent.record(accepted_rows, true);
        const auto migrated = saved_parent.apply(next.bodies.usage, D::AdaptiveLeafBodyRole::Family, &usage);
        EXPECT_EQ(migrated.new_payload_bytes, 0);
        EXPECT_EQ(migrated.usage.payload_bytes(), next.bodies.usage.payload_bytes());
    };
    const auto measured = arena();
    ASSERT_NO_THROW(reproduce(measured));
    ASSERT_GT(measured->peak, 0);
    EXPECT_EQ(measured->used, 0);
    const auto exact = arena(measured->peak);
    EXPECT_NO_THROW(reproduce(exact));
    EXPECT_EQ(exact->used, 0);
    const auto shortfall = arena(measured->peak - 1);
    EXPECT_THROW(reproduce(shortfall), std::exception);
    EXPECT_EQ(shortfall->used, 0);
    EXPECT_EQ(accepted_arena->used, source_bytes);
    EXPECT_EQ(source.at(parent.key).revision, 0);

    const auto receiver_arena = arena();
    const std::array<AMR::AdaptiveLeaf, 1> normalized_rows{replacement};
    const auto received = Pages::build(normalized_rows, receiver_arena);
    const auto leaf_only = initial_usage(received, receiver_arena);
    const D::AdaptiveLeafBodyUsage empty_source;
    D::AdaptiveLeafBodyDelta received_family(receiver_arena);
    received_family.record(accepted_rows, true);
    const auto retained = received_family.apply(leaf_only, D::AdaptiveLeafBodyRole::Family, &empty_source);
    EXPECT_GT(retained.new_payload_bytes, 0);
    EXPECT_EQ(retained.new_payload_bytes, retained.usage.payload_bytes() - retained.usage.leaf_payload_bytes());
    EXPECT_EQ(retained.usage.leaf_payload_bytes(), leaf_only.payload_bytes());
}

TEST(AdaptiveLeafBodyUsage, SavedCartesianParentSurvivesEditAndMigrationWithoutEscapingLeafLayout)
{
    const auto comm = Tpetra::getDefaultComm();
    AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 1, 1};
    grid.max_level = 1;
    AMR::AdaptiveCellKey left, right;
    auto detached = [&]
    {
        AMR::CartesianAMR mesh(grid);
        left = mesh.cell_key(0);
        right = mesh.cell_key(8);
        std::vector<std::uint64_t> selection;
        if (comm->getRank() == 0)
            for (std::uint64_t cell = 0; cell < 16; ++cell)
                selection.push_back(cell);
        return mesh.detach_owned(selection, {'p'});
    }();
    using Hierarchy = AMR::DistributedAdaptiveHierarchy;
    auto source = std::make_shared<const Hierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
    const auto family_key = AMR::adaptive_family_key(left, AMR::AdaptiveSplit::Isotropic);
    const auto refine = [](const std::shared_ptr<const Hierarchy>& state, const AMR::AdaptiveCellKey& key)
    {
        std::vector<AMR::AdaptiveMark> marks;
        if (state->owned_leaves().contains(key))
            marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
        return state->apply(state->plan_adaptation(marks));
    };
    const auto saved = [&](const Hierarchy& state)
    {
        const auto parent = state.families().find(family_key);
        return parent == state.families().end() ? std::weak_ptr<const Descriptor>{} : parent->second.parent.native_recipe;
    };

    auto first = refine(source, left);
    source.reset();
    const auto first_parent = saved(*first);
    if (comm->getRank() == 0)
        EXPECT_FALSE(first_parent.expired());
    auto second = refine(first, right);
    first.reset();
    if (comm->getRank() == 0)
    {
        EXPECT_FALSE(first_parent.expired());
        EXPECT_EQ(first_parent.lock(), saved(*second).lock());
    }

    const auto destination = comm->getSize() - 1;
    const std::vector<int> destinations(second->owned_leaves().size(), destination);
    auto migration = second->plan_migration(destinations);
    EXPECT_EQ(migration.native_migration().self_payload_bytes, 0);
    auto moved = second->apply(migration);
    second.reset();
    const auto escaped = moved->leaf_layout();
    const auto final_parent = saved(*moved);
    if (comm->getRank() == destination)
    {
        const auto family = moved->families().find(family_key);
        EXPECT_NE(family, moved->families().end());
        if (family != moved->families().end())
        {
            EXPECT_DOUBLE_EQ(moved->metrics(family->second.parent).volume, 0.125);
            for (const auto& row : *escaped)
                EXPECT_NE(row.native_recipe, family->second.parent.native_recipe);
        }
        EXPECT_FALSE(final_parent.expired());
    }
    moved.reset();
    // Release the plan too before checking that the layout pins no capsule.
    migration = {};
    EXPECT_TRUE(final_parent.expired());
    EXPECT_EQ(escaped->size(), comm->getRank() == destination ? 128 : 0);
}
} // namespace
