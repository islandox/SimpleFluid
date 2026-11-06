/** Retention and binding regressions for source-only adaptive geometry. */
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <set>
#include <string_view>
#include <utility>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace PLIC = SimpleFluid::PLIC;
namespace R = PLIC::details::retention;
using SimpleFluid::ArithmeticMode;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t limit = 64 * 1024 * 1024;
using Hierarchy = AMR::DistributedAdaptiveHierarchy;
using View = PLIC::CellGeometryView<ordinary>;
using Access = PLIC::details::NativeCellGeometrySourceAccess;
using Snapshot = AMR::AdaptiveGeometrySnapshot;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::vector<std::uint64_t> indices(std::size_t size)
{
    std::vector<std::uint64_t> result(size);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
std::shared_ptr<const Hierarchy> hierarchy(AMR::AdaptiveBootstrap input)
{
    return std::make_shared<const Hierarchy>(
        std::move(input.owned), std::move(input.families), std::move(input.kernel), Tpetra::getDefaultComm());
}
std::shared_ptr<const Hierarchy> cartesian_grid_source(const AMR::AMRGrid& grid)
{
    const AMR::CartesianAMR source(grid);
    return hierarchy(source.detach_owned(indices(source.geometry()->num_cells()), {'c', 'a', 'r', 't'}));
}
std::shared_ptr<const Hierarchy> cartesian_source()
{
    return cartesian_grid_source(AMR::AMRGrid{});
}
std::shared_ptr<const Hierarchy> product_source()
{
    const AMR::ExtrudedAMR source(SimpleFluid::test::mixed_product_grid(true));
    return hierarchy(source.detach_owned(indices(source.geometry()->num_cells()), {'p', 'r', 'o', 'd'}));
}
std::shared_ptr<const Hierarchy> polyhedral_source()
{
    const auto source = SimpleFluid::test::mixed_polyhedron()->adapt({0});
    return hierarchy(source->detach_owned(indices(source->geometry()->num_cells()), {'p', 'o', 'l', 'y'}));
}
std::shared_ptr<const Hierarchy> static_source()
{
    AMR::CartesianAMR source(AMR::AMRGrid{});
    return hierarchy(AMR::detach_static_participant(source.geometry()->regions().front(), 1, 2, 3,
        indices(source.geometry()->num_cells()), {'s', 't', 'a', 't', 'i', 'c'}));
}
std::shared_ptr<const Hierarchy> mixed_source()
{
    auto grid = SimpleFluid::test::mixed_joined_grid();
    const auto product = std::get<AMR::ProductCellAncestry>(grid.product->cell_key(0, 0).ancestry);
    grid.product = grid.product->adapt({product}, {product});
    grid.adaptive.front() = std::dynamic_pointer_cast<const AMR::PolyhedralAMR>(grid.adaptive.front())->adapt({0});
    const AMR::MixedAMR source(std::move(grid));
    return hierarchy(source.detach_owned(indices(source.geometry()->num_cells()), {'m', 'i', 'x', 'e', 'd'}));
}
std::shared_ptr<const Snapshot> snapshot(const Hierarchy& source, std::span<const AMR::AdaptiveLeaf* const> rows)
{
    std::size_t used = 0;
    return source.geometry_snapshot(rows, ordinary, AMR::GeometrySnapshotPurpose::State, used, limit);
}
std::map<const void*, std::size_t> compact_owners(const Snapshot& snapshot, const Hierarchy& hierarchy)
{
    std::set<const void*> recipes;
    for (const auto& row : hierarchy.owned_leaves())
    {
        if (const auto& native = row.native_recipe)
        {
            recipes.insert(native.get());
            for (const auto& part : native->parts)
                recipes.insert(part.get());
        }
        recipes.insert(row.join_recipe.get());
        if (const auto& factor = row.factor_recipe)
        {
            recipes.insert(factor.get());
            recipes.insert(factor->root.get());
            recipes.insert(factor->base.get());
            recipes.insert(factor->axis.get());
        }
    }
    std::map<const void*, std::size_t> owners;
    snapshot.visit_external_owners(
        [&](const void* owner, std::size_t bytes)
        {
            if (!recipes.contains(owner))
                owners.emplace(owner, bytes);
        });
    return owners;
}
// Fixed portions of the previous per-row layout. Dynamic keys/domains and
// surface descriptors are deliberately excluded from this lower bound.
struct PreviousBindingRow
{
    AMR::AdaptiveCellKey key;
    AMR::AdaptiveBytes domain;
    std::array<std::uint64_t, 3> cell_metadata;
    AMR::AdaptiveProviderKind provider;
    std::shared_ptr<const SimpleFluid::Meshes::NativeFragmentDescriptor> native;
    std::shared_ptr<const AMR::AdaptiveBytes> join;
    std::shared_ptr<const AMR::AdaptiveFactorRecipe> factor;
    PLIC::ConvexGeometryPtr<ordinary> geometry;
    std::vector<AMR::GeometrySnapshotSurface> surfaces;
};
constexpr std::size_t previous_binding_bytes =
    sizeof(PreviousBindingRow) + sizeof(std::pair<const AMR::AdaptiveCellKey, std::size_t>) + 4 * sizeof(void*) +
    sizeof(AMR::AdaptiveCellMetrics) + sizeof(AMR::GeometryEvaluationPolicy);
} // namespace

TEST(AdaptiveGeometryStorage, ProviderSnapshotsShareBoundPagesAndPreserveShuffledSelection)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    const std::array fixtures{std::pair{"Cartesian", &cartesian_source}, std::pair{"Product", &product_source},
        std::pair{"Polyhedral", &polyhedral_source}, std::pair{"Static", &static_source},
        std::pair{"Mixed", &mixed_source}};
    for (const auto& [name, make] : fixtures)
    {
        SCOPED_TRACE(name);
        auto source = make();
        std::vector<const AMR::AdaptiveLeaf*> rows;
        for (const auto& leaf : source->owned_leaves())
            rows.push_back(&leaf);
        ASSERT_GT(rows.size(), 2u);

        const auto initial_rows = std::span(rows).first(rows.size() / 2);
        const auto before = snapshot(*source, initial_rows);
        const auto owners_before = compact_owners(*before, *source);
        std::rotate(rows.begin(), rows.begin() + rows.size() / 2, rows.end());
        std::reverse(rows.begin() + 1, rows.end());
        const auto after = snapshot(*source, rows);
        const auto owners_after = compact_owners(*after, *source);
        ASSERT_EQ(after->ordinary_catalog()->size(), rows.size());
        EXPECT_NE(before, after);
        EXPECT_TRUE(after->matches_layout(source->leaf_layout()));
        EXPECT_GT(std::count_if(owners_before.begin(), owners_before.end(),
                      [&](const auto& owner) { return owners_after.contains(owner.first); }),
            0);
        const auto metadata = std::accumulate(owners_after.begin(), owners_after.end(), std::size_t(0),
            [](auto bytes, const auto& owner) { return bytes + owner.second; });
        auto previous_metadata = rows.size() * previous_binding_bytes;
        for (const auto* row : rows)
            if (row->provider == AMR::AdaptiveProviderKind::Polyhedral && row->geometry && row->join_recipe)
                for (const auto& surface : row->surfaces)
                    previous_metadata +=
                        sizeof(AMR::GeometrySnapshotSurface) + surface.key.size() + surface.boundary.size();
        EXPECT_LT(metadata, previous_metadata);

        for (std::size_t slot = 0; slot < rows.size(); ++slot)
        {
            const auto& leaf = *rows[slot];
            ASSERT_EQ(after->find(leaf), slot);
            const auto expected = source->metrics(leaf);
            const auto actual = after->native_metrics().at(slot);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.volume), std::bit_cast<std::uint64_t>(expected.volume));
            for (unsigned axis = 0; axis < 3; ++axis)
                EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.centroid[axis]),
                    std::bit_cast<std::uint64_t>(expected.centroid[axis]));
            auto malformed = leaf;
            ++malformed.native_cell;
            EXPECT_FALSE(after->find(malformed));

            if (const auto old_slot = before->find(leaf))
            {
                const View old_view(before->ordinary_catalog(), *old_slot), new_view(after->ordinary_catalog(), slot);
                EXPECT_EQ(old_view.is_procedural(), new_view.is_procedural());
                if (old_view.is_procedural())
                {
                    const auto old_ref = Access::source_cell(old_view), new_ref = Access::source_cell(new_view);
                    EXPECT_EQ(old_ref.source, new_ref.source);
                    EXPECT_EQ(old_ref.cell_slot, new_ref.cell_slot);
                }
                else
                    EXPECT_EQ(Access::existing_geometry(old_view), Access::existing_geometry(new_view));
            }
        }
        const auto needed = after->retained_bytes();
        PLIC::GeometryAllocationLedger exact_ordinary, exact_bounded, short_ordinary, short_bounded;
        std::size_t exact = 0, short_used = 0;
        EXPECT_NO_THROW(after->retain(exact_ordinary, exact_bounded, exact, needed));
        EXPECT_EQ(exact, needed);
        EXPECT_THROW(after->retain(short_ordinary, short_bounded, short_used, needed - 1), PLIC::GeometryError);

        std::weak_ptr<const AMR::AdaptiveLeafLayout> weak_layout = source->leaf_layout();
        const View surviving(after->ordinary_catalog(), 0);
        const auto volume = surviving.volume().value;
        source.reset();
        EXPECT_TRUE(weak_layout.expired());
        EXPECT_EQ(surviving.volume().value, volume);
    }
}

TEST(AdaptiveGeometryStorage, CatalogReorderSharesExplicitShellPagesAndIndependentRoot)
{
    using Geometry = PLIC::ConvexCellGeometry<ordinary>;
    using Catalog = PLIC::CellGeometryCatalog<ordinary>;
    std::vector<PLIC::ConvexGeometryPtr<ordinary>> cells;
    for (unsigned i = 0; i < 80; ++i)
        cells.push_back(Geometry::box({0, 0, 0}, {double(i + 1), 1, 1}));
    auto first = Catalog::materialized(std::move(cells));
    auto permutation = indices(first->size());
    std::reverse(permutation.begin(), permutation.end());
    const std::vector<std::size_t> order(permutation.begin(), permutation.end());
    const auto reordered = Catalog::reordered(first, order);
    {
        const std::array roots{R::root(*first, 0), R::root(*reordered, 1)};
        std::vector<std::byte> buffer(16 * 1024 * 1024);
        R::ProbeScratch scratch(buffer);
        const auto census = R::inspect_retention(roots, {}, scratch);
        ASSERT_TRUE(census.complete()) << census.detail;
        const auto pages = std::count_if(census.owners.begin(), census.owners.end(), [](const auto& owner)
            { return std::string_view(owner.kind) == "cell_shell_page" && owner.root_groups == 0b11; });
        EXPECT_EQ(pages, 2);
    }
    {
        std::vector<Catalog::Entry> appended;
        for (std::size_t slot = 0; slot < first->size(); ++slot)
            appended.emplace_back(first->explicit_entry(slot));
        appended.emplace_back(Geometry::box({0, 0, 0}, {81, 1, 1}));
        const auto expanded = Catalog::make(Catalog::SourceTable{}, std::move(appended));
        const std::array roots{R::root(*first, 0), R::root(*expanded, 1)};
        std::vector<std::byte> buffer(16 * 1024 * 1024);
        R::ProbeScratch scratch(buffer);
        const auto census = R::inspect_retention(roots, {}, scratch);
        ASSERT_TRUE(census.complete()) << census.detail;
        const auto full_pages = std::count_if(census.owners.begin(), census.owners.end(), [](const auto& owner)
            { return std::string_view(owner.kind) == "cell_shell_page" && owner.root_groups == 0b11; });
        EXPECT_EQ(full_pages, 1);
        EXPECT_EQ(Access::existing_geometry(View(expanded, 79)), Access::existing_geometry(View(first, 79)));
    }
    const std::array<std::size_t, 3> subset{5, 5, 1};
    const auto selected = Catalog::selected(reordered, subset);
    EXPECT_EQ(selected->size(), subset.size());
    for (std::size_t slot = 0; slot < subset.size(); ++slot)
        EXPECT_EQ(View(selected, slot).volume().value, order[subset[slot]] + 1);
    const std::array<std::size_t, 1> invalid{reordered->size()};
    EXPECT_THROW((void) Catalog::selected(reordered, invalid), PLIC::GeometryError);
    std::weak_ptr<const Catalog> old = first;
    first.reset();
    EXPECT_TRUE(old.expired());
    for (std::size_t i = 0; i < order.size(); ++i)
        EXPECT_EQ(View(reordered, i).volume().value, order[i] + 1);
}

TEST(AdaptiveGeometryStorage, CartesianSourcePreservesIrregularIntervalsAndOriginalSupportOverrides)
{
    using Geometry = PLIC::ConvexCellGeometry<ordinary>;
    PLIC::CartesianSourceDescription input;
    for (unsigned x = 0; x <= 80; ++x)
        input.axes[0].push_back(double(x));
    input.axes[1] = input.axes[2] = {0, 1};
    for (unsigned x = 0; x < 80; ++x)
    {
        PLIC::CartesianSourceCellSlot cell;
        cell.lower = {x, 0, 0};
        cell.upper = {x + (x == 17 ? 2 : 1), 1, 1};
        input.cells.push_back(cell);
    }
    const auto shell = View::from_existing(Geometry::box({17, 0, 0}, {19, 1, 1}));
    const auto original = shell.source_support(1);
    input.original_supports.push_back(original);
    input.cells[17].support_slots[1] = 0;
    const auto source = Access::make(std::move(input), {}, PLIC::details::NativeCellGeometryMoments::PrimedState);
    const auto catalog = PLIC::CellGeometryCatalog<ordinary>::cartesian(source);
    EXPECT_EQ(View(catalog, 17).source_support(1), original);
    for (std::size_t row = 0; row < catalog->size(); ++row)
    {
        const View view(catalog, row);
        const auto width = row == 17 ? 2 : 1;
        EXPECT_EQ(view.volume().value, width);
        EXPECT_EQ(view.centroid()[0], double(row) + .5 * width);
        const auto bounds = view.closed_contact_bounds();
        EXPECT_EQ(bounds[0][0], row);
        EXPECT_EQ(bounds[0][1], row + width);
    }
}

TEST(AdaptiveGeometryStorage, MixedCatalogSelectsAcrossProceduralAndShellPageBoundaries)
{
    using Catalog = PLIC::CellGeometryCatalog<ordinary>;
    using Geometry = PLIC::ConvexCellGeometry<ordinary>;
    PLIC::CartesianSourceDescription input;
    for (unsigned x = 0; x <= 64; ++x)
        input.axes[0].push_back(double(x));
    input.axes[1] = input.axes[2] = {0, 1};
    for (unsigned x = 0; x < 64; ++x)
    {
        PLIC::CartesianSourceCellSlot cell;
        cell.lower = {x, 0, 0};
        cell.upper = {x + 1, 1, 1};
        input.cells.push_back(cell);
    }
    const auto source = PLIC::CartesianCellSource::make(std::move(input));
    std::vector<Catalog::Entry> entries;
    for (std::size_t i = 0; i < 131; ++i)
        if (i % 3 == 0)
            entries.emplace_back(PLIC::NativeCellGeometrySlot{0, i / 3});
        else
            entries.emplace_back(Geometry::box({0, 0, 0}, {double(i + 100), 1, 1}));
    auto catalog = Catalog::make(source, std::move(entries));
    std::vector<std::size_t> order(catalog->size());
    std::iota(order.begin(), order.end(), 0);
    std::rotate(order.begin(), order.begin() + 64, order.end());
    const auto reordered = Catalog::reordered(catalog, order);
    std::vector<std::size_t> selected_order;
    for (std::size_t repeat = 0; repeat < 13; ++repeat)
        selected_order.insert(selected_order.end(), {62, 63, 64, 65, 127, 128});
    const auto selected = Catalog::selected(catalog, selected_order);
    const auto check = [&](const auto& result, std::span<const std::size_t> source_slots)
    {
        ASSERT_EQ(result->size(), source_slots.size());
        EXPECT_EQ(result->source_count(), 1u);
        for (std::size_t i = 0; i < source_slots.size(); ++i)
        {
            SCOPED_TRACE(i);
            const auto slot = source_slots[i];
            const View before(catalog, slot), after(result, i);
            ASSERT_EQ(after.is_procedural(), before.is_procedural());
            if (before.is_procedural())
            {
                const auto ref = Access::source_cell(after);
                EXPECT_EQ(ref.source, source);
                EXPECT_EQ(ref.cell_slot, slot / 3);
            }
            else
                EXPECT_EQ(Access::existing_geometry(after), Access::existing_geometry(before));
        }
    };
    check(reordered, order);
    check(selected, selected_order);
    std::weak_ptr<const Catalog> old = catalog;
    catalog.reset();
    EXPECT_TRUE(old.expired());
    EXPECT_EQ(View(reordered, 0).volume().value, 164);
    EXPECT_TRUE(View(selected, 1).is_procedural());
}

TEST(AdaptiveGeometryStorage, CartesianSuccessorSharesUnchangedSourceWithWeakLayoutIdentity)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    auto source = [&]
    {
        const AMR::CartesianAMR mesh(grid);
        return hierarchy(mesh.detach_owned(indices(mesh.geometry()->num_cells()), {'s', 'u', 'c', 'c'}));
    }();
    std::vector<const AMR::AdaptiveLeaf*> rows;
    for (const auto& leaf : source->owned_leaves())
        rows.push_back(&leaf);
    ASSERT_EQ(rows.size(), 16u);
    const auto accepted = snapshot(*source, rows);
    const auto owners_before = compact_owners(*accepted, *source);
    const AMR::AdaptiveMark mark{rows.front()->key, AMR::AdaptiveSplit::Isotropic};
    const auto candidate = source->apply(source->plan_adaptation(std::array{mark}));
    EXPECT_FALSE(accepted->matches_layout(candidate->leaf_layout()));

    std::vector<const AMR::AdaptiveLeaf*> candidate_rows;
    for (const auto& leaf : candidate->owned_leaves())
        candidate_rows.push_back(&leaf);
    const auto successor = snapshot(*candidate, candidate_rows);
    ASSERT_EQ(successor->ordinary_catalog()->size(), candidate_rows.size());
    EXPECT_TRUE(successor->matches_layout(candidate->leaf_layout()));
    EXPECT_FALSE(successor->find(*rows.front()));
    std::size_t reused = 0;
    for (const auto* leaf : candidate_rows)
        if (const auto old_slot = accepted->find(*leaf))
        {
            const auto new_slot = successor->find(*leaf);
            ASSERT_TRUE(new_slot);
            const View before(accepted->ordinary_catalog(), *old_slot), after(successor->ordinary_catalog(), *new_slot);
            ASSERT_TRUE(before.is_procedural());
            ASSERT_TRUE(after.is_procedural());
            const auto old = Access::source_cell(before), current = Access::source_cell(after);
            EXPECT_EQ(current.source, old.source);
            EXPECT_EQ(current.cell_slot, old.cell_slot);
            ++reused;
        }
    EXPECT_EQ(reused, 8u);
    const auto owners_after = compact_owners(*successor, *candidate);
    EXPECT_TRUE(std::any_of(owners_before.begin(), owners_before.end(),
        [&](const auto& owner) { return owners_after.contains(owner.first); }));
    std::weak_ptr<const AMR::AdaptiveLeafLayout> weak_layout = source->leaf_layout();
    source.reset();
    EXPECT_TRUE(weak_layout.expired());
}

TEST(AdaptiveGeometryStorage, StreamingCartesianExpansionKeepsBoundedSourceOwnersAcrossPages)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    AMR::AMRGrid grid;
    grid.block_cells = {8, 4, 4};
    const auto batch_source = cartesian_grid_source(grid);
    std::vector<const AMR::AdaptiveLeaf*> batch_rows;
    for (const auto& leaf : batch_source->owned_leaves())
        batch_rows.push_back(&leaf);
    const auto batch = snapshot(*batch_source, batch_rows);
    const auto batch_bytes = batch->retained_bytes();
    const auto batch_metadata = compact_owners(*batch, *batch_source);
    RecordProperty("BatchRetainedBytes", std::to_string(batch_bytes));
    for (const std::size_t width : {1, 2, 4})
        for (const auto direction : {"Forward", "Reverse", "Alternate"})
        {
            const auto label = std::string(direction) + std::to_string(width);
            SCOPED_TRACE(label);
            const auto source = cartesian_grid_source(grid);
            std::vector<const AMR::AdaptiveLeaf*> rows;
            for (const auto& leaf : source->owned_leaves())
                rows.push_back(&leaf);
            ASSERT_EQ(rows.size(), 128u);
            if (std::string_view(direction) == "Reverse")
                std::reverse(rows.begin(), rows.end());
            if (std::string_view(direction) == "Alternate")
                std::sort(rows.begin(), rows.end(),
                    [](const auto* first, const auto* second)
                    {
                        const auto weave = [](auto ordinal) { return 2 * (ordinal % 64) + ordinal / 64; };
                        return weave(first->native_cell) < weave(second->native_cell);
                    });
            // A native publication may supply equivalent copies of local rows.
            std::vector<AMR::AdaptiveLeaf> copied;
            if (width == 4 && std::string_view(direction) == "Alternate")
            {
                copied.reserve(rows.size());
                for (auto& row : rows)
                {
                    copied.push_back(*row);
                    row = &copied.back();
                }
            }
            const auto selection = std::span(rows);
            const auto first = snapshot(*source, selection.first(width));
            const auto first_ref = Access::source_cell(View(first->ordinary_catalog(), 0));
            auto active = first;
            for (std::size_t offset = width; offset < rows.size(); offset += width)
                active = snapshot(*source, selection.subspan(offset, width));
            ASSERT_EQ(active->ordinary_catalog()->size(), rows.size());
            const auto first_slot = active->find(*rows.front());
            ASSERT_TRUE(first_slot);
            const auto surviving = Access::source_cell(View(active->ordinary_catalog(), *first_slot));
            EXPECT_EQ(surviving.source, first_ref.source);
            EXPECT_EQ(surviving.cell_slot, first_ref.cell_slot);

            std::vector<R::BorrowedRoot> roots{R::root(*active->ordinary_catalog())};
            active->visit_prepared_cartesian_sources([&](const auto& value) { roots.push_back(R::root(value)); });
            std::vector<std::byte> scratch_bytes(8 * 1024 * 1024);
            R::ProbeScratch scratch(scratch_bytes);
            const auto census = R::inspect_retention(roots, {}, scratch);
            ASSERT_TRUE(census.complete()) << census.detail;
            const auto source_owners = std::count_if(census.owners.begin(), census.owners.end(),
                [](const auto& owner) { return std::string_view(owner.kind) == "cartesian_source"; });
            EXPECT_EQ(source_owners, active->ordinary_catalog()->source_count());
            EXPECT_LE(source_owners, 5);
            const auto bytes = active->retained_bytes();
            EXPECT_LE(bytes, batch_bytes + batch_bytes / 2);
            // Partial pages can change identity during a merge. The resulting
            // physical metadata owner count is bounded independently of request count.
            const auto metadata = compact_owners(*active, *source);
            EXPECT_LE(metadata.size(), batch_metadata.size() + 4);
            RecordProperty(label + "SourceOwners", std::to_string(source_owners));
            RecordProperty(label + "MetadataOwners", std::to_string(metadata.size()));
            RecordProperty(label + "RetainedBytes", std::to_string(bytes));
        }
}

TEST(AdaptiveGeometryStorage, TightStreamingCapAdmitsOnlyTheRequiredCandidate)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    std::size_t exact = 0;
    for (const int attempt : {0, 1, 2})
    {
        const auto source = cartesian_source();
        auto row = source->owned_leaves().begin();
        const auto* first_row = &*row;
        ++row;
        const auto* next_row = &*row;
        const auto initial = snapshot(*source, std::array{first_row});
        std::size_t used = 0;
        const auto acquire = [&](std::size_t cap)
        {
            return source->geometry_snapshot(
                std::array{next_row}, ordinary, AMR::GeometrySnapshotPurpose::State, used, cap);
        };
        if (attempt == 2)
        {
            EXPECT_THROW((void) acquire(exact - 1), std::exception);
            EXPECT_EQ(snapshot(*source, std::array{first_row}), initial);
            continue;
        }
        const auto candidate = acquire(attempt == 0 ? initial->retained_bytes() + 12 * 1024 : exact);
        if (attempt == 0)
            exact = used;
        EXPECT_EQ(used, exact);
        std::size_t prepared = 0;
        candidate->visit_prepared_cartesian_sources([&](const auto&) { ++prepared; });
        EXPECT_EQ(prepared, 0u);
        EXPECT_EQ(candidate->ordinary_catalog()->size(), 2u);
    }
}

TEST(AdaptiveGeometryStorage, EqualRecipeWithDifferentOwnerStaysDemandOnly)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    std::size_t demand_cap = 0;
    for (const bool retry : {false, true})
    {
        auto source = cartesian_source();
        auto row = source->owned_leaves().begin();
        auto first = *row++;
        auto second = *row;
        ASSERT_EQ(first.native_recipe, second.native_recipe);
        const std::weak_ptr<const SimpleFluid::Meshes::NativeFragmentDescriptor> original = first.native_recipe;
        auto copy = std::make_shared<SimpleFluid::Meshes::NativeFragmentDescriptor>(*first.native_recipe);
        for (auto& part : copy->parts)
            part = std::make_shared<const AMR::AdaptiveBytes>(*part);
        first.native_recipe = copy;
        second.native_recipe = copy;
        const auto initial = snapshot(*source, std::array<const AMR::AdaptiveLeaf*, 1>{&first});
        std::size_t used = 0;
        const auto expanded = source->geometry_snapshot(std::array<const AMR::AdaptiveLeaf*, 1>{&second},
            ordinary, AMR::GeometrySnapshotPurpose::State, used, retry ? demand_cap : limit);
        if (!retry)
            demand_cap = used;
        EXPECT_EQ(used, demand_cap);
        std::size_t prepared = 0;
        expanded->visit_prepared_cartesian_sources([&](const auto&) { ++prepared; });
        EXPECT_EQ(prepared, 0u);
        EXPECT_EQ(expanded->ordinary_catalog()->size(), 2u);
        EXPECT_TRUE(expanded->find(first));
        EXPECT_TRUE(expanded->find(second));
        source.reset();
        EXPECT_TRUE(original.expired());
        EXPECT_TRUE(initial->find(first));
        EXPECT_TRUE(expanded->find(second));
    }
}

TEST(AdaptiveGeometryStorage, PreparedCartesianPageExcludesStateFallbacksAndChangedControls)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    struct Environment
    {
        std::fenv_t saved{};
        Environment() { (void) std::fegetenv(&saved); }
        ~Environment() { (void) std::fesetenv(&saved); }
    } environment;
    AMR::AMRGrid grid;
    grid.origin[0] = -0x1p-100;
    grid.extent[0] = 4;
    const auto source = cartesian_grid_source(grid);
    std::array<const AMR::AdaptiveLeaf*, 4> row{};
    for (const auto& leaf : source->owned_leaves())
    {
        const auto& key = std::get<AMR::CartesianCellAncestry>(leaf.key.ancestry);
        if (key.index[1] == 0 && key.index[2] == 0)
            row[key.index[0]] = &leaf;
    }
    ASSERT_TRUE(std::all_of(row.begin(), row.end(), [](const auto* leaf) { return leaf != nullptr; }));
    const auto incompatible = source->native_cartesian_geometry_bounds(*row[0]);
    ASSERT_TRUE(incompatible);
    EXPECT_FALSE(Access::state_compatible(incompatible->lower, incompatible->upper));
    const auto first = snapshot(*source, std::array{row[1]});
    ASSERT_TRUE(View(first->ordinary_catalog(), 0).is_procedural());
    const auto warmed = snapshot(*source, std::array{row[2]});
    std::size_t prepared = 0;
    warmed->visit_prepared_cartesian_sources([&](const auto& page) { prepared += page.size(); });
    EXPECT_GT(prepared, 2u);
    EXPECT_LT(prepared, 62u);
    const auto fallback = snapshot(*source, std::array{row[0]});
    const auto bad = fallback->find(*row[0]);
    ASSERT_TRUE(bad);
    EXPECT_FALSE(View(fallback->ordinary_catalog(), *bad).is_procedural());

    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    const auto changed = snapshot(*source, std::array{row[3]});
    EXPECT_EQ(changed->construction_controls().rounding, FE_UPWARD);
    EXPECT_EQ(changed->ordinary_catalog()->size(), 1u);
    EXPECT_FALSE(View(changed->ordinary_catalog(), 0).is_procedural());
    changed->visit_prepared_cartesian_sources(
        [&](const auto&) { ADD_FAILURE() << "Inherited a foreign-control page."; });
    EXPECT_TRUE(View(first->ordinary_catalog(), 0).is_procedural());
}

TEST(AdaptiveGeometryStorage, RawSelectionFitsBelowThePreparedStatePageFactCost)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    struct Sample
    {
        std::shared_ptr<const Hierarchy> source;
        std::array<const AMR::AdaptiveLeaf*, 3> rows{};
        std::shared_ptr<const Snapshot> state;
    };
    const auto make = [](bool prepare)
    {
        Sample sample;
        sample.source = cartesian_source();
        auto row = sample.source->owned_leaves().begin();
        for (auto& selected : sample.rows)
        {
            selected = &*row;
            ++row;
            std::size_t used = 0;
            const auto cap = !sample.state || prepare ? limit : sample.state->retained_bytes() + 12 * 1024;
            sample.state = sample.source->geometry_snapshot(
                std::array{selected}, ordinary, AMR::GeometrySnapshotPurpose::State, used, cap);
        }
        return sample;
    };
    const auto prepared = make(true);
    const auto plain = make(false);
    const auto state_slot = prepared.state->find(*prepared.rows.back());
    ASSERT_TRUE(state_slot);
    const auto page = Access::source_cell(View(prepared.state->ordinary_catalog(), *state_slot)).source;
    ASSERT_GT(page->size(), prepared.rows.size());
    plain.state->visit_prepared_cartesian_sources(
        [&](const auto&) { ADD_FAILURE() << "The demand-only State baseline prepared a page."; });
    const auto exclusive = Access::exclusive_bytes(*page);
    const auto dense_payload = std::accumulate(exclusive.begin(), exclusive.end(), std::size_t(0));
    ASSERT_GT(dense_payload, 0u);
    // Old State ownership participates in both admission scopes. Each raw
    // request has the same extra cap, one byte below a dense clone's payload.
    for (const auto* sample : {&plain, &prepared})
    {
        PLIC::GeometryAllocationLedger ordinary_owners, bounded_owners;
        std::size_t used = 0;
        sample->state->retain(ordinary_owners, bounded_owners, used, limit);
        const auto cap = used + dense_payload - 1;
        const auto raw = sample->source->geometry_snapshot(std::array{sample->rows.back()}, ordinary,
            AMR::GeometrySnapshotPurpose::OriginalSources, used, cap, &ordinary_owners, &bounded_owners);
        EXPECT_LE(used, cap);
        EXPECT_TRUE(raw->native_metrics().empty());
        ASSERT_EQ(raw->ordinary_catalog()->size(), sample->rows.size());
        const auto selected = raw->find(*sample->rows.back());
        ASSERT_TRUE(selected);
        const View view(raw->ordinary_catalog(), *selected);
        ASSERT_TRUE(view.is_procedural());
        EXPECT_LE(Access::source_cell(view).source->size(), sample->rows.size());
        const auto factory = sample->source->geometry(*sample->rows.back());
        const auto reference = View::from_existing(factory);
        EXPECT_EQ(view.volume().value, factory->volume().value);
        for (std::size_t face = 0; face < 6; ++face)
            EXPECT_EQ(view.source_support(face), reference.source_support(face));

        struct Environment
        {
            std::fenv_t saved{};
            Environment() { (void) std::fegetenv(&saved); }
            ~Environment() { (void) std::fesetenv(&saved); }
        } environment;
        ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
        const PLIC::GeometryPoint origin{-.1, .3, -.7};
        const auto before = PLIC::geometry_statistics().moment_evaluations;
        const auto actual = view.moments(origin);
        EXPECT_EQ(PLIC::geometry_statistics().moment_evaluations, before + 1);
        const auto expected = factory->moments(origin);
        EXPECT_EQ(actual.volume.value, expected.volume.value);
        for (unsigned axis = 0; axis < 3; ++axis)
            EXPECT_EQ(actual.first_moment[axis].value, expected.first_moment[axis].value);
    }
}
