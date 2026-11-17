/** Deterministic storage/work gates and supporting timings for the 64-cell
 * prepared-page policy. The small-cap guards remain in testAdaptiveGeometryStorage.
 */
#include "geometry/PLIC/details/CellCatalogDemandAccess.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace PLIC = SimpleFluid::PLIC;
namespace G = PLIC::details::geometry_demand;
using SimpleFluid::ArithmeticMode;
using Hierarchy = AMR::DistributedAdaptiveHierarchy;
using Snapshot = AMR::AdaptiveGeometrySnapshot;
using View = PLIC::CellGeometryView<ArithmeticMode::Ordinary>;
using Access = AMR::adaptive_detail::GeometrySnapshotAccess;
using SourceAccess = PLIC::details::NativeCellGeometrySourceAccess;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t ample = 256 * 1024 * 1024;
// Published persistent page contracts, independent of the measured counters.
constexpr std::size_t leaf_capacity = 64, branch_capacity = 32, control_bytes = 128;
constexpr std::array directions{"Forward", "Reverse", "Alternating"};
constexpr std::array<std::size_t, 3> widths{1, 2, 4};
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::size_t scale_cells()
{
    const auto* text = std::getenv("SIMPLEFLUID_DEMAND_SCALE_CELLS");
    return text ? std::size_t(std::strtoull(text, nullptr, 10)) : 512;
}
std::shared_ptr<const Hierarchy> cartesian_scale_source(std::size_t cells)
{
    AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {cells / 64u, 8, 8};
    const AMR::CartesianAMR mesh(grid);
    std::vector<std::uint64_t> indices(mesh.geometry()->num_cells());
    std::iota(indices.begin(), indices.end(), 0);
    auto detached = mesh.detach_owned(indices, {'s', 'c', 'a', 'l'});
    return std::make_shared<const Hierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), Tpetra::getDefaultComm());
}
std::vector<const AMR::AdaptiveLeaf*> rows_of(const Hierarchy& source)
{
    std::vector<const AMR::AdaptiveLeaf*> rows;
    for (const auto& leaf : source.owned_leaves())
        rows.push_back(&leaf);
    return rows;
}
std::shared_ptr<const Snapshot> snapshot(const Hierarchy& source, std::span<const AMR::AdaptiveLeaf* const> rows,
    ArithmeticMode mode = ordinary, AMR::GeometrySnapshotPurpose purpose = AMR::GeometrySnapshotPurpose::State)
{
    std::size_t used = 0;
    return source.geometry_snapshot(rows, mode, purpose, used, ample);
}
std::size_t physical_owners(const Snapshot& snapshot)
{
    PLIC::GeometryAllocationLedger ordinary_owners, bounded_owners;
    std::size_t used = 0;
    snapshot.retain(ordinary_owners, bounded_owners, used, std::numeric_limits<std::size_t>::max());
    return ordinary_owners.owner_count() + bounded_owners.owner_count();
}
void record(const std::string& key, std::size_t value)
{
    testing::Test::RecordProperty(key, std::to_string(value));
}
void record_ms(const std::string& key, double value)
{
    testing::Test::RecordProperty(key, std::to_string(value));
}
std::vector<const AMR::AdaptiveLeaf*> ordered_rows(const Hierarchy& source, std::string_view direction)
{
    const auto canonical = rows_of(source);
    auto rows = canonical;
    if (direction == "Reverse")
        std::reverse(rows.begin(), rows.end());
    if (direction == "Alternating")
    {
        // Alternate between the lower and upper halves of the complete demand;
        // reversing just one four-row request does not exercise this order.
        rows.clear();
        const auto half = canonical.size() / 2;
        for (std::size_t slot = 0; slot < half; ++slot)
        {
            rows.push_back(canonical[slot]);
            rows.push_back(canonical[half + slot]);
        }
    }
    return rows;
}
struct StreamResult
{
    std::size_t retained = 0, owners = 0, prepared = 0, sources = 0;
    double build_ms = 0;
};
struct ScaleReport
{
    std::size_t cells = 0, batch_bytes = 0;
    std::array<StreamResult, 9> streams;
};
using Payload = std::map<const void*, std::size_t>;
Payload source_payload(const PLIC::CartesianCellSource& source)
{
    Payload result;
    SourceAccess::visit_payload(source, [&](const void* owner, std::size_t bytes) { result.emplace(owner, bytes); });
    return result;
}
struct NativeSelection
{
    struct Row
    {
        const AMR::AdaptiveLeaf* leaf;
        PLIC::details::NativeCartesianCellReference physical;
    };
    std::vector<Row> rows;
    std::map<const void*, Payload> payloads;
};
NativeSelection native_selection(const Snapshot& snapshot, std::span<const AMR::AdaptiveLeaf* const> rows)
{
    NativeSelection result;
    for (const auto* row : rows)
    {
        const auto slot = snapshot.find(*row);
        EXPECT_TRUE(slot);
        if (!slot)
            continue;
        const View view(snapshot.ordinary_catalog(), *slot);
        EXPECT_TRUE(view.is_procedural());
        if (!view.is_procedural())
            continue;
        auto physical = SourceAccess::source_cell(view);
        if (!result.payloads.contains(physical.source.get()))
            result.payloads.emplace(physical.source.get(), source_payload(*physical.source));
        result.rows.push_back({row, std::move(physical)});
    }
    return result;
}
void expect_native_selection_preserved(const NativeSelection& before, const Snapshot& after)
{
    for (const auto& row : before.rows)
    {
        const auto slot = after.find(*row.leaf);
        ASSERT_TRUE(slot);
        const View view(after.ordinary_catalog(), *slot);
        ASSERT_TRUE(view.is_procedural());
        const auto physical = SourceAccess::source_cell(view);
        EXPECT_EQ(physical.source.get(), row.physical.source.get());
        EXPECT_FALSE(physical.source.owner_before(row.physical.source));
        EXPECT_FALSE(row.physical.source.owner_before(physical.source));
        EXPECT_EQ(physical.cell_slot, row.physical.cell_slot);
    }
    // Capture before acquisition, rather than comparing two aliases after it.
    std::set<const void*> checked;
    for (const auto& row : before.rows)
        if (checked.insert(row.physical.source.get()).second)
            EXPECT_EQ(source_payload(*row.physical.source), before.payloads.at(row.physical.source.get()));
}
// Traverse the actual borrowed graph outside the measured interval. Its depth
// and byte envelopes are independent of the construction-work counters.
struct CatalogGraph
{
    std::map<const void*, std::size_t> depth, bytes;
    std::size_t height = 0, largest = 0;
};
CatalogGraph catalog_graph(const Snapshot& snapshot)
{
    CatalogGraph graph;
    PLIC::details::CellCatalogDemandAccess::visit_borrowed_dependencies(*snapshot.ordinary_catalog(),
        {&graph,
            [](void* context, const void* owner, const void* parent, std::size_t bytes)
            {
                auto& graph = *static_cast<CatalogGraph*>(context);
                const auto depth = parent ? graph.depth.at(parent) + 1 : 1;
                graph.height = std::max(graph.height, depth);
                graph.largest = std::max(graph.largest, bytes);
                graph.depth.emplace(owner, depth);
                return graph.bytes.emplace(owner, bytes).second;
            }},
        {nullptr, [](void*, const void*, G::SourceKind, const void*) {}});
    return graph;
}
} // namespace

TEST(AdaptiveGeometryDemandScaling, CompletedStreamsKeepPerCellStorageAndOwnersLinear)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    const auto large = std::max<std::size_t>(scale_cells(), 256u);
    ASSERT_EQ(large % 128, 0u) << "Matched N/2N fixtures require complete 64-cell blocks.";
    const auto measure = [&](std::size_t cells, const std::string& prefix)
    {
        ScaleReport report;
        report.cells = cells;
        const auto batch_source = cartesian_scale_source(cells);
        const auto batch_rows = rows_of(*batch_source);
        EXPECT_EQ(batch_rows.size(), cells);
        const auto begin = std::chrono::steady_clock::now();
        const auto batch = snapshot(*batch_source, batch_rows);
        record_ms(prefix + "BatchMs",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
        report.batch_bytes = batch->retained_bytes();
        record(prefix + "Cells", cells);
        record(prefix + "BatchBytes", report.batch_bytes);
        record(prefix + "BatchOwners", physical_owners(*batch));
        for (std::size_t direction = 0; direction < directions.size(); ++direction)
            for (std::size_t width_slot = 0; width_slot < widths.size(); ++width_slot)
            {
                const auto width = widths[width_slot];
                const auto label = prefix + directions[direction] + "Batch" + std::to_string(width);
                SCOPED_TRACE(label);
                const auto source = cartesian_scale_source(cells);
                const auto rows = ordered_rows(*source, directions[direction]);
                const auto started = std::chrono::steady_clock::now();
                auto active = snapshot(*source, std::span(rows).first(width));
                for (std::size_t offset = width; offset < rows.size(); offset += width)
                    active = snapshot(*source, std::span(rows).subspan(offset, width));
                auto& result = report.streams[direction * widths.size() + width_slot];
                result.build_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                result.retained = active->retained_bytes();
                result.owners = physical_owners(*active);
                active->visit_prepared_cartesian_sources([&](const auto& page) { result.prepared += page.size(); });
                result.sources = active->ordinary_catalog()->source_count();
                EXPECT_EQ(active->ordinary_catalog()->size(), cells);
                for (const auto* row : rows)
                    EXPECT_TRUE(active->find(*row));
                EXPECT_LE(result.prepared, cells);
                EXPECT_LE(result.owners, cells);
                record(label + "Bytes", result.retained);
                record(label + "Owners", result.owners);
                record(label + "Prepared", result.prepared);
                record(label + "Sources", result.sources);
                record_ms(label + "Ms", result.build_ms);
                record_ms(label + "BatchRatio", double(result.retained) / double(report.batch_bytes));
            }
        return report;
    };
    const auto small = measure(large / 2, "Half");
    const auto big = measure(large, "Full");
    // Preserve the existing scale gates for every newly covered order/batch.
    // The 128-cell 1.5 stream/batch and owner allowances remain unchanged in
    // AdaptiveGeometryStorage, together with required-candidate and Raw caps.
    const auto per_cell = [](std::size_t value, std::size_t cells) { return double(value) / double(cells); };
    for (std::size_t slot = 0; slot < small.streams.size(); ++slot)
    {
        SCOPED_TRACE(slot);
        EXPECT_LE(per_cell(big.streams[slot].retained, big.cells),
            per_cell(small.streams[slot].retained, small.cells) * 1.15 + 8);
        EXPECT_LE(per_cell(big.streams[slot].owners, big.cells),
            per_cell(small.streams[slot].owners, small.cells) * 1.15 + 0.05);
    }
}

TEST(AdaptiveGeometryDemandScaling, DoubledUnrelatedWarmSelectionKeepsCopyWorkBounded)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    const auto cells = std::max<std::size_t>(scale_cells(), 384u);
    ASSERT_EQ(cells % 64, 0u);
    const auto measure = [&](std::size_t warm_rows, const std::string& label)
    {
        const auto source = cartesian_scale_source(cells);
        const auto rows = rows_of(*source);
        constexpr std::size_t changed = 4;
        EXPECT_LE(warm_rows + changed, rows.size());
        const auto warm = snapshot(*source, std::span(rows).first(warm_rows));
        const auto graph_before = catalog_graph(*warm);
        const auto physical_before = native_selection(*warm, std::span(rows).first(warm_rows));
        EXPECT_EQ(physical_before.rows.size(), warm_rows);
        const auto before = Access::cache_admission_statistics(*source);
        const auto begin = std::chrono::steady_clock::now();
        const auto active = snapshot(*source, std::span(rows).subspan(warm_rows, changed));
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        const auto after = Access::cache_admission_statistics(*source);
        const auto graph_after = catalog_graph(*active);
        const auto selected = active->ordinary_catalog()->size();
        EXPECT_EQ(selected, warm_rows + changed);
        const auto retained = active->retained_bytes();
        EXPECT_LE(retained / selected, std::size_t{700});
        EXPECT_LE(physical_owners(*active), cells);

        // This disjoint append is permitted to change zero old physical slots.
        // Rebuilding unrelated sources must fail, never enlarge its own bound.
        expect_native_selection_preserved(physical_before, *active);
        constexpr std::size_t permitted_old_rebound = 0;
        const auto height = graph_before.height + 1; // at most one new level from K=4.
        EXPECT_LE(graph_after.height, height);
        const auto edits = 4 * changed + 3;
        const auto selector_pages = 2 + changed;
        const auto page_bound = 6 * edits * (height + 1) + selector_pages;
        const auto visit_bound = 16 * edits * (height + 1);
        const auto row_width = std::max(sizeof(G::OwnerRecord), sizeof(G::OrderedSourceSlot));
        const auto node_bound = graph_before.largest + control_bytes + leaf_capacity * row_width +
                                branch_capacity * sizeof(std::shared_ptr<const void>);
        const auto scratch_bound = (changed + leaf_capacity) *
                                   (sizeof(G::OrderedSourceSlot) + sizeof(G::SourceSlot) + 2 * sizeof(G::OrderStamp));
        const auto byte_bound = (page_bound + 2) * node_bound + scratch_bound;
        const auto copied = after.catalog_copied_metadata_bytes - before.catalog_copied_metadata_bytes;
        const auto copied_pages = after.catalog_copied_pages - before.catalog_copied_pages;
        const auto visited_pages = after.catalog_visited_pages - before.catalog_visited_pages;
        const auto copied_rows = after.catalog_copied_source_rows - before.catalog_copied_source_rows;
        const auto visited_rows = after.catalog_visited_source_rows - before.catalog_visited_source_rows;
        EXPECT_GT(copied_pages, 0u);
        EXPECT_LE(copied, byte_bound);
        EXPECT_LE(copied_pages, page_bound);
        EXPECT_LE(visited_pages, visit_bound);
        EXPECT_LE(copied_rows, selector_pages * leaf_capacity);
        // Binary selector lookups cost at most log2(L64)+one equality check;
        // allow two validations plus one complete touched selector per edit.
        EXPECT_LE(visited_rows, (changed + 1) * (leaf_capacity + 2 * 8));
        record(label + "WarmRows", warm_rows);
        record(label + "ChangedRows", changed);
        record(label + "PermittedOldReboundRows", permitted_old_rebound);
        record(label + "GraphDepth", height);
        record(label + "CopiedBytes", copied);
        record(label + "CopiedBytesBound", byte_bound);
        record(label + "CopiedPages", copied_pages);
        record(label + "CopiedPagesBound", page_bound);
        record(label + "VisitedPages", visited_pages);
        record(label + "VisitedPagesBound", visit_bound);
        record(label + "CopiedSourceRows", copied_rows);
        record(label + "VisitedSourceRows", visited_rows);
        record(label + "Retained", retained);
        record_ms(label + "Ms", elapsed);
        struct Result
        {
            std::size_t retained, selected, copied, copied_pages, visited_pages;
        };
        return Result{retained, selected, copied, copied_pages, visited_pages};
    };
    const auto single = measure(cells / 3, "SingleWarm");
    const auto doubled = measure(2 * (cells / 3), "DoubledWarm");
    // Preserve every previous guard alongside the independent structural bounds.
    EXPECT_LE(doubled.retained / doubled.selected, single.retained / single.selected + 8);
    EXPECT_LE(doubled.copied, single.copied + single.copied / 2 + 4096);
    EXPECT_LE(doubled.copied_pages, single.copied_pages + 8);
    EXPECT_LE(doubled.visited_pages, single.visited_pages + 64);
}

TEST(AdaptiveGeometryDemandScaling, FixedBlockZeroDemandPreservesDistantCompleteWarmChunksForStateAndRaw)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    for (const auto purpose : {AMR::GeometrySnapshotPurpose::State, AMR::GeometrySnapshotPurpose::OriginalSources})
        for (const std::size_t warm_rows : {128, 256})
        {
            const auto label = std::string(purpose == AMR::GeometrySnapshotPurpose::State ? "State" : "Raw") +
                               "FixedBlockZeroWarm" + std::to_string(warm_rows);
            SCOPED_TRACE(label);
            const auto source = cartesian_scale_source(512);
            auto rows = rows_of(*source);
            std::sort(
                rows.begin(), rows.end(), [](const auto* a, const auto* b) { return a->native_cell < b->native_cell; });
            ASSERT_EQ(rows.size(), 512u);
            ASSERT_EQ(rows[0]->native_cell, 0u);
            ASSERT_EQ(rows[3]->native_cell, 3u);
            const auto distant = std::span(rows).subspan(128, warm_rows);
            const auto warm = snapshot(*source, distant, ordinary, purpose);
            const auto physical = native_selection(*warm, distant);
            ASSERT_EQ(physical.rows.size(), warm_rows);
            const auto before = Access::cache_admission_statistics(*source);
            const auto begin = std::chrono::steady_clock::now();
            const auto active = snapshot(*source, std::span(rows).first(4), ordinary, purpose);
            const auto elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            const auto after = Access::cache_admission_statistics(*source);
            ASSERT_EQ(active->ordinary_catalog()->size(), warm_rows + 4);
            expect_native_selection_preserved(physical, *active);
            if (purpose == AMR::GeometrySnapshotPurpose::OriginalSources)
            {
                EXPECT_TRUE(active->native_metrics().empty());
                EXPECT_FALSE(active->finite_ordinary_catalog());
                EXPECT_FALSE(active->bounded_catalog());
                for (const auto* row : std::span(rows).first(4))
                {
                    const auto slot = active->find(*row);
                    ASSERT_TRUE(slot);
                    const View view(active->ordinary_catalog(), *slot);
                    EXPECT_TRUE(view.is_procedural());
                    EXPECT_FALSE(active->evaluation_policy(*slot).original_integrals);
                    EXPECT_EQ(view.volume().value, 1.0 / 512.0);
                }
            }
            record(label + "CopiedSourceRows", after.catalog_copied_source_rows - before.catalog_copied_source_rows);
            record(label + "VisitedSourceRows", after.catalog_visited_source_rows - before.catalog_visited_source_rows);
            record(label + "CopiedPages", after.catalog_copied_pages - before.catalog_copied_pages);
            record(label + "VisitedPages", after.catalog_visited_pages - before.catalog_visited_pages);
            record_ms(label + "Ms", elapsed);
        }
}

TEST(AdaptiveGeometryDemandScaling, RawPreparedSubsetCountsDeferredPhysicalCatalogConstruction)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    const auto source = cartesian_scale_source(512);
    auto rows = rows_of(*source);
    std::sort(rows.begin(), rows.end(), [](const auto* a, const auto* b) { return a->native_cell < b->native_cell; });
    (void) snapshot(*source, std::span(rows).first(1));
    (void) snapshot(*source, std::span(rows).subspan(1, 1));
    const auto state = snapshot(*source, std::span(rows).subspan(2, 1));
    const auto selected = state->find(*rows[2]);
    ASSERT_TRUE(selected);
    const auto page = SourceAccess::source_cell(View(state->ordinary_catalog(), *selected));
    ASSERT_EQ(page.source->size(), 64u);
    const auto before = Access::cache_admission_statistics(*source);
    const auto raw =
        snapshot(*source, std::span(rows).first(4), ordinary, AMR::GeometrySnapshotPurpose::OriginalSources);
    const auto after = Access::cache_admission_statistics(*source);
    ASSERT_EQ(raw->ordinary_catalog()->size(), 4u);
    EXPECT_TRUE(raw->native_metrics().empty());
    EXPECT_FALSE(raw->finite_ordinary_catalog());
    EXPECT_FALSE(raw->bounded_catalog());
    // One four-row catalog initialization, one new selector group maximum read.
    EXPECT_EQ(after.catalog_copied_source_rows - before.catalog_copied_source_rows, 4u);
    EXPECT_EQ(after.catalog_visited_source_rows - before.catalog_visited_source_rows, 5u);
    const auto raw_slot = raw->find(*rows[2]);
    ASSERT_TRUE(raw_slot);
    const auto deferred = SourceAccess::source_cell(View(raw->ordinary_catalog(), *raw_slot));
    EXPECT_NE(deferred.source.get(), page.source.get());
    EXPECT_EQ(SourceAccess::base_owner(*deferred.source), SourceAccess::base_owner(*page.source));
    EXPECT_LE(deferred.source->size(), 4u);
    EXPECT_EQ(View(raw->ordinary_catalog(), *raw_slot).volume().value, 1.0 / 512.0);
}

TEST(AdaptiveGeometryDemandScaling, FallbackStateCountsFiniteOrdinaryAndBoundedCatalogChannels)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    for (const auto mode : {ordinary, ArithmeticMode::BoundedCertification})
    {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        AMR::AMRGrid grid;
        if (mode == ordinary)
            grid.origin[0] = -0x1p-100;
        grid.extent[0] = 4;
        const AMR::CartesianAMR mesh(grid);
        std::vector<std::uint64_t> all(mesh.geometry()->num_cells());
        std::iota(all.begin(), all.end(), 0);
        auto detached = mesh.detach_owned(all, {'c', 'h', 'a', 'n'});
        const auto source = std::make_shared<const Hierarchy>(std::move(detached.owned), std::move(detached.families),
            std::move(detached.kernel), Tpetra::getDefaultComm());
        std::array<const AMR::AdaptiveLeaf*, 2> selected{};
        for (const auto& row : source->owned_leaves())
        {
            const auto& key = std::get<AMR::CartesianCellAncestry>(row.key.ancestry);
            if (key.index[0] == 0 && key.index[2] == 0 && key.index[1] < 2)
                selected[key.index[1]] = &row;
        }
        ASSERT_TRUE(selected[0]);
        ASSERT_TRUE(selected[1]);
        const auto bounds = source->native_cartesian_geometry_bounds(*selected[0]);
        ASSERT_TRUE(bounds);
        if (mode == ordinary)
            EXPECT_FALSE(SourceAccess::state_compatible(bounds->lower, bounds->upper));
        const auto before = Access::cache_admission_statistics(*source);
        const auto first = snapshot(*source, std::span(selected).first(1), mode);
        const auto initialized = Access::cache_admission_statistics(*source);
        ASSERT_FALSE(View(first->ordinary_catalog(), 0).is_procedural());
        if (mode == ordinary)
        {
            ASSERT_TRUE(first->finite_ordinary_catalog());
            EXPECT_NE(first->ordinary_catalog(), first->finite_ordinary_catalog());
        }
        else
            ASSERT_TRUE(first->bounded_catalog());
        // Each catalog reads the selected input, its new group maximum and
        // the partial-tail stamp: three decoded reads for each actual catalog.
        EXPECT_EQ(initialized.catalog_copied_source_rows - before.catalog_copied_source_rows, 2u);
        EXPECT_EQ(initialized.catalog_visited_source_rows - before.catalog_visited_source_rows, 6u);
        const auto active = snapshot(*source, std::span(selected).subspan(1, 1), mode);
        const auto expanded = Access::cache_admission_statistics(*source);
        ASSERT_EQ(active->ordinary_catalog()->size(), 2u);
        if (mode == ordinary)
            ASSERT_EQ(active->finite_ordinary_catalog()->size(), 2u);
        else
            ASSERT_EQ(active->bounded_catalog()->size(), 2u);
        // In each catalog, prefix1+oldfront1 creates2 selector rows; repacking
        // the partial explicit source then rebinds that two-row group once.
        EXPECT_EQ(expanded.catalog_copied_source_rows - initialized.catalog_copied_source_rows, 8u);
    }
}
