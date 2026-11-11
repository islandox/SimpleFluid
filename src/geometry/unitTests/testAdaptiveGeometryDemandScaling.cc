/** Scale-oriented retained-storage evidence for sparse adaptive geometry demand.
 *
 * The 128-cell storage gates are constant-fraction guards: fixed per-structure
 * costs dominate there. This fixture measures the same contracts at sizes where
 * per-cell cost dominates, and asserts ratios and per-cell budgets rather than
 * absolute kilobytes: completed batch selection against completed incremental
 * streams, a bounded physical owner count independent of the number of builds,
 * and a bounded retained-bytes-per-cell ratio. Nothing here relaxes the small
 * constant gates in testAdaptiveGeometryStorage.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace PLIC = SimpleFluid::PLIC;
using SimpleFluid::ArithmeticMode;
using Hierarchy = AMR::DistributedAdaptiveHierarchy;
using Snapshot = AMR::AdaptiveGeometrySnapshot;
using View = PLIC::CellGeometryView<ArithmeticMode::Ordinary>;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t ample = 256 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::size_t scale_cells()
{
    const auto* text = std::getenv("SIMPLEFLUID_DEMAND_SCALE_CELLS");
    return text ? std::size_t(std::strtoull(text, nullptr, 10)) : 512;
}

std::shared_ptr<const Hierarchy> cartesian_scale_source(std::size_t cells)
{
    // One region so every cell shares one native recipe, matching the prepared
    // page key space; the cell count is a product of the block shape.
    AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {cells / 64u, 8, 8};
    const AMR::CartesianAMR mesh(grid);
    std::vector<std::uint64_t> indices(mesh.geometry()->num_cells());
    std::iota(indices.begin(), indices.end(), 0);
    auto detached = mesh.detach_owned(indices, {'s', 'c', 'a', 'l'});
    return std::make_shared<const Hierarchy>(std::move(detached.owned), std::move(detached.families),
        std::move(detached.kernel), Tpetra::getDefaultComm());
}

std::shared_ptr<const Snapshot> snapshot(const Hierarchy& source, std::span<const AMR::AdaptiveLeaf* const> rows)
{
    std::size_t used = 0;
    return source.geometry_snapshot(rows, ordinary, AMR::GeometrySnapshotPurpose::State, used, ample);
}

std::size_t physical_owners(const Snapshot& snapshot)
{
    PLIC::GeometryAllocationLedger ordinary_owners, bounded_owners;
    std::size_t used = 0;
    snapshot.retain(ordinary_owners, bounded_owners, used, std::numeric_limits<std::size_t>::max());
    return ordinary_owners.owner_count() + bounded_owners.owner_count();
}

struct StreamResult
{
    std::size_t retained = 0, owners = 0, prepared = 0, catalog_sources = 0;
    double build_ms = 0;
};
} // namespace

namespace
{
struct ScaleReport
{
    std::size_t cells = 0, batch_bytes = 0, batch_owners = 0;
    StreamResult forward, alternating;
};
} // namespace

TEST(AdaptiveGeometryDemandScaling, CompletedStreamsKeepPerCellStorageAndOwnersLinear)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    const auto large = std::max<std::size_t>(scale_cells(), 256u);
    const auto rows_of = [](const Hierarchy& source)
    {
        std::vector<const AMR::AdaptiveLeaf*> rows;
        for (const auto& leaf : source.owned_leaves())
            rows.push_back(&leaf);
        return rows;
    };
    const auto stream = [&](std::size_t cells, bool alternating)
    {
        const auto source = cartesian_scale_source(cells);
        const auto selected = rows_of(*source);
        const auto begin = std::chrono::steady_clock::now();
        auto active = snapshot(*source, std::span(selected).first(4));
        std::vector<const AMR::AdaptiveLeaf*> next;
        for (std::size_t offset = 4; offset < selected.size(); offset += 4)
        {
            next.assign(selected.begin() + offset, selected.begin() + std::min(selected.size(), offset + 4));
            if (alternating)
                std::reverse(next.begin(), next.end());
            active = snapshot(*source, next);
        }
        StreamResult result;
        result.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        result.retained = active->retained_bytes();
        result.owners = physical_owners(*active);
        active->visit_prepared_cartesian_sources([&](const auto& page) { result.prepared += page.size(); });
        result.catalog_sources = active->ordinary_catalog()->source_count();
        EXPECT_EQ(active->ordinary_catalog()->size(), selected.size());
        for (std::size_t slot = 0; slot < selected.size(); ++slot)
            EXPECT_TRUE(active->find(*selected[slot]));
        EXPECT_LE(result.prepared, cells);
        return result;
    };
    const auto measure = [&](std::size_t cells)
    {
        ScaleReport report;
        report.cells = cells;
        const auto source = cartesian_scale_source(cells);
        const auto rows = rows_of(*source);
        EXPECT_EQ(rows.size(), cells);
        const auto batch = snapshot(*source, rows);
        report.batch_bytes = batch->retained_bytes();
        report.batch_owners = physical_owners(*batch);
        report.forward = stream(cells, false);
        report.alternating = stream(cells, true);
        return report;
    };
    const auto small = measure(large / 2);
    const auto big = measure(large);
    const auto record = [](const char* prefix, const ScaleReport& report)
    {
        RecordProperty((std::string(prefix) + "Cells").c_str(), std::to_string(report.cells));
        RecordProperty((std::string(prefix) + "BatchBytes").c_str(), std::to_string(report.batch_bytes));
        RecordProperty((std::string(prefix) + "BatchOwners").c_str(), std::to_string(report.batch_owners));
        for (const auto* result : {&report.forward, &report.alternating})
        {
            RecordProperty((std::string(prefix) + "StreamBytes").c_str(), std::to_string(result->retained));
            RecordProperty((std::string(prefix) + "StreamOwners").c_str(), std::to_string(result->owners));
            RecordProperty((std::string(prefix) + "StreamPrepared").c_str(), std::to_string(result->prepared));
            RecordProperty((std::string(prefix) + "StreamSources").c_str(), std::to_string(result->catalog_sources));
            RecordProperty((std::string(prefix) + "StreamMs").c_str(), std::to_string(result->build_ms));
        }
    };
    record("Half", small);
    record("Full", big);

    // Scale contract: per-cell retained storage and physical owner count must not
    // grow with the grid. The absolute batch*1.5 ratio is the separate constant
    // guard in testAdaptiveGeometryStorage; here the same ratio is only reported.
    const auto per_cell = [](std::size_t bytes, std::size_t cells) { return double(bytes) / double(cells); };
    for (const auto* pair : {&small, &big})
    {
        EXPECT_LE(pair->forward.owners, pair->cells);
        EXPECT_LE(pair->alternating.owners, pair->cells);
        RecordProperty("ForwardBatchRatio", std::to_string(per_cell(pair->forward.retained, pair->batch_bytes)));
        RecordProperty("AlternatingBatchRatio", std::to_string(per_cell(pair->alternating.retained, pair->batch_bytes)));
    }
    EXPECT_LE(per_cell(big.forward.retained, big.cells), per_cell(small.forward.retained, small.cells) * 1.15 + 8);
    EXPECT_LE(
        per_cell(big.alternating.retained, big.cells), per_cell(small.alternating.retained, small.cells) * 1.15 + 8);
    EXPECT_LE(per_cell(big.forward.owners, big.cells), per_cell(small.forward.owners, small.cells) * 1.15 + 0.05);
    EXPECT_LE(per_cell(big.alternating.owners, big.cells), per_cell(small.alternating.owners, small.cells) * 1.15 + 0.05);
}

TEST(AdaptiveGeometryDemandScaling, DoubledUnrelatedWarmSelectionKeepsCopyWorkBounded)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable floating-point construction controls.";
    const auto cells = std::max<std::size_t>(scale_cells(), 384u);
    const auto rows_of = [](const Hierarchy& source)
    {
        std::vector<const AMR::AdaptiveLeaf*> rows;
        for (const auto& leaf : source.owned_leaves())
            rows.push_back(&leaf);
        return rows;
    };
    // The same fixed 4-cell demand is appended after a warm selection of W and
    // of 2W unrelated cells; copied catalog metadata and page visits must stay
    // bounded instead of growing with the warm selection.
    const auto measure = [&](std::size_t warm_rows)
    {
        struct Result
        {
            std::size_t retained = 0, owners = 0, copied = 0, copied_pages = 0, visited_pages = 0, selected = 0;
        } result;
        const auto source = cartesian_scale_source(cells);
        const auto rows = rows_of(*source);
        EXPECT_LE(warm_rows + 4, rows.size());
        auto warm = snapshot(*source, std::span(rows).first(warm_rows));
        const auto before = AMR::adaptive_detail::GeometrySnapshotAccess::cache_admission_statistics(*source);
        const auto appended = std::span(rows).subspan(warm_rows, 4);
        const auto active = snapshot(*source, appended);
        const auto after = AMR::adaptive_detail::GeometrySnapshotAccess::cache_admission_statistics(*source);
        result.retained = active->retained_bytes();
        result.owners = physical_owners(*active);
        result.selected = active->ordinary_catalog()->size();
        result.copied = after.catalog_copied_metadata_bytes - before.catalog_copied_metadata_bytes;
        result.copied_pages = after.catalog_copied_pages - before.catalog_copied_pages;
        result.visited_pages = after.catalog_visited_pages - before.catalog_visited_pages;
        EXPECT_EQ(result.selected, warm_rows + 4);
        // Per-cell retained storage includes the fixed page/source overheads, so
        // only its growth with the warm selection is bounded.
        EXPECT_LE(result.retained / result.selected, std::size_t{700});
        EXPECT_LE(result.owners, cells);
        return result;
    };
    const auto single = measure(cells / 3);
    const auto doubled = measure(2 * (cells / 3));
    RecordProperty("FixedChangeCells", std::string("4"));
    RecordProperty("SingleWarmCopied", std::to_string(single.copied));
    RecordProperty("DoubledWarmCopied", std::to_string(doubled.copied));
    RecordProperty("SingleWarmCopiedPages", std::to_string(single.copied_pages));
    RecordProperty("DoubledWarmCopiedPages", std::to_string(doubled.copied_pages));
    RecordProperty("SingleWarmVisitedPages", std::to_string(single.visited_pages));
    RecordProperty("DoubledWarmVisitedPages", std::to_string(doubled.visited_pages));
    RecordProperty("SingleWarmRetained", std::to_string(single.retained));
    RecordProperty("DoubledWarmRetained", std::to_string(doubled.retained));
    // Copy work for one fixed change is a bounded function of the change, not of
    // the unrelated warm selection it is appended to, and per-cell retained
    // storage does not grow when that warm selection doubles.
    EXPECT_LE(doubled.retained / doubled.selected, single.retained / single.selected + 8);
    EXPECT_LE(doubled.copied, single.copied + single.copied / 2 + 4096);
    EXPECT_LE(doubled.copied_pages, single.copied_pages + 8);
    EXPECT_LE(doubled.visited_pages, single.visited_pages + 64);
}
