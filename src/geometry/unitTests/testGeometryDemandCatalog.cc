/** Persistent real-cell catalog source, owner and ordering regressions. */
#include "geometry/PLIC/details/CellCatalogDemandAccess.hh"
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <set>

namespace
{
namespace P = SimpleFluid::PLIC;
namespace A = P::details;
namespace R = A::retention;
using SimpleFluid::ArithmeticMode;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t ample = 64 * 1024 * 1024;
P::GeometryLimits limits()
{
    P::GeometryLimits result;
    result.byte_budget = ample;
    return result;
}
auto native_source(std::size_t count)
{
    P::CartesianSourceDescription input;
    for (std::size_t slot = 0; slot <= count; ++slot)
        input.axes[0].push_back(double(slot));
    input.axes[1] = input.axes[2] = {0, 1};
    for (std::size_t slot = 0; slot < count; ++slot)
    {
        P::CartesianSourceCellSlot cell;
        cell.lower = {std::uint32_t(slot), 0, 0};
        cell.upper = {std::uint32_t(slot + 1), 1, 1};
        input.cells.push_back(cell);
    }
    return P::CartesianCellSource::make(std::move(input), limits());
}
template<ArithmeticMode M>
void check_finite_admission(const std::shared_ptr<const P::CellGeometryCatalog<M>>& catalog, const R::Report& report)
{
    std::size_t physical = 0, control_allowance = 0;
    for (const auto& owner : report.owners)
        if (owner.root_groups & 0b10)
        {
            physical += owner.bytes;
            const std::string_view kind(owner.kind);
            // Point snapshot construction admits a qualified control allowance.
            // The physical census reports the actual objects and buffers.
            if (kind == "point_context" || kind == "native_facts" || kind == "compact_batch")
                control_allowance += 2 * sizeof(void*);
        }
    P::GeometryAllocationLedger measured;
    std::size_t required = 0;
    measured.retain(catalog, required, ample);
    EXPECT_EQ(measured.payload_bytes(), physical + control_allowance);
    ASSERT_GT(required, 0u);
    P::GeometryAllocationLedger exact, rejected;
    std::size_t used = 0, short_used = 0;
    exact.retain(catalog, used, required);
    EXPECT_EQ(used, required);
    EXPECT_THROW(rejected.retain(catalog, short_used, required - 1), P::GeometryError);
    EXPECT_THROW(rejected.retain(catalog, short_used, ample), P::GeometryError);
    exact.retain(catalog, used, required);
    EXPECT_EQ(used, required);
}
template<ArithmeticMode M> void explicit_streams()
{
    using Catalog = P::CellGeometryCatalog<M>;
    using Geometry = P::ConvexCellGeometry<M>;
    const auto geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    for (const auto width : {1u, 2u, 4u})
    {
        std::array<std::size_t, 2> cumulative{}, copies{}, visited{}, current_metadata{};
        for (std::size_t scale = 0; scale < 2; ++scale)
        {
            const auto count = std::size_t{8192} << scale;
            auto initial = Catalog::materialized(std::vector<P::ConvexGeometryPtr<M>>(count, geometry), limits());
            std::weak_ptr<const Catalog> initial_lifetime = initial;
            const auto accepted = A::CellCatalogDemandAccess::initialize(initial, limits());
            initial.reset();
            EXPECT_TRUE(initial_lifetime.expired());
            auto current = accepted;
            for (std::size_t batch = 0; batch < 40; ++batch)
            {
                const auto requested =
                    Catalog::materialized(std::vector<P::ConvexGeometryPtr<M>>(width, geometry), limits());
                std::vector<std::optional<std::size_t>> old(width);
                A::CellCatalogDemandWork work;
                current = A::CellCatalogDemandAccess::extended(current, requested, old, limits(), &work);
                cumulative[scale] += work.cumulative_metadata_bytes;
                copies[scale] += work.pages.copied_payload_bytes;
                visited[scale] += work.pages.visited_pages + work.pages.visited_source_rows;
                EXPECT_EQ(current->size(), count + (batch + 1) * width);
                EXPECT_EQ(current->source_count(), 0u);
                for (const auto slot : {std::size_t{0}, current->size() / 2, current->size() - 1})
                    EXPECT_EQ(
                        A::NativeCellGeometrySourceAccess::existing_geometry(P::CellGeometryView<M>(current, slot)),
                        geometry);
            }
            SCOPED_TRACE(static_cast<unsigned>(M));
            SCOPED_TRACE(width);
            SCOPED_TRACE(scale);
            std::size_t shell_pages = 0, shared_pages = 0;
            if constexpr (M == ArithmeticMode::Exact)
            {
                // The finite geometry probe supports Exact identity roots.
                // The catalog visitor also exposes Exact view storage directly.
                struct Owners
                {
                    std::map<const void*, std::size_t> metadata;
                    std::map<const void*, std::size_t> shells;
                };
                const auto capture = [](const Catalog& catalog)
                {
                    namespace G = A::geometry_demand;
                    Owners owners;
                    A::CellCatalogDemandAccess::visit_borrowed_dependencies(catalog,
                        {&owners, [](void* raw, const void* owner, const void*, std::size_t bytes)
                            { return static_cast<Owners*>(raw)->metadata.emplace(owner, bytes).second; }},
                        {&owners, [](void* raw, const void*, G::SourceKind kind, const void* owner)
                            {
                                EXPECT_EQ(kind, G::SourceKind::Explicit);
                                const auto& page = *static_cast<const typename Catalog::ExplicitPage*>(owner);
                                const auto bytes = sizeof(page) + page.cells.capacity() * sizeof(page.cells[0]);
                                static_cast<Owners*>(raw)->shells.emplace(owner, bytes);
                            }});
                    return owners;
                };
                const auto before = capture(*accepted), after = capture(*current);
                shell_pages = after.shells.size();
                std::size_t shell_bytes = 0;
                for (const auto& [page, bytes] : after.shells)
                {
                    shared_pages += before.shells.contains(page);
                    shell_bytes += bytes;
                }
                EXPECT_EQ(shell_bytes, shell_pages * sizeof(typename Catalog::ExplicitPage) +
                                           current->size() * sizeof(P::ConvexGeometryPtr<M>));
                for (const auto& [owner, bytes] : after.metadata)
                    if (!before.metadata.contains(owner))
                        current_metadata[scale] += bytes;
            }
            else
            {
                const std::array roots{R::root(*accepted, 0), R::root(*current, 1)};
                std::vector<std::byte> memory(16 * 1024 * 1024);
                R::ProbeScratch scratch(memory);
                const auto report = R::inspect_retention(roots, {}, scratch);
                ASSERT_TRUE(report.complete()) << report.detail;
                check_finite_admission(current, report);
                for (const auto& owner : report.owners)
                {
                    if (std::string_view(owner.kind) == "cell_shell_page")
                    {
                        shell_pages += (owner.root_groups & 0b10) != 0;
                        shared_pages += owner.root_groups == 0b11;
                    }
                    if (std::string_view(owner.kind) == "external_owner" && owner.root_groups == 0b10)
                        current_metadata[scale] += owner.bytes;
                }
            }
            EXPECT_EQ(shared_pages, count / 64);
            EXPECT_EQ(shell_pages, count / 64 + (40 * width + 63) / 64);
        }
        EXPECT_EQ(cumulative[0], cumulative[1]);
        EXPECT_EQ(copies[0], copies[1]);
        EXPECT_EQ(current_metadata[0], current_metadata[1]);
        // Probe boundary children stay bounded by one 32-child branch page.
        EXPECT_LE(visited[1], visited[0] + 40 * 32);
    }
}
namespace G = A::geometry_demand;
struct CapturedSelector
{
    std::shared_ptr<const void> control;
    const void* parent;
    std::vector<G::OrderedSourceSlot> rows;
    std::vector<G::DemandPageBuffer> buffers;
};
struct CapturedSource
{
    std::shared_ptr<const void> control;
    std::map<const void*, std::size_t> buffers;
};
struct DemandGraph
{
    std::map<const void*, const void*> parent;
    std::map<const void*, std::size_t> bytes, depth;
    std::map<const void*, std::shared_ptr<const void>> controls;
    std::map<const void*, CapturedSelector> selectors;
    std::map<const void*, CapturedSource> sources;
};
template<class T, class U> bool same_control(const std::shared_ptr<T>& a, const std::shared_ptr<U>& b)
{
    return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
template<ArithmeticMode M> DemandGraph demand_graph(const P::CellGeometryCatalog<M>& catalog)
{
    using Catalog = P::CellGeometryCatalog<M>;
    DemandGraph graph;
    A::CellCatalogDemandAccess::visit_borrowed_dependencies(catalog,
        {&graph,
            [](void* raw, const void* owner, const void* parent, std::size_t bytes)
            {
                auto& graph = *static_cast<DemandGraph*>(raw);
                graph.parent.emplace(owner, parent);
                graph.depth.emplace(owner, parent ? graph.depth.at(parent) + 1 : 1);
                return graph.bytes.emplace(owner, bytes).second;
            }},
        {nullptr, [](void*, const void*, G::SourceKind, const void*) {}});
    A::CellCatalogDemandAccess::visit_dependencies(catalog,
        {&graph, [](void* raw, const std::shared_ptr<const void>& owner, std::size_t, std::size_t)
            { return static_cast<DemandGraph*>(raw)->controls.emplace(owner.get(), owner).second; }},
        {&graph, [](void* raw, G::SourceKind kind, const std::shared_ptr<const void>& owner)
            {
                auto& sources = static_cast<DemandGraph*>(raw)->sources;
                if (sources.contains(owner.get()))
                    return;
                CapturedSource source{owner, {}};
                if (kind == G::SourceKind::Explicit)
                {
                    const auto& page = *static_cast<const typename Catalog::ExplicitPage*>(owner.get());
                    source.buffers.emplace(page.cells.data(), page.cells.capacity() * sizeof(page.cells[0]));
                }
                else
                    A::NativeCellGeometrySourceAccess::visit_payload(
                        *static_cast<const P::CartesianCellSource*>(owner.get()),
                        [&](const void* data, std::size_t bytes) { source.buffers.emplace(data, bytes); });
                sources.emplace(owner.get(), std::move(source));
            }});
    A::CellCatalogDemandAccess::visit_selector_pages(catalog,
        [&](const G::BorrowedSelectorPage& page)
        {
            graph.selectors.emplace(page.control.get(),
                CapturedSelector{page.control, page.index_parent, {page.rows.begin(), page.rows.end()},
                    {page.buffers.begin(), page.buffers.end()}});
        });
    return graph;
}
std::vector<const void*> index_path(const DemandGraph& graph, const CapturedSelector& page)
{
    std::vector<const void*> result;
    // Catalog demand State and demand-order State occupy depths one and two.
    for (auto owner = page.parent; owner && graph.depth.at(owner) > 2; owner = graph.parent.at(owner))
        result.push_back(owner);
    return result;
}
void expect_selector_unchanged(const CapturedSelector& before, const CapturedSelector& after)
{
    EXPECT_TRUE(same_control(before.control, after.control));
    EXPECT_EQ(before.buffers, after.buffers);
    ASSERT_EQ(before.rows.size(), after.rows.size());
    for (std::size_t slot = 0; slot < before.rows.size(); ++slot)
    {
        EXPECT_EQ(before.rows[slot].stamp, after.rows[slot].stamp);
        EXPECT_EQ(before.rows[slot].physical.row, after.rows[slot].physical.row);
        EXPECT_EQ(before.rows[slot].physical.kind, after.rows[slot].physical.kind);
        EXPECT_TRUE(same_control(before.rows[slot].physical.source, after.rows[slot].physical.source));
    }
}
template<ArithmeticMode M> void fixed_catalog_frontier()
{
    using Catalog = P::CellGeometryCatalog<M>;
    using Geometry = P::ConvexCellGeometry<M>;
    const auto box = Geometry::box({0, 0, 0}, {1, 1, 1});
    constexpr std::size_t capacity = 64, changed = 4, replacement_groups = 2, ordered_edits = 3;
    for (const std::size_t count : {8192, 16384})
    {
        SCOPED_TRACE(static_cast<unsigned>(M));
        SCOPED_TRACE(count);
        A::CellCatalogDemandWork initialized;
        const auto warm = A::CellCatalogDemandAccess::initialize(
            Catalog::materialized(std::vector<P::ConvexGeometryPtr<M>>(count, box), limits()), limits(), &initialized);
        const auto before = demand_graph(*warm);
        ASSERT_EQ(before.selectors.size(), count / capacity);
        EXPECT_EQ(initialized.pages.copied_source_rows, count);
        EXPECT_EQ(initialized.pages.visited_source_rows, count + count / capacity);
        const auto first = std::find_if(before.selectors.begin(), before.selectors.end(),
            [](const auto& value) { return value.second.rows.front().stamp == G::OrderStamp{1, 0}; });
        ASSERT_NE(first, before.selectors.end());
        ASSERT_EQ(first->second.rows.size(), capacity);
        const auto path = index_path(before, first->second);
        ASSERT_EQ(path.size(), 2u); // 128/256 selector groups, 64-row/32-child index.
        // One deletion may repair its immediate right sibling. All other index
        // nodes, selectors and physical source pages are outside the frontier.
        std::map<std::size_t, const void*> ordered_leaves;
        for (const auto& [identity, page] : before.selectors)
        {
            const auto rank = page.rows.front().stamp.position;
            const auto found = ordered_leaves.find(rank);
            if (found == ordered_leaves.end())
                ordered_leaves.emplace(rank, page.parent);
        }
        std::vector<const void*> leaf_order;
        for (const auto& [rank, leaf] : ordered_leaves)
            if (std::find(leaf_order.begin(), leaf_order.end(), leaf) == leaf_order.end())
                leaf_order.push_back(leaf);
        ASSERT_GE(leaf_order.size(), 2u);
        std::set<const void*> affected(path.begin(), path.end());
        affected.insert(leaf_order[1]);
        affected.insert(first->first);
        for (const auto& [owner, depth] : before.depth)
            if (depth <= 2)
                affected.insert(owner);

        const auto request = Catalog::materialized(std::vector<P::ConvexGeometryPtr<M>>(changed, box), limits());
        const std::array<std::optional<std::size_t>, changed> previous{};
        A::CellCatalogDemandWork work;
        const auto active = A::CellCatalogDemandAccess::extended(warm, request, previous, limits(), &work);
        const auto after = demand_graph(*active);
        ASSERT_EQ(active->size(), count + changed);
        ASSERT_EQ(after.selectors.size(), before.selectors.size() + replacement_groups - 1);
        EXPECT_FALSE(after.selectors.contains(first->first));
        std::map<std::pair<std::uint64_t, std::uint64_t>, G::SourceSlot> current_rows;
        for (const auto& [owner, page] : after.selectors)
            for (const auto& row : page.rows)
                EXPECT_TRUE(
                    current_rows.emplace(std::pair{row.stamp.generation, row.stamp.position}, row.physical).second);
        for (const auto& [owner, page] : before.selectors)
            for (const auto& row : page.rows)
            {
                const auto found = current_rows.find({row.stamp.generation, row.stamp.position});
                ASSERT_NE(found, current_rows.end());
                EXPECT_TRUE(same_control(row.physical.source, found->second.source));
                EXPECT_EQ(row.physical.row, found->second.row);
                EXPECT_EQ(row.physical.kind, found->second.kind);
            }
        for (const auto& [owner, source] : before.sources)
        {
            const auto found = after.sources.find(owner);
            ASSERT_NE(found, after.sources.end());
            EXPECT_TRUE(same_control(source.control, found->second.control));
            EXPECT_EQ(source.buffers, found->second.buffers);
        }
        for (const auto& [owner, page] : before.selectors)
            if (owner != first->first)
            {
                const auto found = after.selectors.find(owner);
                ASSERT_NE(found, after.selectors.end());
                expect_selector_unchanged(page, found->second);
            }
        for (const auto& [owner, control] : before.controls)
            if (!affected.contains(owner))
            {
                const auto found = after.controls.find(owner);
                ASSERT_NE(found, after.controls.end());
                EXPECT_TRUE(same_control(control, found->second));
                EXPECT_EQ(before.bytes.at(owner), after.bytes.at(owner));
            }
        // Expectations come from the input: copy the old first64 plus new4;
        // decode each requested row, the first64, and two new group maxima.
        EXPECT_EQ(work.pages.copied_source_rows, capacity + changed);
        EXPECT_EQ(work.pages.visited_source_rows, changed + capacity + replacement_groups);
        EXPECT_EQ(work.pages.visited_pages, (ordered_edits + 1) * path.size());
        const auto copied_page_bound = 6 * ordered_edits * path.size() + replacement_groups;
        ASSERT_LT(copied_page_bound, before.selectors.size());
        EXPECT_LE(work.pages.copied_pages, copied_page_bound);
        EXPECT_LT(work.pages.copied_source_rows, count);
        EXPECT_LT(work.pages.visited_source_rows, count);
        // All original geometries stay reachable at unchanged order stamps.
        for (const auto slot : {changed, count / 2 + changed, count + changed - 1})
            EXPECT_EQ(A::NativeCellGeometrySourceAccess::existing_geometry(P::CellGeometryView<M>(active, slot)), box);
    }
}
template<ArithmeticMode M> void persistent_requested_work()
{
    using Catalog = P::CellGeometryCatalog<M>;
    const auto box = P::ConvexCellGeometry<M>::box({0, 0, 0}, {1, 1, 1});
    constexpr std::size_t count = 8192, requested_height = 2;
    const auto accepted = A::CellCatalogDemandAccess::initialize(
        Catalog::materialized(std::vector<P::ConvexGeometryPtr<M>>(64, box), limits()), limits());
    const auto flat = Catalog::materialized(std::vector<P::ConvexGeometryPtr<M>>(count, box), limits());
    const auto persistent = A::CellCatalogDemandAccess::initialize(flat, limits());
    const auto graph = demand_graph(*persistent);
    ASSERT_EQ(graph.selectors.size(), count / 64);
    for (const auto& [owner, selector] : graph.selectors)
        ASSERT_EQ(index_path(graph, selector).size(), requested_height);
    const std::vector<std::optional<std::size_t>> previous(count);
    A::CellCatalogDemandWork flat_work, persistent_work;
    const auto flat_result = A::CellCatalogDemandAccess::extended(accepted, flat, previous, limits(), &flat_work);
    const auto persistent_result =
        A::CellCatalogDemandAccess::extended(accepted, persistent, previous, limits(), &persistent_work);
    EXPECT_EQ(flat_result->size(), persistent_result->size());
    EXPECT_EQ(persistent_work.pages.visited_pages, flat_work.pages.visited_pages + count * requested_height);
    EXPECT_EQ(persistent_work.pages.visited_source_rows, flat_work.pages.visited_source_rows);
    EXPECT_EQ(persistent_work.pages.copied_source_rows, flat_work.pages.copied_source_rows);
    EXPECT_EQ(persistent_work.pages.copied_pages, flat_work.pages.copied_pages);
}
template<ArithmeticMode M> void rebound_work()
{
    using Catalog = P::CellGeometryCatalog<M>;
    const auto first = native_source(128), replacement = native_source(128);
    std::vector<typename Catalog::Entry> entries;
    for (std::size_t slot = 0; slot < 128; ++slot)
        entries.emplace_back(P::NativeCellGeometrySlot{0, slot});
    const auto accepted = A::CellCatalogDemandAccess::initialize(
        Catalog::make(typename Catalog::SourceTable{first}, std::move(entries), limits()), limits());
    const auto before = demand_graph(*accepted);
    const std::array<std::size_t, 1> ordinal{0};
    const std::array physical{G::SourceSlot{replacement, 0, G::SourceKind::Cartesian}};
    A::CellCatalogDemandWork work;
    const auto active = A::CellCatalogDemandAccess::rebound(accepted, ordinal, physical, limits(), &work);
    const auto after = demand_graph(*active);
    // Two direct old-order reads, two binary searches of first of64 (7+1),
    // one full64 selector copy and its maximum read. No old row count enters it.
    EXPECT_EQ(work.pages.copied_source_rows, 64u);
    EXPECT_EQ(work.pages.visited_source_rows, 2u + 2 * 8 + 64 + 1);
    for (const auto& [owner, page] : before.selectors)
        if (page.rows.front().stamp.position != 0)
        {
            const auto found = after.selectors.find(owner);
            ASSERT_NE(found, after.selectors.end());
            expect_selector_unchanged(page, found->second);
        }
    EXPECT_TRUE(after.sources.contains(first.get()));
    EXPECT_TRUE(after.sources.contains(replacement.get()));
    EXPECT_TRUE(same_control(before.sources.at(first.get()).control, after.sources.at(first.get()).control));
    EXPECT_EQ(before.sources.at(first.get()).buffers, after.sources.at(first.get()).buffers);
}
} // namespace

TEST(GeometryDemandCatalog, MixedSourcesPreserveValuesFramesAndActualClosedShellPages)
{
    using Catalog = P::CellGeometryCatalog<ordinary>;
    using Geometry = P::ConvexCellGeometry<ordinary>;
    using View = P::CellGeometryView<ordinary>;
    const auto source = native_source(512);
    std::vector<Catalog::Entry> entries;
    for (std::size_t slot = 0; slot < 256; ++slot)
        if (slot % 3)
            entries.emplace_back(P::NativeCellGeometrySlot{0, slot});
        else
            entries.emplace_back(Geometry::box({0, 0, 0}, {double(slot + 1), 1, 1}));
    auto input = Catalog::make(source, std::move(entries), limits());
    std::weak_ptr<const Catalog> source_catalog = input;
    const auto accepted = A::CellCatalogDemandAccess::initialize(input, limits());
    input.reset();
    EXPECT_TRUE(source_catalog.expired());
    const auto old_tail = A::NativeCellGeometrySourceAccess::existing_geometry(View(accepted, 255));
    const auto added = Geometry::box({0, 0, 0}, {777, 1, 1});
    std::vector<Catalog::Entry> requested_entries;
    requested_entries.emplace_back(accepted->explicit_entry(255));
    requested_entries.emplace_back(added);
    requested_entries.emplace_back(P::NativeCellGeometrySlot{0, 1});
    requested_entries.emplace_back(P::NativeCellGeometrySlot{0, 400});
    auto requested = Catalog::make(source, std::move(requested_entries), limits());
    std::weak_ptr<const Catalog> requested_lifetime = requested;
    const std::array<std::optional<std::size_t>, 4> previous{255, std::nullopt, 1, std::nullopt};
    const auto next = A::CellCatalogDemandAccess::extended(accepted, requested, previous, limits());
    requested.reset();
    EXPECT_TRUE(requested_lifetime.expired());
    ASSERT_EQ(next->size(), 258u);
    EXPECT_EQ(next->source_count(), 1u);
    EXPECT_EQ(A::NativeCellGeometrySourceAccess::existing_geometry(View(next, 0)), old_tail);
    EXPECT_EQ(A::NativeCellGeometrySourceAccess::existing_geometry(View(next, 1)), added);
    for (const auto [slot, cell] : {std::pair{2u, 1u}, std::pair{3u, 400u}})
    {
        const auto actual = A::NativeCellGeometrySourceAccess::source_cell(View(next, slot));
        EXPECT_EQ(actual.source, source);
        EXPECT_EQ(actual.cell_slot, cell);
        EXPECT_EQ(View(next, slot).volume().value, 1);
    }
    std::size_t output = 4;
    for (std::size_t old = 0; old < accepted->size(); ++old)
    {
        if (old == 255 || old == 1)
            continue;
        const View before(accepted, old), after(next, output++);
        EXPECT_EQ(before.is_procedural(), after.is_procedural());
        if (before.is_procedural())
        {
            const auto a = A::NativeCellGeometrySourceAccess::source_cell(before);
            const auto b = A::NativeCellGeometrySourceAccess::source_cell(after);
            EXPECT_EQ(a.source, b.source);
            EXPECT_EQ(a.cell_slot, b.cell_slot);
        }
        else
            EXPECT_EQ(A::NativeCellGeometrySourceAccess::existing_geometry(before),
                A::NativeCellGeometrySourceAccess::existing_geometry(after));
    }
    const std::array roots{R::root(*accepted, 0), R::root(*next, 1)};
    std::vector<std::byte> memory(16 * 1024 * 1024);
    R::ProbeScratch scratch(memory);
    const auto report = R::inspect_retention(roots, {}, scratch);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(std::count_if(report.owners.begin(), report.owners.end(), [](const auto& owner)
                  { return std::string_view(owner.kind) == "cell_shell_page" && owner.root_groups == 0b11; }),
        1);
    check_finite_admission(next, report);
}

TEST(GeometryDemandCatalog, ExplicitStreamingCompactsTailMetadataForEveryCatalogArithmeticMode)
{
    explicit_streams<ordinary>();
    explicit_streams<ArithmeticMode::BoundedCertification>();
    explicit_streams<ArithmeticMode::Exact>();
}

TEST(GeometryDemandCatalog, OwningSelectionAndPermutationDetachFromThePersistentCatalogController)
{
    using Catalog = P::CellGeometryCatalog<ordinary>;
    const auto box = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    auto original = A::CellCatalogDemandAccess::initialize(
        Catalog::materialized(std::vector<P::ConvexGeometryPtr<ordinary>>(130, box), limits()), limits());
    std::weak_ptr<const Catalog> lifetime = original;
    const std::array<std::size_t, 4> selection{129, 0, 129, 63};
    const auto selected = Catalog::selected(original, selection, limits());
    std::vector<std::size_t> order(original->size());
    std::iota(order.rbegin(), order.rend(), 0);
    const auto reversed = Catalog::reordered(original, order, limits());
    auto malformed = order;
    malformed[0] = malformed[1];
    EXPECT_THROW((void) Catalog::reordered(original, malformed, limits()), P::GeometryError);
    original.reset();
    EXPECT_TRUE(lifetime.expired());
    for (std::size_t slot = 0; slot < selected->size(); ++slot)
        EXPECT_EQ(
            A::NativeCellGeometrySourceAccess::existing_geometry(P::CellGeometryView<ordinary>(selected, slot)), box);
    EXPECT_EQ(reversed->size(), 130u);
    EXPECT_EQ(A::NativeCellGeometrySourceAccess::existing_geometry(P::CellGeometryView<ordinary>(reversed, 0)), box);
}

TEST(GeometryDemandCatalog, FrozenTypedExtensionAdmitsExactMetadataAndRejectsOneByteShort)
{
    using Catalog = P::CellGeometryCatalog<ordinary>;
    const auto box = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto accepted = A::CellCatalogDemandAccess::initialize(
        Catalog::materialized(std::vector<P::ConvexGeometryPtr<ordinary>>(130, box), limits()), limits());
    const auto requested = Catalog::materialized(std::vector<P::ConvexGeometryPtr<ordinary>>(4, box), limits());
    const std::array<std::optional<std::size_t>, 4> previous{129, std::nullopt, 63, std::nullopt};
    A::CellCatalogDemandWork measured;
    const auto next = A::CellCatalogDemandAccess::extended(accepted, requested, previous, limits(), &measured);
    ASSERT_GT(measured.cumulative_metadata_bytes, 0u);
    auto exact = limits();
    exact.byte_budget = measured.cumulative_metadata_bytes;
    const auto repeated = A::CellCatalogDemandAccess::extended(accepted, requested, previous, exact);
    EXPECT_EQ(repeated->size(), next->size());
    auto short_limit = exact;
    --short_limit.byte_budget;
    A::CellCatalogDemandWork rejected;
    EXPECT_THROW((void) A::CellCatalogDemandAccess::extended(accepted, requested, previous, short_limit, &rejected),
        P::GeometryError);
    EXPECT_GT(rejected.cumulative_metadata_bytes, 0u);
    EXPECT_LE(rejected.cumulative_metadata_bytes, short_limit.byte_budget);
    EXPECT_EQ(accepted->size(), 130u);
    EXPECT_EQ(A::NativeCellGeometrySourceAccess::existing_geometry(P::CellGeometryView<ordinary>(accepted, 129)), box);
    EXPECT_EQ(A::CellCatalogDemandAccess::extended(accepted, requested, previous, limits())->size(), 132u);
}

TEST(GeometryDemandCatalog, QueuedSameAddressWithForeignControlRejectsDuringWalk)
{
    using Kind = P::allocation_detail::Kind;
    const auto geometry = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const P::ConvexGeometryPtr<ordinary> foreign(geometry.get(), [](const auto*) {});
    P::GeometryAllocationLedger::Impl ledger;
    std::size_t used = 0;
    ledger.push(Kind::TypedWrapper, foreign, used, ample);
    ledger.push(Kind::TypedWrapper, geometry, used, ample);
    ASSERT_EQ(ledger.pending.size(), 2u);
    EXPECT_THROW(ledger.walk<ordinary>(used, ample), P::GeometryError);
    EXPECT_TRUE(ledger.owners.contains({Kind::TypedWrapper, geometry.get()}));
}

TEST(GeometryDemandCatalog, NativeExportsKeepDistinctControlsAndReplacementReleasesOnlyRetiredSource)
{
    using Catalog = P::CellGeometryCatalog<ordinary>;
    using View = P::CellGeometryView<ordinary>;
    const auto first = native_source(64);
    auto second = native_source(64);
    ASSERT_NE(first, second);
    std::weak_ptr<const P::CartesianCellSource> lifetime = second;
    std::vector<Catalog::Entry> entries{P::NativeCellGeometrySlot{0, 1}, P::NativeCellGeometrySlot{1, 2},
        P::NativeCellGeometrySlot{0, 3}, P::NativeCellGeometrySlot{1, 4}};
    auto accepted = A::CellCatalogDemandAccess::initialize(
        Catalog::make(Catalog::SourceTable{first, second}, std::move(entries), limits()), limits());
    const std::array<std::size_t, 3> subset{1, 1, 0};
    auto selected = Catalog::selected(accepted, subset, limits());
    const std::array<std::size_t, 4> reverse{3, 2, 1, 0};
    auto reordered = Catalog::reordered(accepted, reverse, limits());
    EXPECT_EQ(selected->source_count(), 2u);
    EXPECT_EQ(reordered->source_count(), 2u);
    for (std::size_t slot = 0; slot < subset.size(); ++slot)
    {
        const auto original = A::NativeCellGeometrySourceAccess::source_cell(View(accepted, subset[slot]));
        const auto exported = A::NativeCellGeometrySourceAccess::source_cell(View(selected, slot));
        EXPECT_EQ(exported.source, original.source);
        EXPECT_EQ(exported.cell_slot, original.cell_slot);
    }
    for (std::size_t slot = 0; slot < reverse.size(); ++slot)
    {
        const auto original = A::NativeCellGeometrySourceAccess::source_cell(View(accepted, reverse[slot]));
        const auto exported = A::NativeCellGeometrySourceAccess::source_cell(View(reordered, slot));
        EXPECT_EQ(exported.source, original.source);
        EXPECT_EQ(exported.cell_slot, original.cell_slot);
    }
    const auto requested =
        Catalog::make(first, {P::NativeCellGeometrySlot{0, 5}, P::NativeCellGeometrySlot{0, 6}}, limits());
    const std::array<std::optional<std::size_t>, 2> previous{1, 3};
    const auto next = A::CellCatalogDemandAccess::extended(accepted, requested, previous, limits());
    EXPECT_EQ(next->source_count(), 1u);
    EXPECT_EQ(accepted->source_count(), 2u);
    second.reset();
    accepted.reset();
    selected.reset();
    EXPECT_FALSE(lifetime.expired());
    reordered.reset();
    EXPECT_TRUE(lifetime.expired());
    for (std::size_t slot = 0; slot < next->size(); ++slot)
        EXPECT_EQ(A::NativeCellGeometrySourceAccess::source_cell(View(next, slot)).source, first);
}

TEST(GeometryDemandCatalog, FixedFourCellAppendPreservesTheIndependentFrontierInEveryArithmeticMode)
{
    fixed_catalog_frontier<ArithmeticMode::Ordinary>();
    fixed_catalog_frontier<ArithmeticMode::BoundedCertification>();
    fixed_catalog_frontier<ArithmeticMode::Exact>();
}
TEST(GeometryDemandCatalog, PersistentRequestedCatalogCountsEveryInputTreeLookup)
{
    persistent_requested_work<ArithmeticMode::Ordinary>();
    persistent_requested_work<ArithmeticMode::BoundedCertification>();
    persistent_requested_work<ArithmeticMode::Exact>();
}
TEST(GeometryDemandCatalog, OrdinaryCartesianSourceReboundCountsEveryDecodedSelector)
{
    rebound_work<ArithmeticMode::Ordinary>();
}

TEST(GeometryDemandCatalog, BoundedCartesianSourceReboundRejectsUnsupportedExplicitSourceWithoutCopyingSelectors)
{
    constexpr auto bounded = ArithmeticMode::BoundedCertification;
    using Catalog = P::CellGeometryCatalog<bounded>;
    const auto box = P::ConvexCellGeometry<bounded>::box({0, 0, 0}, {1, 1, 1});
    const auto accepted = A::CellCatalogDemandAccess::initialize(
        Catalog::materialized(std::vector<P::ConvexGeometryPtr<bounded>>(128, box), limits()), limits());
    const auto source = native_source(128);
    const std::array<std::size_t, 1> ordinal{0};
    const std::array physical{G::SourceSlot{source, 0, G::SourceKind::Cartesian}};
    A::CellCatalogDemandWork work;
    EXPECT_THROW(
        (void) A::CellCatalogDemandAccess::rebound(accepted, ordinal, physical, limits(), &work), P::GeometryError);
    EXPECT_EQ(work.pages.copied_source_rows, 0u);
    EXPECT_EQ(work.pages.visited_source_rows, 1u);
    EXPECT_EQ(accepted->size(), 128u);
}
