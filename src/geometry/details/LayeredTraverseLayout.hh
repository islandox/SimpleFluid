/** Immutable XY metadata with arithmetic Z repetitions. */
#pragma once
#include "geometry/details/StructuredTraverseLayout.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"
#include <algorithm>
#include <vector>
#include <limits>

namespace SimpleFluid::Meshes { class SemiStructuredXY_Z; }
namespace SimpleFluid::detail
{
template<class Geometry> inline constexpr bool has_layered_traverse_layout = false;
template<> inline constexpr bool has_layered_traverse_layout<Meshes::SemiStructuredXY_Z> = true;

class LayeredTraverseLayout
{
public:
    using Indexer = Meshes::SemiStructuredIndexer;
    static constexpr size_t invalid_ordinal = std::numeric_limits<size_t>::max();
    struct ResolvedIncidence
    {
        StructuredTraverseIncidence incidence;
        size_t owner, neighbor;
        Indexer::CellID cell, other;
    };
    struct SideIncidence
    {
        Indexer::Ordinal edge;
        int batch;
        signed char orientation;
        bool interior;
    };
    explicit LayeredTraverseLayout(const Meshes::SemiStructMeshTopo& topology)
        : d_indexer(topology.indexer())
    {
        d_offsets.push_back(0);
        d_run_offsets.push_back(0);
        for (size_t base = 0; base < d_indexer.num_cells_per_layer; ++base)
        {
            bool interior = true;
            size_t slot = 2;
            const auto run_begin = d_incidence_runs.size();
            for (auto edge : topology.cell_side_faces(base))
            {
                const auto& side = topology.side_face(edge);
                const bool face_interior = side.neighbor != Meshes::SemiStructMeshTopo::invalid_ordinal;
                const int batch = face_interior ? -1 : side.boundary_id;
                d_incidences.push_back({edge, batch, static_cast<signed char>(side.owner == base ? 1 : -1), face_interior});
                interior = interior && face_interior;
                append(d_incidence_runs, slot++, face_interior, batch);
            }
            d_offsets.push_back(d_incidences.size());
            const auto run_end = d_incidence_runs.size();
            d_run_offsets.push_back(run_end);
            for (size_t run = run_begin; run < run_end; ++run) d_grouped_runs.push_back(run);
            std::stable_sort(d_grouped_runs.begin() + run_begin, d_grouped_runs.begin() + run_end,
                [&](size_t left, size_t right)
                {
                    const auto& lhs = d_incidence_runs[left];
                    const auto& rhs = d_incidence_runs[right];
                    if (lhs.interior != rhs.interior) return lhs.interior;
                    return lhs.batch < rhs.batch;
                });
            size_t interior_end = run_begin;
            while (interior_end < run_end && d_incidence_runs[d_grouped_runs[interior_end]].interior) ++interior_end;
            d_interior_run_ends.push_back(interior_end);
            append(d_cells, base, interior, -1);
        }
        for (const auto& run : d_cells) if (run.interior) d_interior_cells.push_back(run);
        d_batches = topology.boundary_batch_ids();
        const auto& sides = topology.side_faces();
        for (size_t edge = 0; edge < sides.size(); ++edge)
        {
            const bool interior = sides[edge].neighbor != Meshes::SemiStructMeshTopo::invalid_ordinal;
            append(d_side_runs, edge, interior, interior ? -1 : sides[edge].boundary_id);
            if (!interior) d_batches.push_back(sides[edge].boundary_id);
        }
        if (!d_indexer.axial_periodic) { d_batches.push_back(0); d_batches.push_back(1); }
        std::sort(d_batches.begin(), d_batches.end());
        d_batches.erase(std::unique(d_batches.begin(), d_batches.end()), d_batches.end());
    }
    const Indexer& indexer() const noexcept { return d_indexer; }
    size_t num_cells() const noexcept { return d_indexer.total_cells(); }
    size_t num_faces() const noexcept { return d_indexer.total_faces(); }
    size_t storage_bytes() const noexcept
    {
        return sizeof(*this) + (d_offsets.capacity() + d_run_offsets.capacity()
            + d_grouped_runs.capacity() + d_interior_run_ends.capacity()) * sizeof(size_t)
            + d_incidences.capacity() * sizeof(SideIncidence)
            + (d_cells.capacity() + d_side_runs.capacity() + d_incidence_runs.capacity() + d_interior_cells.capacity()) * sizeof(StructuredTraverseRun)
            + d_batches.capacity() * sizeof(int);
    }
    class Incidences
    {
    public:
        Incidences(const LayeredTraverseLayout& layout, size_t cell)
            : d_layout(&layout), d_cell(layout.d_indexer.cell_id(cell)),
              d_side_begin(layout.d_offsets[d_cell.ij]),
              d_size(2 + layout.d_offsets[d_cell.ij + 1] - d_side_begin),
              d_axial_start(layout.d_indexer.face_ordinal({d_cell.ij, 0, Indexer::AXIAL})),
              d_side_start(layout.d_indexer.face_ordinal({0, d_cell.k, Indexer::SIDE})),
              d_cell_start(static_cast<size_t>(layout.d_indexer.num_cells_per_layer) * d_cell.k) {}
        size_t size() const noexcept { return d_size; }
        /** Derive only the canonical ID for retained cache metric refill. */
        size_t face_ordinal(size_t slot) const
        {
            if (slot < 2)
            {
                const auto k = slot == 0 ? d_cell.k : d_layout->d_indexer.axial_periodic
                    ? (d_cell.k + 1) % d_layout->d_indexer.num_layers : d_cell.k + 1;
                return d_axial_start + k;
            }
            return d_side_start + d_layout->d_incidences[d_side_begin + slot - 2].edge;
        }
        StructuredTraverseIncidence operator[](size_t slot) const
        {
            if (slot < 2)
            {
                const auto& indexer = d_layout->d_indexer;
                const auto k = slot == 0 ? d_cell.k : indexer.axial_periodic
                    ? (d_cell.k + 1) % indexer.num_layers : d_cell.k + 1;
                const bool boundary = !indexer.axial_periodic && (k == 0 || k == indexer.num_layers);
                const auto owner = k == 0 ? (indexer.axial_periodic ? indexer.num_layers - 1 : 0) : k - 1;
                return {d_axial_start + k, owner == d_cell.k ? 1 : -1,
                    !boundary, boundary ? (k == 0 ? 0 : 1) : -1};
            }
            const auto& side = d_layout->d_incidences[d_side_begin + slot - 2];
            return {d_side_start + side.edge, side.orientation, side.interior, side.batch};
        }
        ResolvedIncidence resolved(size_t slot, const Meshes::SemiStructMeshTopo& topology) const
        {
            const auto incidence = (*this)[slot];
            const auto& indexer = d_layout->d_indexer;
            Indexer::CellID owner{}, neighbor{};
            size_t owner_ordinal = invalid_ordinal, neighbor_ordinal = invalid_ordinal;
            if (slot < 2)
            {
                const auto face_layer = slot == 0 ? d_cell.k : indexer.axial_periodic
                    ? (d_cell.k + 1) % indexer.num_layers : d_cell.k + 1;
                const auto owner_layer = face_layer == 0 ? (indexer.axial_periodic ? indexer.num_layers - 1 : 0) : face_layer - 1;
                owner = {d_cell.ij, owner_layer};
                owner_ordinal = indexer.cell_ordinal(owner);
                if (incidence.interior)
                {
                    neighbor = {d_cell.ij, face_layer};
                    neighbor_ordinal = indexer.cell_ordinal(neighbor);
                }
            }
            else
            {
                const auto& side = d_layout->d_incidences[d_side_begin + slot - 2];
                const auto& adjacent = topology.side_face(side.edge);
                if (adjacent.owner != Meshes::SemiStructMeshTopo::invalid_ordinal)
                {
                    owner = {adjacent.owner, d_cell.k};
                    owner_ordinal = d_cell_start + adjacent.owner;
                }
                // Check the base sentinel before widening/adding the layer offset.
                if (adjacent.neighbor != Meshes::SemiStructMeshTopo::invalid_ordinal)
                {
                    neighbor = {adjacent.neighbor, d_cell.k};
                    neighbor_ordinal = d_cell_start + adjacent.neighbor;
                }
            }
            if (d_cell != owner && d_cell != neighbor)
                throw std::invalid_argument("Cell is not adjacent to requested face.");
            return {incidence, owner_ordinal, neighbor_ordinal, d_cell, d_cell == owner ? neighbor : owner};
        }
        bool all_interior() const noexcept
        {
            return d_layout->d_interior_run_ends[d_cell.ij] == d_layout->d_run_offsets[d_cell.ij + 1]
                && (d_layout->d_indexer.axial_periodic
                    || (d_cell.k > 0 && d_cell.k + 1 < d_layout->d_indexer.num_layers));
        }
        /** Visit preclassified XY runs with the two arithmetic axial entries. */
        template<class Visitor>
        void visit_runs(StructuredTraverseSelection selection, bool independent,
            std::optional<int> batch, Visitor&& visitor) const
        {
            if (all_interior())
            {
                if (selection != StructuredTraverseSelection::Boundary)
                {
                    StructuredTraverseRun run{0, size(), true, -1};
                    visitor(run);
                }
                return;
            }
            StructuredTraverseRun pending{};
            bool has_pending = false;
            const auto emit = [&](const StructuredTraverseRun& run)
            {
                if (!selected(selection, run.interior)
                    || (selection == StructuredTraverseSelection::Boundary && batch && *batch != run.batch)) return;
                if (has_pending && pending.end == run.begin
                    && pending.interior == run.interior && pending.batch == run.batch)
                    pending.end = run.end;
                else
                {
                    if (has_pending) visitor(pending);
                    pending = run;
                    has_pending = true;
                }
            };
            const auto lower = (*this)[0], upper = (*this)[1];
            const StructuredTraverseRun axial[2]{{0, 1, lower.interior, lower.batch}, {1, 2, upper.interior, upper.batch}};
            const auto begin = d_layout->d_run_offsets[d_cell.ij];
            const auto end = d_layout->d_run_offsets[d_cell.ij + 1];
            const auto interior_end = d_layout->d_interior_run_ends[d_cell.ij];
            if (selection == StructuredTraverseSelection::Interior)
            {
                emit(axial[0]); emit(axial[1]);
                for (size_t run = begin; run < interior_end; ++run)
                    emit(d_layout->d_incidence_runs[d_layout->d_grouped_runs[run]]);
            }
            else if (selection == StructuredTraverseSelection::Boundary && batch)
            {
                emit(axial[0]); emit(axial[1]);
                auto run = std::lower_bound(d_layout->d_grouped_runs.begin() + interior_end,
                    d_layout->d_grouped_runs.begin() + end, *batch,
                    [&](size_t index, int id) { return d_layout->d_incidence_runs[index].batch < id; });
                while (run != d_layout->d_grouped_runs.begin() + end
                    && d_layout->d_incidence_runs[*run].batch == *batch)
                    emit(d_layout->d_incidence_runs[*run++]);
            }
            else if (!independent || batch)
            {
                emit(axial[0]); emit(axial[1]);
                for (size_t run = begin; run < end; ++run) emit(d_layout->d_incidence_runs[run]);
            }
            else
            {
                if (selection == StructuredTraverseSelection::All)
                {
                    if (axial[0].interior) emit(axial[0]);
                    if (axial[1].interior) emit(axial[1]);
                    for (size_t run = begin; run < interior_end; ++run)
                        emit(d_layout->d_incidence_runs[d_layout->d_grouped_runs[run]]);
                }
                bool lower_pending = !axial[0].interior, upper_pending = !axial[1].interior;
                size_t run = interior_end;
                // Merge the sorted row-local patch runs with at most two axial
                // patches, retaining native order for equal patch IDs.
                while (run < end || lower_pending || upper_pending)
                {
                    const auto* side = run < end
                        ? &d_layout->d_incidence_runs[d_layout->d_grouped_runs[run]] : nullptr;
                    if (lower_pending && (!side || axial[0].batch <= side->batch))
                    { emit(axial[0]); lower_pending = false; }
                    else if (upper_pending && (!side || axial[1].batch <= side->batch))
                    { emit(axial[1]); upper_pending = false; }
                    else { emit(*side); ++run; }
                }
            }
            if (has_pending) visitor(pending);
        }
        struct Iterator
        {
            const Incidences* view; size_t slot;
            auto operator*() const { return (*view)[slot]; }
            Iterator& operator++() { ++slot; return *this; }
            bool operator!=(const Iterator& other) const { return slot != other.slot; }
        };
        Iterator begin() const { return {this, 0}; } Iterator end() const { return {this, size()}; }
    private:
        const LayeredTraverseLayout* d_layout;
        Indexer::CellID d_cell;
        size_t d_side_begin, d_size, d_axial_start, d_side_start, d_cell_start;
    };
    Incidences cell_faces(size_t cell) const { return {*this, cell}; }
    template<class Visitor> void visit_cells(StructuredTraverseSelection selection, bool independent, Visitor&& visit) const
    {
        if (independent && selection == StructuredTraverseSelection::All)
        { visit_cells(StructuredTraverseSelection::Interior, false, visit); visit_cells(StructuredTraverseSelection::Boundary, false, visit); return; }
        if (selection == StructuredTraverseSelection::Interior)
        {
            const size_t first = d_indexer.axial_periodic ? 0 : 1;
            const size_t end = d_indexer.axial_periodic ? d_indexer.num_layers
                : (d_indexer.num_layers > 1 ? d_indexer.num_layers - 1 : 1);
            for (size_t k = first; k < end; ++k)
                for (const auto& run : d_interior_cells)
                    visit(StructuredTraverseRun{run.begin + k*d_indexer.num_cells_per_layer,
                        run.end + k*d_indexer.num_cells_per_layer, true, -1});
            return;
        }
        for (size_t k = 0; k < d_indexer.num_layers; ++k)
            for (const auto& run : d_cells)
            {
                const bool interior = run.interior && (d_indexer.axial_periodic || (k > 0 && k+1 < d_indexer.num_layers));
                if (selected(selection, interior)) visit(StructuredTraverseRun{run.begin+k*d_indexer.num_cells_per_layer, run.end+k*d_indexer.num_cells_per_layer, interior, -1});
            }
    }
    template<class Visitor> void visit_faces(StructuredTraverseSelection selection, bool independent, std::optional<int> batch, Visitor&& visit) const
    {
        if (independent && !batch && selection != StructuredTraverseSelection::Interior)
        {
            if (selection == StructuredTraverseSelection::All) visit_faces(StructuredTraverseSelection::Interior, false, {}, visit);
            for (int id : d_batches) visit_faces(StructuredTraverseSelection::Boundary, false, id, visit);
            return;
        }
        auto emit = [&](size_t begin, size_t end, bool interior, int id)
        { if (begin < end && selected(selection, interior) && (!batch || selection == StructuredTraverseSelection::Interior || *batch == id)) visit(StructuredTraverseRun{begin,end,interior,id}); };
        for (size_t b = 0; b < d_indexer.num_cells_per_layer; ++b)
        {
            const auto start = b*d_indexer.num_node_layers;
            if (d_indexer.axial_periodic) emit(start,start+d_indexer.num_layers,true,-1);
            else { emit(start,start+1,false,0); emit(start+1,start+d_indexer.num_layers,true,-1); emit(start+d_indexer.num_layers,start+d_indexer.num_node_layers,false,1); }
        }
        for (size_t k = 0; k < d_indexer.num_layers; ++k)
            for (const auto& run : d_side_runs)
            { const auto start = d_indexer.face_offsets[Indexer::SIDE]+k*d_indexer.num_side_faces_per_layer; emit(start+run.begin,start+run.end,run.interior,run.batch); }
    }
    size_t cell_count(StructuredTraverseSelection selection) const noexcept
    {
        if (selection == StructuredTraverseSelection::All) return num_cells();
        size_t base_interior=0;
        for (const auto& run : d_cells) if (run.interior) base_interior+=run.end-run.begin;
        const size_t layers=d_indexer.axial_periodic ? d_indexer.num_layers
            : (d_indexer.num_layers > 2 ? d_indexer.num_layers-2 : 0);
        const auto interior=base_interior*layers;
        return selection == StructuredTraverseSelection::Interior ? interior : num_cells()-interior;
    }
    size_t face_count(StructuredTraverseSelection selection, std::optional<int> batch={}) const noexcept
    {
        if (selection == StructuredTraverseSelection::All && !batch) return num_faces();
        size_t side_count=0;
        for (const auto& run : d_side_runs)
            if (selected(selection,run.interior) && (!batch || selection == StructuredTraverseSelection::Interior || run.batch == *batch))
                side_count+=run.end-run.begin;
        size_t axial=0;
        if (selection != StructuredTraverseSelection::Boundary)
            axial=d_indexer.axial_periodic ? d_indexer.num_layers : d_indexer.num_layers-1;
        if (!d_indexer.axial_periodic && selection != StructuredTraverseSelection::Interior)
        { if (!batch || *batch==0) ++axial; if (!batch || *batch==1) ++axial; }
        return axial*d_indexer.num_cells_per_layer+side_count*d_indexer.num_layers;
    }
    std::span<const StructuredTraverseRun> xy_interior_cell_runs() const noexcept { return d_interior_cells; }
    const std::vector<int>& boundary_ids() const noexcept { return d_batches; }
private:
    static bool selected(StructuredTraverseSelection selection, bool interior)
    { return selection == StructuredTraverseSelection::All || (selection == StructuredTraverseSelection::Interior) == interior; }
    static void append(std::vector<StructuredTraverseRun>& runs, size_t ordinal, bool interior, int batch)
    {
        if (!runs.empty() && runs.back().end == ordinal && runs.back().interior == interior && runs.back().batch == batch) ++runs.back().end;
        else runs.push_back({ordinal,ordinal+1,interior,batch});
    }
    Indexer d_indexer;
    std::vector<size_t> d_offsets, d_run_offsets, d_grouped_runs, d_interior_run_ends;
    std::vector<SideIncidence> d_incidences;
    std::vector<StructuredTraverseRun> d_cells, d_side_runs, d_incidence_runs, d_interior_cells;
    std::vector<int> d_batches;
};
}
