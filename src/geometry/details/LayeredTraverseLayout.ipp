/** @file LayeredTraverseLayout.ipp
 * @brief Inline accessors and numerical loops for LayeredTraverseLayout.
 */
#pragma once

#include "geometry/details/LayeredTraverseLayout.hh"

#include <algorithm>
#include <stdexcept>

namespace SimpleFluid::detail
{

// Header-defined queries remain callable by downstream traversal consumers.
inline size_t LayeredTraverseLayout::cell_incidence_count(StructuredTraverseSelection selection) const noexcept
{
    const auto total = 2 * num_cells() + d_incidences.size() * d_indexer.num_layers;
    if (selection == StructuredTraverseSelection::All) return total;
    size_t base_interior = 0;
    for (const auto& run : d_interior_cells)
        for (size_t base = run.begin; base < run.end; ++base)
            base_interior += 2 + d_offsets[base + 1] - d_offsets[base];
    const size_t interior_layers = d_indexer.axial_periodic ? d_indexer.num_layers
        : (d_indexer.num_layers > 2 ? d_indexer.num_layers - 2 : 0);
    const auto interior = base_interior * interior_layers;
    return selection == StructuredTraverseSelection::Interior ? interior : total - interior;
}

inline size_t LayeredTraverseLayout::cell_count(StructuredTraverseSelection selection) const noexcept
{
    if (selection == StructuredTraverseSelection::All)
        return num_cells();
    size_t base_interior = 0;
    for (const auto& run : d_cells)
        if (run.interior)
            base_interior += run.end - run.begin;
    const size_t layers =
        d_indexer.axial_periodic ? d_indexer.num_layers : (d_indexer.num_layers > 2 ? d_indexer.num_layers - 2 : 0);
    const auto interior = base_interior * layers;
    return selection == StructuredTraverseSelection::Interior ? interior : num_cells() - interior;
}

inline size_t LayeredTraverseLayout::face_count(
    StructuredTraverseSelection selection, std::optional<int> batch) const noexcept
{
    if (selection == StructuredTraverseSelection::All && !batch)
        return num_faces();
    size_t side_count = 0;
    for (const auto& run : d_side_runs)
        if (selected(selection, run.interior) &&
            (!batch || selection == StructuredTraverseSelection::Interior || run.batch == *batch))
            side_count += run.end - run.begin;
    size_t axial = 0;
    if (selection != StructuredTraverseSelection::Boundary)
        axial = d_indexer.axial_periodic ? d_indexer.num_layers : d_indexer.num_layers - 1;
    if (!d_indexer.axial_periodic && selection != StructuredTraverseSelection::Interior)
    {
        if (!batch || *batch == 0)
            ++axial;
        if (!batch || *batch == 1)
            ++axial;
    }
    return axial * d_indexer.num_cells_per_layer + side_count * d_indexer.num_layers;
}

inline void LayeredTraverseLayout::append(
    std::vector<StructuredTraverseRun>& runs, size_t ordinal, bool interior, int batch)
{
    if (!runs.empty() && runs.back().end == ordinal && runs.back().interior == interior && runs.back().batch == batch)
        ++runs.back().end;
    else
        runs.push_back({ordinal, ordinal + 1, interior, batch});
}

inline LayeredTraverseLayout::Incidences::Incidences(const LayeredTraverseLayout& layout, size_t cell)
    : d_layout(&layout), d_cell(layout.d_indexer.cell_id(cell)), d_side_begin(layout.d_offsets[d_cell.ij]),
      d_size(2 + layout.d_offsets[d_cell.ij + 1] - d_side_begin),
      d_axial_start(layout.d_indexer.face_ordinal({d_cell.ij, 0, Indexer::AXIAL})),
      d_side_start(layout.d_indexer.face_ordinal({0, d_cell.k, Indexer::SIDE})),
      d_cell_start(static_cast<size_t>(layout.d_indexer.num_cells_per_layer) * d_cell.k)
{
}

inline StructuredTraverseIncidence LayeredTraverseLayout::Incidences::operator[](size_t slot) const
{
    if (slot < 2)
    {
        const auto& indexer = d_layout->d_indexer;
        const auto k = slot == 0                ? d_cell.k
                       : indexer.axial_periodic ? (d_cell.k + 1) % indexer.num_layers
                                                : d_cell.k + 1;
        const bool boundary = !indexer.axial_periodic && (k == 0 || k == indexer.num_layers);
        const auto owner = k == 0 ? (indexer.axial_periodic ? indexer.num_layers - 1 : 0) : k - 1;
        return {d_axial_start + k, owner == d_cell.k ? 1 : -1, !boundary, boundary ? (k == 0 ? 0 : 1) : -1};
    }
    const auto& side = d_layout->d_incidences[d_side_begin + slot - 2];
    return {d_side_start + side.edge, side.orientation, side.interior, side.batch};
}

inline LayeredTraverseLayout::ResolvedIncidence LayeredTraverseLayout::Incidences::resolved(
    size_t slot, const Meshes::SemiStructMeshTopo& topology) const
{
    const auto incidence = (*this)[slot];
    const auto& indexer = d_layout->d_indexer;
    Indexer::CellID owner{}, neighbor{};
    size_t owner_ordinal = invalid_ordinal, neighbor_ordinal = invalid_ordinal;
    if (slot < 2)
    {
        const auto face_layer = slot == 0                ? d_cell.k
                                : indexer.axial_periodic ? (d_cell.k + 1) % indexer.num_layers
                                                         : d_cell.k + 1;
        const auto owner_layer =
            face_layer == 0 ? (indexer.axial_periodic ? indexer.num_layers - 1 : 0) : face_layer - 1;
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

inline LayeredTraverseLayout::Incidences::Iterator& LayeredTraverseLayout::Incidences::Iterator::operator++()
{
    ++slot;
    return *this;
}

template<class Visitor>
inline void LayeredTraverseLayout::visit_cells(
    StructuredTraverseSelection selection, bool independent, Visitor&& visit) const
{
    if (independent && selection == StructuredTraverseSelection::All)
    {
        visit_cells(StructuredTraverseSelection::Interior, false, visit);
        visit_cells(StructuredTraverseSelection::Boundary, false, visit);
        return;
    }
    if (selection == StructuredTraverseSelection::Interior)
    {
        const size_t first = d_indexer.axial_periodic ? 0 : 1;
        const size_t end =
            d_indexer.axial_periodic ? d_indexer.num_layers : (d_indexer.num_layers > 1 ? d_indexer.num_layers - 1 : 1);
        for (size_t k = first; k < end; ++k)
            for (const auto& run : d_interior_cells)
                visit(StructuredTraverseRun{run.begin + k * d_indexer.num_cells_per_layer,
                    run.end + k * d_indexer.num_cells_per_layer, true, -1});
        return;
    }
    for (size_t k = 0; k < d_indexer.num_layers; ++k)
        for (const auto& run : d_cells)
        {
            const bool interior = run.interior && (d_indexer.axial_periodic || (k > 0 && k + 1 < d_indexer.num_layers));
            if (selected(selection, interior))
                visit(StructuredTraverseRun{run.begin + k * d_indexer.num_cells_per_layer,
                    run.end + k * d_indexer.num_cells_per_layer, interior, -1});
        }
}

template<class Visitor>
inline void LayeredTraverseLayout::visit_faces(
    StructuredTraverseSelection selection, bool independent, std::optional<int> batch, Visitor&& visit) const
{
    if (independent && !batch && selection != StructuredTraverseSelection::Interior)
    {
        if (selection == StructuredTraverseSelection::All)
            visit_faces(StructuredTraverseSelection::Interior, false, {}, visit);
        for (int id : d_batches)
            visit_faces(StructuredTraverseSelection::Boundary, false, id, visit);
        return;
    }
    auto emit = [&](size_t begin, size_t end, bool interior, int id)
    {
        if (begin < end && selected(selection, interior) &&
            (!batch || selection == StructuredTraverseSelection::Interior || *batch == id))
            visit(StructuredTraverseRun{begin, end, interior, id});
    };
    for (size_t b = 0; b < d_indexer.num_cells_per_layer; ++b)
    {
        const auto start = b * d_indexer.num_node_layers;
        if (d_indexer.axial_periodic)
            emit(start, start + d_indexer.num_layers, true, -1);
        else
        {
            emit(start, start + 1, false, 0);
            emit(start + 1, start + d_indexer.num_layers, true, -1);
            emit(start + d_indexer.num_layers, start + d_indexer.num_node_layers, false, 1);
        }
    }
    for (size_t k = 0; k < d_indexer.num_layers; ++k)
        for (const auto& run : d_side_runs)
        {
            const auto start = d_indexer.face_offsets[Indexer::SIDE] + k * d_indexer.num_side_faces_per_layer;
            emit(start + run.begin, start + run.end, run.interior, run.batch);
        }
}

template<class Visitor>
inline void LayeredTraverseLayout::Incidences::visit_runs(
    StructuredTraverseSelection selection, bool independent, std::optional<int> batch, Visitor&& visitor) const
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
        if (!selected(selection, run.interior) ||
            (selection == StructuredTraverseSelection::Boundary && batch && *batch != run.batch))
            return;
        if (has_pending && pending.end == run.begin && pending.interior == run.interior && pending.batch == run.batch)
            pending.end = run.end;
        else
        {
            if (has_pending)
                visitor(pending);
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
        emit(axial[0]);
        emit(axial[1]);
        for (size_t run = begin; run < interior_end; ++run)
            emit(d_layout->d_incidence_runs[d_layout->d_grouped_runs[run]]);
    }
    else if (selection == StructuredTraverseSelection::Boundary && batch)
    {
        emit(axial[0]);
        emit(axial[1]);
        auto run =
            std::lower_bound(d_layout->d_grouped_runs.begin() + interior_end, d_layout->d_grouped_runs.begin() + end,
                *batch, [&](size_t index, int id) { return d_layout->d_incidence_runs[index].batch < id; });
        while (run != d_layout->d_grouped_runs.begin() + end && d_layout->d_incidence_runs[*run].batch == *batch)
            emit(d_layout->d_incidence_runs[*run++]);
    }
    else if (!independent || batch)
    {
        emit(axial[0]);
        emit(axial[1]);
        for (size_t run = begin; run < end; ++run)
            emit(d_layout->d_incidence_runs[run]);
    }
    else
    {
        if (selection == StructuredTraverseSelection::All)
        {
            if (axial[0].interior)
                emit(axial[0]);
            if (axial[1].interior)
                emit(axial[1]);
            for (size_t run = begin; run < interior_end; ++run)
                emit(d_layout->d_incidence_runs[d_layout->d_grouped_runs[run]]);
        }
        bool lower_pending = !axial[0].interior, upper_pending = !axial[1].interior;
        size_t run = interior_end;
        // Merge the sorted row-local patch runs with at most two axial
        // patches, retaining native order for equal patch IDs.
        while (run < end || lower_pending || upper_pending)
        {
            const auto* side = run < end ? &d_layout->d_incidence_runs[d_layout->d_grouped_runs[run]] : nullptr;
            if (lower_pending && (!side || axial[0].batch <= side->batch))
            {
                emit(axial[0]);
                lower_pending = false;
            }
            else if (upper_pending && (!side || axial[1].batch <= side->batch))
            {
                emit(axial[1]);
                upper_pending = false;
            }
            else
            {
                emit(*side);
                ++run;
            }
        }
    }
    if (has_pending)
        visitor(pending);
}

} // namespace SimpleFluid::detail
