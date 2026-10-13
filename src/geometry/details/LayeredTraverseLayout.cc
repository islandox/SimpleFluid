/** @file LayeredTraverseLayout.cc
 * @brief Construction of retained XY traversal metadata.
 */
#include "geometry/details/LayeredTraverseLayout.hh"

#include <algorithm>

namespace SimpleFluid::detail
{

LayeredTraverseLayout::LayeredTraverseLayout(const Meshes::SemiStructMeshTopo& topology) : d_indexer(topology.indexer())
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
        for (size_t run = run_begin; run < run_end; ++run)
            d_grouped_runs.push_back(run);
        std::stable_sort(d_grouped_runs.begin() + run_begin, d_grouped_runs.begin() + run_end,
            [&](size_t left, size_t right)
            {
                const auto& lhs = d_incidence_runs[left];
                const auto& rhs = d_incidence_runs[right];
                if (lhs.interior != rhs.interior)
                    return lhs.interior;
                return lhs.batch < rhs.batch;
            });
        size_t interior_end = run_begin;
        while (interior_end < run_end && d_incidence_runs[d_grouped_runs[interior_end]].interior)
            ++interior_end;
        d_interior_run_ends.push_back(interior_end);
        append(d_cells, base, interior, -1);
    }
    for (const auto& run : d_cells)
        if (run.interior)
            d_interior_cells.push_back(run);
    d_batches = topology.boundary_batch_ids();
    const auto& sides = topology.side_faces();
    for (size_t edge = 0; edge < sides.size(); ++edge)
    {
        const bool interior = sides[edge].neighbor != Meshes::SemiStructMeshTopo::invalid_ordinal;
        append(d_side_runs, edge, interior, interior ? -1 : sides[edge].boundary_id);
        if (!interior)
            d_batches.push_back(sides[edge].boundary_id);
    }
    if (!d_indexer.axial_periodic)
    {
        d_batches.push_back(0);
        d_batches.push_back(1);
    }
    std::sort(d_batches.begin(), d_batches.end());
    d_batches.erase(std::unique(d_batches.begin(), d_batches.end()), d_batches.end());
}

size_t LayeredTraverseLayout::cell_incidence_count(StructuredTraverseSelection selection) const noexcept
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

} // namespace SimpleFluid::detail
