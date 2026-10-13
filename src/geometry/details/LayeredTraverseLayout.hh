/** @file LayeredTraverseLayout.hh
 * @brief Immutable XY metadata with arithmetic Z repetitions.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/details/StructuredTraverseLayout.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes
{
class SemiStructuredXY_Z;
}
namespace SimpleFluid::detail
{
template<class Geometry> inline constexpr bool has_layered_traverse_layout = false;
template<> inline constexpr bool has_layered_traverse_layout<Meshes::SemiStructuredXY_Z> = true;

class SIMPLEFLUID_PUBLIC_TYPE LayeredTraverseLayout
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
    explicit LayeredTraverseLayout(const Meshes::SemiStructMeshTopo& topology);
    const Indexer& indexer() const noexcept { return d_indexer; }
    size_t num_cells() const noexcept { return d_indexer.total_cells(); }
    size_t num_faces() const noexcept { return d_indexer.total_faces(); }
    size_t storage_bytes() const noexcept
    {
        return sizeof(*this) +
               (d_offsets.capacity() + d_run_offsets.capacity() + d_grouped_runs.capacity() +
                   d_interior_run_ends.capacity()) *
                   sizeof(size_t) +
               d_incidences.capacity() * sizeof(SideIncidence) +
               (d_cells.capacity() + d_side_runs.capacity() + d_incidence_runs.capacity() +
                   d_interior_cells.capacity()) *
                   sizeof(StructuredTraverseRun) +
               d_batches.capacity() * sizeof(int);
    }
    class Incidences
    {
    public:
        Incidences(const LayeredTraverseLayout& layout, size_t cell);
        size_t size() const noexcept { return d_size; }
        /** Derive only the canonical ID for retained cache metric refill. */
        size_t face_ordinal(size_t slot) const
        {
            if (slot < 2)
            {
                const auto k = slot == 0                            ? d_cell.k
                               : d_layout->d_indexer.axial_periodic ? (d_cell.k + 1) % d_layout->d_indexer.num_layers
                                                                    : d_cell.k + 1;
                return d_axial_start + k;
            }
            return d_side_start + d_layout->d_incidences[d_side_begin + slot - 2].edge;
        }
        StructuredTraverseIncidence operator[](size_t slot) const;
        ResolvedIncidence resolved(size_t slot, const Meshes::SemiStructMeshTopo& topology) const;
        bool all_interior() const noexcept
        {
            return d_layout->d_interior_run_ends[d_cell.ij] == d_layout->d_run_offsets[d_cell.ij + 1] &&
                   (d_layout->d_indexer.axial_periodic ||
                       (d_cell.k > 0 && d_cell.k + 1 < d_layout->d_indexer.num_layers));
        }
        /** Visit preclassified XY runs with the two arithmetic axial entries. */
        template<class Visitor>
        void visit_runs(
            StructuredTraverseSelection selection, bool independent, std::optional<int> batch, Visitor&& visitor) const;
        struct Iterator
        {
            const Incidences* view;
            size_t slot;
            auto operator*() const { return (*view)[slot]; }
            Iterator& operator++();
            bool operator!=(const Iterator& other) const { return slot != other.slot; }
        };
        Iterator begin() const { return {this, 0}; }
        Iterator end() const { return {this, size()}; }

    private:
        const LayeredTraverseLayout* d_layout;
        Indexer::CellID d_cell;
        size_t d_side_begin, d_size, d_axial_start, d_side_start, d_cell_start;
    };
    Incidences cell_faces(size_t cell) const { return {*this, cell}; }
    template<class Visitor>
    void visit_cells(StructuredTraverseSelection selection, bool independent, Visitor&& visit) const;
    template<class Visitor>
    void visit_faces(
        StructuredTraverseSelection selection, bool independent, std::optional<int> batch, Visitor&& visit) const;
    size_t cell_count(StructuredTraverseSelection selection) const noexcept;
    /** Number of all face incidences belonging to the selected cells. */
    size_t cell_incidence_count(StructuredTraverseSelection selection) const noexcept;
    size_t face_count(StructuredTraverseSelection selection, std::optional<int> batch = {}) const noexcept;
    std::span<const StructuredTraverseRun> xy_interior_cell_runs() const noexcept { return d_interior_cells; }
    const std::vector<int>& boundary_ids() const noexcept { return d_batches; }

private:
    static bool selected(StructuredTraverseSelection selection, bool interior)
    {
        return selection == StructuredTraverseSelection::All ||
               (selection == StructuredTraverseSelection::Interior) == interior;
    }
    static void append(std::vector<StructuredTraverseRun>& runs, size_t ordinal, bool interior, int batch);
    Indexer d_indexer;
    std::vector<size_t> d_offsets, d_run_offsets, d_grouped_runs, d_interior_run_ends;
    std::vector<SideIncidence> d_incidences;
    std::vector<StructuredTraverseRun> d_cells, d_side_runs, d_incidence_runs, d_interior_cells;
    std::vector<int> d_batches;
};
} // namespace SimpleFluid::detail

#include "geometry/details/LayeredTraverseLayout.ipp"
