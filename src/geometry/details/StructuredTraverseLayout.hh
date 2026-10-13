/** @file StructuredTraverseLayout.hh
 * @brief Constant-storage canonical traversal ranges for orthogonal meshes.
 */
#pragma once

#include "geometry/mesh/OrthogonalIndexer.hh"

#include <array>
#include <cstddef>
#include <optional>

namespace SimpleFluid::Meshes
{
class OrthogonalCartesian3D;
class OrthogonalCylindrial3D;
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::detail
{

/** Opt in only providers with the canonical OrthogonalIndexer ordering. */
template<class GeometryMesh> inline constexpr bool has_structured_traverse_layout = false;
template<> inline constexpr bool has_structured_traverse_layout<Meshes::OrthogonalCartesian3D> = true;
template<> inline constexpr bool has_structured_traverse_layout<Meshes::OrthogonalCylindrial3D> = true;

enum class StructuredTraverseSelection
{
    All,
    Interior,
    Boundary
};

struct StructuredTraverseRun
{
    size_t begin{}, end{};
    bool interior = false;
    int batch = -1;
};

struct StructuredTraverseIncidence
{
    size_t face{};
    int orientation = 1;
    bool interior = false;
    int batch = -1;
};

/**
 * @brief Arithmetic selections for a full, canonically ordered orthogonal mesh.
 *
 * This descriptor contains only dimensions and periodicity. It neither retains
 * geometry nor materializes cell, face, or incidence schedules. Cell ordinals
 * have I as their fastest coordinate; face ordinals are grouped by orientation
 * with the normal coordinate fastest, as in OrthogonalIndexer. Physical
 * boundary batches are 2*axis and 2*axis+1; periodic axes have no such batches.
 *
 * Public maps must be identity maps covering the entire provider. Partitioned,
 * reordered, and composite meshes must use their mapped traversal schedules.
 */
class StructuredTraverseLayout
{
public:
    explicit StructuredTraverseLayout(const Meshes::OrthogonalIndexer& indexer) noexcept;

    size_t num_cells() const noexcept { return d_cells[0] * d_cells[1] * d_cells[2]; }

    size_t num_faces() const noexcept { return orientation_faces(0) + orientation_faces(1) + orientation_faces(2); }

    size_t cell_count(StructuredTraverseSelection selection) const noexcept
    {
        const auto all = num_cells();
        if (selection == StructuredTraverseSelection::All)
            return all;
        const auto interior = interior_cells(0) * interior_cells(1) * interior_cells(2);
        return selection == StructuredTraverseSelection::Interior ? interior : all - interior;
    }

    size_t face_count(StructuredTraverseSelection selection, std::optional<int> batch = {}) const noexcept;

    /** Emit nonempty homogeneous ordinal ranges, preserving selected order. */
    template<class Visitor>
    void visit_cells(StructuredTraverseSelection selection, bool independent, Visitor&& visitor) const;

    /** Independent faces group interiors first, then ascending boundary IDs. */
    template<class Visitor>
    void visit_faces(
        StructuredTraverseSelection selection, bool independent, std::optional<int> batch, Visitor&& visitor) const;

    /** Native lower/upper I, J, K incidence, including periodic wraparound. */
    std::array<StructuredTraverseIncidence, 6> cell_faces(size_t cell) const noexcept;

private:
    static bool selected(StructuredTraverseSelection selection, bool interior) noexcept
    {
        return selection == StructuredTraverseSelection::All ||
               (selection == StructuredTraverseSelection::Interior) == interior;
    }

    size_t interior_begin(size_t axis) const noexcept { return d_periodic[axis] ? 0 : 1; }
    size_t interior_cells(size_t axis) const noexcept
    {
        return d_periodic[axis] ? d_cells[axis] : (d_cells[axis] > 2 ? d_cells[axis] - 2 : 0);
    }
    size_t face_rows(size_t axis) const noexcept { return d_cells[(axis + 1) % 3] * d_cells[(axis + 2) % 3]; }
    size_t orientation_faces(size_t axis) const noexcept
    {
        return (d_cells[axis] + !d_periodic[axis]) * face_rows(axis);
    }

    std::array<size_t, 3> d_cells;
    std::array<bool, 3> d_periodic;
};

} // namespace SimpleFluid::detail

#include "geometry/details/StructuredTraverseLayout.ipp"
