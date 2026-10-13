/** @file StructuredTraverseLayout.ipp
 * @brief Inline accessors and numerical loops for StructuredTraverseLayout.
 */
#pragma once

#include "geometry/details/StructuredTraverseLayout.hh"

#include <functional>

namespace SimpleFluid::detail
{

inline StructuredTraverseLayout::StructuredTraverseLayout(const Meshes::OrthogonalIndexer& indexer) noexcept
    : d_cells{indexer.num_cells_per_dim[0], indexer.num_cells_per_dim[1], indexer.num_cells_per_dim[2]},
      d_periodic{indexer.periodic_dimensions[0], indexer.periodic_dimensions[1], indexer.periodic_dimensions[2]}
{
}

inline size_t StructuredTraverseLayout::face_count(
    StructuredTraverseSelection selection, std::optional<int> batch) const noexcept
{
    if (selection != StructuredTraverseSelection::Interior && batch)
    {
        if (*batch < 0 || *batch >= 6 || d_periodic[static_cast<size_t>(*batch / 2)])
            return 0;
        return face_rows(static_cast<size_t>(*batch / 2));
    }
    if (selection == StructuredTraverseSelection::All)
        return num_faces();
    size_t boundary = 0;
    for (size_t axis = 0; axis < 3; ++axis)
        if (!d_periodic[axis])
            boundary += 2 * face_rows(axis);
    return selection == StructuredTraverseSelection::Boundary ? boundary : num_faces() - boundary;
}

inline std::array<StructuredTraverseIncidence, 6> StructuredTraverseLayout::cell_faces(size_t cell) const noexcept
{
    const std::array<size_t, 3> coordinates{
        cell % d_cells[0], (cell / d_cells[0]) % d_cells[1], cell / (d_cells[0] * d_cells[1])};
    std::array<StructuredTraverseIncidence, 6> result;
    size_t offset = 0;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const auto width = d_cells[axis] + !d_periodic[axis];
        const auto row = axis == 0   ? coordinates[1] + d_cells[1] * coordinates[2]
                         : axis == 1 ? coordinates[0] + d_cells[0] * coordinates[2]
                                     : coordinates[0] + d_cells[0] * coordinates[1];
        const auto base = offset + row * width;
        const auto lower = coordinates[axis];
        const auto upper = d_periodic[axis] && lower + 1 == d_cells[axis] ? 0 : lower + 1;
        const bool lower_boundary = !d_periodic[axis] && lower == 0;
        const bool upper_boundary = !d_periodic[axis] && upper == d_cells[axis];
        // At a periodic seam the owner is the last coordinate. For a
        // one-cell periodic axis both incidences therefore have sign +1.
        const auto lower_owner = lower == 0 ? (d_periodic[axis] ? d_cells[axis] - 1 : 0) : lower - 1;
        result[2 * axis] = {base + lower, lower_owner == lower ? 1 : -1, !lower_boundary,
            lower_boundary ? static_cast<int>(2 * axis) : -1};
        result[2 * axis + 1] = {base + upper, 1, !upper_boundary, upper_boundary ? static_cast<int>(2 * axis + 1) : -1};
        offset += orientation_faces(axis);
    }
    return result;
}

template<class Visitor>
inline void StructuredTraverseLayout::visit_cells(
    StructuredTraverseSelection selection, bool independent, Visitor&& visitor) const
{
    if (independent && selection == StructuredTraverseSelection::All)
    {
        visit_cells(StructuredTraverseSelection::Interior, false, visitor);
        visit_cells(StructuredTraverseSelection::Boundary, false, visitor);
        return;
    }
    const auto emit = [&](size_t begin, size_t end, bool interior)
    {
        if (begin != end && selected(selection, interior))
            std::invoke(visitor, StructuredTraverseRun{begin, end, interior, -1});
    };
    const auto all = num_cells();
    const auto inner = cell_count(StructuredTraverseSelection::Interior);
    if (inner == 0 || inner == all)
    {
        emit(0, all, inner == all);
        return;
    }

    const auto ni = d_cells[0];
    const auto nj = d_cells[1];
    const auto plane = ni * nj;
    const auto i_begin = interior_begin(0);
    const auto i_end = i_begin + interior_cells(0);
    const auto j_begin = interior_begin(1);
    const auto j_end = j_begin + interior_cells(1);
    const auto k_begin = interior_begin(2);
    const auto k_end = k_begin + interior_cells(2);
    emit(0, k_begin * plane, false);
    for (size_t k = k_begin; k < k_end; ++k)
    {
        const auto plane_begin = k * plane;
        emit(plane_begin, plane_begin + j_begin * ni, false);
        for (size_t j = j_begin; j < j_end; ++j)
        {
            const auto row = plane_begin + j * ni;
            emit(row, row + i_begin, false);
            emit(row + i_begin, row + i_end, true);
            emit(row + i_end, row + ni, false);
        }
        emit(plane_begin + j_end * ni, plane_begin + plane, false);
    }
    emit(k_end * plane, all, false);
}

template<class Visitor>
inline void StructuredTraverseLayout::visit_faces(
    StructuredTraverseSelection selection, bool independent, std::optional<int> batch, Visitor&& visitor) const
{
    if (independent && !batch && selection != StructuredTraverseSelection::Interior)
    {
        if (selection == StructuredTraverseSelection::All)
            visit_faces(StructuredTraverseSelection::Interior, false, {}, visitor);
        for (int boundary = 0; boundary < 6; ++boundary)
            visit_faces(StructuredTraverseSelection::Boundary, false, boundary, visitor);
        return;
    }
    if (selection != StructuredTraverseSelection::Interior && batch && (*batch < 0 || *batch >= 6))
        return;
    const auto emit = [&](size_t begin, size_t end, bool interior, int boundary)
    {
        if (begin != end && selected(selection, interior) &&
            (!batch || selection == StructuredTraverseSelection::Interior || *batch == boundary))
            std::invoke(visitor, StructuredTraverseRun{begin, end, interior, boundary});
    };
    size_t offset = 0;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const auto count = orientation_faces(axis);
        if (batch && selection != StructuredTraverseSelection::Interior && static_cast<size_t>(*batch / 2) != axis)
        {
            offset += count;
            continue;
        }
        if (d_periodic[axis])
            emit(offset, offset + count, true, -1);
        else
        {
            const auto width = d_cells[axis] + 1;
            const auto lower_batch = static_cast<int>(2 * axis);
            for (size_t row = 0; row < face_rows(axis); ++row)
            {
                const auto begin = offset + row * width;
                emit(begin, begin + 1, false, lower_batch);
                emit(begin + 1, begin + d_cells[axis], true, -1);
                emit(begin + d_cells[axis], begin + width, false, lower_batch + 1);
            }
        }
        offset += count;
    }
}

} // namespace SimpleFluid::detail
