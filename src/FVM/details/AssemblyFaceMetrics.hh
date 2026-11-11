/**
 * @file FVM/details/AssemblyFaceMetrics.hh
 * @brief Backend-neutral face metrics and the borrowed row view for transport assembly.
 *
 * The stored-field transport kernels used to receive the concrete
 * `detail::TraverseAccess<MeshHandle, ConcreteGeometry>` object (or a resolved
 * region face adapter) directly. That made every numerical callback depend on
 * the runtime geometry backend, so each large kernel was instantiated once per
 * `MeshHandle` geometry alternative. This header defines a plain record plus a
 * lightweight borrowed row view with the same duck-typed accessor names, so the
 * numerical callbacks are instantiated once per `(Pack, MeshType)` instead.
 *
 * Boundary contract:
 * - The traversal driver fills one `AssemblyFaceMetrics` per face of the
 *   current cell row from the *same* typed provider accessors (and, for
 *   composite meshes, the same resolved region-face snapshots) that the
 *   kernels used before. No metric formula is re-derived here; stored
 *   distances, periodic-image displacements and coarse/fine expansion keep
 *   their previous meaning.
 * - The records and the row are borrowed, stack-scoped snapshots. They are
 *   valid only for the current row callback, are never retained by field
 *   storage or caches, are thread-affine, and must not outlive the visitor.
 * - `normal`, `area_vector`, `centroid` and the `owner_*` fields are oriented
 *   out of the face owner. The row flips the sign of `normal` and
 *   `area_vector` for the incident neighbor cell, exactly as the typed
 *   accessors did, so numerical accumulation order and values are unchanged.
 * - `owner_center_vector`/`neighbor_center_vector` are the provider's
 *   image-aware cell-to-face-center displacements from each incident cell;
 *   they are not recomputed by centroid subtraction.
 *   `owner_to_neighbor`/`neighbor_to_owner` are the provider's image-aware
 *   cell-to-cell displacements. The distances are the provider's own
 *   `cell_to_face_distance`/`face_cell_center_distance` results, so wrapped
 *   periodic, composite-image and stored legacy distances stay bit-exact.
 */
#pragma once

#include "geometry/MeshUtils.hh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace SimpleFluid::FVM::detail
{

/**
 * @brief Backend-neutral metrics of one assembly face, owner-oriented.
 *
 * This is a passive value snapshot. The traversal driver fills it from the
 * typed geometry accessor (or from a resolved composite face) and the
 * numerical kernels consume it through `AssemblyFaceRow`. Fields that are
 * meaningless for an exterior face stay value-initialized.
 */
struct AssemblyFaceMetrics
{
    using Vec3 = MeshUtils::Vec3;

    /** Bits of `flags`; the traversal driver sets exactly the real category. */
    static constexpr std::uint8_t interior_bit = 1U << 0;
    static constexpr std::uint8_t physical_boundary_bit = 1U << 1;
    static constexpr std::uint8_t owned_orientation_bit = 1U << 2;

    real_t area{};
    /** @brief Owner-oriented unit normal. */
    Vec3 normal{};
    /** @brief Owner-oriented normal times area, in the provider's arithmetic order. */
    Vec3 area_vector{};
    Vec3 centroid{};
    /** @brief Image-aware displacement from the owner cell center to the face center. */
    Vec3 owner_center_vector{};
    /** @brief Image-aware displacement from the neighbor cell center to the face center. */
    Vec3 neighbor_center_vector{};
    Vec3 owner_centroid{};
    Vec3 neighbor_centroid{};
    /** @brief Image-aware displacement from the owner cell center to the neighbor cell center. */
    Vec3 owner_to_neighbor{};
    /** @brief Image-aware displacement from the neighbor cell center to the owner cell center. */
    Vec3 neighbor_to_owner{};
    real_t owner_to_face_distance{};
    real_t neighbor_to_face_distance{};
    real_t face_cell_center_distance{};
    int boundary_id{};
    std::uint8_t flags{};
};

/**
 * @brief Borrowed metrics of the faces of the current assembly row.
 *
 * The row is a lightweight view over one reused metric scratch buffer plus the
 * row's cell ordinals. It deliberately exposes the accessor names that the
 * existing transport helper functions and kernels already use, so those
 * bodies compile unchanged for both the typed provider access and this stable
 * record. It allocates nothing and performs no mesh queries.
 *
 * A face-index argument must name the face currently being visited (set by the
 * traversal driver). A cell argument must be the row's cell or, across an
 * interior face, the opposite cell; the corresponding stored orientation is
 * selected. Sign conventions, periodic-image displacements and stored
 * distances are identical to the previous typed accessors. Any other face or
 * cell is a programming error and throws `std::invalid_argument`.
 *
 * The view is thread-affine and expires with the row callback.
 * @tparam Local The public mesh local ordinal type.
 * @tparam Scalar The public mesh scalar type.
 */
template<class Local, class Scalar> class AssemblyFaceRow
{
public:
    using local_ordinal_type = Local;
    using scalar_type = Scalar;
    using Vec3 = MeshUtils::Vec3;

    AssemblyFaceRow() = default;
    AssemblyFaceRow(std::span<const AssemblyFaceMetrics> metrics, Local cell) noexcept
        : d_metrics(metrics), d_cell(cell)
    {
    }

    /** @brief Bind the next row's metrics and cell; reserved for the traversal driver. */
    void reset(std::span<const AssemblyFaceMetrics> metrics, Local cell) noexcept
    {
        d_metrics = metrics;
        d_cell = cell;
        d_other = invalid_cell();
        d_face_lid = std::numeric_limits<size_t>::max();
        d_slot = std::numeric_limits<size_t>::max();
    }

    /** @brief Select the face being visited; reserved for the traversal driver. */
    void select(size_t slot, Local face_lid, Local other) noexcept
    {
        d_slot = slot;
        d_face_lid = static_cast<size_t>(face_lid);
        d_other = other;
    }

    real_t face_area(size_t face) const { return metric(face).area; }
    Vec3 face_normal(size_t face) const { return metric(face).normal; }
    Vec3 face_area_vector(size_t face) const { return metric(face).area_vector; }
    Vec3 face_centroid(size_t face) const { return metric(face).centroid; }
    int boundary_id(size_t face) const { return metric(face).boundary_id; }

    Vec3 face_normal_outward(size_t face, Local cell) const
    {
        const auto& metrics = metric(face);
        return row_is_owner(metrics, cell) ? metrics.normal : metrics.normal * -1.0;
    }

    Vec3 face_area_vector_outward(size_t face, Local cell) const
    {
        const auto& metrics = metric(face);
        return row_is_owner(metrics, cell) ? metrics.area_vector : metrics.area_vector * -1.0;
    }

    Vec3 cell_centroid(Local cell) const
    {
        const auto& metrics = current();
        return row_is_owner(metrics, cell) ? metrics.owner_centroid : metrics.neighbor_centroid;
    }

    Vec3 face_center_vector(size_t face, Local cell) const
    {
        const auto& metrics = metric(face);
        return row_is_owner(metrics, cell) ? metrics.owner_center_vector : metrics.neighbor_center_vector;
    }

    Vec3 cell_center_vector(size_t face, Local cell) const
    {
        const auto& metrics = metric(face);
        if ((metrics.flags & AssemblyFaceMetrics::interior_bit) == 0)
            throw std::invalid_argument("Exterior face has no adjacent cell.");
        return row_is_owner(metrics, cell) ? metrics.owner_to_neighbor : metrics.neighbor_to_owner;
    }

    real_t cell_to_face_distance(size_t face, Local cell) const
    {
        const auto& metrics = metric(face);
        return row_is_owner(metrics, cell) ? metrics.owner_to_face_distance : metrics.neighbor_to_face_distance;
    }

    real_t face_cell_center_distance(size_t face) const { return metric(face).face_cell_center_distance; }

private:
    static constexpr Local invalid_cell() noexcept { return Local{}; }

    const AssemblyFaceMetrics& current() const
    {
        if (d_slot >= d_metrics.size())
            throw std::logic_error("Assembly face row has no selected face.");
        return d_metrics[d_slot];
    }

    /** @brief Reject stale face ids cheaply instead of reading another face's record. */
    const AssemblyFaceMetrics& metric(size_t face) const
    {
        if (face != d_face_lid)
            throw std::invalid_argument("Assembly face metrics are valid only for the current row face.");
        return current();
    }

    /** @brief True when @p cell sees the owner-oriented record as outward. */
    bool row_is_owner(const AssemblyFaceMetrics& metrics, Local cell) const
    {
        const bool interior = (metrics.flags & AssemblyFaceMetrics::interior_bit) != 0;
        if (!interior)
        {
            // A physical or untagged exterior face has one locally visible
            // incident cell, which is its owner.
            if (cell == d_cell)
                return true;
            throw std::invalid_argument("Cell is not incident to the current assembly row face.");
        }
        if (cell == d_cell)
            return (metrics.flags & AssemblyFaceMetrics::owned_orientation_bit) != 0;
        if (cell == d_other && d_other != d_cell)
            return (metrics.flags & AssemblyFaceMetrics::owned_orientation_bit) == 0;
        throw std::invalid_argument("Cell is not incident to the current assembly row face.");
    }

    std::span<const AssemblyFaceMetrics> d_metrics{};
    Local d_cell{};
    Local d_other{};
    size_t d_face_lid{std::numeric_limits<size_t>::max()};
    size_t d_slot{std::numeric_limits<size_t>::max()};
};
} // namespace SimpleFluid::FVM::detail
