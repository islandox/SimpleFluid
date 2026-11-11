/**
 * @file MeshGeometryHandle.hh
 * @brief Borrowed geometry view over an owning MeshHandle.
 *
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "SimpleFluidExport.hh"
#include "dataclass/TpetraTypes.hh"
#include "geometry/MeshBinding.hh"
#include "geometry/MeshUtils.hh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
class MeshHandle;

/**
 * @brief Borrowed, validated geometry view over an owning MeshHandle.
 *
 * MeshGeometryHandle exposes the concrete geometry variant, controlled mutable
 * visitation, the geometry epoch/identity, and every coordinate-derived metric
 * (volume, centroid, area, normal, area vector, and image-aware displacements).
 * It stores no geometry: every query forwards to the owning handle, preserving
 * the established periodic-image arithmetic, composite region resolution, and
 * legacy branches byte-for-byte.
 *
 * Lifetime: the view borrows the MeshHandle it was constructed from. The handle
 * must outlive the view and every result borrowed from it. Construct a view on
 * demand with `MeshHandle::geometry()`; copies are cheap (one pointer plus the
 * validated binding snapshot). The binding never owns Teuchos maps.
 *
 * Geometry-only mutation (ALE trial/rollback) does not replace the view; the
 * live geometry epoch reported by the owning handle remains authoritative.
 *
 * Migration contract: MeshHandle keeps its public geometry API as thin
 * forwarders to this class, so existing callers are source-compatible.
 *
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class SIMPLEFLUID_LOCAL MeshGeometryHandle
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using Vec3 = MeshUtils::Vec3;
    using variant_type = typename MeshHandle<Pack>::variant_type;
    using mutable_variant_type = typename MeshHandle<Pack>::mutable_variant_type;

    /**
     * @brief Borrow the geometry bound by @p owner.
     * @note The owner must outlive this view. Construction copies only the
     *       validated binding snapshot and is cheap.
     */
    explicit MeshGeometryHandle(const MeshHandle<Pack>& owner) noexcept;

    /** @brief Binding snapshot recorded when this view was constructed. */
    [[nodiscard]] const detail::MeshBinding<Pack>& binding() const;

    /** @brief The owning concrete-mesh variant; stable across geometry-only motion. */
    [[nodiscard]] const variant_type& variant() const noexcept;

    /**
     * @brief Invoke a visitor with the concrete mesh object.
     *
     * Pointer alternatives are dereferenced before dispatch.
     */
    template<class Visitor>
    decltype(auto) visit(Visitor&& visitor) const;

    /**
     * @brief Invoke a visitor with mutable access to the concrete geometry.
     *
     * Mutable access is available only when the handle was built from a
     * mutable mesh pointer.
     *
     * @warning This low-level ownership seam does not publish a geometry epoch.
     *          Use a controlled mesh-motion transaction instead.
     *
     * @throws std::logic_error If the handle was built from a const mesh.
     */
    template<class Visitor>
    decltype(auto) visit_mutable(Visitor&& visitor);

    /** @brief True when visit_mutable() can access the concrete geometry. */
    [[nodiscard]] bool has_mutable_geometry() const noexcept;

    /**
     * @brief Pin and validate composite geometry for a bulk operation; native paths are unchanged.
     * @return A Meshes::MultiRegionMesh::ExecutionView by value.
     */
    [[nodiscard]] auto acquire_execution_view() const;

    /**
     * @brief Identity of the concrete geometry shared by this handle.
     *
     * Separately constructed alias handles around the same native mesh report
     * the same identity.
     */
    [[nodiscard]] const void* geometry_identity() const noexcept;

    /**
     * @brief Monotone revision of the fixed-topology geometry.
     *
     * Topology, global IDs, maps, and partitioning are not part of this
     * revision. Geometry-dependent caches must refresh numeric data whenever
     * this value changes.
     */
    [[nodiscard]] std::uint64_t geometry_epoch() const;

    /// @name Cell metrics
    /// @{
    [[nodiscard]] real_t cell_volume(local_ordinal_type cell_lid) const;
    [[nodiscard]] Vec3 cell_centroid(local_ordinal_type cell_lid) const;
    /// @}

    /// @name Face metrics
    /// @{
    [[nodiscard]] real_t face_area(local_ordinal_type face_lid) const;
    [[nodiscard]] Vec3 face_centroid(local_ordinal_type face_lid) const;
    [[nodiscard]] Vec3 face_normal(local_ordinal_type face_lid) const;
    [[nodiscard]] Vec3 face_area_vector(local_ordinal_type face_lid) const;
    [[nodiscard]] Vec3 face_normal_outward(local_ordinal_type face_lid, local_ordinal_type cell_lid) const;
    [[nodiscard]] Vec3 face_area_vector_outward(local_ordinal_type face_lid, local_ordinal_type cell_lid) const;
    [[nodiscard]] real_t face_cell_center_distance(local_ordinal_type face_lid) const;
    [[nodiscard]] Vec3 face_center_vector(local_ordinal_type face_lid, local_ordinal_type cell_lid) const;
    [[nodiscard]] Vec3 cell_center_vector(local_ordinal_type face_lid, local_ordinal_type cell_lid) const;
    [[nodiscard]] real_t cell_to_face_distance(local_ordinal_type face_lid, local_ordinal_type cell_lid) const;
    /// @}

private:
    const MeshHandle<Pack>* d_owner = nullptr;
    std::shared_ptr<const detail::MeshBinding<Pack>> d_binding;
};

} // namespace SimpleFluid
