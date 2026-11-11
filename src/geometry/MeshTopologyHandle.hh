/**
 * @file MeshTopologyHandle.hh
 * @brief Borrowed topology view over an owning MeshHandle.
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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
class MeshHandle;

namespace Meshes
{
class DecomposedOrthogonalTopology;
class DecomposedSemiStructuredTopology;
class OrthogonalMeshDecomposition;
} // namespace Meshes

/**
 * @brief Borrowed, validated topology view over an owning MeshHandle.
 *
 * MeshTopologyHandle exposes counts, ownership, entity identities, incidence,
 * boundary batches, decomposition metadata, and geometry-to-local translation
 * for the concrete mesh bound by a MeshHandle. It stores no topology: every
 * query forwards to the owning handle, and no layout state is copied.
 *
 * Lifetime: the view borrows the MeshHandle it was constructed from. The handle
 * must outlive the view and every result borrowed from it. Construct a view on
 * demand with `MeshHandle::topology()`; copies are cheap (one pointer plus the
 * validated binding snapshot). The binding never owns Teuchos maps.
 *
 * Migration contract: MeshHandle keeps its public topology API as thin
 * forwarders to this class, so existing callers are source-compatible.
 *
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class SIMPLEFLUID_LOCAL MeshTopologyHandle
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;
    using BoundaryFaceBatch = typename MeshHandle<Pack>::BoundaryFaceBatch;
    using SemiStructuredDecomposition = typename MeshHandle<Pack>::SemiStructuredDecomposition;
    using CellFaceRange = typename MeshHandle<Pack>::CellFaceRange;

    static constexpr int invalid_boundary_id = -1;

    /** @brief Sentinel for an unavailable local entity ordinal. */
    static constexpr local_ordinal_type invalid_local_id() noexcept
    {
        if constexpr (std::is_signed_v<local_ordinal_type>)
            return static_cast<local_ordinal_type>(-1);
        return std::numeric_limits<local_ordinal_type>::max();
    }

    /**
     * @brief Borrow the topology bound by @p owner.
     * @note The owner must outlive this view. Construction copies only the
     *       validated binding snapshot and is cheap.
     */
    explicit MeshTopologyHandle(const MeshHandle<Pack>& owner) noexcept;

    /** @brief Binding snapshot recorded when this view was constructed. */
    [[nodiscard]] const detail::MeshBinding<Pack>& binding() const;

    /** @brief Local-layout revision; advanced when cell order is permuted. */
    [[nodiscard]] std::uint64_t layout_generation() const noexcept;

    [[nodiscard]] std::size_t num_owned_cells() const noexcept;
    [[nodiscard]] std::size_t num_local_cells() const noexcept;
    [[nodiscard]] std::size_t num_cells() const noexcept;
    [[nodiscard]] std::size_t num_owned_faces() const noexcept;
    [[nodiscard]] std::size_t num_faces() const noexcept;

    [[nodiscard]] bool is_owned_cell(local_ordinal_type cell_lid) const;
    [[nodiscard]] bool is_owned_face(local_ordinal_type face_lid) const;

    /** Algebraic cell ID in the field maps; distinct from canonical geometry identity. */
    [[nodiscard]] global_ordinal_type cell_global_id(local_ordinal_type cell_lid) const;
    /** Partition-independent geometry identifier of a cell. */
    [[nodiscard]] global_ordinal_type cell_geometry_global_id(local_ordinal_type cell_lid) const;
    /** Algebraic face ID in the field maps. */
    [[nodiscard]] global_ordinal_type face_global_id(local_ordinal_type face_lid) const;
    /** Partition-independent source topology identifier of a face. */
    [[nodiscard]] global_ordinal_type face_geometry_global_id(local_ordinal_type face_lid) const;

    [[nodiscard]] local_ordinal_type owner_cell(local_ordinal_type face_lid) const;
    [[nodiscard]] local_ordinal_type neighbor_cell(local_ordinal_type face_lid) const;
    [[nodiscard]] local_ordinal_type opposite_cell(local_ordinal_type face_lid,
                                                   local_ordinal_type cell_lid) const;
    [[nodiscard]] local_ordinal_type opposite_or_periodic_neighbor_cell(local_ordinal_type face_lid,
                                                                        local_ordinal_type cell_lid) const;

    /**
     * @brief Observe visible incident faces with field-local IDs.
     *
     * Delegates to the owning MeshHandle, which retains the established
     * allocation-free incidence walk. The returned range borrows the owning
     * handle, not this view.
     */
    [[nodiscard]] CellFaceRange faces(local_ordinal_type cell_lid) const;

    /**
     * @brief Visit visible incident faces with one concrete-backend dispatch.
     *
     * Delegates to the owning MeshHandle; callback arguments and incidence
     * order are unchanged. The concrete geometry is borrowed for the callback.
     */
    template<class Visitor>
    void visit_cell_faces(local_ordinal_type cell_lid, Visitor&& visitor) const;

    /** @brief True if the underlying geometry has no cell across a face. */
    [[nodiscard]] bool is_geometry_exterior_face(local_ordinal_type face_lid) const;
    [[nodiscard]] bool is_exterior_face(local_ordinal_type face_lid) const;
    [[nodiscard]] bool is_interior_face(local_ordinal_type face_lid) const;
    [[nodiscard]] int boundary_id(local_ordinal_type face_lid) const;
    [[nodiscard]] bool is_boundary_face(local_ordinal_type face_lid) const;

    /** @brief Locally known boundary batch IDs in ascending order. */
    [[nodiscard]] std::vector<int> boundary_batch_ids() const;
    [[nodiscard]] const std::string& boundary_batch_name(int batch_id) const;
    [[nodiscard]] const BoundaryFaceBatch& boundary_face_batch(int batch_id) const;
    [[nodiscard]] const std::unordered_map<int, BoundaryFaceBatch>& boundary_batches() const noexcept;

    [[nodiscard]] Teuchos::RCP<const map_type> owned_cell_map() const;
    [[nodiscard]] Teuchos::RCP<const map_type> overlap_cell_map() const;
    [[nodiscard]] Teuchos::RCP<const map_type> owned_face_map() const;
    [[nodiscard]] Teuchos::RCP<const map_type> overlap_face_map() const;
    [[nodiscard]] Teuchos::RCP<const map_type> boundary_face_map() const;

    /**
     * @brief Look up the field local ordinal for a geometry cell index.
     * @returns The local ordinal, or invalid_local_id() if not locally available.
     */
    [[nodiscard]] local_ordinal_type geometry_to_local_cell(std::size_t geometry_lid) const noexcept;
    /**
     * @brief Look up the field local ordinal for a geometry face index.
     * @returns The local ordinal, or invalid_local_id() if not locally available.
     */
    [[nodiscard]] local_ordinal_type geometry_to_local_face(std::size_t geometry_lid) const noexcept;

    /** @brief Translate a resolved canonical composite cell into the field map. */
    [[nodiscard]] local_ordinal_type region_cell_local_id(std::uint64_t cell) const;
    /** @brief Translate a resolved canonical composite face into the field map. */
    [[nodiscard]] local_ordinal_type region_face_local_id(std::uint64_t face) const;

    /** Count/width-selected orthogonal plan; null for another mesh family. */
    [[nodiscard]] SP<const Meshes::OrthogonalMeshDecomposition> orthogonal_decomposition() const noexcept;
    /** Compact construction-local orthogonal topology; null for serial identity and other families. */
    [[nodiscard]] SP<const Meshes::DecomposedOrthogonalTopology> decomposed_orthogonal_topology() const noexcept;
    /** @brief Selected compact XY-graph/Z-slab factors; throws for another mesh family. */
    [[nodiscard]] const SemiStructuredDecomposition& semi_structured_decomposition() const;
    /** Read-only compact XY/Z topology in construction order; null for other families. */
    [[nodiscard]] SP<const Meshes::DecomposedSemiStructuredTopology>
    decomposed_semi_structured_topology() const noexcept;

    /** @brief True after a factory has changed this handle's local cell order. */
    [[nodiscard]] bool has_reordered_cells() const noexcept;
    /** @brief Whether owned cells retain increasing canonical composite region order. */
    [[nodiscard]] bool supports_region_execution() const noexcept;

private:
    const MeshHandle<Pack>* d_owner = nullptr;
    std::shared_ptr<const detail::MeshBinding<Pack>> d_binding;
};

} // namespace SimpleFluid
