/**
 * @file MeshBinding.hh
 * @brief Validated distribution binding shared by a MeshHandle and its borrowed views.
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

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
class MeshHandle;

namespace detail
{

/**
 * @brief Immutable, validated snapshot of a MeshHandle's distribution binding.
 *
 * A binding records the identity of the concrete geometry, the local/global
 * ownership counts, the local sizes of the owned/overlap Tpetra maps, and the
 * layout generation observed when a borrowed view was constructed. It owns no
 * mesh, map, or Teuchos reference-counted storage: a borrowed view keeps the
 * owning MeshHandle alive and uses the binding only for O(1) self-validation.
 *
 * The binding is an implementation detail of the mesh-handle split. It is
 * explicitly local so the dynamic-symbol audit never exports
 * `SimpleFluid::detail::` symbols.
 *
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<TpetraTypePack Pack>
struct SIMPLEFLUID_LOCAL MeshBinding
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;

    /** Identity of the concrete geometry object; unchanged by topology edits. */
    const void* geometry_identity = nullptr;
    /** Borrowed map identities; the owning MeshHandle retains the maps. */
    const map_type* owned_cell_map = nullptr;
    const map_type* overlap_cell_map = nullptr;
    const map_type* owned_face_map = nullptr;
    const map_type* overlap_face_map = nullptr;
    const map_type* boundary_face_map = nullptr;
    /** Local cell/face ownership counts. */
    std::size_t num_owned_cells = 0;
    std::size_t num_local_cells = 0;
    std::size_t num_owned_faces = 0;
    std::size_t num_local_faces = 0;
    /** Tpetra map local element counts; must agree with the counts above. */
    std::size_t owned_cell_map_size = 0;
    std::size_t overlap_cell_map_size = 0;
    std::size_t owned_face_map_size = 0;
    std::size_t overlap_face_map_size = 0;
    std::size_t boundary_face_map_size = 0;
    /** Local-layout revision; advanced when MeshReorderingFactory permutes cells. */
    std::uint64_t layout_generation = 0;
    bool cells_reordered = false;
    bool serial_identity = false;

    /**
     * @brief Reject an internally inconsistent binding.
     *
     * @throws std::logic_error If the geometry identity is missing, an owned
     *         count exceeds its local count, a map is missing, or a map size
     *         disagrees with the recorded entity count.
     */
    void validate() const;

    /**
     * @brief Validate this snapshot and check the observed layout generation.
     *
     * @param expected_layout_generation Generation currently observed on the
     *        owning handle. A mismatch means the view predates an in-place
     *        layout edit and must be rebuilt.
     * @throws std::logic_error If self-validation fails or the generation is stale.
     */
    void validate(std::uint64_t expected_layout_generation) const;
};

template<TpetraTypePack Pack>
inline void MeshBinding<Pack>::validate() const
{
    if (geometry_identity == nullptr)
        throw std::logic_error("MeshBinding has no concrete geometry identity.");
    if (num_owned_cells > num_local_cells)
        throw std::logic_error("MeshBinding owned cell count " + std::to_string(num_owned_cells) +
                               " exceeds local cell count " + std::to_string(num_local_cells) + ".");
    if (num_owned_faces > num_local_faces)
        throw std::logic_error("MeshBinding owned face count " + std::to_string(num_owned_faces) +
                               " exceeds local face count " + std::to_string(num_local_faces) + ".");
    if (owned_cell_map == nullptr || overlap_cell_map == nullptr || owned_face_map == nullptr ||
        overlap_face_map == nullptr || boundary_face_map == nullptr)
        throw std::logic_error("MeshBinding is missing an owned, overlap, or boundary Tpetra map.");
    if (owned_cell_map_size != num_owned_cells)
        throw std::logic_error("MeshBinding owned cell map size " + std::to_string(owned_cell_map_size) +
                               " disagrees with owned cell count " + std::to_string(num_owned_cells) + ".");
    if (overlap_cell_map_size != num_local_cells)
        throw std::logic_error("MeshBinding overlap cell map size " + std::to_string(overlap_cell_map_size) +
                               " disagrees with local cell count " + std::to_string(num_local_cells) + ".");
    if (owned_face_map_size != num_owned_faces)
        throw std::logic_error("MeshBinding owned face map size " + std::to_string(owned_face_map_size) +
                               " disagrees with owned face count " + std::to_string(num_owned_faces) + ".");
    if (overlap_face_map_size != num_local_faces)
        throw std::logic_error("MeshBinding overlap face map size " + std::to_string(overlap_face_map_size) +
                               " disagrees with local face count " + std::to_string(num_local_faces) + ".");
    if (boundary_face_map_size > num_owned_faces)
        throw std::logic_error("MeshBinding boundary face map size " + std::to_string(boundary_face_map_size) +
                               " exceeds owned face count " + std::to_string(num_owned_faces) + ".");
}

template<TpetraTypePack Pack>
inline void MeshBinding<Pack>::validate(std::uint64_t expected_layout_generation) const
{
    validate();
    if (layout_generation != expected_layout_generation)
        throw std::logic_error("MeshBinding layout generation " + std::to_string(layout_generation) +
                               " is stale; the handle is at generation " +
                               std::to_string(expected_layout_generation) + ".");
}

} // namespace detail

} // namespace SimpleFluid
