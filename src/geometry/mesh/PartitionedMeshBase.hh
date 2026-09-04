/**
 * @file PartitionedMeshBase.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Static distributed wrapper for CRTP geometry meshes.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/TpetraTypes.hh"
#include "geometry/mesh/MeshBase.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>

#include <concepts>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief CRTP base combining rank-local geometry and global entity indexing.
 *
 * The held geometry mesh uses compact rank-local entity IDs. The indexer
 * preserves IDs from the original global mesh and records owned prefixes.
 *
 * @par Template instantiation
 * SimpleFluid provides compiled definitions for the geometry/type-pack pairs
 * instantiated in `PartitionedMeshBase.cc`. Code using another @p GeometryMesh
 * or @p Pack must make the definitions in
 * `geometry/mesh/PartitionedMeshBase.tcc` visible where they are instantiated,
 * or explicitly instantiate the specialization from that file in one
 * translation unit.
 *
 * @tparam GeometryMesh Rank-local CRTP geometry mesh type.
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<MeshClass GeometryMesh,
         TpetraTypePack Pack = DefaultTpetraTypes>
class PartitionedMesh
{
public:
    using partitioned_mesh_base_tag = void;
    using mesh_type = GeometryMesh;
    using tpetra_type_pack = Pack;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;
    using cell_id_t = typename mesh_type::cell_id_t;
    using face_id_t = typename mesh_type::face_id_t;
    using node_id_t = typename mesh_type::node_id_t;
    using indexer_type = typename mesh_type::template local_global_indexer_t<
        local_ordinal_type,
        global_ordinal_type>;
    using Vec3 = typename mesh_type::Vec3;

    /** @brief Rank-local faces associated with one boundary identifier. */
    struct BoundaryFaceBatch
    {
        int id = -1;
        std::vector<local_ordinal_type> face_lids;
    };

    static constexpr int invalid_boundary_id = -1;

    /**
     * @brief Couple rank-local geometry with global indexing and maps.
     * @throws std::invalid_argument If @p mesh is null or its layout differs
     *         from @p indexer.
     */
    PartitionedMesh(
        SP<const mesh_type> mesh,
        indexer_type indexer,
        Teuchos::RCP<const typename Pack::comm_type> comm =
            Tpetra::getDefaultComm());

    const mesh_type& mesh() const noexcept;
    SP<const mesh_type> mesh_ptr() const noexcept;
    const indexer_type& indexer() const noexcept;

    size_t num_global_cells() const noexcept;
    size_t num_owned_cells() const noexcept;
    size_t num_local_cells() const noexcept;
    size_t num_cells() const noexcept;

    size_t num_global_faces() const noexcept;
    size_t num_owned_faces() const noexcept;
    size_t num_local_faces() const noexcept;
    size_t num_faces() const noexcept;

    size_t num_global_nodes() const noexcept;
    size_t num_owned_nodes() const noexcept;
    size_t num_local_nodes() const noexcept;
    size_t num_nodes() const noexcept;

    global_ordinal_type cell_global_id(local_ordinal_type local_id) const;
    global_ordinal_type face_global_id(local_ordinal_type local_id) const;
    global_ordinal_type node_global_id(local_ordinal_type local_id) const;

    local_ordinal_type cell_local_id(const cell_id_t& global_id) const noexcept;
    local_ordinal_type face_local_id(const face_id_t& global_id) const noexcept;
    local_ordinal_type node_local_id(const node_id_t& global_id) const noexcept;

    bool is_local_cell(local_ordinal_type local_id) const noexcept;
    bool is_local_face(local_ordinal_type local_id) const noexcept;
    bool is_local_node(local_ordinal_type local_id) const noexcept;

    bool is_owned_cell(local_ordinal_type local_id) const;
    bool is_owned_face(local_ordinal_type local_id) const;
    bool is_owned_node(local_ordinal_type local_id) const;

    bool is_local_cell_id(const cell_id_t& global_id) const noexcept;
    bool is_local_face_id(const face_id_t& global_id) const noexcept;
    bool is_local_node_id(const node_id_t& global_id) const noexcept;

    bool is_owned_global_cell(global_ordinal_type global_id) const;
    bool is_owned_global_face(global_ordinal_type global_id) const;
    bool is_owned_global_node(global_ordinal_type global_id) const;

    static constexpr local_ordinal_type invalid_local_id() noexcept;

    real_t cell_volume(local_ordinal_type local_id) const;
    Vec3 cell_centroid(local_ordinal_type local_id) const;
    std::vector<local_ordinal_type> faces(
        local_ordinal_type cell_local_id) const;

    local_ordinal_type owner_cell(local_ordinal_type face_local_id) const;
    local_ordinal_type neighbor_cell(local_ordinal_type face_local_id) const;
    local_ordinal_type opposite_cell(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const;
    local_ordinal_type opposite_or_periodic_neighbor_cell(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const;

    real_t face_area(local_ordinal_type local_id) const;
    Vec3 face_centroid(local_ordinal_type local_id) const;
    Vec3 face_normal(local_ordinal_type local_id) const;
    Vec3 face_area_vector(local_ordinal_type local_id) const;
    Vec3 face_normal_outward(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const;
    Vec3 face_area_vector_outward(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const;
    real_t face_cell_center_distance(local_ordinal_type face_local_id) const;
    Vec3 cell_center_vector(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const;
    real_t cell_to_face_distance(
        local_ordinal_type face_local_id,
        local_ordinal_type cell_local_id) const;

    bool is_exterior_face(local_ordinal_type face_local_id) const;
    bool is_interior_face(local_ordinal_type face_local_id) const;
    bool is_boundary_face(local_ordinal_type face_local_id) const;
    int boundary_id(local_ordinal_type face_local_id) const;
    const std::string& boundary_batch_name(int batch_id) const;
    /**
     * @throws std::out_of_range If @p batch_id is not present locally.
     */
    const BoundaryFaceBatch& boundary_face_batch(int batch_id) const;
    const std::unordered_map<int, BoundaryFaceBatch>&
    boundary_batches() const noexcept;

    Vec3 node_coordinates(local_ordinal_type node_local_id) const;
    Vec3 node_coord(local_ordinal_type node_local_id) const;

    Teuchos::RCP<const map_type> owned_cell_map() const;
    Teuchos::RCP<const map_type> overlap_cell_map() const;
    Teuchos::RCP<const map_type> owned_face_map() const;
    Teuchos::RCP<const map_type> overlap_face_map() const;
    Teuchos::RCP<const map_type> boundary_face_map() const;
    Teuchos::RCP<const map_type> owned_node_map() const;
    Teuchos::RCP<const map_type> overlap_node_map() const;

private:
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh);
    void validate_layout() const;

    static bool valid_local(local_ordinal_type local_id, size_t count);
    static local_ordinal_type checked_local(size_t local_id);

    cell_id_t geometry_cell_id(local_ordinal_type local_id) const;
    face_id_t geometry_face_id(local_ordinal_type local_id) const;
    node_id_t geometry_node_id(local_ordinal_type local_id) const;

    local_ordinal_type adjacent_cell(
        local_ordinal_type face_local_id,
        bool owner) const;

    template<class CommPtr>
    Teuchos::RCP<const map_type> make_map(
        const CommPtr& comm,
        const std::vector<global_ordinal_type>& globals) const;

    void initialize_boundary_batches();
    void initialize_maps(
        const Teuchos::RCP<const typename Pack::comm_type>& comm);

    SP<const mesh_type> d_mesh;
    indexer_type d_indexer;
    std::unordered_map<int, BoundaryFaceBatch> d_boundary_batches;
    Teuchos::RCP<const map_type> d_owned_cell_map;
    Teuchos::RCP<const map_type> d_overlap_cell_map;
    Teuchos::RCP<const map_type> d_global_cell_map;
    Teuchos::RCP<const map_type> d_owned_face_map;
    Teuchos::RCP<const map_type> d_overlap_face_map;
    Teuchos::RCP<const map_type> d_global_face_map;
    Teuchos::RCP<const map_type> d_boundary_face_map;
    Teuchos::RCP<const map_type> d_owned_node_map;
    Teuchos::RCP<const map_type> d_overlap_node_map;
    Teuchos::RCP<const map_type> d_global_node_map;
};


/**
 * @brief Identify distributed mesh wrappers built on PartitionedMesh.
 * @tparam T Candidate distributed mesh type.
 */
template<class T>
concept PartitionedMeshClass = requires {
    typename T::partitioned_mesh_base_tag;
    typename T::mesh_type;
    typename T::indexer_type;
};

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/PartitionedMeshBase.ipp"
