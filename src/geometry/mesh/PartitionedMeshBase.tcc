/**
 * @file PartitionedMeshBase.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method definitions for PartitionedMesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @details PartitionedMeshBase.cc includes this file for the supported
 * explicit instantiations. A translation unit using another geometry mesh or
 * Tpetra type pack may include it to instantiate construction, adjacency, and
 * distributed-map initialization for that combination.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/PartitionedMeshBase.hh"
#include "trilinos_wrapper/Teuchos.hh"
#include "trilinos_wrapper/Tpetra.hh"

namespace SimpleFluid::Meshes
{

/**
 * @brief Couple rank-local geometry with global indexing and Tpetra maps.
 * @tparam GeometryMesh Rank-local CRTP geometry mesh type.
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 * @param mesh Rank-local geometry shared with the wrapper.
 * @param indexer Owned-first local/global entity indexer.
 * @param comm Communicator used to construct distributed maps.
 * @throws std::invalid_argument If the mesh is null or its layout differs
 *         from the indexer.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
PartitionedMesh<GeometryMesh, Pack>::PartitionedMesh(
    SP<const mesh_type> mesh,
    indexer_type indexer,
    Teuchos::RCP<const typename Pack::comm_type> comm)
    : d_mesh(require_mesh(std::move(mesh))),
      d_indexer(std::move(indexer))
{
    validate_layout();
    initialize_boundary_batches();
    initialize_maps(comm ? comm : Tpetra::getDefaultComm());
}

/**
 * @brief Translate a cell's geometry face IDs to local ordinals.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param cell_local_id Local cell ordinal.
 * @return Local ordinals of faces bounding the cell.
 * @throws std::out_of_range If the cell ordinal is invalid.
 * @throws std::overflow_error If a face ordinal cannot be represented.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
std::vector<typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type>
PartitionedMesh<GeometryMesh, Pack>::faces(
    local_ordinal_type cell_local_id) const
{
    std::vector<local_ordinal_type> result;
    for (const auto face : mesh().faces(geometry_cell_id(cell_local_id)))
    {
        result.push_back(checked_local(mesh().face_local_id(face)));
    }
    return result;
}

/**
 * @brief Return the local owner-cell ordinal of a face.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @return Owner-cell ordinal.
 * @throws std::out_of_range If the face ordinal is invalid.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::owner_cell(
    local_ordinal_type face_local_id) const
{
    return adjacent_cell(face_local_id, true);
}

/**
 * @brief Return the local neighbor-cell ordinal of a face.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @return Neighbor ordinal, or @ref invalid_local_id for an exterior face.
 * @throws std::out_of_range If the face ordinal is invalid.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::neighbor_cell(
    local_ordinal_type face_local_id) const
{
    return adjacent_cell(face_local_id, false);
}

/**
 * @brief Return the cell across a face from a supplied adjacent cell.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @param cell_local_id One adjacent cell ordinal.
 * @return The other cell, or @ref invalid_local_id at an exterior boundary.
 * @throws std::invalid_argument If the cell is not adjacent to the face.
 * @throws std::out_of_range If the face ordinal is invalid.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::opposite_cell(
    local_ordinal_type face_local_id,
    local_ordinal_type cell_local_id) const
{
    const auto owner = owner_cell(face_local_id);
    const auto neighbor = neighbor_cell(face_local_id);
    if (cell_local_id == owner)
    {
        return neighbor;
    }
    if (neighbor != invalid_local_id() && cell_local_id == neighbor)
    {
        return owner;
    }
    throw std::invalid_argument(
        "Cell is not adjacent to requested face.");
}

/**
 * @brief Return the opposite cell using the mesh's already-wrapped topology.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @param cell_local_id One adjacent cell ordinal.
 * @return Opposite local cell ordinal.
 * @throws std::invalid_argument If the cell is not adjacent to the face.
 * @throws std::out_of_range If the face ordinal is invalid.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::opposite_or_periodic_neighbor_cell(
    local_ordinal_type face_local_id,
    local_ordinal_type cell_local_id) const
{
    return opposite_cell(face_local_id, cell_local_id);
}

/**
 * @brief Compute a face area vector in the stored owner orientation.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param local_id Local face ordinal.
 * @return Unit normal multiplied by face area.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::face_area_vector(
    local_ordinal_type local_id) const
{
    return face_normal(local_id) * face_area(local_id);
}

/**
 * @brief Orient a face unit normal outward from an adjacent cell.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @param cell_local_id Adjacent cell ordinal.
 * @return Outward unit normal.
 * @throws std::invalid_argument If the cell is not adjacent to the face.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::face_normal_outward(
    local_ordinal_type face_local_id,
    local_ordinal_type cell_local_id) const
{
    if (cell_local_id == owner_cell(face_local_id))
    {
        return face_normal(face_local_id);
    }
    if (cell_local_id == neighbor_cell(face_local_id))
    {
        return face_normal(face_local_id) * -1.0;
    }
    throw std::invalid_argument(
        "Cell is not adjacent to requested face.");
}

/**
 * @brief Compute a face area vector outward from an adjacent cell.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @param cell_local_id Adjacent cell ordinal.
 * @return Outward unit normal multiplied by face area.
 * @throws std::invalid_argument If the cell is not adjacent to the face.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::face_area_vector_outward(
    local_ordinal_type face_local_id,
    local_ordinal_type cell_local_id) const
{
    return face_normal_outward(face_local_id, cell_local_id)
         * face_area(face_local_id);
}

/**
 * @brief Measure the distance between cells adjacent to an interior face.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @return Cell-center distance, or zero for an exterior face.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
real_t PartitionedMesh<GeometryMesh, Pack>::face_cell_center_distance(
    local_ordinal_type face_local_id) const
{
    const auto neighbor = neighbor_cell(face_local_id);
    if (neighbor == invalid_local_id())
    {
        return 0.0;
    }
    return (cell_centroid(neighbor)
          - cell_centroid(owner_cell(face_local_id))).norm();
}

/**
 * @brief Form the vector from a cell center to its opposite neighbor center.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @param cell_local_id Adjacent cell ordinal.
 * @return Vector from @p cell_local_id to the opposite cell.
 * @throws std::invalid_argument If the cell is not adjacent or the face is exterior.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::cell_center_vector(
    local_ordinal_type face_local_id,
    local_ordinal_type cell_local_id) const
{
    const auto other = opposite_cell(face_local_id, cell_local_id);
    if (other == invalid_local_id())
    {
        throw std::invalid_argument(
            "Exterior face does not have an opposite cell.");
    }
    return cell_centroid(other) - cell_centroid(cell_local_id);
}

/**
 * @brief Measure distance from a cell centroid to a face centroid.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @param cell_local_id Local cell ordinal.
 * @return Euclidean centroid-to-face distance.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
real_t PartitionedMesh<GeometryMesh, Pack>::cell_to_face_distance(
    local_ordinal_type face_local_id,
    local_ordinal_type cell_local_id) const
{
    return (face_centroid(face_local_id)
          - cell_centroid(cell_local_id)).norm();
}

/**
 * @brief Require a non-null geometry mesh pointer.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param mesh Geometry pointer to validate.
 * @return The validated pointer.
 * @throws std::invalid_argument If @p mesh is null.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
SP<const GeometryMesh>
PartitionedMesh<GeometryMesh, Pack>::require_mesh(
    SP<const mesh_type> mesh)
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "PartitionedMeshBase requires a non-null mesh.");
    }
    return mesh;
}

/**
 * @brief Verify geometry counts against the local/global indexer layout.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @throws std::invalid_argument If any entity count or owned prefix differs.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
void PartitionedMesh<GeometryMesh, Pack>::validate_layout() const
{
    if (mesh().num_cells() != num_local_cells()
        || mesh().num_owned_cells() != num_owned_cells()
        || mesh().num_faces() != num_local_faces()
        || mesh().num_owned_faces() != num_owned_faces()
        || mesh().num_nodes() != num_local_nodes())
    {
        throw std::invalid_argument(
            "Partitioned mesh geometry does not match its indexer.");
    }
}

/**
 * @brief Translate a geometry-adjacent cell ID to a local ordinal.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param face_local_id Local face ordinal.
 * @param owner Select the owner when true, otherwise the neighbor.
 * @return Adjacent cell ordinal, or @ref invalid_local_id when absent.
 * @throws std::out_of_range If the face ordinal is invalid.
 * @throws std::overflow_error If the cell ordinal cannot be represented.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::adjacent_cell(
    local_ordinal_type face_local_id,
    bool owner) const
{
    const auto cell = owner
        ? mesh().owner_cell(geometry_face_id(face_local_id))
        : mesh().neighbor_cell(geometry_face_id(face_local_id));
    if (cell == mesh_type::invalid_cell_id())
    {
        return invalid_local_id();
    }
    return checked_local(mesh().cell_local_id(cell));
}

/**
 * @brief Create a noncontiguous Tpetra map from local global IDs.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @tparam CommPtr Teuchos communicator pointer type.
 * @param comm Communicator shared by the map.
 * @param globals Global IDs represented on this rank.
 * @return Constructed Tpetra map.
 * @throws std::overflow_error If the local element count cannot be represented.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
template<class CommPtr>
Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::make_map(
    const CommPtr& comm,
    const std::vector<global_ordinal_type>& globals) const
{
    const auto invalid_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    return Teuchos::rcp(new map_type(
        invalid_size,
        globals.data(),
        checked_local(globals.size()),
        global_ordinal_type{},
        comm));
}

/**
 * @brief Translate geometry boundary batches into rank-local face ordinals.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @throws std::overflow_error If a face ordinal cannot be represented.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
void PartitionedMesh<GeometryMesh, Pack>::initialize_boundary_batches()
{
    for (const auto batch_id : mesh().boundary_batch_ids())
    {
        BoundaryFaceBatch batch;
        batch.id = batch_id;
        auto append_faces = [&](const auto& source_faces)
        {
            for (const auto face : source_faces)
            {
                const auto local_id = checked_local(
                    mesh().face_local_id(face));
                if (is_local_face(local_id))
                {
                    batch.face_lids.push_back(local_id);
                }
            }
        };
        const auto& source_batch = mesh().boundary_face_batch(batch_id);
        if constexpr (std::ranges::range<
                          std::remove_cvref_t<decltype(source_batch)>>)
        {
            append_faces(source_batch);
        }
        else
        {
            append_faces(source_batch.face_lids);
        }
        if (!batch.face_lids.empty())
        {
            d_boundary_batches.emplace(batch_id, std::move(batch));
        }
    }
}

/**
 * @brief Construct owned, overlap, global, and boundary Tpetra maps.
 * @tparam GeometryMesh Rank-local geometry mesh type.
 * @tparam Pack Tpetra type pack.
 * @param comm Communicator shared by the maps.
 * @throws std::overflow_error If a local count cannot be represented.
 */
template<MeshClass GeometryMesh, TpetraTypePack Pack>
void PartitionedMesh<GeometryMesh, Pack>::initialize_maps(
    const Teuchos::RCP<const typename Pack::comm_type>& comm)
{
    auto collect_ids = [&](size_t count, const auto& global_id)
    {
        std::vector<global_ordinal_type> ids;
        ids.reserve(count);
        for (size_t local = 0; local < count; ++local)
        {
            ids.push_back(global_id(checked_local(local)));
        }
        return ids;
    };
    const auto owned_cells = collect_ids(
        num_owned_cells(),
        [&](auto local) { return cell_global_id(local); });
    const auto overlap_cells = collect_ids(
        num_local_cells(),
        [&](auto local) { return cell_global_id(local); });
    const auto owned_faces = collect_ids(
        num_owned_faces(),
        [&](auto local) { return face_global_id(local); });
    const auto overlap_faces = collect_ids(
        num_local_faces(),
        [&](auto local) { return face_global_id(local); });
    const auto owned_nodes = collect_ids(
        num_owned_nodes(),
        [&](auto local) { return node_global_id(local); });
    const auto overlap_nodes = collect_ids(
        num_local_nodes(),
        [&](auto local) { return node_global_id(local); });

    d_owned_cell_map = make_map(comm, owned_cells);
    d_overlap_cell_map = make_map(comm, overlap_cells);
    d_owned_face_map = make_map(comm, owned_faces);
    d_overlap_face_map = make_map(comm, overlap_faces);
    d_owned_node_map = make_map(comm, owned_nodes);
    d_overlap_node_map = make_map(comm, overlap_nodes);
    d_global_cell_map = Tpetra::createOneToOne(d_overlap_cell_map);
    d_global_face_map = Tpetra::createOneToOne(d_overlap_face_map);
    d_global_node_map = Tpetra::createOneToOne(d_overlap_node_map);

    std::vector<global_ordinal_type> boundary_faces;
    for (const auto& [batch_id, batch] : d_boundary_batches)
    {
        (void)batch_id;
        for (const auto face_local_id : batch.face_lids)
        {
            if (is_owned_face(face_local_id))
            {
                boundary_faces.push_back(face_global_id(face_local_id));
            }
        }
    }
    d_boundary_face_map = make_map(comm, boundary_faces);
}

} // namespace SimpleFluid::Meshes
