/**
 * @file SemiStructuredXY_Z.hh
 * @brief Finite-volume mesh formed by extruding a polygonal XY mesh in Z.
 */

#pragma once

#include "geometry/mesh/MeshBase.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Serial mesh formed by extruding a polygonal XY topology through Z.
 *
 * XY cells are supplied as counter-clockwise node loops. Shared edges are
 * derived from those loops. The first cell containing an edge is its owner,
 * and the directed edge follows that owner's counter-clockwise loop.
 */
class SemiStructuredXY_Z
    : public MeshBase<SemiStructuredXY_Z,
                      SemiStructuredIndexer::CellID,
                      SemiStructuredIndexer::FaceID,
                      SemiStructuredIndexer::NodeID>
{
public:
    using Indexer = SemiStructuredIndexer;
    using Topology = SemiStructMeshTopo;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using NodeID = Indexer::NodeID;
    using enum Indexer::Dimension;
    using enum Indexer::FaceOrientation;

    using Base = MeshBase<SemiStructuredXY_Z, CellID, FaceID, NodeID>;
    using cell_id_t = typename Base::cell_id_t;
    using face_id_t = typename Base::face_id_t;
    using node_id_t = typename Base::node_id_t;
    using Vec3 = typename Base::Vec3;
    using BoundaryFaceBatch = typename Base::BoundaryFaceBatch;

    enum FaceOrientation : std::uint8_t
    {
        Z_FACE = AXIAL,
        SIDE_FACE = SIDE
    };

    using BoundaryEdge = Topology::BoundaryEdge;

    static constexpr cell_id_t invalid_cell_id() noexcept { return {}; }

    SemiStructuredXY_Z(
        const Arr<Vec3>& xy_nodes,
        const Arr<Arr<unsigned>>& xy_cell_nodes,
        const Arr<real_t>& z_edges,
        const Arr<BoundaryEdge>& boundary_edges = {});

    const Arr<Vec3>& xy_nodes() const noexcept { return d_xy_nodes; }
    const Arr<Arr<unsigned>>& xy_cell_nodes() const noexcept
    {
        return d_xy_cell_nodes;
    }
    const Arr<real_t>& z_edges() const noexcept { return d_z_edges; }
    const Indexer& indexer() const noexcept { return d_topology.indexer(); }
    const Topology& topology() const noexcept { return d_topology; }

private:
    friend Base;

    void check_cell_id(cell_id_t cell_id) const;
    void check_face_id(face_id_t face_id) const;
    void check_node_id(node_id_t node_id) const;

    bool is_owned_cell_impl(cell_id_t) const noexcept { return true; }
    bool is_owned_face_impl(face_id_t) const noexcept { return true; }

    real_t cell_volume_impl(cell_id_t cell_id) const;
    Vec3 cell_centroid_impl(cell_id_t cell_id) const;
    std::vector<face_id_t> cell_faces_impl(cell_id_t cell_id) const;

    cell_id_t owner_cell_impl(face_id_t face_id) const;
    cell_id_t neighbor_cell_impl(face_id_t face_id) const;

    real_t face_area_impl(face_id_t face_id) const;
    Vec3 face_centroid_impl(face_id_t face_id) const;
    Vec3 face_normal_impl(face_id_t face_id) const;
    Vec3 node_coordinates_impl(node_id_t node_id) const;

    int boundary_id_impl(face_id_t face_id) const;
    const std::string& boundary_batch_name_impl(int batch_id) const;
    auto boundary_face_batch_impl(int batch_id) const;
    auto boundary_batch_ids_impl() const;
    int num_boundary_batches_impl() const noexcept;
    auto boundary_batches_impl() const noexcept
    {
        return d_topology.boundary_batches();
    }

    void initialize_xy_geometry();

    Arr<Vec3> d_xy_nodes;
    Arr<Arr<unsigned>> d_xy_cell_nodes;
    Arr<real_t> d_z_edges;
    Arr<real_t> d_z_widths;
    Arr<real_t> d_z_midpoints;

    Arr<real_t> d_xy_cell_areas;
    Arr<Vec3> d_xy_cell_centroids;
    Arr<real_t> d_xy_edge_lengths;
    Arr<Vec3> d_xy_edge_centroids;
    Arr<Vec3> d_xy_edge_normals;

    Topology d_topology;
};

using SemiStructuredXYZ3D = SemiStructuredXY_Z;
using SemiStructuredXY_Z3D = SemiStructuredXY_Z;

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/SemiStructuredXY_Z.ipp"
