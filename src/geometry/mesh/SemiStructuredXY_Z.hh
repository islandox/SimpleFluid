/**
 * @file SemiStructuredXY_Z.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume mesh formed by extruding a polygonal XY mesh in Z.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/GeometryEpoch.hh"
#include "geometry/mesh/MeshBase.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace SimpleFluid::Meshes
{

class PlanarALEGeometryAccess;

/**
 * @brief Serial mesh formed by extruding a polygonal XY topology through Z.
 *
 * XY cells are supplied as counter-clockwise node loops. Shared edges are
 * derived from those loops. The first cell containing an edge is its owner,
 * and the directed edge follows that owner's counter-clockwise loop.
 */
class SemiStructuredXY_Z
    : public MeshBase<SemiStructuredXY_Z, SemiStructuredMeshIndexTypes>
{
public:
    using Indexer = SemiStructuredIndexer;
    using Topology = SemiStructMeshTopo;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using NodeID = Indexer::NodeID;
    using enum Indexer::Dimension;
    using enum Indexer::FaceOrientation;

    using Base = MeshBase<SemiStructuredXY_Z, SemiStructuredMeshIndexTypes>;
    using cell_id_t = typename Base::cell_id_t;
    using face_id_t = typename Base::face_id_t;
    using node_id_t = typename Base::node_id_t;
    using Vec3 = typename Base::Vec3;
    using BoundaryFaceBatch = typename Base::BoundaryFaceBatch;

    /** @brief Public aliases for axial and extruded-side face orientations. */
    enum FaceOrientation : std::uint8_t
    {
        Z_FACE = AXIAL,
        SIDE_FACE = SIDE
    };

    using BoundaryEdge = Topology::BoundaryEdge;

    static constexpr cell_id_t invalid_cell_id() noexcept { return {}; }

    /**
     * @brief Extrude a counter-clockwise polygonal XY mesh through Z layers.
     * @param xy_nodes Base-topology node coordinates in the XY plane.
     * @param xy_cell_nodes Counter-clockwise node loops for base cells.
     * @param z_edges Strictly increasing axial edge coordinates.
     * @param boundary_edges Named exterior edges in the base topology.
     * @throws std::invalid_argument If geometry or connectivity is invalid.
     * @throws std::overflow_error If entity counts exceed supported ID ranges.
     */
    SemiStructuredXY_Z(
        const Arr<Vec3>& xy_nodes,
        const Arr<Arr<unsigned>>& xy_cell_nodes,
        const Arr<real_t>& z_edges,
        const Arr<BoundaryEdge>& boundary_edges = {});

    /**
     * @brief Generate a triangular XY mesh with frontal-Delaunay placement,
     *        then extrude it through the supplied Z edges.
     *
     * @param xy_boundary Convex, counter-clockwise polygon on the XY plane.
     * @param z_edges Strictly increasing axial edge coordinates.
     * @param target_edge_length Requested XY edge length.
     * @param side_batch_name Boundary batch assigned to the polygon sides.
     * @return Complete serial triangular-prism mesh.
     * @throws std::invalid_argument If source geometry or spacing is invalid.
     * @throws std::overflow_error If generated connectivity exceeds its ID type.
     * @throws std::runtime_error If triangulation invariants cannot be satisfied.
     */
    static SemiStructuredXY_Z from_frontal_delaunay(
        const Arr<Vec3>& xy_boundary,
        const Arr<real_t>& z_edges,
        real_t target_edge_length,
        const std::string& side_batch_name = "side");

    const Arr<Vec3>& xy_nodes() const noexcept { return d_xy_nodes; }
    const Arr<Arr<unsigned>>& xy_cell_nodes() const noexcept
    {
        return d_xy_cell_nodes;
    }
    const Arr<real_t>& z_edges() const noexcept { return d_z_edges; }
    const Indexer& indexer() const noexcept { return d_topology.indexer(); }
    const Topology& topology() const noexcept { return d_topology; }

    size_t topology_storage_bytes() const noexcept
    {
        size_t bytes = d_topology.storage_bytes() + d_xy_cell_nodes.capacity() * sizeof(Arr<unsigned>);
        for (const auto& row : d_xy_cell_nodes) bytes += row.capacity() * sizeof(unsigned);
        return bytes;
    }
    size_t geometry_storage_bytes() const noexcept
    {
        return (d_xy_nodes.capacity() + d_xy_cell_centroids.capacity() + d_xy_edge_centroids.capacity()
                   + d_xy_edge_normals.capacity()) * sizeof(Vec3)
            + (d_z_edges.capacity() + d_z_widths.capacity() + d_z_midpoints.capacity()
                   + d_xy_cell_areas.capacity() + d_xy_edge_lengths.capacity()) * sizeof(real_t);
    }

    /** @brief Revision shared by all handles observing this geometry. */
    std::uint64_t geometry_epoch() const noexcept { return d_geometry_state.epoch; }

private:
    friend Base;
    friend class PlanarALEGeometryAccess;

    void replace_axial_edges_fixed_topology(Arr<real_t> edges);

    void check_cell_id(cell_id_t cell_id) const;
    void check_face_id(face_id_t face_id) const;
    void check_node_id(node_id_t node_id) const;

    bool is_owned_cell_impl(cell_id_t) const noexcept { return true; }
    bool is_owned_face_impl(face_id_t) const noexcept { return true; }

    real_t cell_volume_impl(cell_id_t cell_id) const;
    Vec3 cell_centroid_impl(cell_id_t cell_id) const;
    EntityRange<face_id_t> cell_faces_impl(cell_id_t cell_id) const;

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
    auto boundary_batches_impl() const
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
    GeometryEpochState d_geometry_state;
};

using SemiStructuredXYZ3D = SemiStructuredXY_Z;
using SemiStructuredXY_Z3D = SemiStructuredXY_Z;

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/SemiStructuredXY_Z.ipp"
