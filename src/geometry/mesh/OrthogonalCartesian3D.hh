/**
 * @file OrthogonalCartesian3D.hh
 * @brief Three-dimensional orthogonal Cartesian finite-volume mesh.
 */

#pragma once

#include "geometry/mesh/MeshBase.hh"
#include "geometry/mesh/OrthoMeshTopo.hh"
#include "OrthogonalIndexer.hh"

#include <array>
#include <cstddef>
#include <compare>

namespace SimpleFluid::Meshes
{

/**
 * @brief Serial orthogonal Cartesian mesh defined by edge coordinates.
 *
 * Face orientation is X, Y, or Z. For an interior face, the owner is the
 * cell on the lower-coordinate side and the face normal points from owner to
 * neighbor. On exterior faces the normal points out of the domain.
 */
class OrthogonalCartesian3D
    : public MeshBase<OrthogonalCartesian3D,
                      OrthogonalIndexer::CellID,
                      OrthogonalIndexer::FaceID,
                      OrthogonalIndexer::NodeID>
{
public:
    using Indexer = OrthogonalIndexer;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using NodeID = Indexer::NodeID;
    using enum Indexer::Dimension;
    using enum Indexer::FaceOrientation;

    using Base = MeshBase<OrthogonalCartesian3D, CellID, FaceID, NodeID>;
    using cell_id_t = typename Base::cell_id_t;
    using face_id_t = typename Base::face_id_t;
    using node_id_t = typename Base::node_id_t;
    using Vec3 = typename Base::Vec3;
    using BoundaryFacePatch = typename Base::BoundaryFacePatch;

    enum Dimension : uint8_t
    {
        X = I,
        Y = J,
        Z = K
    };
    enum FaceOrientation : uint8_t
    {
        X_FACE = I_FACE,
        Y_FACE = J_FACE,
        Z_FACE = K_FACE
    };

    static constexpr cell_id_t invalid_cell_id() noexcept { return {}; }

    explicit OrthogonalCartesian3D(
        const Vec3D<Arr<real_t>>& cell_edges);

    const Vec3D<unsigned>& num_cells_per_dimension() const noexcept
    {
        return d_indexer.num_cells_per_dim;
    }

    const Vec3D<Arr<real_t>>& cell_edges() const noexcept
    {
        return d_cell_edges;
    }

    const Indexer& indexer() const { return d_indexer; }
    const OrthoMeshTopo& topology() const noexcept { return d_topology; }

private:
    friend Base;

    void check_cell_id(cell_id_t cell_id) const;
    void check_face_id(face_id_t face_id) const;
    void check_node_id(node_id_t node_id) const;

    bool is_owned_cell_impl(cell_id_t) const noexcept { return true; }
    bool is_owned_face_impl(face_id_t) const noexcept { return true; }

    real_t cell_volume_impl(cell_id_t cell_id) const;
    Vec3 cell_centroid_impl(cell_id_t cell_id) const;
    std::array<face_id_t, 6> cell_faces_impl(cell_id_t cell_id) const;

    cell_id_t owner_cell_impl(face_id_t face_id) const;
    cell_id_t neighbor_cell_impl(face_id_t face_id) const;

    real_t face_area_impl(face_id_t face_id) const;
    Vec3 face_centroid_impl(face_id_t face_id) const;
    Vec3 face_normal_impl(face_id_t face_id) const;

    Vec3 node_coordinates_impl(node_id_t node_id) const;

    int boundary_id_impl(face_id_t face_id) const;
    const std::string& boundary_patch_name_impl(int patch_id) const;
    auto boundary_face_patch_impl(int patch_id) const;
    auto boundary_patch_ids_impl() const;
    int num_boundary_patches_impl() const noexcept;

    Vec3D<Arr<real_t>> d_cell_edges;
    Vec3D<Arr<real_t>> d_cell_centroids;
    Vec3D<Arr<real_t>> d_cell_widths;

    Indexer d_indexer;
    OrthoMeshTopo d_topology;
};

} // namespace SimpleFluid

#include "geometry/mesh/OrthogonalCartesian3D.ipp"
