/**
 * @file OrthogonalCylindrial3D.hh
 * @brief Three-dimensional orthogonal cylindrical finite-volume mesh.
 */

#pragma once

#include "geometry/mesh/MeshBase.hh"
#include "geometry/mesh/OrthoMeshTopo.hh"
#include "OrthogonalIndexer.hh"

#include <array>
#include <cmath>
#include <compare>
#include <cstddef>

namespace SimpleFluid::Mesh
{

/**
 * @brief Serial mesh defined by radial, azimuthal, and axial edge arrays.
 *
 * The radial coordinate must start above zero so every cell keeps a
 * non-degenerate six-face topology. Angular spans smaller than 2 pi form an
 * open sector. A 2 pi span is connected periodically in the azimuthal
 * direction and omits the two angular boundary patches.
 */
class OrthogonalCylindrial3D
    : public MeshBase<OrthogonalCylindrial3D,
                      OrthogonalIndexer::CellID,
                      OrthogonalIndexer::FaceID,
                      OrthogonalIndexer::NodeID>
{
public:
    using Indexer = OrthogonalIndexer;
    using enum Indexer::Dimension;
    using enum Indexer::FaceOrientation;

    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using NodeID = Indexer::NodeID;
    using Base = MeshBase<OrthogonalCylindrial3D, CellID, FaceID, NodeID>;
    using cell_id_t = typename Base::cell_id_t;
    using face_id_t = typename Base::face_id_t;
    using node_id_t = typename Base::node_id_t;
    using Vec3 = typename Base::Vec3;
    using BoundaryFacePatch = typename Base::BoundaryFacePatch;

    enum Coordinate : int
    {
        R = I,
        THETA = J,
        AXIAL = K
    };

    enum FaceOrientation : int
    {
        R_FACE = I_FACE,
        THETA_FACE = J_FACE,
        Z_FACE = K_FACE
    };

    static constexpr cell_id_t invalid_cell_id() noexcept { return {}; }

    explicit OrthogonalCylindrial3D(
        const Vec3D<Arr<real_t>>& cell_edges);

    const Vec3D<unsigned>& num_cells_per_dimension() const noexcept
    {
        return d_indexer.num_cells_per_dim;
    }

    const Vec3D<Arr<real_t>>& cell_edges() const noexcept
    {
        return d_cell_edges;
    }

    bool is_theta_periodic() const noexcept
    {
        return d_indexer.periodic_dimensions[THETA];
    }

    const Indexer& indexer() const noexcept { return d_indexer; }
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
    const BoundaryFacePatch& boundary_face_patch_impl(int patch_id) const;
    const std::unordered_map<int, BoundaryFacePatch>&
    boundary_patches_impl() const noexcept
    {
        return d_topology.boundary_patches();
    }

    real_t sector_centroid_radius(size_t radial_cell,
                                  size_t theta_cell) const;
    Vec3 cylindrical_point(real_t radius,
                           real_t theta,
                           real_t z) const;

    Indexer d_indexer;
    OrthoMeshTopo d_topology;

    Vec3D<ArrReal> d_cell_edges;
    Vec3D<ArrReal> d_cell_midpoints;
    Vec3D<ArrReal> d_cell_widths;

    ArrReal d_cell_Dr2;

    Vec3D<Vec3D<ArrReal>> d_face_area_magnitudes;
};

using OrthogonalCylindrical3D = OrthogonalCylindrial3D;

} // namespace SimpleFluid

#include "geometry/mesh/OrthogonalCylindrial3D.ipp"
