/**
 * @file OrthogonalCylindrial3D.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Three-dimensional orthogonal cylindrical finite-volume mesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/mesh/MeshBase.hh"
#include "geometry/mesh/OrthoMeshTopo.hh"
#include "OrthogonalIndexer.hh"

#include <array>
#include <cmath>
#include <compare>
#include <cstddef>

namespace SimpleFluid::Meshes
{

/**
 * @brief Serial mesh defined by radial, azimuthal, and axial edge arrays.
 *
 * The radial coordinate must start above zero so every cell keeps a
 * non-degenerate six-face topology. Angular spans smaller than 2 pi form an
 * open sector. A 2 pi span is connected periodically in the azimuthal
 * direction and omits the two angular boundary batches.
 */
class OrthogonalCylindrial3D
    : public MeshBase<OrthogonalCylindrial3D, OrthogonalMeshIndexTypes>
{
public:
    using Indexer = OrthogonalIndexer;
    using enum Indexer::Dimension;
    using enum Indexer::FaceOrientation;

    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using NodeID = Indexer::NodeID;
    using Base = MeshBase<OrthogonalCylindrial3D, OrthogonalMeshIndexTypes>;
    using cell_id_t = typename Base::cell_id_t;
    using face_id_t = typename Base::face_id_t;
    using node_id_t = typename Base::node_id_t;
    using Vec3 = typename Base::Vec3;
    using BoundaryFaceBatch = typename Base::BoundaryFaceBatch;

    /** @brief Cylindrical coordinate aliases for the generic I/J/K axes. */
    enum Coordinate : int
    {
        R = I,
        THETA = J,
        AXIAL = K
    };

    /** @brief Cylindrical aliases for coordinate-normal face orientations. */
    enum FaceOrientation : int
    {
        R_FACE = I_FACE,
        THETA_FACE = J_FACE,
        Z_FACE = K_FACE
    };

    static constexpr cell_id_t invalid_cell_id() noexcept { return {}; }

    /**
     * @brief Construct a cylindrical mesh from radial, angular, and axial edges.
     * @param cell_edges Strictly increasing R, theta, and Z edge coordinates.
     * @throws std::invalid_argument If coordinates are invalid, the inner
     *         radius is not positive, or the angular span is unsupported.
     * @throws std::overflow_error If entity counts exceed supported ID ranges.
     */
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
    const std::string& boundary_batch_name_impl(int batch_id) const;
    auto boundary_face_batch_impl(int batch_id) const;
    auto boundary_batch_ids_impl() const;
    int num_boundary_batches_impl() const noexcept;

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
