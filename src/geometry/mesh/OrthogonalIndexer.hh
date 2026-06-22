/**
 * @file OrthogonalIndexer.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Indexer for an orthogonal mesh providing dimension sizes and face offsets.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "geometry/mesh/MeshIndexTypes.hh"

namespace SimpleFluid::Meshes
{

/**
 *  @brief Indexer for an orthogonal mesh providing dimension sizes and face offsets.
 */
struct OrthogonalIndexer
{
    using Ordinal = unsigned;

    Vec3D<bool> periodic_dimensions; /**< Periodicity of each dimension. */
    Vec3D<Ordinal> num_cells_per_dim; /**< Number of cells in each dimension. */
    Vec3D<Ordinal> num_nodes_per_dim; /**< Number of nodes in each dimension. */
    Vec3D<Vec3D<Ordinal>> num_faces_per_dim_per_orientation; /**< Number of faces for each orientation and dimension. */
    Vec3D<size_t> num_faces_per_orientation; /**< Number of faces for each orientation. */
    Vec3D<size_t> face_offsets; /**< Starting local ID for each face orientation. */
    Vec3D<Vec3D<size_t>> face_strides; /**< Strides for each face orientation and dimension. */

    enum Dimension : uint8_t
    {
        I = 0,
        J = 1,
        K = 2
    };

    enum FaceOrientation : uint8_t
    {
        I_FACE = I,
        J_FACE = J,
        K_FACE = K
    };

    struct CellID
    {
        Ordinal i = -1;
        Ordinal j = -1;
        Ordinal k = -1;

        constexpr auto operator<=>(const CellID&) const = default;
    };

    struct FaceID
    {
        Ordinal i = -1;
        Ordinal j = -1;
        Ordinal k = -1;
        uint8_t orientation = -1;

        constexpr auto operator<=>(const FaceID&) const = default;
    };

    struct NodeID
    {
        Ordinal i = -1;
        Ordinal j = -1;
        Ordinal k = -1;

        constexpr auto operator<=>(const NodeID&) const = default;
    };

    using cell_id_t = CellID;
    using face_id_t = FaceID;
    using node_id_t = NodeID;
    using ordinal_t = size_t;

    OrthogonalIndexer() = default;

    /**
     * @brief Construct an OrthogonalIndexer.
     * @param ni Number of cells along the first direction.
     * @param nj Number of cells along the second direction.
     * @param nk Number of cells along the third direction.
     * @param I_periodic Whether the first direction is periodic.
     * @param J_periodic Whether the second direction is periodic.
     * @param K_periodic Whether the third direction is periodic.
     */
    constexpr OrthogonalIndexer(
        const Ordinal ni,
        const Ordinal nj,
        const Ordinal nk,
        bool I_periodic = false,
        bool J_periodic = false,
        bool K_periodic = false)
        : periodic_dimensions{I_periodic, J_periodic, K_periodic},
          num_cells_per_dim{ni, nj, nk},
          num_nodes_per_dim{
              I_periodic ? ni : ni + 1,
              J_periodic ? nj : nj + 1,
              K_periodic ? nk : nk + 1},
          num_faces_per_dim_per_orientation{{
              {{num_nodes_per_dim[I], nj,                   nk}},
              {{ni,                   num_nodes_per_dim[J], nk}},
              {{ni,                   nj,                   num_nodes_per_dim[K]}}}},
          num_faces_per_orientation{
              static_cast<size_t>(num_nodes_per_dim[I]) * nj * nk,
              static_cast<size_t>(ni) * num_nodes_per_dim[J] * nk,
              static_cast<size_t>(ni) * nj * num_nodes_per_dim[K]},
          face_offsets{
              0,
              num_faces_per_orientation[I_FACE],
              num_faces_per_orientation[I_FACE] + num_faces_per_orientation[J_FACE]},
          face_strides{{
              {{1, num_nodes_per_dim[I],
                static_cast<size_t>(num_nodes_per_dim[I]) * nj}},
              {{num_nodes_per_dim[J], 1,
                static_cast<size_t>(ni) * num_nodes_per_dim[J]}},
              {{num_nodes_per_dim[K],
                static_cast<size_t>(ni) * num_nodes_per_dim[K], 1}}}}
    {
    }

    constexpr size_t total_cells() const noexcept
    {
        return static_cast<size_t>(num_cells_per_dim[I])
             * num_cells_per_dim[J]
             * num_cells_per_dim[K];
    }

    constexpr size_t total_faces() const noexcept
    {
        return num_faces_per_orientation[I_FACE]
             + num_faces_per_orientation[J_FACE]
             + num_faces_per_orientation[K_FACE];
    }

    constexpr size_t total_nodes() const noexcept
    {
        return static_cast<size_t>(num_nodes_per_dim[I])
             * num_nodes_per_dim[J]
             * num_nodes_per_dim[K];
    }

    constexpr size_t face_ordinal(const FaceID& face_id) const noexcept
    {
        const auto i = face_id.i;
        const auto j = face_id.j;
        const auto k = face_id.k;
        const auto orientation = face_id.orientation;

        return i * face_strides[orientation][I]
             + j * face_strides[orientation][J]
             + k * face_strides[orientation][K]
             + face_offsets[orientation];
    }

    constexpr size_t node_ordinal(const NodeID& node_id) const noexcept
    {
        const auto i = node_id.i;
        const auto j = node_id.j;
        const auto k = node_id.k;

        return i + num_nodes_per_dim[I] * (j + num_nodes_per_dim[J] * k);
    }

    constexpr size_t cell_ordinal(const CellID& cell_id) const noexcept
    {
        const auto i = cell_id.i;
        const auto j = cell_id.j;
        const auto k = cell_id.k;

        return i + num_cells_per_dim[I] * (j + num_cells_per_dim[J] * k);
    }

    constexpr size_t cell_local_id(const CellID& cell_id) const noexcept
    {
        return cell_ordinal(cell_id);
    }

    constexpr size_t face_local_id(const FaceID& face_id) const noexcept
    {
        return face_ordinal(face_id);
    }

    constexpr size_t node_local_id(const NodeID& node_id) const noexcept
    {
        return node_ordinal(node_id);
    }

    constexpr CellID cell_id(size_t cell_ordinal) const noexcept
    {
        const auto ni = num_cells_per_dim[I];
        const auto nj = num_cells_per_dim[J];

        const auto i = cell_ordinal % ni;
        const auto row = cell_ordinal / ni;
        const auto j = row % nj;
        const auto k = row / nj;
        return {
            static_cast<Ordinal>(i),
            static_cast<Ordinal>(j),
            static_cast<Ordinal>(k)};
    }

    constexpr FaceID face_id(size_t face_ordinal) const noexcept
    {
        if (face_ordinal < num_faces_per_orientation[I_FACE])
        {
            const auto ni = num_nodes_per_dim[I];
            const auto nj = num_cells_per_dim[J];

            const auto i = face_ordinal % ni;
            const auto row = face_ordinal / ni;
            const auto j = row % nj;
            const auto k = row / nj;
            return {
                static_cast<Ordinal>(i),
                static_cast<Ordinal>(j),
                static_cast<Ordinal>(k),
                I_FACE};
        }
        if (face_ordinal < num_faces_per_orientation[I_FACE] + num_faces_per_orientation[J_FACE])
        {
            face_ordinal -= num_faces_per_orientation[I_FACE];
            const auto ni = num_cells_per_dim[I];
            const auto nj = num_nodes_per_dim[J];

            const auto j = face_ordinal % nj;
            const auto row = face_ordinal / nj;
            const auto i = row % ni;
            const auto k = row / ni;
            return {
                static_cast<Ordinal>(i),
                static_cast<Ordinal>(j),
                static_cast<Ordinal>(k),
                J_FACE};
        }
        face_ordinal -= num_faces_per_orientation[I_FACE] + num_faces_per_orientation[J_FACE];
        const auto ni = num_cells_per_dim[I];
        const auto nj = num_cells_per_dim[J];
        const auto nk = num_nodes_per_dim[K];

        const auto k = face_ordinal % nk;
        const auto row = face_ordinal / nk;
        const auto j = row / ni;
        const auto i = row % ni;
        return {
            static_cast<Ordinal>(i),
            static_cast<Ordinal>(j),
            static_cast<Ordinal>(k),
            K_FACE};
    }

    constexpr NodeID node_id(size_t node_ordinal) const noexcept
    {
        const auto ni = num_nodes_per_dim[I];
        const auto nj = num_nodes_per_dim[J];

        const auto i = node_ordinal % ni;
        const auto row = node_ordinal / ni;
        const auto j = row % nj;
        const auto k = row / nj;
        return {
            static_cast<Ordinal>(i),
            static_cast<Ordinal>(j),
            static_cast<Ordinal>(k)};
    }
};

template<class LocalOrdinal = size_t,
         class GlobalOrdinal = uint64_t>
struct OrthogonalMeshIndexTypePack
    : MeshIndexTypes<
          OrthogonalIndexer::CellID,
          OrthogonalIndexer::FaceID,
          OrthogonalIndexer::NodeID,
          LocalOrdinal,
          GlobalOrdinal>
{
    using orthogonal_index_type_pack_tag = void;

    template<class NewLocalOrdinal, class NewGlobalOrdinal>
    using rebind_ordinals = OrthogonalMeshIndexTypePack<
        NewLocalOrdinal, NewGlobalOrdinal>;
};

using OrthogonalMeshIndexTypes = OrthogonalMeshIndexTypePack<>;

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/OrthogonalLocalGlobalIndexer.hh"
