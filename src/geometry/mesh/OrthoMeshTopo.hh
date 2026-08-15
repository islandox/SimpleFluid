/**
 * @file OrthoMeshTopo.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Topology queries shared by three-dimensional orthogonal meshes.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/mesh/OrthogonalIndexer.hh"
#include "geometry/mesh/StructuredBatchView.hh"

#include <array>
#include <string>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Connectivity and boundary batches for an orthogonal 3D mesh.
 *
 * Boundary batch IDs are assigned in dimension order: lower and upper I,
 * lower and upper J, then lower and upper K. Periodic dimensions have no
 * boundary batches and wrap the face at coordinate zero.
 *
 * Cell and face batches are returned as lazy
 * `std::ranges::views::cartesian_product` views, avoiding materialized
 * storage for the structured topology.
 */
class OrthoMeshTopo
{
public:
    using Indexer = OrthogonalIndexer;
    using Ordinal = Indexer::Ordinal;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using enum Indexer::Dimension;
    using BoundaryNames = std::array<std::string, 6>;

    static constexpr int invalid_boundary_id = -1;
    static constexpr int max_neighbors = 6;
    static constexpr Ordinal invalid_ordinal =
        static_cast<Ordinal>(-1);

    /** @brief Fixed-capacity list of face-adjacent cells. */
    struct NeighborCells
    {
        unsigned num{};
        std::array<CellID, max_neighbors> neighbors{};

        constexpr bool operator==(const NeighborCells&) const = default;
    };

    OrthoMeshTopo() = default;
    OrthoMeshTopo(const Indexer& indexer, BoundaryNames boundary_names);
    OrthoMeshTopo(Ordinal ni, Ordinal nj, Ordinal nk,
                  bool periodic_i, bool periodic_j, bool periodic_k,
                  BoundaryNames boundary_names);

    const Indexer& indexer() const noexcept { return d_indexer; }

    CellID owner_cell(FaceID face_id) const noexcept;
    CellID neighbor_cell(FaceID face_id) const noexcept;

    /// Lazy cartesian_product view of CellID over interior cells.
    auto interior_cell_batch() const
    {
        const auto ni = d_indexer.num_cells_per_dim[I];
        const auto nj = d_indexer.num_cells_per_dim[J];
        const auto nk = d_indexer.num_cells_per_dim[K];

        const auto i_beg = d_indexer.periodic_dimensions[I]
            ? Ordinal{0} : Ordinal{1};
        const auto i_end = d_indexer.periodic_dimensions[I]
            ? ni : (ni > 0 ? ni - 1 : 0);
        const auto j_beg = d_indexer.periodic_dimensions[J]
            ? Ordinal{0} : Ordinal{1};
        const auto j_end = d_indexer.periodic_dimensions[J]
            ? nj : (nj > 0 ? nj - 1 : 0);
        const auto k_beg = d_indexer.periodic_dimensions[K]
            ? Ordinal{0} : Ordinal{1};
        const auto k_end = d_indexer.periodic_dimensions[K]
            ? nk : (nk > 0 ? nk - 1 : 0);

        return cartesian_product_3d(i_beg, i_end, j_beg, j_end, k_beg, k_end)
            | std::views::transform([](auto t) {
                  auto [i, j, k] = t;
                  return CellID{i, j, k};
              });
    }

    NeighborCells neighbor_cells(CellID cell_id) const noexcept
    {
        const auto& i_neighbors = d_neighbors_per_dim[I][cell_id.i];
        const auto& j_neighbors = d_neighbors_per_dim[J][cell_id.j];
        const auto& k_neighbors = d_neighbors_per_dim[K][cell_id.k];

        NeighborCells result;
        result.num =
            i_neighbors.num + j_neighbors.num + k_neighbors.num;

        unsigned offset = 0;
        for (unsigned i = 0; i < i_neighbors.num; ++i)
        {
            result.neighbors[offset++] =
                {i_neighbors.indices[i], cell_id.j, cell_id.k};
        }
        for (unsigned j = 0; j < j_neighbors.num; ++j)
        {
            result.neighbors[offset++] =
                {cell_id.i, j_neighbors.indices[j], cell_id.k};
        }
        for (unsigned k = 0; k < k_neighbors.num; ++k)
        {
            result.neighbors[offset++] =
                {cell_id.i, cell_id.j, k_neighbors.indices[k]};
        }

        return result;
    }

    bool is_boundary_face(FaceID face_id) const noexcept;
    int boundary_id(FaceID face_id) const noexcept;
    const std::string& boundary_batch_name(int batch_id) const;

    /// Lazy cartesian_product view of FaceID for boundary batch @p batch_id.
    auto boundary_face_batch(int batch_id) const
    {
        validate_boundary_batch(batch_id);

        const auto ni = d_indexer.num_cells_per_dim[Indexer::I];
        const auto nj = d_indexer.num_cells_per_dim[Indexer::J];
        const auto nk = d_indexer.num_cells_per_dim[Indexer::K];

        const auto dim = batch_id / 2;
        const auto is_upper = (batch_id % 2) == 1;

        if (dim == 0) // I faces: vary j, k
        {
            const auto i = is_upper ? ni : Ordinal{0};
            return cartesian_product_2d(Ordinal{0}, nj, Ordinal{0}, nk)
                | std::views::transform(
                      FaceBatchMapper{i, Indexer::I_FACE});
        }
        if (dim == 1) // J faces: vary i, k
        {
            const auto j = is_upper ? nj : Ordinal{0};
            return cartesian_product_2d(Ordinal{0}, ni, Ordinal{0}, nk)
                | std::views::transform(
                      FaceBatchMapper{j, Indexer::J_FACE});
        }
        // K faces: vary i, j
        const auto k = is_upper ? nk : Ordinal{0};
        return cartesian_product_2d(Ordinal{0}, ni, Ordinal{0}, nj)
            | std::views::transform(
                  FaceBatchMapper{k, Indexer::K_FACE});
    }

    /// IDs of all available (non-periodic) boundary batches.
    std::vector<int> boundary_batch_ids() const;

    /// Number of available boundary batches.
    int num_boundary_batches() const noexcept;

private:
    /** @brief Owner and neighbor coordinates for one face plane. */
    struct FaceCells
    {
        Ordinal owner = invalid_ordinal;
        Ordinal neighbor = invalid_ordinal;
    };

    /** @brief Neighbor coordinates available along one mesh dimension. */
    struct DimensionNeighbors
    {
        unsigned num{};
        std::array<Ordinal, 2> indices{};
    };

    /// Functor that maps a 2-D cartesian-product tuple to a FaceID,
    /// with a consistent type across all face orientations.
    struct FaceBatchMapper
    {
        Ordinal fixed_coord;
        uint8_t orientation;

        FaceID operator()(std::tuple<Ordinal, Ordinal> t) const noexcept
        {
            auto [a, b] = t;
            if (orientation == Indexer::I_FACE)
            {
                return {fixed_coord, a, b, orientation};
            }
            if (orientation == Indexer::J_FACE)
            {
                return {a, fixed_coord, b, orientation};
            }
            return {a, b, fixed_coord, orientation};
        }
    };

    void initialize_face_adjacency();
    void initialize_cell_adjacency();
    void validate_boundary_batch(int batch_id) const;

    Indexer d_indexer;
    BoundaryNames d_boundary_names{};
    Vec3D<Arr<FaceCells>> d_face_cells_per_dim;
    Vec3D<Arr<DimensionNeighbors>> d_neighbors_per_dim;
};

} // namespace SimpleFluid::Meshes
