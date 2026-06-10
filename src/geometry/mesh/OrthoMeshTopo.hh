/**
 * @file OrthoMeshTopo.hh
 * @brief Topology queries shared by three-dimensional orthogonal meshes.
 */

#pragma once

#include "geometry/mesh/OrthogonalIndexer.hh"
#include "geometry/mesh/StructuredPatchView.hh"

#include <array>
#include <string>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Connectivity and boundary patches for an orthogonal 3D mesh.
 *
 * Boundary patch IDs are assigned in dimension order: lower and upper I,
 * lower and upper J, then lower and upper K. Periodic dimensions have no
 * boundary patches and wrap the face at coordinate zero.
 *
 * Cell and face patches are returned as lazy
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
    using BoundaryNames = std::array<std::string, 6>;
    using NeighborCells = std::vector<CellID>;

    static constexpr int invalid_boundary_id = -1;

    OrthoMeshTopo() = default;
    OrthoMeshTopo(const Indexer& indexer, BoundaryNames boundary_names);
    OrthoMeshTopo(Ordinal ni, Ordinal nj, Ordinal nk,
                  bool periodic_i, bool periodic_j, bool periodic_k,
                  BoundaryNames boundary_names);

    const Indexer& indexer() const noexcept { return d_indexer; }

    CellID owner_cell(FaceID face_id) const noexcept;
    CellID neighbor_cell(FaceID face_id) const noexcept;

    /// Lazy cartesian_product view of CellID over interior cells.
    auto interior_cell_patch() const
    {
        const auto ni = d_indexer.num_cells_per_dim[Indexer::I];
        const auto nj = d_indexer.num_cells_per_dim[Indexer::J];
        const auto nk = d_indexer.num_cells_per_dim[Indexer::K];

        const auto i_beg = d_indexer.periodic_dimensions[Indexer::I]
            ? Ordinal{0} : Ordinal{1};
        const auto i_end = d_indexer.periodic_dimensions[Indexer::I]
            ? ni : (ni > 0 ? ni - 1 : 0);
        const auto j_beg = d_indexer.periodic_dimensions[Indexer::J]
            ? Ordinal{0} : Ordinal{1};
        const auto j_end = d_indexer.periodic_dimensions[Indexer::J]
            ? nj : (nj > 0 ? nj - 1 : 0);
        const auto k_beg = d_indexer.periodic_dimensions[Indexer::K]
            ? Ordinal{0} : Ordinal{1};
        const auto k_end = d_indexer.periodic_dimensions[Indexer::K]
            ? nk : (nk > 0 ? nk - 1 : 0);

        return cartesian_product_3d(i_beg, i_end, j_beg, j_end, k_beg, k_end)
            | std::views::transform([](auto t) {
                  auto [i, j, k] = t;
                  return CellID{i, j, k};
              });
    }

    const NeighborCells& neighbor_cells(CellID cell_id) const noexcept
    {
        return d_neighbor_cells[d_indexer.cell_local_id(cell_id)];
    }

    bool is_boundary_face(FaceID face_id) const noexcept;
    int boundary_id(FaceID face_id) const noexcept;
    const std::string& boundary_patch_name(int patch_id) const;

    /// Lazy cartesian_product view of FaceID for boundary patch @p patch_id.
    auto boundary_face_patch(int patch_id) const
    {
        validate_boundary_patch(patch_id);

        const auto ni = d_indexer.num_cells_per_dim[Indexer::I];
        const auto nj = d_indexer.num_cells_per_dim[Indexer::J];
        const auto nk = d_indexer.num_cells_per_dim[Indexer::K];

        const auto dim = patch_id / 2;
        const auto is_upper = (patch_id % 2) == 1;

        if (dim == 0) // I faces: vary j, k
        {
            const auto i = is_upper ? ni : Ordinal{0};
            return cartesian_product_2d(Ordinal{0}, nj, Ordinal{0}, nk)
                | std::views::transform(
                      FacePatchMapper{i, Indexer::I_FACE});
        }
        if (dim == 1) // J faces: vary i, k
        {
            const auto j = is_upper ? nj : Ordinal{0};
            return cartesian_product_2d(Ordinal{0}, ni, Ordinal{0}, nk)
                | std::views::transform(
                      FacePatchMapper{j, Indexer::J_FACE});
        }
        // K faces: vary i, j
        const auto k = is_upper ? nk : Ordinal{0};
        return cartesian_product_2d(Ordinal{0}, ni, Ordinal{0}, nj)
            | std::views::transform(
                  FacePatchMapper{k, Indexer::K_FACE});
    }

    /// IDs of all available (non-periodic) boundary patches.
    std::vector<int> boundary_patch_ids() const;

    /// Number of available boundary patches.
    int num_boundary_patches() const noexcept;

private:
    /// Functor that maps a 2-D cartesian-product tuple to a FaceID,
    /// with a consistent type across all face orientations.
    struct FacePatchMapper
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

    void initialize_cell_adjacency();
    void validate_boundary_patch(int patch_id) const;

    Indexer d_indexer;
    BoundaryNames d_boundary_names{};
    std::vector<NeighborCells> d_neighbor_cells;
};

} // namespace SimpleFluid::Meshes
