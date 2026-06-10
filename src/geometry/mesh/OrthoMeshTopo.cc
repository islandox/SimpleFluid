/**
 * @file OrthoMeshTopo.cc
 * @brief Topology queries for three-dimensional orthogonal meshes.
 */

#include "geometry/mesh/OrthoMeshTopo.hh"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes
{

OrthoMeshTopo::OrthoMeshTopo(
    const Indexer& indexer,
    BoundaryNames boundary_names)
    : d_indexer(indexer),
      d_boundary_names(std::move(boundary_names))
{
    initialize_cell_adjacency();
}

OrthoMeshTopo::OrthoMeshTopo(Ordinal ni, Ordinal nj, Ordinal nk,
    bool periodic_i,
    bool periodic_j,
    bool periodic_k,
    BoundaryNames boundary_names)
    : d_indexer(ni, nj, nk, periodic_i, periodic_j, periodic_k),
      d_boundary_names(std::move(boundary_names))
{
    initialize_cell_adjacency();
}

auto OrthoMeshTopo::owner_cell(FaceID id) const noexcept -> CellID
{
    if (id.orientation == Indexer::I_FACE)
    {
        auto i = id.i == 0 ? 0 : id.i - 1;
        if (id.i == 0 && d_indexer.periodic_dimensions[Indexer::I])
        {
            i = d_indexer.num_cells_per_dim[Indexer::I] - 1;
        }
        return {i, id.j, id.k};
    }
    if (id.orientation == Indexer::J_FACE)
    {
        auto j = id.j == 0 ? 0 : id.j - 1;
        if (id.j == 0 && d_indexer.periodic_dimensions[Indexer::J])
        {
            j = d_indexer.num_cells_per_dim[Indexer::J] - 1;
        }
        return {id.i, j, id.k};
    }

    auto k = id.k == 0 ? 0 : id.k - 1;
    if (id.k == 0 && d_indexer.periodic_dimensions[Indexer::K])
    {
        k = d_indexer.num_cells_per_dim[Indexer::K] - 1;
    }
    return {id.i, id.j, k};
}

auto OrthoMeshTopo::neighbor_cell(FaceID id) const noexcept -> CellID
{
    if (id.orientation == Indexer::I_FACE)
    {
        return d_indexer.periodic_dimensions[Indexer::I]
                || (id.i != 0
                    && id.i != d_indexer.num_cells_per_dim[Indexer::I])
             ? CellID{id.i, id.j, id.k}
             : CellID{};
    }
    if (id.orientation == Indexer::J_FACE)
    {
        return d_indexer.periodic_dimensions[Indexer::J]
                || (id.j != 0
                    && id.j != d_indexer.num_cells_per_dim[Indexer::J])
             ? CellID{id.i, id.j, id.k}
             : CellID{};
    }
    return d_indexer.periodic_dimensions[Indexer::K]
            || (id.k != 0
                && id.k != d_indexer.num_cells_per_dim[Indexer::K])
         ? CellID{id.i, id.j, id.k}
         : CellID{};
}

bool OrthoMeshTopo::is_boundary_face(FaceID face_id) const noexcept
{
    return boundary_id(face_id) != invalid_boundary_id;
}

int OrthoMeshTopo::boundary_id(FaceID id) const noexcept
{
    const auto orientation = id.orientation;
    if (d_indexer.periodic_dimensions[orientation])
    {
        return invalid_boundary_id;
    }

    unsigned coordinate = id.k;
    if (orientation == Indexer::I_FACE)
    {
        coordinate = id.i;
    }
    else if (orientation == Indexer::J_FACE)
    {
        coordinate = id.j;
    }

    if (coordinate == 0)
    {
        return 2 * static_cast<int>(orientation);
    }
    if (coordinate == d_indexer.num_cells_per_dim[orientation])
    {
        return 2 * static_cast<int>(orientation) + 1;
    }
    return invalid_boundary_id;
}

const std::string&
OrthoMeshTopo::boundary_patch_name(int patch_id) const
{
    if (patch_id < 0
        || static_cast<size_t>(patch_id) >= d_boundary_names.size())
    {
        throw std::out_of_range("Requested boundary patch is not found.");
    }
    return d_boundary_names[static_cast<size_t>(patch_id)];
}

void OrthoMeshTopo::validate_boundary_patch(int patch_id) const
{
    if (patch_id < 0 || patch_id >= 6)
    {
        throw std::out_of_range("Requested boundary patch is not found.");
    }
    const auto dim = static_cast<size_t>(patch_id / 2);
    if (d_indexer.periodic_dimensions[dim])
    {
        throw std::out_of_range(
            "Requested boundary patch is in a periodic dimension.");
    }
}

std::vector<int> OrthoMeshTopo::boundary_patch_ids() const
{
    std::vector<int> ids;
    ids.reserve(6);
    for (int dim = 0; dim < 3; ++dim)
    {
        if (!d_indexer.periodic_dimensions[static_cast<size_t>(dim)])
        {
            ids.push_back(2 * dim);
            ids.push_back(2 * dim + 1);
        }
    }
    return ids;
}

int OrthoMeshTopo::num_boundary_patches() const noexcept
{
    int count = 0;
    for (int dim = 0; dim < 3; ++dim)
    {
        if (!d_indexer.periodic_dimensions[static_cast<size_t>(dim)])
        {
            count += 2;
        }
    }
    return count;
}

void OrthoMeshTopo::initialize_cell_adjacency()
{
    d_neighbor_cells.resize(d_indexer.total_cells());

    const auto ni = d_indexer.num_cells_per_dim[Indexer::I];
    const auto nj = d_indexer.num_cells_per_dim[Indexer::J];
    const auto nk = d_indexer.num_cells_per_dim[Indexer::K];

    for (size_t local_id = 0;
         local_id < d_indexer.total_cells();
         ++local_id)
    {
        const auto cell = d_indexer.cell_id(local_id);
        auto& neighbors = d_neighbor_cells[local_id];
        neighbors.reserve(6);

        if (cell.i > 0)
        {
            neighbors.push_back({cell.i - 1, cell.j, cell.k});
        }
        else if (d_indexer.periodic_dimensions[Indexer::I])
        {
            neighbors.push_back({ni - 1, cell.j, cell.k});
        }
        if (cell.i + 1 < ni)
        {
            neighbors.push_back({cell.i + 1, cell.j, cell.k});
        }
        else if (d_indexer.periodic_dimensions[Indexer::I])
        {
            neighbors.push_back({0, cell.j, cell.k});
        }

        if (cell.j > 0)
        {
            neighbors.push_back({cell.i, cell.j - 1, cell.k});
        }
        else if (d_indexer.periodic_dimensions[Indexer::J])
        {
            neighbors.push_back({cell.i, nj - 1, cell.k});
        }
        if (cell.j + 1 < nj)
        {
            neighbors.push_back({cell.i, cell.j + 1, cell.k});
        }
        else if (d_indexer.periodic_dimensions[Indexer::J])
        {
            neighbors.push_back({cell.i, 0, cell.k});
        }

        if (cell.k > 0)
        {
            neighbors.push_back({cell.i, cell.j, cell.k - 1});
        }
        else if (d_indexer.periodic_dimensions[Indexer::K])
        {
            neighbors.push_back({cell.i, cell.j, nk - 1});
        }
        if (cell.k + 1 < nk)
        {
            neighbors.push_back({cell.i, cell.j, cell.k + 1});
        }
        else if (d_indexer.periodic_dimensions[Indexer::K])
        {
            neighbors.push_back({cell.i, cell.j, 0});
        }
    }
}

} // namespace SimpleFluid::Meshes
