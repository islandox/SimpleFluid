/**
 * @file OrthoMeshTopo.cc
 * @brief Topology queries for three-dimensional orthogonal meshes.
 */

#include "geometry/mesh/OrthoMeshTopo.hh"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Mesh
{

OrthoMeshTopo::OrthoMeshTopo(
    const Indexer& indexer,
    BoundaryNames boundary_names)
    : d_indexer(indexer),
      d_boundary_names(std::move(boundary_names))
{
    initialize_boundary_patches();
}

OrthoMeshTopo::OrthoMeshTopo(Ordinal ni, Ordinal nj, Ordinal nk,
    bool periodic_i,
    bool periodic_j,
    bool periodic_k,
    BoundaryNames boundary_names)
    : d_indexer(ni, nj, nk, periodic_i, periodic_j, periodic_k),
      d_boundary_names(std::move(boundary_names))
{
    initialize_boundary_patches();
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

const OrthoMeshTopo::BoundaryPatch&
OrthoMeshTopo::boundary_face_patch(int patch_id) const
{
    const auto patch = d_boundary_patches.find(patch_id);
    if (patch == d_boundary_patches.end())
    {
        throw std::out_of_range("Requested boundary patch is not found.");
    }
    return patch->second;
}

void OrthoMeshTopo::initialize_boundary_patches()
{
    for (size_t dim = 0; dim < 3; ++dim)
    {
        if (d_indexer.periodic_dimensions[dim])
        {
            continue;
        }

        const auto lower_patch_id = static_cast<int>(2 * dim);
        const auto upper_patch_id = lower_patch_id + 1;
        d_boundary_patches.emplace(
            lower_patch_id,
            BoundaryPatch{lower_patch_id, {}});
        d_boundary_patches.emplace(
            upper_patch_id,
            BoundaryPatch{upper_patch_id, {}});
    }

    const auto ni = d_indexer.num_cells_per_dim[Indexer::I];
    const auto nj = d_indexer.num_cells_per_dim[Indexer::J];
    const auto nk = d_indexer.num_cells_per_dim[Indexer::K];

    if (!d_indexer.periodic_dimensions[Indexer::I])
    {
        for (unsigned k = 0; k < nk; ++k)
        {
            for (unsigned j = 0; j < nj; ++j)
            {
                d_boundary_patches.at(0).face_lids.push_back(
                    {0, j, k, Indexer::I_FACE});
                d_boundary_patches.at(1).face_lids.push_back(
                    {ni, j, k, Indexer::I_FACE});
            }
        }
    }

    if (!d_indexer.periodic_dimensions[Indexer::J])
    {
        for (unsigned k = 0; k < nk; ++k)
        {
            for (unsigned i = 0; i < ni; ++i)
            {
                d_boundary_patches.at(2).face_lids.push_back(
                    {i, 0, k, Indexer::J_FACE});
                d_boundary_patches.at(3).face_lids.push_back(
                    {i, nj, k, Indexer::J_FACE});
            }
        }
    }

    if (!d_indexer.periodic_dimensions[Indexer::K])
    {
        for (unsigned j = 0; j < nj; ++j)
        {
            for (unsigned i = 0; i < ni; ++i)
            {
                d_boundary_patches.at(4).face_lids.push_back(
                    {i, j, 0, Indexer::K_FACE});
                d_boundary_patches.at(5).face_lids.push_back(
                    {i, j, nk, Indexer::K_FACE});
            }
        }
    }
}

} // namespace SimpleFluid::Mesh
