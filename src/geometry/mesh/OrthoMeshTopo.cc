/**
 * @file OrthoMeshTopo.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Topology queries for three-dimensional orthogonal meshes.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/OrthoMeshTopo.hh"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes
{

/**
 * @brief Build topology queries around an existing structured indexer.
 * @param indexer Structured dimensions and periodicity.
 * @param boundary_names Names in lower/upper I, J, K order.
 */
OrthoMeshTopo::OrthoMeshTopo(
    const Indexer& indexer,
    BoundaryNames boundary_names)
    : d_indexer(indexer),
      d_boundary_names(std::move(boundary_names))
{
    initialize_face_adjacency();
    initialize_cell_adjacency();
}

/**
 * @brief Build topology queries directly from dimensions and periodicity.
 * @param ni Cell count along I.
 * @param nj Cell count along J.
 * @param nk Cell count along K.
 * @param periodic_i Whether I wraps periodically.
 * @param periodic_j Whether J wraps periodically.
 * @param periodic_k Whether K wraps periodically.
 * @param boundary_names Names in lower/upper I, J, K order.
 */
OrthoMeshTopo::OrthoMeshTopo(Ordinal ni, Ordinal nj, Ordinal nk,
    bool periodic_i,
    bool periodic_j,
    bool periodic_k,
    BoundaryNames boundary_names)
    : d_indexer(ni, nj, nk, periodic_i, periodic_j, periodic_k),
      d_boundary_names(std::move(boundary_names))
{
    initialize_face_adjacency();
    initialize_cell_adjacency();
}

/**
 * @brief Return the owner cell on the lower-coordinate side of a face.
 * @param id Structured face identifier.
 * @return Owner cell identifier.
 */
auto OrthoMeshTopo::owner_cell(FaceID id) const noexcept -> CellID
{
    std::array<Ordinal, 3> coordinates{id.i, id.j, id.k};
    const auto orientation = id.orientation;
    const auto face_coordinate = coordinates[orientation];
    coordinates[orientation] =
        d_face_cells_per_dim[orientation][face_coordinate].owner;
    return {coordinates[Indexer::I],
            coordinates[Indexer::J],
            coordinates[Indexer::K]};
}

/**
 * @brief Return the adjacent neighbor cell of a face.
 * @param id Structured face identifier.
 * @return Neighbor cell, or an invalid ID for an exterior face.
 */
auto OrthoMeshTopo::neighbor_cell(FaceID id) const noexcept -> CellID
{
    std::array<Ordinal, 3> coordinates{id.i, id.j, id.k};
    const auto orientation = id.orientation;
    const auto face_coordinate = coordinates[orientation];
    const auto neighbor =
        d_face_cells_per_dim[orientation][face_coordinate].neighbor;
    if (neighbor == invalid_ordinal)
    {
        return {};
    }
    coordinates[orientation] = neighbor;
    return {coordinates[Indexer::I],
            coordinates[Indexer::J],
            coordinates[Indexer::K]};
}

/**
 * @brief Test whether a face belongs to a non-periodic exterior boundary.
 * @param face_id Structured face identifier.
 * @return True for physical boundary faces.
 */
bool OrthoMeshTopo::is_boundary_face(FaceID face_id) const noexcept
{
    return boundary_id(face_id) != invalid_boundary_id;
}

/**
 * @brief Map an exterior face to its lower or upper boundary batch.
 * @param id Structured face identifier.
 * @return Boundary batch ID, or @ref invalid_boundary_id for other faces.
 */
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

/**
 * @brief Return the configured name of a boundary batch.
 * @param batch_id Boundary batch ID.
 * @return Boundary name.
 * @throws std::out_of_range If @p batch_id is outside the name table.
 */
const std::string&
OrthoMeshTopo::boundary_batch_name(int batch_id) const
{
    if (batch_id < 0
        || static_cast<size_t>(batch_id) >= d_boundary_names.size())
    {
        throw std::out_of_range("Requested boundary batch is not found.");
    }
    return d_boundary_names[static_cast<size_t>(batch_id)];
}

/**
 * @brief Validate that a boundary batch exists and is not periodic.
 * @param batch_id Boundary batch ID.
 * @throws std::out_of_range If the ID is invalid or belongs to a periodic axis.
 */
void OrthoMeshTopo::validate_boundary_batch(int batch_id) const
{
    if (batch_id < 0 || batch_id >= 6)
    {
        throw std::out_of_range("Requested boundary batch is not found.");
    }
    const auto dim = static_cast<size_t>(batch_id / 2);
    if (d_indexer.periodic_dimensions[dim])
    {
        throw std::out_of_range(
            "Requested boundary batch is in a periodic dimension.");
    }
}

/**
 * @brief Enumerate boundary batches for non-periodic dimensions.
 * @return Available boundary batch IDs in dimension order.
 */
std::vector<int> OrthoMeshTopo::boundary_batch_ids() const
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

/**
 * @brief Count boundary batches from non-periodic dimensions.
 * @return Number of available lower and upper batches.
 */
int OrthoMeshTopo::num_boundary_batches() const noexcept
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

/** @brief Precompute owner and neighbor coordinates for every face plane. */
void OrthoMeshTopo::initialize_face_adjacency()
{
    using enum Indexer::Dimension;

    auto init_dim = [&](Indexer::Dimension dim) {
        const auto cells = d_indexer.num_cells_per_dim[dim];
        auto& faces = d_face_cells_per_dim[dim];
        faces.resize(d_indexer.num_nodes_per_dim[dim]);

        if (d_indexer.periodic_dimensions[dim])
        {
            for (Ordinal face = 0; face < cells; ++face)
            {
                faces[face] = {
                    face == 0 ? cells - 1 : face - 1,
                    face};
            }
            return;
        }

        faces[0] = {0, invalid_ordinal};
        for (Ordinal face = 1; face < cells; ++face)
        {
            faces[face] = {face - 1, face};
        }
        faces[cells] = {
            cells == 0 ? 0 : cells - 1,
            invalid_ordinal};
    };

    init_dim(I);
    init_dim(J);
    init_dim(K);
}

/** @brief Precompute neighboring cell coordinates along every dimension. */
void OrthoMeshTopo::initialize_cell_adjacency()
{
    using enum Indexer::Dimension;

    const auto ni = d_indexer.num_cells_per_dim[I];
    const auto nj = d_indexer.num_cells_per_dim[J];
    const auto nk = d_indexer.num_cells_per_dim[K];

    d_neighbors_per_dim[I].resize(ni);
    d_neighbors_per_dim[J].resize(nj);
    d_neighbors_per_dim[K].resize(nk);

    auto init_dim = [&](Indexer::Dimension dim, Ordinal n) {
        if (n == 0)
        {
            return;
        }

        auto& neighbors = d_neighbors_per_dim[dim];
        if (n == 1)
        {
            if (d_indexer.periodic_dimensions[dim])
            {
                neighbors[0] = {2, {0, 0}};
            }
            return;
        }

        for (Ordinal idx = 1; idx < n - 1; ++idx)
        {
            neighbors[idx] = {2, {idx - 1, idx + 1}};
        }
        if (!d_indexer.periodic_dimensions[dim])
        {
            neighbors[0] = {1, {1}};
            neighbors[n - 1] = {1, {n - 2}};
        }
        else
        {
            neighbors[0] = {2, {n - 1, 1}};
            neighbors[n - 1] = {2, {n - 2, 0}};
        }
    };

    init_dim(I, ni);
    init_dim(J, nj);
    init_dim(K, nk);
}

} // namespace SimpleFluid::Meshes
