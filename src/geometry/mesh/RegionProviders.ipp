/**
 * @file RegionProviders.ipp
 * @author islandox
 * @brief Ordinal incidence adapters over native compact meshes.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once

#include "geometry/mesh/OrthogonalIndexer.hh"
#include <array>
#include <cstdint>

namespace SimpleFluid::Meshes
{
namespace region_detail
{
inline uint64_t cartesian_cell_node(const OrthogonalIndexer& indexer, size_t cell, size_t corner)
{
    constexpr unsigned corners[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                      {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const auto c = indexer.cell_id(cell);
    return indexer.node_ordinal({(c.i+corners[corner][0])%indexer.num_nodes_per_dim[0],
        (c.j+corners[corner][1])%indexer.num_nodes_per_dim[1],(c.k+corners[corner][2])%indexer.num_nodes_per_dim[2]});
}
inline uint64_t cartesian_face_node(const OrthogonalIndexer& indexer, size_t face, size_t corner)
{
    const auto f = indexer.face_id(face);
    std::array<unsigned, 3> n{f.i, f.j, f.k};
    constexpr unsigned corners[4][2] = {{0,0},{1,0},{1,1},{0,1}};
    size_t tangent = 0;
    for (size_t axis = 0; axis < 3; ++axis)
        if (axis != f.orientation) n[axis] += corners[corner][tangent++];
    for(size_t a=0;a<3;++a) n[a]%=indexer.num_nodes_per_dim[a];
    return indexer.node_ordinal({n[0], n[1], n[2]});
}
} // namespace region_detail

} // namespace SimpleFluid::Meshes
