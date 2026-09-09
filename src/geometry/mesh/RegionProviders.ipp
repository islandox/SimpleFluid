/**
 * @file RegionProviders.ipp
 * @author islandox
 * @brief Ordinal incidence adapters over native compact meshes.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once

namespace SimpleFluid::Meshes
{
namespace region_detail
{
inline uint64_t cartesian_cell_node(const OrthogonalIndexer& indexer, size_t cell, size_t corner)
{
    constexpr unsigned corners[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                      {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const auto c = indexer.cell_id(cell);
    return indexer.node_ordinal({c.i + corners[corner][0], c.j + corners[corner][1], c.k + corners[corner][2]});
}
inline uint64_t cartesian_face_node(const OrthogonalIndexer& indexer, size_t face, size_t corner)
{
    const auto f = indexer.face_id(face);
    std::array<unsigned, 3> n{f.i, f.j, f.k};
    constexpr unsigned corners[4][2] = {{0,0},{1,0},{1,1},{0,1}};
    size_t tangent = 0;
    for (size_t axis = 0; axis < 3; ++axis)
        if (axis != f.orientation) n[axis] += corners[corner][tangent++];
    return indexer.node_ordinal({n[0], n[1], n[2]});
}
} // namespace region_detail

template<class Native>
EntityRange<uint64_t> NativeRegionProvider<Native>::cell_nodes(size_t c) const
{
    size_t count = 8;
    if constexpr (std::same_as<Native, SemiStructuredXY_Z>)
        count = 2 * d_mesh->xy_cell_nodes().at(d_mesh->cell_id(c).ij).size();
    else if constexpr (std::same_as<Native, UnstructuredMesh>) count = d_mesh->cell_nodes(c).size();
    return {this, c, count, [](const void* source, size_t c, size_t i) -> ID
    {
        const auto& m = *static_cast<const NativeRegionProvider*>(source)->d_mesh;
        if constexpr (std::same_as<Native, OrthogonalCartesian3D>)
            return region_detail::cartesian_cell_node(m.indexer(), c, i);
        else if constexpr (std::same_as<Native, SemiStructuredXY_Z>)
        {
            const auto id = m.cell_id(c);
            const auto& base = m.xy_cell_nodes()[id.ij];
            return m.node_local_id({base[i % base.size()], static_cast<unsigned>(id.k + i / base.size())});
        }
        else return m.cell_nodes(c)[i];
    }};
}

template<class Native>
EntityRange<uint64_t> NativeRegionProvider<Native>::face_nodes(size_t f) const
{
    size_t count = 4;
    if constexpr (std::same_as<Native, SemiStructuredXY_Z>)
    {
        const auto id = d_mesh->face_id(f);
        if (id.orientation == Native::Z_FACE) count = d_mesh->xy_cell_nodes().at(id.ij).size();
    }
    else if constexpr (std::same_as<Native, UnstructuredMesh>) count = d_mesh->face_nodes(f).size();
    return {this, f, count, [](const void* source, size_t f, size_t i) -> ID
    {
        const auto& m = *static_cast<const NativeRegionProvider*>(source)->d_mesh;
        if constexpr (std::same_as<Native, OrthogonalCartesian3D>)
            return region_detail::cartesian_face_node(m.indexer(), f, i);
        else if constexpr (std::same_as<Native, SemiStructuredXY_Z>)
        {
            const auto id = m.face_id(f);
            if (id.orientation == Native::Z_FACE) return m.node_local_id({m.xy_cell_nodes()[id.ij][i], id.k});
            const auto& edge = m.topology().side_face(id.ij);
            return m.node_local_id({edge.nodes[(i == 1 || i == 2) ? 1 : 0], id.k + (i >= 2 ? 1U : 0U)});
        }
        else return m.face_nodes(f)[i];
    }};
}

template<class Native>
MeshUtils::CellType NativeRegionProvider<Native>::cell_type(size_t c) const
{
    if constexpr (std::same_as<Native, OrthogonalCartesian3D>) return MeshUtils::CellType::HEXAHEDRON;
    else if constexpr (std::same_as<Native, UnstructuredMesh>) return d_mesh->cell_type(c);
    else
    {
        const auto count = d_mesh->xy_cell_nodes().at(d_mesh->cell_id(c).ij).size();
        if (count == 3) return MeshUtils::CellType::TRIPRISM;
        if (count == 4) return MeshUtils::CellType::HEXAHEDRON;
        return MeshUtils::CellType::INVALID;
    }
}

template<class Native>
MeshStorageReport NativeRegionProvider<Native>::storage_report() const
{
    return {.topology = d_mesh->topology_storage_bytes(), .geometry = d_mesh->geometry_storage_bytes()};
}
} // namespace SimpleFluid::Meshes
