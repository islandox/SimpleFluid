/** @file MeshRegion.tcc
 * @brief Native adapter implementations for compiled or downstream instantiation.
 * Include in an owning implementation unit, never from a public header.
 */
#pragma once

#include "geometry/mesh/MeshRegion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include "geometry/mesh/RegionProviders.ipp"

namespace SimpleFluid::Meshes
{
template<class Native>
NativeRegionProvider<Native>::NativeRegionProvider(std::shared_ptr<const Native> mesh) : d_mesh(std::move(mesh))
{
    if (!d_mesh) throw std::invalid_argument("Region provider requires a non-null native mesh.");
    if (d_mesh->num_owned_cells() != d_mesh->num_cells() || d_mesh->num_owned_faces() != d_mesh->num_faces())
        throw std::invalid_argument("A region must be standalone serial native geometry.");
}

template<class Native>
RegionLayout NativeRegionProvider<Native>::layout() const
{
    if constexpr (std::same_as<Native, OrthogonalCartesian3D> || std::same_as<Native, OrthogonalCylindrial3D>)
    {
        const auto& n = d_mesh->indexer().num_cells_per_dim;
        const auto& p=d_mesh->indexer().periodic_dimensions;
        return {std::same_as<Native,OrthogonalCartesian3D>?RegionLayout::Family::Rectilinear:RegionLayout::Family::Cylindrical,
            d_mesh->num_cells(),d_mesh->num_faces(),d_mesh->num_nodes(),{n[0],n[1],n[2]},nullptr,{p[0],p[1],p[2]}};
    }
    else if constexpr (std::same_as<Native, SemiStructuredXY_Z>)
    {
        const auto& i = d_mesh->indexer();
        return {RegionLayout::Family::Extruded, d_mesh->num_cells(), d_mesh->num_faces(),
            d_mesh->num_nodes(), {i.num_cells_per_layer, i.num_side_faces_per_layer, i.num_layers}, d_mesh.get()};
    }
    else
        return {RegionLayout::Family::Explicit, d_mesh->num_cells(), d_mesh->num_faces(),
            d_mesh->num_nodes(), {}, d_mesh.get()};
}

template<class Native>
typename NativeRegionProvider<Native>::ID NativeRegionProvider<Native>::neighbor_cell(size_t f) const
{
    const auto c = d_mesh->neighbor_cell(d_mesh->face_id(f));
    return c == Native::invalid_cell_id() ? UnstructuredMesh::invalid_ordinal : d_mesh->cell_local_id(c);
}

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
        if constexpr (std::same_as<Native, OrthogonalCartesian3D> || std::same_as<Native, OrthogonalCylindrial3D>)
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
        if constexpr (std::same_as<Native, OrthogonalCartesian3D> || std::same_as<Native, OrthogonalCylindrial3D>)
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
    if constexpr (std::same_as<Native, OrthogonalCartesian3D> || std::same_as<Native, OrthogonalCylindrial3D>) return MeshUtils::CellType::HEXAHEDRON;
    else if constexpr (std::same_as<Native, UnstructuredMesh>) return d_mesh->cell_type(c);
    else
    {
        const auto count = d_mesh->xy_cell_nodes().at(d_mesh->cell_id(c).ij).size();
        if (count == 3) return MeshUtils::CellType::TRIPRISM;
        if (count == 4) return MeshUtils::CellType::HEXAHEDRON;
        return MeshUtils::CellType::POLYHEDRON;
    }
}

template<class Native>
MeshStorageReport NativeRegionProvider<Native>::storage_report() const
{
    return {.topology = d_mesh->topology_storage_bytes(), .geometry = d_mesh->geometry_storage_bytes()};
}
} // namespace SimpleFluid::Meshes
