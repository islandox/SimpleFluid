/**
 * @file RegionProviders.cc
 * @author islandox
 * @brief Independent rectilinear provider implementations.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#include "geometry/mesh/RegionProviders.hh"

#include <cmath>
#include <limits>
#include <numeric>

namespace SimpleFluid::Meshes
{
namespace
{
RegionLayout layout_of(const OrthogonalIndexer& i)
{
    const auto& n = i.num_cells_per_dim;
    return {RegionLayout::Family::Rectilinear, i.total_cells(), i.total_faces(), i.total_nodes(), {n[0], n[1], n[2]}};
}
} // namespace

RectilinearTopology::RectilinearTopology(std::shared_ptr<const OrthoMeshTopo> topology) : d_topology(std::move(topology))
{
    if (!d_topology) throw std::invalid_argument("Rectilinear topology requires a live template.");
    for (size_t a = 0; a < 3; ++a)
        if (indexer().periodic_dimensions[a] || indexer().num_cells_per_dim[a] == 0)
            throw std::invalid_argument("Composite rectilinear topology requires nonempty, nonperiodic axes.");
}
RegionLayout RectilinearTopology::layout() const { return layout_of(indexer()); }
EntityRange<uint64_t> RectilinearTopology::cell_faces(size_t c) const
{
    return {this, c, 6, [](const void* source, size_t c, size_t i) -> ID
    {
        const auto& indexer = static_cast<const RectilinearTopology*>(source)->indexer();
        const auto cell = indexer.cell_id(c);
        std::array<unsigned, 3> coords{cell.i, cell.j, cell.k};
        coords[i / 2] += i % 2;
        return indexer.face_ordinal({coords[0], coords[1], coords[2], static_cast<uint8_t>(i / 2)});
    }};
}
EntityRange<uint64_t> RectilinearTopology::cell_nodes(size_t c) const
{
    return {this, c, 8, [](const void* source, size_t c, size_t i)
        { return region_detail::cartesian_cell_node(static_cast<const RectilinearTopology*>(source)->indexer(), c, i); }};
}
EntityRange<uint64_t> RectilinearTopology::face_nodes(size_t f) const
{
    return {this, f, 4, [](const void* source, size_t f, size_t i)
        { return region_detail::cartesian_face_node(static_cast<const RectilinearTopology*>(source)->indexer(), f, i); }};
}
uint64_t RectilinearTopology::owner_cell(size_t f) const
{
    return indexer().cell_ordinal(d_topology->owner_cell(indexer().face_id(f)));
}
uint64_t RectilinearTopology::neighbor_cell(size_t f) const
{
    const auto c = d_topology->neighbor_cell(indexer().face_id(f));
    return c == OrthogonalIndexer::CellID{} ? UnstructuredMesh::invalid_ordinal : indexer().cell_ordinal(c);
}
MeshStorageReport RectilinearTopology::storage_report() const { return {.topology = d_topology->storage_bytes()}; }

RectilinearGeometry::RectilinearGeometry(Vec3D<ArrReal> edges) : d_edges(std::move(edges))
{
    size_t product = 1;
    for (size_t a = 0; a < 3; ++a)
    {
        const auto& e = d_edges[a];
        if (e.size() < 2 || e.size() > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Rectilinear geometry requires supported nonempty axis extents.");
        if (product > std::numeric_limits<size_t>::max() / e.size() / 3)
            throw std::overflow_error("Rectilinear entity counts overflow.");
        product *= e.size();
        for (size_t i = 0; i < e.size(); ++i)
            if (!std::isfinite(e[i]) || (i && e[i] <= e[i - 1]))
                throw std::invalid_argument("Rectilinear edges must be finite and strictly increasing.");
    }
    d_indexer = OrthogonalIndexer(static_cast<unsigned>(d_edges[0].size() - 1),
        static_cast<unsigned>(d_edges[1].size() - 1), static_cast<unsigned>(d_edges[2].size() - 1));
}
RegionLayout RectilinearGeometry::layout() const { return layout_of(d_indexer); }
real_t RectilinearGeometry::cell_volume(size_t c) const
{
    const auto id = d_indexer.cell_id(c);
    return (d_edges[0][id.i + 1] - d_edges[0][id.i]) * (d_edges[1][id.j + 1] - d_edges[1][id.j])
        * (d_edges[2][id.k + 1] - d_edges[2][id.k]);
}
RectilinearGeometry::Vec3 RectilinearGeometry::cell_centroid(size_t c) const
{
    const auto id = d_indexer.cell_id(c);
    return {std::midpoint(d_edges[0][id.i], d_edges[0][id.i + 1]),
        std::midpoint(d_edges[1][id.j], d_edges[1][id.j + 1]), std::midpoint(d_edges[2][id.k], d_edges[2][id.k + 1])};
}
real_t RectilinearGeometry::face_area(size_t f) const
{
    const auto id = d_indexer.face_id(f);
    const std::array<unsigned, 3> n{id.i, id.j, id.k};
    real_t area = 1;
    for (size_t a = 0; a < 3; ++a) if (a != id.orientation) area *= d_edges[a][n[a] + 1] - d_edges[a][n[a]];
    return area;
}
RectilinearGeometry::Vec3 RectilinearGeometry::face_centroid(size_t f) const
{
    const auto id = d_indexer.face_id(f);
    const std::array<unsigned, 3> n{id.i, id.j, id.k};
    std::array<real_t, 3> result;
    for (size_t a = 0; a < 3; ++a)
        result[a] = a == id.orientation ? d_edges[a][n[a]] : std::midpoint(d_edges[a][n[a]], d_edges[a][n[a] + 1]);
    return {result[0], result[1], result[2]};
}
RectilinearGeometry::Vec3 RectilinearGeometry::face_area_vector(size_t f) const
{
    const auto id = d_indexer.face_id(f);
    const std::array<unsigned, 3> n{id.i, id.j, id.k};
    std::array<real_t, 3> result{};
    result[id.orientation] = (n[id.orientation] == 0 ? -1 : 1) * face_area(f);
    return {result[0], result[1], result[2]};
}
RectilinearGeometry::Vec3 RectilinearGeometry::node_coordinates(size_t n) const
{
    const auto id = d_indexer.node_id(n);
    return {d_edges[0][id.i], d_edges[1][id.j], d_edges[2][id.k]};
}
MeshStorageReport RectilinearGeometry::storage_report() const
{
    return {.geometry = (d_edges[0].capacity() + d_edges[1].capacity() + d_edges[2].capacity()) * sizeof(real_t)};
}
CartesianRegion cartesian_region(std::string name, Vec3D<ArrReal> edges)
{
    auto geometry = std::make_shared<const RectilinearGeometry>(std::move(edges));
    const auto n = geometry->layout().extents;
    auto topology = std::make_shared<const RectilinearTopology>(std::make_shared<const OrthoMeshTopo>(
        static_cast<unsigned>(n[0]), static_cast<unsigned>(n[1]), static_cast<unsigned>(n[2]), false, false, false,
        OrthoMeshTopo::BoundaryNames{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}));
    return {std::move(name), std::move(topology), std::move(geometry)};
}
} // namespace SimpleFluid::Meshes
