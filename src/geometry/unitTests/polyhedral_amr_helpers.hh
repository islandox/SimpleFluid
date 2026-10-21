/** @file polyhedral_amr_helpers.hh
 * @brief Connected native convex fixtures for hierarchy, packet and remap tests.
 */
#pragma once
#include "geometry/PolyhedralAMR.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <map>
#include <tuple>
namespace SimpleFluid::test
{
inline SP<const Meshes::MultiRegionMesh> polyhedral_pair(double shift = 0, bool labelled = true)
{
    using U = Meshes::UnstructuredMesh;
    Arr<MeshUtils::Vec3> nodes;
    std::map<std::tuple<double, double, double>, U::NodeID> ids;
    auto node = [&](double x, double y, double z)
    {
        const auto [it, inserted] = ids.emplace(std::tuple{x + shift, y, z}, nodes.size());
        if (inserted)
            nodes.push_back({x + shift, y, z});
        return it->second;
    };
    const std::array<std::array<double, 2>, 4> left{{{0, -0.25}, {1, 0}, {1, 1}, {0, 1}}};
    const std::array<std::array<double, 2>, 4> right{{{1, 0}, {2, 0}, {2, 1.5}, {1, 1}}};
    Arr<U::CellDefinition> cells;
    for (const auto& base : {left, right})
    {
        U::CellDefinition cell;
        cell.type = MeshUtils::CellType::HEXAHEDRON;
        for (const auto z : {0.0, 1.0})
            for (const auto& p : base)
                cell.node_ids.push_back(node(p[0], p[1], z));
        cells.push_back(std::move(cell));
    }
    const U untagged(nodes, cells);
    Arr<U::BoundaryFaceDefinition> boundaries;
    for (std::size_t f = 0; f < untagged.num_faces(); ++f)
        if (labelled && untagged.is_exterior_face(f))
        {
            const auto center = untagged.face_centroid(f);
            const auto id = center.z == 0 ? 0 : center.z == 1 ? 1 : 2;
            boundaries.push_back({untagged.face_nodes(f), id, id == 0 ? "bottom" : id == 1 ? "top" : "wall"});
        }
    const auto native = std::make_shared<U>(nodes, cells, boundaries);
    return std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{Meshes::native_region("native", native)},
        std::vector<Meshes::MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
}
inline SP<const Meshes::MultiRegionMesh> polyhedral_tetrahedron()
{
    using U = Meshes::UnstructuredMesh;
    Arr<MeshUtils::Vec3> nodes{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    U::CellDefinition cell;
    cell.type = MeshUtils::CellType::POLYHEDRON;
    cell.face_node_ids = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    Arr<U::CellDefinition> cells{std::move(cell)};
    const U untagged(nodes, cells);
    Arr<U::BoundaryFaceDefinition> boundaries;
    for (std::size_t f = 0; f < untagged.num_faces(); ++f)
        boundaries.push_back({untagged.face_nodes(f), 0, "wall"});
    return std::make_shared<Meshes::MultiRegionMesh>(std::vector<Meshes::MultiRegionMesh::Region>{Meshes::native_region(
                                                         "tetra", std::make_shared<U>(nodes, cells, boundaries))},
        std::vector<Meshes::MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
}
inline Meshes::PolyhedralAMRGrid polyhedral_grid(SP<const Meshes::MultiRegionMesh> mesh = polyhedral_pair())
{
    Meshes::PolyhedralAMRGrid result;
    result.mesh = std::move(mesh);
    result.physical_regions.assign(result.mesh->num_cells(), (std::uint64_t{1} << 55) + 19);
    for (std::uint64_t c = 0; c < result.mesh->num_cells(); ++c)
        result.materials.push_back(17 + 6 * c);
    return result;
}
} // namespace SimpleFluid::test
