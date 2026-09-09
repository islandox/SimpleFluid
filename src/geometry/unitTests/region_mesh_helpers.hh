/**
 * @file region_mesh_helpers.hh
 * @author islandox
 * @brief Composite fixtures and test-only explicit reference conversion.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once
#include "geometry/mesh/MultiRegionMesh.hh"
#include <map>

namespace SimpleFluid::test
{
inline SP<Meshes::MultiRegionMesh> two_regions(size_t n = 2)
{
    ArrReal x, y;
    for (size_t i = 0; i <= n; ++i) { x.push_back(real_t(i) / n); y.push_back(real_t(i) / n); }
    auto right = x; for (auto& v : right) v += 1;
    return std::make_shared<Meshes::MultiRegionMesh>(std::vector<Meshes::MultiRegionMesh::Region>{
        Meshes::cartesian_region("left", {{x, y, y}}), Meshes::cartesian_region("right", {{right, y, y}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{Meshes::StructuredPatchInterface{{0, 1}, {1, 0}}});
}
inline SP<Meshes::MultiRegionMesh> mixed_regions()
{
    using namespace Meshes;
    auto prism = std::make_shared<SemiStructuredXY_Z>(Arr<MeshUtils::Vec3>{{1,0,0},{2,0,0},{1,1,0}},
        Arr<Arr<unsigned>>{{0,1,2}}, ArrReal{0,0.5,1},
        Arr<SemiStructuredXY_Z::BoundaryEdge>{{2,0,"join"},{0,1,"bottom"},{1,2,"slope"}});
    auto cart = cartesian_region("hex", {{{0,0.5,1},{0,1},{0,0.5,1}}});
    int join = -1;
    for (auto b : prism->boundary_batch_ids()) if (prism->boundary_batch_name(b) == "join") join = b;
    ExplicitConformingInterface interface{0,1,1,join,{}};
    for (size_t a = 0; a < cart.layout().faces; ++a)
        if (cart.topology().boundary_id(a) == 1)
            for (size_t b = 0; b < prism->num_faces(); ++b)
                if (prism->boundary_id(prism->face_id(b)) == join
                    && (cart.geometry().face_centroid(a) - prism->face_centroid(prism->face_id(b))).norm() < 1e-12)
                    interface.faces.emplace_back(a, b);
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{std::move(cart), native_region("prism", prism)},
        std::vector<MultiRegionMesh::Interface>{std::move(interface)});
}

/** @brief Explicit geometry exists solely as an independent operator reference in tests. */
inline SP<Meshes::UnstructuredMesh> explicit_reference(const Meshes::MultiRegionMesh& mesh)
{
    using U = Meshes::UnstructuredMesh;
    std::map<std::tuple<real_t,real_t,real_t>, uint64_t> node_ids;
    Arr<MeshUtils::Vec3> nodes;
    auto node = [&](uint64_t n)
    {
        const auto p = mesh.node_coordinates(n);
        auto [it, added] = node_ids.emplace(std::tuple{p.x,p.y,p.z}, nodes.size());
        if (added) nodes.push_back(p);
        return it->second;
    };
    Arr<U::CellDefinition> cells;
    for (size_t c = 0; c < mesh.num_cells(); ++c)
    {
        U::CellDefinition cell{mesh.cell_type(c), {}};
        for (auto n : mesh.cell_nodes(c)) cell.node_ids.push_back(node(n));
        cells.push_back(std::move(cell));
    }
    Arr<U::BoundaryFaceDefinition> boundaries;
    for (size_t f = 0; f < mesh.num_faces(); ++f)
        if (mesh.is_boundary_face(f))
        {
            U::BoundaryFaceDefinition b{{}, mesh.boundary_id(f), mesh.boundary_name(f)};
            for (auto n : mesh.face_nodes(f)) b.node_ids.push_back(node(n));
            boundaries.push_back(std::move(b));
        }
    return std::make_shared<U>(nodes, cells, boundaries);
}
} // namespace SimpleFluid::test
