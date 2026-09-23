/** @file composite_partition_helpers.hh
 *  @brief Connected Cartesian, XY-Z and pentagonal-prism partition fixture.
 */
#pragma once
#include "geometry/mesh/MultiRegionMesh.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <numeric>

namespace SimpleFluid::test
{
using namespace Meshes;
using ID = MultiRegionMesh::ID;
inline std::shared_ptr<MultiRegionMesh> mixed_partition_source(unsigned layers = 8)
{
    ArrReal z(layers + 1); std::iota(z.begin(), z.end(), 0.0);
    auto cartesian = cartesian_region("cartesian", {{{0,1},{0,1},z}});
    auto xy_topology = std::make_shared<const ExtrudedTopology>(4, Arr<Arr<unsigned>>{{0,1,2,3}}, layers,
        Arr<SemiStructMeshTopo::BoundaryEdge>{{0,1,"bottom"},{1,2,"right"},{2,3,"top"},{3,0,"left"}});
    auto xy = extruded_region("xy_z", xy_topology, {{1,0,0},{2,0,0},{2,1,0},{1,1,0}}, z);
    Arr<MeshUtils::Vec3> nodes;
    const std::array<MeshUtils::Vec3, 5> base{{{2,0,0},{3,0,0},{3.2,0.5,0},{3,1,0},{2,1,0}}};
    for (unsigned k = 0; k <= layers; ++k)
        for (auto point : base) { point.z = k; nodes.push_back(point); }
    Arr<UnstructuredMesh::CellDefinition> cells;
    Arr<UnstructuredMesh::BoundaryFaceDefinition> boundaries;
    for (unsigned k = 0; k < layers; ++k)
    {
        const ID lower = k*5, upper = (k+1)*5;
        UnstructuredMesh::CellDefinition cell;
        cell.type = MeshUtils::CellType::POLYHEDRON;
        cell.face_node_ids.push_back({lower+4,lower+3,lower+2,lower+1,lower});
        cell.face_node_ids.push_back({upper,upper+1,upper+2,upper+3,upper+4});
        for (ID side = 0; side < 5; ++side)
            cell.face_node_ids.push_back({lower+side,lower+(side+1)%5,upper+(side+1)%5,upper+side});
        boundaries.push_back({{lower+4,lower,upper,upper+4}, 7, "left"});
        cells.push_back(std::move(cell));
    }
    const UnstructuredMesh untagged(nodes, cells, boundaries);
    for (ID face = 0; face < untagged.num_faces(); ++face)
        if (untagged.is_exterior_face(face) && untagged.boundary_id(face) == -1)
            boundaries.push_back({untagged.face_nodes(face), 8, "walls"});
    auto explicit_mesh = std::make_shared<UnstructuredMesh>(std::move(nodes), std::move(cells), std::move(boundaries));
    auto poly = native_region("polyhedra", explicit_mesh);
    int xy_left = -1, xy_right = -1;
    for (auto boundary : xy_topology->boundary_batch_ids())
    {
        if (xy_topology->boundary_batch_name(boundary) == "left") xy_left = boundary;
        if (xy_topology->boundary_batch_name(boundary) == "right") xy_right = boundary;
    }
    ExplicitConformingInterface first{0,1,1,xy_left,{}}, second{1,2,xy_right,7,{}};
    auto connect = [](const auto& a, int a_boundary, const auto& b, int b_boundary, auto& output)
    {
        for (ID f = 0; f < a.layout().faces; ++f)
            if (a.topology().boundary_id(f) == a_boundary)
                for (ID g = 0; g < b.layout().faces; ++g)
                    if (b.topology().boundary_id(g) == b_boundary
                        && (a.geometry().face_centroid(f)-b.geometry().face_centroid(g)).norm() < 1e-12)
                        output.emplace_back(f,g);
    };
    connect(cartesian,1,xy,xy_left,first.faces); connect(xy,xy_right,poly,7,second.faces);
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{cartesian,xy,poly},
        std::vector<MultiRegionMesh::Interface>{first,second}, Teuchos::rcp(new Teuchos::SerialComm<int>));
}

} // namespace SimpleFluid::test
