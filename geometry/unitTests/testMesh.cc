/**
 * @file testMesh.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for Mesh class
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <gtest/gtest.h>
#include "geometry/Mesh.hh"

TEST(MeshTest, AssembleMeshWithoutThrowing)
{
    using MeshType = SimpleFluid::Mesh<SimpleFluid::TpetraTypes<>>;
    MeshType mesh;

    // Add some cells, faces, and nodes to the mesh
    auto cell0 = mesh.add_cell(0, MeshType::CellType::HEXAHEDRON, 1.0, {0.5, 0.5, 0.5});
    auto cell1 = mesh.add_cell(1, MeshType::CellType::HEXAHEDRON, 1.0, {1.5, 0.5, 0.5});

    auto face0 = mesh.add_face(cell0, cell1, MeshType::FaceType::QUAD, 0, 1.0, {1.0, 0.0, 0.0}, {1.0, 0.5, 0.5});
    auto face1 = mesh.add_face(cell1, cell0, MeshType::FaceType::QUAD, 0, 1.0, {-1.0, 0.0, 0.0}, {1.5, 0.5, 0.5});

    auto node0 = mesh.add_node({0.0, 0.0, 0.0});
    auto node1 = mesh.add_node({1.0, 0.0, 0.0});
    auto node2 = mesh.add_node({1.0, 1.0, 0.0});
    auto node3 = mesh.add_node({0.0, 1.0, 0.0});
    auto node4 = mesh.add_node({0.5, 0.5, 1.0});

    // Set connectivity
    mesh.set_cell_faces(cell0, {face0});
    mesh.set_cell_faces(cell1, {face1});

    mesh.set_cell_nodes(cell0, {node0, node1, node2, node3});
    mesh.set_cell_nodes(cell1, {node4});

    mesh.set_face_nodes(face0, {node1});
    mesh.set_face_nodes(face1, {node4});

    // Assemble the mesh (create maps and device views)
    mesh.assemble();
}