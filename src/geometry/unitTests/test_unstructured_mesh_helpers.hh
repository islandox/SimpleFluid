#pragma once

#include "geometry/mesh/UnstructuredMesh.hh"

#include <memory>

namespace SimpleFluid::test
{

/** @brief Shared serial hex-line input for partition and field integration tests. */
inline SimpleFluid::SP<SimpleFluid::Meshes::UnstructuredMesh>
make_partition_test_hex_line(unsigned cells)
{
    using UnstructuredMesh = SimpleFluid::Meshes::UnstructuredMesh;
    using Vec3 = UnstructuredMesh::Vec3;
    using CellDefinition = UnstructuredMesh::CellDefinition;
    using BoundaryFaceDefinition =
        UnstructuredMesh::BoundaryFaceDefinition;
    const auto nodes_per_x = cells + 1U;
    auto node_id = [nodes_per_x](unsigned i,
                                 unsigned j,
                                 unsigned k)
    {
        return i + nodes_per_x * (j + 2U * k);
    };

    SimpleFluid::Arr<Vec3> nodes;
    nodes.reserve(static_cast<size_t>(nodes_per_x) * 4U);
    for (unsigned k = 0; k < 2U; ++k)
    {
        for (unsigned j = 0; j < 2U; ++j)
        {
            for (unsigned i = 0; i <= cells; ++i)
            {
                nodes.push_back({
                    static_cast<SimpleFluid::real_t>(i),
                    static_cast<SimpleFluid::real_t>(j),
                    static_cast<SimpleFluid::real_t>(k)});
            }
        }
    }

    SimpleFluid::Arr<CellDefinition> cell_defs;
    cell_defs.reserve(cells);
    for (unsigned cell = 0; cell < cells; ++cell)
    {
        cell_defs.push_back({
            UnstructuredMesh::CellType::HEXAHEDRON,
            {
                node_id(cell, 0, 0),
                node_id(cell + 1U, 0, 0),
                node_id(cell + 1U, 1, 0),
                node_id(cell, 1, 0),
                node_id(cell, 0, 1),
                node_id(cell + 1U, 0, 1),
                node_id(cell + 1U, 1, 1),
                node_id(cell, 1, 1),
            }});
    }

    SimpleFluid::Arr<BoundaryFaceDefinition> boundaries{
        {{node_id(0, 1, 0),
          node_id(0, 1, 1),
          node_id(0, 0, 1),
          node_id(0, 0, 0)}, 1, "xmin"},
        {{node_id(cells, 0, 0),
          node_id(cells, 0, 1),
          node_id(cells, 1, 1),
          node_id(cells, 1, 0)}, 2, "xmax"}};

    return std::make_shared<UnstructuredMesh>(
        nodes,
        cell_defs,
        boundaries);
}

} // namespace SimpleFluid::test
