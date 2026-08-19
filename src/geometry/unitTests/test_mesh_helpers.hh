/**
 * @file test_mesh_helpers.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shared test utilities for building meshes via MeshFactory.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 * Include this header in unit test files to avoid duplicating mesh setup.
 */
#pragma once

#include "dataclass/Database.hh"
#include "geometry/MeshFactory.hh"
#include "geometry/mesh/UnstructuredMesh.hh"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace SimpleFluid::test
{

/**
 * @brief Create a Database configured for an Nx × Ny × Nz structured box mesh.
 *
 * @param n_x Number of cells in the X direction.
 * @param n_y Number of cells in the Y direction.
 * @param n_z Number of cells in the Z direction.
 * @param mesh_size Size of the domain in each dimension.
 * @return Shared pointer to the configured Database.
 */
inline SP<const Database> make_box_database(size_t n_x,
                                             size_t n_y,
                                             size_t n_z,
                                             real_t mesh_size = 1.0)
{
    auto db = std::make_shared<Database>();
    db->set("dimension", 3);
    db->set("mesh_size", mesh_size);
    db->set("domain_type",
            static_cast<int>(MeshFactory::DomainType::BOX));

    ArrReal x_edges(n_x + 1);
    for (size_t i = 0; i <= n_x; ++i)
        x_edges[i] = static_cast<real_t>(i) * mesh_size;

    ArrReal y_edges(n_y + 1);
    for (size_t i = 0; i <= n_y; ++i)
        y_edges[i] = static_cast<real_t>(i) * mesh_size;

    ArrReal z_edges(n_z + 1);
    for (size_t i = 0; i <= n_z; ++i)
        z_edges[i] = static_cast<real_t>(i) * mesh_size;

    db->set("X", std::move(x_edges));
    db->set("Y", std::move(y_edges));
    db->set("Z", std::move(z_edges));
    db->set("domain_exterior_face_types",
            ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    return db;
}

/**
 * @brief Build a mesh from a Database using MeshFactory.
 *
 * @tparam Pack Tpetra type pack.
 * @param db Shared pointer to the configured Database.
 * @return Shared pointer to the assembled mesh.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
SP<Mesh<Pack>> build_mesh(const SP<const Database>& db)
{
    MeshFactory factory(db);
    return factory.template build<Pack>();
}

// -- convenience aliases for common meshes --------------------------------

/// Single hex cell: 1×1×1 (good for minimal equation tests)
inline SP<const Database> make_single_hex_database()
{
    return make_box_database(1, 1, 1);
}

/// Two hex cells along X: 2×1×1 (good for basic field tests)
inline SP<const Database> make_two_hex_database()
{
    return make_box_database(2, 1, 1);
}

/// 2×2×2 mesh (8 cells, good for topology tests)
inline SP<const Database> make_2x2x2_database()
{
    return make_box_database(2, 2, 2);
}

/// 3×3×3 mesh (27 cells, good for factory tests)
inline SP<const Database> make_3x3x3_database()
{
    return make_box_database(3, 3, 3);
}

/// 4×4×4 mesh (64 cells, good for multi-rank tests)
inline SP<const Database> make_4x4x4_database()
{
    return make_box_database(4, 4, 4);
}

/// 10×10×10 mesh (1000 cells, good for scaling tests)
inline SP<const Database> make_10x10x10_database()
{
    return make_box_database(10, 10, 10);
}

/**
 * @brief Build a native unstructured line of axis-aligned hexahedra.
 *
 * All six exterior sides are assigned the same boundary names used by the
 * structured box fixtures so operator and solver tests can compare the two
 * backends directly.
 *
 * @param cells Number of cells along X.
 * @param cell_width Width of each cell along X.
 * @return Replicated unstructured geometry ready for serial use or explicit
 *         partitioning with MeshPartitioner.
 */
inline SP<Meshes::UnstructuredMesh> make_unstructured_hex_line(
    size_t cells,
    real_t cell_width = 1.0)
{
    using Mesh = Meshes::UnstructuredMesh;
    using Vec3 = Mesh::Vec3;
    using CellDefinition = Mesh::CellDefinition;
    using BoundaryFaceDefinition = Mesh::BoundaryFaceDefinition;

    if (cells == 0)
    {
        throw std::invalid_argument(
            "Unstructured hex-line fixture requires at least one cell.");
    }
    if (!(cell_width > 0.0))
    {
        throw std::invalid_argument(
            "Unstructured hex-line fixture requires positive cell width.");
    }

    const auto nodes_per_x = cells + 1;
    const auto node_id = [nodes_per_x](size_t i, size_t j, size_t k)
    {
        return static_cast<Mesh::NodeID>(
            i + nodes_per_x * (j + 2 * k));
    };

    Arr<Vec3> nodes;
    nodes.reserve(nodes_per_x * 4);
    for (size_t k = 0; k < 2; ++k)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            for (size_t i = 0; i <= cells; ++i)
            {
                nodes.push_back({
                    static_cast<real_t>(i) * cell_width,
                    static_cast<real_t>(j),
                    static_cast<real_t>(k)});
            }
        }
    }

    Arr<CellDefinition> cell_definitions;
    cell_definitions.reserve(cells);
    Arr<BoundaryFaceDefinition> boundaries;
    boundaries.reserve(4 * cells + 2);
    for (size_t cell = 0; cell < cells; ++cell)
    {
        const auto x0y0z0 = node_id(cell, 0, 0);
        const auto x1y0z0 = node_id(cell + 1, 0, 0);
        const auto x1y1z0 = node_id(cell + 1, 1, 0);
        const auto x0y1z0 = node_id(cell, 1, 0);
        const auto x0y0z1 = node_id(cell, 0, 1);
        const auto x1y0z1 = node_id(cell + 1, 0, 1);
        const auto x1y1z1 = node_id(cell + 1, 1, 1);
        const auto x0y1z1 = node_id(cell, 1, 1);

        cell_definitions.push_back({
            Mesh::CellType::HEXAHEDRON,
            {x0y0z0, x1y0z0, x1y1z0, x0y1z0,
             x0y0z1, x1y0z1, x1y1z1, x0y1z1}});

        if (cell == 0)
        {
            boundaries.push_back(
                {{x0y1z0, x0y1z1, x0y0z1, x0y0z0}, 1, "xmin"});
        }
        if (cell + 1 == cells)
        {
            boundaries.push_back(
                {{x1y0z0, x1y0z1, x1y1z1, x1y1z0}, 2, "xmax"});
        }
        boundaries.push_back(
            {{x0y0z0, x0y0z1, x1y0z1, x1y0z0}, 3, "ymin"});
        boundaries.push_back(
            {{x0y1z0, x1y1z0, x1y1z1, x0y1z1}, 4, "ymax"});
        boundaries.push_back(
            {{x0y0z0, x1y0z0, x1y1z0, x0y1z0}, 5, "zmin"});
        boundaries.push_back(
            {{x0y0z1, x0y1z1, x1y1z1, x1y0z1}, 6, "zmax"});
    }

    return std::make_shared<Mesh>(
        nodes, cell_definitions, boundaries);
}

} // namespace SimpleFluid::test
