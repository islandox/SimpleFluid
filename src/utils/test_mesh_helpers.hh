/**
 * @file test_mesh_helpers.hh
 * @brief Shared test utilities for building meshes via MeshFactory.
 *
 * Include this header in unit test files to avoid duplicating mesh setup.
 */
#pragma once

#include "dataclass/Database.hh"
#include "geometry/MeshFactory.hh"

#include <cstddef>
#include <memory>

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
inline SP<const Database> make_box_database(std::size_t n_x,
                                             std::size_t n_y,
                                             std::size_t n_z,
                                             real_t mesh_size = 1.0)
{
    auto db = std::make_shared<Database>();
    db->set("dimension", 3);
    db->set("mesh_size", mesh_size);
    db->set("domain_type",
            static_cast<int>(MeshFactory::DomainType::BOX));

    ArrReal x_edges(n_x + 1);
    for (std::size_t i = 0; i <= n_x; ++i)
        x_edges[i] = static_cast<real_t>(i) * mesh_size;

    ArrReal y_edges(n_y + 1);
    for (std::size_t i = 0; i <= n_y; ++i)
        y_edges[i] = static_cast<real_t>(i) * mesh_size;

    ArrReal z_edges(n_z + 1);
    for (std::size_t i = 0; i <= n_z; ++i)
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

} // namespace SimpleFluid::test
