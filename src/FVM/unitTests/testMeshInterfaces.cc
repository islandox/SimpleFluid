/**
 * @file testMeshInterfaces.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for the FVM-facing interface of statically dispatched meshes.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "FVM/OperatorDetails.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>

namespace
{

using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
using SemiStructured = SimpleFluid::Meshes::SemiStructuredXY_Z;

template<class Mesh>
void expect_identifier_conversion_interface(const Mesh& mesh)
{
    static_assert(
        std::is_same_v<typename Mesh::local_ordinal_type, size_t>);
    static_assert(
        std::is_same_v<typename Mesh::scalar_type, SimpleFluid::real_t>);

    for (size_t local_id = 0; local_id < mesh.num_cells(); ++local_id)
    {
        const auto cell_id = mesh.cell_id(local_id);
        EXPECT_EQ(mesh.cell_local_id(cell_id), local_id);
        EXPECT_GT(mesh.cell_volume(cell_id), 0.0);
    }
    for (size_t local_id = 0; local_id < mesh.num_faces(); ++local_id)
    {
        const auto face_id = mesh.face_id(local_id);
        EXPECT_EQ(mesh.face_local_id(face_id), local_id);
        EXPECT_GT(mesh.face_area(face_id), 0.0);
    }
    for (size_t local_id = 0; local_id < mesh.num_nodes(); ++local_id)
    {
        const auto node_id = mesh.node_id(local_id);
        EXPECT_EQ(mesh.node_local_id(node_id), local_id);
        EXPECT_EQ(mesh.node_coord(node_id), mesh.node_coordinates(node_id));
    }

    for (size_t face_lid = 0; face_lid < mesh.num_faces(); ++face_lid)
    {
        const auto face_id = mesh.face_id(face_lid);
        if (mesh.is_exterior_face(face_id))
        {
            EXPECT_EQ(mesh.neighbor_cell(face_id), Mesh::invalid_cell_id());
            return;
        }
    }
    FAIL() << "Expected at least one exterior face.";
}

template<class Mesh>
void expect_fvm_geometry_helpers(const Mesh& mesh)
{
    bool tested_interior = false;
    bool tested_boundary = false;

    for (size_t cell_lid = 0;
         cell_lid < mesh.num_owned_cells();
         ++cell_lid)
    {
        const auto cell_id = mesh.cell_id(cell_lid);
        for (const auto face_id : mesh.faces(cell_id))
        {
            const auto face_lid = mesh.face_local_id(face_id);
            if (!tested_interior && mesh.is_interior_face(face_id))
            {
                const auto other_id =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_id, cell_id);
                const auto other_lid = mesh.cell_local_id(other_id);
                EXPECT_GT(
                    SimpleFluid::FVM::detail::
                        interior_diffusion_coefficient(
                            mesh, face_lid, cell_lid, other_lid, 1.0),
                    0.0);
                tested_interior = true;
            }
            if (!tested_boundary && mesh.is_boundary_face(face_id))
            {
                EXPECT_GT(
                    SimpleFluid::FVM::detail::
                        boundary_diffusion_coefficient(
                            mesh, face_lid, cell_lid, 1.0),
                    0.0);
                tested_boundary = true;
            }
        }
    }

    EXPECT_TRUE(tested_interior);
    EXPECT_TRUE(tested_boundary);

    const auto stencils =
        SimpleFluid::FVM::detail::least_squares_gradient_stencils(mesh);
    EXPECT_EQ(stencils.size(), mesh.num_owned_cells());
    for (const auto& stencil : stencils)
    {
        for (const auto& entry : stencil)
        {
            EXPECT_LT(entry.cell_lid, mesh.num_local_cells());
        }
    }

    const auto locations =
        SimpleFluid::FVM::detail::boundary_face_locations(mesh);
    EXPECT_EQ(locations.size(), mesh.num_faces());

    if constexpr (std::ranges::range<
                      decltype(mesh.boundary_face_batch(0))>)
    {
        // View-based API
        for (int batch_id : mesh.boundary_batch_ids())
        {
            size_t in_batch_id = 0;
            for (auto face_id : mesh.boundary_face_batch(batch_id))
            {
                const auto face_lid = mesh.face_local_id(face_id);
                EXPECT_TRUE(locations[face_lid].active);
                EXPECT_EQ(locations[face_lid].batch_id, batch_id);
                EXPECT_EQ(locations[face_lid].in_batch_id, in_batch_id);
                ++in_batch_id;
            }
        }
    }
    else
    {
        // Legacy materialized API
        for (const auto& [batch_id, batch] : mesh.boundary_batches())
        {
            for (size_t in_batch_id = 0;
                 in_batch_id < batch.face_lids.size();
                 ++in_batch_id)
            {
                const auto face_lid =
                    mesh.face_local_id(batch.face_lids[in_batch_id]);
                EXPECT_TRUE(locations[face_lid].active);
                EXPECT_EQ(locations[face_lid].batch_id, batch_id);
                EXPECT_EQ(locations[face_lid].in_batch_id, in_batch_id);
            }
        }
    }
}

Cartesian make_cartesian()
{
    return Cartesian({{
        {0.0, 1.0, 2.0},
        {0.0, 1.0, 2.0},
        {0.0, 1.0, 2.0}}});
}

Cylindrical make_cylindrical()
{
    constexpr auto pi = std::numbers::pi_v<SimpleFluid::real_t>;
    return Cylindrical({{
        {1.0, 2.0, 3.0},
        {0.0, 0.5 * pi, pi},
        {0.0, 1.0, 2.0}}});
}

SemiStructured make_semi_structured()
{
    return SemiStructured(
        {{0.0, 0.0, 0.0},
         {1.0, 0.0, 0.0},
         {1.0, 1.0, 0.0},
         {0.0, 1.0, 0.0}},
        {{0, 1, 3}, {1, 2, 3}},
        {0.0, 1.0, 2.0});
}

} // namespace

/** @brief Verifies mesh identifier conversion and identifier-based queries. */
TEST(MeshInterfacesTest, ConvertsIdentifiersAndQueriesById)
{
    expect_identifier_conversion_interface(make_cartesian());
    expect_identifier_conversion_interface(make_cylindrical());
    expect_identifier_conversion_interface(make_semi_structured());
}

/** @brief Verifies mesh implementations support finite-volume geometry helpers. */
TEST(MeshInterfacesTest, SupportsFiniteVolumeGeometryHelpers)
{
    expect_fvm_geometry_helpers(make_cartesian());
    expect_fvm_geometry_helpers(make_cylindrical());
    expect_fvm_geometry_helpers(make_semi_structured());
}
