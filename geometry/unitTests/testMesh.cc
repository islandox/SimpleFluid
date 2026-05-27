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

#include "utils/testing_environment.hh"

#include <stdexcept>
#include <string>

namespace
{

using MeshType = SimpleFluid::Mesh<SimpleFluid::TpetraTypes<>>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

} // namespace

TEST(MeshTest, EmptyMeshAssemblesMapsAndViews)
{
    MeshType mesh;

    mesh.assemble();

    EXPECT_EQ(mesh.spatial_dimension(), 0u);
    EXPECT_EQ(mesh.num_local_cells(), 0u);
    EXPECT_EQ(mesh.num_owned_cells(), 0u);
    EXPECT_EQ(mesh.num_faces(), 0u);
    EXPECT_TRUE(mesh.owned_cell_map() != Teuchos::null);
    EXPECT_TRUE(mesh.overlap_cell_map() != Teuchos::null);
    EXPECT_EQ(mesh.owned_cell_map()->getLocalNumElements(), 0u);
    EXPECT_EQ(mesh.overlap_cell_map()->getLocalNumElements(), 0u);
    EXPECT_EQ(mesh.global_to_local_cell(7),
              SimpleFluid::invalid_id<MeshType::local_ordinal_type>());
}

TEST(MeshTest, BaseExportVtuRequiresConcreteBackend)
{
    MeshType mesh;

    EXPECT_THROW(mesh.export_vtu("unused.vtu"), std::runtime_error);
}
