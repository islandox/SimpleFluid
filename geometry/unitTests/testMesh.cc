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

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <stdexcept>
#include <string>

namespace
{

using MeshType = SimpleFluid::Mesh<SimpleFluid::TpetraTypes<>>;

/**
 * @brief Global Google Test environment to initialize MPI and Kokkos.
 */
class KokkosEnvironment : public testing::Environment
{
public:
    void SetUp() override
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized)
        {
            MPI_Init(nullptr, nullptr);
            d_initialized_mpi = true;
        }

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
            d_initialized_kokkos = true;
        }
    }

    void TearDown() override
    {
        if (d_initialized_kokkos && Kokkos::is_initialized())
        {
            Kokkos::finalize();
        }

        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (d_initialized_mpi && !mpi_finalized)
        {
            MPI_Finalize();
        }
    }

private:
    bool d_initialized_mpi = false;
    bool d_initialized_kokkos = false;
};

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
