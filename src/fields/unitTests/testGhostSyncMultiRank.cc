/**
 * @file testGhostSyncMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Multi-rank ghost synchronization tests for cell fields.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"
#include "parallel/MPI_interface.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cstddef>
#include <memory>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_4x4x4_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_4x4x4_database());
}

SimpleFluid::SP<MeshType> make_10x10x10_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_10x10x10_database());
}

void expect_partitioned_mesh(const MeshType& mesh,
                             std::size_t expected_global_cells)
{
    const auto comm = mesh.owned_cell_map()->getComm();
    const int local_owned = static_cast<int>(mesh.num_owned_cells());
    const int local_ghosts = static_cast<int>(
        mesh.num_local_cells() - mesh.num_owned_cells());

    int global_owned = 0;
    int global_ghosts = 0;
    my_mpi::global_sum(local_owned, global_owned);
    my_mpi::global_sum(local_ghosts, global_ghosts);

    EXPECT_GT(comm->getSize(), 1);
    EXPECT_GT(mesh.num_owned_cells(), 0u)
        << "Rank " << comm->getRank() << " owns no cells.";
    EXPECT_GT(mesh.num_local_cells(), mesh.num_owned_cells())
        << "Rank " << comm->getRank()
        << " has no ghost cells, so ghost sync is not exercised.";
    EXPECT_EQ(static_cast<std::size_t>(global_owned), expected_global_cells);
    EXPECT_GT(global_ghosts, 0);
}

} // namespace

/**
 * @brief Sets each owned cell to its global ID and verifies ghost sync propagates values to all local cells.
 */
TEST(GhostSyncMultiRankTest, SyncsOwnedGlobalIdValuesToAllLocalCells)
{
    auto mesh = make_4x4x4_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }
    expect_partitioned_mesh(*mesh, 64u);

    FieldType field(mesh, "global_id_field");
    for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        field.set_owned_value(
            cell_lid,
            static_cast<Pack::scalar_type>(mesh->cell_global_id(cell_lid)));
    }

    field.sync_ghosts();

    for (std::size_t cell = 0; cell < mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(cell);
        EXPECT_DOUBLE_EQ(
            field.local_value(cell_lid),
            static_cast<Pack::scalar_type>(mesh->cell_global_id(cell_lid)));
    }
}

/**
 * @brief Verifies sync_periodic_boundaries propagates scalar and vector values across periodic pairs.
 */
TEST(GhostSyncMultiRankTest, SyncPeriodicBoundariesUpdatesScalarAndVectorGhosts)
{
    auto mesh = make_4x4x4_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }
    expect_partitioned_mesh(*mesh, 64u);

    FieldType scalar(mesh, "periodic_scalar", false);
    VectorFieldType vector(mesh, "periodic_vector", false);
    for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        const auto gid =
            static_cast<Pack::scalar_type>(mesh->cell_global_id(cell_lid));
        scalar.set_owned_value(cell_lid, gid + 0.25);
        vector.set_owned_value(cell_lid, {gid, 2.0 * gid, -gid});
    }

    mesh->sync_periodic_boundaries(scalar);
    mesh->sync_periodic_boundaries(vector);

    for (std::size_t cell = 0; cell < mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(cell);
        const auto gid =
            static_cast<Pack::scalar_type>(mesh->cell_global_id(cell_lid));
        EXPECT_DOUBLE_EQ(scalar.local_value(cell_lid), gid + 0.25);

        const auto value = vector.local_value(cell_lid);
        EXPECT_DOUBLE_EQ(value.x, gid);
        EXPECT_DOUBLE_EQ(value.y, 2.0 * gid);
        EXPECT_DOUBLE_EQ(value.z, -gid);
    }
}

/**
 * @brief Verifies ghost sync on a larger 10×10×10 mesh requiring at least four MPI ranks.
 */
TEST(GhostSyncMultiRankTest, SyncsOwnedGlobalIdValuesOnLargeMesh)
{
    auto mesh = make_10x10x10_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 4)
    {
        GTEST_SKIP() << "This test requires at least four MPI ranks.";
    }
    expect_partitioned_mesh(*mesh, 1000u);

    FieldType field(mesh, "global_id_field_large");
    for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        field.set_owned_value(
            cell_lid,
            static_cast<Pack::scalar_type>(mesh->cell_global_id(cell_lid)));
    }

    field.sync_ghosts();

    for (std::size_t cell = 0; cell < mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(cell);
        EXPECT_DOUBLE_EQ(
            field.local_value(cell_lid),
            static_cast<Pack::scalar_type>(mesh->cell_global_id(cell_lid)));
    }
}
