/**
 * @file testGhostSyncMultiRank.cc
 * @brief Multi-rank ghost synchronization tests for cell fields.
 */

#include <gtest/gtest.h>

#include "fields/CellField.hh"
#include "geometry/MeshFactory.hh"
#include "utils/testing_environment.hh"

#include <mpi.h>

#include <cstddef>
#include <memory>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_4x4x4_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));

    SimpleFluid::ArrReal edges(5);
    for (std::size_t i = 0; i <= 4; ++i)
        edges[i] = static_cast<SimpleFluid::real_t>(i);

    db->set("X", edges);
    db->set("Y", edges);
    db->set("Z", edges);
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

SimpleFluid::SP<MeshType> make_10x10x10_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));

    SimpleFluid::ArrReal edges(11);
    for (std::size_t i = 0; i <= 10; ++i)
        edges[i] = static_cast<SimpleFluid::real_t>(i);

    db->set("X", edges);
    db->set("Y", edges);
    db->set("Z", edges);
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
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
    MPI_Allreduce(&local_owned, &global_owned, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_ghosts, &global_ghosts, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

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
