/** @brief Fields-owned partitioning and selected-first ghost-sync integration tests. */
#include <gtest/gtest.h>

#include "fields/CellField.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/unitTests/test_unstructured_mesh_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <map>
#include <memory>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Factory = SimpleFluid::MeshReorderingFactory<Pack>;
using GlobalOrdinal = Pack::global_ordinal_type;
using LocalOrdinal = Pack::local_ordinal_type;
using Vec3 = Handle::Vec3;
using FieldType = SimpleFluid::CellField<Pack>;
using PartitionedUnstructured =
    SimpleFluid::Meshes::PartitionedMesh<SimpleFluid::Meshes::UnstructuredMesh, Pack>;
using PartitionedCellField = SimpleFluid::ScalarCellFieldStored<Pack, PartitionedUnstructured>;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

LocalOrdinal local(std::size_t value) { return static_cast<LocalOrdinal>(value); }

TEST(PartitionedFieldSyncTest, FieldSyncAfterPartitioning)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    FieldType field(mesh, "sync_test", false);

    // Set each owned cell's value to its global ID
    for (size_t o = 0; o < mesh->num_owned_cells(); ++o)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(o);
        const auto gid = mesh->cell_global_id(lid);
        field.set_owned_value(
            lid, static_cast<Pack::scalar_type>(gid));
    }

    // Sync to ghost cells
    field.sync_ghosts();

    // Verify every local cell (including ghosts) has correct GID-based value
    for (size_t c = 0; c < mesh->num_local_cells(); ++c)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(c);
        const auto gid = mesh->cell_global_id(lid);
        EXPECT_DOUBLE_EQ(
            field.local_value(lid),
            static_cast<Pack::scalar_type>(gid))
            << "Mismatch for cell LID " << c << " GID " << gid
            << " on rank " << comm->getRank();
    }
}

TEST(PartitionedFieldSyncTest, NativeUnstructuredFieldSyncAfterPartitioning)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    auto mesh = SimpleFluid::test::make_partition_test_hex_line(8);
    const auto partition = SimpleFluid::MeshPartitioner<Pack>::partition(*mesh, comm);
    const auto partitioned = std::make_shared<PartitionedUnstructured>(mesh, partition.indexer, comm);
    PartitionedCellField stored(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("partitioned_value"),
        partitioned);
    for (size_t local = 0; local < partitioned->num_owned_cells(); ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        stored.set_owned_value(
            local_id,
            static_cast<double>(partitioned->cell_global_id(local_id)));
    }
    stored.sync_ghosts();
    for (size_t local = 0; local < partitioned->num_local_cells(); ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_DOUBLE_EQ(
            stored.local_value(local_id),
            static_cast<double>(partitioned->cell_global_id(local_id)));
    }

}

TEST(PartitionedFieldSyncTest, MultiRegionOwnershipHalosPreserveFieldValues)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Requires MPI ranks.";
    const auto geometry = SimpleFluid::test::two_regions(2);
    SimpleFluid::SP<const Handle> mesh = std::make_shared<Handle>(geometry);
    SimpleFluid::ScalarCellFieldStored<Pack> field(mesh, "gid");
    for (std::size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        field.set_owned_value(local(cell), mesh->cell_global_id(local(cell)) + 0.5);
    field.sync_ghosts();
    for (std::size_t cell = 0; cell < mesh->num_cells(); ++cell)
        EXPECT_DOUBLE_EQ(field.local_value(local(cell)), mesh->cell_global_id(local(cell)) + 0.5);
}

TEST(PartitionedFieldSyncTest, SemiStructuredSelectedFirstPreservesGhostImports)
{
    using Native = SimpleFluid::Meshes::SemiStructuredXY_Z;
    const auto comm = Tpetra::getDefaultComm();
    for (const bool sparse_slab : {false, true})
    {
        SCOPED_TRACE(sparse_slab ? "one Z layer with empty slab owners" : "hybrid graph/slab owners");
        SimpleFluid::Arr<Native::Vec3> points;
        for (unsigned y = 0; y < 3; ++y)
            for (unsigned x = 0; x < 4; ++x)
                points.push_back({double(x) + .2 * y, double(y) + .05 * x, 0});
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>> polygons;
        for (unsigned x = 0; x < 3; ++x)
            for (unsigned y = 0; y < 2; ++y)
            {
                const unsigned a = x + 4 * y;
                polygons.push_back({a, a + 1, a + 5, a + 4});
            }
        const SimpleFluid::ArrReal z = sparse_slab ? SimpleFluid::ArrReal{0, .8} : SimpleFluid::ArrReal{0, .2, .9, 1.7};
        const auto geometry = std::make_shared<Native>(points, polygons, z);
        Handle::DistributionOptions options;
        options.allow_empty_partitions = true;
        options.ghost_layers = 1;
        const auto ranks = std::size_t(comm->getSize());
        options.xy_partitions = sparse_slab ? 1 : (ranks >= 2 && ranks % 2 == 0 ? 2 : ranks);
        options.z_partitions = ranks / *options.xy_partitions;
        auto parent = std::make_shared<Handle>(geometry, options);
        int local_empty = parent->num_owned_cells() == 0 ? 1 : 0, empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        if (sparse_slab)
            EXPECT_EQ(empty, comm->getSize() - 1);

        const auto reorder = [&](SimpleFluid::SP<Handle> input)
        {
            std::map<GlobalOrdinal, GlobalOrdinal> before;
            for (std::size_t cell = 0; cell < input->num_local_cells(); ++cell)
                before.emplace(input->cell_geometry_global_id(local(cell)), input->cell_global_id(local(cell)));
            std::vector<GlobalOrdinal> faces;
            for (std::size_t face = 0; face < input->num_faces(); ++face)
                faces.push_back(input->face_global_id(local(face)));
            const auto total = geometry->num_cells();
            std::vector<int> local_selected(total), selected(total);
            for (std::size_t cell = input->num_owned_cells() / 2; cell < input->num_owned_cells(); ++cell)
                local_selected[std::size_t(input->cell_geometry_global_id(local(cell)))] = 1;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(total), local_selected.data(), selected.data());
            // Match the geometry regression's pre-existing caches and repeated overlay composition.
            static_cast<void>(input->indexer());
            input->materialize_cell_faces();
            const auto layout = Factory::selected_cells_first(std::move(input),
                [selected](GlobalOrdinal canonical, const Vec3&) { return selected[std::size_t(canonical)] != 0; });
            SimpleFluid::ScalarCellFieldStored<Pack> imported(layout.mesh, "algebraic_cell_probe");
            SimpleFluid::ScalarFaceFieldStored<Pack> face_imported(layout.mesh, "algebraic_face_probe");
            for (std::size_t cell = 0; cell < layout.mesh->num_owned_cells(); ++cell)
                imported.set_owned_value(local(cell), double(layout.mesh->cell_global_id(local(cell))) + .125);
            for (std::size_t face = 0; face < layout.mesh->num_owned_faces(); ++face)
                face_imported.set_owned_value(local(face), double(layout.mesh->face_global_id(local(face))) + .25);
            imported.sync_ghosts();
            face_imported.sync_ghosts();
            for (std::size_t cell = 0; cell < layout.mesh->num_local_cells(); ++cell)
                EXPECT_DOUBLE_EQ(imported.local_value(local(cell)),
                    double(before.at(layout.mesh->cell_geometry_global_id(local(cell)))) + .125);
            for (std::size_t face = 0; face < layout.mesh->num_faces(); ++face)
                EXPECT_DOUBLE_EQ(face_imported.local_value(local(face)), double(faces[face]) + .25);
            return layout;
        };
        const auto first = reorder(std::move(parent));
        auto repeated = std::make_shared<Handle>(*first.mesh);
        const auto second = reorder(std::move(repeated));
        (void)second;
    }
}
} // namespace
