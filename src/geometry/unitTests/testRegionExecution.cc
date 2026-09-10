#include <gtest/gtest.h>
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Handle = MeshHandle<>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

auto one_native(const std::shared_ptr<OrthogonalCartesian3D>& native)
{
    return std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{native_region("native", native)},
        std::vector<MultiRegionMesh::Interface>{});
}
}

TEST(RegionExecutionTest, PinsNativeReplacementAndMoveBeforeAnyStorageChanges)
{
    const Vec3D<ArrReal> edges{{{0,0.5,1},{0,1},{0,1}}};
    auto native = std::make_shared<OrthogonalCartesian3D>(edges);
    auto mesh = one_native(native);
    auto replacement = OrthogonalCartesian3D({{{0,2},{0,2},{0,2}}});
    {
        const auto execution = mesh->acquire_execution_view();
        EXPECT_THROW(*native = replacement, std::logic_error);
        EXPECT_THROW(*native = std::move(replacement), std::logic_error);
        auto* alias = native.get();
        EXPECT_THROW(*native = std::move(*alias), std::logic_error);
        EXPECT_THROW((OrthogonalCartesian3D(std::move(*native))), std::logic_error);
        EXPECT_EQ(native->num_cells(), 2U);
        EXPECT_DOUBLE_EQ(mesh->cell_volume(0), 0.5);
        EXPECT_EQ(native->geometry_epoch(), 0U);
        auto independent_copy = *native;
        EXPECT_DOUBLE_EQ(independent_copy.cell_volume(independent_copy.cell_id(0)), 0.5);
        EXPECT_THROW(mesh->cell_volume(mesh->num_cells()), std::out_of_range);
        EXPECT_THROW(mesh->face_area(mesh->num_faces()), std::out_of_range);
        EXPECT_THROW(mesh->node_coordinates(mesh->num_nodes()), std::out_of_range);
    }
    *native = OrthogonalCartesian3D(edges);
    EXPECT_THROW(static_cast<void>(mesh->acquire_execution_view()), std::logic_error);
    EXPECT_THROW(mesh->cell_volume(0), std::logic_error);
}

TEST(RegionExecutionTest, PinsSharedProviderAndTopologyAliases)
{
    auto native = std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    using Provider = NativeRegionProvider<OrthogonalCartesian3D>;
    auto provider = std::make_shared<Provider>(native);
    MultiRegionMesh native_mesh({NativeIsoRegion<OrthogonalCartesian3D>("native", provider, provider)}, {});
    {
        const auto execution = native_mesh.acquire_execution_view();
        EXPECT_THROW(*provider = Provider(native), std::logic_error);
        EXPECT_DOUBLE_EQ(native_mesh.cell_volume(0), 1.0);
    }
    auto topology = std::make_shared<OrthoMeshTopo>(1,1,1,false,false,false,
        OrthoMeshTopo::BoundaryNames{"xmin","xmax","ymin","ymax","zmin","zmax"});
    auto rect = std::make_shared<RectilinearTopology>(topology);
    auto geometry = std::make_shared<RectilinearGeometry>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    MultiRegionMesh independent({CartesianRegion("independent", rect, geometry)}, {});
    {
        const auto execution = independent.acquire_execution_view();
        EXPECT_THROW(*topology = OrthoMeshTopo(2,1,1,false,false,false,
            OrthoMeshTopo::BoundaryNames{"xmin","xmax","ymin","ymax","zmin","zmax"}), std::logic_error);
        EXPECT_THROW(*rect = RectilinearTopology(topology), std::logic_error);
        EXPECT_EQ(independent.cell_faces(0).size(), 6U);
    }
}

TEST(RegionExecutionTest, NestedScopesAndExceptionsReleaseEveryLease)
{
    auto native = std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    auto first = one_native(native);
    auto second = test::two_regions();
    {
        const auto outer = first->acquire_execution_view();
        try
        {
            const auto other = second->acquire_execution_view();
            const auto nested = first->acquire_execution_view();
            EXPECT_DOUBLE_EQ(first->cell_volume(0), 1.0);
            throw std::runtime_error("kernel failure");
        }
        catch (const std::runtime_error&) {}
        EXPECT_THROW(*native = OrthogonalCartesian3D({{{0,2},{0,1},{0,1}}}), std::logic_error);
    }
    EXPECT_NO_THROW(*native = OrthogonalCartesian3D({{{0,2},{0,1},{0,1}}}));
    EXPECT_THROW(first->geometry_epoch(), std::logic_error);
    EXPECT_NO_THROW(second->validate_static());
}

TEST(RegionExecutionTest, CanonicalAndRegionNativeTraversalAgreeWithoutMaterialization)
{
    for (const auto& geometry : {test::two_regions(), test::coarse_fine_regions(),
             test::periodic_regions(), test::cylindrical_regions(), test::independent_extruded_regions()})
    {
        Handle mesh(geometry);
        real_t canonical = 0, regional = 0;
        const auto execution = mesh.acquire_execution_view();
        for (size_t c = 0; c < geometry->num_cells(); ++c) canonical += geometry->cell_volume(c);
        execution.visit_regions([&](size_t, uint64_t, const auto& region)
        {
            for (size_t c = 0; c < region.layout().cells; ++c) regional += region.geometry().cell_volume(c);
        });
        EXPECT_NEAR(canonical, regional, 1e-12);
        EXPECT_EQ(mesh.connectivity_storage_bytes(), 0U);
        EXPECT_FALSE(mesh.has_materialized_indexer());
    }
}

TEST(RegionExecutionTest, CompositeAndChildMotionAreRejectedUntilViewEnds)
{
    auto native = std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    auto composite = one_native(native);
    auto child_handle = std::make_shared<Handle>(native);
    auto handle = std::make_shared<Handle>(composite);
    PlanarALEMeshMotion<> motion(handle), child_motion(child_handle);
    {
        const auto execution = handle->acquire_execution_view();
        EXPECT_THROW(motion.begin_trial(1.2,0.1), std::logic_error);
        EXPECT_THROW(child_motion.begin_trial(1.2,0.1), std::logic_error);
        EXPECT_FALSE(motion.has_active_trial());
        EXPECT_FALSE(child_motion.has_active_trial());
        EXPECT_DOUBLE_EQ(composite->cell_volume(0), 1.0);
    }
    EXPECT_NO_THROW(motion.begin_trial(1.2,0.1));
    motion.rollback_trial();
    EXPECT_DOUBLE_EQ(composite->cell_volume(0), 1.0);
}

TEST(RegionExecutionMultiRankTest, OneRankExecutionLeaseRejectsMotionCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1) GTEST_SKIP() << "Requires an MPI decomposition.";
    auto geometry = test::two_regions();
    auto mesh = std::make_shared<Handle>(geometry);
    PlanarALEMeshMotion<> motion(mesh);
    std::unique_ptr<MultiRegionMesh::ExecutionView> execution;
    if (comm->getRank() == 0) execution = std::make_unique<MultiRegionMesh::ExecutionView>(geometry.get());
    EXPECT_THROW(motion.begin_trial(1.2,0.1), std::logic_error);
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(mesh->geometry_epoch(), 0U);
    execution.reset();
    EXPECT_NO_THROW(motion.begin_trial(1.2,0.1));
    motion.rollback_trial();
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
}
