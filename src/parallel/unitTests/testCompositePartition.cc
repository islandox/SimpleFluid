/** @file testCompositePartition.cc
 *  @brief Native composite ownership, transport, and collective validation tests.
 */
#include <gtest/gtest.h>
#include "parallel/MeshPartitioner.hh"
#include "geometry/MeshHandle.hh"
#include "fields/FieldStored.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/unitTests/composite_partition_helpers.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Partitioner = MeshPartitioner<DefaultTpetraTypes>;
using Handle = MeshHandle<>;
using ID = MultiRegionMesh::ID;
using Policy = CompositePartitionOptions::Policy;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

Teuchos::RCP<const Teuchos::Comm<int>> serial_comm()
{ return Teuchos::rcp(new Teuchos::SerialComm<int>); }


std::shared_ptr<MultiRegionMesh> one_box(unsigned count)
{
    ArrReal x(count+1); std::iota(x.begin(),x.end(),0.0);
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{cartesian_region("box",{{x,{0,1},{0,1}}})},
        std::vector<MultiRegionMesh::Interface>{}, serial_comm());
}

void check_partition(const CompositePartition<>& partition, unsigned layers)
{
    const auto& plan = partition.plan();
    const auto& geometry = *partition.geometry();
    auto handle = std::make_shared<Handle>(partition);
    EXPECT_EQ(handle->num_owned_cells(),plan.owned_cells().size());
    EXPECT_EQ(handle->num_cells(),plan.cell_ids().size());
    EXPECT_EQ(handle->owned_cell_map()->getGlobalNumElements(),3*layers);
    EXPECT_EQ(handle->owned_face_map()->getGlobalNumElements(),plan.global_faces());
    EXPECT_EQ(handle->connectivity_storage_bytes(),0U);
    EXPECT_TRUE(handle->supports_region_execution());
    EXPECT_EQ(geometry.region_layout(0).family,RegionLayout::Family::Rectilinear);
    EXPECT_EQ(geometry.region_layout(1).family,RegionLayout::Family::Extruded);
    EXPECT_EQ(geometry.region_layout(2).family,RegionLayout::Family::Explicit);
    ScalarCellFieldStored<> cell_values(handle,"canonical_cell");
    ScalarFaceFieldStored<> face_values(handle,"canonical_face");
    for (size_t local = 0; local < handle->num_owned_cells(); ++local)
        cell_values.set_owned_value(local,handle->cell_global_id(local)+0.25);
    for (size_t local = 0; local < handle->num_owned_faces(); ++local)
        face_values.set_owned_value(local,handle->face_global_id(local)+0.5);
    cell_values.sync_ghosts(); face_values.sync_ghosts();
    std::set<ID> visited;
    handle->visit_owned_region_cells([&](auto lid, ID canonical, ID native, const auto& region)
    {
        EXPECT_EQ(handle->cell_global_id(lid),canonical);
        EXPECT_TRUE(visited.insert(canonical).second);
        EXPECT_DOUBLE_EQ(region.cell_volume(native),handle->cell_volume(lid));
    });
    EXPECT_EQ(visited.size(),handle->num_owned_cells());
    for (size_t local = 0; local < handle->num_cells(); ++local)
    {
        const auto global = handle->cell_global_id(local);
        EXPECT_EQ(global,plan.cell_ids()[local]);
        EXPECT_DOUBLE_EQ(cell_values.local_value(local),global+0.25);
        const auto [region,native] = geometry.native_cell(global);
        EXPECT_NEAR(handle->cell_volume(local),region == 2 ? 1.1 : 1.0,1e-13);
        EXPECT_NEAR(handle->cell_centroid(local).z,native+0.5,1e-13);
        EXPECT_EQ(geometry.cell_type(global),region == 2 ? MeshUtils::CellType::POLYHEDRON : MeshUtils::CellType::HEXAHEDRON);
        if (region == 2) EXPECT_EQ(geometry.cell_faces(global).size(),7U);
        for (const auto face : handle->faces(local))
        {
            if (handle->is_owned_cell(local) && !handle->is_geometry_exterior_face(face))
                EXPECT_NE(handle->opposite_cell(face,local),Handle::invalid_local_id());
            const auto global_face = handle->face_global_id(face);
            EXPECT_TRUE(geometry.owner_cell(global_face) == static_cast<ID>(global)
                || geometry.neighbor_cell(global_face) == static_cast<ID>(global));
        }
    }
    for (size_t local = 0; local < handle->num_faces(); ++local)
        EXPECT_DOUBLE_EQ(face_values.local_value(local),handle->face_global_id(local)+0.5);
    const auto output = handle->vtu_topology();
    EXPECT_EQ(output->cell_offsets.size(),handle->num_owned_cells());
}
}

TEST(CompositePartitionTest, MixedFamiliesKeepCanonicalGeometryAndFieldCommunication)
{
    const auto comm = Tpetra::getDefaultComm();
    const int source_rank = comm->getSize()-1;
    CompositeMeshSource source(comm->getRank() == source_rank ? test::mixed_partition_source() : nullptr,source_rank);
    for (auto policy : {Policy::TopologyAware,Policy::CellGraph,Policy::ContiguousCanonical})
    {
        CompositePartitionOptions options; options.policy = policy; options.ghost_layers = 2;
        const auto plan = Partitioner::make_plan(source,options,comm);
        const auto partition = Partitioner::distribute(source,plan);
        check_partition(partition,8);
        if (policy == Policy::TopologyAware)
            for (const auto& fragment : plan.fragments())
            {
                if (fragment.region == 0) EXPECT_TRUE(std::holds_alternative<CompositeRegionFragment::OrthogonalBox>(fragment.selection));
                if (fragment.region == 1) EXPECT_TRUE(std::holds_alternative<CompositeRegionFragment::ExtrudedProduct>(fragment.selection));
                if (fragment.region == 2) EXPECT_TRUE(std::holds_alternative<CompositeRegionFragment::ExplicitCells>(fragment.selection));
            }
    }
}

TEST(CompositePartitionTest, EmptyRanksKeepGlobalCatalogAndEmptyFields)
{
    const auto comm = Tpetra::getDefaultComm();
    CompositeMeshSource source(comm->getRank() == 0 ? one_box(1) : nullptr);
    const auto partition = Partitioner::partition(source,{},comm);
    auto handle = std::make_shared<Handle>(partition);
    EXPECT_EQ(handle->num_owned_cells(),comm->getRank() == 0 ? 1U : 0U);
    EXPECT_EQ(handle->owned_cell_map()->getGlobalNumElements(),1U);
    ScalarCellFieldStored<> values(handle,"empty_rank"); values.sync_ghosts();
    EXPECT_EQ(handle->vtu_topology()->cell_offsets.size(),handle->num_owned_cells());
}

TEST(CompositePartitionTest, PlanRejectsDifferentSourceAndInvalidCollectiveOptions)
{
    const auto comm = Tpetra::getDefaultComm();
    CompositeMeshSource source(comm->getRank() == 0 ? one_box(8) : nullptr);
    const auto plan = Partitioner::make_plan(source,{},comm);
    CompositeMeshSource changed(comm->getRank() == 0 ? one_box(8) : nullptr);
    EXPECT_THROW(Partitioner::distribute(changed,plan),std::invalid_argument);
    CompositePartitionOptions invalid; invalid.ghost_layers = comm->getRank() == 0 ? 0 : 1;
    EXPECT_THROW(Partitioner::make_plan(source,invalid,comm),std::invalid_argument);
    if (comm->getSize() > 1)
    {
        CompositePartitionOptions different; different.target_cells_per_fragment = comm->getRank() == 0 ? 32 : 64;
        EXPECT_THROW(Partitioner::make_plan(source,different,comm),std::invalid_argument);
    }
}

TEST(CompositePartitionTest, SourceRevisionChangeInvalidatesPlanBeforeDistribution)
{
    const auto comm = Tpetra::getDefaultComm();
    std::shared_ptr<OrthogonalCartesian3D> native;
    std::shared_ptr<MultiRegionMesh> source_mesh;
    if (comm->getRank() == 0)
    {
        native = std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1,2},{0,1},{0,1}}});
        source_mesh = std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("native",native)},
            std::vector<MultiRegionMesh::Interface>{},serial_comm());
    }
    CompositeMeshSource source(source_mesh);
    const auto plan = Partitioner::make_plan(source,{},comm);
    if (native) *native = OrthogonalCartesian3D(Vec3D<ArrReal>{{{0,1,3},{0,1},{0,1}}});
    EXPECT_THROW(Partitioner::distribute(source,plan),std::exception);
}

TEST(CompositePartitionTest, IndependentCommunicatorControlsOwnership)
{
    const auto world = Tpetra::getDefaultComm();
    const auto comm = world->split(world->getRank()%2,world->getRank());
    CompositeMeshSource source(comm->getRank() == 0 ? one_box(8) : nullptr);
    const auto partition = Partitioner::partition(source,{},comm);
    Handle handle(partition);
    EXPECT_EQ(handle.owned_cell_map()->getComm()->getSize(),comm->getSize());
    EXPECT_EQ(handle.owned_cell_map()->getComm()->getRank(),comm->getRank());
    EXPECT_EQ(handle.owned_cell_map()->getGlobalNumElements(),8U);
}

TEST(CompositePartitionTest, CanonicalCoarseFineAndPeriodicIncidenceSurvivesGraphPartition)
{
    const auto comm = Tpetra::getDefaultComm();
    for (auto reference : {test::coarse_fine_regions(),test::periodic_regions(),test::cylindrical_regions()})
    {
        CompositeMeshSource source(comm->getRank() == 0 ? reference : nullptr);
        const auto partition = Partitioner::partition(source,{},comm);
        auto handle = std::make_shared<Handle>(partition);
        EXPECT_EQ(handle->owned_cell_map()->getGlobalNumElements(),reference->num_cells());
        EXPECT_EQ(handle->owned_face_map()->getGlobalNumElements(),reference->num_faces());
        for (size_t local = 0; local < handle->num_owned_cells(); ++local)
        {
            const ID global = handle->cell_global_id(local);
            std::vector<ID> observed;
            for (const auto face : handle->faces(local))
            {
                const ID canonical = handle->face_global_id(face);
                observed.push_back(canonical);
                EXPECT_EQ(partition.geometry()->owner_cell(canonical),reference->owner_cell(canonical));
                EXPECT_EQ(partition.geometry()->neighbor_cell(canonical),reference->neighbor_cell(canonical));
                EXPECT_NEAR((handle->face_center_vector(face,local)-reference->face_center_vector(canonical,global)).norm(),0,1e-13);
                if (!reference->is_exterior_face(canonical))
                {
                    EXPECT_NE(handle->opposite_cell(face,local),Handle::invalid_local_id());
                    EXPECT_NEAR((handle->cell_center_vector(face,local)-reference->cell_center_vector(canonical,global)).norm(),0,1e-13);
                }
            }
            const auto native = reference->cell_faces(global);
            EXPECT_EQ(observed,std::vector<ID>(native.begin(),native.end()));
        }
        double local_flux = 0;
        for (size_t local = 0; local < handle->num_owned_cells(); ++local)
            for (auto face : handle->faces(local))
            {
                const auto global_face = handle->face_global_id(face);
                if (reference->is_exterior_face(global_face)) continue;
                local_flux += (handle->cell_global_id(local) == static_cast<DefaultTpetraTypes::global_ordinal_type>(reference->owner_cell(global_face)) ? 1 : -1)
                    * (global_face+0.125);
            }
        double global_flux = 0;
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&local_flux,&global_flux);
        EXPECT_DOUBLE_EQ(global_flux,0);
    }
}

TEST(CompositePartitionTest, ReportsActualBalanceAndCanRequireFeasibleTolerance)
{
    const auto comm = Tpetra::getDefaultComm();
    CompositeMeshSource source(comm->getRank() == 0 ? one_box(1) : nullptr);
    const auto plan = Partitioner::make_plan(source,{},comm);
    EXPECT_EQ(plan.balance().total_cells,1U);
    EXPECT_EQ(plan.balance().max_owned_cells,1U);
    EXPECT_DOUBLE_EQ(plan.balance().measured_imbalance,comm->getSize());
    EXPECT_DOUBLE_EQ(plan.balance().minimum_imbalance_bound,comm->getSize());
    CompositePartitionOptions strict; strict.require_balance = true;
    if (comm->getSize() > 1) EXPECT_THROW(Partitioner::make_plan(source,strict,comm),std::invalid_argument);
    else EXPECT_NO_THROW(Partitioner::make_plan(source,strict,comm));
}
