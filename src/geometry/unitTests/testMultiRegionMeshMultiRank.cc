/**
 * @file testMultiRegionMeshMultiRank.cc
 * @author islandox
 * @brief Compact composite ownership, halo and collective-failure checks.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#include <gtest/gtest.h>
#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "fields/FieldStored.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <set>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Handle = MeshHandle<>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
}
TEST(MultiRegionMeshMultiRankTest, OwnershipHalosAndCanonicalFaces)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1) GTEST_SKIP() << "Requires MPI ranks.";
    const auto geometry = test::two_regions(2);
    SP<const Handle> mesh = std::make_shared<Handle>(geometry);
    EXPECT_EQ(mesh->owned_cell_map()->getGlobalNumElements(), geometry->num_cells());
    EXPECT_EQ(mesh->owned_face_map()->getGlobalNumElements(), geometry->num_faces());
    EXPECT_EQ(mesh->num_owned_cells(), geometry->num_cells() / comm->getSize());
    EXPECT_FALSE(mesh->has_materialized_indexer()); EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
    ScalarCellFieldStored<> field(mesh, "gid");
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c) field.set_owned_value(c, mesh->cell_global_id(c) + 0.5);
    field.sync_ghosts();
    for (size_t c = 0; c < mesh->num_cells(); ++c)
    {
        EXPECT_DOUBLE_EQ(field.local_value(c), mesh->cell_global_id(c) + 0.5);
        for (const auto f : mesh->faces(c))
        {
            const auto global_face = mesh->face_global_id(f);
            EXPECT_EQ(mesh->cell_global_id(c), geometry->owner_cell(global_face) == static_cast<uint64_t>(mesh->cell_global_id(c))
                ? geometry->owner_cell(global_face) : geometry->neighbor_cell(global_face));
            if (mesh->is_owned_cell(c) && !mesh->is_geometry_exterior_face(f))
                EXPECT_NE(mesh->opposite_cell(f, c), Handle::invalid_local_id());
        }
    }
    EXPECT_EQ(mesh->vtu_topology()->cell_offsets.size(), mesh->num_owned_cells());
}
TEST(MultiRegionMeshMultiRankTest, MultipleRegionsPerRank)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1) GTEST_SKIP() << "Requires MPI ranks.";
    std::vector<MultiRegionMesh::Region> regions;
    std::vector<MultiRegionMesh::Interface> interfaces;
    for (int r = 0; r < 2 * comm->getSize(); ++r)
    {
        regions.push_back(cartesian_region(std::to_string(r), {{{real_t(r),real_t(r+1)},{0,1},{0,1}}}));
        if (r) interfaces.push_back(StructuredPatchInterface{{static_cast<size_t>(r-1),1},{static_cast<size_t>(r),0}});
    }
    auto geometry = std::make_shared<MultiRegionMesh>(std::move(regions),std::move(interfaces));
    Handle mesh(geometry);
    std::set<size_t> owned_regions;
    for(size_t c=0;c<mesh.num_owned_cells();++c) owned_regions.insert(geometry->native_cell(mesh.cell_global_id(c)).first);
    EXPECT_EQ(owned_regions.size(),2U);
    EXPECT_EQ(mesh.connectivity_storage_bytes(),0U);
}
TEST(MultiRegionMeshMultiRankTest, RankDivergentGeometryAndInterfacesFailCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1) GTEST_SKIP() << "Requires MPI ranks.";
    const real_t shift = comm->getRank() == 0 ? 0.0 : 0.1;
    std::vector<MultiRegionMesh::Region> regions{
        cartesian_region("a",{{{shift,shift+1},{0,1},{0,1}}}),
        cartesian_region("b",{{{shift+1,shift+2},{0,1},{0,1}}})};
    EXPECT_THROW((MultiRegionMesh(regions,{StructuredPatchInterface{{0,1},{1,0}}})),std::invalid_argument);
    auto interfaces = std::vector<MultiRegionMesh::Interface>{StructuredPatchInterface{{0,1},{1,0}}};
    if(comm->getRank()==0) interfaces.push_back(interfaces.front());
    EXPECT_THROW((MultiRegionMesh(regions,interfaces)),std::invalid_argument);
}

TEST(MultiRegionMeshMultiRankTest, ExplicitGlobalRegionsAndStaleChildrenFailBeforeMaps)
{
    const auto comm=Tpetra::getDefaultComm();
    if(comm->getSize()==1) GTEST_SKIP()<<"Requires MPI ranks.";
    auto source=test::two_regions(1);
    auto explicit_geometry=test::explicit_reference(*source);
    EXPECT_THROW((MultiRegionMesh({native_region("explicit",explicit_geometry)},{})),std::invalid_argument);
    auto native=std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1,2,3,4},{0,1},{0,1}}});
    auto geometry=std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("native",native)},
        std::vector<MultiRegionMesh::Interface>{});
    if(comm->getRank()==0) *native=OrthogonalCartesian3D(Vec3D<ArrReal>{{{0,1,2,3,5},{0,1},{0,1}}});
    EXPECT_THROW((Handle(geometry)),std::logic_error);
}

TEST(MultiRegionMeshMultiRankTest, HaloOptionsPreserveMutableGeometryOwnership)
{
    const auto comm=Tpetra::getDefaultComm(); if(comm->getSize()==1) GTEST_SKIP()<<"Requires MPI ranks.";
    const auto geometry=test::two_regions();
    Handle mutable_handle(geometry,Handle::DistributionOptions{.ghost_layers=2});
    Handle observer(std::static_pointer_cast<const MultiRegionMesh>(geometry),Handle::DistributionOptions{.ghost_layers=2});
    EXPECT_TRUE(mutable_handle.has_mutable_geometry()); EXPECT_FALSE(observer.has_mutable_geometry());
    EXPECT_EQ(mutable_handle.geometry_identity(),observer.geometry_identity());
    EXPECT_EQ(mutable_handle.num_cells(),observer.num_cells());
    EXPECT_EQ(mutable_handle.connectivity_storage_bytes(),0U);
}
