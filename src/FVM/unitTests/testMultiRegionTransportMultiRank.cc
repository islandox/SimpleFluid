/**
 * @file testMultiRegionTransportMultiRank.cc
 * @author islandox
 * @brief Distributed composite reconstruction, diffusion and conservation.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#include <gtest/gtest.h>
#include "FVM/Operators.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
namespace
{
using namespace SimpleFluid;
using Pack=DefaultTpetraTypes;
using Handle=MeshHandle<>;
testing::Environment* const environment=testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
real_t analytic(MeshUtils::Vec3 p) { return 1+p.x+2*p.y-0.5*p.z; }
}
TEST(MultiRegionTransportMultiRankTest, LinearGradientAndDiffusionAcrossRankAndRegionSeams)
{
    if(Tpetra::getDefaultComm()->getSize()==1) GTEST_SKIP()<<"Requires MPI ranks.";
    SP<const Handle> mesh=std::make_shared<Handle>(test::two_regions());
    ScalarCellFieldStored<Pack> scalar(mesh,"phi"); VectorCellFieldStored<Pack> gradient(mesh,"gradient");
    for(size_t c=0;c<mesh->num_owned_cells();++c) scalar.set_owned_value(c,analytic(mesh->cell_centroid(c)));
    scalar.sync_ghosts();
    auto boundary=[&](int b,size_t i) { return analytic(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
    auto condition=[&](int b,size_t i) { return BoundaryCondition{BoundaryConditionType::Dirichlet,boundary(b,i)}; };
    FVM::cell_gradient(scalar,condition,boundary,gradient);
    for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR((gradient.value(c)-MeshUtils::Vec3{1,2,-0.5}).norm(),0,2e-11);
    auto system=FVM::diffusion_system<Pack>(*mesh,1.0,condition,[](int){return 0.0;});
    Pack::vector_type result(mesh->owned_cell_map(),true);
    BelosLinearSolver<> solver; LinearSolverOptions options; options.tolerance=1e-12;
    ASSERT_TRUE(solver.solve(system.matrix,*system.rhs,result,options));
    const auto values=result.getData();
    for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(values[c],analytic(mesh->cell_centroid(c)),2e-10);
    EXPECT_EQ(mesh->connectivity_storage_bytes(),0U); EXPECT_FALSE(mesh->has_materialized_indexer());
}
TEST(MultiRegionTransportMultiRankTest, SeamFluxPreservesGlobalScalarInventory)
{
    const auto comm=Tpetra::getDefaultComm();
    if(comm->getSize()==1) GTEST_SKIP()<<"Requires MPI ranks.";
    SP<const Handle> mesh=std::make_shared<Handle>(test::two_regions());
    ScalarCellFieldStored<Pack> old(mesh,"old"); ScalarFaceFieldStored<Pack> flux(mesh,0.0,"flux");
    for(size_t c=0;c<mesh->num_owned_cells();++c) old.set_owned_value(c,1+mesh->cell_centroid(c).x);
    for(size_t f=0;f<mesh->num_owned_faces();++f)
        if(std::abs(mesh->face_centroid(f).x-1)<1e-12 && std::abs(mesh->face_normal(f).x)>0.5) flux.set_owned_value(f,0.125);
    old.sync_ghosts(); flux.sync_ghosts();
    auto system=FVM::transport_system<Pack>(old,flux,0.1,0.0,
        [](int,size_t){return BoundaryCondition{BoundaryConditionType::Neumann,0};},
        [](int,size_t){return 0.0;},[](int){return 0.0;});
    Pack::vector_type result(mesh->owned_cell_map(),true); BelosLinearSolver<> solver;
    LinearSolverOptions options; options.tolerance=1e-12;
    ASSERT_TRUE(solver.solve(system.matrix,*system.rhs,result,options));
    const auto values=result.getData();
    real_t local[2]{},global[2]{};
    for(size_t c=0;c<mesh->num_owned_cells();++c)
    { local[0]+=mesh->cell_volume(c)*old.value(c); local[1]+=mesh->cell_volume(c)*values[c]; }
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,2,local,global);
    EXPECT_NEAR(global[0],global[1],2e-11);
    EXPECT_EQ(mesh->connectivity_storage_bytes(),0U); EXPECT_FALSE(mesh->has_materialized_indexer());
}

TEST(MultiRegionTransportMultiRankTest, ExtendedCompactFamiliesRunDistributedDiffusion)
{
    if(Tpetra::getDefaultComm()->getSize()==1) GTEST_SKIP()<<"Requires MPI ranks.";
    for(const auto& geometry:{test::coarse_fine_regions(),test::periodic_regions(),test::cylindrical_regions(),test::independent_extruded_regions()})
    {
        SP<const Handle> mesh=std::make_shared<Handle>(geometry);
        auto condition=[&](int b,size_t i){return BoundaryCondition{BoundaryConditionType::Dirichlet,
            mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i]).z};};
        const auto system=FVM::diffusion_system<Pack>(*mesh,1.0,condition,[](int){return 0.0;});
        Pack::vector_type result(mesh->owned_cell_map(),true); BelosLinearSolver<> solver;
        LinearSolverOptions options; options.tolerance=1e-12;
        ASSERT_TRUE(solver.solve(system.matrix,*system.rhs,result,options));
        const auto values=result.getData();
        for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(values[c],mesh->cell_centroid(c).z,2e-10);
        EXPECT_EQ(mesh->connectivity_storage_bytes(),0U); EXPECT_FALSE(mesh->has_materialized_indexer());
    }
}

TEST(MultiRegionTransportMultiRankTest, RanksWithoutOwnedCellsParticipateInDiffusion)
{
    if(Tpetra::getDefaultComm()->getSize()<4) GTEST_SKIP()<<"Requires more ranks than cells.";
    SP<const Handle> mesh=std::make_shared<Handle>(test::two_regions(1));
    auto condition=[&](int b,size_t i){return BoundaryCondition{BoundaryConditionType::Dirichlet,
        mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i]).x};};
    const auto system=FVM::diffusion_system<Pack>(*mesh,1.0,condition,[](int){return 0.0;});
    Pack::vector_type result(mesh->owned_cell_map(),true); BelosLinearSolver<> solver;
    ASSERT_TRUE(solver.solve(system.matrix,*system.rhs,result));
    const auto values=result.getData();
    for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(values[c],mesh->cell_centroid(c).x,2e-10);
}

TEST(MultiRegionTransportMultiRankTest, NonconformingSeamTransportConservesInventory)
{
    const auto comm=Tpetra::getDefaultComm(); if(comm->getSize()==1) GTEST_SKIP()<<"Requires MPI ranks.";
    SP<const Handle> mesh=std::make_shared<Handle>(test::coarse_fine_regions());
    ScalarCellFieldStored<Pack> old(mesh,"old"); ScalarFaceFieldStored<Pack> flux(mesh,0.0,"flux");
    for(size_t c=0;c<mesh->num_owned_cells();++c) old.set_owned_value(c,1+mesh->cell_centroid(c).x);
    for(size_t f=0;f<mesh->num_owned_faces();++f)
        if(std::abs(mesh->face_centroid(f).x-1)<1e-12 && std::abs(mesh->face_normal(f).x)>0.5)
            flux.set_owned_value(f,0.25*mesh->face_area_vector(f).x);
    old.sync_ghosts(); flux.sync_ghosts();
    auto system=FVM::transport_system<Pack>(old,flux,0.1,0.0,
        [](int,size_t){return BoundaryCondition{BoundaryConditionType::Neumann,0};},
        [](int,size_t){return 0.0;},[](int){return 0.0;});
    Pack::vector_type result(mesh->owned_cell_map(),true); BelosLinearSolver<> solver;
    ASSERT_TRUE(solver.solve(system.matrix,*system.rhs,result));
    const auto values=result.getData(); real_t local[3]{},global[3]{};
    for(size_t c=0;c<mesh->num_owned_cells();++c)
    {
        local[0]+=old.value(c)*mesh->cell_volume(c); local[1]+=values[c]*mesh->cell_volume(c);
        local[2]+=std::abs(values[c]-old.value(c));
    }
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,3,local,global);
    EXPECT_NEAR(global[0],global[1],2e-10); EXPECT_GT(global[2],1e-4);
    EXPECT_EQ(mesh->connectivity_storage_bytes(),0U);
}
