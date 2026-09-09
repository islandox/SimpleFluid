/**
 * @file testMultiRegionOperators.cc
 * @author islandox
 * @brief Existing assembled FVM operators across canonical region seams.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#include <gtest/gtest.h>
#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<>;
using Scalar = ScalarCellFieldStored<Pack>;
using Flux = ScalarFaceFieldStored<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
real_t linear(MeshUtils::Vec3 p) { return 1 + 2*p.x - 0.75*p.y + 0.4*p.z; }

std::vector<real_t> solve(const auto& system, const SP<const Handle>& mesh)
{
    Pack::vector_type x(mesh->owned_cell_map(), true);
    BelosLinearSolver<> solver; LinearSolverOptions options; options.tolerance = 1e-12;
    EXPECT_TRUE(solver.solve(system.matrix, *system.rhs, x, options));
    const auto data = x.getData(); return {data.begin(), data.end()};
}
void expect_compact(const Handle& mesh)
{
    EXPECT_EQ(mesh.connectivity_storage_bytes(),0U);
    EXPECT_FALSE(mesh.has_materialized_indexer());
}
}

TEST(MultiRegionOperatorsTest, LinearGradientAndManufacturedDiffusionAcrossSeam)
{
    SP<const Handle> mesh = std::make_shared<Handle>(test::two_regions(3));
    Scalar scalar(mesh,"linear"); VectorCellFieldStored<Pack> gradient(mesh,"gradient");
    for (size_t c=0;c<mesh->num_cells();++c) scalar.set_owned_value(c,linear(mesh->cell_centroid(c)));
    scalar.sync_ghosts();
    auto boundary = [&](int b,size_t i) { return linear(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
    auto condition = [&](int b,size_t i) { return BoundaryCondition{BoundaryConditionType::Dirichlet,boundary(b,i)}; };
    FVM::CellGradientCache<Pack,Handle> cache(mesh);
    FVM::cell_gradient(scalar,condition,boundary,gradient,cache);
    for (size_t c=0;c<mesh->num_cells();++c)
        EXPECT_NEAR((gradient.value(c) - MeshUtils::Vec3{2,-0.75,0.4}).norm(),0,2e-12);
    const auto system = FVM::diffusion_system<Pack>(*mesh,1.0,condition,[](int){return 0.0;});
    const auto values = solve(system,mesh);
    for (size_t c=0;c<mesh->num_cells();++c) EXPECT_NEAR(values[c],linear(mesh->cell_centroid(c)),2e-10);
    expect_compact(*mesh);
}
TEST(MultiRegionOperatorsTest, NonOrthogonalMixedMeshMatchesExplicitReference)
{
    auto composite = test::mixed_regions();
    std::vector<SP<const Handle>> meshes{std::make_shared<Handle>(composite), std::make_shared<Handle>(test::explicit_reference(*composite))};
    std::vector<std::vector<real_t>> solutions;
    for (const auto& mesh : meshes)
    {
        Scalar scalar(mesh,"linear"); Flux flux(mesh,0.0,"flux");
        for (size_t c=0;c<mesh->num_cells();++c) scalar.set_owned_value(c,linear(mesh->cell_centroid(c)));
        scalar.sync_ghosts();
    auto boundary = [&](int b,size_t i) { return linear(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
        auto condition = [&](int b,size_t i) { return BoundaryCondition{BoundaryConditionType::Dirichlet,boundary(b,i)}; };
        FVM::TransportGeometryCache<Handle> cache(*mesh);
        const auto system = FVM::non_orthogonal_transport_system<Pack>(scalar,flux,0.2,0.5,condition,boundary,
            [](int){return 0.0;},FVM::NonOrthogonalTreatment::Implicit,nullptr,Teuchos::null,&cache);
        solutions.push_back(solve(system,mesh));
        for (size_t c=0;c<mesh->num_cells();++c) EXPECT_NEAR(solutions.back()[c],linear(mesh->cell_centroid(c)),2e-10);
        expect_compact(*mesh);
    }
    // Pair by physical centroid, independent of face or native cell ordering.
    for (size_t c=0;c<meshes[0]->num_cells();++c)
    {
        bool found = false;
        for (size_t d=0;d<meshes[1]->num_cells();++d)
            if ((meshes[0]->cell_centroid(c)-meshes[1]->cell_centroid(d)).norm()<1e-12)
            { EXPECT_NEAR(solutions[0][c],solutions[1][d],2e-10); found=true; }
        EXPECT_TRUE(found);
    }
}
TEST(MultiRegionOperatorsTest, CanonicalFluxAndTransportConserveVolumeWeightedInventory)
{
    auto composite = test::two_regions(); SP<const Handle> mesh = std::make_shared<Handle>(composite);
    Flux flux(mesh,0.0,"seam_flux"); Scalar old(mesh,"old");
    for (size_t c=0;c<mesh->num_cells();++c) old.set_owned_value(c,1+mesh->cell_centroid(c).x);
    for (size_t f=0;f<mesh->num_faces();++f)
        if (std::abs(mesh->face_centroid(f).x-1)<1e-12 && std::abs(mesh->face_normal(f).x)>0.5) flux.set_owned_value(f,0.125);
    old.sync_ghosts(); flux.sync_ghosts();
    const auto divergence = FVM::cell_divergence_from_fluxes(*mesh,flux);
    real_t balance = 0; for (size_t c=0;c<mesh->num_cells();++c) balance += divergence[c]*mesh->cell_volume(c);
    EXPECT_NEAR(balance,0,1e-14);
    const auto system = FVM::transport_system<Pack>(old,flux,0.1,0.0,
        [](int,size_t){return BoundaryCondition{BoundaryConditionType::Neumann,0};},
        [](int,size_t){return 0.0;},[](int){return 0.0;});
    const auto result = solve(system,mesh);
    real_t before=0,after=0;
    for (size_t c=0;c<mesh->num_cells();++c)
    { before+=old.value(c)*mesh->cell_volume(c); after+=result[c]*mesh->cell_volume(c); }
    EXPECT_NEAR(after,before,2e-12);
    EXPECT_GT(std::abs(result[0]-old.value(0))+std::abs(result[1]-old.value(1)),1e-5);
    EXPECT_EQ(flux.num_owned_faces(),mesh->num_faces());
    expect_compact(*mesh);
}
TEST(MultiRegionOperatorsTest, ConstantFieldAndUniformVelocityRemainConstant)
{
    SP<const Handle> mesh=std::make_shared<Handle>(test::two_regions());
    Scalar old(mesh,3.0,"constant"); Flux flux(mesh,"flux");
    for (size_t f=0;f<mesh->num_faces();++f) flux.set_owned_value(f,mesh->face_area_vector(f).x);
    old.sync_ghosts(); flux.sync_ghosts();
    const auto system=FVM::transport_system<Pack>(old,flux,0.1,0.2,
        [](int,size_t){return BoundaryCondition{BoundaryConditionType::Dirichlet,3};},
        [](int,size_t){return 3.0;},[](int){return 0.0;});
    const auto values=solve(system,mesh);
    for (const auto x:values) EXPECT_NEAR(x,3,2e-11);
    expect_compact(*mesh);
}
TEST(MultiRegionOperatorsTest, CachedOperatorsRejectNativeConstituentReplacement)
{
    using namespace Meshes;
    auto native=std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1,2},{0,1},{0,1}}});
    auto composite=std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("one",native)},std::vector<MultiRegionMesh::Interface>{});
    SP<const Handle> mesh=std::make_shared<Handle>(composite);
    Scalar scalar(mesh,1.0,"one"); VectorCellFieldStored<Pack> gradient(mesh,"gradient");
    FVM::CellGradientCache<Pack,Handle> cache(mesh);
    *native=OrthogonalCartesian3D(Vec3D<ArrReal>{{{0,1,3},{0,1},{0,1}}});
    EXPECT_THROW(FVM::cell_gradient(scalar,gradient,cache),std::logic_error);
}
