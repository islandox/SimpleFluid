/**
 * @file testMultiRegionFlow.cc
 * @author islandox
 * @brief Existing segregated and coupled flow paths on connected composite domains.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#include <gtest/gtest.h>
#include "solvers/FluidSolver.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"
namespace
{
using namespace SimpleFluid;
testing::Environment* const environment=testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
void run_flow(PressureVelocityCoupling mode)
{
    SP<const MeshHandle<>> composite=std::make_shared<MeshHandle<>>(test::two_regions());
    SP<const MeshHandle<>> single=std::make_shared<MeshHandle<>>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0,0.5,1,1.5,2},{0,0.5,1},{0,0.5,1}}}));
    std::vector<SP<const MeshHandle<>>> meshes{composite,single};
    std::vector<std::vector<real_t>> pressure;
    std::vector<std::vector<MeshUtils::Vec3>> velocity;
    for (const auto& mesh:meshes)
    {
        BoundaryConditionSet bc;
        for (const auto& [b,batch]:mesh->boundary_batches())
        {
            const auto name=mesh->boundary_batch_name(b);
            bc.velocity[name]={BoundaryConditionType::Dirichlet,{1,0,0}};
        }
        TimeStepperOptions options; options.time_step=0.01; options.kinematic_viscosity=0.1;
        options.pressure_velocity_coupling=mode;
        FluidSolver<> solver(mesh,bc,options);
        // A nonuniform predictor forces projection to communicate across the seam.
        for (size_t c=0;c<mesh->num_cells();++c)
            solver.velocity().set_owned_value(c,{1+0.05*std::sin(mesh->cell_centroid(c).x),0,0});
        solver.step();
        EXPECT_LT(solver.last_volume_continuity_residuals().maximum,1e-7);
        pressure.emplace_back(); velocity.emplace_back();
        for (size_t c=0;c<mesh->num_cells();++c)
        { pressure.back().push_back(solver.pressure().value(c)); velocity.back().push_back(solver.velocity().value(c)); }
        EXPECT_EQ(mesh->connectivity_storage_bytes(),0U); EXPECT_FALSE(mesh->has_materialized_indexer());
    }
    for (size_t c=0;c<composite->num_cells();++c)
        for (size_t d=0;d<single->num_cells();++d)
            if ((composite->cell_centroid(c)-single->cell_centroid(d)).norm()<1e-12)
            {
                EXPECT_NEAR(pressure[0][c]-pressure[0][0],pressure[1][d]-pressure[1][0],2e-7);
                EXPECT_NEAR((velocity[0][c]-velocity[1][d]).norm(),0,2e-8);
            }
}
}
TEST(MultiRegionFlowTest,PisoMatchesSingleMesh) { run_flow(PressureVelocityCoupling::PISO); }
TEST(MultiRegionFlowTest,CoupledMatchesSingleMesh) { run_flow(PressureVelocityCoupling::CoupledKrylov); }
