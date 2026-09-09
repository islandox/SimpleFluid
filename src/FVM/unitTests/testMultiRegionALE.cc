/**
 * @file testMultiRegionALE.cc
 * @author islandox
 * @brief Composite affine ALE through the existing motion and transport contracts.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#include <gtest/gtest.h>
#include "FVM/Operators.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"
namespace
{
using namespace SimpleFluid;
using Pack=DefaultTpetraTypes;
using Handle=MeshHandle<>;
testing::Environment* const environment=testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
}
TEST(MultiRegionALETest, AffineMotionPreservesConstantsAndCanonicalGcl)
{
    auto geometry=test::two_regions(); auto mesh=std::make_shared<Handle>(geometry);
    auto alias=std::make_shared<Handle>(std::static_pointer_cast<const Meshes::MultiRegionMesh>(geometry));
    const auto topology=geometry->regions();
    ScalarCellFieldStored<Pack> old(mesh,2.75,"old"), unit(mesh,1.0,"unit"), zero(mesh,0.0,"zero");
    FVM::TransportGeometryCache<Handle> cache(*mesh);
    PlanarALEMeshMotion<> motion(mesh);
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh)),std::logic_error);
    motion.begin_trial(1.3,0.2);
    EXPECT_EQ(alias->geometry_epoch(),mesh->geometry_epoch());
    EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12);
    auto ale=FVM::make_ale_control_volume_state(*mesh,motion);
    ScalarFaceFieldStored<Pack> absolute(mesh,0.0,"absolute"), relative(mesh,"relative");
    FVM::mesh_relative_face_fluxes(absolute,ale,relative);
    const auto system=FVM::weighted_scalar_transport_system<Pack>(FVM::MeshWeightedScalarTransportRequest<Pack,Handle>{
        .old_values=old,.face_fluxes=relative,.time_step=0.2,.storage_weight=unit,.advection_weight=unit,.diffusivity=zero,
        .boundary_condition=[](int,size_t){return BoundaryCondition{BoundaryConditionType::Dirichlet,2.75};},
        .boundary_value=[](int,size_t){return 2.75;},.source=[](int){return 0.0;},
        .treatment=FVM::NonOrthogonalTreatment::Explicit,.ale=&ale});
    Pack::vector_type result(mesh->owned_cell_map(),true); BelosLinearSolver<> solver;
    LinearSolverOptions options; options.tolerance=1e-12;
    EXPECT_TRUE(solver.solve(system.matrix,*system.rhs,result,options));
    const auto values=result.getData(); for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(values[c],2.75,2e-11);
    motion.rollback_trial(); EXPECT_EQ(mesh->geometry_epoch(),2U);
    for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(mesh->cell_volume(c),0.125,2e-14);
    motion.begin_trial(0.8,0.2); EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12); motion.accept_trial();
    EXPECT_EQ(mesh->connectivity_storage_bytes(),0U); EXPECT_FALSE(mesh->has_materialized_indexer());
    // Constituent coordinates stay immutable; only the composite geometry moves.
    std::visit([](const auto& r){EXPECT_EQ(r.geometry().geometry_epoch(),0U);},topology[0]);
}
TEST(MultiRegionALETest, InvalidMotionAndQualityFailureLeaveAcceptedGeometry)
{
    auto mesh=std::make_shared<Handle>(test::two_regions());
    PlanarALEMeshMotionOptions invalid; invalid.axis=Dimension::X;
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh,invalid)),std::invalid_argument);
    invalid.axis=Dimension::Z; invalid.deformation_start_elevation=0.4;
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh,invalid)),std::invalid_argument);
    PlanarALEMeshMotionOptions options; options.quality_limits.maximum_aspect_ratio=2.0;
    PlanarALEMeshMotion<> motion(mesh,options);
    EXPECT_THROW(motion.begin_trial(5.0,0.2),std::runtime_error);
    EXPECT_FALSE(motion.has_active_trial());
    for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(mesh->cell_volume(c),0.125,2e-14);
    motion.begin_trial(1.1,0.2); motion.accept_trial();
}

TEST(MultiRegionALETest, ExtendedFamiliesPreserveAffineGcl)
{
    for(const auto& geometry:{test::coarse_fine_regions(),test::periodic_regions(),test::cylindrical_regions(),test::independent_extruded_regions()})
    {
        auto mesh=std::make_shared<Handle>(geometry);
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.25,0.2);
        EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12);
        motion.rollback_trial();
        motion.begin_trial(0.75,0.2);
        EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12);
        motion.accept_trial();
        EXPECT_EQ(mesh->connectivity_storage_bytes(),0U);
    }
}
TEST(MultiRegionALETest, ChangingThePeriodicLengthIsRejected)
{
    using namespace Meshes;
    StructuredPatchInterface periodic{{0,4},{0,5}}; periodic.periodic_translation=MeshUtils::Vec3{0,0,1};
    auto geometry=std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{
        cartesian_region("periodic",{{{0,1},{0,1},{0,0.5,1}}})},std::vector<MultiRegionMesh::Interface>{periodic});
    auto mesh=std::make_shared<Handle>(geometry);
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh)),std::invalid_argument);
}
