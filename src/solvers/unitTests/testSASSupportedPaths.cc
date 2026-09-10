/** @file testSASSupportedPaths.cc @brief Transient and transaction gates for SAS extensions. */
#include <gtest/gtest.h>
#include "solvers/IncompressibleIsothermalSolver.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"
#include <numbers>

namespace
{
using namespace SimpleFluid;
using Pack=DefaultTpetraTypes;
using Handle=MeshHandle<Pack>;
auto* const environment=testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TimeStepperOptions transient_options()
{
    TimeStepperOptions result;
    result.time_step=1e-3;
    result.kinematic_viscosity=.001;
    result.thermal_expansion=0;
    result.gravity_z=0;
    result.n_pressure_correctors=2;
    result.non_orthogonal_treatment=FVM::NonOrthogonalTreatment::Explicit;
    return result;
}

TurbulenceModelOptions sas_options()
{
    TurbulenceModelOptions result;
    result.model=TurbulenceModelType::SSTKOmegaSAS;
    result.initial_turbulent_kinetic_energy=.2;
    result.initial_specific_dissipation_rate=2;
    result.initial_wall_distance=.25;
    result.sas.diagnostics=true;
    return result;
}

BoundaryConditionSet closed_boundaries(const Handle& mesh)
{
    BoundaryConditionSet result;
    for (const auto& [batch,faces] : mesh.boundary_batches())
        result.velocity[mesh.boundary_batch_name(batch)]={BoundaryConditionType::NoSlip,{}};
    return result;
}

template<class Solver> void expect_active_bounded_step(Solver& solver)
{
    ASSERT_NO_THROW(solver.step());
    const auto& model=*solver.find_turbulence_model();
    ASSERT_TRUE(model.sas_statistics().valid);
    EXPECT_GT(model.sas_statistics().max_source,1e-6);
    EXPECT_LT(solver.last_pressure_velocity_residuals().continuity,1e-6);
    for (size_t i=0; i<solver.velocity().mesh().num_owned_cells(); ++i)
    {
        EXPECT_GT(model.turbulent_kinetic_energy().value(i),0);
        EXPECT_GT(model.specific_dissipation_rate()->value(i),0);
        EXPECT_TRUE(std::isfinite(model.specific_dissipation_rate()->value(i)));
    }
}
}

TEST(SASSupportedPathsTest, CylindricalCartesianComponentsAdvanceTransientSwirl)
{
    ArrReal theta(17);
    for (size_t i=0; i<theta.size(); ++i) theta[i]=2*std::numbers::pi*i/16;
    SP<const Handle> mesh=std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCylindrial3D>(
        Vec3D<ArrReal>{{{1.,1.25,1.5,1.75,2.},theta,{0.,.25,.5,.75,1.}}}));
    for (bool slip : {false,true})
    {
        auto boundaries=closed_boundaries(*mesh);
        if (slip)
            for (auto& [name,condition] : boundaries.velocity) condition.type=BoundaryConditionType::Slip;
        IncompressibleIsothermalSolver<Pack> solver(mesh,boundaries,transient_options());
        solver.configure_turbulence(sas_options());
        for (size_t i=0; i<mesh->num_owned_cells(); ++i)
        {
            const auto p=mesh->cell_centroid(i);
            const auto amplitude=std::sin(std::numbers::pi*(std::hypot(p.x,p.y)-1))
                                 *std::sin(std::numbers::pi*p.z);
            solver.velocity().set_owned_value(i,{-p.y*amplitude,p.x*amplitude,0});
        }
        solver.velocity().sync_ghosts();
        expect_active_bounded_step(solver);
        expect_active_bounded_step(solver);
    }
}

TEST(SASSupportedPathsTest, SemiStructuredPrismsAdvanceWithoutLegacyMesh)
{
    if (Tpetra::getDefaultComm()->getSize()!=1) GTEST_SKIP()<<"Existing serial-only semi-structured backend";
    using Semi=Meshes::SemiStructuredXY_Z;
    auto geometry=std::make_shared<Semi>(
        Arr<Semi::Vec3>{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{.4,.4,0}},
        Arr<Arr<unsigned>>{{0,1,4},{1,2,4},{2,3,4},{3,0,4}},ArrReal{0,.25,.5,.75,1});
    SP<const Handle> mesh=std::make_shared<Handle>(geometry);
    ASSERT_FALSE(mesh->legacy_mesh());
    IncompressibleIsothermalSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
    solver.configure_turbulence(sas_options());
    for (size_t i=0; i<mesh->num_owned_cells(); ++i)
    {
        const auto p=mesh->cell_centroid(i);
        solver.velocity().set_owned_value(i,{std::sin(std::numbers::pi*p.z)*p.y,0,0});
    }
    solver.velocity().sync_ghosts();
    expect_active_bounded_step(solver);
    expect_active_bounded_step(solver);
}

namespace
{
SP<const Handle> box_mesh()
{
    return std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0.,.25,.5,.75,1.},{0.,.25,.5,.75,1.},{0.,.25,.5,.75,1.}}}));
}

template<class Solver> void initialize_circulation(Solver& solver)
{
    const auto& mesh=solver.velocity().mesh();
    for (size_t i=0; i<mesh.num_owned_cells(); ++i)
    {
        const auto p=mesh.cell_centroid(i); const auto pi=std::numbers::pi;
        solver.velocity().set_owned_value(i,{std::sin(pi*p.x)*std::sin(pi*p.x)*std::sin(2*pi*p.y),
            -std::sin(2*pi*p.x)*std::sin(pi*p.y)*std::sin(pi*p.y),0});
    }
    solver.velocity().sync_ghosts();
}
}

TEST(SASSupportedPathsTest, SlipWallsAdvanceWithNonzeroSourceAndContinuity)
{
    auto mesh=box_mesh(); auto bc=closed_boundaries(*mesh);
    bc.velocity["zmin"]={BoundaryConditionType::Slip,{}};
    bc.velocity["zmax"]={BoundaryConditionType::Slip,{}};
    IncompressibleIsothermalSolver<Pack> solver(mesh,bc,transient_options());
    solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    expect_active_bounded_step(solver); expect_active_bounded_step(solver);
}

TEST(SASSupportedPathsTest, PeriodicNativeFlowAdvancesWithNonzeroSource)
{
    ArrReal edges(9); for(size_t i=0;i<edges.size();++i) edges[i]=double(i)/8;
    SP<const Handle> mesh=std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{edges,edges,edges}},Vec3D<bool>{true,true,true}));
    IncompressibleIsothermalSolver<Pack> solver(mesh,{},transient_options());
    solver.configure_turbulence(sas_options());
    for(size_t i=0;i<mesh->num_owned_cells();++i)
    {
        const auto p=mesh->cell_centroid(i);
        solver.velocity().set_owned_value(i,{std::sin(2*std::numbers::pi*p.y),std::sin(2*std::numbers::pi*p.z),std::sin(2*std::numbers::pi*p.x)});
    }
    solver.velocity().sync_ghosts();
    expect_active_bounded_step(solver); expect_active_bounded_step(solver);
}

TEST(SASSupportedPathsTest, GaussLinearSASAdvancesWithSlipAndRestarts)
{
    auto mesh=box_mesh(); auto bc=closed_boundaries(*mesh);
    bc.velocity["zmin"].type=BoundaryConditionType::Slip;
    bc.velocity["zmax"].type=BoundaryConditionType::Slip;
    auto options=sas_options(); options.gradient_scheme=FVM::CellGradientScheme::GaussLinear;
    IncompressibleIsothermalSolver<Pack> solver(mesh,bc,transient_options());
    auto& model=solver.configure_turbulence(options); initialize_circulation(solver);
    expect_active_bounded_step(solver);
    auto saved=model.snapshot(); const auto q=model.sas_statistics().max_source;
    expect_active_bounded_step(solver); model.restore(saved);
    EXPECT_DOUBLE_EQ(model.sas_statistics().max_source,q);
}

TEST(SASSupportedPathsTest, GaussLinearSASCouplesSignedBoussinesqProduction)
{
    auto mesh=box_mesh(); auto time=transient_options();
    time.thermal_expansion=.001; time.gravity_z=-9.81;
    BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),time);
    solver.initialize_linear_temperature({0,0,1},2,1);
    auto options=sas_options(); options.gradient_scheme=FVM::CellGradientScheme::GaussLinear;
    options.buoyancy_model=TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
    const auto& model=solver.configure_turbulence(options); initialize_circulation(solver);
    expect_active_bounded_step(solver);
    double local=0,global=0;
    for(size_t i=0;i<mesh->num_owned_cells();++i) local+=std::abs(model.output_fields().at("buoyancy_production")->value(i));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(),Teuchos::REDUCE_SUM,1,&local,&global);
    EXPECT_GT(global,1e-6);
}
