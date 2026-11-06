#include <gtest/gtest.h>
#include "FVM/SteadyDiagnostics.hh"
#include "FVM/FaceFlux.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include <Tpetra_Core.hpp>
#include <limits>
namespace {
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Scalar = SimpleFluid::ScalarCellFieldStored<Pack>;
using Vector = SimpleFluid::VectorCellFieldStored<Pack>;
using Flux = SimpleFluid::ScalarFaceFieldStored<Pack>;
using namespace SimpleFluid::FVM;
using utils_test::KokkosEnvironment;
testing::Environment* const env = testing::AddGlobalTestEnvironment(new KokkosEnvironment);
auto mesh()
{
 return std::make_shared<Mesh>(std::make_shared<Mesh::Cartesian>(
  SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.,1.,2.,3.,4.},{0.,1.},{0.,1.}}}));
}
TEST(SteadyDiagnostics, NativeBalancePerturbationAndTimeStepIndependence)
{
 auto m=mesh(); Scalar q(m,"q"), storage(m,"storage"), adv(m,"adv"), diffusion(m,"diffusion");
 Flux flux(m,"flux"); q.put_value(2.); storage.put_value(3.); adv.put_value(1.); diffusion.put_value(0.); flux.put_value(0.);
 q.sync_ghosts();storage.sync_ghosts();adv.sync_ghosts();diffusion.sync_ghosts();flux.sync_ghosts();
 auto assemble=[&](double dt) {
  FieldStoredWeightedScalarTransportRequest<Pack,Mesh> request{
    .old_values=q,.face_fluxes=flux,.time_step=dt,.storage_weight=storage,.advection_weight=adv,.diffusivity=diffusion,
    .boundary_condition=[](int,size_t){return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Neumann,0.};},
    .boundary_value=[](int,size_t){return 0.;},.source=[](auto){return 4.;},
    .treatment=NonOrthogonalTreatment::Explicit,
    .discretization={.mode=AssemblyMode::Steady},.implicit_sink=[](auto){return 2.;}};
  return weighted_scalar_transport_system<Pack>(request);
 };
 auto a=assemble(0.01), b=assemble(100.);
 std::vector<double> volumes(m->num_owned_cells(),1.);
 auto zero=steady_residual_report<Pack>(*a.matrix,*a.rhs,q.owned_data(),volumes,{},true);
 EXPECT_NEAR(zero.volume_rms,0.,1e-14); EXPECT_DOUBLE_EQ(zero.scale,8.);
 auto other=steady_residual_report<Pack>(*b.matrix,*b.rhs,q.owned_data(),volumes);
 EXPECT_DOUBLE_EQ(other.scale,zero.scale); EXPECT_DOUBLE_EQ(other.volume_rms,zero.volume_rms);
 q.put_value(3.); auto perturb=steady_residual_report<Pack>(*a.matrix,*a.rhs,q.owned_data(),volumes,{},true);
 EXPECT_DOUBLE_EQ(perturb.volume_rms,2.); EXPECT_DOUBLE_EQ(perturb.scale,10.); EXPECT_DOUBLE_EQ(perturb.dimensionless,0.2);
 EXPECT_EQ(perturb.included_count,4u); EXPECT_EQ(perturb.integrated.size(),volumes.size());
 std::vector<unsigned char> excluded(volumes.size(),1);
 auto constraints=steady_residual_report<Pack>(*a.matrix,*a.rhs,q.owned_data(),volumes,excluded);
 EXPECT_EQ(constraints.included_count,0u); EXPECT_EQ(constraints.excluded_count,4u); EXPECT_EQ(constraints.volume_rms,0.);
}
TEST(SteadyDiagnostics, LegacyScalarBalanceAndTimeStepIndependence)
{
 auto m=SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4,1,1));
 SimpleFluid::CellField<Pack> q(m,2.),storage(m,3.),adv(m,1.),diffusion(m,0.);SimpleFluid::FaceField<Pack> flux(m,0.);
 auto assemble=[&](double dt){
 WeightedScalarTransportRequest<Pack> request{.old_values=q,.face_fluxes=flux,.time_step=dt,
 .storage_weight=storage,.advection_weight=adv,.diffusivity=diffusion,
 .boundary_condition=[](int,size_t){return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Neumann,0.};},
 .boundary_value=[](int,size_t){return 0.;},.source=[](auto){return 4.;},.treatment=NonOrthogonalTreatment::Explicit,
 .discretization={.mode=AssemblyMode::Steady},.implicit_sink=[](auto){return 2.;}};
 return weighted_scalar_transport_system<Pack>(request);};
 auto a=assemble(0.01),b=assemble(10.);std::vector<double> volumes;
 for(size_t i=0;i<m->num_owned_cells();++i)volumes.push_back(m->cell_volume(static_cast<Pack::local_ordinal_type>(i)));
 auto ra=steady_residual_report<Pack>(*a.matrix,*a.rhs,q.vector(),volumes);
 auto rb=steady_residual_report<Pack>(*b.matrix,*b.rhs,q.vector(),volumes);
 EXPECT_NEAR(ra.volume_rms,0.,1e-14);EXPECT_DOUBLE_EQ(ra.scale,rb.scale);EXPECT_DOUBLE_EQ(ra.scale,8.);
}
TEST(SteadyDiagnostics, LegacyPhysicalMomentumAndThermalOmitStorage)
{
 auto m=SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4,1,1));
 SimpleFluid::VectorCellField<Pack> u(m,SimpleFluid::vec3<double>{2.,3.,4.});
 SimpleFluid::CellField<Pack> temperature(m,2.),rho(m,3.),cp(m,4.),zero(m,0.);SimpleFluid::FaceField<Pack> flux(m,0.);
 auto momentum=[&](double dt){return physical_momentum_transport_system<Pack>(u,flux,dt,zero,1.,
  [](int,size_t){return SimpleFluid::vec3<double>{2.,3.,4.};},[](auto){return SimpleFluid::vec3<double>{1.,2.,3.};},
  NonOrthogonalTreatment::Explicit,nullptr,Teuchos::null,detail::AlwaysDiffuseBoundary{},nullptr,nullptr,
  FaceCoefficientInterpolation::Harmonic,AssemblyMode::Steady);};
 auto thermal=[&](double dt){return physical_temperature_transport_system<Pack>(temperature,flux,dt,rho,cp,zero,
  [](int,size_t){return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Neumann,0.};},
  [](int,size_t){return 0.;},[](auto){return 4.;},NonOrthogonalTreatment::Explicit,nullptr,Teuchos::null,
  nullptr,nullptr,FaceCoefficientInterpolation::Harmonic,AssemblyMode::Steady);};
 auto a=momentum(0.01),b=momentum(10.);auto ta=thermal(0.01),tb=thermal(10.);std::vector<double> volumes;
 for(size_t i=0;i<m->num_owned_cells();++i)volumes.push_back(m->cell_volume(static_cast<Pack::local_ordinal_type>(i)));
 for(int c=0;c<3;++c){
  auto ra=steady_residual_report<Pack>(*a.matrix,*a.rhs->getVector(c),*u.owned_data().getVector(c),volumes);
  auto rb=steady_residual_report<Pack>(*b.matrix,*b.rhs->getVector(c),*u.owned_data().getVector(c),volumes);
  EXPECT_DOUBLE_EQ(ra.volume_rms,c+1.);EXPECT_DOUBLE_EQ(ra.scale,c+1.);EXPECT_DOUBLE_EQ(rb.scale,ra.scale);
 }
 auto ra=steady_residual_report<Pack>(*ta.matrix,*ta.rhs,temperature.owned_data(),volumes);
 auto rb=steady_residual_report<Pack>(*tb.matrix,*tb.rhs,temperature.owned_data(),volumes);
 EXPECT_DOUBLE_EQ(ra.volume_rms,4.);EXPECT_DOUBLE_EQ(ra.scale,4.);EXPECT_DOUBLE_EQ(rb.scale,ra.scale);
}
TEST(SteadyDiagnostics, PhysicalMomentumAndThermalOmitStorage)
{
 auto m=mesh();Vector u(m,"u"); Scalar temperature(m,"T"),rho(m,"rho"),cp(m,"cp"),zero(m,"zero");Flux flux(m,"flux");
 u.put_value({2.,3.,4.});temperature.put_value(2.);rho.put_value(3.);cp.put_value(4.);zero.put_value(0.);flux.put_value(0.);
 u.sync_ghosts();temperature.sync_ghosts();rho.sync_ghosts();cp.sync_ghosts();zero.sync_ghosts();flux.sync_ghosts();
 auto momentum=[&](double dt){return physical_momentum_transport_system<Pack>(u,flux,dt,zero,1.,
  [](int,size_t){return SimpleFluid::vec3<double>{2.,3.,4.};},[](auto){return SimpleFluid::vec3<double>{1.,2.,3.};},
  NonOrthogonalTreatment::Explicit,nullptr,Teuchos::null,detail::AlwaysDiffuseBoundary{},nullptr,nullptr,
  FaceCoefficientInterpolation::Harmonic,nullptr,nullptr,{},AssemblyMode::Steady);};
 auto thermal=[&](double dt){return physical_temperature_transport_system<Pack>(temperature,flux,dt,rho,cp,zero,
  [](int,size_t){return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Neumann,0.};},
  [](int,size_t){return 0.;},[](auto){return 4.;},NonOrthogonalTreatment::Explicit,nullptr,Teuchos::null,
  nullptr,nullptr,FaceCoefficientInterpolation::Harmonic,nullptr,nullptr,nullptr,false,AssemblyMode::Steady);};
 const auto a=momentum(0.01),b=momentum(10.);const auto ta=thermal(0.01),tb=thermal(10.);
 std::vector<double> volumes(m->num_owned_cells(),1.);
 for(int c=0;c<3;++c){
  auto ra=steady_residual_report<Pack>(*a.matrix,*a.rhs->getVector(c),*u.owned_data().getVector(c),volumes);
  auto rb=steady_residual_report<Pack>(*b.matrix,*b.rhs->getVector(c),*u.owned_data().getVector(c),volumes);
  EXPECT_DOUBLE_EQ(ra.volume_rms,c+1.);EXPECT_DOUBLE_EQ(ra.scale,c+1.);EXPECT_DOUBLE_EQ(rb.scale,ra.scale);EXPECT_DOUBLE_EQ(ra.volume_rms,rb.volume_rms);
 }
 auto ra=steady_residual_report<Pack>(*ta.matrix,*ta.rhs,temperature.owned_data(),volumes);
 auto rb=steady_residual_report<Pack>(*tb.matrix,*tb.rhs,temperature.owned_data(),volumes);
 EXPECT_DOUBLE_EQ(ra.volume_rms,4.);EXPECT_DOUBLE_EQ(ra.scale,4.);EXPECT_DOUBLE_EQ(rb.scale,ra.scale);EXPECT_DOUBLE_EQ(ra.volume_rms,rb.volume_rms);
}
TEST(SteadyDiagnostics, RawReductionScaleZeroAndEmptyRanks)
{
 const auto comm=Tpetra::getDefaultComm(); const bool owner=comm->getRank()==0;
 std::vector<double> r(owner?1:0,2.), scale(owner?1:0,0.), volumes(owner?1:0,2.);
 auto result=integrated_residual_report(r,scale,volumes,*comm,true);
 EXPECT_DOUBLE_EQ(result.signed_sum,2.);EXPECT_DOUBLE_EQ(result.volume_rms,1.);EXPECT_FALSE(result.dimensionless_available);
 r.assign(r.size(),0.);result=integrated_residual_report(r,scale,volumes,*comm);
 EXPECT_TRUE(result.dimensionless_available);EXPECT_DOUBLE_EQ(result.dimensionless,0.);
 if(owner)r[0]=std::numeric_limits<double>::infinity();
 EXPECT_THROW(integrated_residual_report(r,scale,volumes,*comm),std::invalid_argument);
}
void check_flux_components(const std::shared_ptr<Mesh>& m,double xmax_pressure)
{
 Vector u(m,"u");Scalar p(m,"p");Flux total(m,"total"),v(m,"v"),s(m,"s"),reference(m,"reference"),scaled(m,"scaled"),v2(m,"v2"),s2(m,"s2");
 for(size_t i=0;i<m->num_owned_cells();++i){auto lid=static_cast<Pack::local_ordinal_type>(i);u.set_owned_value(lid,{1.,0.,0.});auto x=m->cell_centroid(lid).x;p.set_owned_value(lid,x*x);}
 u.sync_ghosts();p.sync_ghosts();
 SimpleFluid::BoundaryConditionSet bc; for(auto name:{"xmin","xmax","ymin","ymax","zmin","zmax"})bc.velocity[name]={SimpleFluid::BoundaryConditionType::Neumann,{}};
 SimpleFluid::SP<const Mesh> cm=m;auto cache=cache_velocity_boundary_conditions<Pack>(cm,bc);
 SimpleFluid::BoundaryConditionMap pressure_bc;pressure_bc["xmax"]={SimpleFluid::BoundaryConditionType::Dirichlet,xmax_pressure};
 FieldStoredPressureWeightedFaceFluxWorkspace<Pack> workspace(m);
 pressure_weighted_face_fluxes(u,p,0.2,cache,pressure_bc,workspace,reference);
 pressure_weighted_face_flux_components(u,p,0.2,cache,pressure_bc,workspace,total,v,s);
 pressure_weighted_face_flux_components(u,p,0.4,cache,pressure_bc,workspace,scaled,v2,s2);
 Flux zero_total(m,"zero_total"),zero_velocity(m,"zero_velocity"),zero_pressure(m,"zero_pressure"),zero_plain(m,"zero_plain");
 pressure_weighted_face_flux_components(u,p,0.,cache,pressure_bc,workspace,zero_total,zero_velocity,zero_pressure);
 pressure_weighted_face_fluxes(u,p,0.,cache,pressure_bc,workspace,zero_plain);
 double local_max=0.;
 for(auto face:total.owned_face_ids()){
  local_max=std::max(local_max,std::abs(s.local_value(face)));
  EXPECT_DOUBLE_EQ(total.local_value(face),reference.local_value(face));
  EXPECT_DOUBLE_EQ(zero_total.local_value(face),v.local_value(face));
  EXPECT_DOUBLE_EQ(zero_plain.local_value(face),v.local_value(face));
  EXPECT_DOUBLE_EQ(zero_velocity.local_value(face),v.local_value(face));
  EXPECT_DOUBLE_EQ(zero_pressure.local_value(face),0.);
  if(m->is_boundary_face(face) && std::abs(m->face_centroid(face).x-std::sqrt(xmax_pressure))<1e-14)
   EXPECT_GT(std::abs(v.local_value(face)),0.);
  EXPECT_NEAR(total.local_value(face),v.local_value(face)+s.local_value(face),1e-14);
  EXPECT_DOUBLE_EQ(v.local_value(face),v2.local_value(face));
  EXPECT_NEAR(2.*s.local_value(face),s2.local_value(face),1e-14);
 }
 double global_max=0.;Teuchos::reduceAll(*m->owned_cell_map()->getComm(),Teuchos::REDUCE_MAX,1,&local_max,&global_max);
 EXPECT_GT(global_max,1e-10);
 // One rank supplying a nonfinite coefficient rejects before any output writes.
 const auto comm=m->owned_cell_map()->getComm();
 const auto nan_coefficient=comm->getRank()==0?std::numeric_limits<double>::quiet_NaN():0.2;
 EXPECT_THROW(pressure_weighted_face_flux_components(u,p,nan_coefficient,cache,pressure_bc,workspace,total,v,s),std::invalid_argument);
 const auto inf_coefficient=comm->getRank()==0?std::numeric_limits<double>::infinity():0.2;
 EXPECT_THROW(pressure_weighted_face_flux_components(u,p,inf_coefficient,cache,pressure_bc,workspace,total,v,s),std::invalid_argument);
}
TEST(SteadyDiagnostics, FluxComponentsMatchProductionAndScaleLinearly)
{
 check_flux_components(mesh(),16.);
}
TEST(SteadyDiagnostics, FluxComponentsSupportEmptyPartitions)
{
 auto geometry=std::make_shared<Mesh::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.,1.,2.},{0.,1.},{0.,1.}}});
 auto tiny=std::make_shared<Mesh>(geometry,Mesh::DistributionOptions{.allow_empty_partitions=true});
 check_flux_components(tiny,4.);
 const auto comm=tiny->owned_cell_map()->getComm();
 if(comm->getSize()>2){int empty=tiny->num_owned_cells()==0?1:0,total=0;
  Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&empty,&total);EXPECT_GT(total,0);}
}

}
