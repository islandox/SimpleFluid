/** @file testSASSupportedPaths.cc @brief Transient and transaction gates for SAS extensions. */
#include <gtest/gtest.h>
#include "solvers/IncompressibleIsothermalSolver.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"
#include <limits>
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

namespace
{
struct SavedSASFields
{
    using Field=ScalarCellFieldStored<Pack>;
    std::map<const Field*,std::vector<double>> values;
    template<class Solver> explicit SavedSASFields(const Solver& solver)
    {
        capture(solver.temperature()); capture(solver.pressure());
        const auto& m=solver.material_properties();
        capture(m.density); capture(m.dynamic_viscosity); capture(m.specific_heat_capacity); capture(m.thermal_conductivity);
        auto model=[&](const auto* p) { if(p) for(const auto& [name,field]:p->output_fields()) capture(*field); };
        model(solver.find_turbulence_model()); model(solver.find_material_feedback_model());
        model(solver.find_scalar_void_fraction_model()); model(solver.find_precursor_model());
        model(solver.find_radiolytic_gas_model()); model(solver.find_boiling_source_model());
        if(solver.find_free_surface_model())
        {
            capture(solver.clear_level()); capture(solver.pool_level());
            capture(solver.headspace_pressure()); capture(solver.pool_occupancy());
        }
        if(const auto* inventory=solver.find_liquid_mass_inventory())
        {
            capture(inventory->pureLiquidDensity()); capture(inventory->cellMassInventory());
        }
    }
    void capture(const Field& field)
    {
        auto& saved=values[&field];
        for(size_t i=0;i<field.num_owned_cells();++i) saved.push_back(field.value(i));
    }
    void expect_restored() const
    {
        for(const auto& [field,saved]:values)
            for(size_t i=0;i<saved.size();++i) EXPECT_DOUBLE_EQ(field->value(i),saved[i]);
    }
};

// Feedback runs after temperature, radiolysis and precursors. Inputs remain finite
// until this final feedback stage; overflow rejects the trial after upstream work.
void expect_late_multiphysics_rollback(BoussinesqSolver<Pack>& solver, std::function<void()> after_restore={})
{
    solver.temperature().put_scalar(300.); solver.temperature().sync_ghosts();
    MaterialFeedbackOptions original;
    if(auto* feedback=solver.find_material_feedback_model()) original=feedback->options();
    else { original.reference_dynamic_viscosity=.001; solver.configure_material_feedback(original); }
    auto reject=original;
    reject.density_mode=DensityFeedbackMode::BoussinesqTemperatureOnly;
    reject.reference_density=1.; reject.reference_temperature=300.; reject.thermal_expansion=1e308;
    solver.configure_material_feedback(reject);
    const SavedSASFields saved(solver);
    const auto time=solver.time(); const auto step=solver.step_index();
    const auto source=solver.find_turbulence_model()->sas_statistics().max_source;
    solver.add_temperature_source("sas_rejection_cooling",-2000.);
    EXPECT_ANY_THROW(solver.step());
    saved.expect_restored();
    EXPECT_DOUBLE_EQ(solver.time(),time); EXPECT_EQ(solver.step_index(),step);
    EXPECT_DOUBLE_EQ(solver.find_turbulence_model()->sas_statistics().max_source,source);
    if(after_restore) after_restore();
    solver.remove_temperature_source("sas_rejection_cooling");
    solver.configure_material_feedback(original);
    expect_active_bounded_step(solver);
}
}

TEST(SASSupportedPathsTest, MaterialFeedbackPublishesAndRollsBackWithSAS)
{
    auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
    solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    MaterialFeedbackOptions feedback;
    feedback.density_mode=DensityFeedbackMode::BoussinesqTemperatureOnly;
    feedback.reference_density=1.2; feedback.reference_temperature=300;
    feedback.thermal_expansion=.01; feedback.reference_dynamic_viscosity=.002;
    solver.configure_material_feedback(feedback);
    expect_active_bounded_step(solver);
    for(size_t i=0;i<mesh->num_owned_cells();++i)
    {
        EXPECT_NEAR(solver.material_properties().density.value(i),1.2,1e-12);
        EXPECT_DOUBLE_EQ(solver.material_properties().dynamic_viscosity.value(i),.002);
    }
    expect_late_multiphysics_rollback(solver);
}

TEST(SASSupportedPathsTest, ScalarVoidDiffusionCollapseAndFeedbackRollBackWithSAS)
{
    auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
    solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    ScalarVoidFractionOptions options; options.initial_alpha=.2; options.alpha_collapse_time=.1; options.alpha_diffusivity=.01;
    auto& phase=solver.configure_scalar_void_fraction(options);
    ScalarCellFieldStored<Pack> initial(mesh,"initial_void");
    for(size_t i=0;i<mesh->num_owned_cells();++i) initial.set_owned_value(i,.2+.05*mesh->cell_centroid(i).x);
    initial.sync_ghosts(); phase.initialize_from(initial);
    MaterialFeedbackOptions feedback; feedback.density_mode=DensityFeedbackMode::Mixture;
    feedback.liquid_density=1; feedback.gas_density=.1; feedback.reference_dynamic_viscosity=.001;
    solver.configure_material_feedback(feedback);
    double before=0,after=0;
    for(size_t i=0;i<mesh->num_owned_cells();++i) before+=phase.alpha_g().value(i)*mesh->cell_volume(i);
    expect_active_bounded_step(solver);
    for(size_t i=0;i<mesh->num_owned_cells();++i)
    {
        const auto alpha=phase.alpha_g().value(i);
        after+=alpha*mesh->cell_volume(i);
        EXPECT_NEAR(phase.alpha_l().value(i),1-alpha,1e-14);
        EXPECT_NEAR(solver.material_properties().density.value(i),1-.9*alpha,1e-13);
    }
    std::array<double,2> local{before,after},total{};
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(),Teuchos::REDUCE_SUM,2,local.data(),total.data());
    EXPECT_NEAR(total[1],total[0]*(1-.001/.1),1e-11);
    expect_late_multiphysics_rollback(solver);
}

TEST(SASSupportedPathsTest, PrecursorInventoryAndDiagnosticsRollBackWithSAS)
{
    auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
    solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    ScalarVoidFractionOptions phase; phase.initial_alpha=.2;
    solver.configure_scalar_void_fraction(phase);
    DelayedNeutronPrecursorOptions options;
    options.group_count=2; options.initial_concentrations={1,2}; options.decay_constants={.1,.2};
    options.source_terms={.2,.1}; options.effective_diffusivity=.01;
    auto& precursors=solver.configure_precursors(options);
    expect_active_bounded_step(solver);
    const auto before=precursors.last_inventory_diagnostics(0);
    const double expected=.8*std::exp(-.1*.001)+.2*(-std::expm1(-.1*.001))/.1;
    EXPECT_NEAR(before.inventory_after,expected,1e-11);
    EXPECT_NEAR(before.balance_error,0,1e-11);
    expect_late_multiphysics_rollback(solver,[&]
    {
        const auto& restored=precursors.last_inventory_diagnostics(0);
        EXPECT_DOUBLE_EQ(restored.inventory_after,before.inventory_after);
        EXPECT_DOUBLE_EQ(restored.source_added,before.source_added);
        EXPECT_DOUBLE_EQ(restored.balance_error,before.balance_error);
    });
    const auto stale=precursors.snapshot(); precursors.configure(options);
    EXPECT_ANY_THROW(precursors.restore(stale));
}

namespace
{
RadiolyticGasOptions seeded_radiolysis_options(RadiolyticGasMode mode)
{
    RadiolyticGasOptions options; options.mode=mode; options.max_source_alpha_rate=1.;
    options.hydrogen_yield_mol_per_j=2e-7; options.reference_pressure=1e5;
    // Unit-density fixture has molecular nu=.001; D=1e-5 gives admissible Sc=100.
    options.henry_coefficient=1e-5; options.surface_tension=.07; options.hydrogen_diffusivity=1e-5;
    options.uranium_concentration_mol_per_m3=1000; options.hydrogen_yield_molecules_per_100_ev=1.8;
    options.dissolved_transport=RadiolyticTransportMode::Advective;
    options.rise_velocity_mode=BubbleRiseVelocityMode::ConstantSlip; options.constant_slip_velocity=.05;
    options.initial_dissolved_hydrogen=1e-5; options.initial_micro_number_density=1e10;
    options.initial_micro_moles=1e-5; options.initial_large_number_density=1e8; options.initial_large_moles=5e-6;
    options.microbubble_lifetime=1e30; options.large_bubble_dissolution_time=1e30;
    options.micro_to_large_conversion_coefficient=0; options.min_radius=1e-12; options.max_radius=1e-3;
    options.min_population=1e-40; options.max_population=1e40;
    return options;
}
}

TEST(SASSupportedPathsTest, RadiolysisAndHydrogenLedgersRollBackWithSAS)
{
    for(auto mode:{RadiolyticGasMode::IdealGasSource,RadiolyticGasMode::Sheng2024TwoPopulation})
    {
        auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
        solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
        FissionPowerSourceOptions power; power.profile=FissionPowerProfile::Constant; power.power_density=1.;
        solver.configure_fission_power_source(power);
        auto options=seeded_radiolysis_options(mode);
        auto& gas=solver.configure_radiolytic_gas(options);
        expect_active_bounded_step(solver);
        if(mode==RadiolyticGasMode::Sheng2024TwoPopulation)
        {
            EXPECT_GT(gas.last_statistics().hydrogen_produced,0);
            EXPECT_NEAR(gas.last_statistics().inventory_error,0,1e-14);
        }
        else
        {
            double local=0,global=0;
            for(size_t i=0;i<mesh->num_owned_cells();++i) local+=solver.find_scalar_void_fraction_model()->alpha_g().value(i);
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(),Teuchos::REDUCE_SUM,1,&local,&global);
            EXPECT_GT(global,0);
        }
        const auto statistics=gas.last_statistics();
        const auto inventory=gas.global_submerged_hydrogen_moles();
        expect_late_multiphysics_rollback(solver,[&]
        {
            EXPECT_DOUBLE_EQ(gas.last_statistics().hydrogen_after,statistics.hydrogen_after);
            EXPECT_DOUBLE_EQ(gas.last_statistics().cumulative_hydrogen_produced,statistics.cumulative_hydrogen_produced);
            EXPECT_DOUBLE_EQ(gas.global_submerged_hydrogen_moles(),inventory);
        });
    }
}

TEST(SASSupportedPathsTest, BoilingSourcesAndPendingStateRollBackWithSAS)
{
    auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
    solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    BoilingSourceOptions options; options.enable_bulk_boiling=true; options.enable_wall_boiling=true;
    options.saturation_temperature=299.; options.latent_heat=1000.; options.gas_density=1.;
    options.wall_heat_flux=1.; options.wall_evaporation_fraction=.2; options.wall_boiling_patches={"zmin"};
    auto& boiling=solver.configure_boiling_source(options);
    expect_active_bounded_step(solver);
    const auto phase=boiling.last_phase_change_diagnostics();
    EXPECT_GT(phase.accepted_evaporation_mass,0);
    double local=0,total=0;
    for(size_t i=0;i<mesh->num_owned_cells();++i) local+=boiling.latent_heat_sink().value(i)*mesh->cell_volume(i)*.001;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(),Teuchos::REDUCE_SUM,1,&local,&total);
    EXPECT_NEAR(total,phase.accepted_evaporation_mass*options.latent_heat,1e-13);
    expect_late_multiphysics_rollback(solver,[&]
    {
        EXPECT_DOUBLE_EQ(boiling.last_phase_change_diagnostics().accepted_evaporation_mass,phase.accepted_evaporation_mass);
        EXPECT_FALSE(boiling.phase_change_completion_pending());
    });
    const auto stale=boiling.snapshot(); boiling.configure(options);
    EXPECT_ANY_THROW(boiling.restore(stale));
}

namespace
{
FreeSurfaceOptions fixed_surface_options()
{
    FreeSurfaceOptions result;
    result.enabled=true; result.mode=FreeSurfaceMode::PlanarVolumeBudget;
    result.initial_liquid_volume=.5;
    result.vessel.bottom_elevation=0; result.vessel.top_elevation=1;
    result.vessel.cross_section_area=1; result.vessel.total_internal_volume=1;
    result.liquid_mass.mode=LiquidVolumeMode::CellMassInventory;
    result.headspace.mode=HeadspaceMode::Vented;
    result.headspace.initial_temperature=300; result.headspace.total_internal_volume=1;
    return result;
}
}

TEST(SASSupportedPathsTest, FixedFreeSurfaceRestoresBoilingInventoryHistoryAndRetry)
{
    for(auto inventory_mode:{LiquidVolumeMode::GlobalConstantMass,LiquidVolumeMode::CellMassInventory})
    {
        auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
        solver.initialize_heated_box(301,301); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
        BoilingSourceOptions boiling; boiling.enable_bulk_boiling=true;
        boiling.saturation_temperature=300; boiling.latent_heat=1000; boiling.gas_density=.01;
        auto& phase=solver.configure_boiling_source(boiling);
        auto options=fixed_surface_options(); options.liquid_mass.mode=inventory_mode;
        options.headspace.temperature_mode=HeadspaceTemperatureMode::Prescribed;
        options.headspace.prescribed_temperature_times={0.,.0015};
        options.headspace.prescribed_temperature_values={300.,300.};
        ASSERT_NE(solver.configure_free_surface(options),nullptr);
        const auto mass_before=solver.liquid_mass_inventory().totalMass();
        expect_active_bounded_step(solver);
        const auto mass=solver.liquid_mass_inventory().totalMass();
        EXPECT_GT(phase.last_phase_change_diagnostics().accepted_evaporation_mass,0);
        EXPECT_NEAR(mass,mass_before-phase.last_phase_change_diagnostics().accepted_evaporation_mass,1e-12);
        const auto history=solver.free_surface_history().size();
        const auto surface=solver.free_surface_diagnostics();
        const auto steam=phase.global_submerged_steam_mass();
        const auto q=solver.find_turbulence_model()->sas_statistics().max_source;
        const SavedSASFields saved(solver);
        EXPECT_ANY_THROW(solver.step()); // Headspace history ends after the next physical time.
        saved.expect_restored();
        EXPECT_DOUBLE_EQ(solver.time(),.001); EXPECT_EQ(solver.step_index(),1);
        EXPECT_EQ(solver.free_surface_history().size(),history);
        EXPECT_DOUBLE_EQ(solver.liquid_mass_inventory().totalMass(),mass);
        EXPECT_DOUBLE_EQ(solver.free_surface_diagnostics().pool_level,surface.pool_level);
        EXPECT_DOUBLE_EQ(phase.global_submerged_steam_mass(),steam);
        EXPECT_DOUBLE_EQ(solver.find_turbulence_model()->sas_statistics().max_source,q);
        solver.set_time_step(.00025); expect_active_bounded_step(solver);
        EXPECT_EQ(solver.free_surface_history().size(),history+1);
    }
}

TEST(SASSupportedPathsTest, LazyFreeSurfaceInitializationRemainsOwnedAfterRejection)
{
    auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
    solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    solver.temperature().put_scalar(300.); solver.temperature().sync_ghosts();
    auto options=fixed_surface_options(); options.liquid_mass.initial_liquid_mass=.5;
    ASSERT_NE(solver.configure_free_surface(options),nullptr);
    ASSERT_FALSE(solver.find_free_surface_model()->initialized());
    bool inconsistent_density=true;
    solver.set_material_updater([&](const auto&,auto& material){material.density.put_scalar(inconsistent_density ? 2. : 1.);});
    EXPECT_ANY_THROW(solver.step());
    ASSERT_NE(solver.find_free_surface_model(),nullptr);
    EXPECT_FALSE(solver.find_free_surface_model()->initialized());
    EXPECT_FALSE(solver.liquid_mass_inventory().initialized());
    EXPECT_TRUE(solver.free_surface_history().empty());
    inconsistent_density=false;
    solver.add_temperature_source("late_overflow_one",std::numeric_limits<double>::max());
    solver.add_temperature_source("late_overflow_two",std::numeric_limits<double>::max());
    EXPECT_ANY_THROW(solver.step()); // Successful lazy initialization must also roll back.
    ASSERT_NE(solver.find_free_surface_model(),nullptr);
    EXPECT_FALSE(solver.find_free_surface_model()->initialized());
    EXPECT_TRUE(solver.free_surface_history().empty());
    EXPECT_FALSE(solver.find_turbulence_model()->sas_statistics().valid);
    solver.remove_temperature_source("late_overflow_one"); solver.remove_temperature_source("late_overflow_two");
    expect_active_bounded_step(solver);
    EXPECT_TRUE(solver.find_free_surface_model()->initialized());
    EXPECT_EQ(solver.free_surface_history().size(),2U);
}

TEST(SASSupportedPathsTest, FixedFreeSurfaceRadiolysisRestoresClosedHeadspaceAndEscape)
{
    auto mesh=box_mesh(); BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),transient_options());
    solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    auto gas_options=seeded_radiolysis_options(RadiolyticGasMode::Sheng2024TwoPopulation);
    gas_options.free_surface_patches={"zmax"};
    auto& gas=solver.configure_radiolytic_gas(gas_options);
    solver.add_fission_power_source().initialize_constant(1.);
    auto options=fixed_surface_options(); options.headspace.mode=HeadspaceMode::Closed;
    options.coupling.maximum_absolute_pressure=2e5;
    options.headspace.temperature_mode=HeadspaceTemperatureMode::Prescribed;
    options.headspace.prescribed_temperature_times={0.,.0015};
    options.headspace.prescribed_temperature_values={300.,300.};
    ASSERT_NE(solver.configure_free_surface(options),nullptr);
    expect_active_bounded_step(solver);
    const SavedSASFields saved(solver);
    const auto headspace=solver.free_surface_diagnostics().headspace.pressure;
    const auto history=solver.free_surface_history().size();
    const auto escape=gas.last_statistics().cumulative_hydrogen_escaped;
    EXPECT_GT(escape,0);
    EXPECT_ANY_THROW(solver.step());
    saved.expect_restored();
    EXPECT_DOUBLE_EQ(gas.last_statistics().cumulative_hydrogen_escaped,escape);
    EXPECT_DOUBLE_EQ(solver.free_surface_diagnostics().headspace.pressure,headspace);
    EXPECT_EQ(solver.free_surface_history().size(),history);
    solver.set_time_step(.00025); expect_active_bounded_step(solver);
}

TEST(SASSupportedPathsTest, FixedFreeSurfaceStillRejectsSASALE)
{
    auto mesh=std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0.,.5,1.},{0.,.5,1.},{0.,.5,1.}}}));
    auto time=transient_options();
    BoussinesqSolver<Pack> solver(mesh,closed_boundaries(*mesh),time,{},BoussinesqModelOptions::legacy_defaults(time));
    solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options());
    auto options=fixed_surface_options(); options.mode=FreeSurfaceMode::PlanarALE; options.ale.top_boundary="zmax";
    EXPECT_ANY_THROW(solver.configure_free_surface(options));
}

namespace
{
// NOX requires the named configuration to be replicated even on ranks which
// do not own a particular patch. closed_boundaries() enumerates local patches.
BoundaryConditionSet nox_box_boundaries()
{
    BoundaryConditionSet result;
    for(const auto* name:{"xmin","xmax","ymin","ymax","zmin","zmax"})
        result.velocity[name]={BoundaryConditionType::NoSlip,{}};
    return result;
}

TimeStepperOptions nox_options()
{
    auto time=transient_options();
    time.pressure_velocity_coupling=PressureVelocityCoupling::CoupledNonlinear;
    time.nonlinear.linear_backend=LinearSolverBackend::Gmres;
    time.nonlinear.maximum_iterations=40;
    return time;
}

template<class Field> std::vector<double> owned_values(const Field& field)
{
    const auto view=field.owned_data().getLocalViewHost(Tpetra::Access::ReadOnly);
    std::vector<double> result;
    for(size_t i=0;i<view.extent(0);++i)
        for(size_t j=0;j<view.extent(1);++j) result.push_back(view(i,j));
    return result;
}

void expect_same_turbulence(const TurbulenceModel<Pack,Handle>& actual,
    const TurbulenceModel<Pack,Handle>& expected)
{
    ASSERT_EQ(actual.output_fields().size(),expected.output_fields().size());
    for(const auto& [name,field]:expected.output_fields())
    {
        SCOPED_TRACE(name);
        const auto* other=actual.output_fields().at(name);
        for(size_t cell=0;cell<field->num_owned_cells();++cell)
            EXPECT_NEAR(other->value(cell),field->value(cell),1e-11*std::max(1.,std::abs(field->value(cell))));
    }
}
}

#ifdef SIMPLEFLUID_ENABLE_NOX
TEST(SASSupportedPathsTest, NoxIsothermalAdvancesSASOnceAcrossBackendsAndGradients)
{
    for(auto backend:{CoupledOperatorBackend::Assembled,CoupledOperatorBackend::BlockComposite})
        for(auto gradient:{FVM::CellGradientScheme::LeastSquares,FVM::CellGradientScheme::GaussLinear})
        {
            auto mesh=box_mesh(); auto bc=nox_box_boundaries();
            bc.velocity["zmin"].type=bc.velocity["zmax"].type=BoundaryConditionType::Slip;
            auto time=nox_options(); time.coupled_operator_backend=backend; time.pressure_gradient_scheme=gradient;
            auto options=sas_options(); options.gradient_scheme=gradient;
            if(gradient==FVM::CellGradientScheme::GaussLinear)
            {
                options.wall_treatment=TurbulenceWallTreatmentType::ResolvedLowReSST;
                options.wall_options.boundary_names={"xmin","xmax","ymin","ymax"};
            }
            LinearSolverOptions linear; linear.tolerance=1e-11; linear.max_iterations=250;
            IncompressibleIsothermalSolver<Pack> solver(mesh,bc,time,linear,7.);
            auto& model=solver.configure_turbulence(options); initialize_circulation(solver);
            TurbulenceModel<Pack,Handle> once(mesh,bc);
            once.configure(options,solver.material_properties(),solver.reference_density());
            const auto velocity_bc=FVM::cache_velocity_boundary_conditions<Pack>(mesh,bc);
            for(int step=1;step<=3;++step)
            {
                ASSERT_NO_FATAL_FAILURE(expect_active_bounded_step(solver));
                EXPECT_TRUE(solver.last_nonlinear_result().converged);
                EXPECT_GT(solver.last_nonlinear_result().residual_evaluations,1);
                EXPECT_LT(solver.last_volume_continuity_residuals().maximum,time.nonlinear.continuity_tolerance);
                EXPECT_EQ(solver.step_index(),step); EXPECT_NEAR(solver.time(),step*time.time_step,1e-15);
                // An independent model advances the previous accepted k/omega
                // exactly once with the final flow and the physical dt. This
                // also checks the applied SAS record survives closure refresh.
                ASSERT_NO_THROW(once.advance(solver.velocity(),solver.pressure_corrected_face_fluxes(),velocity_bc,
                    time.time_step,solver.material_properties(),solver.reference_density(),time.non_orthogonal_treatment,linear));
                expect_same_turbulence(model,once);
                if(step==3) EXPECT_GT(solver.last_nonlinear_result().native_workspace_reuses,0U);
            }
        }
}

TEST(SASSupportedPathsTest, NoxDisabledSourceReproducesParentSST)
{
    auto mesh=box_mesh(); const auto bc=nox_box_boundaries(); const auto time=nox_options();
    IncompressibleIsothermalSolver<Pack> parent(mesh,bc,time), disabled(mesh,bc,time);
    auto options=sas_options(); options.model=TurbulenceModelType::SSTKOmega;
    parent.configure_turbulence(options);
    options.model=TurbulenceModelType::SSTKOmegaSAS; options.sas.enabled=false;
    disabled.configure_turbulence(options);
    initialize_circulation(parent); initialize_circulation(disabled);
    for(int step=0;step<2;++step)
    {
        ASSERT_NO_THROW(parent.step());
        ASSERT_NO_THROW(disabled.step());
        EXPECT_EQ(owned_values(parent.velocity()),owned_values(disabled.velocity()));
        EXPECT_EQ(owned_values(parent.pressure()),owned_values(disabled.pressure()));
        EXPECT_EQ(owned_values(parent.pressure_corrected_face_fluxes()),owned_values(disabled.pressure_corrected_face_fluxes()));
        expect_same_turbulence(*parent.find_turbulence_model(),*disabled.find_turbulence_model());
        EXPECT_FALSE(disabled.find_turbulence_model()->sas_statistics().valid);
    }
}

TEST(SASSupportedPathsTest, NoxPeriodicIsothermalFlowActivatesSAS)
{
    const ArrReal edges{0.,.25,.5,.75,1.};
    SP<const Handle> mesh=std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{edges,edges,edges}},Vec3D<bool>{true,true,true}));
    IncompressibleIsothermalSolver<Pack> solver(mesh,{},nox_options());
    solver.configure_turbulence(sas_options());
    for(size_t i=0;i<mesh->num_owned_cells();++i)
    {
        const auto p=mesh->cell_centroid(i); const auto pi=std::numbers::pi;
        solver.velocity().set_owned_value(i,{std::sin(2*pi*p.y),std::sin(2*pi*p.z),std::sin(2*pi*p.x)});
    }
    solver.velocity().sync_ghosts();
    ASSERT_NO_FATAL_FAILURE(expect_active_bounded_step(solver));
    ASSERT_NO_FATAL_FAILURE(expect_active_bounded_step(solver));
    EXPECT_TRUE(solver.last_nonlinear_result().converged);
}

TEST(SASSupportedPathsTest, NoxRejectedFlowRestoresSASAndRetries)
{
    auto mesh=box_mesh(); auto time=nox_options();
    time.nonlinear.forcing_initial=time.nonlinear.forcing_minimum=1e-10;
    LinearSolverOptions linear; linear.max_iterations=250; linear.tolerance=1e-11;
    IncompressibleIsothermalSolver<Pack> solver(mesh,nox_box_boundaries(),time,linear);
    auto& model=solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    ASSERT_NO_FATAL_FAILURE(expect_active_bounded_step(solver));
    const auto velocity=owned_values(solver.velocity()), pressure=owned_values(solver.pressure());
    const auto flux=owned_values(solver.pressure_corrected_face_fluxes());
    std::map<std::string,std::vector<double>> turbulence;
    for(const auto& [name,field]:model.output_fields()) turbulence[name]=owned_values(*field);
    const auto report=solver.last_nonlinear_result();
    linear.max_iterations=1; solver.set_linear_solver_options(linear);
    for(int attempt=0;attempt<2;++attempt)
    {
        EXPECT_THROW(solver.step(),std::runtime_error);
        EXPECT_EQ(solver.step_index(),1); EXPECT_DOUBLE_EQ(solver.time(),time.time_step);
        EXPECT_EQ(owned_values(solver.velocity()),velocity); EXPECT_EQ(owned_values(solver.pressure()),pressure);
        EXPECT_EQ(owned_values(solver.pressure_corrected_face_fluxes()),flux);
        for(const auto& [name,field]:model.output_fields()) EXPECT_EQ(owned_values(*field),turbulence.at(name));
        EXPECT_DOUBLE_EQ(solver.last_nonlinear_result().total_seconds,report.total_seconds);
        EXPECT_EQ(solver.last_nonlinear_result().residual_evaluations,report.residual_evaluations);
    }
    linear.max_iterations=250; solver.set_linear_solver_options(linear);
    ASSERT_NO_FATAL_FAILURE(expect_active_bounded_step(solver));
}

TEST(SASSupportedPathsTest, NoxBoussinesqRestoresActiveSourceAndMultiphysicsAfterLateRejection)
{
    auto mesh=box_mesh(); auto time=nox_options();
    time.thermal_expansion=.001; time.gravity_z=-9.81; time.reference_temperature=300;
    BoussinesqSolver<Pack> solver(mesh,nox_box_boundaries(),time);
    solver.initialize_heated_box(300,300); solver.configure_turbulence(sas_options()); initialize_circulation(solver);
    ScalarVoidFractionOptions phase; phase.initial_alpha=.1; phase.alpha_collapse_time=1.;
    solver.configure_scalar_void_fraction(phase);
    MaterialFeedbackOptions feedback; feedback.reference_dynamic_viscosity=.001;
    solver.configure_material_feedback(feedback);
    auto options=seeded_radiolysis_options(RadiolyticGasMode::IdealGasSource);
    auto& gas=solver.configure_radiolytic_gas(options);
    solver.add_fission_power_source().initialize_constant(1.);
    ASSERT_NO_FATAL_FAILURE(expect_active_bounded_step(solver));
    ASSERT_TRUE(solver.last_nonlinear_result().converged);
    const auto report=solver.last_nonlinear_result();
    const auto velocity=owned_values(solver.velocity()), flux=owned_values(solver.pressure_corrected_face_fluxes());
    const auto gas_before=gas.last_statistics();
    expect_late_multiphysics_rollback(solver,[&]
    {
        EXPECT_EQ(owned_values(solver.velocity()),velocity);
        EXPECT_EQ(owned_values(solver.pressure_corrected_face_fluxes()),flux);
        EXPECT_DOUBLE_EQ(solver.last_nonlinear_result().total_seconds,report.total_seconds);
        EXPECT_EQ(solver.last_nonlinear_result().residual_evaluations,report.residual_evaluations);
        EXPECT_DOUBLE_EQ(gas.last_statistics().hydrogen_after,gas_before.hydrogen_after);
    });
    EXPECT_EQ(solver.step_index(),2);
    EXPECT_TRUE(solver.last_nonlinear_result().converged);
}

TEST(SASSupportedPathsTest, NoxSASRetainsPseudoTimeAndNonorthogonalRejections)
{
    auto mesh=box_mesh(); auto time=nox_options(); time.physical_time=false;
    IncompressibleIsothermalSolver<Pack> pseudo(mesh,nox_box_boundaries(),time);
    pseudo.configure_turbulence(sas_options()); initialize_circulation(pseudo);
    const auto saved=owned_values(pseudo.velocity());
    EXPECT_THROW(pseudo.step(),std::invalid_argument);
    EXPECT_EQ(pseudo.step_index(),0); EXPECT_EQ(owned_values(pseudo.velocity()),saved);
    if(Tpetra::getDefaultComm()->getSize()!=1) return; // Underlying semi-structured mesh is serial-only.
    using Semi=Meshes::SemiStructuredXY_Z;
    SP<const Handle> skewed=std::make_shared<Handle>(std::make_shared<Semi>(
        Arr<Semi::Vec3>{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{.4,.4,0}},
        Arr<Arr<unsigned>>{{0,1,4},{1,2,4},{2,3,4},{3,0,4}},ArrReal{0,.25,.5,.75,1}));
    IncompressibleIsothermalSolver<Pack> solver(skewed,closed_boundaries(*skewed),nox_options());
    solver.configure_turbulence(sas_options());
    EXPECT_THROW(solver.step(),std::invalid_argument);
    EXPECT_EQ(solver.step_index(),0);
}
#else
TEST(SASSupportedPathsTest, NoxDisabledBuildRejectsSASBeforeMutation)
{
    auto mesh=box_mesh(); auto time=nox_options();
    IncompressibleIsothermalSolver<Pack> isothermal(mesh,nox_box_boundaries(),time);
    isothermal.configure_turbulence(sas_options()); initialize_circulation(isothermal);
    const auto before=owned_values(isothermal.velocity());
    EXPECT_THROW(isothermal.step(),std::invalid_argument);
    EXPECT_EQ(isothermal.step_index(),0); EXPECT_EQ(owned_values(isothermal.velocity()),before);
    EXPECT_FALSE(isothermal.find_turbulence_model()->sas_statistics().valid);
    BoussinesqSolver<Pack> thermal(mesh,nox_box_boundaries(),time);
    thermal.configure_turbulence(sas_options());
    EXPECT_THROW(thermal.step(),std::invalid_argument);
    EXPECT_EQ(thermal.step_index(),0); EXPECT_FALSE(thermal.find_turbulence_model()->sas_statistics().valid);
}
#endif
