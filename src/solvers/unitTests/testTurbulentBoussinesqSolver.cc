/**
 * @file testTurbulentBoussinesqSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief End-to-end tests for Problem-owned turbulence in BoussinesqSolver.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/YPlusBoundaryLayerController.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build slip velocity conditions on every box wall. @return Boundary set. */
SimpleFluid::BoundaryConditionSet slip_box_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    }
    return boundaries;
}

/** @brief Build a box with one no-slip wall and five slip walls. @return Boundary set. */
SimpleFluid::BoundaryConditionSet single_wall_box_boundaries()
{
    auto boundaries = slip_box_boundaries();
    boundaries.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    return boundaries;
}

/** @brief Build a side-heated box with a linear-temperature boundary set. */
SimpleFluid::BoundaryConditionSet buoyant_box_boundaries()
{
    auto boundaries = slip_box_boundaries();
    boundaries.temperature["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 2.0};
    boundaries.temperature["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {
            SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    return boundaries;
}

/**
 * @brief Build stable one-step options for a coupling mode.
 *
 * @param coupling Pressure-velocity coupling algorithm.
 * @return Configured time-step options.
 */
SimpleFluid::TimeStepperOptions stable_time_options(SimpleFluid::PressureVelocityCoupling coupling)
{
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-3;
    options.steps = 1;
    options.thermal_diffusivity = 1.0e-2;
    options.kinematic_viscosity = 1.0e-2;
    options.thermal_expansion = 0.0;
    options.gravity_x = 0.0;
    options.gravity_y = 0.0;
    options.gravity_z = 0.0;
    options.reference_temperature = 1.0;
    options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;
    options.pressure_velocity_coupling = coupling;
    options.n_pressure_correctors = 1;
    return options;
}

/** @brief Build baseline standard k-epsilon options. @return Turbulence options. */
SimpleFluid::TurbulenceModelOptions standard_k_epsilon_options()
{
    SimpleFluid::TurbulenceModelOptions options;
    options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    options.initial_turbulent_kinetic_energy = 0.1;
    options.initial_dissipation_rate = 0.009;
    options.min_turbulent_kinetic_energy = 1.0e-10;
    options.min_dissipation_rate = 1.0e-10;
    return options;
}

/**
 * @brief Initialize a checkerboard shear field in a Boussinesq solver.
 *
 * @param solver Solver whose velocity field is initialized.
 */
void initialize_shear(SimpleFluid::BoussinesqSolver<Pack>& solver)
{
    solver.initialize_heated_box(1.0, 1.0);
    const auto& mesh = solver.velocity().mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        const auto x = mesh.cell_centroid(cell_lid).x;
        const auto y = mesh.cell_centroid(cell_lid).y;
        solver.velocity().set_owned_value(
            cell_lid, {0.0, x < 1.0 ? (y < 1.0 ? 0.2 : -0.2) : (y < 1.0 ? -0.2 : 0.2), 0.0});
    }
    solver.velocity().sync_ghosts();
}

/**
 * @brief Assert positive finite turbulence and effective-transport fields.
 *
 * @param model Turbulence model to inspect.
 */
void expect_positive_turbulence_fields(const SimpleFluid::TurbulenceModel<Pack>& model)
{
    ASSERT_NE(model.dissipation_rate(), nullptr);
    ASSERT_EQ(model.specific_dissipation_rate(), nullptr);
    const auto& k = model.turbulent_kinetic_energy();
    const auto& epsilon = *model.dissipation_rate();
    const auto& nu_t = model.turbulent_kinematic_viscosity();
    const auto& mu_eff = model.effective_dynamic_viscosity();
    const auto& lambda_eff = model.effective_thermal_conductivity();
    for (size_t owned = 0; owned < k.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        EXPECT_TRUE(std::isfinite(k.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(epsilon.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(nu_t.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(mu_eff.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(lambda_eff.value(cell_lid)));
        EXPECT_GT(k.value(cell_lid), 0.0);
        EXPECT_GT(epsilon.value(cell_lid), 0.0);
        EXPECT_GT(nu_t.value(cell_lid), 0.0);
        EXPECT_GT(mu_eff.value(cell_lid), 0.0);
        EXPECT_GT(lambda_eff.value(cell_lid), 0.0);
    }
}

} // namespace

/** @brief Verify turbulence is Problem-owned and laminar mode can be restored. */
TEST(TurbulentBoussinesqSolverTest, ConfiguresProblemOwnedModelAndRestoresLaminarMode)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, slip_box_boundaries(),
        stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO));

    EXPECT_EQ(solver.find_turbulence_model(), nullptr);
    auto& configured = solver.configure_turbulence(standard_k_epsilon_options());
    EXPECT_EQ(solver.find_turbulence_model(), &configured);
    EXPECT_EQ(std::as_const(solver).find_turbulence_model(), &configured);
    EXPECT_TRUE(configured.enabled());
    EXPECT_EQ(configured.type(), SimpleFluid::TurbulenceModelType::StandardKEpsilon);
    EXPECT_TRUE(solver.remove_turbulence_model());
    EXPECT_EQ(solver.find_turbulence_model(), nullptr);
    EXPECT_FALSE(solver.remove_turbulence_model());
}

/** @brief Advance turbulence through segregated and coupled pressure-velocity solves. */
TEST(TurbulentBoussinesqSolverTest,
     AdvancesTurbulenceThroughSegregatedAndCoupledPressureVelocitySolves)
{
    for (const auto coupling : {SimpleFluid::PressureVelocityCoupling::PISO,
                                SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        SCOPED_TRACE("coupling=" + std::to_string(static_cast<int>(coupling)));
        auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-11;
        linear_options.max_iterations = 300;
        SimpleFluid::BoussinesqSolver<Pack> solver(mesh, slip_box_boundaries(),
                                                   stable_time_options(coupling), linear_options);
        initialize_shear(solver);

        auto& model = solver.configure_turbulence(standard_k_epsilon_options());
        const auto molecular_viscosity =
            std::as_const(solver).material_properties().dynamic_viscosity.value(0);
        solver.step();

        ASSERT_EQ(solver.step_index(), 1);
        ASSERT_EQ(solver.find_turbulence_model(), &model);
        EXPECT_TRUE(solver.last_step_statistics().converged);
        EXPECT_GE(solver.last_step_statistics().linear_solves, 3);
        EXPECT_DOUBLE_EQ(std::as_const(solver).material_properties().dynamic_viscosity.value(0),
                         molecular_viscosity);
        expect_positive_turbulence_fields(model);
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
            EXPECT_NEAR(model.effective_dynamic_viscosity().value(cell_lid),
                        molecular_viscosity + model.turbulent_kinematic_viscosity().value(cell_lid),
                        1.0e-12);
        }
    }
}

/**
 * @brief Verify that the solver supplies its Boussinesq state to turbulence.
 */
TEST(TurbulentBoussinesqSolverTest,
     PassesDirectBuoyancyContextThroughSegregatedAndCoupledSolvers)
{
    for (const auto coupling :
         {SimpleFluid::PressureVelocityCoupling::PISO,
          SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        SCOPED_TRACE(
            "coupling=" + std::to_string(static_cast<int>(coupling)));
        auto mesh = SimpleFluid::test::build_mesh<Pack>(
            SimpleFluid::test::make_2x2x2_database());
        auto time_options = stable_time_options(coupling);
        time_options.time_step = 1.0e-4;
        time_options.thermal_expansion = 1.0e-2;
        time_options.gravity_x = -1.0;
        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-11;
        linear_options.max_iterations = 400;
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh, buoyant_box_boundaries(), time_options,
            linear_options);
        solver.initialize_heated_box(2.0, 0.0);

        auto options = standard_k_epsilon_options();
        options.buoyancy_model =
            SimpleFluid::TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
        auto& model = solver.configure_turbulence(options);
        ASSERT_NE(model.buoyancy_production(), nullptr);
        ASSERT_NO_THROW(solver.step());
        ASSERT_TRUE(solver.last_step_statistics().converged);

        const auto* production = model.buoyancy_production();
        ASSERT_NE(production, nullptr);
        EXPECT_EQ(
            model.output_fields().at("buoyancy_production"),
            production);
        double minimum_production = 0.0;
        for (size_t owned = 0;
             owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            const auto value = production->value(cell_lid);
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_LE(value, 0.0);
            minimum_production = std::min(minimum_production, value);
        }
        double global_minimum_production = 0.0;
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(),
            Teuchos::REDUCE_MIN, 1,
            &minimum_production,
            &global_minimum_production);
        EXPECT_LT(global_minimum_production, 0.0);
    }
}

/** @brief Enable turbulence after a laminar implicit non-orthogonal step. */
TEST(TurbulentBoussinesqSolverTest,
     EnablesTurbulenceAfterLaminarStepWithImplicitNonOrthogonalAssembly)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
    auto time_options = stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO);
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;
    time_options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Implicit;
    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, slip_box_boundaries(), time_options);
    solver.initialize_heated_box(1.0, 1.0);

    solver.step();
    ASSERT_EQ(solver.step_index(), 1);
    ASSERT_EQ(solver.find_turbulence_model(), nullptr);

    auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
    EXPECT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_EQ(solver.find_turbulence_model(), &turbulence);
    expect_positive_turbulence_fields(turbulence);
}

/** @brief Verify turbulence expands zero-diffusivity physical matrix graphs. */
TEST(TurbulentBoussinesqSolverTest, ExpandsZeroDiffusivityPhysicalGraphsWhenTurbulenceIsEnabled)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
    auto time_options = stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO);
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;
    time_options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid;
    auto model_options = SimpleFluid::BoussinesqModelOptions::legacy_defaults(time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, slip_box_boundaries(), time_options, {},
                                               model_options);
    solver.initialize_heated_box(1.0, 1.0);

    solver.step();
    ASSERT_EQ(solver.step_index(), 1);

    auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
    EXPECT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_EQ(solver.find_turbulence_model(), &turbulence);
    expect_positive_turbulence_fields(turbulence);
}

/** @brief Exercise all wall-treatment/closure pairings through both solvers. */
TEST(TurbulentBoussinesqSolverTest,
     AdvancesWallTreatmentClosurePairingsThroughSegregatedAndCoupledSolvers)
{
    /** @brief Turbulence model and compatible wall treatment under test. */
    struct WallCase
    {
        SimpleFluid::TurbulenceModelType model;
        SimpleFluid::TurbulenceWallTreatmentType treatment;
    };
    const WallCase wall_cases[] = {
        {SimpleFluid::TurbulenceModelType::StandardKEpsilon,
         SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon},
        {SimpleFluid::TurbulenceModelType::StandardKEpsilon,
         SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReKEpsilon},
        {SimpleFluid::TurbulenceModelType::RealizableKEpsilon,
         SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReKEpsilon},
        {SimpleFluid::TurbulenceModelType::SSTKOmega,
         SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReSST}};

    for (const auto coupling : {SimpleFluid::PressureVelocityCoupling::PISO,
                                SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        for (const auto wall_case : wall_cases)
        {
            SCOPED_TRACE("coupling=" + std::to_string(static_cast<int>(coupling)) +
                         ", wall=" +
                         std::string(SimpleFluid::to_string(wall_case.treatment)));
            auto mesh = SimpleFluid::test::build_mesh<Pack>(
                SimpleFluid::test::make_2x2x2_database());
            auto time_options = stable_time_options(coupling);
            time_options.time_step = 1.0e-5;
            SimpleFluid::LinearSolverOptions linear_options;
            linear_options.tolerance = 1.0e-10;
            linear_options.max_iterations = 400;
            SimpleFluid::BoussinesqSolver<Pack> solver(
                mesh, single_wall_box_boundaries(), time_options,
                linear_options);
            initialize_shear(solver);

            SimpleFluid::TurbulenceModelOptions options;
            options.model = wall_case.model;
            options.initial_turbulent_kinetic_energy = 0.1;
            options.initial_dissipation_rate = 0.009;
            options.initial_specific_dissipation_rate = 2.0;
            options.min_turbulent_kinetic_energy = 1.0e-10;
            options.min_dissipation_rate = 1.0e-10;
            options.min_specific_dissipation_rate = 1.0e-10;
            options.wall_treatment = wall_case.treatment;
            options.wall_options.boundary_names = {"xmin"};
            if (wall_case.model == SimpleFluid::TurbulenceModelType::SSTKOmega)
                options.initial_wall_distance = 0.5;

            auto& model = solver.configure_turbulence(options);
            EXPECT_NO_THROW(solver.step());
            EXPECT_TRUE(solver.last_step_statistics().converged);
            EXPECT_EQ(solver.step_index(), 1);
            ASSERT_NE(model.wall_y_plus(), nullptr);
            ASSERT_NE(model.effective_dynamic_viscosity_boundary_cache(), nullptr);
            ASSERT_NE(model.effective_thermal_conductivity_boundary_cache(), nullptr);
            const auto& wall_statistics =
                model.wall_y_plus_statistics();
            ASSERT_EQ(wall_statistics.size(), 1U);
            EXPECT_EQ(
                wall_statistics.front().boundary_name, "xmin");
            EXPECT_EQ(
                wall_statistics.front().global_face_count, 4U);
            EXPECT_GT(wall_statistics.front().maximum, 0.0);
            for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
            {
                const auto lid = static_cast<MeshType::local_ordinal_type>(owned);
                EXPECT_TRUE(std::isfinite(model.turbulent_kinetic_energy().value(lid)));
                EXPECT_GT(model.turbulent_kinetic_energy().value(lid), 0.0);
                const auto* secondary = wall_case.model ==
                                                SimpleFluid::TurbulenceModelType::SSTKOmega
                                          ? model.specific_dissipation_rate()
                                          : model.dissipation_rate();
                ASSERT_NE(secondary, nullptr);
                EXPECT_TRUE(std::isfinite(secondary->value(lid)));
                EXPECT_GT(secondary->value(lid), 0.0);
            }
            if (wall_case.model
                == SimpleFluid::TurbulenceModelType::SSTKOmega)
            {
                SimpleFluid::YPlusBoundaryLayerControllerOptions
                    controller_options;
                controller_options.target_y_plus =
                    0.5 * wall_statistics.front().maximum;
                controller_options.adaptation_exponent = 1.0;
                controller_options.minimum_height_ratio = 0.1;
                controller_options.relative_tolerance = 0.0;
                const SimpleFluid::YPlusBoundaryLayerController
                    controller(controller_options);
                const auto update = controller.update_layer_specs(
                    {{"xmin", 4, 0.5, 1.2}},
                    wall_statistics);
                ASSERT_EQ(update.layer_specs.size(), 1U);
                EXPECT_NEAR(
                    update.layer_specs.front().first_cell_height,
                    0.25, 1.0e-14);
            }
        }
    }
}

/** @brief Verify turbulence fields are emitted only when VTU output requests them. */
TEST(TurbulentBoussinesqSolverTest, TurbulenceFieldsAreOptInForSolutionOutput)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, slip_box_boundaries(),
        stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO));
    solver.initialize_heated_box(1.0, 1.0);
    solver.configure_turbulence(standard_k_epsilon_options());

    const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path();
    const auto default_file =
        directory / ("SimpleFluid_laminar_output_" + std::to_string(unique_id) + ".vtu");
    const auto turbulence_file =
        directory / ("SimpleFluid_turbulence_output_" + std::to_string(unique_id) + ".vtu");
    solver.write_solution_vtu(default_file.string());
    solver.write_solution_vtu(turbulence_file.string(), {.include_turbulence_fields = true});

    auto read = [](const std::filesystem::path& path)
    {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    const auto default_contents = read(default_file);
    const auto turbulence_contents = read(turbulence_file);
    EXPECT_EQ(default_contents.find("Name=\"nu_t\""), std::string::npos);
    EXPECT_EQ(default_contents.find("Name=\"epsilon\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"k\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"epsilon\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"nu_t\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"mu_eff\""), std::string::npos);
    EXPECT_NE(turbulence_contents.find("Name=\"lambda_eff\""), std::string::npos);

    std::filesystem::remove(default_file);
    std::filesystem::remove(turbulence_file);
}
