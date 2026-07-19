/**
 * @file testTurbulentBoussinesqSolver.cc
 * @brief End-to-end tests for Problem-owned turbulence in BoussinesqSolver.
 */

#include <gtest/gtest.h>

#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

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

SimpleFluid::BoundaryConditionSet slip_box_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    }
    return boundaries;
}

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
