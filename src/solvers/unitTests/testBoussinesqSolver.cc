/**
 * @file testBoussinesqSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief smoke tests for the Boussinesq solver
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "FVM/Operators.hh"
#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_box_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

SimpleFluid::SP<MeshType> make_single_cell_box_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

SimpleFluid::SP<MeshType> make_boundary_layer_box_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.25});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 0.25, 0.5, 0.75, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 0.25, 0.5, 0.75, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 0.25, 0.5, 0.75, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    db->set("boundary_layer_boundary_names",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    db->set("boundary_layer_counts", SimpleFluid::ArrInt{1, 1, 1, 1, 1, 1});
    db->set("boundary_layer_first_cell_heights",
            SimpleFluid::ArrReal{0.05, 0.05, 0.05, 0.05, 0.05, 0.05});
    db->set("boundary_layer_growth_ratios",
            SimpleFluid::ArrReal{1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

SimpleFluid::SP<MeshType> make_cylinder_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("height", SimpleFluid::real_t{2.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"radial", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

SimpleFluid::SP<MeshType> make_boundaried_cylinder_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.75});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("height", SimpleFluid::real_t{2.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"radial", "zmin", "zmax"});
    db->set("boundary_layer_boundary_names",
            SimpleFluid::ArrString{"radial", "zmin", "zmax"});
    db->set("boundary_layer_counts", SimpleFluid::ArrInt{1, 1, 1});
    db->set("boundary_layer_first_cell_heights",
            SimpleFluid::ArrReal{0.1, 0.1, 0.1});
    db->set("boundary_layer_growth_ratios",
            SimpleFluid::ArrReal{1.0, 1.0, 1.0});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

SimpleFluid::SP<MeshType> make_split_sphere_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::SPHERE));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"lower_surface", "upper_surface"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

void expect_finite_solution(const MeshType& mesh,
                            const SimpleFluid::BoussinesqSolver<Pack>& solver)
{
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh.num_owned_cells());
         ++lid)
    {
        EXPECT_TRUE(std::isfinite(solver.temperature().value(lid)));
        EXPECT_TRUE(std::isfinite(solver.pressure().value(lid)));
        const auto velocity = solver.velocity().value(lid);
        EXPECT_TRUE(std::isfinite(velocity.x));
        EXPECT_TRUE(std::isfinite(velocity.y));
        EXPECT_TRUE(std::isfinite(velocity.z));
    }
}

void expect_zero_boundary_velocity(
    const MeshType& mesh,
    const SimpleFluid::VectorFaceField<Pack>& face_velocity,
    const SimpleFluid::FaceField<Pack>& face_fluxes,
    const char* boundary_name)
{
    bool saw_boundary = false;
    for (const auto& [patch_id, patch] : mesh.boundary_patches())
    {
        if (mesh.boundary_patch_name(patch_id) != boundary_name)
        {
            continue;
        }

        saw_boundary = true;
        for (const auto face_lid : patch.face_lids)
        {
            if (!face_velocity.is_owned_face(face_lid))
            {
                continue;
            }

            const auto value = face_velocity.value(face_lid);
            EXPECT_NEAR(value.x, 0.0, 1.0e-12);
            EXPECT_NEAR(value.y, 0.0, 1.0e-12);
            EXPECT_NEAR(value.z, 0.0, 1.0e-12);
            EXPECT_NEAR(face_fluxes.value(face_lid), 0.0, 1.0e-12);
        }
    }

    EXPECT_TRUE(saw_boundary);
}

Pack::scalar_type continuity_imbalance_norm(
    const SimpleFluid::VectorCellField<Pack>& velocity,
    const SimpleFluid::FVM::VelocityBoundaryCache<Pack>& cache)
{
    SimpleFluid::FaceField<Pack> face_fluxes(velocity.mesh_ptr(), "face_flux");
    SimpleFluid::FVM::face_fluxes(velocity, cache, face_fluxes);

    Pack::scalar_type norm_squared = 0.0;
    for (size_t owned = 0; owned < velocity.mesh().num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto value =
            SimpleFluid::FVM::cell_flux_balance<Pack>(
                velocity.mesh(), face_fluxes, cell_lid);
        norm_squared += value * value;
    }

    return std::sqrt(norm_squared);
}

} // namespace

/**
 * @brief Verifies a single buoyancy step produces the analytical velocity increment in all three components.
 */
TEST(BoussinesqSolverTest, OneCellBuoyancyStepMatchesAnalyticalVelocityIncrement)
{
    auto mesh = make_single_cell_box_mesh();

    SimpleFluid::BoundaryConditionSet bcs;

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.steps = 1;
    time_options.thermal_diffusivity = 0.0;
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_expansion = 2.0;
    time_options.gravity_x = -10.0;
    time_options.gravity_y = -20.0;
    time_options.gravity_z = -30.0;
    time_options.reference_temperature = 0.5;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options,
                                               linear_options);
    solver.initialize_heated_box(1.5, 0.0);
    solver.run();

    ASSERT_EQ(mesh->num_owned_cells(), 1U);
    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_NEAR(solver.time(), time_options.time_step, 1.0e-14);
    EXPECT_NEAR(solver.temperature().value(0), 1.5, 1.0e-12);
    EXPECT_NEAR(solver.pressure().value(0), 0.0, 1.0e-12);

    const auto velocity = solver.velocity().value(0);
    EXPECT_NEAR(velocity.x, 2.0, 1.0e-10);
    EXPECT_NEAR(velocity.y, 4.0, 1.0e-10);
    EXPECT_NEAR(velocity.z, 6.0, 1.0e-10);
}

/**
 * @brief Runs a small heated-box simulation for 2 steps and checks finite temperature and velocity values.
 */
TEST(BoussinesqSolverTest, RunsHeatedBoxSmokeCase)
{
    auto mesh = make_box_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 2;
    time_options.thermal_diffusivity = 1.0e-2;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options, linear_options);
    solver.initialize_heated_box(1.0, 0.0);
    solver.run();

    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_GT(solver.time(), 0.0);
    expect_finite_solution(*mesh, solver);

    const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_file = std::filesystem::temp_directory_path()
                           / ("SimpleFluid_testBoussinesq_solution_"
                              + std::to_string(unique_id) + ".vtu");
    solver.write_solution_vtu(output_file.string());

    std::ifstream input(output_file);
    ASSERT_TRUE(input.good());
    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    EXPECT_NE(contents.find("Name=\"temperature\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"pressure\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"velocity\""), std::string::npos);
    EXPECT_NE(contents.find("NumberOfComponents=\"3\""), std::string::npos);

    std::filesystem::remove(output_file);
}

/**
 * @brief Runs a bottom-hot cylinder vessel simulation for 2 steps.
 */
TEST(BoussinesqSolverTest, RunsBottomHotCylinderSmokeCase)
{
    auto mesh = make_cylinder_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["zmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    bcs.temperature["radial"] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 2;
    time_options.thermal_diffusivity = 1.0e-2;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options);
    solver.initialize_bottom_hot_top_cold(1.0, 0.0);
    solver.run();

    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_GT(solver.time(), 0.0);
    expect_finite_solution(*mesh, solver);
}

/**
 * @brief Runs a bottom-hot cylinder vessel with boundary layers and no-slip walls.
 */
TEST(BoussinesqSolverTest, RunsBoundariedCylinderSmokeCase)
{
    auto mesh = make_boundaried_cylinder_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["zmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    bcs.temperature["radial"] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    for (const auto* name : {"radial", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 2;
    time_options.thermal_diffusivity = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.reference_temperature = 0.5;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 100;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options,
                                               linear_options);
    solver.initialize_bottom_hot_top_cold(1.0, 0.0);
    solver.run();

    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_GT(solver.time(), 0.0);
    expect_finite_solution(*mesh, solver);

    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_flux");
    SimpleFluid::FVM::face_velocities(solver.velocity(), cache,
                                               face_velocity);
    SimpleFluid::FVM::normal_face_fluxes(face_velocity, face_fluxes);

    expect_zero_boundary_velocity(*mesh, face_velocity, face_fluxes, "radial");
    expect_zero_boundary_velocity(*mesh, face_velocity, face_fluxes, "zmin");
    expect_zero_boundary_velocity(*mesh, face_velocity, face_fluxes, "zmax");
}

/**
 * @brief Runs a bottom-hot sphere vessel simulation for 2 steps.
 */
TEST(BoussinesqSolverTest, RunsBottomHotSphereSmokeCase)
{
    auto mesh = make_split_sphere_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["lower_surface"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["upper_surface"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 2;
    time_options.thermal_diffusivity = 1.0e-2;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options);
    solver.initialize_bottom_hot_top_cold(1.0, 0.0);
    solver.run();

    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_GT(solver.time(), 0.0);
    expect_finite_solution(*mesh, solver);
}

/**
 * @brief Runs a boundary-layer box simulation with three-directional gravity forcing and NoSlip walls.
 */
TEST(BoussinesqSolverTest, RunsBoundaryLayerBoxWithThreeDirectionGravity)
{
    auto mesh = make_boundary_layer_box_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 1;
    time_options.thermal_diffusivity = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.gravity_x = -1.0;
    time_options.gravity_y = -2.0;
    time_options.gravity_z = -9.81;
    time_options.reference_temperature = 0.5;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options);
    solver.initialize_heated_box(1.0, 0.0);
    solver.run();

    EXPECT_EQ(solver.step_index(), 1);
    expect_finite_solution(*mesh, solver);
}

TEST(BoussinesqSolverTest, RunsPressureVelocityCouplingModesAndReportsResiduals)
{
    const std::vector<SimpleFluid::PressureVelocityCoupling> modes{
        SimpleFluid::PressureVelocityCoupling::SIMPLE,
        SimpleFluid::PressureVelocityCoupling::PISO,
        SimpleFluid::PressureVelocityCoupling::PIMPLE
    };

    for (const auto mode : modes)
    {
        auto mesh = make_box_mesh();

        SimpleFluid::BoundaryConditionSet bcs;
        bcs.temperature["xmin"] =
            {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
        bcs.temperature["xmax"] =
            {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

        SimpleFluid::TimeStepperOptions time_options;
        time_options.time_step = 1.0e-2;
        time_options.steps = 1;
        time_options.thermal_diffusivity = 0.0;
        time_options.kinematic_viscosity = 1.0e-2;
        time_options.thermal_expansion = 1.0;
        time_options.gravity_x = -1.0;
        time_options.gravity_y = -0.5;
        time_options.gravity_z = -0.25;
        time_options.reference_temperature = 0.5;
        time_options.pressure_velocity_coupling = mode;
        time_options.n_pressure_correctors = 2;
        time_options.n_outer_correctors = 2;

        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-12;
        linear_options.max_iterations = 200;

        SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options,
                                                   linear_options);
        solver.initialize_heated_box(1.0, 0.0);

        const auto cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
                mesh, bcs);
        solver.step();

        const auto after = continuity_imbalance_norm(solver.velocity(), cache);
        const auto residuals = solver.last_pressure_velocity_residuals();

        EXPECT_EQ(solver.step_index(), 1);
        expect_finite_solution(*mesh, solver);
        EXPECT_NEAR(after, residuals.continuity, 1.0e-10);
        EXPECT_TRUE(std::isfinite(residuals.momentum));
        EXPECT_TRUE(std::isfinite(residuals.pressure));
        EXPECT_TRUE(std::isfinite(residuals.continuity));
        EXPECT_GE(residuals.momentum, 0.0);
        EXPECT_GE(residuals.pressure, 0.0);
        EXPECT_GE(residuals.continuity, 0.0);
    }
}

TEST(BoussinesqSolverTest, RejectsInvalidPressureVelocityLoopCounts)
{
    auto mesh = make_box_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TimeStepperOptions time_options;
    time_options.n_pressure_correctors = 0;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options);
    solver.initialize_heated_box(1.0, 0.0);

    EXPECT_THROW(solver.step(), std::invalid_argument);
}
