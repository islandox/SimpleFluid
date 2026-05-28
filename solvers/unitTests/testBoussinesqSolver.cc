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
        const auto velocity = solver.velocity().value(lid);
        EXPECT_TRUE(std::isfinite(velocity.x));
        EXPECT_TRUE(std::isfinite(velocity.y));
        EXPECT_TRUE(std::isfinite(velocity.z));
    }
}

} // namespace

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
