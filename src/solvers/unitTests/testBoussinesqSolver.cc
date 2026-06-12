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
#include <limits>
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

SimpleFluid::SP<MeshType> make_checkerboard_box_mesh()
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
            SimpleFluid::ArrString{
                "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

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
    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        if (mesh.boundary_batch_name(batch_id) != boundary_name)
        {
            continue;
        }

        saw_boundary = true;
        for (const auto face_lid : batch.face_lids)
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
    const SimpleFluid::CellField<Pack>& pressure,
    Pack::scalar_type pressure_coefficient,
    const SimpleFluid::FVM::VelocityBoundaryCache<Pack>& cache)
{
    SimpleFluid::FaceField<Pack> face_fluxes(velocity.mesh_ptr(), "face_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, pressure_coefficient, cache, face_fluxes);

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

TEST(BoussinesqSolverTest, RunsCartesianCRTPMeshSmokeCase)
{
    auto cartesian =
        std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                {0.0, 0.5, 1.0},
                {0.0, 1.0},
                {0.0, 1.0}}});
    auto mesh =
        std::make_shared<SimpleFluid::MeshHandle<Pack>>(cartesian);

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 1;
    time_options.thermal_diffusivity = 1.0e-2;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, bcs, time_options);
    solver.initialize_heated_box(1.0, 0.0);
    solver.run();

    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_TRUE(std::isfinite(solver.temperature().value(0)));
    EXPECT_TRUE(std::isfinite(solver.pressure().value(0)));
    const auto velocity = solver.velocity().value(0);
    EXPECT_TRUE(std::isfinite(velocity.x));
    EXPECT_TRUE(std::isfinite(velocity.y));
    EXPECT_TRUE(std::isfinite(velocity.z));
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
    double simple_continuity = std::numeric_limits<double>::infinity();
    double coupled_continuity = std::numeric_limits<double>::infinity();
    const std::vector<SimpleFluid::PressureVelocityCoupling> modes{
        SimpleFluid::PressureVelocityCoupling::SIMPLE,
        SimpleFluid::PressureVelocityCoupling::PISO,
        SimpleFluid::PressureVelocityCoupling::PIMPLE,
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov
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

        const auto after = continuity_imbalance_norm(
            solver.velocity(), solver.pressure(),
            time_options.time_step, cache);
        const auto residuals = solver.last_pressure_velocity_residuals();
        const auto statistics = solver.last_step_statistics();

        EXPECT_EQ(solver.step_index(), 1);
        expect_finite_solution(*mesh, solver);
        EXPECT_NEAR(after, residuals.continuity, 1.0e-10);
        EXPECT_TRUE(std::isfinite(residuals.momentum));
        EXPECT_TRUE(std::isfinite(residuals.pressure));
        EXPECT_TRUE(std::isfinite(residuals.continuity));
        EXPECT_GE(residuals.momentum, 0.0);
        EXPECT_GE(residuals.pressure, 0.0);
        EXPECT_GE(residuals.continuity, 0.0);
        EXPECT_TRUE(statistics.converged);
        EXPECT_GE(statistics.nonlinear_iterations, 1);
        EXPECT_GE(statistics.linear_solves, 1);
        EXPECT_GE(statistics.krylov_iterations, 0);
        EXPECT_NEAR(statistics.momentum, residuals.momentum, 1.0e-12);
        EXPECT_NEAR(statistics.pressure, residuals.pressure, 1.0e-12);
        EXPECT_NEAR(statistics.continuity, residuals.continuity, 1.0e-12);
        if (mode == SimpleFluid::PressureVelocityCoupling::SIMPLE)
        {
            simple_continuity = residuals.continuity;
        }
        if (mode == SimpleFluid::PressureVelocityCoupling::CoupledKrylov)
        {
            coupled_continuity = residuals.continuity;
            EXPECT_GE(residuals.linear_iterations, 0);
            EXPECT_TRUE(std::isfinite(residuals.achieved_tolerance));
            EXPECT_GE(residuals.achieved_tolerance, 0.0);
        }
    }

    EXPECT_LT(simple_continuity, 1.0e-10);
    EXPECT_LT(coupled_continuity, 1.0e-10);
}

TEST(BoussinesqSolverTest, CoupledKrylovAssemblesVelocityPressureBlocks)
{
    auto mesh = make_checkerboard_box_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    SimpleFluid::CellField<Pack> temperature(mesh, 1.0, "temperature");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_fluxes");
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FVM::face_fluxes(
        velocity, cache, face_fluxes);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-2;
    options.kinematic_viscosity = 1.0e-2;
    options.thermal_expansion = 1.0;
    options.gravity_x = -1.0;
    options.reference_temperature = 0.0;

    SimpleFluid::BoussinesqMomentumEquation<Pack>
        momentum_equation(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack>
        coupled_solver(mesh);
    const auto system = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        temperature,
        face_fluxes,
        cache,
        bcs,
        options);

    EXPECT_EQ(system.map->getLocalNumElements(),
              4 * mesh->num_owned_cells());
    EXPECT_EQ(system.matrix->getLocalNumRows(),
              4 * mesh->num_owned_cells());
    EXPECT_EQ(system.schur->getLocalNumRows(),
              mesh->num_owned_cells());
    EXPECT_GT(system.matrix->getGlobalNumEntries(), 0U);
    EXPECT_GT(system.pressure_stabilization->getGlobalNumEntries(), 0U);
    EXPECT_GT(system.schur->getGlobalNumEntries(), 0U);

    typename Pack::vector_type checkerboard(
        mesh->owned_cell_map(), true);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        const auto parity =
            static_cast<int>(center.x * 4.0)
          + static_cast<int>(center.y * 4.0)
          + static_cast<int>(center.z * 4.0);
        checkerboard.replaceLocalValue(
            cell_lid, parity % 2 == 0 ? 1.0 : -1.0);
    }
    typename Pack::vector_type stabilized(
        mesh->owned_cell_map(), true);
    system.pressure_stabilization->apply(
        checkerboard, stabilized);
    EXPECT_GT(stabilized.norm2(), 1.0e-12);
    EXPECT_GT(checkerboard.dot(stabilized), 0.0);
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

TEST(BoussinesqSolverTest, PhysicalHeatSourcesGiveAnalyticalOneCellRise)
{
    auto mesh = make_single_cell_box_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.thermal_expansion = 0.0;
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 2.0;
    model_options.density = 2.0;
    model_options.specific_heat_capacity = 5.0;
    model_options.dynamic_viscosity = 0.0;
    model_options.thermal_conductivity = 0.0;
    model_options.temperature_source_names = {"heating", "sink"};
    model_options.temperature_source_power_densities = {20.0, -5.0};

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh,
        {},
        time_options,
        {},
        model_options);
    solver.temperature().put_scalar(10.0);
    solver.step();

    EXPECT_NEAR(
        solver.temperature().value(0),
        10.0 + 0.1 * 15.0 / (2.0 * 5.0),
        1.0e-12);
}

TEST(BoussinesqSolverTest, RefreshesMaterialBeforeSourcesAtStepStart)
{
    auto mesh = make_single_cell_box_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.thermal_expansion = 0.0;
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 2.0;
    model_options.density = 2.0;
    model_options.specific_heat_capacity = 4.0;
    model_options.dynamic_viscosity = 0.0;
    model_options.thermal_conductivity = 0.0;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.temperature().put_scalar(10.0);

    std::vector<std::string> updates;
    solver.set_material_updater(
        [&](const auto& context, auto& material)
        {
            updates.push_back("material");
            EXPECT_DOUBLE_EQ(context.time, 0.0);
            EXPECT_EQ(context.step_index, 0);
            material.specific_heat_capacity.put_scalar(5.0);
        });
    auto& zeta_source = solver.add_temperature_source("zeta");
    zeta_source.set_updater(
        [&](const auto& context, auto& field)
        {
            updates.push_back("zeta");
            EXPECT_DOUBLE_EQ(context.temperature.value(0), 10.0);
            field.put_scalar(0.0);
        });
    auto& alpha_source = solver.add_temperature_source("alpha");
    alpha_source.set_updater(
        [&](const auto& context, auto& field)
        {
            updates.push_back("alpha");
            EXPECT_DOUBLE_EQ(context.temperature.value(0), 10.0);
            field.put_scalar(20.0);
        });

    solver.step();

    EXPECT_EQ(
        updates,
        (std::vector<std::string>{"material", "alpha", "zeta"}));
    EXPECT_DOUBLE_EQ(
        solver.material_properties()
            .specific_heat_capacity.value(0),
        5.0);
    EXPECT_DOUBLE_EQ(alpha_source.field().value(0), 20.0);
    EXPECT_DOUBLE_EQ(zeta_source.field().value(0), 0.0);
    EXPECT_NEAR(solver.temperature().value(0), 10.2, 1.0e-12);
}

TEST(BoussinesqSolverTest, DensityFeedbackUsesReferenceDensityBuoyancy)
{
    auto mesh = make_single_cell_box_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;
    time_options.gravity_x = 0.0;
    time_options.gravity_y = 0.0;
    time_options.gravity_z = -10.0;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 1.0;
    model_options.density = 0.8;
    model_options.specific_heat_capacity = 1.0;
    model_options.dynamic_viscosity = 0.0;
    model_options.thermal_conductivity = 0.0;
    model_options.density_feedback_enabled = true;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.temperature().put_scalar(1.0);
    solver.step();

    const auto velocity = solver.velocity().value(0);
    EXPECT_NEAR(velocity.x, 0.0, 1.0e-12);
    EXPECT_NEAR(velocity.y, 0.0, 1.0e-12);
    EXPECT_NEAR(velocity.z, 0.2, 1.0e-12);
}

TEST(BoussinesqSolverTest, PhysicalDefaultsMatchLegacySolver)
{
    auto legacy_mesh = make_box_mesh();
    auto physical_mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    for (const auto* name :
         {"ymin", "ymax", "zmin", "zmax"})
    {
        bcs.temperature[name] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.01;
    time_options.thermal_diffusivity = 0.01;
    time_options.kinematic_viscosity = 0.01;

    SimpleFluid::BoussinesqSolver<Pack> legacy(
        legacy_mesh, bcs, time_options);
    SimpleFluid::BoussinesqSolver<Pack> physical(
        physical_mesh,
        bcs,
        time_options,
        {},
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options));
    legacy.initialize_heated_box(1.0, 0.0);
    physical.initialize_heated_box(1.0, 0.0);
    legacy.step();
    physical.step();

    for (size_t owned = 0;
         owned < legacy_mesh->num_owned_cells();
         ++owned)
    {
        const auto lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        EXPECT_NEAR(
            legacy.temperature().value(lid),
            physical.temperature().value(lid),
            1.0e-11);
        const auto legacy_velocity = legacy.velocity().value(lid);
        const auto physical_velocity = physical.velocity().value(lid);
        EXPECT_NEAR(
            legacy_velocity.x, physical_velocity.x, 1.0e-11);
        EXPECT_NEAR(
            legacy_velocity.y, physical_velocity.y, 1.0e-11);
        EXPECT_NEAR(
            legacy_velocity.z, physical_velocity.z, 1.0e-11);
    }
}

TEST(BoussinesqSolverTest, AuxiliaryFieldsAreOptInForVtuOutput)
{
    auto mesh = make_single_cell_box_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.add_temperature_source("qdot_test", 12.0);

    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base =
        std::filesystem::temp_directory_path()
      / ("SimpleFluid_phase10_" + std::to_string(unique_id));
    const auto default_file = base.string() + "_default.vtu";
    const auto auxiliary_file = base.string() + "_auxiliary.vtu";

    solver.write_solution_vtu(default_file);
    solver.write_solution_vtu(
        auxiliary_file,
        SimpleFluid::SolutionOutputOptions{
            .include_sources = true,
            .include_material_properties = true});

    auto read_file =
        [](const std::string& filename)
    {
        std::ifstream input(filename);
        return std::string(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    };
    const auto default_contents = read_file(default_file);
    const auto auxiliary_contents = read_file(auxiliary_file);

    EXPECT_EQ(
        default_contents.find("Name=\"qdot_test\""),
        std::string::npos);
    EXPECT_EQ(
        default_contents.find("Name=\"density\""),
        std::string::npos);
    EXPECT_NE(
        auxiliary_contents.find("Name=\"qdot_test\""),
        std::string::npos);
    EXPECT_NE(
        auxiliary_contents.find("Name=\"density\""),
        std::string::npos);
    EXPECT_NE(
        auxiliary_contents.find(
            "Name=\"specific_heat_capacity\""),
        std::string::npos);
    EXPECT_NE(
        auxiliary_contents.find("Name=\"dynamic_viscosity\""),
        std::string::npos);
    EXPECT_NE(
        auxiliary_contents.find("Name=\"thermal_conductivity\""),
        std::string::npos);

    std::filesystem::remove(default_file);
    std::filesystem::remove(auxiliary_file);
}

TEST(BoussinesqSolverTest, RejectsNonFiniteSourceUpdaterOutput)
{
    auto mesh = make_single_cell_box_mesh();
    const SimpleFluid::TimeStepperOptions time_options;
    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    auto& source = solver.add_temperature_source("invalid");
    source.set_updater(
        [](const auto&, auto& field)
        {
            field.put_scalar(
                std::numeric_limits<double>::quiet_NaN());
        });

    EXPECT_THROW(solver.step(), std::invalid_argument);
}

TEST(BoussinesqSolverTest, CoupledKrylovUsesPhysicalMaterialPath)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.01;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;
    time_options.kinematic_viscosity = 0.01;
    time_options.thermal_diffusivity = 0.0;
    time_options.thermal_expansion = 0.0;

    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    model_options.dynamic_viscosity = 0.02;
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, bcs, time_options, {}, model_options);
    solver.temperature().put_scalar(0.5);
    solver.step();

    expect_finite_solution(*mesh, solver);
    EXPECT_GT(solver.last_step_statistics().linear_solves, 0);
}
