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
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
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

/** @brief Build the standard 2-by-2-by-2 box test mesh. @return Assembled mesh. */
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

/** @brief Build the 4-by-4-by-4 checkerboard test mesh. @return Assembled mesh. */
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

/** @brief Build a single-cell box test mesh. @return Assembled mesh. */
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

/** @brief Build a box refined toward all six boundaries. @return Assembled mesh. */
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

/** @brief Build the baseline cylindrical test mesh. @return Assembled mesh. */
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

/** @brief Build a cylinder with boundary-layer refinement. @return Assembled mesh. */
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

/** @brief Build a sphere split into upper and lower boundary patches. @return Assembled mesh. */
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

/**
 * @brief Assert that all primary solution values are finite.
 *
 * @param mesh Mesh defining owned cells.
 * @param solver Solver containing fields to inspect.
 */
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

/**
 * @brief Assert zero boundary velocity and flux on a named patch.
 *
 * @param mesh Mesh defining boundary batches.
 * @param face_velocity Reconstructed face velocity.
 * @param face_fluxes Face-normal volumetric flux.
 * @param boundary_name Patch to inspect.
 */
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

/**
 * @brief Compute the cellwise L2 norm of pressure-weighted flux imbalance.
 *
 * @param velocity Cell-centered velocity field.
 * @param pressure Cell-centered pressure field.
 * @param pressure_coefficient Pressure contribution coefficient.
 * @param cache Cached velocity boundary conditions.
 * @param pressure_boundaries Pressure boundary conditions.
 * @return Square-root sum of squared cell flux balances.
 */
Pack::scalar_type continuity_imbalance_norm(
    const SimpleFluid::VectorCellField<Pack>& velocity,
    const SimpleFluid::CellField<Pack>& pressure,
    Pack::scalar_type pressure_coefficient,
    const SimpleFluid::FVM::VelocityBoundaryCache<Pack>& cache,
    const SimpleFluid::BoundaryConditionMap& pressure_boundaries)
{
    SimpleFluid::FaceField<Pack> face_fluxes(velocity.mesh_ptr(), "face_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity,
        pressure,
        pressure_coefficient,
        cache,
        pressure_boundaries,
        face_fluxes);

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

/** @brief Verify the solver accepts the Cartesian CRTP mesh backend. */
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

/** @brief Exercise SIMPLE, PISO, PIMPLE, and coupled modes and inspect residuals. */
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
        SCOPED_TRACE(
            "pressure-velocity coupling mode "
            + std::to_string(static_cast<int>(mode)));
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
            time_options.time_step, cache, bcs.pressure);
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

/**
 * @brief Stored gauge pressure scales with density while velocity does not.
 */
TEST(BoussinesqSolverTest,
     StoresPressureInPascalsAcrossSegregatedAndCoupledSolves)
{
    /** @brief Pressure and velocity snapshots returned by one coupling run. */
    struct CaseState
    {
        std::vector<double> pressure;
        std::vector<MeshType::Vec3> velocity;
    };

    auto run_case = [](
        SimpleFluid::PressureVelocityCoupling coupling,
        double reference_density)
    {
        auto mesh = make_box_mesh();
        SimpleFluid::BoundaryConditionSet bcs;
        bcs.temperature["xmin"] =
            {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
        bcs.temperature["xmax"] =
            {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

        SimpleFluid::TimeStepperOptions time_options;
        time_options.time_step = 1.0e-2;
        time_options.steps = 2;
        time_options.thermal_diffusivity = 0.0;
        time_options.kinematic_viscosity = 1.0e-2;
        time_options.thermal_expansion = 1.0;
        time_options.gravity_x = -1.0;
        time_options.gravity_y = -0.5;
        time_options.gravity_z = -0.25;
        time_options.reference_temperature = 0.5;
        time_options.pressure_velocity_coupling = coupling;
        time_options.n_pressure_correctors = 2;

        SimpleFluid::BoussinesqModelOptions model_options;
        model_options.reference_density = reference_density;
        model_options.density = reference_density;
        model_options.specific_heat_capacity = 1.0;
        model_options.dynamic_viscosity =
            reference_density * time_options.kinematic_viscosity;
        model_options.thermal_conductivity = 0.0;

        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-12;
        linear_options.max_iterations = 200;

        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            bcs,
            time_options,
            linear_options,
            model_options);
        solver.initialize_heated_box(1.0, 0.0);
        solver.run();

        CaseState state;
        state.pressure.reserve(mesh->num_owned_cells());
        state.velocity.reserve(mesh->num_owned_cells());
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            state.pressure.push_back(solver.pressure().value(cell_lid));
            state.velocity.push_back(solver.velocity().value(cell_lid));
        }
        return state;
    };

    constexpr double water_density = 1000.0;
    for (const auto coupling : {
             SimpleFluid::PressureVelocityCoupling::PISO,
             SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        const auto unit_density = run_case(coupling, 1.0);
        const auto water = run_case(coupling, water_density);
        ASSERT_EQ(water.pressure.size(), unit_density.pressure.size());
        ASSERT_EQ(water.velocity.size(), unit_density.velocity.size());

        double maximum_pressure = 0.0;
        for (size_t cell = 0; cell < unit_density.pressure.size(); ++cell)
        {
            maximum_pressure = std::max(
                maximum_pressure,
                std::abs(unit_density.pressure[cell]));
            const auto pressure_tolerance = std::max(
                1.0e-10,
                std::abs(unit_density.pressure[cell]) * 1.0e-8);
            EXPECT_NEAR(
                water.pressure[cell] / water_density,
                unit_density.pressure[cell],
                pressure_tolerance);

            const auto velocity_tolerance = 1.0e-9;
            EXPECT_NEAR(
                water.velocity[cell].x,
                unit_density.velocity[cell].x,
                velocity_tolerance);
            EXPECT_NEAR(
                water.velocity[cell].y,
                unit_density.velocity[cell].y,
                velocity_tolerance);
            EXPECT_NEAR(
                water.velocity[cell].z,
                unit_density.velocity[cell].z,
                velocity_tolerance);
        }
        EXPECT_GT(maximum_pressure, 1.0e-8);
    }
}

/** @brief Verify coupled Krylov assembly produces all velocity-pressure blocks. */
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
    EXPECT_DOUBLE_EQ(system.reference_density, 1.0);

    const auto gauge_cell_gid =
        mesh->owned_cell_map()->getMinAllGlobalIndex();
    const auto coupled_gauge_row_gid = 4 * gauge_cell_gid + 3;
    const auto coupled_gauge_row_lid =
        system.matrix->getRowMap()->getLocalElement(
            coupled_gauge_row_gid);
    typename Pack::matrix_type::local_inds_host_view_type
        gauge_columns;
    typename Pack::matrix_type::values_host_view_type gauge_values;
    system.matrix->getLocalRowView(
        coupled_gauge_row_lid, gauge_columns, gauge_values);
    ASSERT_EQ(gauge_columns.extent(0), 1U);
    EXPECT_EQ(
        system.matrix->getColMap()->getGlobalElement(
            gauge_columns[0]),
        coupled_gauge_row_gid);
    EXPECT_DOUBLE_EQ(gauge_values[0], 1.0);
    const auto coupled_rhs_view = system.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    EXPECT_DOUBLE_EQ(
        coupled_rhs_view(coupled_gauge_row_lid, 0), 0.0);

    const auto schur_gauge_row_lid =
        system.schur->getRowMap()->getLocalElement(gauge_cell_gid);
    system.schur->getLocalRowView(
        schur_gauge_row_lid, gauge_columns, gauge_values);
    ASSERT_EQ(gauge_columns.extent(0), 1U);
    EXPECT_EQ(
        system.schur->getColMap()->getGlobalElement(
            gauge_columns[0]),
        gauge_cell_gid);
    EXPECT_DOUBLE_EQ(gauge_values[0], 1.0);

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

/** @brief Verify coupled setup reuse is observable and numerically neutral. */
TEST(BoussinesqSolverTest,
     CoupledKrylovReusesCompatibleSetupAndMatchesForcedRebuild)
{
    struct RunResult
    {
        std::vector<MeshType::Vec3> velocity;
        std::vector<double> pressure;
        SimpleFluid::CoupledPressureVelocityCacheStatistics statistics;
    };

    auto run = [](SimpleFluid::CoupledRebuildPolicy policy)
    {
        auto mesh = make_box_mesh();
        SimpleFluid::BoundaryConditionSet boundaries;
        for (const auto* name :
             {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        {
            boundaries.velocity[name] = {
                SimpleFluid::BoundaryConditionType::NoSlip, {}};
        }
        const auto boundary_cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
                mesh, boundaries);

        SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
        SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
        SimpleFluid::CellField<Pack> temperature(mesh, "temperature");
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            temperature.set_owned_value(
                cell_lid, mesh->cell_centroid(cell_lid).x);
        }
        temperature.sync_ghosts();
        SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_fluxes");

        SimpleFluid::TimeStepperOptions time_options;
        time_options.time_step = 1.0e-2;
        time_options.kinematic_viscosity = 1.0e-2;
        time_options.thermal_expansion = 1.0;
        time_options.gravity_x = -1.0;
        time_options.reference_temperature = 0.5;
        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-12;
        linear_options.max_iterations = 200;

        SimpleFluid::BoussinesqMomentumEquation<Pack> momentum(mesh);
        SimpleFluid::CoupledPressureVelocitySolver<Pack> solver(mesh);
        solver.set_rebuild_policy(policy);
        for (size_t step = 0; step < 2; ++step)
        {
            SimpleFluid::FVM::face_fluxes(
                velocity, boundary_cache, face_fluxes);
            const auto system = solver.assemble(
                momentum,
                velocity,
                pressure,
                temperature,
                face_fluxes,
                boundary_cache,
                boundaries,
                time_options);
            const auto result = solver.solve(
                system, velocity, pressure, linear_options);
            EXPECT_TRUE(result.converged);
        }

        RunResult result;
        result.statistics = solver.cache_statistics();
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            result.velocity.push_back(velocity.value(cell_lid));
            result.pressure.push_back(pressure.value(cell_lid));
        }
        return result;
    };

    const auto cached = run(
        SimpleFluid::CoupledRebuildPolicy::OnOperatorGraphChange);
    const auto rebuilt = run(SimpleFluid::CoupledRebuildPolicy::Always);
    ASSERT_EQ(cached.velocity.size(), rebuilt.velocity.size());
    ASSERT_EQ(cached.pressure.size(), rebuilt.pressure.size());
    for (size_t cell = 0; cell < cached.velocity.size(); ++cell)
    {
        EXPECT_NEAR(cached.velocity[cell].x, rebuilt.velocity[cell].x, 1.0e-12);
        EXPECT_NEAR(cached.velocity[cell].y, rebuilt.velocity[cell].y, 1.0e-12);
        EXPECT_NEAR(cached.velocity[cell].z, rebuilt.velocity[cell].z, 1.0e-12);
        EXPECT_NEAR(cached.pressure[cell], rebuilt.pressure[cell], 1.0e-12);
    }

    EXPECT_EQ(cached.statistics.coupled_map_builds, 1U);
    EXPECT_EQ(cached.statistics.static_geometry_builds, 1U);
    EXPECT_EQ(cached.statistics.static_geometry_reuses, 1U);
    EXPECT_EQ(cached.statistics.matrix_graph_reuses, 1U);
    EXPECT_EQ(cached.statistics.schur_product_reuses, 1U);
    EXPECT_EQ(cached.statistics.preconditioner_builds, 1U);
    EXPECT_EQ(cached.statistics.preconditioner_numeric_reuses, 1U);
    EXPECT_EQ(cached.statistics.belos_solver_builds, 1U);
    EXPECT_EQ(cached.statistics.belos_solver_reuses, 1U);
    EXPECT_EQ(cached.statistics.preconditioner_scratch_allocations, 1U);

    EXPECT_EQ(rebuilt.statistics.static_geometry_reuses, 0U);
    EXPECT_EQ(rebuilt.statistics.matrix_graph_reuses, 0U);
    EXPECT_EQ(rebuilt.statistics.schur_product_reuses, 0U);
    EXPECT_EQ(rebuilt.statistics.preconditioner_numeric_reuses, 0U);
    EXPECT_EQ(rebuilt.statistics.belos_solver_reuses, 0U);
    EXPECT_EQ(rebuilt.statistics.preconditioner_builds, 2U);
    EXPECT_EQ(rebuilt.statistics.belos_solver_builds, 2U);
}

/**
 * @brief A dormant Belos owner must not hide an external cached-matrix owner.
 */
TEST(BoussinesqSolverTest,
     CoupledKrylovDoesNotMutateRetainedUnsolvedGraphChange)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundaries);

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    SimpleFluid::CellField<Pack> temperature(
        mesh, 0.5, "temperature");
    SimpleFluid::FaceField<Pack> face_fluxes(
        mesh, 0.0, "face_fluxes");
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;
    linear_options.max_iterations = 200;

    SimpleFluid::BoussinesqMomentumEquation<Pack> momentum(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack> solver(mesh);
    const auto initial = solver.assemble(
        momentum, velocity, pressure, temperature, face_fluxes,
        boundary_cache, boundaries, time_options);
    ASSERT_TRUE(
        solver.solve(
            initial, velocity, pressure, linear_options).converged);

    auto changed_boundaries = boundaries;
    changed_boundaries.pressure["xmin"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    Teuchos::RCP<Pack::matrix_type> retained_changed_matrix;
    {
        const auto changed = solver.assemble(
            momentum, velocity, pressure, temperature, face_fluxes,
            boundary_cache, changed_boundaries, time_options);
        retained_changed_matrix = changed.matrix;
    }
    const auto reuses_before =
        solver.cache_statistics().matrix_graph_reuses;

    const auto next = solver.assemble(
        momentum, velocity, pressure, temperature, face_fluxes,
        boundary_cache, changed_boundaries, time_options);

    EXPECT_NE(
        next.matrix.getRawPtr(),
        retained_changed_matrix.getRawPtr());
    EXPECT_EQ(
        solver.cache_statistics().matrix_graph_reuses,
        reuses_before);
}

/** @brief Verify coupled assembly honors a dynamic-viscosity override field. */
TEST(BoussinesqSolverTest,
     CoupledKrylovUsesDynamicViscosityOverride)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    SimpleFluid::CellField<Pack> temperature(
        mesh, 0.5, "temperature");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, 0.0, "face_fluxes");
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.thermal_expansion = 0.0;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.dynamic_viscosity = 0.1;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);
    SimpleFluid::CellField<Pack> effective_viscosity(
        mesh, 1.0, "effective_viscosity");

    SimpleFluid::BoussinesqMomentumEquation<Pack> momentum_equation(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack> coupled_solver(mesh);
    EXPECT_THROW(
        coupled_solver.assemble(
            momentum_equation,
            velocity,
            pressure,
            temperature,
            face_fluxes,
            cache,
            bcs,
            time_options,
            nullptr,
            1.0,
            false,
            &effective_viscosity),
        std::invalid_argument);
    const auto molecular = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        temperature,
        face_fluxes,
        cache,
        bcs,
        time_options,
        &material,
        1.0,
        false);
    const auto effective = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        temperature,
        face_fluxes,
        cache,
        bcs,
        time_options,
        &material,
        1.0,
        false,
        &effective_viscosity);

    const auto cell_gid =
        mesh->owned_cell_map()->getMinAllGlobalIndex();
    const auto velocity_row_gid = 4 * cell_gid;
    auto velocity_diagonal =
        [velocity_row_gid](const auto& system)
    {
        const auto row_lid =
            system.matrix->getRowMap()->getLocalElement(
                velocity_row_gid);
        typename Pack::matrix_type::local_inds_host_view_type columns;
        typename Pack::matrix_type::values_host_view_type values;
        system.matrix->getLocalRowView(row_lid, columns, values);
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            const auto column_gid =
                system.matrix->getColMap()->getGlobalElement(
                    columns[entry]);
            if (column_gid == velocity_row_gid)
            {
                return values[entry];
            }
        }
        return std::numeric_limits<Pack::scalar_type>::quiet_NaN();
    };

    const auto molecular_diagonal = velocity_diagonal(molecular);
    const auto effective_diagonal = velocity_diagonal(effective);
    ASSERT_TRUE(std::isfinite(molecular_diagonal));
    ASSERT_TRUE(std::isfinite(effective_diagonal));
    EXPECT_GT(effective_diagonal, molecular_diagonal);
}

/** @brief Verify coupled momentum includes isotropic turbulent kinetic-energy stress. */
TEST(BoussinesqSolverTest,
     CoupledKrylovIncludesIsotropicTurbulentKineticEnergyStress)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundaries);

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    SimpleFluid::CellField<Pack> temperature(mesh, 0.5, "temperature");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, 0.0, "face_fluxes");
    const SimpleFluid::VectorCellField<Pack>::vec_type uniform_k_gradient{
        3.0, -6.0, 9.0};
    SimpleFluid::VectorCellField<Pack> k_gradient(
        mesh, uniform_k_gradient, "k_gradient");

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.thermal_expansion = 0.0;
    SimpleFluid::BoussinesqModelOptions model_options;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);
    SimpleFluid::BoussinesqMomentumEquation<Pack> momentum_equation(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack> coupled_solver(mesh);

    const auto base = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        temperature,
        face_fluxes,
        boundary_cache,
        boundaries,
        time_options,
        &material,
        1.0,
        false);
    const auto turbulent = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        temperature,
        face_fluxes,
        boundary_cache,
        boundaries,
        time_options,
        &material,
        1.0,
        false,
        nullptr,
        &k_gradient);

    const auto base_rhs = base.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    const auto turbulent_rhs = turbulent.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        for (size_t component = 0; component < 3; ++component)
        {
            const auto row_lid = base.map->getLocalElement(
                4 * cell_gid + static_cast<Pack::global_ordinal_type>(component));
            const auto expected =
                mesh->cell_volume(cell_lid)
                * Pack::scalar_type{-2.0 / 3.0}
                * uniform_k_gradient.component(component);
            EXPECT_NEAR(
                turbulent_rhs(row_lid, 0) - base_rhs(row_lid, 0),
                expected,
                1.0e-12);
        }
    }
}

/** @brief Compare effective transpose-stress terms in both coupling paths. */
TEST(BoussinesqSolverTest,
     SegregatedAndCoupledMomentumCarryEffectiveTransposeStress)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "shear_velocity");
    SimpleFluid::CellField<Pack> effective_viscosity(
        mesh, "effective_viscosity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        velocity.set_owned_value(cell_lid, {0.0, center.x, 0.0});
        effective_viscosity.set_owned_value(
            cell_lid, center.y < 0.5 ? 1.0 : 3.0);
    }
    velocity.sync_ghosts();
    effective_viscosity.sync_ghosts();

    auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    for (auto& [batch_id, values] : boundary_cache.value)
    {
        const auto& batch = mesh->boundary_face_batch(batch_id);
        for (size_t in_batch = 0; in_batch < values.size(); ++in_batch)
        {
            const auto face_lid = batch.face_lids[in_batch];
            values[in_batch] = {
                0.0, mesh->face_centroid(face_lid).x, 0.0};
        }
    }

    SimpleFluid::CellField<Pack> pressure(mesh, 0.0, "pressure");
    SimpleFluid::CellField<Pack> temperature(mesh, 0.5, "temperature");
    SimpleFluid::FaceField<Pack> zero_fluxes(
        mesh, 0.0, "zero_fluxes");
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.thermal_expansion = 0.0;
    time_options.non_orthogonal_treatment =
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 1.0;
    model_options.density = 1.0;
    model_options.dynamic_viscosity = 2.0;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);

    auto zero_source = [](MeshType::local_ordinal_type)
    {
        return SimpleFluid::VectorCellField<Pack>::vec_type{};
    };
    SimpleFluid::BoussinesqMomentumEquation<Pack> momentum_equation(mesh);
    const auto segregated = momentum_equation.assemble_physical_system(
        velocity,
        zero_fluxes,
        temperature,
        boundary_cache,
        time_options,
        material,
        1.0,
        false,
        zero_source,
        &velocity,
        &effective_viscosity);

    SimpleFluid::CoupledPressureVelocitySolver<Pack> coupled_solver(mesh);
    const auto molecular = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        temperature,
        zero_fluxes,
        boundary_cache,
        bcs,
        time_options,
        &material,
        1.0,
        false);
    const auto effective = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        temperature,
        zero_fluxes,
        boundary_cache,
        bcs,
        time_options,
        &material,
        1.0,
        false,
        &effective_viscosity);

    const auto segregated_x = segregated.rhs->getData(0);
    const auto molecular_rhs = molecular.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    const auto effective_rhs = effective.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    double maximum_transpose_stress = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto coupled_x_row = 4 * owned;
        EXPECT_NEAR(molecular_rhs(coupled_x_row, 0), 0.0, 1.0e-12);
        EXPECT_NEAR(
            effective_rhs(coupled_x_row, 0),
            segregated_x[owned],
            1.0e-12);
        maximum_transpose_stress = std::max(
            maximum_transpose_stress,
            std::abs(effective_rhs(coupled_x_row, 0)));
    }
    EXPECT_GT(maximum_transpose_stress, 1.0e-6);
}

/** @brief Verify coupled pressure boundary data is normalized by reference density. */
TEST(BoussinesqSolverTest,
     CoupledKrylovNormalizesPhysicalPressureBoundaryData)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name :
         {"xmin", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, {}};
    bcs.pressure["xmin"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 2.5};
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 3.0};

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_fluxes");
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FVM::face_fluxes(velocity, cache, face_fluxes);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-2;
    options.kinematic_viscosity = 1.0e-2;
    SimpleFluid::IncompressibleMomentumEquation<Pack>
        momentum_equation(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack>
        coupled_solver(mesh);

    auto assemble = [&](double reference_density)
    {
        auto scaled_bcs = bcs;
        scaled_bcs.pressure["xmin"].value *= reference_density;
        scaled_bcs.pressure["xmax"].value *= reference_density;
        return coupled_solver.assemble(
            momentum_equation,
            velocity,
            pressure,
            face_fluxes,
            cache,
            scaled_bcs,
            options,
            reference_density);
    };

    const auto unit_density = assemble(1.0);
    const auto water = assemble(1000.0);
    const auto unit_rhs = unit_density.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    const auto water_rhs = water.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    ASSERT_EQ(unit_rhs.extent(0), water_rhs.extent(0));

    double maximum_rhs = 0.0;
    for (size_t row = 0; row < unit_rhs.extent(0); ++row)
    {
        maximum_rhs = std::max(maximum_rhs, std::abs(unit_rhs(row, 0)));
        EXPECT_NEAR(
            water_rhs(row, 0),
            unit_rhs(row, 0),
            std::max(1.0e-12, std::abs(unit_rhs(row, 0)) * 1.0e-12));
    }
    EXPECT_GT(maximum_rhs, 0.0);
    EXPECT_DOUBLE_EQ(water.reference_density, 1000.0);
}

/** @brief Verify skew Neumann pressure data uses boundary-normal distance. */
TEST(BoussinesqSolverTest,
     CoupledKrylovUsesNormalDistanceForSkewBoundaryNeumannData)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    constexpr double reference_density = 4.0;
    constexpr double prescribed_pressure_gradient = 20.0;

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.pressure["xmin"] = {
        SimpleFluid::BoundaryConditionType::Neumann,
        prescribed_pressure_gradient};
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_fluxes");
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FVM::face_fluxes(velocity, cache, face_fluxes);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-2;
    options.kinematic_viscosity = 1.0e-2;
    SimpleFluid::IncompressibleMomentumEquation<Pack>
        momentum_equation(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack>
        coupled_solver(mesh);
    const auto system = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        face_fluxes,
        cache,
        bcs,
        options,
        reference_density);
    const auto rhs = system.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);

    SimpleFluid::VectorCellField<Pack> reconstructed_gradient(
        mesh, "reconstructed_gradient");
    SimpleFluid::FVM::cell_gradient(
        pressure, bcs.pressure, reconstructed_gradient);
    const auto coupled_stencils =
        SimpleFluid::detail::pressure_gradient_stencils<Pack>(
            *mesh, bcs.pressure, reference_density);
    ASSERT_EQ(coupled_stencils.size(), mesh->num_owned_cells());

    const auto boundary_locations =
        SimpleFluid::FVM::detail::boundary_face_locations(*mesh);
    double maximum_euclidean_error = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto expected_stencil_constant =
            reconstructed_gradient.value(cell_lid) / reference_density;
        const auto actual_stencil_constant =
            coupled_stencils[owned].constant;
        EXPECT_NEAR(
            actual_stencil_constant.x,
            expected_stencil_constant.x,
            1.0e-12);
        EXPECT_NEAR(
            actual_stencil_constant.y,
            expected_stencil_constant.y,
            1.0e-12);
        EXPECT_NEAR(
            actual_stencil_constant.z,
            expected_stencil_constant.z,
            1.0e-12);

        std::array<double, 3> expected_momentum_rhs{};
        std::array<double, 3> euclidean_momentum_rhs{};
        for (const auto face_lid : mesh->faces(cell_lid))
        {
            if (!mesh->is_boundary_face(face_lid))
            {
                continue;
            }
            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            if (!location.active
                || mesh->boundary_batch_name(location.batch_id)
                   != "xmin")
            {
                continue;
            }

            const auto direction =
                mesh->face_centroid(face_lid)
              - mesh->cell_centroid(cell_lid);
            const auto normal_distance = direction.dot(
                mesh->face_normal_outward(face_lid, cell_lid));
            const auto euclidean_distance =
                mesh->cell_to_face_distance(face_lid, cell_lid);
            const auto area =
                mesh->face_area_vector_outward(face_lid, cell_lid);
            const std::array<double, 3> area_components{
                area.x, area.y, area.z};
            for (size_t component = 0; component < 3; ++component)
            {
                const auto factor =
                    -prescribed_pressure_gradient
                    / reference_density
                    * area_components[component];
                expected_momentum_rhs[component] +=
                    factor * normal_distance;
                euclidean_momentum_rhs[component] +=
                    factor * euclidean_distance;
            }
        }

        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        for (size_t component = 0; component < 3; ++component)
        {
            const auto row_lid = system.map->getLocalElement(
                4 * cell_gid
              + static_cast<Pack::global_ordinal_type>(component));
            ASSERT_NE(
                row_lid,
                Teuchos::OrdinalTraits<Pack::local_ordinal_type>::invalid());
            EXPECT_NEAR(
                rhs(row_lid, 0),
                expected_momentum_rhs[component],
                1.0e-12);
            maximum_euclidean_error = std::max(
                maximum_euclidean_error,
                std::abs(expected_momentum_rhs[component]
                         - euclidean_momentum_rhs[component]));
        }
    }
    EXPECT_GT(maximum_euclidean_error, 1.0e-6);
}

/** @brief Verify Dirichlet pressure preserves continuity and the physical gauge shift. */
TEST(BoussinesqSolverTest,
     CoupledKrylovDirichletPressurePreservesContinuityAndGaugeShift)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name :
         {"xmin", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, {}};

    constexpr double reference_density = 1000.0;
    constexpr double base_pressure = 2000.0;
    constexpr double pressure_shift = 5000.0;
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        base_pressure};

    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_fluxes");
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FVM::face_fluxes(velocity, cache, face_fluxes);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-2;
    options.kinematic_viscosity = 1.0e-2;
    SimpleFluid::IncompressibleMomentumEquation<Pack>
        momentum_equation(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack>
        coupled_solver(mesh);

    auto assemble = [&](double boundary_pressure)
    {
        auto shifted_bcs = bcs;
        shifted_bcs.pressure["xmax"].value = boundary_pressure;
        return coupled_solver.assemble(
            momentum_equation,
            velocity,
            pressure,
            face_fluxes,
            cache,
            shifted_bcs,
            options,
            reference_density);
    };
    const auto base = assemble(base_pressure);
    const auto shifted = assemble(base_pressure + pressure_shift);

    const auto former_gauge_cell_gid =
        mesh->owned_cell_map()->getMinAllGlobalIndex();
    const auto former_gauge_row_gid =
        4 * former_gauge_cell_gid + 3;
    const auto former_gauge_row_lid =
        base.matrix->getRowMap()->getLocalElement(
            former_gauge_row_gid);
    typename Pack::matrix_type::local_inds_host_view_type
        former_gauge_columns;
    typename Pack::matrix_type::values_host_view_type
        former_gauge_values;
    base.matrix->getLocalRowView(
        former_gauge_row_lid,
        former_gauge_columns,
        former_gauge_values);
    bool has_continuity_velocity_entry = false;
    for (size_t entry = 0;
         entry < former_gauge_columns.extent(0);
         ++entry)
    {
        const auto column_gid =
            base.matrix->getColMap()->getGlobalElement(
                former_gauge_columns[entry]);
        has_continuity_velocity_entry =
            has_continuity_velocity_entry
            || (column_gid % 4 != 3
                && std::abs(former_gauge_values[entry]) > 1.0e-14);
    }
    EXPECT_TRUE(has_continuity_velocity_entry);

    const auto former_schur_row_lid =
        base.schur->getRowMap()->getLocalElement(
            former_gauge_cell_gid);
    typename Pack::matrix_type::local_inds_host_view_type
        former_schur_columns;
    typename Pack::matrix_type::values_host_view_type
        former_schur_values;
    base.schur->getLocalRowView(
        former_schur_row_lid,
        former_schur_columns,
        former_schur_values);
    const auto schur_row_is_gauge_identity =
        former_schur_columns.extent(0) == 1
        && base.schur->getColMap()->getGlobalElement(
               former_schur_columns[0]) == former_gauge_cell_gid
        && former_schur_values[0] == 1.0;
    EXPECT_FALSE(schur_row_is_gauge_identity);

    typename Pack::vector_type constant_pressure(
        mesh->owned_cell_map(), true);
    constant_pressure.putScalar(1.0);
    typename Pack::vector_type stabilized_pressure(
        mesh->owned_cell_map(), true);
    base.pressure_stabilization->apply(
        constant_pressure, stabilized_pressure);
    EXPECT_GT(stabilized_pressure.norm2(), 1.0e-12);

    typename Pack::vector_type normalized_pressure_shift(
        base.map, true);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        normalized_pressure_shift.replaceGlobalValue(
            4 * cell_gid + 3,
            pressure_shift / reference_density);
    }
    typename Pack::vector_type matrix_shift_response(base.map, true);
    base.matrix->apply(
        normalized_pressure_shift,
        matrix_shift_response);

    const auto response = matrix_shift_response.getLocalViewHost(
        Tpetra::Access::ReadOnly);
    const auto base_rhs = base.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    const auto shifted_rhs = shifted.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);
    double maximum_rhs_shift = 0.0;
    for (size_t row = 0; row < response.extent(0); ++row)
    {
        const auto rhs_shift = shifted_rhs(row, 0) - base_rhs(row, 0);
        maximum_rhs_shift = std::max(
            maximum_rhs_shift, std::abs(rhs_shift));
        EXPECT_NEAR(
            response(row, 0),
            rhs_shift,
            std::max(1.0e-12, std::abs(rhs_shift) * 1.0e-12));
    }
    EXPECT_GT(maximum_rhs_shift, 0.0);
}

/** @brief Verify coupled continuity accounts for prescribed boundary face flux. */
TEST(BoussinesqSolverTest,
     CoupledKrylovDirichletContinuityMatchesBoundaryFaceFlux)
{
    auto mesh = make_box_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name :
         {"xmin", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, {}};
    bcs.pressure["xmin"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 200.0};
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 3000.0};

    constexpr double reference_density = 1000.0;
    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    SimpleFluid::CellField<Pack> pressure(mesh, "pressure");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        velocity.set_owned_value(
            cell_lid,
            {0.2 + 0.1 * center.x,
             -0.05 * center.y,
             0.03 * center.z});
        pressure.set_owned_value(
            cell_lid,
            500.0 + 300.0 * center.x
                  - 100.0 * center.y
                  + 50.0 * center.z);
    }
    velocity.sync_ghosts();
    pressure.sync_ghosts();

    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-2;
    options.kinematic_viscosity = 1.0e-2;
    SimpleFluid::FaceField<Pack> face_fluxes(mesh, "face_fluxes");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity,
        pressure,
        options.time_step / reference_density,
        cache,
        bcs.pressure,
        face_fluxes);

    SimpleFluid::IncompressibleMomentumEquation<Pack>
        momentum_equation(mesh);
    SimpleFluid::CoupledPressureVelocitySolver<Pack>
        coupled_solver(mesh);
    const auto system = coupled_solver.assemble(
        momentum_equation,
        velocity,
        pressure,
        face_fluxes,
        cache,
        bcs,
        options,
        reference_density);

    typename Pack::vector_type state(system.map, true);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        const auto cell_velocity = velocity.value(cell_lid);
        state.replaceGlobalValue(4 * cell_gid, cell_velocity.x);
        state.replaceGlobalValue(4 * cell_gid + 1, cell_velocity.y);
        state.replaceGlobalValue(4 * cell_gid + 2, cell_velocity.z);
        state.replaceGlobalValue(
            4 * cell_gid + 3,
            pressure.value(cell_lid) / reference_density);
    }
    typename Pack::vector_type applied(system.map, true);
    system.matrix->apply(state, applied);
    const auto applied_view = applied.getLocalViewHost(
        Tpetra::Access::ReadOnly);
    const auto rhs_view = system.rhs->getLocalViewHost(
        Tpetra::Access::ReadOnly);

    double maximum_face_flux_balance = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        const auto pressure_row_lid =
            system.map->getLocalElement(4 * cell_gid + 3);
        const auto coupled_residual =
            applied_view(pressure_row_lid, 0)
          - rhs_view(pressure_row_lid, 0);
        const auto face_flux_balance =
            SimpleFluid::FVM::cell_flux_balance<Pack>(
                *mesh, face_fluxes, cell_lid);
        maximum_face_flux_balance = std::max(
            maximum_face_flux_balance,
            std::abs(face_flux_balance));
        EXPECT_NEAR(
            coupled_residual,
            face_flux_balance,
            std::max(
                1.0e-12,
                std::abs(face_flux_balance) * 1.0e-11));
    }
    EXPECT_GT(maximum_face_flux_balance, 1.0e-12);
}

/** @brief Verify segregated coupling rejects invalid corrector counts. */
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

/** @brief Compare one-cell physical heat-source integration to its analytic rise. */
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

/** @brief Verify fission power adds to other registered heat sources. */
TEST(BoussinesqSolverTest, FissionPowerAddsToOtherPhysicalHeatSources)
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

    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.temperature().put_scalar(10.0);
    solver.add_temperature_source("heat_sink", -5.0);
    auto& fission = solver.add_fission_power_source();
    fission.initialize_constant(20.0);
    solver.step();

    EXPECT_NEAR(solver.temperature().value(0), 10.15, 1.0e-12);
    EXPECT_DOUBLE_EQ(fission.field().value(0), 20.0);
    EXPECT_NE(
        solver.find_temperature_source("qdot_fission"),
        nullptr);
}

/** @brief Verify fission source callbacks observe the step-start context. */
TEST(BoussinesqSolverTest, FissionMultiplierUsesStepStartContext)
{
    auto mesh = make_single_cell_box_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.thermal_expansion = 0.0;
    time_options.kinematic_viscosity = 0.0;
    time_options.thermal_diffusivity = 0.0;

    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.temperature().put_scalar(0.0);
    auto& fission = solver.add_fission_power_source();
    fission.initialize_constant(10.0);

    std::vector<double> update_times;
    fission.set_time_multiplier(
        [&](const auto& context)
        {
            update_times.push_back(context.time);
            return 1.0 + context.time;
        });

    solver.step();
    EXPECT_NEAR(solver.temperature().value(0), 1.0, 1.0e-12);
    EXPECT_DOUBLE_EQ(fission.field().value(0), 10.0);
    solver.step();
    EXPECT_NEAR(solver.temperature().value(0), 2.1, 1.0e-12);
    EXPECT_DOUBLE_EQ(fission.field().value(0), 11.0);
    EXPECT_EQ(update_times, (std::vector<double>{0.0, 0.1}));
}

/** @brief Exercise creation, lookup, configuration, and removal of fission sources. */
TEST(BoussinesqSolverTest, ManagesSpecializedFissionSourceLifecycle)
{
    auto mesh = make_single_cell_box_mesh();
    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, {});

    EXPECT_EQ(solver.find_fission_power_source(), nullptr);
    EXPECT_THROW(
        solver.add_temperature_source("qdot_fission", 1.0),
        std::invalid_argument);
    EXPECT_THROW(
        solver.remove_temperature_source("qdot_fission"),
        std::invalid_argument);

    SimpleFluid::FissionPowerSourceOptions options;
    options.profile =
        SimpleFluid::FissionPowerProfile::Constant;
    options.power_density = 4.0;
    solver.configure_fission_power_source(options);
    ASSERT_NE(solver.find_fission_power_source(), nullptr);
    EXPECT_DOUBLE_EQ(
        solver.find_fission_power_source()->field().value(0),
        4.0);
    EXPECT_THROW(
        solver.add_fission_power_source(),
        std::invalid_argument);

    options.profile =
        SimpleFluid::FissionPowerProfile::Disabled;
    solver.configure_fission_power_source(options);
    EXPECT_EQ(solver.find_fission_power_source(), nullptr);
    EXPECT_EQ(
        solver.find_temperature_source("qdot_fission"),
        nullptr);
    EXPECT_FALSE(solver.remove_fission_power_source());
}

/** @brief Verify material properties refresh before step-start source evaluation. */
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

/** @brief Verify density feedback retains reference-density buoyancy scaling. */
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

/** @brief Compare explicit physical defaults against the legacy transport path. */
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

/** @brief Verify auxiliary multiphysics fields are opt-in for VTU output. */
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
    auto& fission = solver.add_fission_power_source();
    fission.initialize_constant(8.0);

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
        default_contents.find("Name=\"qdot_fission\""),
        std::string::npos);
    EXPECT_EQ(
        default_contents.find("Name=\"density\""),
        std::string::npos);
    EXPECT_NE(
        auxiliary_contents.find("Name=\"qdot_test\""),
        std::string::npos);
    EXPECT_NE(
        auxiliary_contents.find("Name=\"qdot_fission\""),
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

/** @brief Verify source updates reject non-finite power density values. */
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

/** @brief Verify coupled Krylov assembly uses configured physical material fields. */
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
