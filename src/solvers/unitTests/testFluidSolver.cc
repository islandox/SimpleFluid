/**
 * @file testFluidSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for the reusable incompressible fluid solver.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/MeshFactory.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/FluidSolver.hh"
#include "utils/testing_environment.hh"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build a single-cell box mesh. @return Assembled mesh. */
SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    database->set("X", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return SimpleFluid::MeshFactory(database).template build<Pack>();
}

/** @brief Build a two-cell line mesh along x. @return Assembled mesh. */
SimpleFluid::SP<MeshType> make_two_cell_line_mesh()
{
    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{0.5});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    database->set("X", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    database->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return SimpleFluid::MeshFactory(database).template build<Pack>();
}

/** @brief Expose protected FluidSolver hooks for focused unit tests. */
class TestFluidSolver : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

    SimpleFluid::LinearSolveSummary advance_momentum_once()
    {
        return advance_momentum();
    }

    bool pressure_preconditioner_reuse_enabled()
    {
        return pressure_projection()
            .linear_solver_options().reuse_preconditioner;
    }

    auto runtime_mesh_handle() const
    {
        return d_problem.mesh_ptr();
    }

    auto pressure_mesh_handle() const
    {
        return pressure().mesh_ptr();
    }

    auto velocity_mesh_handle() const
    {
        return velocity().mesh_ptr();
    }

    auto wrapped_legacy_mesh() const
    {
        return d_legacy_mesh;
    }

    bool has_legacy_backend() const
    {
        return uses_legacy_backend();
    }

    auto* historical_pressure_flux_workspace()
    {
        return &pressure_face_flux_workspace();
    }

    auto* explicit_legacy_pressure_flux_workspace()
    {
        return &legacy_pressure_face_flux_workspace();
    }
};

/** @brief Override pressure normalization to represent water-density scaling. */
class WaterPressureFluidSolver : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

protected:
    Pack::scalar_type pressure_reference_density() const noexcept override
    {
        return 1000.0;
    }
};

/** @brief Exercise the public FieldStored contract of the virtual hook. */
class PublicMomentumHookFluidSolver
    : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

protected:
    SimpleFluid::LinearSolveSummary advance_momentum() override
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            velocity().set_value(
                static_cast<Pack::local_ordinal_type>(owned),
                {2.0, 0.0, 0.0});
        }
        return {};
    }
};

/** @brief Verify native mesh dispatch remains an overridable hook. */
class NativeMomentumEquationHookFluidSolver
    : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

    bool native_hook_called() const noexcept
    {
        return d_native_hook_called;
    }

protected:
    momentum_equation_type& native_momentum_equation() override
    {
        d_native_hook_called = true;
        return SimpleFluid::FluidSolver<Pack>::native_momentum_equation();
    }

private:
    bool d_native_hook_called = false;
};

} // namespace

static_assert(std::is_base_of_v<
              SimpleFluid::FluidSolver<Pack>,
              SimpleFluid::BoussinesqSolver<Pack>>);

/** @brief A supplied runtime handle is shared rather than reconstructed. */
TEST(FluidSolverTest, ReusesProvidedRuntimeMeshHandle)
{
    auto legacy = make_single_cell_mesh();
    auto handle =
        std::make_shared<SimpleFluid::MeshHandle<Pack>>(legacy);
    TestFluidSolver solver(handle, {});

    EXPECT_EQ(solver.runtime_mesh_handle(), handle);
    EXPECT_EQ(solver.pressure_mesh_handle(), handle);
    EXPECT_EQ(solver.velocity_mesh_handle(), handle);
    EXPECT_EQ(solver.wrapped_legacy_mesh(), legacy);
    EXPECT_TRUE(solver.has_legacy_backend());
}

/** @brief The legacy constructor wraps, but never rebuilds, its input mesh. */
TEST(FluidSolverTest, ReusesExactLegacyMeshWithoutConversion)
{
    auto legacy = make_single_cell_mesh();
    TestFluidSolver solver(legacy, {});

    ASSERT_TRUE(solver.runtime_mesh_handle());
    EXPECT_EQ(solver.runtime_mesh_handle()->legacy_mesh(), legacy);
    EXPECT_EQ(solver.wrapped_legacy_mesh(), legacy);
    EXPECT_EQ(solver.pressure_mesh_handle(), solver.runtime_mesh_handle());
    EXPECT_EQ(solver.velocity_mesh_handle(), solver.runtime_mesh_handle());
    EXPECT_EQ(solver.historical_pressure_flux_workspace(),
              solver.explicit_legacy_pressure_flux_workspace());
}

/** @brief Reordered legacy handles cannot use ordinal-based field mirrors. */
TEST(FluidSolverTest, RejectsReorderedLegacyCompatibilityHandle)
{
    auto handle = std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        make_two_cell_line_mesh());
    const auto layout =
        SimpleFluid::MeshReorderingFactory<Pack>::selected_cells_first(
            std::move(handle),
            [](Pack::global_ordinal_type,
               const SimpleFluid::MeshHandle<Pack>::Vec3& center)
            {
                return center.x > 0.5;
            });

    ASSERT_TRUE(layout.mesh->has_reordered_cells());
    EXPECT_THROW(
        static_cast<void>(TestFluidSolver(layout.mesh, {})),
        std::invalid_argument);
}

/** @brief Legacy mirrors do not overwrite a derived public momentum update. */
TEST(FluidSolverTest, PreservesPublicMomentumHookUpdatesOnLegacyMesh)
{
    PublicMomentumHookFluidSolver solver(make_single_cell_mesh(), {});
    solver.step();

    const auto updated = solver.velocity().value(0);
    EXPECT_NEAR(updated.x, 2.0, 1.0e-12);
    EXPECT_NEAR(updated.y, 0.0, 1.0e-12);
    EXPECT_NEAR(updated.z, 0.0, 1.0e-12);
}

/** @brief Native mesh handles run without constructing a legacy mesh. */
TEST(FluidSolverTest, RunsNativeMeshHandlesWithoutLegacyConversion)
{
    using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
    using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
    using SemiStructured = SimpleFluid::Meshes::SemiStructuredXY_Z;

    std::vector<SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>>> meshes;
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        std::make_shared<Cartesian>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                {0.0, 0.5, 1.0},
                {0.0, 1.0},
                {0.0, 1.0}}})));
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        std::make_shared<Cylindrical>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                {1.0, 2.0},
                {0.0, std::numbers::pi, 2.0 * std::numbers::pi},
                {0.0, 1.0}}})));
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        std::make_shared<SemiStructured>(
            SimpleFluid::Arr<SemiStructured::Vec3>{
                {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0},
                {1.0, 1.0, 0.0},
                {0.0, 1.0, 0.0}},
            SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{
                {0, 1, 3}, {1, 2, 3}},
            SimpleFluid::ArrReal{0.0, 1.0})));
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        SimpleFluid::test::make_unstructured_hex_line(2)));

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 1;
    time_options.kinematic_viscosity = 0.0;

    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index)
    {
        SCOPED_TRACE(mesh_index);
        const auto& mesh = meshes[mesh_index];
        ASSERT_FALSE(mesh->legacy_mesh());
        TestFluidSolver solver(mesh, {}, time_options);
        EXPECT_EQ(solver.runtime_mesh_handle(), mesh);
        EXPECT_EQ(solver.pressure_mesh_handle(), mesh);
        EXPECT_EQ(solver.velocity_mesh_handle(), mesh);
        EXPECT_FALSE(solver.wrapped_legacy_mesh());
        EXPECT_FALSE(solver.has_legacy_backend());

        solver.run();
        EXPECT_EQ(solver.step_index(), 1);
        EXPECT_TRUE(std::isfinite(solver.pressure().value(0)));
        const auto velocity = solver.velocity().value(0);
        EXPECT_TRUE(std::isfinite(velocity.x));
        EXPECT_TRUE(std::isfinite(velocity.y));
        EXPECT_TRUE(std::isfinite(velocity.z));

        const auto output_file = std::filesystem::temp_directory_path() /
                                 ("SimpleFluid_native_solver_" +
                                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                                     std::to_string(mesh_index) + ".vtu");
        solver.write_solution_vtu(output_file.string());
        EXPECT_TRUE(std::filesystem::exists(output_file));
        std::filesystem::remove(output_file);
    }
}

/** @brief Native handles execute the monolithic solver without conversion. */
TEST(FluidSolverTest, RunsCoupledKrylovOnNativeMeshHandle)
{
    using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
    using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
    using SemiStructured = SimpleFluid::Meshes::SemiStructuredXY_Z;

    std::vector<SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>>> meshes;
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 0.5, 1.0}, {0.0, 1.0}, {0.0, 1.0}}})));
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        std::make_shared<Cylindrical>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{
            {{1.0, 2.0}, {0.0, std::numbers::pi, 2.0 * std::numbers::pi}, {0.0, 1.0}}})));
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 3}, {1, 2, 3}}, SimpleFluid::ArrReal{0.0, 1.0})));
    meshes.push_back(std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        SimpleFluid::test::make_unstructured_hex_line(2)));

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index)
    {
        SCOPED_TRACE(mesh_index);
        const auto& mesh = meshes[mesh_index];
        ASSERT_FALSE(mesh->legacy_mesh());
        TestFluidSolver solver(mesh, {}, time_options);

        EXPECT_NO_THROW(solver.step());
        EXPECT_EQ(solver.step_index(), 1);
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
            EXPECT_TRUE(std::isfinite(solver.pressure().value(cell_lid)));
            const auto velocity = solver.velocity().value(cell_lid);
            EXPECT_TRUE(std::isfinite(velocity.x));
            EXPECT_TRUE(std::isfinite(velocity.y));
            EXPECT_TRUE(std::isfinite(velocity.z));
        }
    }
}

/** @brief Native handles dispatch through their dedicated equation hook. */
TEST(FluidSolverTest, DispatchesNativeMomentumEquationHook)
{
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<SimpleFluid::MeshHandle<Pack>>(cartesian);
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    NativeMomentumEquationHookFluidSolver solver(mesh, {}, time_options);

    solver.step();
    EXPECT_TRUE(solver.native_hook_called());
}

/**
 * @brief Verify transport and pressure policies are independently selectable.
 */
TEST(FluidSolverTest, ExposesIndependentLinearSolverPolicies)
{
    auto mesh = make_single_cell_mesh();

    SimpleFluid::LinearSolverOptions transport_options;
    transport_options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    transport_options.preconditioner = SimpleFluid::LinearPreconditioner::ILU0;
    TestFluidSolver solver(mesh, {}, {}, transport_options);

    EXPECT_EQ(
        solver.linear_solver_options().backend,
        SimpleFluid::LinearSolverBackend::BiCGStab);
    EXPECT_EQ(
        solver.linear_solver_options().preconditioner,
        SimpleFluid::LinearPreconditioner::ILU0);
    EXPECT_EQ(
        solver.pressure_linear_solver_options().backend,
        SimpleFluid::LinearSolverBackend::BiCGStab);
    EXPECT_EQ(
        solver.pressure_linear_solver_options().preconditioner,
        SimpleFluid::LinearPreconditioner::MueLu);
    EXPECT_TRUE(
        solver.pressure_linear_solver_options().reuse_preconditioner);

    auto replacement_transport = solver.linear_solver_options();
    replacement_transport.backend =
        SimpleFluid::LinearSolverBackend::Gmres;
    replacement_transport.preconditioner =
        SimpleFluid::LinearPreconditioner::Jacobi;
    solver.set_linear_solver_options(replacement_transport);

    EXPECT_EQ(
        solver.linear_solver_options().backend,
        SimpleFluid::LinearSolverBackend::Gmres);
    EXPECT_EQ(
        solver.linear_solver_options().preconditioner,
        SimpleFluid::LinearPreconditioner::Jacobi);
    EXPECT_EQ(
        solver.pressure_linear_solver_options().preconditioner,
        SimpleFluid::LinearPreconditioner::MueLu);

    auto replacement_pressure =
        solver.pressure_linear_solver_options();
    replacement_pressure.preconditioner =
        SimpleFluid::LinearPreconditioner::Jacobi;
    replacement_pressure.reuse_preconditioner = false;
    solver.set_pressure_linear_solver_options(replacement_pressure);

    EXPECT_EQ(
        solver.pressure_linear_solver_options().backend,
        SimpleFluid::LinearSolverBackend::BiCGStab);
    EXPECT_EQ(
        solver.pressure_linear_solver_options().preconditioner,
        SimpleFluid::LinearPreconditioner::Jacobi);
    EXPECT_FALSE(
        solver.pressure_linear_solver_options().reuse_preconditioner);
}

/**
 * @brief Verify that FluidSolver advances pressure and velocity over two
 *        time steps on a single-cell mesh with zero viscosity.
 */
TEST(FluidSolverTest, AdvancesPressureVelocityWithoutThermalFields)
{
    auto mesh = make_single_cell_mesh();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.steps = 2;
    time_options.kinematic_viscosity = 0.0;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;

    TestFluidSolver solver(
        mesh, {}, time_options, linear_options);
    EXPECT_TRUE(solver.pressure_preconditioner_reuse_enabled());
    solver.run();

    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_NEAR(solver.time(), 0.2, 1.0e-14);
    EXPECT_NEAR(solver.pressure().value(0), 0.0, 1.0e-12);
    const auto velocity = solver.velocity().value(0);
    EXPECT_NEAR(velocity.x, 0.0, 1.0e-12);
    EXPECT_NEAR(velocity.y, 0.0, 1.0e-12);
    EXPECT_NEAR(velocity.z, 0.0, 1.0e-12);

    const auto residuals = solver.last_pressure_velocity_residuals();
    const auto statistics = solver.last_step_statistics();
    EXPECT_TRUE(statistics.converged);
    EXPECT_NEAR(statistics.momentum, residuals.momentum, 1.0e-12);
    EXPECT_NEAR(statistics.pressure, residuals.pressure, 1.0e-12);
    EXPECT_NEAR(statistics.continuity, residuals.continuity, 1.0e-12);
}

/**
 * @brief Confirm that the momentum predictor uses the previous pressure
 *        gradient, producing a linear velocity ramp on a two-cell mesh.
 */
TEST(FluidSolverTest, MomentumPredictorIncludesOldPressureGradient)
{
    auto mesh = make_two_cell_line_mesh();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.kinematic_viscosity = 0.0;

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.pressure["xmin"] = {
        SimpleFluid::BoundaryConditionType::Neumann, -1.0};
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 1.0};
    TestFluidSolver solver(mesh, bcs, time_options);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        solver.pressure().set_value(
            cell_lid, mesh->cell_centroid(cell_lid).x);
    }
    solver.pressure().sync_ghosts();

    const auto summary = solver.advance_momentum_once();

    EXPECT_TRUE(summary.converged);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto velocity = solver.velocity().value(cell_lid);
        EXPECT_NEAR(velocity.x, -time_options.time_step, 1.0e-12);
        EXPECT_NEAR(velocity.y, 0.0, 1.0e-12);
        EXPECT_NEAR(velocity.z, 0.0, 1.0e-12);
    }
}

/** @brief Verify segregated coupling modes enforce physical outlet pressure. */
TEST(FluidSolverTest,
     SegregatedModesHonorPhysicalDirichletPressureBoundary)
{
    for (const auto coupling : {
             SimpleFluid::PressureVelocityCoupling::SIMPLE,
             SimpleFluid::PressureVelocityCoupling::PISO,
             SimpleFluid::PressureVelocityCoupling::PIMPLE})
    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoundaryConditionSet bcs;
        bcs.pressure["xmax"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 1000.0};
        bcs.velocity["xmax"] = {
            SimpleFluid::BoundaryConditionType::Neumann, {}};
        for (const auto* name :
             {"xmin", "ymin", "ymax", "zmin", "zmax"})
        {
            bcs.velocity[name] = {
                SimpleFluid::BoundaryConditionType::NoSlip, {}};
        }

        SimpleFluid::TimeStepperOptions time_options;
        time_options.time_step = 0.1;
        time_options.kinematic_viscosity = 0.0;
        time_options.pressure_velocity_coupling = coupling;
        time_options.n_pressure_correctors = 2;
        time_options.n_outer_correctors = 1;

        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-12;
        WaterPressureFluidSolver solver(
            mesh, bcs, time_options, linear_options);
        solver.step();

        EXPECT_NEAR(
            solver.pressure().value(0),
            1000.0,
            1.0e-8);
        const auto velocity = solver.velocity().value(0);
        EXPECT_NEAR(
            velocity.x, 0.0, 1.0e-10);
        EXPECT_NEAR(velocity.y, 0.0, 1.0e-12);
        EXPECT_NEAR(velocity.z, 0.0, 1.0e-12);
        const auto cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
                solver.velocity().mesh_ptr(), bcs);
        SimpleFluid::ScalarFaceFieldStored<Pack> final_fluxes(
            solver.velocity().mesh_ptr(),
            "final_pressure_weighted_fluxes");
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            solver.velocity(),
            solver.pressure(),
            time_options.time_step / 1000.0,
            cache,
            bcs.pressure,
            final_fluxes);
        const auto final_imbalance = std::abs(
            SimpleFluid::FVM::cell_flux_balance<Pack>(
                solver.velocity().mesh(), final_fluxes, 0));
        EXPECT_NEAR(
            solver.last_pressure_velocity_residuals().continuity,
            final_imbalance,
            1.0e-12);
        EXPECT_TRUE(solver.last_step_statistics().converged);
    }
}

/** @brief Verify coupled Krylov enforces a Dirichlet pressure outlet. */
TEST(FluidSolverTest,
     CoupledKrylovHonorsDirichletPressureOutlet)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, {}};
    for (const auto* name :
         {"xmin", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.kinematic_viscosity = 0.0;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;
    linear_options.max_iterations = 200;
    SimpleFluid::FluidSolver<Pack> solver(
        mesh, bcs, time_options, linear_options);
    solver.step();

    EXPECT_TRUE(solver.last_step_statistics().converged);
    EXPECT_NEAR(solver.pressure().value(0), 1.0, 1.0e-10);
    const auto velocity = solver.velocity().value(0);
    EXPECT_NEAR(velocity.x, 0.0, 1.0e-10);
    EXPECT_NEAR(velocity.y, 0.0, 1.0e-10);
    EXPECT_NEAR(velocity.z, 0.0, 1.0e-10);
    EXPECT_NEAR(
        solver.last_pressure_velocity_residuals().continuity,
        0.0,
        1.0e-10);
}

/**
 * @brief Ensure VTU output contains pressure and velocity but excludes
 *        temperature when no thermal model is active.
 */
TEST(FluidSolverTest, WritesOnlyCoreFluidFields)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::FluidSolver<Pack> solver(mesh, {});

    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_file =
        std::filesystem::temp_directory_path()
        / ("SimpleFluid_testFluidSolver_"
           + std::to_string(unique_id) + ".vtu");
    solver.write_solution_vtu(output_file.string());

    std::ifstream input(output_file);
    ASSERT_TRUE(input.good());
    const std::string contents(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    EXPECT_NE(contents.find("Name=\"pressure\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"velocity\""), std::string::npos);
    EXPECT_EQ(contents.find("Name=\"temperature\""), std::string::npos);

    std::filesystem::remove(output_file);
}

/**
 * @brief Validate a single coupled-Krylov pressure-velocity step on a
 *        single-cell mesh with zero viscosity.
 */
TEST(FluidSolverTest, SupportsCoupledKrylovPressureVelocitySolve)
{
    auto mesh = make_single_cell_mesh();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.kinematic_viscosity = 0.0;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    SimpleFluid::FluidSolver<Pack> solver(mesh, {}, time_options);
    solver.step();

    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_TRUE(solver.last_step_statistics().converged);
    EXPECT_EQ(solver.last_step_statistics().nonlinear_iterations, 1);
    EXPECT_TRUE(std::isfinite(
        solver.last_pressure_velocity_residuals().continuity));
}
