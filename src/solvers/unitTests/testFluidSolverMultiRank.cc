/**
 * @file testFluidSolverMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI tests for globally reduced pressure-velocity residual norms.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "examples/ExampleRunner.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/FluidSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;
using FaceFieldType = SimpleFluid::FaceField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Expose the protected global velocity-update norm for MPI tests. */
class ExposedFluidSolver : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

    Pack::scalar_type update_norm(
        const VelocityFieldType& before,
        const VelocityFieldType& after) const
    {
        return velocity_update_norm(before, after);
    }

    SimpleFluid::VTUWriter solution_writer() const
    {
        return fluid_solution_writer();
    }
};

/**
 * @brief Sum a scalar value over the mesh communicator.
 *
 * @param mesh Mesh providing the communicator.
 * @param local_value Local contribution.
 * @return Global sum across ranks.
 */
double global_sum(const MeshType& mesh, double local_value)
{
    double global_value = 0.0;
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_SUM,
        1,
        &local_value,
        &global_value);
    return global_value;
}

/** @brief Build the distributed cavity mesh. @return Assembled mesh. */
SimpleFluid::SP<MeshType> distributed_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 8, 1, 0.125));
}

/** @brief Build the distributed line mesh. @return Assembled mesh. */
SimpleFluid::SP<MeshType> distributed_line_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
}

/**
 * @brief Construct moving-lid cavity velocity boundary conditions.
 *
 * @return Boundary conditions for the distributed cavity fixture.
 */
SimpleFluid::BoundaryConditionSet cavity_boundary_conditions()
{
    SimpleFluid::BoundaryConditionSet conditions;
    for (const auto* name : {"xmin", "xmax", "ymin"})
    {
        conditions.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    conditions.velocity["ymax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        {1.0, 0.0, 0.0}};
    conditions.velocity["zmin"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    conditions.velocity["zmax"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    return conditions;
}

} // namespace

/** @brief Verify distributed examples never select the same VTU filename. */
TEST(FluidSolverOutputTest, DistributedVtuFilenamesAreRankSpecific)
{
    EXPECT_EQ(
        SimpleFluid::detail::rank_local_vtu_filename(
            "solution.vtu", 0, 1),
        "solution.vtu");
    EXPECT_EQ(
        SimpleFluid::detail::rank_local_vtu_filename(
            "solution.vtu", 0, 2),
        "solution_rank0.vtu");
    EXPECT_EQ(
        SimpleFluid::detail::rank_local_vtu_filename(
            "solution.vtu", 1, 2),
        "solution_rank1.vtu");
}

/** @brief Verify the velocity update norm is reduced over all ranks. */
TEST(FluidSolverMultiRankTest, VelocityUpdateNormIsGlobal)
{
    auto mesh = distributed_mesh();
    if (mesh->owned_cell_map()->getComm()->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    ExposedFluidSolver solver(mesh, {});
    VelocityFieldType before(
        mesh, MeshType::Vec3{}, "velocity_before");
    VelocityFieldType after(
        mesh, MeshType::Vec3{}, "velocity_after");

    double local_squared_norm = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        const auto scale = static_cast<double>(cell_gid + 1);
        const MeshType::Vec3 value{
            scale, -0.5 * scale, 2.0 * scale};
        after.set_value(cell_lid, value);
        local_squared_norm +=
            value.dot(value) * mesh->cell_volume(cell_lid);
    }

    const auto expected =
        std::sqrt(global_sum(*mesh, local_squared_norm));
    ASSERT_GT(expected, 0.0);
    EXPECT_NEAR(
        solver.update_norm(before, after),
        expected,
        std::max(1.0e-14, expected * 1.0e-12));
}

/** @brief Verify rank-local VTU pieces omit overlap ghost cells. */
TEST(FluidSolverMultiRankTest, SolutionOutputContainsOwnedCellsOnly)
{
    auto mesh = distributed_line_mesh();
    if (mesh->owned_cell_map()->getComm()->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    ASSERT_GT(mesh->num_local_cells(), mesh->num_owned_cells());
    ExposedFluidSolver solver(mesh, {});
    const auto first_writer = solver.solution_writer();
    const auto second_writer = solver.solution_writer();
    EXPECT_EQ(first_writer.num_cells(), mesh->num_owned_cells());
    ASSERT_TRUE(first_writer.topology_handle());
    ASSERT_TRUE(second_writer.topology_handle());
    EXPECT_EQ(
        first_writer.topology_handle().get(),
        second_writer.topology_handle().get());
}

/** @brief Verify distributed output publishes binary pieces and one PVTU index. */
TEST(FluidSolverMultiRankTest, ParallelSolutionOutputPublishesIndex)
{
    auto mesh = distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, {});
    const auto base_filename =
        (std::filesystem::temp_directory_path()
         / ("SimpleFluid_parallel_solution_"
            + std::to_string(rank_count) + ".vtu"))
            .string();
    const auto piece_filename =
        SimpleFluid::VTUWriter::rank_piece_filename(
            base_filename, rank, rank_count);
    const auto index_filename =
        SimpleFluid::VTUWriter::parallel_index_filename(
            base_filename);
    std::error_code cleanup_error;
    std::filesystem::remove(piece_filename, cleanup_error);
    EXPECT_FALSE(cleanup_error);
    if (rank == 0)
    {
        cleanup_error.clear();
        std::filesystem::remove(index_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();

    solver.write_parallel_solution_vtu(base_filename);
    EXPECT_TRUE(std::filesystem::exists(piece_filename));

    if (rank == 0)
    {
        std::ifstream input(index_filename);
        EXPECT_TRUE(input.good());
        if (input.good())
        {
            const std::string contents(
                (std::istreambuf_iterator<char>(input)),
                std::istreambuf_iterator<char>());
            EXPECT_NE(contents.find("PUnstructuredGrid"),
                      std::string::npos);
            EXPECT_NE(contents.find("Name=\"temperature\""),
                      std::string::npos);
            for (int piece_rank = 0;
                 piece_rank < rank_count;
                 ++piece_rank)
            {
                const auto source = std::filesystem::path(
                    SimpleFluid::VTUWriter::rank_piece_filename(
                        base_filename, piece_rank, rank_count))
                                        .filename()
                                        .string();
                EXPECT_NE(contents.find(source), std::string::npos);
            }
        }
        input.close();
        EXPECT_FALSE(input.fail());
    }

    communicator->barrier();
    cleanup_error.clear();
    std::filesystem::remove(piece_filename, cleanup_error);
    EXPECT_FALSE(cleanup_error);
    if (rank == 0)
    {
        cleanup_error.clear();
        std::filesystem::remove(index_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();
}

/** @brief Reject rank-varying CellData before pieces or an index are written. */
TEST(FluidSolverMultiRankTest, ParallelOutputRejectsRankVaryingCellDataSchema)
{
    auto mesh = distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    const auto base_filename =
        (std::filesystem::temp_directory_path()
         / ("SimpleFluid_parallel_schema_failure_"
            + std::to_string(rank_count) + ".vtu"))
            .string();
    const auto piece_filename =
        SimpleFluid::VTUWriter::rank_piece_filename(
            base_filename, rank, rank_count);
    const auto index_filename =
        SimpleFluid::VTUWriter::parallel_index_filename(
            base_filename);
    std::error_code cleanup_error;
    std::filesystem::remove(piece_filename, cleanup_error);
    EXPECT_FALSE(cleanup_error);
    if (rank == 0)
    {
        cleanup_error.clear();
        std::filesystem::remove(index_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, {});
    // Deliberately violate the registry's cross-rank mutation contract to
    // verify output catches the mismatch independently.
    solver.add_temperature_source(
        "qdot_rank_" + std::to_string(rank), 1.0);
    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_sources = true;
    EXPECT_THROW(
        solver.write_parallel_solution_vtu(
            base_filename, output_options),
        std::invalid_argument);

    std::error_code exists_error;
    EXPECT_FALSE(std::filesystem::exists(piece_filename, exists_error));
    EXPECT_FALSE(exists_error);
    if (rank == 0)
    {
        exists_error.clear();
        EXPECT_FALSE(std::filesystem::exists(index_filename, exists_error));
        EXPECT_FALSE(exists_error);
    }
    communicator->barrier();

    cleanup_error.clear();
    std::filesystem::remove(piece_filename, cleanup_error);
    EXPECT_FALSE(cleanup_error);
    if (rank == 0)
    {
        cleanup_error.clear();
        std::filesystem::remove(index_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();
}

/** @brief Reject rank-varying filenames and field-selection options. */
TEST(FluidSolverMultiRankTest, ParallelOutputRejectsRankVaryingArguments)
{
    auto mesh = distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, {});
    const auto base = std::filesystem::temp_directory_path()
        / ("SimpleFluid_parallel_argument_failure_"
           + std::to_string(rank_count));
    const auto local_filename =
        base.string() + "_rank" + std::to_string(rank) + ".vtu";
    EXPECT_THROW(
        solver.write_parallel_solution_vtu(local_filename),
        std::invalid_argument);

    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_sources = rank == 0;
    EXPECT_THROW(
        solver.write_parallel_solution_vtu(
            base.string() + ".vtu", output_options),
        std::invalid_argument);

    std::error_code cleanup_error;
    std::filesystem::remove(
        SimpleFluid::VTUWriter::rank_piece_filename(
            local_filename, rank, rank_count),
        cleanup_error);
    EXPECT_FALSE(cleanup_error);
    cleanup_error.clear();
    const auto common_filename = base.string() + ".vtu";
    std::filesystem::remove(
        SimpleFluid::VTUWriter::rank_piece_filename(
            common_filename, rank, rank_count),
        cleanup_error);
    EXPECT_FALSE(cleanup_error);
    if (rank == 0)
    {
        cleanup_error.clear();
        std::filesystem::remove(
            SimpleFluid::VTUWriter::parallel_index_filename(
                local_filename),
            cleanup_error);
        EXPECT_FALSE(cleanup_error);
        cleanup_error.clear();
        std::filesystem::remove(
            SimpleFluid::VTUWriter::parallel_index_filename(
                common_filename),
            cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();
}

/** @brief Verify one rank's piece error is propagated before index output. */
TEST(FluidSolverMultiRankTest, ParallelSolutionOutputFailureIsRankCoherent)
{
    auto mesh = distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    const auto base_filename =
        (std::filesystem::temp_directory_path()
         / ("SimpleFluid_parallel_solution_failure_"
            + std::to_string(rank_count) + ".vtu"))
            .string();
    const auto piece_filename =
        SimpleFluid::VTUWriter::rank_piece_filename(
            base_filename, rank, rank_count);
    const auto index_filename =
        SimpleFluid::VTUWriter::parallel_index_filename(
            base_filename);

    std::error_code cleanup_error;
    if (rank == 0)
    {
        std::filesystem::remove_all(piece_filename, cleanup_error);
        cleanup_error.clear();
        std::filesystem::remove(index_filename, cleanup_error);
        cleanup_error.clear();
        const auto directory_created =
            std::filesystem::create_directory(
                piece_filename, cleanup_error);
        EXPECT_TRUE(directory_created);
        EXPECT_FALSE(cleanup_error);
    }
    else
    {
        std::filesystem::remove(piece_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, {});
    EXPECT_THROW(
        solver.write_parallel_solution_vtu(base_filename),
        std::exception);

    cleanup_error.clear();
    if (rank == 0)
    {
        std::filesystem::remove_all(piece_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
        cleanup_error.clear();
        std::filesystem::remove(index_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    else
    {
        std::filesystem::remove(piece_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();
}

/** @brief Verify a rank-zero index error is propagated to every rank. */
TEST(FluidSolverMultiRankTest, ParallelIndexOutputFailureIsRankCoherent)
{
    auto mesh = distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    const auto base_filename =
        (std::filesystem::temp_directory_path()
         / ("SimpleFluid_parallel_index_failure_"
            + std::to_string(rank_count) + ".vtu"))
            .string();
    const auto piece_filename =
        SimpleFluid::VTUWriter::rank_piece_filename(
            base_filename, rank, rank_count);
    const auto index_filename =
        SimpleFluid::VTUWriter::parallel_index_filename(
            base_filename);

    std::error_code cleanup_error;
    std::filesystem::remove(piece_filename, cleanup_error);
    EXPECT_FALSE(cleanup_error);
    if (rank == 0)
    {
        cleanup_error.clear();
        std::filesystem::remove_all(index_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
        cleanup_error.clear();
        const auto directory_created =
            std::filesystem::create_directory(
                index_filename, cleanup_error);
        EXPECT_TRUE(directory_created);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, {});
    EXPECT_THROW(
        solver.write_parallel_solution_vtu(base_filename),
        std::exception);

    cleanup_error.clear();
    std::filesystem::remove(piece_filename, cleanup_error);
    EXPECT_FALSE(cleanup_error);
    if (rank == 0)
    {
        cleanup_error.clear();
        std::filesystem::remove_all(index_filename, cleanup_error);
        EXPECT_FALSE(cleanup_error);
    }
    communicator->barrier();
}

/** @brief Verify coupled continuity residuals use a global reduction. */
TEST(FluidSolverMultiRankTest, CoupledContinuityResidualIsGlobal)
{
    auto mesh = distributed_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    const auto boundary_conditions = cavity_boundary_conditions();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 2.0e-3;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 300;
    // Retain a measurable converged residual so the global reduction check
    // cannot become vacuous when the coupled discretization is exact.
    linear_options.tolerance = 1.0e-6;

    SimpleFluid::FluidSolver<Pack> solver(
        mesh, boundary_conditions, time_options, linear_options);
    solver.step();
    ASSERT_TRUE(solver.last_step_statistics().converged);

    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    FaceFieldType face_fluxes(mesh, "independent_coupled_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        solver.velocity(),
        solver.pressure(),
        time_options.time_step,
        boundary_cache,
        boundary_conditions.pressure,
        face_fluxes);

    double local_squared_norm = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto balance =
            SimpleFluid::FVM::cell_flux_balance<Pack>(
                *mesh, face_fluxes, cell_lid);
        local_squared_norm += balance * balance;
    }

    const auto global_squared_norm =
        global_sum(*mesh, local_squared_norm);
    ASSERT_GT(global_squared_norm, 1.0e-24)
        << "The fixture must exercise a nonzero continuity norm.";
    const auto expected = std::sqrt(global_squared_norm);
    const auto actual =
        solver.last_pressure_velocity_residuals().continuity;
    EXPECT_NEAR(
        actual,
        expected,
        std::max(1.0e-14, expected * 1.0e-12));
}

/** @brief Verify a remote Dirichlet pressure patch disables the global gauge row. */
TEST(FluidSolverMultiRankTest,
     CoupledDirichletPressureSuppressesGaugeOnEveryRank)
{
    auto mesh = distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    int local_owns_pressure_boundary = 0;
    for (const auto& [batch_id, boundary_batch] :
         mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) != "xmax")
        {
            continue;
        }
        for (const auto face_lid : boundary_batch.face_lids)
        {
            if (mesh->is_owned_face(face_lid)
                && mesh->is_boundary_face(face_lid))
            {
                local_owns_pressure_boundary = 1;
                break;
            }
        }
    }
    int minimum_boundary_presence = 0;
    int maximum_boundary_presence = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_owns_pressure_boundary,
        &minimum_boundary_presence);
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_owns_pressure_boundary,
        &maximum_boundary_presence);
    ASSERT_EQ(minimum_boundary_presence, 0);
    ASSERT_EQ(maximum_boundary_presence, 1);

    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    boundary_conditions.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, {}};
    for (const auto* name :
         {"xmin", "ymin", "ymax", "zmin", "zmax"})
    {
        boundary_conditions.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 400;
    // The traction-free velocity outlet leaves pressure accuracy controlled
    // directly by the coupled Krylov solve. Keep the solve tolerance one
    // decade below the asserted boundary-pressure error.
    linear_options.tolerance = 1.0e-11;
    SimpleFluid::FluidSolver<Pack> solver(
        mesh,
        boundary_conditions,
        time_options,
        linear_options);
    solver.step();
    ASSERT_TRUE(solver.last_step_statistics().converged);

    double local_pressure_error = 0.0;
    double local_velocity_error = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        local_pressure_error = std::max(
            local_pressure_error,
            std::abs(solver.pressure().value(cell_lid) - 1.0));
        const auto velocity = solver.velocity().value(cell_lid);
        local_velocity_error = std::max(
            local_velocity_error,
            std::sqrt(velocity.dot(velocity)));
    }
    double global_pressure_error = 0.0;
    double global_velocity_error = 0.0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_pressure_error,
        &global_pressure_error);
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_velocity_error,
        &global_velocity_error);

    EXPECT_LT(global_pressure_error, 1.0e-8);
    EXPECT_LT(global_velocity_error, 1.0e-8);
    EXPECT_LT(
        solver.last_pressure_velocity_residuals().continuity,
        1.0e-8);
}
