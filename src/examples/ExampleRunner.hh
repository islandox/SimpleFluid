/**
 * @file ExampleRunner.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shared boilerplate for Boussinesq example executables.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace SimpleFluid
{

/** @brief Allocate the database shared by an example configuration. */
inline std::shared_ptr<Database> make_example_database()
{
    return std::make_shared<Database>();
}

namespace detail
{

/**
 * @brief Return a collision-free VTU piece filename for this communicator rank.
 *
 * Serial output preserves the caller-provided filename. Distributed output
 * inserts a rank suffix before the conventional `.vtu` extension so that no
 * two ranks truncate or overwrite the same file; the solver also emits a
 * `.pvtu` index referencing these pieces.
 */
inline std::string rank_local_vtu_filename(
    const std::string& filename,
    int rank,
    int communicator_size)
{
    return VTUWriter::rank_piece_filename(
        filename, rank, communicator_size);
}

} // namespace detail

/**
 * @brief Build a mesh, run a configured Boussinesq example, and write VTU output.
 *
 * The caller owns Tpetra initialization so ScopeGuard lifetime remains in main().
 *
 * @tparam Pack Tpetra type pack for mesh storage and solver.
 * @tparam Initializer Callable type that accepts a BoussinesqSolver reference
 *                     and initialises the fields (e.g., temperature ramp).
 * @param db Database with mesh configuration parameters.
 * @param boundary_conditions Boundary condition set for temperature and velocity.
 * @param time_options Time-stepping parameters.
 * @param linear_options Linear solver parameters.
 * @param initialize Callable used to initialise solver fields before running.
 * @param vtu_filename Output VTU file path.
 * @param output_options Optional field-selection controls for solution output.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class Initializer>
void run_boussinesq_example(
    const std::shared_ptr<Database>& db,
    const BoundaryConditionSet& boundary_conditions,
    const TimeStepperOptions& time_options,
    const LinearSolverOptions& linear_options,
    Initializer&& initialize,
    const std::string& vtu_filename,
    const SolutionOutputOptions& output_options = {})
{
    MeshFactory factory(db);
    auto mesh = factory.build_handle<Pack>();

    BoussinesqSolver<Pack> solver(mesh, boundary_conditions,
                                  time_options, linear_options);
    std::forward<Initializer>(initialize)(solver);
    solver.run();

    const auto communicator = mesh->owned_cell_map()->getComm();
    solver.write_parallel_solution_vtu(vtu_filename, output_options);

    if (communicator->getRank() == 0)
    {
        std::cout << "Wrote "
                  << (communicator->getSize() == 1
                    ? vtu_filename
                    : VTUWriter::parallel_index_filename(vtu_filename))
                  << " at t=" << solver.time() << "\n";
    }
}

/**
 * @brief Build and run a Boussinesq example with explicit material options.
 *
 * This overload is intended for physical multiphysics examples whose
 * dimensional boundary data must be consistent with the material fields.
 *
 * @tparam Pack Tpetra type pack for mesh storage and solver.
 * @tparam Initializer Callable type that accepts a BoussinesqSolver reference.
 * @param db Database with mesh configuration parameters.
 * @param boundary_conditions Boundary condition set for temperature and velocity.
 * @param time_options Time-stepping parameters.
 * @param linear_options Linear solver parameters.
 * @param model_options Dimensional material-property options.
 * @param initialize Callable used to initialise and configure the solver.
 * @param vtu_filename Output VTU file path.
 * @param output_options Optional field-selection controls for solution output.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class Initializer>
void run_boussinesq_example(
    const std::shared_ptr<Database>& db,
    const BoundaryConditionSet& boundary_conditions,
    const TimeStepperOptions& time_options,
    const LinearSolverOptions& linear_options,
    BoussinesqModelOptions model_options,
    Initializer&& initialize,
    const std::string& vtu_filename,
    const SolutionOutputOptions& output_options = {})
{
    MeshFactory factory(db);
    auto mesh = factory.build_handle<Pack>();

    BoussinesqSolver<Pack> solver(
        mesh,
        boundary_conditions,
        time_options,
        linear_options,
        std::move(model_options));
    std::forward<Initializer>(initialize)(solver);
    solver.run();

    const auto communicator = mesh->owned_cell_map()->getComm();
    solver.write_parallel_solution_vtu(vtu_filename, output_options);

    if (communicator->getRank() == 0)
    {
        std::cout << "Wrote "
                  << (communicator->getSize() == 1
                    ? vtu_filename
                    : VTUWriter::parallel_index_filename(vtu_filename))
                  << " at t=" << solver.time() << "\n";
    }
}

} // namespace SimpleFluid
