/**
 * @file ExampleRunner.hh
 * @brief Shared boilerplate for Boussinesq example executables.
 */
#pragma once

#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Build a mesh, run a configured Boussinesq example, and write VTU output.
 *
 * The caller owns Tpetra initialization so ScopeGuard lifetime remains in main().
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class Initializer>
void run_boussinesq_example(
    const std::shared_ptr<Database>& db,
    const BoundaryConditionSet& boundary_conditions,
    const TimeStepperOptions& time_options,
    const LinearSolverOptions& linear_options,
    Initializer&& initialize,
    const std::string& vtu_filename)
{
    MeshFactory factory(db);
    auto mesh = factory.build<Pack>();

    BoussinesqSolver<Pack> solver(mesh, boundary_conditions,
                                  time_options, linear_options);
    std::forward<Initializer>(initialize)(solver);
    solver.run();
    solver.write_solution_vtu(vtu_filename);

    if (mesh->owned_cell_map()->getComm()->getRank() == 0)
    {
        std::cout << "Wrote " << vtu_filename << " at t="
                  << solver.time() << "\n";
    }
}

} // namespace SimpleFluid
