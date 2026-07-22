module;

#include "cmake/StandardHeaders.hh"

#include "solvers/BelosLinearSolver.hh"

export module SimpleFluid.LinearSolvers;

export import SimpleFluid.Fields;

export namespace SimpleFluid
{
using ::SimpleFluid::LinearPreconditioner;
using ::SimpleFluid::to_string;
using ::SimpleFluid::LinearSolveStatistics;
using ::SimpleFluid::LinearSolveSummary;
using ::SimpleFluid::LinearSolverOptions;
using ::SimpleFluid::BelosLinearSolver;
using ::SimpleFluid::solve_linear_system;
}
