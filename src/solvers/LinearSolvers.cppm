module;

#if !defined(SIMPLEFLUID_USE_STD_MODULE)
#include "cmake/StandardHeaders.hh"
#endif

#include "solvers/BelosLinearSolver.hh"

export module SimpleFluid.LinearSolvers;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export import SimpleFluid.Fields;

export namespace SimpleFluid
{
using ::SimpleFluid::LinearSolverBackend;
using ::SimpleFluid::parse_linear_solver_backend;
using ::SimpleFluid::LinearPreconditioner;
using ::SimpleFluid::parse_linear_preconditioner;
using ::SimpleFluid::to_string;
using ::SimpleFluid::LinearResidualScaling;
using ::SimpleFluid::LinearSolveStatistics;
using ::SimpleFluid::LinearSolveSummary;
using ::SimpleFluid::LinearSolverOptions;
using ::SimpleFluid::BelosLinearSolver;
using ::SimpleFluid::solve_linear_system;
}
