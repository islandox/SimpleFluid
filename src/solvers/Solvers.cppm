module;

#if !defined(SIMPLEFLUID_USE_STD_MODULE)
#include "cmake/StandardHeaders.hh"
#endif

#include "solvers/CoupledPressureVelocitySolver.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/SolverProgress.hh"
#include "solvers/SteadyStateSearch.hh"

export module SimpleFluid.Solvers;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export import SimpleFluid.Problems;

export namespace SimpleFluid
{
using ::SimpleFluid::CoupledPressureVelocitySystem;
using ::SimpleFluid::CoupledPressureVelocityResult;
using ::SimpleFluid::CoupledRebuildPolicy;
using ::SimpleFluid::CoupledPressureVelocityCacheStatistics;
using ::SimpleFluid::CoupledPressureVelocitySolver;
using ::SimpleFluid::FluidSolver;
using ::SimpleFluid::BoussinesqSolver;
using ::SimpleFluid::ProgressLineFormatter;
using ::SimpleFluid::ProgressStream;
using ::SimpleFluid::SteadyStateUpdateScales;
using ::SimpleFluid::SteadyStateSearchOptions;
using ::SimpleFluid::validate_steady_state_search_options;
using ::SimpleFluid::SteadyStateUpdateRates;
using ::SimpleFluid::SteadyStateStepStatistics;
using ::SimpleFluid::AdaptiveSteadyStateController;
using ::SimpleFluid::SteadyStateFieldMonitor;
using ::SimpleFluid::SteadyStateProgressLineFormatter;
using ::SimpleFluid::SteadyStateProgressStream;
}
