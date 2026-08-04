module;

#if !defined(SIMPLEFLUID_USE_STD_MODULE)
#include "cmake/StandardHeaders.hh"
#endif

#include "solvers/CoupledPressureVelocitySolver.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/BoussinesqSolver.hh"

export module SimpleFluid.Solvers;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export import SimpleFluid.Problems;

export namespace SimpleFluid
{
using ::SimpleFluid::CoupledPressureVelocitySystem;
using ::SimpleFluid::CoupledPressureVelocityResult;
using ::SimpleFluid::CoupledPressureVelocitySolver;
using ::SimpleFluid::FluidSolver;
using ::SimpleFluid::BoussinesqSolver;
}
