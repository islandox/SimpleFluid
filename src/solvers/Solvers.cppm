module;

#include "cmake/StandardHeaders.hh"

#include "solvers/CoupledPressureVelocitySolver.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/BoussinesqSolver.hh"

export module SimpleFluid.Solvers;

export import SimpleFluid.Problems;

export namespace SimpleFluid
{
using ::SimpleFluid::CoupledPressureVelocitySystem;
using ::SimpleFluid::CoupledPressureVelocityResult;
using ::SimpleFluid::CoupledPressureVelocitySolver;
using ::SimpleFluid::FluidSolver;
using ::SimpleFluid::BoussinesqSolver;
}
