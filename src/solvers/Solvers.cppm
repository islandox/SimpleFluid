module;

#include "cmake/StandardHeaders.hh"

#include "problems/Problem.hh"
#include "solvers/CoupledPressureVelocitySolver.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/BoussinesqSolver.hh"

export module SimpleFluid.Solvers;

export import SimpleFluid.Equations;

export namespace SimpleFluid
{
using ::SimpleFluid::Problem;
using ::SimpleFluid::CoupledPressureVelocitySystem;
using ::SimpleFluid::CoupledPressureVelocityResult;
using ::SimpleFluid::CoupledPressureVelocitySolver;
using ::SimpleFluid::FluidSolver;
using ::SimpleFluid::BoussinesqSolver;
}
