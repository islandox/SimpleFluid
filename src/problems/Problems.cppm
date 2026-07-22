module;

#include "cmake/StandardHeaders.hh"

#include "problems/Problem.hh"

export module SimpleFluid.Problems;

export import SimpleFluid.Equations;

export namespace SimpleFluid
{
using ::SimpleFluid::Problem;
}
