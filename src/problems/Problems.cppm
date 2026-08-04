module;

#if !defined(SIMPLEFLUID_USE_STD_MODULE)
#include "cmake/StandardHeaders.hh"
#endif

#include "problems/Problem.hh"

export module SimpleFluid.Problems;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export import SimpleFluid.Equations;

export namespace SimpleFluid
{
using ::SimpleFluid::Problem;
}
