module;

#include "cmake/StandardHeaders.hh"

#include "examples/ExampleRunner.hh"

export module SimpleFluid.Examples;

export namespace SimpleFluid
{
using ::SimpleFluid::real_t;
using ::SimpleFluid::ArrInt;
using ::SimpleFluid::ArrReal;
using ::SimpleFluid::ArrString;
using ::SimpleFluid::Database;
using ::SimpleFluid::MeshFactory;
using ::SimpleFluid::BoundaryConditionSet;
using ::SimpleFluid::BoundaryConditionType;
using ::SimpleFluid::TimeStepperOptions;
using ::SimpleFluid::LinearSolverOptions;
using ::SimpleFluid::SolutionOutputOptions;
using ::SimpleFluid::FissionPowerProfile;
using ::SimpleFluid::FissionPowerSourceOptions;
using ::SimpleFluid::BoilingSourceOptions;
using ::SimpleFluid::DensityFeedbackMode;
using ::SimpleFluid::MaterialFeedbackOptions;
using ::SimpleFluid::RadiolyticGasMode;
using ::SimpleFluid::RadiolyticGasOptions;
using ::SimpleFluid::ScalarVoidFractionOptions;
using ::SimpleFluid::make_example_database;
using ::SimpleFluid::run_boussinesq_example;
}

export namespace Tpetra
{
using ::Tpetra::ScopeGuard;
}
