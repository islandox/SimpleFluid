set(SIMPLEFLUID_EQUATION_SOURCES
    explicit_instantiation/BoilingSourceModel.cc
    explicit_instantiation/DelayedNeutronPrecursorModel.cc
    explicit_instantiation/FissionPowerSource.cc
    explicit_instantiation/ScalarVoidFractionModel.cc
    explicit_instantiation/RadiolyticGasModel.cc
    explicit_instantiation/BoussinesqMomentumEquation.cc
    explicit_instantiation/IncompressibleMomentumEquation.cc
    explicit_instantiation/PressureProjectionEquation.cc
    RadiolyticGasPhysics.cc
    RadiolyticGasProperties.cc
    explicit_instantiation/SolidHeatConductionEquation.cc
    explicit_instantiation/TemperatureDiffusionEquation.cc
    turbulence/TurbulenceModel.cc
    explicit_instantiation/TurbulenceScalarTransportEquation.cc
    turbulence/TurbulenceWallTreatment.cc
    turbulence/WallDistanceEquation.cc
)
simplefluid_absolute_sources(SIMPLEFLUID_EQUATION_SOURCES)
set(SIMPLEFLUID_EQUATION_SOURCES "${SIMPLEFLUID_EQUATION_SOURCES}" PARENT_SCOPE)

# Equations and solvers share one physical DSO; retain the logical facade.
add_library(SimpleFluidEquations INTERFACE)
add_library(SimpleFluid::Equations ALIAS SimpleFluidEquations)
target_link_libraries(SimpleFluidEquations INTERFACE SimpleFluidSolvers)
