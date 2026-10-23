set(SIMPLEFLUID_EQUATION_SOURCES
    ReactionMechanism.cc
    ReactingThermodynamics.cc
    explicit_instantiation/BoilingSourceModel.cc
    explicit_instantiation/ScalarSourceDecayTransport.cc
    explicit_instantiation/PrescribedHeatSource.cc
    explicit_instantiation/ScalarVoidFractionModel.cc
    explicit_instantiation/ReactingGasModel.cc
    explicit_instantiation/BoussinesqMomentumEquation.cc
    explicit_instantiation/IncompressibleMomentumEquation.cc
    LumpedInnerCavityModel.cc
    GrayDiffuseEnclosure.cc
    VentedLumpedCavityModel.cc
    LumpedInnerCavityCoupling.cc
    explicit_instantiation/PressureProjectionEquation.cc
    BubbleGasPhysics.cc
    ReactingGasProperties.cc
    BubblePopulationKinetics.cc
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
