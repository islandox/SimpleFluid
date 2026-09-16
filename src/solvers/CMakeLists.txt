set(SIMPLEFLUID_SOLVER_SOURCES
    explicit_instantiation/LiquidMassInventory.cc
    explicit_instantiation/VolumeContinuityModel.cc
    explicit_instantiation/PlanarALEBoundary.cc
    explicit_instantiation/CoupledBlockOperator.cc
    explicit_instantiation/DICPreconditioner.cc
    explicit_instantiation/BoussinesqSolver.cc
    explicit_instantiation/CoupledPressureVelocitySolver.cc
    CoupledNonlinearProblem.cc
    NoxNonlinearSolver.cc
    explicit_instantiation/FluidSolver.cc
    explicit_instantiation/IncompressibleIsothermalSolver.cc
    PlanarFreeSurfaceModel.cc
    SteadyStateSearch.cc
)

# Keep equations and solvers in one physical DSO to preserve cross-layer type
# identity. The FVM DSO below it now supplies the canonical Kokkos runtime
# bridge on ELF when Trilinos is static; the version map localizes the other
# embedded vendor implementations. The logical equation and solver targets
# remain separate.
add_library(SimpleFluidSolvers SHARED
    ${SIMPLEFLUID_EQUATION_SOURCES}
    ${SIMPLEFLUID_SOLVER_SOURCES}
)
add_library(SimpleFluid::Solvers ALIAS SimpleFluidSolvers)
simplefluid_configure_component(SimpleFluidSolvers)
target_compile_definitions(SimpleFluidSolvers PRIVATE
    SimpleFluidEquations_EXPORTS)
target_link_libraries(SimpleFluidSolvers PUBLIC SimpleFluidFVM)
if(SIMPLEFLUID_ENABLE_NOX)
    target_compile_definitions(SimpleFluidSolvers PUBLIC SIMPLEFLUID_ENABLE_NOX=1)
endif()

simplefluid_add_equations_linker_alias()

# SimpleFluid does not expose ELF symbol interposition as an extension API.
# Let optimized shared-library builds bind their own definitions locally when
# the compiler supports that model.
if(SIMPLEFLUID_CXX_HAS_FNO_SEMANTIC_INTERPOSITION)
    target_compile_options(SimpleFluidSolvers PRIVATE
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-fno-semantic-interposition>")
endif()
