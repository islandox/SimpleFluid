set(SIMPLEFLUID_SOLVER_SOURCES
    GeneralVoFSolver.cc
    VoFSolver.cc
    details/ConservativeVoFTransport.cc
    details/VoFProjectionWorkspace.cc
    details/GeneralVoFPhiClosure.cc
    details/GeneralVoFFinitePhiClosure.inc
    details/GeneralVoFProjectionArithmetic.cc
    details/GeneralVoFExactPressure.cc
    details/GeneralVoFFlowWorkspace.cc
    details/GeneralVoFFiniteProjection.inc
    details/GeneralVoFCurvature.cc
    details/GeneralVoFExactStorage.cc
    details/ConservativeVoFExactArithmetic.cc
    details/GeneralVoFTransferBinding.cc
    AdaptiveVoFSolver.cc
    VoFLowMach.cc
    RadiolyticVoFCoupler.cc
    BubbleRepresentation.cc
    explicit_instantiation/LiquidMassInventory.cc
    explicit_instantiation/VolumeContinuityModel.cc
    explicit_instantiation/PlanarALEBoundary.cc
    explicit_instantiation/CoupledBlockOperator.cc
    explicit_instantiation/BoussinesqSolver.cc
    explicit_instantiation/MultiphysicsSolverCoupler.cc
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
simplefluid_exclude_from_pgo_sources(VoFSolver.cc)
simplefluid_exclude_from_pgo_sources(details/ConservativeVoFTransport.cc)
simplefluid_exclude_from_pgo_sources(details/VoFProjectionWorkspace.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFFlowWorkspace.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFCurvature.cc)
simplefluid_exclude_from_pgo_sources(details/ConservativeVoFExactArithmetic.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFTransferBinding.cc)
simplefluid_exclude_from_pgo_sources(AdaptiveVoFSolver.cc)
simplefluid_exclude_from_pgo_sources(VoFLowMach.cc)
simplefluid_exclude_from_pgo_sources(RadiolyticVoFCoupler.cc)
simplefluid_exclude_from_pgo_sources(BubbleRepresentation.cc)
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

simplefluid_exclude_from_pgo_sources(details/GeneralVoFPhiClosure.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFProjectionArithmetic.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFExactPressure.cc)

simplefluid_exclude_from_pgo_sources(details/GeneralVoFExactStorage.cc)

simplefluid_exclude_from_pgo_sources(GeneralVoFSolver.cc)

# Bounded interval/ledger proofs constrain every caller, including inline helpers.
# Per-source options follow target/global options and keep relaxed arithmetic or
# link-time transforms from crossing this numerical boundary.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        GeneralVoFSolver.cc
        details/ConservativeVoFTransport.cc
        details/GeneralVoFFlowWorkspace.cc
        details/GeneralVoFPhiClosure.cc
        details/GeneralVoFCurvature.cc
        PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()
