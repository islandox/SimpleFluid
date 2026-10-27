set(SIMPLEFLUID_GENERAL_VOF_COMPONENT_SOURCES
    details/GeneralVoF/Resources.cc
    explicit_instantiation/GeneralVoF/Initialization.cc
    explicit_instantiation/GeneralVoF/Advance.cc
    explicit_instantiation/GeneralVoF/Remap.cc
    details/GeneralVoF/exact/FaceRates.cc
    details/GeneralVoF/exact/Advance.cc
    details/GeneralVoF/exact/Remap.cc
    explicit_instantiation/GeneralVoF/AdaptedFaceTransfer.cc
    explicit_instantiation/GeneralVoF/Curvature.cc
    explicit_instantiation/GeneralVoF/Diagnostics.cc
    details/GeneralVoF/Execution.cc
    explicit_instantiation/GeneralVoF/FaceRates.cc
    details/GeneralVoF/FaceTransfer.cc
    details/GeneralVoF/NodeVelocity.cc
    explicit_instantiation/GeneralVoF/OwnedFaceTransfer.cc
    explicit_instantiation/GeneralVoF/PhaseStorage.cc
    details/GeneralVoF/exact/Diagnostics.cc
    details/GeneralVoF/exact/PhaseStorage.cc
    details/GeneralVoF/exact/Reduction.cc
)
set(SIMPLEFLUID_VOF_SOLVER_SOURCES
    GeneralVoFSolver.cc
    ${SIMPLEFLUID_GENERAL_VOF_COMPONENT_SOURCES}
    details/GeneralVoF/SolverImpl.cc
    details/GeneralVoFFiniteProjection.tcc
    details/ConservativeVoFFiniteTransport.tcc
    VoFSolver.cc
    details/ConservativeVoFTransport.cc
    details/ConservativeVoFExactTransport.cc
    details/VoFProjectionWorkspace.cc
    details/GeneralVoFPhiClosure.cc
    details/GeneralVoFExactPhiClosure.cc
    details/GeneralVoFPhiClosureSupport.cc
    details/GeneralVoFProjectionArithmetic.cc
    details/GeneralVoFExactPressure.cc
    details/GeneralVoFFlowWorkspace.cc
    details/GeneralVoFExactProjection.cc
    details/GeneralVoFFiniteProjectionArithmetic.hh
    details/GeneralVoFCurvature.cc
    details/GeneralVoF/OrdinaryCurvature.cc
    details/GeneralVoFExactStorage.cc
    details/ConservativeVoFExactArithmetic.cc
    details/GeneralVoFTransferBinding.cc
    AdaptiveVoFSolver.cc
    details/AdaptiveVoFContext.cc
    details/AdaptiveVoFLowMach.cc
    details/AdaptiveVoFThermodynamics.cc
    details/AdaptiveVoFLowMachTransport.cc
    details/AdaptiveVoFRepresentation.cc
    VoFLowMach.cc
    # ReactingVoFCoupler depends on retired single-gas source options.
    # Keep its implementation excluded pending species-schema migration.
    BubbleRepresentation.cc
)

set(SIMPLEFLUID_SOLVER_SOURCES
    LowMachMixtureSolver.cc
    ReactingMixtureCoupler.cc
    details/LowMachFaceTransportAdapter.cc
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
    ClosedMixtureHeadspaceModel.cc
    SteadyStateSearch.cc
)

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
    list(APPEND SIMPLEFLUID_SOLVER_SOURCES ${SIMPLEFLUID_VOF_SOLVER_SOURCES})
endif()

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

if(SIMPLEFLUID_ENABLE_EXPERIMENTAL_VOF)
    simplefluid_exclude_from_pgo_sources(VoFSolver.cc)
    simplefluid_exclude_from_pgo_sources(details/ConservativeVoFTransport.cc)
    simplefluid_exclude_from_pgo_sources(details/VoFProjectionWorkspace.cc)
    simplefluid_exclude_from_pgo_sources(details/GeneralVoFFlowWorkspace.cc)
    simplefluid_exclude_from_pgo_sources(details/GeneralVoFCurvature.cc details/GeneralVoF/OrdinaryCurvature.cc)
    simplefluid_exclude_from_pgo_sources(details/ConservativeVoFExactArithmetic.cc)
    simplefluid_exclude_from_pgo_sources(details/GeneralVoFTransferBinding.cc)
    simplefluid_exclude_from_pgo_sources(AdaptiveVoFSolver.cc)
    simplefluid_exclude_from_pgo_sources(VoFLowMach.cc)
    simplefluid_exclude_from_pgo_sources(ReactingVoFCoupler.cc)
    simplefluid_exclude_from_pgo_sources(BubbleRepresentation.cc)
    simplefluid_exclude_from_pgo_sources(details/GeneralVoFPhiClosure.cc)
    simplefluid_exclude_from_pgo_sources(details/GeneralVoFProjectionArithmetic.cc)
    simplefluid_exclude_from_pgo_sources(details/GeneralVoFExactPressure.cc)
    simplefluid_exclude_from_pgo_sources(details/GeneralVoFExactStorage.cc)
    simplefluid_exclude_from_pgo_sources(GeneralVoFSolver.cc details/GeneralVoF/SolverImpl.cc)
    # Bounded interval/ledger proofs constrain every caller, including inline helpers.
    # Per-source options follow target/global options and keep relaxed arithmetic or
    # link-time transforms from crossing this numerical boundary.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            GeneralVoFSolver.cc
    details/GeneralVoF/SolverImpl.cc
    details/GeneralVoFFiniteProjection.tcc
    details/ConservativeVoFFiniteTransport.tcc
            details/ConservativeVoFTransport.cc
            details/GeneralVoFFlowWorkspace.cc
            details/GeneralVoFPhiClosure.cc
            details/GeneralVoFCurvature.cc
    details/GeneralVoF/OrdinaryCurvature.cc
            PROPERTIES
            COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
            SKIP_UNITY_BUILD_INCLUSION TRUE
            SKIP_PRECOMPILE_HEADERS TRUE)
    endif()
endif()

# Both arithmetic backends retain the owning implementation's numerical boundary.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        details/ConservativeVoFExactTransport.cc
        details/GeneralVoFExactProjection.cc
        details/GeneralVoFExactPhiClosure.cc
        details/GeneralVoFPhiClosureSupport.cc
        PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()
simplefluid_exclude_from_pgo_sources(details/ConservativeVoFExactTransport.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFExactProjection.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFExactPhiClosure.cc)
simplefluid_exclude_from_pgo_sources(details/GeneralVoFPhiClosureSupport.cc)

simplefluid_exclude_from_pgo_sources(details/AdaptiveVoFContext.cc)
simplefluid_exclude_from_pgo_sources(details/AdaptiveVoFLowMach.cc)
simplefluid_exclude_from_pgo_sources(details/AdaptiveVoFRepresentation.cc)
simplefluid_exclude_from_pgo_sources(details/AdaptiveVoFThermodynamics.cc)
simplefluid_exclude_from_pgo_sources(details/AdaptiveVoFLowMachTransport.cc)

# General VoF operation components retain their original numerical boundary.
foreach(source IN LISTS SIMPLEFLUID_GENERAL_VOF_COMPONENT_SOURCES)
    simplefluid_exclude_from_pgo_sources(${source})
endforeach()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${SIMPLEFLUID_GENERAL_VOF_COMPONENT_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()
