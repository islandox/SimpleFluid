add_library(SimpleFluidFVM SHARED
    GeneralVoF.cc
    ExactPhaseTransfer.cc
    ExactFaceTransfer.cc
    PolyhedralReconstruction.cc
    FinitePolyhedralReconstruction.cc
    GeometricVoF.cc
    AdaptiveVoF.cc
    ConservativeAMRRemap.cc
    DistributedAdaptiveState.cc
    details/DistributedAMROverlap.cc
    details/ExactPhaseLabelMap.cc
    details/ExactLiveStorage.cc
    details/ExactPhaseAccumulator.cc
    details/PhaseArithmetic.cc
    details/ExperimentalPhaseBackend.cc
    # Non-orthogonal FVM solves and higher-level solvers share this instance.
    explicit_instantiation/DICPreconditioner.cc
    explicit_instantiation/ALEControlVolumeState.cc
    explicit_instantiation/CellGradientCache.cc
    explicit_instantiation/GaussLinearGradientCache.cc
    explicit_instantiation/NonOrthogonalCorrection.cc
    explicit_instantiation/TransportSystem.cc
)
add_library(SimpleFluid::FVM ALIAS SimpleFluidFVM)
simplefluid_configure_component(SimpleFluidFVM)
target_link_libraries(SimpleFluidFVM PUBLIC SimpleFluidFields)

# Forward-error bounds require the same strict arithmetic used by the numerical
# geometry kernel. Do not import relaxed arithmetic through LTO.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(FinitePolyhedralReconstruction.cc PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION OFF
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto")
endif()

# The independent VoF workstream is outside the existing Shiri PGO workload.
simplefluid_exclude_from_pgo_sources(GeneralVoF.cc ExactPhaseTransfer.cc ExactFaceTransfer.cc PolyhedralReconstruction.cc FinitePolyhedralReconstruction.cc GeometricVoF.cc AdaptiveVoF.cc ConservativeAMRRemap.cc DistributedAdaptiveState.cc details/DistributedAMROverlap.cc details/ExactPhaseLabelMap.cc details/ExactLiveStorage.cc details/ExactPhaseAccumulator.cc details/PhaseArithmetic.cc details/ExperimentalPhaseBackend.cc)

# Error-free transforms and directed residual bounds require strict arithmetic in
# every caller translation unit; a strictly compiled geometry callee alone does
# not constrain reassociation of its inline finite ledger helpers.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(GeneralVoF.cc PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()
