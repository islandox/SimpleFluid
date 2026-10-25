# PLIC reconstruction and phase arithmetic live with geometry, while their
# established shared-library symbols remain owned/exported by SimpleFluidFVM.
add_library(SimpleFluidFVM SHARED
    GeneralVoF.cc
    details/GeneralVoFState.cc
    details/GeneralVoFFiniteTransport.cc
    details/GeneralVoFFiniteTransport.tcc
    details/GeneralVoFExactTransport.cc
    details/GeneralVoFExactTransfer.cc
    ExactPhaseTransfer.cc
    ExactFaceTransfer.cc
    ../geometry/PLIC/details/exact/PolyhedralReconstruction.cc
    ../geometry/PLIC/PolyhedralReconstruction.cc
    ../geometry/PLIC/details/FinitePolyhedralReconstruction.tcc
    GeometricVoF.cc
    AdaptiveVoF.cc
    ConservativeAMRRemap.cc
    DistributedAdaptiveState.cc
    details/DistributedAMROverlap.cc
    details/ExactPhaseLabelMap.cc
    details/ExactLiveStorage.cc
    details/ExactPhaseAccumulator.cc
    details/PhaseArithmetic.cc
    ../geometry/PLIC/details/PhaseArithmetic.cc
    details/FiniteVoFTransport.cc
    details/FiniteVoFTransport.tcc
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
    set_source_files_properties(../geometry/PLIC/PolyhedralReconstruction.cc ../geometry/PLIC/details/PhaseArithmetic.cc
        details/FiniteVoFTransport.cc PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION OFF
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()

# The independent VoF workstream is outside the existing Shiri PGO workload.
simplefluid_exclude_from_pgo_sources(GeneralVoF.cc details/GeneralVoFState.cc details/GeneralVoFFiniteTransport.cc details/GeneralVoFExactTransport.cc details/GeneralVoFExactTransfer.cc ExactPhaseTransfer.cc ExactFaceTransfer.cc ../geometry/PLIC/details/exact/PolyhedralReconstruction.cc ../geometry/PLIC/PolyhedralReconstruction.cc GeometricVoF.cc AdaptiveVoF.cc ConservativeAMRRemap.cc DistributedAdaptiveState.cc details/DistributedAMROverlap.cc details/ExactPhaseLabelMap.cc details/ExactLiveStorage.cc details/ExactPhaseAccumulator.cc details/PhaseArithmetic.cc ../geometry/PLIC/details/PhaseArithmetic.cc
    details/FiniteVoFTransport.cc details/ExperimentalPhaseBackend.cc)

# Error-free transforms and directed residual bounds require strict arithmetic in
# every caller translation unit; a strictly compiled geometry callee alone does
# not constrain reassociation of its inline finite ledger helpers.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(GeneralVoF.cc details/GeneralVoFState.cc details/GeneralVoFFiniteTransport.cc details/GeneralVoFExactTransport.cc details/GeneralVoFExactTransfer.cc PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()
