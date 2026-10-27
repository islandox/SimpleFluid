# Provider responsibilities are grouped within their established owning DSO.
# General-native source-free and shared VoF providers remain unconditional in FVM.
set(SIMPLEFLUID_GENERAL_VOF_INSTANTIATION_SOURCES
    explicit_instantiation/GeneralVoF/Transport.cc
    explicit_instantiation/GeneralVoF/Remap.cc
    explicit_instantiation/GeneralVoF/PhaseAllocation.cc
)
set(SIMPLEFLUID_GENERAL_NATIVE_VOF_FVM_SOURCES
    GeneralVoF.cc
    details/GeneralVoFState.cc
    details/GeneralVoF/Transport.cc
    details/GeneralVoF/LocalGeometry.cc
    details/GeneralVoF/AggregateBudgets.cc
    details/GeneralVoF/LocalTransport.cc
    details/GeneralVoF/FaceTransport.cc
    details/GeneralVoF/PreparedTransport.cc
    details/GeneralVoF/Surface.cc
    details/GeneralVoF/FilteredSurface.cc
    details/GeneralVoF/FilteredSurfaceGeometry.cc
    details/GeneralVoF/CapillaryPlanes.cc
    details/GeneralVoF/CapillaryNodeFans.cc
    details/GeneralVoF/Remap.cc
    ${SIMPLEFLUID_GENERAL_VOF_INSTANTIATION_SOURCES}
    details/GeneralVoF/Wire.cc
    details/GeneralVoFExactTransport.cc
    details/GeneralVoFExactTransfer.cc
)
set(SIMPLEFLUID_GENERAL_NATIVE_VOF_FVM_FRAGMENTS
    details/GeneralVoF/Transport.tcc
    details/GeneralVoF/Remap.tcc
    details/GeneralVoF/PhaseAllocation.tcc
)
set(SIMPLEFLUID_ADAPTIVE_VOF_FVM_SOURCES
    AdaptiveVoF.cc
    ConservativeAMRRemap.cc
    DistributedAdaptiveState.cc
    details/DistributedAMROverlap.cc
)
# PLIC reconstruction and phase arithmetic live with geometry, while their
# established shared-library symbols remain owned/exported by SimpleFluidFVM.
set(SIMPLEFLUID_COMMON_VOF_FVM_SOURCES
    ExactPhaseTransfer.cc
    ExactFaceTransfer.cc
    ../geometry/PLIC/details/exact/PolyhedralReconstruction.cc
    ../geometry/PLIC/PolyhedralReconstruction.cc
    GeometricVoF.cc
    details/ExactPhaseLabelMap.cc
    details/ExactLiveStorage.cc
    details/ExactPhaseAccumulator.cc
    details/PhaseArithmetic.cc
    ../geometry/PLIC/details/PhaseArithmetic.cc
    details/FiniteVoFTransport.cc
    details/ExperimentalPhaseBackend.cc
)
set(SIMPLEFLUID_COMMON_VOF_FVM_FRAGMENTS
    ../geometry/PLIC/details/PolyhedralReconstruction.tcc
    details/FiniteVoFTransport.tcc
)
set(SIMPLEFLUID_FVM_COMMON_SOURCES
    details/VoFFailure.cc
    # Non-orthogonal FVM solves and higher-level solvers share this instance.
    explicit_instantiation/DICPreconditioner.cc
    explicit_instantiation/ALEControlVolumeState.cc
    explicit_instantiation/CellGradientCache.cc
    explicit_instantiation/GaussLinearGradientCache.cc
    explicit_instantiation/NonOrthogonalCorrection.cc
    explicit_instantiation/TransportSystem.cc
)
add_library(SimpleFluidFVM SHARED
    ${SIMPLEFLUID_GENERAL_NATIVE_VOF_FVM_SOURCES}
    ${SIMPLEFLUID_GENERAL_NATIVE_VOF_FVM_FRAGMENTS}
    ${SIMPLEFLUID_ADAPTIVE_VOF_FVM_SOURCES}
    ${SIMPLEFLUID_COMMON_VOF_FVM_SOURCES}
    ${SIMPLEFLUID_COMMON_VOF_FVM_FRAGMENTS}
    ${SIMPLEFLUID_FVM_COMMON_SOURCES}
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
simplefluid_exclude_from_pgo_sources(
    ${SIMPLEFLUID_GENERAL_NATIVE_VOF_FVM_SOURCES}
    ${SIMPLEFLUID_ADAPTIVE_VOF_FVM_SOURCES}
    ${SIMPLEFLUID_COMMON_VOF_FVM_SOURCES})

# Error-free transforms and directed residual bounds require strict arithmetic in
# every caller translation unit; a strictly compiled geometry callee alone does
# not constrain reassociation of its inline finite ledger helpers.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${SIMPLEFLUID_GENERAL_NATIVE_VOF_FVM_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()
