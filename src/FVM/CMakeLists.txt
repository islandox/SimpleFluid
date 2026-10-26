# PLIC reconstruction and phase arithmetic live with geometry, while their
# established shared-library symbols remain owned/exported by SimpleFluidFVM.
set(SIMPLEFLUID_GENERAL_VOF_INSTANTIATION_SOURCES
    explicit_instantiation/GeneralVoF/Transport.cc
    explicit_instantiation/GeneralVoF/Remap.cc
    explicit_instantiation/GeneralVoF/PhaseAllocation.cc
)
add_library(SimpleFluidFVM SHARED
    GeneralVoF.cc
    details/GeneralVoFState.cc
    details/GeneralVoF/Transport.cc
    details/GeneralVoF/FaceTransport.cc
    details/GeneralVoF/Surface.cc
    details/GeneralVoF/Transport.tcc
    details/GeneralVoF/Remap.cc
    details/GeneralVoF/Remap.tcc
    ${SIMPLEFLUID_GENERAL_VOF_INSTANTIATION_SOURCES}
    details/GeneralVoF/PhaseAllocation.tcc
    details/GeneralVoF/Wire.cc
    details/GeneralVoFExactTransport.cc
    details/GeneralVoFExactTransfer.cc
    ExactPhaseTransfer.cc
    ExactFaceTransfer.cc
    ../geometry/PLIC/details/exact/PolyhedralReconstruction.cc
    ../geometry/PLIC/PolyhedralReconstruction.cc
    ../geometry/PLIC/details/PolyhedralReconstruction.tcc
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
    # Assembly-only providers; iterative solves live in SimpleFluidSolvers.
    explicit_instantiation/ALEControlVolumeState.cc
    explicit_instantiation/CellGradientCache.cc
    explicit_instantiation/GaussLinearGradientCache.cc
    explicit_instantiation/NonOrthogonalCorrection.cc
    explicit_instantiation/TransportGeometryCache.cc
    explicit_instantiation/MappedWeightedTransportSystem.cc
    explicit_instantiation/FieldStoredTransportSystem/ScalarMeshHandle.cc
    explicit_instantiation/FieldStoredTransportSystem/BasicVectorMeshHandle.cc
    explicit_instantiation/FieldStoredTransportSystem/VectorMeshHandle.cc
    explicit_instantiation/FieldStoredTransportSystem/MomentumMeshHandle.cc
    explicit_instantiation/FieldStoredTransportSystem/ScalarPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/WeightedTemperaturePartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/VectorPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/MomentumPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/SupportPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/ScalarSolidSubdomain.cc
    explicit_instantiation/FieldStoredTransportSystem/TemperatureSolidSubdomain.cc
    explicit_instantiation/FieldStoredOperators/CellMeshHandle.cc
    explicit_instantiation/FieldStoredOperators/DiffusionMeshHandle.cc
    explicit_instantiation/FieldStoredOperators/FaceMeshHandle.cc
    explicit_instantiation/FieldStoredOperators/PressureMeshHandle.cc
    explicit_instantiation/FieldStoredOperators/WorkspaceMeshHandle.cc
    explicit_instantiation/FieldStoredOperators/CellPartitionedCartesian.cc
    explicit_instantiation/FieldStoredOperators/DiffusionPartitionedCartesian.cc
    explicit_instantiation/FieldStoredOperators/FacePartitionedCartesian.cc
    explicit_instantiation/FieldStoredOperators/PressurePartitionedCartesian.cc
    explicit_instantiation/FieldStoredOperators/WorkspacePartitionedCartesian.cc
    explicit_instantiation/FieldStoredOperators/SupportSolidSubdomain.cc
    explicit_instantiation/FieldStoredOperators/CellSolidSubdomain.cc
    explicit_instantiation/FieldStoredOperators/DiffusionSolidSubdomain.cc
)
add_library(SimpleFluid::FVM ALIAS SimpleFluidFVM)
simplefluid_configure_component(SimpleFluidFVM)
simplefluid_debug_symbols_target(SimpleFluidFVM FVM)
target_link_libraries(SimpleFluidFVM PUBLIC SimpleFluidFields)

# Splitting providers lowers memory per compiler process, but each still parses
# the transport implementation. Keep FVM compilation bounded under Ninja even
# when the enclosing build uses a larger --parallel value.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS simplefluid_fvm_compile=1)
set_property(TARGET SimpleFluidFVM PROPERTY JOB_POOL_COMPILE simplefluid_fvm_compile)

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
simplefluid_exclude_from_pgo_sources(GeneralVoF.cc details/GeneralVoFState.cc details/GeneralVoF/Surface.cc details/GeneralVoF/FaceTransport.cc details/GeneralVoF/Transport.cc details/GeneralVoF/Remap.cc ${SIMPLEFLUID_GENERAL_VOF_INSTANTIATION_SOURCES} details/GeneralVoF/Wire.cc details/GeneralVoFExactTransport.cc details/GeneralVoFExactTransfer.cc ExactPhaseTransfer.cc ExactFaceTransfer.cc ../geometry/PLIC/details/exact/PolyhedralReconstruction.cc ../geometry/PLIC/PolyhedralReconstruction.cc GeometricVoF.cc AdaptiveVoF.cc ConservativeAMRRemap.cc DistributedAdaptiveState.cc details/DistributedAMROverlap.cc details/ExactPhaseLabelMap.cc details/ExactLiveStorage.cc details/ExactPhaseAccumulator.cc details/PhaseArithmetic.cc ../geometry/PLIC/details/PhaseArithmetic.cc
    details/FiniteVoFTransport.cc details/ExperimentalPhaseBackend.cc)

# Error-free transforms and directed residual bounds require strict arithmetic in
# every caller translation unit; a strictly compiled geometry callee alone does
# not constrain reassociation of its inline finite ledger helpers.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(GeneralVoF.cc details/GeneralVoFState.cc details/GeneralVoF/Surface.cc details/GeneralVoF/FaceTransport.cc details/GeneralVoF/Transport.cc details/GeneralVoF/Remap.cc ${SIMPLEFLUID_GENERAL_VOF_INSTANTIATION_SOURCES} details/GeneralVoF/Wire.cc details/GeneralVoFExactTransport.cc details/GeneralVoFExactTransfer.cc PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto"
        SKIP_UNITY_BUILD_INCLUSION TRUE
        SKIP_PRECOMPILE_HEADERS TRUE)
endif()
