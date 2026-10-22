add_library(SimpleFluidFVM SHARED
    GeneralVoF.cc
    ExactPhaseTransfer.cc
    ExactFaceTransfer.cc
    PolyhedralReconstruction.cc
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

# The independent VoF workstream is outside the existing Shiri PGO workload.
simplefluid_exclude_from_pgo_sources(GeneralVoF.cc ExactPhaseTransfer.cc ExactFaceTransfer.cc PolyhedralReconstruction.cc GeometricVoF.cc AdaptiveVoF.cc ConservativeAMRRemap.cc DistributedAdaptiveState.cc details/DistributedAMROverlap.cc details/ExactPhaseLabelMap.cc details/ExactLiveStorage.cc details/ExactPhaseAccumulator.cc details/PhaseArithmetic.cc details/ExperimentalPhaseBackend.cc)
