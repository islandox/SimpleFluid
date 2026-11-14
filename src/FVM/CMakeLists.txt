add_library(SimpleFluidFVM SHARED
    # Assembly-only providers; iterative solves live in SimpleFluidSolvers.
    explicit_instantiation/ALEControlVolumeState.cc
    explicit_instantiation/CellGradientCache.cc
    explicit_instantiation/GaussLinearGradientCache.cc
    explicit_instantiation/NonOrthogonalCorrection.cc
    explicit_instantiation/TransportGeometryCache.cc
    explicit_instantiation/MappedWeightedTransportSystem.cc
    explicit_instantiation/FieldStoredTransportSystem/ScalarMeshHandle.cc
    explicit_instantiation/FieldStoredTransportSystem/VectorMeshHandle.cc
    explicit_instantiation/FieldStoredTransportSystem/MomentumMeshHandle.cc
    explicit_instantiation/FieldStoredTransportSystem/ScalarPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/WeightedTemperaturePartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/VectorPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/MomentumPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/SupportPartitionedCartesian.cc
    explicit_instantiation/FieldStoredTransportSystem/ScalarSolidSubdomain.cc
    explicit_instantiation/FieldStoredTransportSystem/TemperatureSolidSubdomain.cc
    explicit_instantiation/WeightedScalarTransportSystem.cc
    explicit_instantiation/PhysicalTemperatureTransportSystem.cc
    explicit_instantiation/PhysicalMomentumTransportSystem.cc
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
    explicit_instantiation/TransportRhs.cc
    explicit_instantiation/TransportSystem.cc
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
