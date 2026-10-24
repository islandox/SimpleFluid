set(SIMPLEFLUID_MESH_SOURCES
    mesh/AMR/AdaptiveHierarchy.cc
    mesh/AMR/DistributedAdaptiveTypes.cc
    mesh/AMR/DistributedAdaptivePackets.cc
    mesh/AMR/DistributedAdaptiveNative.cc
    mesh/AMR/DistributedAdaptiveNativeGhosts.cc
    mesh/AMR/DistributedAncestryDirectory.cc
    mesh/AMR/AdaptiveProviderKernel.cc
    mesh/AMR/DistributedAdaptiveAuthority.cc
    mesh/AMR/DistributedAdaptiveHierarchy.cc
    mesh/AMR/DistributedAdaptiveClosure.cc
    mesh/AMR/DistributedAdaptiveBalance.cc
    mesh/AMR/DistributedCartesianKernel.cc
    mesh/AMR/DistributedMixedKernel.cc
    mesh/AMR/FrozenNative.cc
    mesh/MultiRegionMeshFrozen.cc
    mesh/AMR/AdaptiveOverlap.cc
    mesh/AMR/ExtrudedAMR.cc
    mesh/AMR/PolyhedralAMR.cc
    mesh/AMR/MixedAMR.cc
    ConvexCellGeometry.cc
    BoundedConvexGeometry.cc
    details/BoundedConvexArithmetic.inc
    details/BoundedConvexPredicates.inc
    details/BoundedConvexMeasures.inc
    details/BoundedConvexConstruction.inc
    details/BoundedConvexClipping.inc
    details/BoundedConvexPartitions.inc
    details/BoundedConvexTrajectories.inc
    details/BoundedConvexPackets.inc
    details/BoundedPlanarFace.inc
    AnnularSemiStructuredMeshFactory.cc
    BoundaryLayerMeshFactory.cc
    mesh/AMR/CartesianAMR.cc
    explicit_instantiation/Mesh.cc
    explicit_instantiation/MeshHandle.cc
    explicit_instantiation/MeshTraverse.cc
    details/LayeredTraverseLayout.cc
    details/SemiStructuredPartition.cc
    explicit_instantiation/PlanarALEMeshMotion.cc
    explicit_instantiation/MeshReorderingFactory.cc
    explicit_instantiation/SolidSubdomain.cc
    explicit_instantiation/STKMesh.cc
    MeshFactory.cc
    explicit_instantiation/LocalGlobalIndexer.cc
    explicit_instantiation/OrthogonalLocalGlobalIndexer.cc
    explicit_instantiation/PartitionedMeshBase.cc
    mesh/OrthoMeshPartitioner.cc
    mesh/OrthogonalMeshDecomposition.cc
    mesh/OrthoMeshTopo.cc
    mesh/OrthogonalCartesian3D.cc
    mesh/OrthogonalCylindrial3D.cc
    mesh/FrontalDelaunay2D.cc
    mesh/PredicateSigns2D.cc
    mesh/SemiStructMeshTopo.cc
    mesh/DecomposedOrthogonalTopology.cc
    mesh/DecomposedSemiStructuredTopology.cc
    mesh/SemiStructuredXY_Z.cc
    mesh/UnstructuredMesh.cc
    mesh/MeshRegion.cc
    mesh/RegionProviders.cc
    mesh/ConservativeInterface.cc
    mesh/ExtrudedRegionProviders.cc
    mesh/ExplicitRegionProvider.cc
    mesh/SweptRZRegionProviders.cc
    mesh/MultiRegionMesh.cc
    mesh/MultiRegionMeshSparse.cc
    mesh/MultiRegionMeshPartition.cc
)

# Private implementation fragments are included only by BoundedConvexGeometry.cc.
set_source_files_properties(
    details/BoundedConvexArithmetic.inc
    details/BoundedConvexPredicates.inc
    details/BoundedConvexMeasures.inc
    details/BoundedConvexConstruction.inc
    details/BoundedConvexClipping.inc
    details/BoundedConvexPartitions.inc
    details/BoundedConvexTrajectories.inc
    details/BoundedConvexPackets.inc
    PROPERTIES HEADER_FILE_ONLY TRUE)

simplefluid_absolute_sources(SIMPLEFLUID_MESH_SOURCES)
list(APPEND SIMPLEFLUID_MESH_SOURCES ${SIMPLEFLUID_IO_SOURCES} ${SIMPLEFLUID_PARALLEL_SOURCES})

# Geometry, partitioning and mesh output retain one archive and type identity.
find_package(Boost CONFIG REQUIRED)
add_library(SimpleFluidMesh STATIC ${SIMPLEFLUID_MESH_SOURCES})
add_library(SimpleFluid::Mesh ALIAS SimpleFluidMesh)
simplefluid_configure_component(SimpleFluidMesh)
target_link_libraries(SimpleFluidMesh PUBLIC SimpleFluidCore PRIVATE Boost::headers)

# The certified predicate filter requires ordinary IEEE binary64 arithmetic.
# Keep this isolated object out of IPO so link-time transforms cannot contract
# its operations or import fast-math assumptions from unrelated code.
set_source_files_properties(mesh/PredicateSigns2D.cc PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(mesh/PredicateSigns2D.cc PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto")
    set_source_files_properties(ConvexCellGeometry.cc BoundedConvexGeometry.cc PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION OFF
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-lto")
endif()

# Require profiles for every Mesh archive member linked into Shiri. The listed
# alternate-path objects are compiled with ordinary LTO optimization in USE
# builds instead of silently applying profile-use without matching data.
set(SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES
    mesh/AMR/AdaptiveHierarchy.cc
    mesh/AMR/DistributedAdaptiveTypes.cc
    mesh/AMR/DistributedAdaptivePackets.cc
    mesh/AMR/DistributedAdaptiveNative.cc
    mesh/AMR/DistributedAdaptiveNativeGhosts.cc
    mesh/AMR/DistributedAncestryDirectory.cc
    mesh/AMR/AdaptiveProviderKernel.cc
    mesh/AMR/DistributedAdaptiveAuthority.cc
    mesh/AMR/DistributedAdaptiveHierarchy.cc
    mesh/AMR/DistributedAdaptiveClosure.cc
    mesh/AMR/DistributedAdaptiveBalance.cc
    mesh/AMR/DistributedCartesianKernel.cc
    mesh/AMR/DistributedMixedKernel.cc
    mesh/AMR/FrozenNative.cc
    mesh/MultiRegionMeshFrozen.cc
    mesh/AMR/AdaptiveOverlap.cc
    mesh/AMR/ExtrudedAMR.cc
    ConvexCellGeometry.cc
    BoundedConvexGeometry.cc
    mesh/DecomposedOrthogonalTopology.cc
    mesh/DecomposedSemiStructuredTopology.cc
    details/SemiStructuredPartition.cc
    AnnularSemiStructuredMeshFactory.cc
    BoundaryLayerMeshFactory.cc
    mesh/AMR/CartesianAMR.cc
    explicit_instantiation/MeshReorderingFactory.cc
    explicit_instantiation/PlanarALEMeshMotion.cc
    explicit_instantiation/SolidSubdomain.cc
    explicit_instantiation/OrthogonalLocalGlobalIndexer.cc
    explicit_instantiation/PartitionedMeshBase.cc
    mesh/SemiStructuredXY_Z.cc
    mesh/SweptRZRegionProviders.cc)
simplefluid_absolute_sources(SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES)
set(SIMPLEFLUID_SHIRI_PROFILED_MESH_SOURCES ${SIMPLEFLUID_MESH_SOURCES})
list(REMOVE_ITEM SIMPLEFLUID_SHIRI_PROFILED_MESH_SOURCES
    ${SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES})
simplefluid_require_complete_pgo_sources(
    ${SIMPLEFLUID_SHIRI_PROFILED_MESH_SOURCES})
simplefluid_exclude_from_pgo_sources(
    ${SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES})
