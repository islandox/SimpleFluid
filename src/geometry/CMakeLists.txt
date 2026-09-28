set(SIMPLEFLUID_MESH_SOURCES
    AnnularSemiStructuredMeshFactory.cc
    BoundaryLayerMeshFactory.cc
    explicit_instantiation/Mesh.cc
    explicit_instantiation/MeshHandle.cc
    explicit_instantiation/PlanarALEMeshMotion.cc
    explicit_instantiation/MeshReorderingFactory.cc
    explicit_instantiation/SolidSubdomain.cc
    explicit_instantiation/STKMesh.cc
    MeshFactory.cc
    explicit_instantiation/LocalGlobalIndexer.cc
    explicit_instantiation/OrthogonalLocalGlobalIndexer.cc
    explicit_instantiation/PartitionedMeshBase.cc
    mesh/OrthoMeshPartitioner.cc
    mesh/OrthoMeshTopo.cc
    mesh/OrthogonalCartesian3D.cc
    mesh/OrthogonalCylindrial3D.cc
    mesh/FrontalDelaunay2D.cc
    mesh/PredicateSigns2D.cc
    mesh/SemiStructMeshTopo.cc
    mesh/SemiStructuredXY_Z.cc
    mesh/UnstructuredMesh.cc
    mesh/RegionProviders.cc
    mesh/ConservativeInterface.cc
    mesh/ExtrudedRegionProviders.cc
    mesh/SweptRZRegionProviders.cc
    mesh/MultiRegionMesh.cc
    mesh/MultiRegionMeshPartition.cc
)

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
endif()

# Require profiles for every Mesh archive member linked into Shiri. The listed
# alternate-path objects are compiled with ordinary LTO optimization in USE
# builds instead of silently applying profile-use without matching data.
set(SIMPLEFLUID_SHIRI_UNPROFILED_MESH_SOURCES
    AnnularSemiStructuredMeshFactory.cc
    BoundaryLayerMeshFactory.cc
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
