
# Unit tests for geometry

# Frozen legacy geometry oracle; never part of a production library.
add_library(SimpleFluidLegacyMeshTestFixture STATIC
    fixtures/LegacySTKFixture.cc
    fixtures/LegacyMesh.cc
    fixtures/LegacyMeshPartitioner.cc)
simplefluid_configure_component(SimpleFluidLegacyMeshTestFixture)
target_link_libraries(SimpleFluidLegacyMeshTestFixture PUBLIC SimpleFluid::Mesh)

add_test(NAME simplefluid_plic_backend_boundary
    COMMAND "${CMAKE_COMMAND}" "-DSIMPLEFLUID_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        -P "${PROJECT_SOURCE_DIR}/cmake/testing/CheckPLICBackendBoundary.cmake")
set_tests_properties(simplefluid_plic_backend_boundary PROPERTIES LABELS "geometry;architecture")

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testOrthogonalDiffusionProfile\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  if(TEST_NAME STREQUAL "testMeshTraverse")
    target_sources(${TEST_NAME} PRIVATE
        mesh_traverse/SelectionAndLifetimeCases.cc
        mesh_traverse/StructuredCases.cc
        mesh_traverse/MpiCases.cc
        mesh_traverse/LayeredCases.cc
        mesh_traverse/LayeredIncidenceCases.cc
        mesh_traverse/StructuredIncidenceCases.cc)
  endif()
  target_link_libraries(${TEST_NAME} SimpleFluid::Mesh GTest::gtest_main)
  simplefluid_configure_test_executable(${TEST_NAME})
  if((TEST_NAME STREQUAL "testOrdinaryPredicateAccounting" OR TEST_NAME STREQUAL "testGeometryProvenanceBorrow"
      OR TEST_NAME STREQUAL "testPointSnapshot"
      OR TEST_NAME STREQUAL "testGeometryRetentionCensus"
      OR TEST_NAME STREQUAL "testClosedContactIndex"
      OR TEST_NAME STREQUAL "testContactCoverIndex"
      OR TEST_NAME STREQUAL "testAdaptiveClosureRecords"
      OR TEST_NAME STREQUAL "testSpatialAdaptiveClosure"
      OR TEST_NAME STREQUAL "testCompleteSupportOverlay"
      OR TEST_NAME STREQUAL "testCompactIncidenceMetadata"
      OR TEST_NAME STREQUAL "testDistributedAdaptiveOwnerOverlay"
      OR TEST_NAME STREQUAL "testIncrementalAdaptiveClosure"
      OR TEST_NAME STREQUAL "testIncrementalAdaptiveHierarchy"
      OR TEST_NAME STREQUAL "testAdaptiveIdentityPool"
      OR TEST_NAME STREQUAL "testContactPointArray"
      OR TEST_NAME STREQUAL "testPooledContactRows"
      OR TEST_NAME STREQUAL "testTileBodyPages"
      OR TEST_NAME STREQUAL "testContactPairSet"
      OR TEST_NAME STREQUAL "testContactRecordStorage"
      OR TEST_NAME STREQUAL "testDeferredOwnerCharges"
      OR TEST_NAME STREQUAL "testPooledLineExchange"
      OR TEST_NAME STREQUAL "testFrozenTopology"
      OR TEST_NAME STREQUAL "testOrdinaryPredicateBatch")
     AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${TEST_NAME} PRIVATE -fno-fast-math -ffp-contract=off -frounding-math -fno-lto)
    set_target_properties(${TEST_NAME} PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION OFF
      INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF
      INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF
      DISABLE_PRECOMPILE_HEADERS ON)
  endif()
  if(TEST_NAME STREQUAL "testMeshQualityAccess")
    simplefluid_discover_tests(${TEST_NAME}
        TEST_FILTER "QualityAccessTest.*"
        PROPERTIES TIMEOUT 30)
  elseif(TEST_NAME STREQUAL "testCartesianAMR")
    simplefluid_discover_tests(${TEST_NAME}
        PROPERTIES LABELS "vof\\;amr\\;geometry\\;conservation\\;collective")
  else()
    simplefluid_discover_tests(${TEST_NAME})
  endif()
endforeach()

simplefluid_add_mpi_test(
    NAME PolyhedralAMR_2procs
    TARGET testPolyhedralAMR
    PROCS 2
    LABELS "amr;polyhedral;collective"
    TIMEOUT 180)

simplefluid_add_mpi_test(
    NAME ExtrudedAMR_2procs
    TARGET testExtrudedAMR
    PROCS 2
    ARGS "--gtest_filter=ExtrudedAMRTest.*"
    LABELS "mpi;geometry;amr;multi_region"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME CartesianAMR_2procs
    TARGET testCartesianAMR
    PROCS 2
    ARGS "--gtest_filter=CartesianAMRTest.*:CartesianAMRMultiRankTest.*"
    LABELS "mpi;geometry;amr;multi_region"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME MeshTraverse_2procs
    TARGET testMeshTraverse
    PROCS 2
    ARGS "--gtest_filter=MeshTraverseMultiRankTest.*"
    LABELS "mpi;geometry;mesh_traverse"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME MeshQualityAccess_2procs
    TARGET testMeshQualityAccess
    PROCS 2
    ARGS "--gtest_filter=QualityAccessTest.*:QualityAccessMultiRankTest.*"
    LABELS "mpi;geometry;quality;collective"
    TIMEOUT 30)

simplefluid_add_mpi_test(
    NAME SweptRZRegionProviders_2procs
    TARGET testSweptRZRegionProviders
    PROCS 2
    ARGS "--gtest_filter=SweptRZRegionProvidersTest.CoincidentSelfSeamRetainsChecksAndCanonicalIncidence:SweptRZRegionProvidersTest.CompactOwnershipAndAffineMotionPreserveGCLAndRollback:SweptRZRegionProvidersTest.RankDivergentSweepDescriptorsFailCollectively"
    LABELS "mpi;multi_region;swept;ale"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME AnnularSemiStructuredMeshFactory_2procs
    TARGET testAnnularSemiStructuredMeshFactory
    PROCS 2
    ARGS "--gtest_filter=AnnularSemiStructuredMeshFactoryTest.CompositeWrapperPartitionsAndMovesWithoutMutatingChild:AnnularSemiStructuredMeshFactoryTest.CornerTransitionsPreserveConformingFacesAndAffineMotion"
    LABELS "mpi;mesh;semi_structured;ale"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME RegionExecution_2procs
    TARGET testRegionExecution
    PROCS 2
    ARGS "--gtest_filter=RegionExecutionMultiRankTest.*"
    LABELS "mpi;multi_region;execution;ale"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME WallYPlusStatistics_2procs
    TARGET testYPlusBoundaryLayerController
    PROCS 2
    ARGS "--gtest_filter=WallYPlusStatisticsTest.ReducesAcrossRanksWhenSomeRanksOwnNoWallFaces"
    LABELS "mpi;turbulence;mesh_adaptation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME SemiStructuredMeshHandle_2procs
    TARGET testMeshHandle
    PROCS 2
    ARGS "--gtest_filter=MeshHandleTest.SemiStructuredConstructionPreservesDefaultDistributedOwnership"
    LABELS "mpi;mesh_handle;semi_structured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PartitionedMeshHandleCommunicator_2procs
    TARGET testMeshHandle
    PROCS 2
    ARGS "--gtest_filter=MeshHandleTest.PreservesPartitionedMeshCommunicator"
    LABELS "mpi;mesh_handle;partitioned_unstructured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME MeshReorderingFactory_2procs
    TARGET testMeshReorderingFactory
    PROCS 2
    ARGS "--gtest_filter=MeshReorderingFactoryMultiRankTest.*"
    LABELS "mpi;mesh_reordering;solid_subdomain"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PlanarALEMeshMotion_2procs
    TARGET testPlanarALEMeshMotion
    PROCS 2
    ARGS "--gtest_filter=PlanarALEMeshMotionMultiRankTest.*"
    LABELS "mpi;mesh_motion;ale;gcl"
    TIMEOUT 120)

# The diffusion profiling test also needs the solvers library
add_executable(testOrthogonalDiffusionProfile testOrthogonalDiffusionProfile.cc)
target_link_libraries(testOrthogonalDiffusionProfile
      SimpleFluid::Mesh
      SimpleFluid::Solvers
      GTest::gtest_main)
simplefluid_configure_test_executable(testOrthogonalDiffusionProfile)
# This million-cell profiling executable is run manually with
# --gtest_also_run_disabled_tests and is intentionally not registered with CTest.

foreach(_region_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(
        NAME RegionMaterialization_${_region_procs}procs
        TARGET testRegionMaterialization
        PROCS ${_region_procs}
        LABELS "mpi;multi_region;compatibility"
        TIMEOUT 120)
    simplefluid_add_mpi_test(
        NAME MultiRegionMesh_${_region_procs}procs
        TARGET testMultiRegionMeshMultiRank
        PROCS ${_region_procs}
        LABELS "mpi;multi_region"
        TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(NAME MixedAMR_2procs TARGET testMixedAMR PROCS 2 LABELS "mpi;amr;mixed" TIMEOUT 180)

simplefluid_add_mpi_test(NAME LocalProductAMR_2procs TARGET testLocalProductAMR PROCS 2
    ARGS "--gtest_filter=LocalProductAMRTest.*" LABELS "mpi;amr;local_product" TIMEOUT 180)

simplefluid_add_mpi_test(
    NAME DistributedAdaptiveAuthority_2procs
    TARGET testDistributedAdaptiveAuthority
    PROCS 2)
simplefluid_add_mpi_test(
    NAME DistributedAdaptiveAuthority_4procs
    TARGET testDistributedAdaptiveAuthority
    PROCS 4)

simplefluid_add_mpi_test(
    NAME DistributedCartesianKernel_2procs
    TARGET testDistributedCartesianKernel
    PROCS 2)
simplefluid_add_mpi_test(
    NAME DistributedCartesianKernel_4procs
    TARGET testDistributedCartesianKernel
    PROCS 4)

simplefluid_add_mpi_test(NAME DistributedAdaptiveClosure_2procs TARGET testDistributedAdaptiveClosure PROCS 2)
simplefluid_add_mpi_test(NAME DistributedAdaptiveClosure_4procs TARGET testDistributedAdaptiveClosure PROCS 4)
simplefluid_add_mpi_test(NAME DistributedProductKernel_2procs TARGET testDistributedProductKernel PROCS 2)
simplefluid_add_mpi_test(NAME DistributedProductKernel_4procs TARGET testDistributedProductKernel PROCS 4)

simplefluid_add_mpi_test(NAME DistributedAdaptiveBalance_2procs TARGET testDistributedAdaptiveBalance PROCS 2)
simplefluid_add_mpi_test(NAME DistributedAdaptiveBalance_4procs TARGET testDistributedAdaptiveBalance PROCS 4)

simplefluid_add_mpi_test(NAME DistributedMixedKernel_2procs TARGET testDistributedMixedKernel PROCS 2 TIMEOUT 900)
simplefluid_add_mpi_test(NAME DistributedMixedKernel_4procs TARGET testDistributedMixedKernel PROCS 4 TIMEOUT 900)

foreach(_semi_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME SemiStructuredDecomposedTopology_${_semi_ranks}procs
        TARGET testDecomposedSemiStructuredTopology PROCS ${_semi_ranks}
        LABELS "mpi;geometry;semi_structured;topology;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME SemiStructuredDistributed_${_semi_ranks}procs
        TARGET testSemiStructuredDistributed PROCS ${_semi_ranks}
        LABELS "mpi;geometry;semi_structured;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME SemiStructuredALEGeometry_${_semi_ranks}procs
        TARGET testPlanarALEMeshMotion PROCS ${_semi_ranks}
        ARGS "--gtest_filter=PlanarALEMeshMotionTest.SemiStructuredAxialMotionPreservesXYTopologyAndSatisfiesGCL"
        LABELS "mpi;geometry;semi_structured;ale;empty_rank" TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(NAME SemiStructuredReordering_4procs
    TARGET testMeshReorderingFactory PROCS 4
    ARGS "--gtest_filter=MeshReorderingFactoryMultiRankTest.*SemiStructured*"
    LABELS "mpi;geometry;semi_structured;mesh_reordering;empty_rank" TIMEOUT 120)

foreach(_ortho_ranks IN ITEMS 2 4 8)
    # Run each fixed-rank width oracle where its required layout is available.
    if(_ortho_ranks EQUAL 2)
        set(_ortho_filter "-OrthogonalDistributedTest.AutomaticGridBalancesCountsBeforePhysicalAspect")
    else()
        set(_ortho_filter "-OrthogonalDistributedTest.PhysicalWidthsBreakBalancedCutLocationTies")
    endif()
    simplefluid_add_mpi_test(NAME OrthogonalDistributed_${_ortho_ranks}procs
        TARGET testOrthogonalDistributed PROCS ${_ortho_ranks}
        ARGS "--gtest_filter=${_ortho_filter}"
        LABELS "mpi;geometry;orthogonal;partition;empty_rank" TIMEOUT 120)
endforeach()

foreach(_ortho_reorder_ranks IN ITEMS 4 8)
    simplefluid_add_mpi_test(NAME OrthogonalReordering_${_ortho_reorder_ranks}procs
        TARGET testMeshReorderingFactory PROCS ${_ortho_reorder_ranks}
        ARGS "--gtest_filter=MeshReorderingFactoryMultiRankTest.*Orthogonal*"
        LABELS "mpi;geometry;orthogonal;mesh_reordering;empty_rank" TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(NAME NativeUnstructuredMapping_2procs
    TARGET testUnstructuredMeshMapping PROCS 2 LABELS "mpi;geometry;unstructured;mapping" TIMEOUT 120)

# Closed-contact spatial exchange, including idle owners and protocol rejection.
foreach(_spatial_closure_ranks IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME SpatialAdaptiveClosure_${_spatial_closure_ranks}procs
    TARGET testSpatialAdaptiveClosure PROCS ${_spatial_closure_ranks}
    ARGS "--gtest_filter=SpatialAdaptiveClosure.*-SpatialAdaptiveClosure.CubicN32*"
    LABELS "mpi;amr;closure;spatial;empty_rank" TIMEOUT 240)
endforeach()
simplefluid_add_mpi_test(NAME SpatialAdaptiveClosureN32_4procs
  TARGET testSpatialAdaptiveClosure PROCS 4
  ARGS "--gtest_filter=SpatialAdaptiveClosure.CubicN32FourSlabHaloUnder64MiB"
  LABELS "mpi;amr;closure;spatial;resource" TIMEOUT 600)
if(TEST SpatialAdaptiveClosureN32_4procs)
  set_tests_properties(SpatialAdaptiveClosureN32_4procs PROPERTIES
    ENVIRONMENT "SIMPLEFLUID_SPATIAL_CLOSURE_N32=1;OMP_NUM_THREADS=1")
endif()

simplefluid_add_mpi_test(NAME SpatialAdaptiveCanonicalN16_4procs
  TARGET testSpatialAdaptiveClosure PROCS 4
  ARGS "--gtest_filter=SpatialAdaptiveClosure.CubicN16CanonicalCacheStorageObservation"
  LABELS "mpi;amr;closure;spatial;resource" TIMEOUT 600)
if(TEST SpatialAdaptiveCanonicalN16_4procs)
  set_tests_properties(SpatialAdaptiveCanonicalN16_4procs PROPERTIES
    ENVIRONMENT "SIMPLEFLUID_SPATIAL_CLOSURE_N32=1;OMP_NUM_THREADS=1")
endif()

# Full canonical cache construction is a separate Debug storage probe. Its
# byte cap is unchanged; allow the setup walk to reach byte admission.
simplefluid_add_mpi_test(NAME SpatialAdaptiveCanonicalN32_4procs
  TARGET testSpatialAdaptiveClosure PROCS 4
  ARGS "--gtest_filter=SpatialAdaptiveClosure.CubicN32CanonicalCacheUnder64MiB"
  LABELS "mpi;amr;closure;spatial;resource" TIMEOUT 1800)
if(TEST SpatialAdaptiveCanonicalN32_4procs)
  set_tests_properties(SpatialAdaptiveCanonicalN32_4procs PROPERTIES
    ENVIRONMENT "SIMPLEFLUID_SPATIAL_CLOSURE_N32=1;OMP_NUM_THREADS=1")
endif()

# Sparse owner authority and complete support partition regressions. The
# support geometry fixture is rank-local; the owner fixture uses collectives.
foreach(_incremental_ranks IN ITEMS 2 4)
  simplefluid_add_mpi_test(NAME DistributedAdaptiveOwnerOverlay_${_incremental_ranks}procs
    TARGET testDistributedAdaptiveOwnerOverlay PROCS ${_incremental_ranks}
    ARGS "--gtest_filter=DistributedAdaptiveOwnerOverlay.*:DistributedAncestryRegistrationLease.*"
    LABELS "mpi;amr;owner_overlay;collective" TIMEOUT 180)
  simplefluid_add_mpi_test(NAME CompleteSupportOverlay_${_incremental_ranks}procs
    TARGET testCompleteSupportOverlay PROCS ${_incremental_ranks}
    ARGS "--gtest_filter=CompleteSupportOverlay.*"
    LABELS "mpi;amr;support;rank_local" TIMEOUT 180)
  simplefluid_add_mpi_test(NAME IncrementalAdaptiveClosure_${_incremental_ranks}procs
    TARGET testIncrementalAdaptiveClosure PROCS ${_incremental_ranks}
    ARGS "--gtest_filter=IncrementalAdaptiveClosure.*"
    LABELS "mpi;amr;closure;incremental;empty_rank" TIMEOUT 240)
  simplefluid_add_mpi_test(NAME PooledLineExchange_${_incremental_ranks}procs
    TARGET testPooledLineExchange PROCS ${_incremental_ranks}
    ARGS "--gtest_filter=PooledLineExchange.*"
    LABELS "mpi;amr;closure;incremental;protocol;empty_rank" TIMEOUT 180)
  simplefluid_add_mpi_test(NAME IncrementalAdaptiveHierarchy_${_incremental_ranks}procs
    TARGET testIncrementalAdaptiveHierarchy PROCS ${_incremental_ranks}
    ARGS "--gtest_filter=IncrementalAdaptiveHierarchy.*"
    LABELS "mpi;amr;closure;incremental;empty_rank" TIMEOUT 240)
endforeach()
