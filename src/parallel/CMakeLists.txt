# Mesh partitioning shares the geometry archive and its runtime dependencies.
set(SIMPLEFLUID_PARALLEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/explicit_instantiation/MeshPartitioner.cc"
    PARENT_SCOPE)
