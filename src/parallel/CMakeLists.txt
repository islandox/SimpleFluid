# Mesh partitioning shares the geometry archive and its runtime dependencies.
set(SIMPLEFLUID_PARALLEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/explicit_instantiation/MeshPartitioner.cc"
    PARENT_SCOPE)
simplefluid_debug_symbols_sources("PARALLEL;MESH"
    explicit_instantiation/MeshPartitioner.cc)
