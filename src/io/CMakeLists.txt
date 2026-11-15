# Mesh output is compiled into SimpleFluidMesh to preserve its existing ABI.
set(SIMPLEFLUID_IO_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/VTUWriter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/STKMeshInput.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/details/STKFaceAuthority.cc"
    )
simplefluid_debug_symbols_sources("IO;MESH" VTUWriter.cc STKMeshInput.cc details/STKFaceAuthority.cc)

# Shared/unwrapped Trilinos has no separate archive owner for the I/O bridge.
if(NOT TARGET SimpleFluidTrilinosBackend)
    set(SIMPLEFLUID_IO_SOURCES ${SIMPLEFLUID_IO_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/details/STKBackendBridge.cc")
endif()

set(SIMPLEFLUID_IO_SOURCES ${SIMPLEFLUID_IO_SOURCES} PARENT_SCOPE)
