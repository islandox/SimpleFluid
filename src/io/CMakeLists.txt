# Mesh output is compiled into SimpleFluidMesh to preserve its existing ABI.
set(SIMPLEFLUID_IO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/VTUWriter.cc" PARENT_SCOPE)
simplefluid_debug_symbols_sources("IO;MESH" VTUWriter.cc)
