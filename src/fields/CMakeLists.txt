add_library(SimpleFluidFields STATIC
    explicit_instantiation/MeshToMeshTransfer.cc
)
add_library(SimpleFluid::Fields ALIAS SimpleFluidFields)
simplefluid_configure_component(SimpleFluidFields)
target_link_libraries(SimpleFluidFields PUBLIC SimpleFluidMesh)
simplefluid_exclude_from_pgo_sources(
    "${CMAKE_CURRENT_SOURCE_DIR}/explicit_instantiation/MeshToMeshTransfer.cc")
