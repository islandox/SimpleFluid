add_library(SimpleFluidFields INTERFACE)
add_library(SimpleFluid::Fields ALIAS SimpleFluidFields)
target_include_directories(SimpleFluidFields INTERFACE
    ${SIMPLEFLUID_PUBLIC_INCLUDE_DIRS}
)
target_link_libraries(SimpleFluidFields INTERFACE SimpleFluidMesh)
