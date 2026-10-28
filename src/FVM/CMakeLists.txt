add_library(SimpleFluidFVM SHARED
    # Non-orthogonal FVM solves and higher-level solvers share this instance.
    explicit_instantiation/DICPreconditioner.cc
    explicit_instantiation/ALEControlVolumeState.cc
    explicit_instantiation/CellGradientCache.cc
    explicit_instantiation/GaussLinearGradientCache.cc
    explicit_instantiation/NonOrthogonalCorrection.cc
    explicit_instantiation/TransportSystem.cc
)
add_library(SimpleFluid::FVM ALIAS SimpleFluidFVM)
simplefluid_configure_component(SimpleFluidFVM)
simplefluid_debug_symbols_target(SimpleFluidFVM FVM)
foreach(source IN LISTS SIMPLEFLUID_COMMON_VOF_FVM_SOURCES)
    if(source MATCHES "/PLIC/")
        simplefluid_debug_symbols_sources("PLIC;FVM" "${source}")
    elseif(source MATCHES "/swept_volume/")
        simplefluid_debug_symbols_sources("SWEPT_VOLUME;FVM" "${source}")
    endif()
endforeach()
target_link_libraries(SimpleFluidFVM PUBLIC SimpleFluidFields)
