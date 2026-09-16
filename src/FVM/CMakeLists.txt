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
target_link_libraries(SimpleFluidFVM PUBLIC SimpleFluidFields)
