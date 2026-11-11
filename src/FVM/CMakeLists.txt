add_library(SimpleFluidFVM SHARED
    # Five assembly-only discretization units; iterative solves live in SimpleFluidSolvers.
    explicit_instantiation/ALEControlVolumeState.cc
    explicit_instantiation/CellGradientCache.cc
    explicit_instantiation/GaussLinearGradientCache.cc
    explicit_instantiation/NonOrthogonalCorrection.cc
    explicit_instantiation/TransportSystem.cc
)
add_library(SimpleFluid::FVM ALIAS SimpleFluidFVM)
simplefluid_configure_component(SimpleFluidFVM)
simplefluid_debug_symbols_target(SimpleFluidFVM FVM)
target_link_libraries(SimpleFluidFVM PUBLIC SimpleFluidFields)
