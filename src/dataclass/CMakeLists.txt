# Low-level data structures and the runtime database.
add_library(SimpleFluidCore STATIC Database.cc)
add_library(SimpleFluid::Core ALIAS SimpleFluidCore)
simplefluid_configure_component(SimpleFluidCore)
