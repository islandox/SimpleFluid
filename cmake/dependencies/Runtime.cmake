include_guard(GLOBAL)

set(SIMPLEFLUID_EXTERNAL_LIBRARIES
    Trilinos::all_selected_libs
    Zoltan2Core::all_libs
    ${Trilinos_TPL_LIBRARIES}
    MPI::MPI_CXX
)
set(SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME ${Trilinos_BUILD_SHARED_LIBS})
if(SIMPLEFLUID_WRAP_STATIC_TRILINOS AND NOT Trilinos_BUILD_SHARED_LIBS)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "SIMPLEFLUID_WRAP_STATIC_TRILINOS currently requires Linux")
    endif()
    include("${PROJECT_SOURCE_DIR}/cmake/dependencies/StaticTrilinosBackend.cmake")
    simplefluid_add_static_trilinos_backend(${SIMPLEFLUID_EXTERNAL_LIBRARIES})
    set(SIMPLEFLUID_EXTERNAL_LIBRARIES SimpleFluid::TrilinosBackend)
    set(SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME ON)
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles([[#include <version>
#ifndef _LIBCPP_VERSION
#error "not libc++"
#endif
int main() { return 0; }
]] SIMPLEFLUID_CXX_USES_LIBCXX)

# Apple libc++ requires default visibility for the complete image.  ELF libc++
# keeps SimpleFluid hidden here and gives only the Trilinos header region in the
# shared PCH default visibility; the Linux version script remains the final DSO
# export boundary.
set(SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET hidden)
set(SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN YES)
if(APPLE)
    set(SIMPLEFLUID_SHARED_CXX_VISIBILITY_PRESET default)
    set(SIMPLEFLUID_SHARED_VISIBILITY_INLINES_HIDDEN NO)
endif()
