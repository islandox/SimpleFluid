if(NOT EXISTS "${SIMPLEFLUID_LIBRARY}")
    message(FATAL_ERROR "SIMPLEFLUID_LIBRARY must name the FVM shared library")
endif()
foreach(limit IN ITEMS SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING)
    if(NOT DEFINED ${limit} OR NOT "${${limit}}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "${limit} must be a non-negative integer")
    endif()
endforeach()

execute_process(
    COMMAND "${SIMPLEFLUID_NM}" -D --defined-only "${SIMPLEFLUID_LIBRARY}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE symbols
    ERROR_VARIABLE nm_error)
if(NOT nm_result EQUAL 0)
    message(FATAL_ERROR "Cannot inspect FVM exports: ${nm_error}")
endif()

# The existing version map allows only SimpleFluid API and the deliberately
# shared Kokkos/Teuchos/Tpetra runtime bridges out of embedded static archives.
set(api_symbols)
set(api_symbol_types)
set(shared_trilinos_rtti_symbols)
set(kokkos_count 0)
set(tpetra_count 0)
string(REPLACE "\n" ";" symbol_lines "${symbols}")
foreach(symbol IN LISTS symbol_lines)
    if(symbol STREQUAL "" OR symbol MATCHES " A SIMPLEFLUID_")
        continue()
    endif()
    if(NOT symbol MATCHES "@@SIMPLEFLUID_(1\\.0|KOKKOS_RUNTIME_1\\.0|TEUCHOS_COMM_RUNTIME_1\\.0|TRILINOS_RTTI_1\\.0|TPETRA_RTTI_1\\.0)$")
        message(FATAL_ERROR "Unexpected FVM export: ${symbol}")
    endif()
    if(symbol MATCHES "^[0-9A-Fa-f]+[ \t]+[^ \t]+[ \t]+(.+)@@SIMPLEFLUID_TRILINOS_RTTI_1[.]0$")
        list(APPEND shared_trilinos_rtti_symbols "${CMAKE_MATCH_1}")
    endif()
    if(symbol MATCHES "^[0-9A-Fa-f]+[ \t]+([^ \t]+)[ \t]+(.+)@@SIMPLEFLUID_1[.]0$")
        list(APPEND api_symbol_types "${CMAKE_MATCH_1}")
        list(APPEND api_symbols "${CMAKE_MATCH_2}")
    endif()
    if(symbol MATCHES "@@SIMPLEFLUID_KOKKOS_RUNTIME_1\\.0$")
        math(EXPR kokkos_count "${kokkos_count} + 1")
    elseif(symbol MATCHES "@@SIMPLEFLUID_TPETRA_RTTI_1\\.0$")
        math(EXPR tpetra_count "${tpetra_count} + 1")
    endif()
endforeach()
include("${CMAKE_CURRENT_LIST_DIR}/CheckSharedTrilinosRtti.cmake")
simplefluid_check_shared_trilinos_rtti("${shared_trilinos_rtti_symbols}"
    "${SIMPLEFLUID_REQUIRE_SHARED_LIBCXX_RTTI_BRIDGES}")
if(kokkos_count GREATER SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING
   OR tpetra_count GREATER SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING)
    message(FATAL_ERROR
        "FVM exceeds runtime export limits: ${kokkos_count} Kokkos, ${tpetra_count} Tpetra RTTI")
endif()

if(NOT api_symbols)
    message(FATAL_ERROR "FVM exports no versioned SimpleFluid API")
endif()
if(NOT EXISTS "${SIMPLEFLUID_CXXFILT}")
    message(FATAL_ERROR "SIMPLEFLUID_CXXFILT must name a C++ demangler")
endif()

# Demangle only public API symbols, after stripping their ELF version tags.
# Name-only output excludes return and argument types, so private types in a
# public signature cannot be mistaken for a private function owner.
foreach(mode IN ITEMS full names)
    set(demangle_options)
    if(mode STREQUAL "names")
        set(demangle_options --no-params)
    endif()
    execute_process(
        COMMAND "${SIMPLEFLUID_CXXFILT}" ${demangle_options} ${api_symbols}
        RESULT_VARIABLE demangle_result
        OUTPUT_VARIABLE demangled_${mode}
        ERROR_VARIABLE demangle_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT demangle_result EQUAL 0)
        message(FATAL_ERROR "Cannot demangle FVM exports: ${demangle_error}")
    endif()
    string(REPLACE "\r\n" "\n" demangled_${mode} "${demangled_${mode}}")
endforeach()

string(REPLACE "\n" ";" api_names "${demangled_names}")
list(LENGTH api_symbols symbol_count)
list(LENGTH api_names name_count)
if(NOT symbol_count EQUAL name_count)
    message(FATAL_ERROR "The demangler returned an incomplete FVM symbol list")
endif()
math(EXPR last_symbol "${symbol_count} - 1")
foreach(index RANGE ${last_symbol})
    list(GET api_names ${index} name)
    list(GET api_symbol_types ${index} symbol_type)
    if(name MATCHES "^_Z")
        message(FATAL_ERROR
            "Cannot decode FVM export with ${SIMPLEFLUID_CXXFILT}: ${name}; use a compatible llvm-cxxfilt")
    endif()
    # DIC's public interface is shared by FVM and header-defined Belos users.
    # Its private helpers remain subject to the detail-namespace prohibition.
    if(name MATCHES "^SimpleFluid::detail::DICPreconditioner<.*>[ ]*::(DICPreconditioner|~DICPreconditioner|apply|getDomainMap|getRangeMap)([(]|$)")
        continue()
    endif()
    if(symbol_type MATCHES "^[TW]$" AND name MATCHES "^SimpleFluid::(FVM::)?detail::")
        message(FATAL_ERROR "Private implementation exported: ${name}")
    endif()
endforeach()

# The lower-level FVM DSO must own the compiled DIC API; solvers consume it.
foreach(method IN ITEMS DICPreconditioner apply)
    if(NOT demangled_names MATCHES "(^|\n)SimpleFluid::detail::DICPreconditioner<[^\n]+>::${method}([(]|\n|$)")
        message(FATAL_ERROR "Missing compiled DIC ${method} in FVM")
    endif()
endforeach()

# These functions have declarations in public headers and definitions in the
# compiled FVM library. A hidden instantiation breaks downstream linking.
foreach(api IN ITEMS
        add_explicit_non_orthogonal_correction
        add_variable_explicit_non_orthogonal_correction
        add_explicit_deviatoric_transpose_gradient_stress
        explicit_non_orthogonal_diffusion_system
        implicit_non_orthogonal_diffusion_system
        fully_implicit_non_orthogonal_diffusion_system
        non_orthogonal_diffusion_system
        full_diffusion_residual
        solve_explicit_non_orthogonal_diffusion
        solve_non_orthogonal_diffusion
        transport_system
        non_orthogonal_transport_system
        weighted_scalar_transport_system
        physical_temperature_transport_system
        physical_momentum_transport_system)
    if(NOT demangled_names MATCHES "(^|\n)SimpleFluid::FVM::${api}<")
        message(FATAL_ERROR "Missing versioned FVM export: ${api}")
    endif()
endforeach()

foreach(mesh IN ITEMS Mesh MeshHandle)
    foreach(method IN ITEMS CellGradientCache refresh)
        if(NOT demangled_full MATCHES "(^|\n)SimpleFluid::FVM::CellGradientCache<[^\n]+SimpleFluid::${mesh}<[^\n]+::${method}[(]")
            message(FATAL_ERROR "Missing compiled cell gradient cache ${method} for ${mesh}")
        endif()
    endforeach()
    if(NOT demangled_names MATCHES "(^|\n)SimpleFluid::FVM::ALEControlVolumeState::validate<SimpleFluid::${mesh}<")
        message(FATAL_ERROR "Missing compiled ALE validation for ${mesh}")
    endif()
    foreach(method IN ITEMS GaussLinearGradientCache require_mesh refresh)
        if(NOT demangled_full MATCHES "(^|\n)SimpleFluid::FVM::GaussLinearGradientCache<[^\n]+SimpleFluid::${mesh}<[^\n]+::${method}[(]")
            message(FATAL_ERROR "Missing compiled Gauss gradient cache ${method} for ${mesh}")
        endif()
    endforeach()
    if(NOT demangled_full MATCHES "(^|\n)SimpleFluid::FVM::TransportGeometryCache<SimpleFluid::${mesh}<[^\n]+::refresh[(][)](\n|$)")
        message(FATAL_ERROR "Missing compiled geometry cache for ${mesh}")
    endif()
    if(NOT demangled_full MATCHES "(^|\n)SimpleFluid::FVM::TransportGeometryCache<SimpleFluid::${mesh}<[^\n]+::refresh[(][^\n]*shared_ptr<SimpleFluid::FVM::TransportGeometryCache<SimpleFluid::${mesh}<[^\n]+::SharedGeometry const>[)](\n|$)")
        message(FATAL_ERROR "Missing compiled shared geometry refresh for ${mesh}")
    endif()
endforeach()

message(STATUS "FVM shared-library exports verified")
