if(NOT DEFINED SIMPLEFLUID_LIBRARY
   OR NOT EXISTS "${SIMPLEFLUID_LIBRARY}")
    message(FATAL_ERROR
        "SIMPLEFLUID_LIBRARY must name an existing shared library")
endif()

if(NOT DEFINED SIMPLEFLUID_NM
   OR NOT EXISTS "${SIMPLEFLUID_NM}")
    message(FATAL_ERROR
        "SIMPLEFLUID_NM must name the configured symbol inspection tool")
endif()
if(NOT DEFINED SIMPLEFLUID_CXXFILT
   OR NOT EXISTS "${SIMPLEFLUID_CXXFILT}")
    message(FATAL_ERROR
        "SIMPLEFLUID_CXXFILT must name a configured C++ demangler")
endif()
if(NOT DEFINED SIMPLEFLUID_READELF
   OR NOT EXISTS "${SIMPLEFLUID_READELF}")
    message(FATAL_ERROR
        "SIMPLEFLUID_READELF must name the configured ELF inspection tool")
endif()
if(NOT DEFINED SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING
   OR NOT SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING must be a non-negative integer")
endif()
if(NOT DEFINED SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING
   OR NOT SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING must be a non-negative integer")
endif()
if(NOT DEFINED SIMPLEFLUID_REQUIRE_STATIC_LIBCXX_RTTI_BRIDGES)
    message(FATAL_ERROR
        "SIMPLEFLUID_REQUIRE_STATIC_LIBCXX_RTTI_BRIDGES must be defined")
endif()

execute_process(
    COMMAND "${SIMPLEFLUID_NM}" -D --defined-only -j
            "${SIMPLEFLUID_LIBRARY}"
    RESULT_VARIABLE simplefluid_nm_result
    OUTPUT_VARIABLE simplefluid_dynamic_symbols
    ERROR_VARIABLE simplefluid_nm_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT simplefluid_nm_result EQUAL 0)
    message(FATAL_ERROR
        "${SIMPLEFLUID_NM} failed for ${SIMPLEFLUID_LIBRARY}: "
        "${simplefluid_nm_error}")
endif()

execute_process(
    COMMAND "${SIMPLEFLUID_READELF}" --wide --syms
            "${SIMPLEFLUID_LIBRARY}"
    RESULT_VARIABLE simplefluid_readelf_result
    OUTPUT_VARIABLE simplefluid_elf_symbols
    ERROR_VARIABLE simplefluid_readelf_error)
if(NOT simplefluid_readelf_result EQUAL 0)
    message(FATAL_ERROR
        "${SIMPLEFLUID_READELF} failed for ${SIMPLEFLUID_LIBRARY}: "
        "${simplefluid_readelf_error}")
endif()
string(REPLACE "\r\n" "\n"
       simplefluid_elf_symbols "${simplefluid_elf_symbols}")
string(REPLACE "\n" ";"
       simplefluid_elf_symbols "${simplefluid_elf_symbols}")
set(simplefluid_hidden_undefined_symbols)
set(simplefluid_missing_local_definition_symbols
    "_ZTTN5MueLu14TpetraOperatorIdixN6Tpetra12KokkosCompat23KokkosDeviceWrapperNodeIN6Kokkos6SerialENS4_9HostSpaceEEEEE"
    "_ZTCN5MueLu14TpetraOperatorIdixN6Tpetra12KokkosCompat23KokkosDeviceWrapperNodeIN6Kokkos6SerialENS4_9HostSpaceEEEEE0_NS1_8OperatorIdixS7_EE"
    "_ZTCN5MueLu14TpetraOperatorIdixN6Tpetra12KokkosCompat23KokkosDeviceWrapperNodeIN6Kokkos6SerialENS4_9HostSpaceEEEEE0_N7Teuchos11DescribableE")
foreach(simplefluid_elf_symbol_line IN LISTS simplefluid_elf_symbols)
    if(simplefluid_elf_symbol_line MATCHES
       "[A-Z]+[ \t]+HIDDEN[ \t]+UND[ \t]+([^ \t]+)$")
        set(simplefluid_hidden_undefined_symbol "${CMAKE_MATCH_1}")
        if(NOT simplefluid_hidden_undefined_symbol MATCHES
           "^(__((start|stop)___llvm_|llvm_)(prf|cov))")
            list(APPEND simplefluid_hidden_undefined_symbols
                 "${simplefluid_hidden_undefined_symbol}")
        endif()
    endif()
    if(simplefluid_elf_symbol_line MATCHES
       "[ \t]+([^ \t]+)[ \t]+([^ \t]+)$")
        set(simplefluid_elf_symbol_index "${CMAKE_MATCH_1}")
        set(simplefluid_elf_symbol_name "${CMAKE_MATCH_2}")
        if(NOT simplefluid_elf_symbol_index STREQUAL "UND"
           AND simplefluid_elf_symbol_name IN_LIST
               simplefluid_missing_local_definition_symbols)
            list(REMOVE_ITEM simplefluid_missing_local_definition_symbols
                 "${simplefluid_elf_symbol_name}")
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES simplefluid_hidden_undefined_symbols)

string(REPLACE "\r\n" "\n"
       simplefluid_dynamic_symbols "${simplefluid_dynamic_symbols}")
string(REPLACE "\n" ";"
       simplefluid_dynamic_symbols "${simplefluid_dynamic_symbols}")

set(simplefluid_found_api FALSE)
set(simplefluid_api_symbol_count 0)
set(simplefluid_kokkos_bridge_symbol_count 0)
set(simplefluid_required_teuchos_comm_bridge_symbols
    "_ZTIN7Teuchos4CommIiEE"
    "_ZTSN7Teuchos4CommIiEE"
    "_ZTIN7Teuchos7MpiCommIiEE"
    "_ZTSN7Teuchos7MpiCommIiEE"
    "_ZTIN7Teuchos10SerialCommIiEE"
    "_ZTSN7Teuchos10SerialCommIiEE"
    "_ZTIN7Teuchos10CommStatusIiEE"
    "_ZTSN7Teuchos10CommStatusIiEE"
    "_ZTIN7Teuchos11CommRequestIiEE"
    "_ZTSN7Teuchos11CommRequestIiEE"
    "_ZTIN7Teuchos13MpiCommStatusIiEE"
    "_ZTSN7Teuchos13MpiCommStatusIiEE"
    "_ZTIN7Teuchos14MpiCommRequestIiEE"
    "_ZTSN7Teuchos14MpiCommRequestIiEE"
    "_ZTIN7Teuchos18MpiCommRequestBaseIiEE"
    "_ZTSN7Teuchos18MpiCommRequestBaseIiEE")
set(simplefluid_missing_teuchos_comm_bridge_symbols
    ${simplefluid_required_teuchos_comm_bridge_symbols})
set(simplefluid_tpetra_rtti_symbol_count 0)
set(simplefluid_required_tpetra_rtti_patterns
    "^_ZTIN6Tpetra13SrcDistObjectE$"
    "^_ZTSN6Tpetra13SrcDistObjectE$"
    "^_ZTIN6Tpetra10DistObjectIdix"
    "^_ZTSN6Tpetra10DistObjectIdix"
    "^_ZTIN6Tpetra11MultiVectorIdix"
    "^_ZTSN6Tpetra11MultiVectorIdix"
    "^_ZTIN6Tpetra6VectorIdix"
    "^_ZTSN6Tpetra6VectorIdix")
set(simplefluid_missing_tpetra_rtti_patterns
    ${simplefluid_required_tpetra_rtti_patterns})
set(simplefluid_symbols_to_demangle)
set(simplefluid_unexpected_symbols)
foreach(simplefluid_symbol IN LISTS simplefluid_dynamic_symbols)
    string(REGEX REPLACE "@.*$" ""
           simplefluid_unversioned_symbol "${simplefluid_symbol}")
    if(simplefluid_symbol MATCHES "^_ZGIW11SimpleFluid")
        if(NOT simplefluid_symbol MATCHES "@@SIMPLEFLUID_1[.]0$")
            list(APPEND simplefluid_unexpected_symbols
                 "${simplefluid_symbol} (wrong module initializer version)")
        endif()
    elseif(simplefluid_symbol MATCHES
       "^(_ZN11SimpleFluid|_ZNK11SimpleFluid|_ZNV11SimpleFluid|_ZNKV11SimpleFluid|_ZTIN11SimpleFluid|_ZTSN11SimpleFluid|_ZTVN11SimpleFluid|_ZTTN11SimpleFluid)")
        list(APPEND simplefluid_symbols_to_demangle
             "${simplefluid_unversioned_symbol}")
        set(simplefluid_found_api TRUE)
        math(EXPR simplefluid_api_symbol_count
             "${simplefluid_api_symbol_count} + 1")
        if(NOT simplefluid_symbol MATCHES
           "@@SIMPLEFLUID_1[.]0$")
            list(APPEND simplefluid_unexpected_symbols
                 "${simplefluid_symbol} (wrong API version)")
        endif()
    elseif(simplefluid_symbol MATCHES
           "^(_ZN6Kokkos|_ZNK6Kokkos|_ZNV6Kokkos|_ZNKV6Kokkos|_ZTIN6Kokkos|_ZTSN6Kokkos|_ZTVN6Kokkos|_ZTTN6Kokkos|Kokkos_|kokkosp_)")
        if(simplefluid_unversioned_symbol MATCHES
           "^(_ZN6Kokkos10initializeERKNS_22InitializationSettingsE|_ZN6Kokkos8finalizeEv|_ZN6Kokkos14is_initializedEv|_ZN6Kokkos4Impl16ExecSpaceManager12get_instanceEv)$")
            list(APPEND simplefluid_symbols_to_demangle
                 "${simplefluid_unversioned_symbol}")
        endif()
        math(EXPR simplefluid_kokkos_bridge_symbol_count
             "${simplefluid_kokkos_bridge_symbol_count} + 1")
        if(NOT simplefluid_symbol MATCHES
           "@@SIMPLEFLUID_KOKKOS_RUNTIME_1[.]0$")
            list(APPEND simplefluid_unexpected_symbols
                 "${simplefluid_symbol} (wrong Kokkos runtime version)")
        endif()
    elseif(simplefluid_unversioned_symbol IN_LIST
           simplefluid_required_teuchos_comm_bridge_symbols)
        list(REMOVE_ITEM simplefluid_missing_teuchos_comm_bridge_symbols
             "${simplefluid_unversioned_symbol}")
        if(NOT simplefluid_symbol MATCHES
           "@@SIMPLEFLUID_TEUCHOS_COMM_RUNTIME_1[.]0$")
            list(APPEND simplefluid_unexpected_symbols
                 "${simplefluid_symbol} (wrong Teuchos Comm runtime version)")
        endif()
    elseif(simplefluid_unversioned_symbol MATCHES
           "^_ZT[IS]N6Tpetra(13SrcDistObjectE$|10DistObject|11MultiVector|6Vector)")
        math(EXPR simplefluid_tpetra_rtti_symbol_count
             "${simplefluid_tpetra_rtti_symbol_count} + 1")
        foreach(simplefluid_required_tpetra_rtti_pattern
                IN LISTS simplefluid_missing_tpetra_rtti_patterns)
            if(simplefluid_unversioned_symbol MATCHES
               "${simplefluid_required_tpetra_rtti_pattern}")
                list(REMOVE_ITEM simplefluid_missing_tpetra_rtti_patterns
                     "${simplefluid_required_tpetra_rtti_pattern}")
            endif()
        endforeach()
        if(NOT simplefluid_symbol MATCHES
           "@@SIMPLEFLUID_TPETRA_RTTI_1[.]0$")
            list(APPEND simplefluid_unexpected_symbols
                 "${simplefluid_symbol} (wrong Tpetra RTTI version)")
        endif()
    elseif(NOT simplefluid_unversioned_symbol MATCHES
           "^SIMPLEFLUID_(1[.]0|KOKKOS_RUNTIME_1[.]0|TEUCHOS_COMM_RUNTIME_1[.]0|TPETRA_RTTI_1[.]0)$")
        list(APPEND simplefluid_unexpected_symbols "${simplefluid_symbol}")
    endif()
endforeach()

execute_process(
    COMMAND "${SIMPLEFLUID_CXXFILT}" ${simplefluid_symbols_to_demangle}
    RESULT_VARIABLE simplefluid_cxxfilt_result
    OUTPUT_VARIABLE simplefluid_demangled_symbols
    ERROR_VARIABLE simplefluid_cxxfilt_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT simplefluid_cxxfilt_result EQUAL 0)
    message(FATAL_ERROR
        "${SIMPLEFLUID_CXXFILT} failed: ${simplefluid_cxxfilt_error}")
endif()
string(REPLACE "\n" ";"
       simplefluid_demangled_symbols "${simplefluid_demangled_symbols}")

set(simplefluid_forbidden_api_patterns
    "^SimpleFluid::detail::"
    "SimpleFluid::FluidSolver<.*>[ ]*::(run_momentum_predictor|global_sum|write_step_progress|run_pressure_correction|solve_coupled_krylov)[(]"
    "SimpleFluid::FluidSolver<.*>[ ]*::FluidSolver[(].*bool[)]$"
    "SimpleFluid::BoussinesqSolver<.*>[ ]*::(temperature_equation|stored_material_properties|stored_turbulence_model|physical_transport_enabled|solution_writer|stored_temperature_sources|refresh_physical_models|refresh_material_feedback|initialize_radiolytic_gas_state|update_void_fraction_models|active_alpha_g_field|active_alpha_l_field|ensure_scalar_void_fraction_model)[(]"
    "PhysicalModelTag"
    "SimpleFluid::BoussinesqMomentumEquation<.*>[ ]*::select_dynamic_viscosity[(]"
    "SimpleFluid::IncompressibleMomentumEquation<.*>[ ]*::validate_transport_inputs[(]"
    "SimpleFluid::PressureProjectionEquation<.*>[ ]*::(require_owned_cell_map|project_impl)[(]"
    "SimpleFluid::TurbulenceModel<.*>[ ]*::State::"
    "SimpleFluid::TurbulenceModel<.*>[ ]*::(require_state|stage_effective_properties|stage_menter_eddy_viscosity|commit_effective_properties)[(]"
    "SimpleFluid::TurbulenceWallTreatment<.*>[ ]*::Evaluation::(Evaluation|check_owned_cell)[(]"
    "SimpleFluid::TurbulenceWallTreatment<.*>[ ]*::(initialize|validate_velocity_cache)[(]"
    "SimpleFluid::AdaptiveSteadyStateController::adapted_time_step[(]"
    "SimpleFluid::SteadyStateFieldMonitor<.*>[ ]*::(require_mesh|require_field_mesh|capture_current_state)[(]")
set(simplefluid_forbidden_api_symbols)
foreach(simplefluid_demangled_symbol IN LISTS simplefluid_demangled_symbols)
    foreach(simplefluid_forbidden_api_pattern
            IN LISTS simplefluid_forbidden_api_patterns)
        if(simplefluid_demangled_symbol MATCHES
           "${simplefluid_forbidden_api_pattern}")
            list(APPEND simplefluid_forbidden_api_symbols
                 "${simplefluid_demangled_symbol}")
            break()
        endif()
    endforeach()
endforeach()

set(simplefluid_required_api_patterns
    "^SimpleFluid::FluidSolver<.*>[ ]*::step[(][)]$"
    "^SimpleFluid::FluidSolver<.*>[ ]*::run[(]int[)]$"
    "^SimpleFluid::FluidSolver<.*>[ ]*::pressure[(][)]$"
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::step[(][)]$"
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::temperature[(][)]$"
    "^SimpleFluid::TurbulenceModel<.*>[ ]*::advance[(]"
    "^SimpleFluid::PressureProjectionEquation<.*>[ ]*::project[(]"
    "^SimpleFluid::IncompressibleMomentumEquation<.*>[ ]*::advance_velocity[(]"
    "^SimpleFluid::CoupledPressureVelocitySolver<.*>[ ]*::solve[(]"
    "^SimpleFluid::AdaptiveSteadyStateController::observe[(]")
set(simplefluid_missing_api_patterns)
foreach(simplefluid_required_api_pattern
        IN LISTS simplefluid_required_api_patterns)
    set(simplefluid_found_required_api FALSE)
    foreach(simplefluid_demangled_symbol IN LISTS simplefluid_demangled_symbols)
        if(simplefluid_demangled_symbol MATCHES
           "${simplefluid_required_api_pattern}")
            set(simplefluid_found_required_api TRUE)
            break()
        endif()
    endforeach()
    if(NOT simplefluid_found_required_api)
        list(APPEND simplefluid_missing_api_patterns
             "${simplefluid_required_api_pattern}")
    endif()
endforeach()

set(simplefluid_required_kokkos_patterns
    "^Kokkos::initialize[(]Kokkos::InitializationSettings const&[)]$"
    "^Kokkos::finalize[(][)]$"
    "^Kokkos::is_initialized[(][)]$"
    "^Kokkos::Impl::ExecSpaceManager::get_instance[(][)]$")
set(simplefluid_missing_kokkos_patterns)
if(simplefluid_kokkos_bridge_symbol_count GREATER 0)
    foreach(simplefluid_required_kokkos_pattern
            IN LISTS simplefluid_required_kokkos_patterns)
        set(simplefluid_found_required_kokkos FALSE)
        foreach(simplefluid_demangled_symbol
                IN LISTS simplefluid_demangled_symbols)
            if(simplefluid_demangled_symbol MATCHES
               "${simplefluid_required_kokkos_pattern}")
                set(simplefluid_found_required_kokkos TRUE)
                break()
            endif()
        endforeach()
        if(NOT simplefluid_found_required_kokkos)
            list(APPEND simplefluid_missing_kokkos_patterns
                 "${simplefluid_required_kokkos_pattern}")
        endif()
    endforeach()
endif()

if(NOT simplefluid_found_api)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} exports no versioned SimpleFluid API")
endif()
if(simplefluid_hidden_undefined_symbols)
    list(JOIN simplefluid_hidden_undefined_symbols "\n  "
         simplefluid_hidden_undefined_symbols)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} contains local hidden undefined symbols; "
        "the linker discarded required archive definitions:\n  "
        "${simplefluid_hidden_undefined_symbols}")
endif()
if(simplefluid_missing_local_definition_symbols)
    list(JOIN simplefluid_missing_local_definition_symbols "\n  "
         simplefluid_missing_local_definition_symbols)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} is missing required local MueLu COMDAT "
        "definitions:\n  ${simplefluid_missing_local_definition_symbols}")
endif()
if(simplefluid_missing_api_patterns)
    list(JOIN simplefluid_missing_api_patterns "\n  "
         simplefluid_missing_api_patterns)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} is missing required public API anchors:\n  "
        "${simplefluid_missing_api_patterns}")
endif()
if(simplefluid_missing_kokkos_patterns)
    list(JOIN simplefluid_missing_kokkos_patterns "\n  "
         simplefluid_missing_kokkos_patterns)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} has an incomplete Kokkos runtime bridge:\n  "
        "${simplefluid_missing_kokkos_patterns}")
endif()
if(SIMPLEFLUID_REQUIRE_STATIC_LIBCXX_RTTI_BRIDGES
   AND simplefluid_missing_teuchos_comm_bridge_symbols)
    list(JOIN simplefluid_missing_teuchos_comm_bridge_symbols "\n  "
         simplefluid_missing_teuchos_comm_bridge_symbols)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} has an incomplete Teuchos Comm RTTI bridge:\n  "
        "${simplefluid_missing_teuchos_comm_bridge_symbols}")
endif()
if(SIMPLEFLUID_REQUIRE_STATIC_LIBCXX_RTTI_BRIDGES
   AND simplefluid_missing_tpetra_rtti_patterns)
    list(JOIN simplefluid_missing_tpetra_rtti_patterns "\n  "
         simplefluid_missing_tpetra_rtti_patterns)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} has an incomplete Tpetra RTTI bridge:\n  "
        "${simplefluid_missing_tpetra_rtti_patterns}")
endif()
if(simplefluid_forbidden_api_symbols)
    list(LENGTH simplefluid_forbidden_api_symbols
         simplefluid_forbidden_api_symbol_count)
    list(SUBLIST simplefluid_forbidden_api_symbols 0 20
         simplefluid_forbidden_api_symbol_sample)
    list(JOIN simplefluid_forbidden_api_symbol_sample "\n  "
         simplefluid_forbidden_api_symbol_sample)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} exports "
        "${simplefluid_forbidden_api_symbol_count} library-internal "
        "SimpleFluid definitions:\n  "
        "${simplefluid_forbidden_api_symbol_sample}")
endif()
if(simplefluid_api_symbol_count LESS 300
   OR simplefluid_api_symbol_count GREATER 400)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} exports ${simplefluid_api_symbol_count} "
        "SimpleFluid symbols; the reviewed public-API range is 300 to 400")
endif()
if(simplefluid_kokkos_bridge_symbol_count GREATER
   SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} exports "
        "${simplefluid_kokkos_bridge_symbol_count} Kokkos runtime symbols; "
        "the validated static-Trilinos compatibility ceiling is "
        "${SIMPLEFLUID_KOKKOS_BRIDGE_SYMBOL_CEILING}")
endif()
if(simplefluid_tpetra_rtti_symbol_count GREATER
   SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} exports "
        "${simplefluid_tpetra_rtti_symbol_count} Tpetra RTTI symbols; "
        "the reviewed transfer-RTTI ceiling is "
        "${SIMPLEFLUID_TPETRA_RTTI_SYMBOL_CEILING}")
endif()
if(simplefluid_unexpected_symbols)
    list(LENGTH simplefluid_unexpected_symbols
         simplefluid_unexpected_symbol_count)
    list(SUBLIST simplefluid_unexpected_symbols 0 20
         simplefluid_unexpected_symbol_sample)
    list(JOIN simplefluid_unexpected_symbol_sample "\n  "
         simplefluid_unexpected_symbol_sample)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} has ${simplefluid_unexpected_symbol_count} "
        "unexpected dynamic definitions:\n  "
        "${simplefluid_unexpected_symbol_sample}")
endif()
