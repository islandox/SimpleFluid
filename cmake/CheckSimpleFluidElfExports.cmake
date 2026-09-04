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
    if(simplefluid_symbol MATCHES
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
    "SimpleFluid::IncompressibleIsothermalSolver<.*>[ ]*::(isothermal_momentum_equation|isothermal_velocity_boundary_cache|isothermal_pressure_face_flux_workspace|isothermal_coupled_pressure_velocity_solver|stored_material_properties|stored_turbulence_model|solution_writer)[(]"
    "PhysicalModelTag"
    "SimpleFluid::PlanarALEMeshMotion<.*>[ ]*::(detect_family|current_axis_edges|geometry_edge_coordinates|geometry_motion_available|geometry_motion_owned|claim_geometry_motion|release_geometry_motion|replace_axis_edges|candidate_axis_edges|candidate_geometry_edges|capture_cell_volumes|capture_face_centroids|validate_collective_construction|validate_collective_trial|validate_collective_transaction|compute_trial_state|rollback_impl|reset_stationary_state)[(]"
    "SimpleFluid::MeshHandle<.*>[ ]*::(require_mesh|checked_local|checked_global_ids|local_output_filename|add_geometry_cell_data|collect_vtu_points|write_vtu|legacy_vtu_topology|orthogonal_vtu_topology|semi_structured_vtu_topology|unstructured_vtu_topology|geometry_cell_lid|geometry_face_lid|visit_geometry_cell|visit_geometry_face|adjacent_cell|cell_local_id|face_local_id|initialize_orthogonal|initialize_semi_structured|initialize_unstructured|initialize_stk|initialize_serial|initialize_cells|initialize_faces|initialize_indexer|initialize_cell_faces|materialize_legacy_indexer|initialize_boundary_batches|make_map|create_maps|check_cell|check_face|visit_indexed_cell|visit_indexed_face|geometry_to_local_cell|geometry_to_local_face)[(]"
    "SimpleFluid::VesselVolumeMap::boundAndEvaluate[(]"
    "SimpleFluid::(ConstantAreaVesselVolumeMap|TabulatedVesselVolumeMap)::(volumeBelowInRange|areaAtInRange|levelForVolumeInRange|heightSegment|volumeSegment)[(]"
    "SimpleFluid::PlanarFreeSurfaceModel::(solveClosure|solveVented|solveClosed|evaluateClosedTrial|makeDiagnostics|validateUpdate)[(]"
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
    "^SimpleFluid::IncompressibleIsothermalSolver<.*>[ ]*::step[(][)]$"
    "^SimpleFluid::TurbulenceModel<.*>[ ]*::advance[(]"
    "^SimpleFluid::PressureProjectionEquation<.*>[ ]*::project[(]"
    "^SimpleFluid::PressureProjectionEquation<.*SimpleFluid::Mesh<.*>[ ]*::project[(]"
    "^SimpleFluid::PressureProjectionEquation<.*SimpleFluid::MeshHandle<.*>[ ]*::project[(]"
    "^SimpleFluid::IncompressibleMomentumEquation<.*>[ ]*::advance_velocity[(]"
    "^SimpleFluid::IncompressibleMomentumEquation<.*SimpleFluid::Mesh<.*>[ ]*::advance_velocity[(]"
    "^SimpleFluid::IncompressibleMomentumEquation<.*SimpleFluid::MeshHandle<.*>[ ]*::advance_velocity[(]"
    "^SimpleFluid::CoupledPressureVelocitySolver<.*>[ ]*::solve[(]"
    "^SimpleFluid::CoupledPressureVelocitySolver<.*SimpleFluid::Mesh<.*>[ ]*::solve[(]"
    "^SimpleFluid::CoupledPressureVelocitySolver<.*SimpleFluid::MeshHandle<.*>[ ]*::solve[(]"
    "^SimpleFluid::TemperatureDiffusionEquation<.*SimpleFluid::MeshHandle<.*>[ ]*::advance_physical[(]"
    "^SimpleFluid::SolidHeatConductionEquation<.*SimpleFluid::SolidSubdomain<.*>[ ]*::advance[(]"
    "^SimpleFluid::BoussinesqMomentumEquation<.*SimpleFluid::MeshHandle<.*>[ ]*::advance_velocity_physical[(]"
    "^SimpleFluid::TurbulenceModel<.*SimpleFluid::MeshHandle<.*>[ ]*::advance[(]"
    "^SimpleFluid::AdaptiveSteadyStateController::observe[(]"
    "^SimpleFluid::AdaptiveSteadyStateController::observe[(].*, bool, bool[)]$"
    "^SimpleFluid::AdaptiveLinearToleranceController::observe[(]"
    "^SimpleFluid::AdaptiveLinearToleranceController::current_linear_tolerance[(]"
    "^SimpleFluid::AdaptiveLinearToleranceController::full_accuracy_requested[(]")
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

# Exact-count anchors distinguish overload families that a broad method-name
# match cannot. Constructor counts are two because the Itanium ABI emits both
# complete-object and base-object constructor entry points.
set(simplefluid_required_exact_api_patterns)
set(simplefluid_required_exact_api_counts)
macro(simplefluid_require_exact_api simplefluid_pattern simplefluid_count)
    list(APPEND simplefluid_required_exact_api_patterns
         "${simplefluid_pattern}")
    list(APPEND simplefluid_required_exact_api_counts
         "${simplefluid_count}")
endmacro()

# Preserve all seven established FluidSolver/BoussinesqSolver constructor
# signatures and the four mutable-MeshHandle additions.
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::FluidSolver[(].*shared_ptr<SimpleFluid::Mesh<.*> const>, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::FluidSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> const>, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::FluidSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> const>, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions, SimpleFluid::FluidSolver<.*>::DeferredMomentumEquationTag[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::BoussinesqSolver[(].*shared_ptr<SimpleFluid::Mesh<.*> const>, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::BoussinesqSolver[(].*shared_ptr<SimpleFluid::Mesh<.*> const>, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions, SimpleFluid::BoussinesqModelOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::BoussinesqSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> const>, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::BoussinesqSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> const>, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions, SimpleFluid::BoussinesqModelOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::FluidSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> >, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::FluidSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> >, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions, SimpleFluid::FluidSolver<.*>::DeferredMomentumEquationTag[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::BoussinesqSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> >, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::BoussinesqSolver[(].*shared_ptr<SimpleFluid::MeshHandle<.*> >, SimpleFluid::BoundaryConditionSet, SimpleFluid::TimeStepperOptions, SimpleFluid::LinearSolverOptions, SimpleFluid::BoussinesqModelOptions[)]$"
    2)

# ALE adds overloads; it does not replace either explicitly instantiated Mesh
# family. Anchor every pre-ALE endpoint and its ALEControlVolumeState peer.
set(simplefluid_exact_mesh_specialization_patterns
    "SimpleFluid::Mesh<.*>"
    "SimpleFluid::MeshHandle<.*>")
foreach(simplefluid_exact_mesh_specialization_pattern
        IN LISTS simplefluid_exact_mesh_specialization_patterns)
    if(simplefluid_exact_mesh_specialization_pattern MATCHES "MeshHandle")
        set(simplefluid_legacy_vector_pointer_pattern
            "SimpleFluid::MeshHandle<.*> > const[*]")
        set(simplefluid_legacy_boundary_pointer_pattern
            "SimpleFluid::MeshHandle<.*> > const[*]")
    else()
        set(simplefluid_legacy_vector_pointer_pattern
            "SimpleFluid::VectorCellField<.*> const[*]")
        set(simplefluid_legacy_boundary_pointer_pattern
            "SimpleFluid::BoundaryCache<.*> const[*]")
    endif()

    foreach(simplefluid_momentum_equation
            IN ITEMS IncompressibleMomentumEquation BoussinesqMomentumEquation)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::advance_velocity[(].*SimpleFluid::LinearSolverOptions const&[)] const$"
            2)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::advance_velocity[(].*SimpleFluid::LinearSolverOptions const&, SimpleFluid::FVM::ALEControlVolumeState const[*][)] const$"
            2)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::assemble_system[(].*, ${simplefluid_legacy_vector_pointer_pattern}[)] const$"
            2)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::assemble_system[(].*SimpleFluid::FVM::ALEControlVolumeState const[*][)] const$"
            2)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::advance_velocity_physical[(].*, ${simplefluid_legacy_boundary_pointer_pattern}[)] const$"
            1)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::advance_velocity_physical[(].*SimpleFluid::FVM::ALEControlVolumeState const[*][)] const$"
            1)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::assemble_physical_system[(].*, ${simplefluid_legacy_boundary_pointer_pattern}[)] const$"
            1)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_momentum_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::assemble_physical_system[(].*SimpleFluid::FVM::ALEControlVolumeState const[*][)] const$"
            1)
    endforeach()

    simplefluid_require_exact_api(
        "^SimpleFluid::TemperatureDiffusionEquation<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::advance_physical[(].*SimpleFluid::FVM::FaceCoefficientInterpolation[)] const$"
        1)
    simplefluid_require_exact_api(
        "^SimpleFluid::TemperatureDiffusionEquation<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::advance_physical[(].*SimpleFluid::FVM::FaceCoefficientInterpolation, SimpleFluid::FVM::ALEControlVolumeState const[*], .* const[*], .* const[*], .* const[*][)] const$"
        1)

    # Geometry refresh, continuity-target projection/assembly, and fixed-flux
    # control are public for both explicit mesh specializations.
    foreach(simplefluid_geometry_equation
            IN ITEMS IncompressibleMomentumEquation PressureProjectionEquation TemperatureDiffusionEquation)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_geometry_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::refresh_geometry[(][)]$"
            1)
    endforeach()
    simplefluid_require_exact_api(
        "^SimpleFluid::PressureProjectionEquation<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::project[(].*SimpleFluid::VolumeContinuityTarget<.*> const&[)]$"
        2)
    simplefluid_require_exact_api(
        "^SimpleFluid::CoupledPressureVelocitySolver<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::assemble[(].*SimpleFluid::VolumeContinuityTarget<.*> const&.*[)] const$"
        3)
    foreach(simplefluid_fixed_flux_owner
            IN ITEMS PressureProjectionEquation CoupledPressureVelocitySolver)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_fixed_flux_owner}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::set_fixed_boundary_flux_provider[(].*[)]$"
            1)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_fixed_flux_owner}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::clear_fixed_boundary_flux_provider[(][)]$"
            1)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_fixed_flux_owner}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::pressure_flux_boundary_cache[(].*[)] const$"
            1)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_fixed_flux_owner}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::apply_fixed_boundary_fluxes[(].*[)] const$"
            1)
    endforeach()
endforeach()

# The protected solver seam owns target installation and geometry invalidation.
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::refine_volume_continuity[(].*[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::refresh_geometry_dependent_state[(][)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::refresh_pressure_velocity_geometry_state[(][)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::set_volume_continuity_target[(].*[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::FluidSolver<.*>[ ]*::clear_volume_continuity_target[(][)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::refresh_geometry_dependent_state[(][)]$" 1)

# Anchor the out-of-line planar-motion lifecycle and solver observer. Inline
# getters remain consumer-instantiable and are not stable per-configuration
# dynamic-symbol anchors.
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarALEMeshMotion<.*>[ ]*::PlanarALEMeshMotion[(].*[)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarALEMeshMotion<.*>[ ]*::~PlanarALEMeshMotion[(][)]$" 3)
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarALEMeshMotion<.*>[ ]*::begin_trial[(].*[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarALEMeshMotion<.*>[ ]*::accept_trial[(][)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarALEMeshMotion<.*>[ ]*::rollback_trial[(][)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::mesh_relative_face_fluxes[(][)] const$" 1)

# The fixed-grid closure exposes an explicit preview/commit transaction and an
# accepted-state snapshot/restore transaction.
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarFreeSurfaceModel::previewUpdate[(]SimpleFluid::FreeSurfaceUpdate const&[)] const$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarFreeSurfaceModel::commitUpdate[(]SimpleFluid::PlanarFreeSurfaceModel::UpdatePreview const&[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarFreeSurfaceModel::snapshot[(][)] const$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::PlanarFreeSurfaceModel::restore[(]SimpleFluid::PlanarFreeSurfaceModel::StateSnapshot const&[)]$" 1)

# Intentional private test seam: testPhysicalEquations calls these two member
# overloads from a separate executable, so the explicit Mesh and MeshHandle
# instantiations must remain dynamically linkable. Keep exactly four exports;
# do not treat this as permission to export other private equation helpers.
simplefluid_require_exact_api(
    "^SimpleFluid::PressureProjectionEquation<.*>[ ]*::project_reusing_cached_predictor[(].*, SimpleFluid::(VectorCellField|FieldStored)<.*>[ ]*&[)]$"
    2)
simplefluid_require_exact_api(
    "^SimpleFluid::PressureProjectionEquation<.*>[ ]*::project_reusing_cached_predictor[(].*, SimpleFluid::VolumeContinuityTarget<.*> const&[)]$"
    2)

set(simplefluid_missing_exact_api_patterns)
list(LENGTH simplefluid_required_exact_api_patterns
     simplefluid_required_exact_api_pattern_count)
math(EXPR simplefluid_required_exact_api_last
     "${simplefluid_required_exact_api_pattern_count} - 1")
foreach(simplefluid_required_exact_api_index
        RANGE 0 ${simplefluid_required_exact_api_last})
    list(GET simplefluid_required_exact_api_patterns
         ${simplefluid_required_exact_api_index}
         simplefluid_required_exact_api_pattern)
    list(GET simplefluid_required_exact_api_counts
         ${simplefluid_required_exact_api_index}
         simplefluid_required_exact_api_count)
    set(simplefluid_found_exact_api_count 0)
    foreach(simplefluid_demangled_symbol IN LISTS simplefluid_demangled_symbols)
        if(simplefluid_demangled_symbol MATCHES
           "${simplefluid_required_exact_api_pattern}")
            math(EXPR simplefluid_found_exact_api_count
                 "${simplefluid_found_exact_api_count} + 1")
        endif()
    endforeach()
    if(NOT simplefluid_found_exact_api_count EQUAL
       simplefluid_required_exact_api_count)
        list(APPEND simplefluid_missing_exact_api_patterns
             "${simplefluid_required_exact_api_pattern} (expected ${simplefluid_required_exact_api_count}, found ${simplefluid_found_exact_api_count})")
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
if(simplefluid_missing_exact_api_patterns)
    list(JOIN simplefluid_missing_exact_api_patterns "\n  "
         simplefluid_missing_exact_api_patterns)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} has missing or duplicated exact API anchors:\n  "
        "${simplefluid_missing_exact_api_patterns}")
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
# The isothermal/adaptive, solid-conduction, and fixed-grid planar free-surface
# APIs extend the reviewed library shape without widening the export map to
# vendor or implementation-detail namespaces.
# The planar-ALE API adds explicit old/new-volume overloads while retaining
# the established protected solver subclass seam and exact pre-ALE entry
# points. A current GCC Debug build has 787 exported SimpleFluid symbols; 800
# leaves narrow growth headroom without claiming that an untested compiler or
# configuration has the same symbol count. Private-family exclusions and exact
# API anchors above remain authoritative.
if(simplefluid_api_symbol_count LESS 300
   OR simplefluid_api_symbol_count GREATER 800)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} exports ${simplefluid_api_symbol_count} "
        "SimpleFluid symbols; the reviewed public-API range is 300 to 800")
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
