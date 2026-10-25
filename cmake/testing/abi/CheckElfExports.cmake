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
set(simplefluid_kokkos_anchor_symbol_pattern
    "^(_ZN6Kokkos10initializeERKNS_22InitializationSettingsE|_ZN6Kokkos8finalizeEv|_ZN6Kokkos14is_initializedEv|_ZN6Kokkos4Impl16ExecSpaceManager12get_instanceEv)$")
set(simplefluid_symbols_to_demangle)
set(simplefluid_shared_trilinos_rtti_symbols)
set(simplefluid_unexpected_symbols)
foreach(simplefluid_symbol IN LISTS simplefluid_dynamic_symbols)
    string(REGEX REPLACE "@.*$" ""
           simplefluid_unversioned_symbol "${simplefluid_symbol}")
    if(simplefluid_symbol MATCHES "@@SIMPLEFLUID_TRILINOS_RTTI_1[.]0$"
       AND NOT simplefluid_unversioned_symbol STREQUAL "SIMPLEFLUID_TRILINOS_RTTI_1.0")
        list(APPEND simplefluid_shared_trilinos_rtti_symbols "${simplefluid_unversioned_symbol}")
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
           "${simplefluid_kokkos_anchor_symbol_pattern}")
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
           "^SIMPLEFLUID_(1[.]0|KOKKOS_RUNTIME_1[.]0|TEUCHOS_COMM_RUNTIME_1[.]0|TRILINOS_RTTI_1[.]0|TPETRA_RTTI_1[.]0)$")
        list(APPEND simplefluid_unexpected_symbols "${simplefluid_symbol}")
    endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/CheckSharedTrilinosRtti.cmake")
simplefluid_check_shared_trilinos_rtti("${simplefluid_shared_trilinos_rtti_symbols}"
    "${SIMPLEFLUID_REQUIRE_SHARED_LIBCXX_RTTI_BRIDGES}")

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
    "SimpleFluid::PrescribedHeatSource<.*>[ ]*::require_finite[(]"
    # Removed source-specific APIs have no compatibility/export aliases.
    "^SimpleFluid::(FissionPowerSource|RadiolyticGasModel|RadiolyticBubbleKinetics|DelayedNeutronPrecursorModel)(<|::)"
    "^SimpleFluid::BubbleGasPhysics::(nucleation_radius|hydrogen_diffusivity|ideal_gas_alpha_source)[(]"
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::(synchronize_void_fraction|set_skip_zero_auxiliary_transport|enable_donor_hydrogen_deficit_tracking|dissolved_hydrogen|donor_hydrogen_deficit|bubble_slip_volume_flux|cumulative_hydrogen_produced|dissolved_hydrogen_inventory|global_dissolved_hydrogen_moles|global_submerged_hydrogen_moles|global_microbubble_hydrogen_moles|global_large_bubble_hydrogen_moles|cumulative_dissolved_hydrogen_outflow|global_submerged_bubble_hydrogen_moles|donor_hydrogen_deficit_tracking_enabled|cumulative_submerged_bubble_hydrogen_escaped)[(]"
    "^SimpleFluid::NOXNonlinearSolver::Impl::"
    "SimpleFluid::LiquidMassInventory<.*>[ ]*::speciesMassClosureTolerance[(]"
    "^SimpleFluid::Reacting::(LowMachMixtureSolver|LowMachPressureEquation|ReactingMixtureCoupler)::Implementation::"
    "^SimpleFluid::Reacting::ReactingMixtureCoupler::Trial::Data::"
    "^SimpleFluid::Reacting::LowMachMixtureSolver::(require_controller|release_controller|preview_impl|preview_planar_impl|capture_accepted_geometry|restore_accepted_after_motion)[(]"
    "^SimpleFluid::Reacting::LowMachPressureEquation::project_impl[(]"
    "^SimpleFluid::Reacting::source_stage_detail::"
    "^SimpleFluid::Reacting::history_detail::"
    "^SimpleFluid::Reacting::low_mach_faces::"
    "^SimpleFluid::Reacting::population_source_detail::"
    "^SimpleFluid::detail::surface_species::"
    "^SimpleFluid::ClosedMixtureHeadspaceModel::(prospective|recover)[(]"
    "^SimpleFluid::ClosedMixtureHeadspaceModel::InventoryCandidate::"
    "^SimpleFluid::PlanarFreeSurfaceModel::(install_collective_guards|local_stage|replicated_update|replicated_headspace|checked_callback_volume)[(]"
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::(species_compartment_index|publish_species_history|validate_species_pressure_metadata|validate_species_candidate_pressure|set_species_absolute_pressure_offset)[(]"
    "^SimpleFluid::Reacting::LowMachMixtureSolver::ControllerLease::ControllerLease[(]SimpleFluid::Reacting::LowMachMixtureSolver[*], (unsigned long|unsigned long long)[)]$"
    "SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::MultiphysicsSolverCoupler[(].*BoussinesqSolver<.*>&[)]$"
    "SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::PlanarALEStep::"
    "SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::BoussWallTrial::"
    "SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::(prepare_bouss_wall_heat|validate_bouss_wall_heat|restore_bouss_wall_heat)[(]"
    "SimpleFluid::BoilingSourceModel<.*>[ ]*::(validate_void_inputs|limit_to_available_void|add_bulk_boiling|add_wall_boiling)[(]"
    "SimpleFluid::ScalarSourceDecayTransport<.*>[ ]*::(validate_collective_configuration|validate_collective_advance_selection|validate_liquid_fraction_field|reconstruct_concentrations|transport|update_inventory_diagnostics)[(]"
    "SimpleFluid::PrescribedHeatSource<.*>[ ]*::(require_uniform_value|require_uniform_configuration|copy_scaled_field|integrate|apply_base_profile|apply_time_multiplier)[(]"
    "SimpleFluid::ScalarVoidFractionModel<.*>[ ]*::(diffuse|restore_for_ale_replay|restore_state)[(]"
    "SimpleFluid::ReactingGasModel<.*>[ ]*::(restore_for_ale_replay|restore_state|restore_for_initialization_rollback|allocate_species_generation|invalidate_transport_operator)[(]"
    "SimpleFluid::LiquidMassInventory<.*>[ ]*::(updateVolumeFromStoredDensity)[(]"
    "SimpleFluid::PlanarALEBoundary<.*>[ ]*::(validate|validate_collective_controls)[(]"
    "^SimpleFluid::detail::"
    "^SimpleFluid::Meshes::DecomposedSemiStructuredTopology::(view|source_entity|position_in_view|lookup|band_for_local)[(]"
    "^SimpleFluid::Meshes::DecomposedOrthogonalTopology::(subtract|insert|count|finalize|flatten|expand|owned_face_local_id|owned_source_face|lookup|select|add_wrapped|shape)[(]"
    "^SimpleFluid::Meshes::DecomposedSemiStructuredTopology::(View|Group)::"
    "SimpleFluid::FluidSolver<.*>[ ]*::(run_momentum_predictor|global_sum|write_step_progress|run_pressure_correction|solve_coupled_krylov)[(]"
    "SimpleFluid::FluidSolver<.*>[ ]*::FluidSolver[(].*bool[)]$"
    "SimpleFluid::BoussinesqSolver<.*>[ ]*::(temperature_equation|stored_material_properties|stored_turbulence_model|physical_transport_enabled|solution_writer|solve_flow|momentum_transport_flux|advance_temperature_transport|stored_temperature_sources|refresh_physical_models|refresh_material_feedback|initialize_reacting_gas_state|update_void_fraction_models|active_alpha_g_field|active_alpha_l_field|ensure_scalar_void_fraction_model)[(]"
    "SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::(momentum_transport_flux|temperature_transport_flux|temperature_transport_state|initialize_primary_state|refresh_geometry_dependent_models|append_material_feedback_fields|append_solution_fields|refresh_physical_models|refresh_material_feedback|initialize_reacting_gas_state|update_void_fraction_models|active_alpha_g_field|active_alpha_l_field|ensure_scalar_void_fraction_model|step_planar_ale|validate_step_coupling|validate_coupling_checkpoint|coupling_configuration_signature)[(]"
    "SimpleFluid::IncompressibleIsothermalSolver<.*>[ ]*::(isothermal_momentum_equation|isothermal_velocity_boundary_cache|isothermal_pressure_face_flux_workspace|isothermal_coupled_pressure_velocity_solver|stored_material_properties|stored_turbulence_model|solution_writer)[(]"
    "PhysicalModelTag"
    "SimpleFluid::PlanarALEMeshMotion<.*>[ ]*::(detect_family|current_axis_edges|geometry_edge_coordinates|geometry_motion_available|geometry_motion_owned|claim_geometry_motion|release_geometry_motion|replace_axis_edges|candidate_axis_edges|candidate_geometry_edges|capture_cell_volumes|capture_face_centroids|validate_collective_construction|validate_collective_trial|validate_collective_transaction|compute_trial_state|rollback_impl|reset_stationary_state)[(]"
    "^SimpleFluid::MeshHandle<.*>[ ]*::(require_mesh|checked_local|checked_global_ids|local_output_filename|add_geometry_cell_data|collect_vtu_points|write_vtu|legacy_vtu_topology|orthogonal_vtu_topology|semi_structured_vtu_topology|unstructured_vtu_topology|geometry_cell_lid|geometry_face_lid|visit_geometry_cell|visit_geometry_face|adjacent_cell|cell_local_id|face_local_id|initialize_orthogonal|initialize_semi_structured|initialize_contiguous_native_maps|initialize_unstructured|initialize_stk|initialize_serial|initialize_cells|initialize_faces|initialize_indexer|initialize_cell_faces|materialize_legacy_indexer|initialize_boundary_batches|make_map|create_maps|check_cell|check_face|visit_indexed_cell|visit_indexed_face|geometry_to_local_cell|geometry_to_local_face)[(]"
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
    "^SimpleFluid::BubbleGasPhysics::advance_microbubble_substep[(]"
    "^SimpleFluid::BubbleGasPhysics::advance_upper_bubble_population[(]"
    "^SimpleFluid::BoilingSourceModel<.*>[ ]*::update[(]"
    "^SimpleFluid::ScalarSourceDecayTransport<.*>[ ]*::advance[(]"
    "^SimpleFluid::ScalarVoidFractionModel<.*>[ ]*::update_explicit[(]"
    "^SimpleFluid::PrescribedHeatSource<.*>[ ]*::initialize_gaussian[(]"
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::advance[(]"
    "^SimpleFluid::LiquidMassInventory<.*>[ ]*::previewCellwiseAdvance[(]"
    "^SimpleFluid::VolumeContinuityModel<.*>[ ]*::preview[(]"
    "^SimpleFluid::PlanarALEBoundary<.*>[ ]*::apply_kinematic_velocity[(]"
    "^SimpleFluid::CoupledBlockOperator<.*>[ ]*::CoupledBlockOperator[(]"
    "^SimpleFluid::CoupledBlockOperator<.*>[ ]*::apply[(]"
    "^SimpleFluid::FluidSolver<.*>[ ]*::step[(][)]$"
    "^SimpleFluid::FluidSolver<.*>[ ]*::run[(]int[)]$"
    "^SimpleFluid::FluidSolver<.*>[ ]*::pressure[(][)]$"
    "^SimpleFluid::FluidSolver<.*>[ ]*::pressure_corrected_face_fluxes[(][)] const$"
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::step[(][)]$"
    "^SimpleFluid::BoussinesqSolver<.*>[ ]*::temperature[(][)]$"
    # libc++ names shared_ptr inside its versioned inline ABI namespace.
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::MultiphysicsSolverCoupler[(]std::(__[0-9]+::)?shared_ptr<SimpleFluid::BoussinesqSolver<.*>[ ]*>[)]$"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::step[(][)]$"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::configure_reacting_gas[(]"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::configure_free_surface[(]"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::create_coupling_checkpoint[(][)]$"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::restore_coupling_checkpoint[(]"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::accept_coupling_checkpoint[(]"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::validate_coupling_checkpoint_acceptance[(]"
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::set_thermal_step_participant[(]"
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
    "^SimpleFluid::VentedLumpedCavityModel::preview[(]"
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
# complete-object and base-object constructor entry points for concrete classes.
# Counts may list alternatives for compiler-dependent abstract/inline-only entries.
set(simplefluid_required_exact_api_patterns)
set(simplefluid_required_exact_api_counts)
macro(simplefluid_require_exact_api simplefluid_pattern simplefluid_count)
    list(APPEND simplefluid_required_exact_api_patterns
         "${simplefluid_pattern}")
    list(APPEND simplefluid_required_exact_api_counts
         "${simplefluid_count}")
endmacro()

# must contain no solver/controller/caloric exports from that path; reusable FVM
# geometry and transport kernels are audited separately in their owning library.

# The standalone reacting schema adds exactly 21 reviewed definitions across
# 18 signatures: SpeciesCatalog (7), ReactionMechanism (6), CellSpeciesState
# (6), SpeciesSourceLedger::thermal_energy (1), and advance_reactions (1).
# Constructors contribute both complete-object and base-object ABI entries.
# Match their complete argument lists, including libc++ inline namespaces.
set(simplefluid_reacting_schema_public_symbol_budget 21)
set(simplefluid_reviewed_reacting_api_patterns)
macro(simplefluid_require_reacting_api simplefluid_pattern simplefluid_count)
    list(APPEND simplefluid_reviewed_reacting_api_patterns "${simplefluid_pattern}")
    simplefluid_require_exact_api("${simplefluid_pattern}" "${simplefluid_count}")
endmacro()
set(simplefluid_reacting_catalog_vector_pattern
    "std::(__[0-9]+::)?vector<SimpleFluid::Reacting::SpeciesDefinition, std::(__[0-9]+::)?allocator<SimpleFluid::Reacting::SpeciesDefinition>[ ]*>")
set(simplefluid_reacting_reaction_vector_pattern
    "std::(__[0-9]+::)?vector<SimpleFluid::Reacting::ReactionDefinition, std::(__[0-9]+::)?allocator<SimpleFluid::Reacting::ReactionDefinition>[ ]*>")
set(simplefluid_reacting_catalog_pointer_pattern
    "std::(__[0-9]+::)?shared_ptr<SimpleFluid::Reacting::SpeciesCatalog const>")
set(simplefluid_reacting_string_view_pattern
    "std::(__[0-9]+::)?basic_string_view<char, std::(__[0-9]+::)?char_traits<char>[ ]*>")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::SpeciesCatalog::SpeciesCatalog[(]${simplefluid_reacting_catalog_vector_pattern}, (unsigned long|unsigned long long)[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::SpeciesCatalog::species[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::SpeciesCatalog::index[(]unsigned int[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::SpeciesCatalog::revision[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::SpeciesCatalog::fingerprint[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::SpeciesCatalog::allows[(]unsigned int, SimpleFluid::Reacting::SpeciesPhase[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactionMechanism::ReactionMechanism[(]${simplefluid_reacting_catalog_pointer_pattern}, ${simplefluid_reacting_reaction_vector_pattern}, SimpleFluid::Reacting::EnergyConvention[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactionMechanism::catalog[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactionMechanism::reactions[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactionMechanism::index[(]unsigned int[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactionMechanism::energy_convention[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::CellSpeciesState::CellSpeciesState[(]SimpleFluid::Reacting::SpeciesCatalog const&[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::CellSpeciesState::amount[(]SimpleFluid::Reacting::SpeciesCatalog const&, unsigned int, SimpleFluid::Reacting::SpeciesPhase, ${simplefluid_reacting_string_view_pattern}[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::CellSpeciesState::phase_amount[(]SimpleFluid::Reacting::SpeciesCatalog const&, unsigned int, SimpleFluid::Reacting::SpeciesPhase[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::CellSpeciesState::set_amount[(]SimpleFluid::Reacting::SpeciesCatalog const&, unsigned int, SimpleFluid::Reacting::SpeciesPhase, double, ${simplefluid_reacting_string_view_pattern}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::CellSpeciesState::validate[(]SimpleFluid::Reacting::SpeciesCatalog const&[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::SpeciesSourceLedger::thermal_energy[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::advance_reactions[(]SimpleFluid::Reacting::ReactionMechanism const&, SimpleFluid::Reacting::CellSpeciesState const&, SimpleFluid::Reacting::ReactionEnvironment const&, double, SimpleFluid::Reacting::ReactionSourceInput const&, SimpleFluid::Reacting::ReactionAdvanceOptions const&[)]$" 1)

# Composition-aware caloric providers replace the five scalar-liquid definitions.
# Seven class definitions (including the two constructor entries) and six pure
# factory/recovery/exchange functions are the entire new thermodynamic surface.
set(simplefluid_reacting_thermodynamic_public_symbol_budget 13)
set(simplefluid_reacting_const_double_span
    "std::(__[0-9]+::)?span<double const, [0-9]+u?l*>")
set(simplefluid_reacting_phase_span
    "std::(__[0-9]+::)?span<SimpleFluid::Reacting::PhaseThermalState const, [0-9]+u?l*>")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::MixtureThermodynamics::MixtureThermodynamics[(]${simplefluid_reacting_catalog_pointer_pattern}, SimpleFluid::Reacting::ThermodynamicBounds, SimpleFluid::Reacting::EOSKind, std::(__[0-9]+::)?function<SimpleFluid::Reacting::MixtureThermodynamicPoint [(]double, double, ${simplefluid_reacting_const_double_span}[)]>, SimpleFluid::Reacting::EnergyConvention, std::(__[0-9]+::)?function<.* [(]double, double, ${simplefluid_reacting_const_double_span}[)]>[)]$" 2)
foreach(simplefluid_thermodynamic_query IN ITEMS bounds catalog kind energy_convention)
    simplefluid_require_reacting_api(
        "^SimpleFluid::Reacting::MixtureThermodynamics::${simplefluid_thermodynamic_query}[(][)] const$" 1)
endforeach()
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::MixtureThermodynamics::evaluate[(]double, double, ${simplefluid_reacting_const_double_span}[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::make_ideal_gas_mixture[(]${simplefluid_reacting_catalog_pointer_pattern}, std::(__[0-9]+::)?vector<SimpleFluid::Reacting::IdealGasCaloricSpecies, .*>, SimpleFluid::Reacting::ThermodynamicBounds, SimpleFluid::Reacting::EnergyConvention, double, std::(__[0-9]+::)?function<.* [(]double, double, ${simplefluid_reacting_const_double_span}[)]>[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::make_linear_compressible_mixture_liquid[(]${simplefluid_reacting_catalog_pointer_pattern}, SimpleFluid::Reacting::ThermodynamicBounds, std::(__[0-9]+::)?function<SimpleFluid::Reacting::LinearCompressibleMixtureLiquid [(]${simplefluid_reacting_const_double_span}[)]>, SimpleFluid::Reacting::EnergyConvention[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::recover_phase_temperatures[(]${simplefluid_reacting_phase_span}, double, SimpleFluid::Reacting::PhaseThermalMode[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::solve_phase_pressure[(]${simplefluid_reacting_phase_span}, double, double, double, double, SimpleFluid::Reacting::PhaseThermalMode[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::solve_bubble_radius[(]SimpleFluid::Reacting::MixtureThermodynamics const&, ${simplefluid_reacting_const_double_span}, double, double, double, double, double, double[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::preview_interphase_exchange[(]${simplefluid_reacting_phase_span}, (unsigned long|unsigned long long), (unsigned long|unsigned long long), std::(__[0-9]+::)?function<SimpleFluid::Reacting::InterphaseExchange [(]SimpleFluid::Reacting::PhaseThermalState const&, SimpleFluid::Reacting::MixtureThermodynamicPoint const&, SimpleFluid::Reacting::PhaseThermalState const&, SimpleFluid::Reacting::MixtureThermodynamicPoint const&[)]> const&, SimpleFluid::Reacting::PhaseThermalMode[)]$" 1)

# Caloric feasibility adds five functions (including two constructor entries)
# and up to three RTTI/vtable definitions of its typed unresolved error.
# The original EOS constructor and every earlier thermodynamic anchor remain.
set(simplefluid_caloric_domain_public_symbol_budget 8)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::MixtureThermodynamics::MixtureThermodynamics[(]${simplefluid_reacting_catalog_pointer_pattern}, SimpleFluid::Reacting::ThermodynamicBounds, SimpleFluid::Reacting::EOSKind, SimpleFluid::Reacting::EnthalpyPressureDependence, std::(__[0-9]+::)?function<SimpleFluid::Reacting::MixtureThermodynamicPoint [(]double, double, ${simplefluid_reacting_const_double_span}[)]>, SimpleFluid::Reacting::EnergyConvention, std::(__[0-9]+::)?function<.* [(]double, double, ${simplefluid_reacting_const_double_span}[)]>[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::MixtureThermodynamics::enthalpy_pressure_dependence[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::exact_caloric_pressure_domain[(]SimpleFluid::Reacting::CaloricPressureQuery const&[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::evaluate_caloric_pressure_margins[(]SimpleFluid::Reacting::CaloricPressureQuery const&, double[)]$" 1)
foreach(simplefluid_caloric_rtti IN ITEMS "typeinfo for" "typeinfo name for" "vtable for")
    set(simplefluid_caloric_rtti_count 1)
    if(simplefluid_caloric_rtti STREQUAL "vtable for")
        # An inline-only exception has no key function; Clang can omit its
        # unused vtable in Release while retaining the cross-DSO catch RTTI.
        set(simplefluid_caloric_rtti_count "0|1")
    endif()
    simplefluid_require_exact_api(
        "^${simplefluid_caloric_rtti} SimpleFluid::Reacting::CaloricDomainUnresolved$" "${simplefluid_caloric_rtti_count}")
endforeach()

# The distributed face adapter stays private. Its pure finite-counter packet
# kernel adds exactly one public function; raw reaction receipts are aggregates.
set(simplefluid_species_diffusion_public_symbol_budget 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::preview_species_diffusion[(]SimpleFluid::Reacting::SpeciesCatalog const&, std::(__[0-9]+::)?span<SimpleFluid::Reacting::CellSpeciesState const, [0-9]+u?l*>, SimpleFluid::Reacting::SpeciesDiffusionBinding const&, std::(__[0-9]+::)?span<SimpleFluid::Reacting::CanonicalDiffusionFace const, [0-9]+u?l*>[)]$" 1)

# The explicit local-predecessor overload retains the original represented-state API.
set(simplefluid_raw_amount_array
    "std::(__[0-9]+::)?array<long double, 2u?l*>")
set(simplefluid_raw_amount_vector
    "std::(__[0-9]+::)?vector<${simplefluid_raw_amount_array}, std::(__[0-9]+::)?allocator<${simplefluid_raw_amount_array}[ ]*>[ ]*>")
set(simplefluid_raw_amount_span
    "std::(__[0-9]+::)?span<${simplefluid_raw_amount_vector} const, [0-9]+u?l*>")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::preview_species_diffusion[(]SimpleFluid::Reacting::SpeciesCatalog const&, std::(__[0-9]+::)?span<SimpleFluid::Reacting::CellSpeciesState const, [0-9]+u?l*>, SimpleFluid::Reacting::SpeciesDiffusionBinding const&, std::(__[0-9]+::)?span<SimpleFluid::Reacting::CanonicalDiffusionFace const, [0-9]+u?l*>, ${simplefluid_raw_amount_span}[)]$" 1)

# Sixteen public field-model operations for each of the two supported meshes,
# three inventory-follower operations for each mesh and one pure multispecies
# population kernel: exactly 39 additional definitions.
set(simplefluid_species_adapters_public_symbol_budget 39)
foreach(simplefluid_model_query IN ITEMS species_enabled source_ledgers population_ledgers escape_ledgers
        thermal_source material_volume_source liquid_mass_density pure_liquid_density)
    simplefluid_require_exact_api(
        "^SimpleFluid::ReactingGasModel<.*>[ ]*::${simplefluid_model_query}[(][)] const$" 2)
endforeach()
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::initialize_species[(]std::(__[0-9]+::)?span<SimpleFluid::Reacting::CellSpeciesState const, [0-9]+u?l*>, std::(__[0-9]+::)?span<std::(__[0-9]+::)?vector<double, .* const, [0-9]+u?l*>[)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::species_field[(]unsigned int, SimpleFluid::Reacting::SpeciesPhase, ${simplefluid_reacting_string_view_pattern}[)] const$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::cell_species_state[(]int[)] const$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::cell_species_state[(]int, double[)] const$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::population_number_field[(]${simplefluid_reacting_string_view_pattern}[)] const$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::global_species_moles[(]unsigned int[)] const$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::refresh_species_thermodynamics[(]double, .* const&, .* const&, .* const&, SimpleFluid::MaterialPropertyFields<.*>[ ]*const&[)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ReactingGasModel<.*>[ ]*::liquid_volume[(]int[)] const$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::LiquidMassInventory<.*>[ ]*::speciesBound[(][)] const$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::LiquidMassInventory<.*>[ ]*::initializeFromSpecies[(].* const&, .* const&[)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::LiquidMassInventory<.*>[ ]*::previewSpeciesMass[(].* const&, .* const&, double, double, SimpleFluid::FVM::ALEControlVolumeState const[*][)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::BubbleGasPhysics::advance_multispecies_population_exchange[(]SimpleFluid::Reacting::SpeciesCatalog const&, SimpleFluid::Reacting::CellSpeciesState const&, std::(__[0-9]+::)?vector<SimpleFluid::BubbleGasPhysics::PopulationNumberState, .* const&, SimpleFluid::Reacting::ReactionEnvironment const&, SimpleFluid::BubbleGasPhysics::MultispeciesPopulationPacket const&[)]$" 1)

# Stage 3 adds exactly 51 public definitions, without changing any existing
# signature: LowMachMixtureSolver (19) plus its move-only ControllerLease (5),
# LowMachPressureEquation (5), ReactingMixtureCoupler including Trial queries
# (14), and VariableDensityMomentumEquation (4 per native/legacy mesh).
# Complete/base constructors and nonvirtual destructors each have two entries.
# Inline trial/checkpoint storage and private implementation helpers are not API.
set(simplefluid_low_mach_public_symbol_budget 51)
set(simplefluid_stage3_std "std::(__[0-9]+::)?")
set(simplefluid_stage3_size "(unsigned long|unsigned long long)")
set(simplefluid_stage3_native_mesh
    "SimpleFluid::MeshHandle<SimpleFluid::TpetraTypes<.*>[ ]*>")
set(simplefluid_stage3_legacy_mesh
    "SimpleFluid::Mesh<SimpleFluid::TpetraTypes<.*>[ ]*>")
set(simplefluid_stage3_eos_pointer
    "${simplefluid_stage3_std}shared_ptr<SimpleFluid::Reacting::MixtureThermodynamics const>")
set(simplefluid_stage3_eos_vector
    "${simplefluid_stage3_std}vector<${simplefluid_stage3_eos_pointer}, ${simplefluid_stage3_std}allocator<${simplefluid_stage3_eos_pointer}[ ]*>[ ]*>")
set(simplefluid_stage3_sources_vector
    "${simplefluid_stage3_std}vector<SimpleFluid::Reacting::LowMachMixtureSolver::CellSources, ${simplefluid_stage3_std}allocator<SimpleFluid::Reacting::LowMachMixtureSolver::CellSources>[ ]*>")
set(simplefluid_stage3_controller_pointer
    "SimpleFluid::Reacting::LowMachMixtureSolver::ControllerLease const[*]")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::LowMachMixtureSolver[(]${simplefluid_stage3_std}shared_ptr<${simplefluid_stage3_native_mesh} const>, ${simplefluid_stage3_eos_vector}, SimpleFluid::Reacting::LowMachMixtureSolver::Options[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::~LowMachMixtureSolver[(][)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::ControllerLease::ControllerLease[(]SimpleFluid::Reacting::LowMachMixtureSolver::ControllerLease&&[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::ControllerLease::operator=[(]SimpleFluid::Reacting::LowMachMixtureSolver::ControllerLease&&[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::ControllerLease::~ControllerLease[(][)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::acquire_controller[(][)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::initialize[(]SimpleFluid::Reacting::LowMachMixtureSolver::State, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::preview[(]double, ${simplefluid_stage3_sources_vector} const&, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::commit[(]SimpleFluid::Reacting::LowMachMixtureSolver::Trial const&, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::advance[(]double, ${simplefluid_stage3_sources_vector} const&[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::restore[(]SimpleFluid::Reacting::LowMachMixtureSolver::Checkpoint const&, ${simplefluid_stage3_controller_pointer}[)]$" 1)
foreach(simplefluid_stage3_solver_query IN ITEMS
        initialized mesh options state diagnostics checkpoint volume_flux density velocity)
    simplefluid_require_reacting_api(
        "^SimpleFluid::Reacting::LowMachMixtureSolver::${simplefluid_stage3_solver_query}[(][)] const$" 1)
endforeach()

# Pressure correction uses native scalar-cell, scalar-face and vector-cell fields;
# retain the complete field order rather than accepting arbitrary project overloads.
set(simplefluid_stage3_scalar_cell
    "SimpleFluid::FieldStored<SimpleFluid::Field<double, SimpleFluid::CellLocation>, SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_native_mesh}[ ]*>")
set(simplefluid_stage3_scalar_face
    "SimpleFluid::FieldStored<SimpleFluid::Field<double, SimpleFluid::FaceLocation>, SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_native_mesh}[ ]*>")
set(simplefluid_stage3_vector_cell
    "SimpleFluid::FieldStored<SimpleFluid::Field<SimpleFluid::vec3<double>, SimpleFluid::CellLocation>, SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_native_mesh}[ ]*>")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachPressureEquation::LowMachPressureEquation[(]${simplefluid_stage3_std}shared_ptr<${simplefluid_stage3_native_mesh} const>[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachPressureEquation::~LowMachPressureEquation[(][)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachPressureEquation::project[(]${simplefluid_stage3_scalar_cell} const&, double, ${simplefluid_reacting_const_double_span}, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face}&, ${simplefluid_stage3_scalar_cell}&, ${simplefluid_stage3_vector_cell}&, SimpleFluid::LinearSolverOptions const&, double[)]$" 1)

simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactingMixtureCoupler::ReactingMixtureCoupler[(]${simplefluid_stage3_std}shared_ptr<SimpleFluid::Reacting::LowMachMixtureSolver>, SimpleFluid::Reacting::ReactingMixtureOptions[)]$" 2)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactingMixtureCoupler::~ReactingMixtureCoupler[(][)]$" 2)
foreach(simplefluid_stage3_coupler_operation IN ITEMS preview advance)
    simplefluid_require_reacting_api(
        "^SimpleFluid::Reacting::ReactingMixtureCoupler::${simplefluid_stage3_coupler_operation}[(]double[)]$" 1)
endforeach()
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactingMixtureCoupler::commit[(]SimpleFluid::Reacting::ReactingMixtureCoupler::Trial const&[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactingMixtureCoupler::restore[(]SimpleFluid::Reacting::ReactingMixtureCoupler::Checkpoint const&[)]$" 1)
foreach(simplefluid_stage3_coupler_query IN ITEMS checkpoint diagnostics solver)
    simplefluid_require_reacting_api(
        "^SimpleFluid::Reacting::ReactingMixtureCoupler::${simplefluid_stage3_coupler_query}[(][)] const$" 1)
endforeach()
foreach(simplefluid_stage3_trial_query IN ITEMS state diagnostics local_sources)
    simplefluid_require_reacting_api(
        "^SimpleFluid::Reacting::ReactingMixtureCoupler::Trial::${simplefluid_stage3_trial_query}[(][)] const$" 1)
endforeach()

# Review this template family independently of the reacting namespace: only the
# constructor, geometry refresh and conservative advance are public definitions.
set(simplefluid_reviewed_variable_density_api_patterns)
macro(simplefluid_require_variable_density_api simplefluid_pattern simplefluid_count)
    list(APPEND simplefluid_reviewed_variable_density_api_patterns "${simplefluid_pattern}")
    simplefluid_require_exact_api("${simplefluid_pattern}" "${simplefluid_count}")
endmacro()
set(simplefluid_stage3_force
    "${simplefluid_stage3_std}function<SimpleFluid::vec3<double> [(]int[)]>")
set(simplefluid_stage3_boundary
    "${simplefluid_stage3_std}function<SimpleFluid::BoundaryCondition [(]int, ${simplefluid_stage3_size}, ${simplefluid_stage3_size}[)]>")
set(simplefluid_stage3_boundary_value
    "${simplefluid_stage3_std}function<double [(]int, ${simplefluid_stage3_size}, ${simplefluid_stage3_size}[)]>")
foreach(simplefluid_stage3_mesh_kind IN ITEMS native legacy)
    if(simplefluid_stage3_mesh_kind STREQUAL "native")
        set(simplefluid_stage3_momentum_mesh "${simplefluid_stage3_native_mesh}")
        set(simplefluid_stage3_momentum_scalar "${simplefluid_stage3_scalar_cell}")
        set(simplefluid_stage3_momentum_vector "${simplefluid_stage3_vector_cell}")
        set(simplefluid_stage3_momentum_face "${simplefluid_stage3_scalar_face}")
    else()
        set(simplefluid_stage3_momentum_mesh "${simplefluid_stage3_legacy_mesh}")
        set(simplefluid_stage3_momentum_scalar "SimpleFluid::CellField<SimpleFluid::TpetraTypes<.*>[ ]*>")
        set(simplefluid_stage3_momentum_vector "SimpleFluid::VectorCellField<SimpleFluid::TpetraTypes<.*>[ ]*>")
        set(simplefluid_stage3_momentum_face "SimpleFluid::FaceField<SimpleFluid::TpetraTypes<.*>[ ]*>")
    endif()
    set(simplefluid_stage3_momentum_class
        "SimpleFluid::VariableDensityMomentumEquation<SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_momentum_mesh}[ ]*>")
    simplefluid_require_variable_density_api(
        "^${simplefluid_stage3_momentum_class}::VariableDensityMomentumEquation[(]${simplefluid_stage3_std}shared_ptr<${simplefluid_stage3_momentum_mesh} const>[)]$" 2)
    simplefluid_require_variable_density_api(
        "^${simplefluid_stage3_momentum_class}::refresh_geometry[(][)]$" 1)
    simplefluid_require_variable_density_api(
        "^${simplefluid_stage3_momentum_class}::advance[(]${simplefluid_stage3_momentum_vector} const&, ${simplefluid_stage3_momentum_scalar} const&, ${simplefluid_stage3_momentum_scalar} const&, ${simplefluid_stage3_momentum_face} const&, ${simplefluid_stage3_momentum_scalar} const&, double, ${simplefluid_stage3_momentum_vector}&, ${simplefluid_stage3_force}, ${simplefluid_stage3_boundary}, ${simplefluid_stage3_boundary_value}, SimpleFluid::LinearSolverOptions const&, SimpleFluid::FVM::NonOrthogonalTreatment, SimpleFluid::FVM::ALEControlVolumeState const[*][)]$" 1)
endforeach()

# Transport-first source evaluation and accepted species-history reporting add
# exactly twelve definitions: four concrete trial/provider/pressure methods and
# four queries for each native/legacy field model. No Stage 3 signatures were removed.
# Explicit dispersed-count ownership adds only its accepted event receipt query.
set(simplefluid_population_events_public_symbol_budget 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactingMixtureCoupler::Trial::local_population_events[(][)] const$" 1)

set(simplefluid_source_history_public_symbol_budget 12)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::preview_with_sources[(]double, ${simplefluid_stage3_std}shared_ptr<SimpleFluid::Reacting::LowMachMixtureSolver::SourceStageProvider const>, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::ReactingMixtureCoupler::Trial::transport_snapshot[(][)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachPressureEquation::project_with_eos[(]${simplefluid_stage3_scalar_cell} const&, double, ${simplefluid_reacting_const_double_span}, ${simplefluid_reacting_const_double_span}, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face}&, ${simplefluid_stage3_scalar_cell}&, ${simplefluid_stage3_vector_cell}&, SimpleFluid::LinearSolverOptions const&, double[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachPressureEquation::project_material_with_eos[(]${simplefluid_stage3_scalar_cell} const&, double, ${simplefluid_reacting_const_double_span}, ${simplefluid_reacting_const_double_span}, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face}&, ${simplefluid_stage3_scalar_cell}&, ${simplefluid_stage3_vector_cell}&, SimpleFluid::LinearSolverOptions const&, double[)]$" 1)
foreach(simplefluid_stage4_mesh_kind IN ITEMS native legacy)
    set(simplefluid_stage4_model
        "SimpleFluid::ReactingGasModel<SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_${simplefluid_stage4_mesh_kind}_mesh}[ ]*>")
    foreach(simplefluid_stage4_query IN ITEMS species_histories species_step_events physical_escape_events)
        simplefluid_require_exact_api(
            "^${simplefluid_stage4_model}::${simplefluid_stage4_query}[(][)] const$" 1)
    endforeach()
    simplefluid_require_exact_api(
        "^${simplefluid_stage4_model}::evaluate_liquid_volume[(]double[)] const$" 1)
endforeach()

# Final schema-only field API: exactly one heat-independent advance and one
# accepted number-escape query per supported mesh. No old heat/void overload remains.
set(simplefluid_final_species_native_cell "${simplefluid_stage3_scalar_cell}")
set(simplefluid_final_species_native_face "${simplefluid_stage3_scalar_face}")
set(simplefluid_final_species_native_velocity "${simplefluid_stage3_vector_cell}")
set(simplefluid_final_species_legacy_cell
    "SimpleFluid::CellField<SimpleFluid::TpetraTypes<.*>[ ]*>")
set(simplefluid_final_species_legacy_face
    "SimpleFluid::FaceField<SimpleFluid::TpetraTypes<.*>[ ]*>")
set(simplefluid_final_species_legacy_velocity
    "SimpleFluid::VectorCellField<SimpleFluid::TpetraTypes<.*>[ ]*>")
foreach(simplefluid_final_mesh_kind IN ITEMS native legacy)
    set(simplefluid_final_model
        "SimpleFluid::ReactingGasModel<SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_${simplefluid_final_mesh_kind}_mesh}[ ]*>")
    set(simplefluid_final_cell "${simplefluid_final_species_${simplefluid_final_mesh_kind}_cell}")
    set(simplefluid_final_face "${simplefluid_final_species_${simplefluid_final_mesh_kind}_face}")
    set(simplefluid_final_velocity "${simplefluid_final_species_${simplefluid_final_mesh_kind}_velocity}")
    set(simplefluid_final_material
        "SimpleFluid::MaterialPropertyFields<SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_${simplefluid_final_mesh_kind}_mesh}[ ]*>")
    simplefluid_require_exact_api(
        "^${simplefluid_final_model}::advance[(]double, double, ${simplefluid_final_cell} const&, ${simplefluid_final_cell} const&, ${simplefluid_final_velocity} const&, ${simplefluid_final_face} const&, ${simplefluid_final_material} const&, SimpleFluid::FVM::ALEControlVolumeState const[*], SimpleFluid::Dimension[)]$" 1)
    simplefluid_require_exact_api(
        "^${simplefluid_final_model}::population_escape_receipt[(][)] const$" 1)
endforeach()
simplefluid_require_exact_api("^SimpleFluid::ReactingGasModel<.*>[ ]*::advance[(].*[)]$" 2)

# This preexisting facade has the same eleven definitions in NOX-OFF stubs and
# NOX-ON implementations. Configuration alone grants no additional API budget.
set(simplefluid_nox_vector
    "Tpetra::Vector<double, int, long long, Tpetra::KokkosCompat::KokkosDeviceWrapperNode<.*>[ ]*>")
simplefluid_require_exact_api(
    "^SimpleFluid::NOXNonlinearSolver::NOXNonlinearSolver[(]SimpleFluid::NonlinearCallbacks[)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::NOXNonlinearSolver::NOXNonlinearSolver[(]SimpleFluid::NOXNonlinearSolver&&[)]$" 2)
simplefluid_require_exact_api("^SimpleFluid::NOXNonlinearSolver::~NOXNonlinearSolver[(][)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::NOXNonlinearSolver::operator=[(]SimpleFluid::NOXNonlinearSolver&&[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::NOXNonlinearSolver::set_callbacks[(]SimpleFluid::NonlinearCallbacks[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::NOXNonlinearSolver::solve[(]${simplefluid_nox_vector}&, SimpleFluid::NonlinearSolverOptions const&, SimpleFluid::LinearSolverOptions const&[)]$" 1)
simplefluid_require_exact_api("^SimpleFluid::NOXNonlinearSolver::cache_statistics[(][)] const$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::solve_nox[(]SimpleFluid::NonlinearCallbacks const&, ${simplefluid_nox_vector}&, SimpleFluid::NonlinearSolverOptions const&, SimpleFluid::LinearSolverOptions const&[)]$" 1)

# Native prescribed closed planar ALE adds exactly six public entry points.
# Trial geometry certificates and replay helpers remain private implementation.
set(simplefluid_native_ale_public_symbol_budget 6)
set(simplefluid_native_ale_motion
    "SimpleFluid::PlanarALEMeshMotion<SimpleFluid::TpetraTypes<.*>[ ]*>")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::preview_ale[(]double, ${simplefluid_native_ale_motion}&, double, ${simplefluid_stage3_sources_vector} const&, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::preview_ale_with_sources[(]double, ${simplefluid_native_ale_motion}&, double, ${simplefluid_stage3_std}shared_ptr<SimpleFluid::Reacting::LowMachMixtureSolver::SourceStageProvider const>, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::commit_ale[(]SimpleFluid::Reacting::LowMachMixtureSolver::Trial const&, ${simplefluid_native_ale_motion}&, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::checkpoint[(]${simplefluid_native_ale_motion} const&[)] const$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachMixtureSolver::restore[(]SimpleFluid::Reacting::LowMachMixtureSolver::Checkpoint const&, ${simplefluid_native_ale_motion}&, ${simplefluid_stage3_controller_pointer}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::LowMachPressureEquation::project_material_ale_with_eos[(]${simplefluid_stage3_scalar_cell} const&, double, ${simplefluid_reacting_const_double_span}, ${simplefluid_reacting_const_double_span}, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face} const&, ${simplefluid_stage3_scalar_face}&, ${simplefluid_stage3_scalar_cell}&, ${simplefluid_stage3_vector_cell}&, SimpleFluid::FVM::ALEControlVolumeState const&, SimpleFluid::LinearSolverOptions const&, double[)]$" 1)

# Pressure-conditioned local thermal transfer: four pure kernels, two receipt
# queries and the three definitions of the typed caloric-range error.
set(simplefluid_conditioned_thermal_public_symbol_budget 10)
set(simplefluid_thermal_double_vector
    "${simplefluid_stage3_std}vector<double, ${simplefluid_stage3_std}allocator<double>[ ]*>")
set(simplefluid_thermal_long_vector
    "${simplefluid_stage3_std}vector<long double, ${simplefluid_stage3_std}allocator<long double>[ ]*>")
set(simplefluid_thermal_double_vectors
    "${simplefluid_stage3_std}vector<${simplefluid_thermal_double_vector}, ${simplefluid_stage3_std}allocator<${simplefluid_thermal_double_vector}[ ]*>[ ]*>")
set(simplefluid_thermal_long_vectors
    "${simplefluid_stage3_std}vector<${simplefluid_thermal_long_vector}, ${simplefluid_stage3_std}allocator<${simplefluid_thermal_long_vector}[ ]*>[ ]*>")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::validate_interphase_configuration[(]SimpleFluid::Reacting::InterphaseConfiguration const&, ${simplefluid_reacting_phase_span}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::compile_interphase_plan[(]${simplefluid_stage3_std}shared_ptr<SimpleFluid::Reacting::InterphaseConfiguration const>, ${simplefluid_reacting_phase_span}, ${simplefluid_thermal_double_vectors}, ${simplefluid_thermal_long_vectors}, ${simplefluid_thermal_long_vector}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::compile_interphase_plan[(]${simplefluid_stage3_std}shared_ptr<SimpleFluid::Reacting::InterphaseConfiguration const>, ${simplefluid_reacting_phase_span}, ${simplefluid_thermal_double_vectors}, ${simplefluid_thermal_long_vectors}, ${simplefluid_thermal_long_vector}, ${simplefluid_raw_amount_span}[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::evaluate_pressure_conditioned_thermal[(]SimpleFluid::Reacting::FrozenInterphasePlan const&, ${simplefluid_reacting_phase_span}, ${simplefluid_reacting_const_double_span}, ${simplefluid_reacting_const_double_span}, double, SimpleFluid::Reacting::PhaseThermalMode, SimpleFluid::Reacting::ThermalEvaluationControls[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::interphase_pressure_bounds[(]SimpleFluid::Reacting::FrozenInterphasePlan const&, ${simplefluid_reacting_phase_span}[)]$" 1)
foreach(simplefluid_thermal_query IN ITEMS interphase_receipt final_thermal_evaluations)
    simplefluid_require_reacting_api(
        "^SimpleFluid::Reacting::ReactingMixtureCoupler::Trial::${simplefluid_thermal_query}[(][)] const$" 1)
endforeach()
foreach(simplefluid_thermal_rtti IN ITEMS "typeinfo for" "typeinfo name for" "vtable for")
    set(simplefluid_thermal_rtti_count 1)
    if(simplefluid_thermal_rtti STREQUAL "vtable for")
        set(simplefluid_thermal_rtti_count "0|1")
    endif()
    simplefluid_require_exact_api(
        "^${simplefluid_thermal_rtti} SimpleFluid::Reacting::ThermalCaloricRangeError$" "${simplefluid_thermal_rtti_count}")
endforeach()

# Fixed-reference finite vapor: seven pure functions, one controller entry point,
# and one accepted-receipt query for each supported field mesh. No private wall
# certificate or orchestration helper is part of the dynamic API.
set(simplefluid_bouss_vapor_public_symbol_budget 10)
set(simplefluid_bouss_number_span
    "${simplefluid_stage3_std}span<SimpleFluid::BubbleGasPhysics::PopulationNumberState const, [0-9]+u?l*>")
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::bouss_configuration_signature[(]SimpleFluid::Reacting::BoussReferenceVolume const&, SimpleFluid::Reacting::BoussCoordinatedSources const&[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::validate_bouss_reference[(]SimpleFluid::Reacting::ReactionMechanism const&, SimpleFluid::Reacting::BoussReferenceVolume const&[)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::validate_bouss_sources[(]SimpleFluid::Reacting::ReactionMechanism const&, SimpleFluid::Reacting::BoussReferenceVolume const&, SimpleFluid::Reacting::BoussCoordinatedSources const&[)]$" 1)
foreach(simplefluid_bouss_volume IN ITEMS gas liquid)
    simplefluid_require_reacting_api(
        "^SimpleFluid::Reacting::bouss_${simplefluid_bouss_volume}_volume[(]SimpleFluid::Reacting::SpeciesCatalog const&, SimpleFluid::Reacting::BoussReferenceVolume const&, SimpleFluid::Reacting::CellSpeciesState const&[)]$" 1)
endforeach()
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::advance_bouss_sources[(]SimpleFluid::Reacting::ReactionMechanism const&, SimpleFluid::Reacting::BoussReferenceVolume const&, SimpleFluid::Reacting::BoussCoordinatedSources const&, SimpleFluid::Reacting::CellSpeciesState const&, ${simplefluid_bouss_number_span}, SimpleFluid::Reacting::ReactionEnvironment const&, double, double, SimpleFluid::Reacting::ReactionSourceInput const&, SimpleFluid::Reacting::ReactionAdvanceOptions const&, SimpleFluid::Reacting::BoussWallHeatBudget const[*][)]$" 1)
simplefluid_require_reacting_api(
    "^SimpleFluid::Reacting::advance_bouss_population_exchange[(]SimpleFluid::Reacting::SpeciesCatalog const&, SimpleFluid::Reacting::BoussReferenceVolume const&, SimpleFluid::Reacting::CellSpeciesState const&, ${simplefluid_bouss_number_span}, SimpleFluid::Reacting::ReactionEnvironment const&, double, SimpleFluid::BubbleGasPhysics::MultispeciesPopulationPacket const&[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::MultiphysicsSolverCoupler<SimpleFluid::TpetraTypes<.*>[ ]*>::configure_bouss_sources[(]${simplefluid_stage3_std}shared_ptr<SimpleFluid::Reacting::BoussCoordinatedSources const>[)]$" 1)
foreach(simplefluid_bouss_mesh_kind IN ITEMS native legacy)
    simplefluid_require_exact_api(
        "^SimpleFluid::ReactingGasModel<SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_${simplefluid_bouss_mesh_kind}_mesh}[ ]*>::bouss_receipts[(][)] const$" 1)
endforeach()

# Typed surface receipts (8), supplied headspace configuration (1), base metadata
# constructor/query (3), and closed-mixture headspace methods/RTTI (13): exactly 25.
set(simplefluid_species_surface_public_symbol_budget 25)
foreach(simplefluid_surface_mesh_kind IN ITEMS native legacy)
    set(simplefluid_surface_model
        "SimpleFluid::ReactingGasModel<SimpleFluid::TpetraTypes<.*>, ${simplefluid_stage3_${simplefluid_surface_mesh_kind}_mesh}[ ]*>")
    foreach(simplefluid_surface_query IN ITEMS species_event_receipt species_subset_source_changes
            liquid_transport_volume_receipt transported_liquid_material_volume_flux)
        simplefluid_require_exact_api(
            "^${simplefluid_surface_model}::${simplefluid_surface_query}[(][)] const$" 1)
    endforeach()
endforeach()
simplefluid_require_exact_api(
    "^SimpleFluid::MultiphysicsSolverCoupler<SimpleFluid::TpetraTypes<.*>[ ]*>::configure_free_surface[(]SimpleFluid::FreeSurfaceOptions const&, ${simplefluid_stage3_std}unique_ptr<SimpleFluid::HeadspaceModel, ${simplefluid_stage3_std}default_delete<SimpleFluid::HeadspaceModel>[ ]*>[)]$" 1)
# Keep both base-object constructors. Clang omits complete-object constructors
# for abstract classes; GCC can also export those otherwise unused C1 entries.
foreach(simplefluid_headspace_ctor IN ITEMS
        "_ZN11SimpleFluid14HeadspaceModelC2ENS_16HeadspaceOptionsE"
        "_ZN11SimpleFluid14HeadspaceModelC2ENS_16HeadspaceOptionsENS_17HeadspaceMetadataE")
    set(simplefluid_found_headspace_ctor FALSE)
    foreach(simplefluid_symbol IN LISTS simplefluid_dynamic_symbols)
        if(simplefluid_symbol MATCHES "^${simplefluid_headspace_ctor}(@@SIMPLEFLUID_[0-9.]+)?$")
            set(simplefluid_found_headspace_ctor TRUE)
        endif()
    endforeach()
    if(NOT simplefluid_found_headspace_ctor)
        message(FATAL_ERROR "Missing abstract base-object constructor: ${simplefluid_headspace_ctor}")
    endif()
endforeach()
simplefluid_require_exact_api(
    "^SimpleFluid::HeadspaceModel::HeadspaceModel[(]SimpleFluid::HeadspaceOptions[)]$" "1|2")
simplefluid_require_exact_api(
    "^SimpleFluid::HeadspaceModel::HeadspaceModel[(]SimpleFluid::HeadspaceOptions, SimpleFluid::HeadspaceMetadata[)]$" "1|2")
simplefluid_require_exact_api("^SimpleFluid::HeadspaceModel::metadata[(][)] const$" 1)
set(simplefluid_surface_string
    "std::(__[0-9]+::|__cxx11::)?basic_string<char, ${simplefluid_stage3_std}char_traits<char>, ${simplefluid_stage3_std}allocator<char>[ ]*>")
set(simplefluid_surface_moles
    "${simplefluid_stage3_std}map<${simplefluid_surface_string}, double, ${simplefluid_stage3_std}less<${simplefluid_surface_string}[ ]*>[ ]*, ${simplefluid_stage3_std}allocator<${simplefluid_stage3_std}pair<${simplefluid_surface_string} const, double>[ ]*>[ ]*>")
simplefluid_require_exact_api(
    "^SimpleFluid::ClosedMixtureHeadspaceModel::ClosedMixtureHeadspaceModel[(]SimpleFluid::HeadspaceOptions, ${simplefluid_stage3_eos_pointer}, ${simplefluid_surface_string}, (unsigned long|unsigned long long)[)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::ClosedMixtureHeadspaceModel::initialize[(]double, double[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::ClosedMixtureHeadspaceModel::commit[(]SimpleFluid::HeadspaceState const&, ${simplefluid_surface_moles} const&[)]$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::ClosedMixtureHeadspaceModel::trialState[(]double, double, ${simplefluid_surface_moles} const&[)] const$" 1)
foreach(simplefluid_surface_query IN ITEMS mode clone)
    simplefluid_require_exact_api("^SimpleFluid::ClosedMixtureHeadspaceModel::${simplefluid_surface_query}[(][)] const$" 1)
endforeach()
foreach(simplefluid_surface_query IN ITEMS ventedMoles headspaceMoles accumulatedEscapedMoles)
    simplefluid_require_exact_api(
        "^SimpleFluid::ClosedMixtureHeadspaceModel::${simplefluid_surface_query}([[]abi:cxx11[]])?[(][)] const$" 1)
endforeach()
foreach(simplefluid_surface_rtti IN ITEMS "typeinfo for" "typeinfo name for" "vtable for")
    simplefluid_require_exact_api("^${simplefluid_surface_rtti} SimpleFluid::ClosedMixtureHeadspaceModel$" 1)
endforeach()

# Isothermal NOX adds one public context constructor (two Itanium entries) and
# two virtual hooks. Even private overrides must remain exported: downstream
# derived-driver vtables refer to them. The custom-driver link test covers this.
simplefluid_require_exact_api(
    "^SimpleFluid::CoupledNonlinearProblem::CoupledNonlinearProblem[(].*SimpleFluid::CoupledNonlinearProblem::FrozenIsothermalInput const&, SimpleFluid::CoupledNonlinearWorkspace[*][)]$" 2)
simplefluid_require_exact_api(
    "^SimpleFluid::IncompressibleIsothermalSolver<.*>[ ]*::supports_coupled_nonlinear[(][)] const$" 1)
simplefluid_require_exact_api(
    "^SimpleFluid::IncompressibleIsothermalSolver<.*>[ ]*::make_coupled_nonlinear_problem[(][)]$" 1)

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
            IN ITEMS IncompressibleMomentumEquation PressureProjectionEquation TemperatureDiffusionEquation
                     ReactingGasModel ScalarVoidFractionModel)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_geometry_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::refresh_geometry[(][)]$"
            1)
    endforeach()
    # Shared-cache overloads must coexist with the no-argument entry points.
    # A default argument preserves call syntax, but replaces the exported symbol.
    foreach(simplefluid_geometry_equation
            IN ITEMS IncompressibleMomentumEquation TemperatureDiffusionEquation
                     ReactingGasModel ScalarVoidFractionModel)
        simplefluid_require_exact_api(
            "^SimpleFluid::${simplefluid_geometry_equation}<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::refresh_geometry[(].*shared_ptr<SimpleFluid::FVM::TransportGeometryCache<${simplefluid_exact_mesh_specialization_pattern}[ ]*::SharedGeometry const>[)]$"
            1)
    endforeach()
    simplefluid_require_exact_api(
        "^SimpleFluid::PressureProjectionEquation<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::project[(].*SimpleFluid::VolumeContinuityTarget<.*> const&[)]$"
        2)
    simplefluid_require_exact_api(
        "^SimpleFluid::CoupledPressureVelocitySolver<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::assemble[(].*SimpleFluid::VolumeContinuityTarget<.*> const&.*SimpleFluid::FVM::ALEControlVolumeState const[*][)] const$"
        3)
    # Keep the historical target overloads above independently anchored. The
    # explicit-purpose overloads add residual-only assembly without replacing
    # those exported entry points.
    simplefluid_require_exact_api(
        "^SimpleFluid::CoupledPressureVelocitySolver<.*${simplefluid_exact_mesh_specialization_pattern}[ ]*::assemble[(].*SimpleFluid::VolumeContinuityTarget<.*> const&.*SimpleFluid::CoupledAssemblyPurpose[)] const$"
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
    "^SimpleFluid::FluidSolver<.*>[ ]*::courant_transport_face_fluxes[(][)] const$" 1)
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
    "^SimpleFluid::MultiphysicsSolverCoupler<.*>[ ]*::mesh_relative_face_fluxes[(][)] const$" 1)

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
    if(NOT "${simplefluid_found_exact_api_count}" MATCHES
       "^(${simplefluid_required_exact_api_count})$")
        list(APPEND simplefluid_missing_exact_api_patterns
             "${simplefluid_required_exact_api_pattern} (expected ${simplefluid_required_exact_api_count}, found ${simplefluid_found_exact_api_count})")
    endif()
endforeach()

# Verify the runtime provider: shared Kokkos, or FVM's versioned bridge when
# Trilinos is static. The solver's own API and export checks remain separate.
set(simplefluid_runtime_demangled_symbols "${simplefluid_demangled_symbols}")
set(simplefluid_runtime_has_kokkos_bridge ${simplefluid_kokkos_bridge_symbol_count})
if(DEFINED SIMPLEFLUID_RUNTIME_LIBRARY)
    if(NOT EXISTS "${SIMPLEFLUID_RUNTIME_LIBRARY}")
        message(FATAL_ERROR "SIMPLEFLUID_RUNTIME_LIBRARY must exist")
    endif()
    execute_process(
        COMMAND "${SIMPLEFLUID_NM}" -D --defined-only -j
                "${SIMPLEFLUID_RUNTIME_LIBRARY}"
        RESULT_VARIABLE runtime_nm_result
        OUTPUT_VARIABLE simplefluid_runtime_symbols)
    execute_process(
        COMMAND "${SIMPLEFLUID_READELF}" --dynamic "${SIMPLEFLUID_LIBRARY}"
        RESULT_VARIABLE runtime_readelf_result
        OUTPUT_VARIABLE runtime_dependencies)
    get_filename_component(runtime_name "${SIMPLEFLUID_RUNTIME_LIBRARY}" NAME)
    if(DEFINED SIMPLEFLUID_RUNTIME_SONAME)
        set(runtime_name "${SIMPLEFLUID_RUNTIME_SONAME}")
    endif()
    string(FIND "${runtime_dependencies}" "[${runtime_name}]" runtime_dependency)
    if(NOT runtime_nm_result EQUAL 0 OR NOT runtime_readelf_result EQUAL 0
       OR runtime_dependency EQUAL -1)
        message(FATAL_ERROR "Cannot verify the solver's runtime dependency: ${runtime_name}")
    endif()
    set(simplefluid_runtime_demangled_symbols)
    set(simplefluid_runtime_symbols_to_demangle)
    set(simplefluid_runtime_has_kokkos_bridge FALSE)
    if(SIMPLEFLUID_TRILINOS_SHARED OR SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME
       OR simplefluid_runtime_symbols MATCHES "@@SIMPLEFLUID_KOKKOS_RUNTIME_1[.]0")
        set(simplefluid_runtime_has_kokkos_bridge TRUE)
    endif()
    string(REPLACE "\r\n" "\n" simplefluid_runtime_symbols "${simplefluid_runtime_symbols}")
    string(REPLACE "\n" ";" simplefluid_runtime_symbols "${simplefluid_runtime_symbols}")
    # The bundled backend exports the complete selected dependency closure.
    # Keep its potentially large symbol table out of the per-symbol CMake loop.
    string(REGEX REPLACE "\\$$" "" runtime_anchor_pattern "${simplefluid_kokkos_anchor_symbol_pattern}")
    list(FILTER simplefluid_runtime_symbols INCLUDE REGEX "${runtime_anchor_pattern}(@.*)?$")
    foreach(simplefluid_runtime_symbol IN LISTS simplefluid_runtime_symbols)
        if(SIMPLEFLUID_TRILINOS_SHARED OR SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME)
            string(REGEX REPLACE "@.*$" "" simplefluid_unversioned_symbol "${simplefluid_runtime_symbol}")
            if(simplefluid_unversioned_symbol MATCHES "${simplefluid_kokkos_anchor_symbol_pattern}")
                list(APPEND simplefluid_runtime_symbols_to_demangle "${simplefluid_unversioned_symbol}")
            endif()
        elseif(simplefluid_runtime_symbol MATCHES "^(.+)@@SIMPLEFLUID_KOKKOS_RUNTIME_1[.]0$")
            set(simplefluid_unversioned_symbol "${CMAKE_MATCH_1}")
            set(simplefluid_runtime_has_kokkos_bridge TRUE)
            if(simplefluid_unversioned_symbol MATCHES "${simplefluid_kokkos_anchor_symbol_pattern}")
                list(APPEND simplefluid_runtime_symbols_to_demangle "${simplefluid_unversioned_symbol}")
            endif()
        endif()
    endforeach()
    # Reuse the solver's anchor filter rather than demangling thousands of
    # unrelated Kokkos template exports from the runtime provider.
    if(simplefluid_runtime_symbols_to_demangle)
        execute_process(
            COMMAND "${SIMPLEFLUID_CXXFILT}" ${simplefluid_runtime_symbols_to_demangle}
            RESULT_VARIABLE runtime_cxxfilt_result
            OUTPUT_VARIABLE simplefluid_runtime_demangled_symbols
            ERROR_VARIABLE runtime_cxxfilt_error
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT runtime_cxxfilt_result EQUAL 0)
            message(FATAL_ERROR "Cannot demangle Kokkos runtime exports: ${runtime_cxxfilt_error}")
        endif()
        string(REPLACE "\n" ";" simplefluid_runtime_demangled_symbols
               "${simplefluid_runtime_demangled_symbols}")
    endif()
endif()
set(simplefluid_required_kokkos_patterns
    "^Kokkos::initialize[(]Kokkos::InitializationSettings const&[)]$"
    "^Kokkos::finalize[(][)]$"
    "^Kokkos::is_initialized[(][)]$"
    "^Kokkos::Impl::ExecSpaceManager::get_instance[(][)]$")
set(simplefluid_missing_kokkos_patterns)
if(simplefluid_runtime_has_kokkos_bridge)
    foreach(simplefluid_required_kokkos_pattern IN LISTS simplefluid_required_kokkos_patterns)
        set(simplefluid_found_required_kokkos FALSE)
        foreach(simplefluid_demangled_symbol IN LISTS simplefluid_runtime_demangled_symbols)
            if(simplefluid_demangled_symbol MATCHES "${simplefluid_required_kokkos_pattern}")
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
if(NOT SIMPLEFLUID_TRILINOS_SHARED AND NOT SIMPLEFLUID_EXTERNAL_TRILINOS_RUNTIME
   AND simplefluid_missing_local_definition_symbols)
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
        "${SIMPLEFLUID_LIBRARY} has an incomplete Kokkos runtime dependency:\n  "
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
# Reject any extra reacting definition, including implementation helpers;
# the namespace itself is not an export allowlist.
foreach(simplefluid_demangled_symbol IN LISTS simplefluid_demangled_symbols)
    if(simplefluid_demangled_symbol MATCHES "^SimpleFluid::Reacting::")
        set(simplefluid_reviewed_reacting_api FALSE)
        foreach(simplefluid_reacting_api_pattern IN LISTS simplefluid_reviewed_reacting_api_patterns)
            if(simplefluid_demangled_symbol MATCHES "${simplefluid_reacting_api_pattern}")
                set(simplefluid_reviewed_reacting_api TRUE)
                break()
            endif()
        endforeach()
        if(NOT simplefluid_reviewed_reacting_api)
            message(FATAL_ERROR
                "Unreviewed reacting dynamic definition: ${simplefluid_demangled_symbol}")
        endif()
    endif()

    if(simplefluid_demangled_symbol MATCHES "^(typeinfo( name)? for|vtable for) SimpleFluid::Reacting::"
       AND NOT simplefluid_demangled_symbol MATCHES "^(typeinfo( name)? for|vtable for) SimpleFluid::Reacting::(CaloricDomainUnresolved|ThermalCaloricRangeError)$")
        message(FATAL_ERROR
            "Unreviewed reacting RTTI definition: ${simplefluid_demangled_symbol}")
    endif()

    if(simplefluid_demangled_symbol MATCHES "^SimpleFluid::VariableDensityMomentumEquation<")
        set(simplefluid_reviewed_variable_density_api FALSE)
        foreach(simplefluid_variable_density_api_pattern
                IN LISTS simplefluid_reviewed_variable_density_api_patterns)
            if(simplefluid_demangled_symbol MATCHES "${simplefluid_variable_density_api_pattern}")
                set(simplefluid_reviewed_variable_density_api TRUE)
                break()
            endif()
        endforeach()
        if(NOT simplefluid_reviewed_variable_density_api)
            message(FATAL_ERROR
                "Unreviewed variable-density momentum dynamic definition: ${simplefluid_demangled_symbol}")
        endif()
    endif()
endforeach()

# The isothermal/adaptive, solid-conduction, and fixed-grid planar free-surface
# APIs extend the reviewed library shape without widening the export map to
# vendor or implementation-detail namespaces.
# Fixed reviewed reacting API budgets retain develop's ten-percent headroom.
# The preexisting substrate excludes the 142 definitions maintained on the
# separate VoF branch. Exact API anchors and private-family guards above remain
# authoritative; the ceiling never depends on the inspected library.
set(simplefluid_api_symbol_lower_bound 1250)
set(simplefluid_final_species_cleanup_symbol_delta -35)
set(simplefluid_preexisting_api_symbol_upper_bound 1408)
math(EXPR simplefluid_api_symbol_upper_bound
    "${simplefluid_preexisting_api_symbol_upper_bound} + ${simplefluid_reacting_schema_public_symbol_budget} + ${simplefluid_reacting_thermodynamic_public_symbol_budget} + ${simplefluid_species_adapters_public_symbol_budget} + ${simplefluid_low_mach_public_symbol_budget} + ${simplefluid_source_history_public_symbol_budget} + ${simplefluid_caloric_domain_public_symbol_budget} + ${simplefluid_species_diffusion_public_symbol_budget} + ${simplefluid_population_events_public_symbol_budget} + ${simplefluid_species_surface_public_symbol_budget} + ${simplefluid_native_ale_public_symbol_budget} + ${simplefluid_conditioned_thermal_public_symbol_budget} + ${simplefluid_bouss_vapor_public_symbol_budget} + ${simplefluid_final_species_cleanup_symbol_delta}")
math(EXPR simplefluid_api_symbol_reviewed_maximum
    "${simplefluid_api_symbol_upper_bound} - 3")
math(EXPR simplefluid_api_symbol_upper_bound
    "(${simplefluid_api_symbol_reviewed_maximum} * 110 + 99) / 100")
if(simplefluid_api_symbol_count LESS simplefluid_api_symbol_lower_bound
   OR simplefluid_api_symbol_count GREATER simplefluid_api_symbol_upper_bound)
    message(FATAL_ERROR
        "${SIMPLEFLUID_LIBRARY} exports ${simplefluid_api_symbol_count} "
        "SimpleFluid symbols; the reviewed public-API range is "
        "${simplefluid_api_symbol_lower_bound} to ${simplefluid_api_symbol_upper_bound}")
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
message(STATUS
    "ELF exports verified: ${simplefluid_api_symbol_count} SimpleFluid API "
    "symbols (reviewed range ${simplefluid_api_symbol_lower_bound} to "
    "${simplefluid_api_symbol_upper_bound})")
