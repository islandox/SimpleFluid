add_executable(verification_boundary_layer_mesh verification_boundary_layer_mesh.cc)
target_link_libraries(verification_boundary_layer_mesh PRIVATE SimpleFluid::Mesh)
simplefluid_exclude_from_pgo_sources(verification_boundary_layer_mesh.cc)

simplefluid_add_example(planar_free_surface_verification planar_free_surface_verification.cc)

# This analytic verification driver is not part of the production PGO training
# workload, so optimize it normally in profile-use builds.
simplefluid_exclude_from_pgo_sources(
    planar_free_surface_verification.cc)

simplefluid_add_example(planar_ale_verification planar_ale_verification.cc)
simplefluid_exclude_from_pgo_sources(
    planar_ale_verification.cc)

foreach(VERIFICATION_TARGET IN ITEMS dispersed_bubble_verification planar_ale_comparison)
  if(SIMPLEFLUID_ENABLE_IF97)
    add_executable(${VERIFICATION_TARGET} ${VERIFICATION_TARGET}.cc)
    target_link_libraries(${VERIFICATION_TARGET} PRIVATE SimpleFluid SimpleFluid::IF97)
    simplefluid_exe_reuse_precompiled_header(${VERIFICATION_TARGET})
    simplefluid_exclude_from_pgo_sources(${VERIFICATION_TARGET}.cc)
  else()
    # Select a separate source instead of changing compile definitions on a
    # target that reuses the shared executable PCH. Default builds retain a
    # useful launcher diagnostic without linking or discovering IF97.
    add_executable(${VERIFICATION_TARGET} if97_verification_unavailable.cc)
  endif()
  if(BUILD_TESTING AND SIMPLEFLUID_ENABLE_IF97)
    if(VERIFICATION_TARGET STREQUAL "dispersed_bubble_verification")
      set(VERIFICATION_SMALL_ARGS --mesh-file "${PROJECT_SOURCE_DIR}/verification/openfoam/dispersedBubbleFlow/mesh_scale1.dat"
        --parameters "${PROJECT_SOURCE_DIR}/verification/openfoam/dispersedBubbleFlow/reference_scale1.properties")
    else()
      set(VERIFICATION_SMALL_ARGS --mesh-file "${PROJECT_SOURCE_DIR}/verification/openfoam/planarALE/mesh_scale1.dat")
    endif()
    foreach(VERIFICATION_MODE IN ITEMS steady transient)
      add_test(
        NAME ${VERIFICATION_TARGET}_${VERIFICATION_MODE}
        COMMAND ${VERIFICATION_TARGET} --mode ${VERIFICATION_MODE}
          ${VERIFICATION_SMALL_ARGS}
          --output "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}/verification/${VERIFICATION_TARGET}/${VERIFICATION_MODE}")
      set_tests_properties(${VERIFICATION_TARGET}_${VERIFICATION_MODE} PROPERTIES
        LABELS "verification;conservation;small_fixture"
        TIMEOUT 120
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endforeach()
  endif()
endforeach()

if(SIMPLEFLUID_ENABLE_IF97)
  add_executable(bottom_heated_bubbly_convection bottom_heated_bubbly_convection.cc)
  target_link_libraries(bottom_heated_bubbly_convection PRIVATE SimpleFluid SimpleFluid::IF97)
  simplefluid_exe_reuse_precompiled_header(bottom_heated_bubbly_convection)
  simplefluid_exclude_from_pgo_sources(bottom_heated_bubbly_convection.cc)
  if(BUILD_TESTING)
    add_test(NAME bottom_convection_zero_source
      COMMAND bottom_heated_bubbly_convection --steps 10 --source-scale 0
        --mesh-file "${PROJECT_SOURCE_DIR}/verification/openfoam/bottomHeatedBubblyConvection/mesh_scale1.dat"
        --properties "${PROJECT_SOURCE_DIR}/verification/openfoam/bottomHeatedBubblyConvection/reference_scale1.properties"
        --output "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}/verification/bottom_convection_zero_source")
    set_tests_properties(bottom_convection_zero_source PROPERTIES
      LABELS "verification;conservation;small_fixture;turbulence" TIMEOUT 60
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    add_test(NAME bottom_convection_graded_transport
      COMMAND bottom_heated_bubbly_convection --steps 5
        --mesh-file "${PROJECT_SOURCE_DIR}/verification/openfoam/bottomHeatedBubblyConvection/mesh_scale1.dat"
        --properties "${PROJECT_SOURCE_DIR}/verification/openfoam/bottomHeatedBubblyConvection/reference_scale1.properties"
        --output "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}/verification/bottom_convection_graded_transport")
    set_tests_properties(bottom_convection_graded_transport PROPERTIES
      LABELS "verification;conservation;small_fixture;turbulence" TIMEOUT 60
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  endif()
else()
  add_executable(bottom_heated_bubbly_convection if97_verification_unavailable.cc)
endif()

simplefluid_add_example(natural_convection_box natural_convection_box.cc)

simplefluid_add_example(natural_convection_cylinder natural_convection_cylinder.cc)

simplefluid_add_example(natural_convection_sphere natural_convection_sphere.cc)

simplefluid_add_example(natural_convection_boundary_layer_box natural_convection_boundary_layer_box.cc)

simplefluid_add_example(natural_convection_shiri natural_convection_shiri.cc)

simplefluid_add_example(pitz_daily pitz_daily.cc)

if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME cavity_verification_launcher
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/verification/openfoam/cavityFlow/test_run_simplefluid.py")
    set_tests_properties(cavity_verification_launcher PROPERTIES
      LABELS "verification;integration" TIMEOUT 30)
    add_test(
      NAME pitz_daily_profile_comparator_fixture
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/verification/openfoam/pitzDaily/test_compare_profiles.py")
    set_tests_properties(pitz_daily_profile_comparator_fixture PROPERTIES
      LABELS "verification;integration"
      TIMEOUT 30)
    add_test(
      NAME openfoam_verification_comparator
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/verification/openfoam/test_compare_verification.py")
    set_tests_properties(openfoam_verification_comparator PROPERTIES
      LABELS "verification;integration"
      TIMEOUT 30)
    add_test(
      NAME verification_performance_outputs
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/verification/openfoam/test_run_performance.py")
    set_tests_properties(verification_performance_outputs PROPERTIES
      LABELS "verification;integration"
      TIMEOUT 30)
    add_test(
      NAME openfoam_water_field_figures
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/verification/openfoam/test_plot_water_fields.py")
    set_tests_properties(openfoam_water_field_figures PROPERTIES
      LABELS "verification;integration"
      TIMEOUT 30)
    add_test(NAME openfoam_sst_field_figures
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/verification/openfoam/test_plot_sst_fields.py")
    set_tests_properties(openfoam_sst_field_figures PROPERTIES LABELS "verification;turbulence" TIMEOUT 30)
    add_test(NAME openfoam_boundary_layer_mesh
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/verification/openfoam/test_structured_mesh.py")
    set_tests_properties(openfoam_boundary_layer_mesh PROPERTIES LABELS "verification;geometry" TIMEOUT 30)
  else()
    message(WARNING
      "Python 3 was not found; the OpenFOAM comparator fixture tests are disabled")
  endif()
endif()

simplefluid_add_example(fissile_solution_tank_demo fissile_solution_tank_demo.cc)

simplefluid_add_example(fissile_solution_tank_sst fissile_solution_tank_sst.cc)

simplefluid_add_example(constant_power_cylinder_vessel constant_power_cylinder_vessel.cc)

# Any production executable requested from a USE tree must have been present in
# the training run. The presets build Shiri only; building another application
# later therefore fails loudly until its workload is added to training.
set(SIMPLEFLUID_EXAMPLE_TARGETS
    planar_free_surface_verification
    planar_ale_verification
    natural_convection_box
    natural_convection_cylinder
    natural_convection_sphere
    natural_convection_boundary_layer_box
    natural_convection_shiri
    pitz_daily
    fissile_solution_tank_demo
    fissile_solution_tank_sst
    constant_power_cylinder_vessel)
foreach(EXAMPLE_TARGET IN LISTS SIMPLEFLUID_EXAMPLE_TARGETS)
  simplefluid_require_complete_pgo(${EXAMPLE_TARGET})
endforeach()

if(BUILD_TESTING)
  foreach(EXAMPLE_TARGET IN LISTS SIMPLEFLUID_EXAMPLE_TARGETS)
    add_test(NAME ${EXAMPLE_TARGET}_smoke COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_TARGET}_smoke PROPERTIES
        LABELS integration
        WORKING_DIRECTORY "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}")
  endforeach()

  set(SIMPLEFLUID_PLANAR_FREE_SURFACE_VERIFICATION_SUBCASES
      uniformHeating
      gasGeneration
      completeEscape
      closedHeadspace
      boilingMassLoss)
  foreach(SUBCASE IN LISTS SIMPLEFLUID_PLANAR_FREE_SURFACE_VERIFICATION_SUBCASES)
    add_test(
        NAME planar_free_surface_${SUBCASE}
        COMMAND planar_free_surface_verification ${SUBCASE})
    set_tests_properties(planar_free_surface_${SUBCASE} PROPERTIES
        LABELS verification
        TIMEOUT 10
        WORKING_DIRECTORY "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}")
  endforeach()

  set(SIMPLEFLUID_PLANAR_ALE_VERIFICATION_SUBCASES
      aleUniformHeating
      aleGasGeneration
      aleCompleteEscape
      aleFailureRollback)
  foreach(SUBCASE IN LISTS SIMPLEFLUID_PLANAR_ALE_VERIFICATION_SUBCASES)
    add_test(
        NAME planar_ale_${SUBCASE}
        COMMAND planar_ale_verification ${SUBCASE})
    set_tests_properties(planar_ale_${SUBCASE} PROPERTIES
        LABELS "verification;ale;conservation"
        TIMEOUT 30
        WORKING_DIRECTORY "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}")
  endforeach()

  set_tests_properties(natural_convection_shiri_smoke PROPERTIES
      ENVIRONMENT
          "SIMPLEFLUID_SHIRI_NR=4;SIMPLEFLUID_SHIRI_NTHETA=4;SIMPLEFLUID_SHIRI_NZ=6;SIMPLEFLUID_SHIRI_STEPS=1")

  set_tests_properties(fissile_solution_tank_sst_smoke PROPERTIES
      ENVIRONMENT
          "SIMPLEFLUID_TANK_RZ_SPACING=0.05;SIMPLEFLUID_TANK_CIRCUMFERENTIAL_SPACING=0.1;SIMPLEFLUID_TANK_BOUNDARY_LAYER_COUNT=1;SIMPLEFLUID_TANK_BOUNDARY_LAYER_FIRST_HEIGHT=0.01;SIMPLEFLUID_TANK_STEPS=1;SIMPLEFLUID_TANK_DT=0.001")

  set_tests_properties(pitz_daily_smoke PROPERTIES
      ENVIRONMENT
          "SIMPLEFLUID_PITZ_MESH_DIVISOR=40;SIMPLEFLUID_PITZ_STEPS=1;SIMPLEFLUID_PITZ_DT=1e-6")

  simplefluid_add_pitz_daily_steady_test(pitz_daily
      "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}/pitz_daily_steady_smoke")
endif()

add_executable(region_diffusion region_diffusion.cc)
target_link_libraries(region_diffusion PRIVATE SimpleFluid::Solvers)
simplefluid_exe_reuse_precompiled_header(region_diffusion)
simplefluid_exclude_from_pgo_sources(region_diffusion.cc)
if(BUILD_TESTING)
    add_test(NAME region_diffusion_smoke COMMAND region_diffusion "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}/region_diffusion_$<CONFIG>.vtu")
    set_tests_properties(region_diffusion_smoke PROPERTIES LABELS "multi_region;example" TIMEOUT 120)
endif()

if(BUILD_TESTING)
    foreach(_region_procs IN ITEMS 2 4)
        simplefluid_add_mpi_test(NAME region_diffusion_${_region_procs}procs TARGET region_diffusion
            PROCS ${_region_procs} ARGS "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}/region_diffusion_${_region_procs}_$<CONFIG>.vtu"
            LABELS "mpi;multi_region;example" TIMEOUT 120)
    endforeach()
endif()

# Moving this directory must not change default CTest working directories.
if(BUILD_TESTING)
    get_property(example_tests DIRECTORY PROPERTY TESTS)
    foreach(test_name IN LISTS example_tests)
        get_test_property(${test_name} WORKING_DIRECTORY working_directory)
        if(NOT working_directory)
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY "${SIMPLEFLUID_EXAMPLE_OUTPUT_DIR}")
        endif()
    endforeach()
endif()

add_executable(sst_sas_activation sst_sas_activation.cc)
target_link_libraries(sst_sas_activation PRIVATE SimpleFluid::Solvers)
simplefluid_exe_reuse_precompiled_header(sst_sas_activation)
if(BUILD_TESTING)
  add_test(NAME sst_sas_activation_small COMMAND sst_sas_activation)
  set_tests_properties(sst_sas_activation_small PROPERTIES
    LABELS "verification;turbulence;sas;small_fixture" TIMEOUT 60)
  simplefluid_add_mpi_test(NAME SSTSASActivation_2procs TARGET sst_sas_activation
    PROCS 2 LABELS "mpi;turbulence;sas;small_fixture" TIMEOUT 60)
endif()
