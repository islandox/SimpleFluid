# Unit tests for solvers

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testReactingGasModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFluidSolverMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testFeedbackMapMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testPlanarFreeSurfaceModelMultiRank\\.cc$")
list(FILTER TEST_SOURCES EXCLUDE REGEX
     ".*/testDICPreconditionerMultiRank\\.cc$")
foreach(TEST_SRC IN LISTS TEST_SOURCES)
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  add_executable(${TEST_NAME} "${TEST_SRC}")
  target_link_libraries(${TEST_NAME} SimpleFluid::Solvers GTest::gtest_main)
  if(TEST_NAME STREQUAL "testLowMachFaceTransportAdapter")
    # Exercise the private adapter directly without exporting it from the DSO.
    target_sources(${TEST_NAME} PRIVATE ../details/LowMachFaceTransportAdapter.cc)
  endif()
  if(TEST_NAME STREQUAL "testLowMachQbmmKineticTransport")
    # Use the production operator with prescribed fluxes and exact donor traces.
    target_sources(${TEST_NAME} PRIVATE
      ../details/LowMachQbmmExplicitTransport.cc
      ../details/LowMachQbmmKineticTransport.cc)
  endif()
  if(TEST_NAME STREQUAL "testLowMachQbmmLinearTransport")
    target_sources(${TEST_NAME} PRIVATE
      ../details/LowMachQbmmExplicitTransport.cc
      ../details/LowMachQbmmLinearTransport.cc)
  endif()
  if(TEST_NAME STREQUAL "testVolumeContinuityGeometryAccess")
    target_sources(${TEST_NAME} PRIVATE
      VolumeContinuityALETestProvider.cc
      VolumeContinuityModelTestProvider.cc)
    set_source_files_properties(
      testVolumeContinuityGeometryAccess.cc
      VolumeContinuityALETestProvider.cc
      VolumeContinuityModelTestProvider.cc
      PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
  endif()
  simplefluid_configure_test_executable(${TEST_NAME})
  if(TEST_NAME MATCHES "^test(LowMachQbmmKineticTransport|QbmmKineticGravity|QbmmLinearGravityCirculation)$")
    simplefluid_discover_tests(${TEST_NAME}
      TEST_FILTER "QbmmCylindricalKinetic*.*"
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q8d/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;kinetic\\;low_mach\\;conservation"
        TIMEOUT 900 ENVIRONMENT "SIMPLEFLUID_Q8D_EXPECTED_RANKS=1")
  endif()
  if(TEST_NAME STREQUAL "testVerificationCases")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "verification\\;integration"
    )
  elseif(TEST_NAME STREQUAL "testALESpeciesConvergence")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "ale\\;reacting_gas\\;convergence\\;roundoff"
    )
  elseif(TEST_NAME STREQUAL "testBoussinesqPlanarALE")
    simplefluid_discover_tests(
      ${TEST_NAME}
      PROPERTIES LABELS "ale\\;free_surface\\;boussinesq\\;conservation\\;transaction"
    )
  elseif(TEST_NAME STREQUAL "testLowMachQbmmTransport")
    simplefluid_discover_tests(${TEST_NAME} TEST_FILTER "-QbmmMixedCarrierTransport.*")
  elseif(TEST_NAME STREQUAL "testLowMachQbmmLinearTransport")
    simplefluid_discover_tests(${TEST_NAME}
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q7c/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;low_mach\\;conservation")
  elseif(TEST_NAME MATCHES "^testLowMachQbmmKinetic(Transport)?$")
    simplefluid_discover_tests(${TEST_NAME}
      TEST_FILTER "-QbmmCylindricalKinetic*.*"
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q7b/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;kinetic\\;low_mach\\;conservation")
  elseif(TEST_NAME MATCHES "^testQbmmLinearGravity(Circulation)?$")
    simplefluid_discover_tests(${TEST_NAME}
      TEST_FILTER "-QbmmCylindricalKinetic*.*"
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q8c/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;gravity\\;low_mach\\;conservation"
        TIMEOUT 900 ENVIRONMENT "SIMPLEFLUID_Q8C_EXPECTED_RANKS=1")
  elseif(TEST_NAME STREQUAL "testQbmmKineticGravity")
    simplefluid_discover_tests(${TEST_NAME}
      TEST_FILTER "-QbmmCylindricalKinetic*.*"
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q8b/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;kinetic\\;low_mach\\;conservation")
  elseif(TEST_NAME STREQUAL "testQbmmApplicationBenchmarks")
    simplefluid_discover_tests(${TEST_NAME}
      TEST_FILTER "QbmmApplicationBenchmarks.*"
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q6/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;low_mach\\;application")
    simplefluid_discover_tests(${TEST_NAME}
      TEST_FILTER "QbmmFourierCavity.*"
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q8a/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;low_mach\\;application\\;fourier"
        TIMEOUT 1800 ENVIRONMENT "SIMPLEFLUID_CAVITY_EXPECTED_RANKS=1")
  elseif(TEST_NAME STREQUAL "testQbmmSlipAndGravity")
    simplefluid_discover_tests(${TEST_NAME}
      XML_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/qbmm-q6/$<CONFIG>"
      PROPERTIES LABELS "qbmm\\;low_mach\\;application")
  else()
    simplefluid_discover_tests(${TEST_NAME})
  endif()
endforeach()

# Finite-solvent fixed-reference Bouss bulk/wall sources and rank-coherent setup.
foreach(_bouss_vapor_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME BoussFiniteVapor_${_bouss_vapor_ranks}procs
        TARGET testBoussFiniteVaporCoupling PROCS ${_bouss_vapor_ranks}
        LABELS "mpi;species;boussinesq;boiling;wall_heat;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

# Clean schema configuration and explicit gas-only reference scheduling.
# Serial discovery uses TEST_SOURCES above; these are actual 2/4-rank groups.
foreach(_schema_api_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME ReactingLegacyCleanup_${_schema_api_ranks}procs
        TARGET testReactingLegacyCleanup PROCS ${_schema_api_ranks}
        LABELS "mpi;species;configuration;conservation;transaction;empty_rank" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME ReactingReferenceSchedule_${_schema_api_ranks}procs
        TARGET testReactingReferenceSchedule PROCS ${_schema_api_ranks}
        LABELS "mpi;species;reference;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

# Typed species surface coupling and exact transported liquid-volume packets.
foreach(_species_surface_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME BoussinesqSpeciesFreeSurface_${_species_surface_ranks}procs
        TARGET testBoussinesqSpeciesFreeSurface PROCS ${_species_surface_ranks}
        LABELS "mpi;species;free_surface;boussinesq;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME SpeciesLiquidVolumePacket_${_species_surface_ranks}procs
        TARGET testSpeciesLiquidVolumePacket PROCS ${_species_surface_ranks}
        LABELS "mpi;species;ale;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME BoussinesqSpeciesPlanarALE_${_species_surface_ranks}procs
        TARGET testBoussinesqPlanarALE PROCS ${_species_surface_ranks}
        ARGS "--gtest_filter=BoussinesqSpeciesPlanarALETest.*"
        LABELS "mpi;species;ale;boussinesq;conservation;transaction;empty_rank" TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(NAME ALESpeciesDenseOracle_2procs
    TARGET testBoussinesqPlanarALE PROCS 2
    ARGS "--gtest_filter=ALESpeciesDenseOracleTest.*"
    LABELS "mpi;species;ale;boussinesq;convergence;transaction" TIMEOUT 120)

# Raw accepted population events preserve tiny material/count endpoints and replay.
foreach(_raw_population_ranks IN ITEMS 2 4)
    foreach(_raw_population_case IN ITEMS PopulationRawCountReplay PopulationRawMaterialReplay PopulationChemicalDonors)
        simplefluid_add_mpi_test(NAME ${_raw_population_case}_${_raw_population_ranks}procs
            TARGET test${_raw_population_case} PROCS ${_raw_population_ranks}
            LABELS "mpi;species;population;conservation;transaction;empty_rank" TIMEOUT 180)
    endforeach()
endforeach()

# Packed family transport on native overlap maps.
foreach(ranks IN ITEMS 2 4)
    set(test "LowMachQbmmPackedTransport_${ranks}procs")
    simplefluid_add_mpi_test(NAME "${test}" TARGET testLowMachQbmmPackedTransport PROCS "${ranks}"
        LABELS "mpi;low_mach;qbmm;multi_rhs;ale;conservation;empty_rank")
    if(TEST "${test}")
        set_tests_properties("${test}" PROPERTIES ENVIRONMENT "SIMPLEFLUID_QBMM_PACKED_EXPECTED_RANKS=${ranks}")
    endif()
endforeach()

# Prescribed native ALE transactions and actual source geometry frames.
foreach(_native_ale_ranks IN ITEMS 2 4)
    foreach(_native_ale_case IN ITEMS LowMachPlanarALE LowMachALEConservation LowMachQbmmPlanarALE ReactingMixtureALE ReactingSourceLocation)
        simplefluid_add_mpi_test(NAME ${_native_ale_case}_${_native_ale_ranks}procs
            TARGET test${_native_ale_case} PROCS ${_native_ale_ranks}
            LABELS "mpi;reacting;low_mach;ale;conservation;transaction;empty_rank" TIMEOUT 180)
    endforeach()
endforeach()

# Use independent test processes and partition-local shards for each portable restart.
function(add_qbmm_restart_tests name target filter environment_prefix labels)
    foreach(ranks IN ITEMS 1 2 4)
        set(fixture "${name}_${ranks}procs")
        set(prefix "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>-${fixture}")
        foreach(stage IN ITEMS Write Read)
            set(test "${fixture}_${stage}")
            if(ranks EQUAL 1)
                add_test(NAME "${test}" COMMAND "$<TARGET_FILE:${target}>" "--gtest_filter=${filter}")
                set_tests_properties("${test}" PROPERTIES LABELS "${labels}" TIMEOUT 180)
            else()
                simplefluid_add_mpi_test(NAME "${test}" TARGET "${target}" PROCS "${ranks}"
                    ARGS "--gtest_filter=${filter}" LABELS "${labels}" TIMEOUT 180)
            endif()
            if(TEST "${test}")
                if(stage STREQUAL "Write")
                    set_tests_properties("${test}" PROPERTIES FIXTURES_SETUP "${fixture}"
                        ENVIRONMENT "${environment_prefix}_WRITE=${prefix};${environment_prefix}_RANKS=${ranks}")
                else()
                    set_tests_properties("${test}" PROPERTIES FIXTURES_REQUIRED "${fixture}"
                        ENVIRONMENT "${environment_prefix}_READ=${prefix};${environment_prefix}_RANKS=${ranks}")
                endif()
            endif()
        endforeach()
    endforeach()
endfunction()

add_qbmm_restart_tests(QbmmPoolFixedQ9cRestart testReactingMixtureQbmmPool
    ReactingMixtureQbmmPool.FixedModesPortableContinuation SIMPLEFLUID_Q9C_FIXED
    "qbmm;reacting;pool;slip;gravity;restart;conservation;empty_rank")
add_qbmm_restart_tests(QbmmPoolRestart testReactingMixtureQbmmPool ReactingMixtureQbmmPool.PortableContinuation
    SIMPLEFLUID_Q9B "qbmm;reacting;pool;restart;conservation;empty_rank")
foreach(ranks IN ITEMS 1 2 4)
    set(test "QbmmPoolContracts_${ranks}procs")
    simplefluid_add_mpi_test(NAME "${test}" TARGET testReactingMixtureQbmmPool PROCS ${ranks}
        ARGS "--gtest_filter=ReactingMixtureQbmmPool.*-*.*PortableContinuation"
        LABELS "mpi;qbmm;reacting;pool;multi_rhs;conservation;transaction;empty_rank" TIMEOUT 240)
    if(TEST "${test}")
        set_tests_properties("${test}" PROPERTIES ENVIRONMENT "SIMPLEFLUID_Q9B_RANKS=${ranks}")
    endif()
endforeach()

foreach(geometry IN ITEMS Fixed Planar)
    add_qbmm_restart_tests(QbmmSourceSubsteps${geometry}Restart testReactingMixtureQbmmSubstepping
        ReactingMixtureQbmmSubstepping.${geometry}PortableContinuation
        SIMPLEFLUID_QBMM_SUBSTEP "qbmm;reacting;substepping;restart;conservation;empty_rank")
endforeach()
foreach(ranks IN ITEMS 1 2 4)
    set(test QbmmSourceSubsteps_${ranks}procs)
    simplefluid_add_mpi_test(NAME "${test}" TARGET testReactingMixtureQbmmSubstepping PROCS ${ranks}
        ARGS "--gtest_filter=ReactingMixtureQbmmSubstepping.*-*.*PortableContinuation"
        LABELS "mpi;qbmm;reacting;substepping;ale;conservation;transaction;empty_rank" TIMEOUT 240)
    if(TEST "${test}")
        set_tests_properties("${test}" PROPERTIES ENVIRONMENT "SIMPLEFLUID_QBMM_SUBSTEP_RANKS=${ranks}")
    endif()
endforeach()

foreach(geometry IN ITEMS Fixed Planar)
    add_qbmm_restart_tests(QbmmFilm${geometry}Restart testReactingMixtureQbmmFilm
        ReactingMixtureQbmmFilm.${geometry}PortableContinuation
        SIMPLEFLUID_FILM "qbmm;film;reacting;restart;conservation;empty_rank")
endforeach()
foreach(ranks IN ITEMS 2 4)
    set(test QbmmFilmContracts_${ranks}procs)
    simplefluid_add_mpi_test(NAME "${test}" TARGET testReactingMixtureQbmmFilm PROCS ${ranks}
        ARGS "--gtest_filter=ReactingMixtureQbmmFilm.*-*.*PortableContinuation"
        LABELS "mpi;qbmm;film;reacting;ale;conservation;transaction;empty_rank" TIMEOUT 180)
    if(TEST "${test}")
        set_tests_properties("${test}" PROPERTIES ENVIRONMENT "SIMPLEFLUID_FILM_RANKS=${ranks}")
    endif()
endforeach()

foreach(velocity IN ITEMS Shared SuppliedSlip)
    add_qbmm_restart_tests(QbmmPoolPlanar${velocity}Restart testReactingMixtureQbmmPoolALE
        ReactingMixtureQbmmPoolALE.${velocity}PortableContinuation SIMPLEFLUID_Q9C_ALE
        "qbmm;pool;ale;restart;transaction;empty_rank")
endforeach()
foreach(ranks IN ITEMS 1 2 4)
    set(test "QbmmPoolPlanarContracts_${ranks}procs")
    simplefluid_add_mpi_test(NAME "${test}" TARGET testReactingMixtureQbmmPoolALE PROCS ${ranks}
        ARGS "--gtest_filter=ReactingMixtureQbmmPoolALE.*-*.*PortableContinuation"
        LABELS "mpi;qbmm;pool;ale;multi_rhs;conservation;transaction;empty_rank" TIMEOUT 240)
    if(TEST "${test}")
        set_tests_properties("${test}" PROPERTIES ENVIRONMENT "SIMPLEFLUID_Q9C_ALE_RANKS=${ranks}")
    endif()
endforeach()

add_qbmm_restart_tests(QbmmPlanarRestart testLowMachQbmmPlanarALE
    LowMachQbmmPlanarALETest.PortableRestartContinuesFromOriginalReference SIMPLEFLUID_Q7A_RESTART "qbmm;ale;restart")
add_qbmm_restart_tests(QbmmPlanarControllerRestart testReactingMixtureQbmmSources
    ReactingMixtureQbmmSourcesTest.PlanarEventsUseLiveGeometryAndReplayAfterSingleOwnerFailure
    SIMPLEFLUID_Q7A_RESTART "qbmm;ale;restart")
foreach(limiter IN ITEMS Realizability TvdMc)
    add_qbmm_restart_tests(QbmmPlanarLinear${limiter}Restart testReactingMixtureQbmmSources
        ReactingMixtureQbmmSourcesTest.PlanarLinear${limiter}PortableContinuation
        SIMPLEFLUID_PLANAR_LINEAR_RESTART "qbmm;ale;restart;transaction;empty_rank")
endforeach()
foreach(ranks IN ITEMS 2 4)
    set(test "QbmmPlanarLinearRestartContracts_${ranks}procs")
    simplefluid_add_mpi_test(NAME "${test}" TARGET testReactingMixtureQbmmSources PROCS ${ranks}
        ARGS "--gtest_filter=ReactingMixtureQbmmSourcesTest.PlanarLinear*PortableContinuation"
        LABELS "mpi;qbmm;ale;restart;transaction;empty_rank" TIMEOUT 180)
    if(TEST "${test}")
        set_tests_properties("${test}" PROPERTIES ENVIRONMENT "SIMPLEFLUID_PLANAR_LINEAR_RESTART_RANKS=${ranks}")
    endif()
endforeach()
add_qbmm_restart_tests(QbmmKineticControllerRestart testReactingMixtureQbmmSources
    ReactingMixtureQbmmSourcesTest.KineticTransportFeedsCoherentEventsThermalsCommitAndRestart
    SIMPLEFLUID_Q7B_RESTART "qbmm;kinetic;restart;transaction;conservation")
add_qbmm_restart_tests(QbmmLinearControllerRestart testReactingMixtureQbmmSources
    ReactingMixtureQbmmSourcesTest.LinearTransportFeedsCoherentEventsThermalsCommitAndRestart
    SIMPLEFLUID_Q7C_RESTART "qbmm;restart;transaction;conservation")

foreach(kind IN ITEMS Unforced Forced)
    add_qbmm_restart_tests(QbmmTvdMc${kind}Restart testReactingMixtureQbmmSources
        ReactingMixtureQbmmSourcesTest.TvdMc${kind}PortableContinuation
        SIMPLEFLUID_TVD_MC_RESTART "qbmm;tvd_mc;restart;transaction")
endforeach()

add_qbmm_restart_tests(QbmmCylindricalKineticRestart testLowMachQbmmCylindricalRestart
    LowMachQbmmCylindricalRestartTest.PortableContinuation SIMPLEFLUID_Q8D_RESTART "qbmm;kinetic;gravity;restart;transaction")
foreach(ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME QbmmCylindricalKineticRestartContracts_${ranks}procs TARGET testLowMachQbmmCylindricalRestart
        PROCS ${ranks} ARGS "--gtest_filter=LowMachQbmmCylindricalRestartTest.*-*.PortableContinuation"
        LABELS "mpi;qbmm;kinetic;restart;transaction")
endforeach()

foreach(kind IN ITEMS Forced ZeroForce)
    add_qbmm_restart_tests(Qbmm${kind}KineticRestart testReactingMixtureQbmmSources
        ReactingMixtureQbmmSourcesTest.${kind}KineticTaggedRestart
        SIMPLEFLUID_Q8B_RESTART "qbmm;kinetic;gravity;restart;transaction")
    add_qbmm_restart_tests(Qbmm${kind}LinearRestart testReactingMixtureQbmmSources
        ReactingMixtureQbmmSourcesTest.${kind}LinearTaggedRestart
        SIMPLEFLUID_Q8C_RESTART "qbmm;gravity;restart;transaction")
endforeach()

foreach(_linear_ranks IN ITEMS 2 4)
    foreach(target IN ITEMS LowMachQbmmLinearTransport QbmmLinearGravity QbmmLinearGravityCirculation
            ReactingMixtureQbmmSources)
        set(test QbmmTvdMc${target}_${_linear_ranks}procs)
        simplefluid_add_mpi_test(NAME ${test}
            TARGET test${target} PROCS ${_linear_ranks}
            ARGS "--gtest_filter=*.TvdMc*" LABELS "mpi;qbmm;tvd_mc;conservation" TIMEOUT 900)
        if(TEST ${test})
            set_tests_properties(${test} PROPERTIES ENVIRONMENT
                "SIMPLEFLUID_Q8C_EXPECTED_RANKS=${_linear_ranks};SIMPLEFLUID_TVD_MC_RESTART_RANKS=${_linear_ranks}")
        endif()
    endforeach()
    foreach(_gravity_case IN ITEMS QbmmLinearGravity QbmmLinearGravityCirculation)
        set(_test "${_gravity_case}_${_linear_ranks}procs")
        simplefluid_add_mpi_test(NAME ${_test} TARGET test${_gravity_case} PROCS ${_linear_ranks}
            ARGS "--gtest_filter=-QbmmCylindricalKinetic*.*:*.TvdMc*"
            LABELS "mpi;qbmm;gravity;low_mach;conservation" TIMEOUT 900)
        if(TEST ${_test})
            set_tests_properties(${_test} PROPERTIES ENVIRONMENT "SIMPLEFLUID_Q8C_EXPECTED_RANKS=${_linear_ranks}")
        endif()
    endforeach()
    simplefluid_add_mpi_test(NAME LowMachQbmmLinearTransport_${_linear_ranks}procs
        TARGET testLowMachQbmmLinearTransport PROCS ${_linear_ranks}
        ARGS "--gtest_filter=-*.TvdMc*:QbmmMovingLinearTransport.*"
        LABELS "mpi;qbmm;low_mach;conservation;empty_rank" TIMEOUT 900)
endforeach()

foreach(ranks IN ITEMS 2 4)
    set(test QbmmMovingLinearTransport_${ranks}procs)
    simplefluid_add_mpi_test(NAME ${test} TARGET testLowMachQbmmLinearTransport PROCS ${ranks}
        ARGS "--gtest_filter=QbmmMovingLinearTransport.*" LABELS "mpi;qbmm;ale;conservation;empty_rank" TIMEOUT 300)
    if(TEST ${test})
        set_tests_properties(${test} PROPERTIES ENVIRONMENT "SIMPLEFLUID_Q8C_EXPECTED_RANKS=${ranks}")
    endif()
    set(test QbmmMovingLinearPiston_${ranks}procs)
    simplefluid_add_mpi_test(NAME ${test} TARGET testLowMachQbmmPlanarALE PROCS ${ranks}
        ARGS "--gtest_filter=LowMachQbmmPlanarALETest.LinearClosedPiston*"
        LABELS "mpi;qbmm;ale;conservation;empty_rank")
    if(TEST ${test})
        set_tests_properties(${test} PROPERTIES ENVIRONMENT "SIMPLEFLUID_QBMM_PLANAR_EXPECTED_RANKS=${ranks}")
    endif()
endforeach()

foreach(_kinetic_ranks IN ITEMS 2 4)
    foreach(_kinetic_case IN ITEMS LowMachQbmmKinetic LowMachQbmmKineticTransport QbmmKineticGravity)
        simplefluid_add_mpi_test(NAME ${_kinetic_case}_${_kinetic_ranks}procs
            TARGET test${_kinetic_case} PROCS ${_kinetic_ranks}
            ARGS "--gtest_filter=-QbmmCylindricalKinetic*.*"
            LABELS "mpi;qbmm;kinetic;low_mach;conservation;transaction;empty_rank" TIMEOUT 180)
    endforeach()
endforeach()

set(_q8d_suites QbmmCylindricalKineticMetric QbmmCylindricalKinetic QbmmCylindricalKineticHeat)
set(_q8d_targets testLowMachQbmmKineticTransport testQbmmKineticGravity testQbmmLinearGravityCirculation)
foreach(suite target IN ZIP_LISTS _q8d_suites _q8d_targets)
    foreach(ranks IN ITEMS 2 4)
        set(name "${suite}_${ranks}procs")
        simplefluid_add_mpi_test(NAME ${name} TARGET ${target} PROCS ${ranks}
            ARGS "--gtest_filter=${suite}.*" LABELS "mpi;qbmm;kinetic;conservation" TIMEOUT 900)
        if(TEST ${name})
            set_tests_properties(${name} PROPERTIES ENVIRONMENT "SIMPLEFLUID_Q8D_EXPECTED_RANKS=${ranks}")
        endif()
    endforeach()
endforeach()

# Configured local interphase thermal events, including the prescribed ALE transaction.
foreach(_interphase_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME ReactingMixtureInterphaseTransfer_${_interphase_ranks}procs
        TARGET testReactingMixtureInterphaseTransfer PROCS ${_interphase_ranks}
        LABELS "mpi;reacting;low_mach;thermal;ale;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

foreach(ranks IN ITEMS 1 2 4)
    set(test QbmmMixedCarrierTransport_${ranks}procs)
    if(ranks EQUAL 1)
        add_test(NAME "${test}" COMMAND "$<TARGET_FILE:testLowMachQbmmTransport>"
            "--gtest_filter=QbmmMixedCarrierTransport.*")
        set_tests_properties("${test}" PROPERTIES LABELS "low_mach;qbmm;multi_rhs;conservation;restart" TIMEOUT 300)
    else()
        simplefluid_add_mpi_test(NAME "${test}" TARGET testLowMachQbmmTransport PROCS "${ranks}"
            ARGS "--gtest_filter=QbmmMixedCarrierTransport.*"
            LABELS "mpi;low_mach;qbmm;multi_rhs;conservation;restart;empty_rank" TIMEOUT 300)
    endif()
    if(TEST ${test})
        set_tests_properties(${test} PROPERTIES ENVIRONMENT "SIMPLEFLUID_MIXED_CARRIER_RANKS=${ranks}")
    endif()
endforeach()

# Conservative native mixture flow and its external reaction controller.
foreach(_mixture_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME QbmmSlipAndGravity_${_mixture_ranks}procs
        TARGET testQbmmSlipAndGravity PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;qbmm;conservation;transaction;application;empty_rank" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME QbmmApplicationBenchmarks_${_mixture_ranks}procs
        TARGET testQbmmApplicationBenchmarks PROCS ${_mixture_ranks}
        ARGS "--gtest_filter=QbmmApplicationBenchmarks.*"
        LABELS "mpi;low_mach;reacting;qbmm;conservation;transaction;application;empty_rank" TIMEOUT 600)
    foreach(_cavity_cells IN ITEMS 16 32)
        set(_cavity_test QbmmFourierCavity${_cavity_cells}_${_mixture_ranks}procs)
        simplefluid_add_mpi_test(NAME ${_cavity_test}
            TARGET testQbmmApplicationBenchmarks PROCS ${_mixture_ranks}
            ARGS "--gtest_filter=QbmmFourierCavity.Matched${_cavity_cells}"
            LABELS "mpi;low_mach;reacting;qbmm;conservation;transaction;application;fourier" TIMEOUT 1800)
        if(TEST ${_cavity_test})
            set_tests_properties(${_cavity_test} PROPERTIES ENVIRONMENT "SIMPLEFLUID_CAVITY_EXPECTED_RANKS=${_mixture_ranks}")
        endif()
    endforeach()
    simplefluid_add_mpi_test(NAME LowMachQbmmState_${_mixture_ranks}procs
        TARGET testLowMachQbmmState PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;qbmm;population;transaction;empty_rank" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME ReactingMixtureQbmmReleaseGate_${_mixture_ranks}procs
        TARGET testReactingMixtureQbmmReleaseGate PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;qbmm;population;conservation;transaction;empty_rank" TIMEOUT 300)
    simplefluid_add_mpi_test(NAME LowMachQbmmTransport_${_mixture_ranks}procs
        TARGET testLowMachQbmmTransport PROCS ${_mixture_ranks} ARGS "--gtest_filter=-QbmmMixedCarrierTransport.*"
        LABELS "mpi;low_mach;qbmm;population;conservation;transaction;empty_rank" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME ReactingMixtureQbmmSources_${_mixture_ranks}procs
        TARGET testReactingMixtureQbmmSources PROCS ${_mixture_ranks} ARGS "--gtest_filter=-*.TvdMc*"
        LABELS "mpi;low_mach;reacting;qbmm;population;conservation;transaction;empty_rank" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME NativeSourceDonors_${_mixture_ranks}procs
        TARGET testNativeSourceDonors PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;diffusion;thermal;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachMixture_${_mixture_ranks}procs
        TARGET testLowMachMixtureSolver PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachMultiRhsTransport_${_mixture_ranks}procs
        TARGET testLowMachMultiRhsTransport PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;multi_rhs;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingMixtureCoupler_${_mixture_ranks}procs
        TARGET testReactingMixtureCoupler PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachSourceContracts_${_mixture_ranks}procs
        TARGET testLowMachSourceContracts PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachPopulationNumbers_${_mixture_ranks}procs
        TARGET testLowMachPopulationNumbers PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;population;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingMixturePopulationEvents_${_mixture_ranks}procs
        TARGET testReactingMixturePopulationEvents PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;population;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachFaceTransport_${_mixture_ranks}procs
        TARGET testLowMachFaceTransportAdapter PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;diffusion;thermal;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingMixtureFaces_${_mixture_ranks}procs
        TARGET testReactingMixtureFaceCoupling PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;diffusion;thermal;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachCaloricFeasibility_${_mixture_ranks}procs
        TARGET testLowMachCaloricFeasibility PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;reacting;thermodynamics;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME LowMachPressureContracts_${_mixture_ranks}procs
        TARGET testLowMachPressureContracts PROCS ${_mixture_ranks}
        LABELS "mpi;low_mach;pressure;conservation;transaction;empty_rank" TIMEOUT 120)
    simplefluid_add_mpi_test(NAME ReactingSpeciesHistory_${_mixture_ranks}procs
        TARGET testReactingSpeciesHistory PROCS ${_mixture_ranks}
        LABELS "mpi;reacting;conservation;transaction;empty_rank" TIMEOUT 120)
endforeach()

# Schema-indexed reacting fields and the liquid inventory follower.
simplefluid_add_mpi_test(NAME ReactingSpeciesModel_2procs TARGET testReactingSpeciesModel PROCS 2
    ARGS "--gtest_filter=-*FactoryNativeMesh*"
    LABELS "mpi;reacting_gas;species;conservation;transaction" TIMEOUT 180)
simplefluid_add_mpi_test(NAME ReactingSpeciesModel_EmptyRank_4procs TARGET testReactingSpeciesModel PROCS 4
    ARGS "--gtest_filter=*CollectiveInitialization*:*RankAsymmetric*"
    LABELS "mpi;reacting_gas;species;collective;empty_rank" TIMEOUT 120)
simplefluid_add_mpi_test(NAME ReactingBoussinesqCoupling_2procs TARGET testReactingBoussinesqCoupling PROCS 2
    LABELS "mpi;reacting_gas;boussinesq;conservation;transaction" TIMEOUT 180)
simplefluid_add_mpi_test(NAME SpeciesLiquidMassInventory_2procs TARGET testSpeciesLiquidMassInventory PROCS 2
    LABELS "mpi;reacting_gas;free_surface;ale;conservation;transaction" TIMEOUT 180)

simplefluid_add_mpi_test(
    NAME CoupledPreconditionerValidation_2procs
    TARGET testCoupledPressureVelocitySolver
    PROCS 2
    ARGS "--gtest_filter=*RejectsRankLocalInvalidPreconditionerStateCollectively*"
    LABELS "mpi;coupled;collective"
    TIMEOUT 60)

simplefluid_add_mpi_test(
    NAME CoupledPressureSmoothing_2procs
    TARGET testCoupledPressureVelocitySolver
    PROCS 2
    ARGS "--gtest_filter=CoupledPressureVelocitySolverTest.PressureMultigridContractsCheckerboardModeAndRefreshesNumerics"
    LABELS "mpi;pressure_velocity;preconditioner;accuracy;reuse"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BelosAccurateResidual_2procs
    TARGET testBelosLinearSolver
    PROCS 2
    ARGS "--gtest_filter=BelosLinearSolverTest.AccurateStoredCrsResidual*:BelosLinearSolverTest.RefinesGmresLossOfAccuracyAgainstOriginalTrueResidualGate:BelosLinearSolverTest.GmresLossOfAccuracyRefinementKeepsIterationAndRestartBounds:BelosLinearSolverTest.GmresDoesNotRetryOrdinaryFailureLargeGapOrNonfiniteResidual:BelosLinearSolverTest.GmresDefectCorrectionRestoresOriginalVectorsAfterException"
    LABELS "mpi;linear_solver;accuracy"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BelosSymmetricAMG_2procs
    TARGET testBelosLinearSolver
    PROCS 2
    ARGS "--gtest_filter=BelosLinearSolverTest.SymmetricSmoothedAmg*"
    LABELS "mpi;linear_solver;preconditioner"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BelosMultiRhs_2procs
    TARGET testBelosLinearSolver
    PROCS 2
    ARGS "--gtest_filter=BelosLinearSolverTest.GmresResetsBlockSizeAcrossConsecutiveSolvesAndRefinements:BelosLinearSolverTest.BatchWidthChunkingInvarianceAndExecutionModes:BelosLinearSolverTest.PerColumnTrueResidualRejectsTraceMaskingAndHonorsActiveMask:BelosLinearSolverTest.CgSolvesMultiRhsColumnSeriallyAndReportsExecutionMode:BelosLinearSolverTest.PerColumnPolicyTinyAcceptedColumnDoesNotBlockGmresRefinement:BelosLinearSolverTest.NonReusablePreconditionerRebuildsWhenLeadingChunkIsSkipped:BelosLinearSolverTest.BatchedColumnIterationsMatchColumnSerialConvergence:BelosLinearSolverTest.CgColumnsKeepIndependentIterationCounts:BelosLinearSolverTest.GmresRefinementPassesResetAndAccumulatePerColumn:BelosLinearSolverTest.GmresRefinementUsesEachColumnsRemainingBudget:BelosLinearSolverTest.GmresRefinementGroupsColumnsByRemainingBudget:BelosLinearSolverTest.GmresRefinementRetriesLoaGroupAfterLaterGroupSucceeds:BelosLinearSolverTest.GmresAsymmetricGapAccountingMatchesScalarColumns:BelosLinearSolverTest.GmresColumnIterationsOnEmptyRank:BelosLinearSolverTest.BatchedCgAndBiCGStab*:BelosLinearSolverTest.BatchedBiCGStab*:BelosLinearSolverTest.BatchedHealthyColumnCorrectionIsolatedFromFailedNeighbor"
    LABELS "mpi;linear_solver;multi_rhs"
    TIMEOUT 120)

target_sources(testBoussinesqSolver PRIVATE
  PressureGradientStencilTestSupport.cc)

if(SIMPLEFLUID_ENABLE_NOX)
  simplefluid_add_mpi_test(
      NAME CavityNoxVerification_2procs
      TARGET testVerificationCases
      PROCS 2
      ARGS "--gtest_filter=VerificationCasesTest.LidDrivenCavityRe100Nox:VerificationCasesTest.LidDrivenCavityRe1000Nox"
      LABELS "verification;integration;mpi;nonlinear_solver"
      TIMEOUT 180)
  if(TEST CavityNoxVerification_2procs)
    set_tests_properties(CavityNoxVerification_2procs PROPERTIES
        ENVIRONMENT "SIMPLEFLUID_PROFILE_OUTPUT_DIR=")
  endif()
endif()

simplefluid_add_mpi_test(
    NAME CoupledNonlinearSolver_2procs
    TARGET testCoupledNonlinearSolver
    PROCS 2
    LABELS "mpi;nonlinear_solver;pressure_velocity"
    TIMEOUT 180)

add_executable(
  testReactingGasModelMultiRank
  testReactingGasModelMultiRank.cc)
target_link_libraries(
  testReactingGasModelMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testReactingGasModelMultiRank)

add_executable(
  testFluidSolverMultiRank
  testFluidSolverMultiRank.cc)
target_link_libraries(
  testFluidSolverMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testFluidSolverMultiRank)

add_executable(
  testFeedbackMapMultiRank
  testFeedbackMapMultiRank.cc)
target_link_libraries(
  testFeedbackMapMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testFeedbackMapMultiRank)

add_executable(
  testPlanarFreeSurfaceModelMultiRank
  testPlanarFreeSurfaceModelMultiRank.cc)
target_link_libraries(
  testPlanarFreeSurfaceModelMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testPlanarFreeSurfaceModelMultiRank)

add_executable(
  testDICPreconditionerMultiRank
  testDICPreconditionerMultiRank.cc)
target_link_libraries(
  testDICPreconditionerMultiRank
  SimpleFluid::Solvers
  GTest::gtest_main)
simplefluid_configure_test_executable(testDICPreconditionerMultiRank)

foreach(_dic_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(
      NAME DICPreconditioner_${_dic_procs}procs
      TARGET testDICPreconditionerMultiRank
      PROCS ${_dic_procs}
      LABELS "mpi;linear_solver;dic"
      TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(
    NAME ReactingGasModel_2procs
    TARGET testReactingGasModelMultiRank
    PROCS 2
    LABELS "mpi;reacting_gas"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FluidSolver_2procs
    TARGET testFluidSolverMultiRank
    PROCS 2
    LABELS "mpi;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FluidSolverGeometryRefresh_2procs
    TARGET testFluidSolver
    PROCS 2
    ARGS "--gtest_filter=FluidSolverTest.CentralGeometryRefreshRestoresCoreCachesAndVtuPoints"
    LABELS "mpi;ale;geometry_epoch;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqMutableGeometryRefresh_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.MutableNativeConstructionPreservesFieldIdentityAndConstOverload"
    LABELS "mpi;ale;geometry_epoch;boussinesq"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME CoupledKrylovDirichletContinuity_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.CoupledKrylovDirichletContinuityMatchesBoundaryFaceFlux"
    LABELS "mpi;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativePhysicalBoussinesq_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.NativePhysicalModelsRunCoupledKrylovOnDistributedCartesianMesh"
    LABELS "mpi;boussinesq;physical_models;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativeUnstructuredPhysicalBoussinesq_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.NativePhysicalModelsRunCoupledKrylovOnPartitionedUnstructuredMesh"
    LABELS "mpi;boussinesq;physical_models;pressure_velocity;unstructured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME NativeUnstructuredCoupledFlow_2procs
    TARGET testBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=BoussinesqSolverTest.NativePartitionedUnstructuredCoupledKrylovAdvancesForcedFlow"
    LABELS "mpi;boussinesq;physical_models;pressure_velocity;unstructured"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME TurbulentBoussinesq_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.AdvancesTurbulenceThroughSegregatedAndCoupledPressureVelocitySolves:TurbulentBoussinesqSolverTest.PassesDirectBuoyancyContextThroughSegregatedAndCoupledSolvers"
    LABELS "mpi;turbulence;pressure_velocity"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME IncompressibleIsothermalSolver_2procs
    TARGET testIncompressibleIsothermalSolver
    PROCS 2
    ARGS "--gtest_filter=IncompressibleIsothermalSolverTest.AdvancesStandardKEpsilonThroughPisoAndCoupledKrylov:IncompressibleIsothermalSolverTest.AdvancesStandardKEpsilonOnNativeMeshHandle:IncompressibleIsothermalSolverTest.ModifiedPressureAbsorbsNonuniformK:IncompressibleIsothermalSolverTest.ZeroGradientOutletStressMatchesManufacturedShear"
    LABELS "mpi;turbulence;pressure_velocity;isothermal"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME CombinedTurbulentReactingGas_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.CouplesShearedRansSpeciesTransportPrescribedHeatAndDensityFeedbackOnTwoRanks"
    LABELS "mpi;turbulence;reacting_gas;physical_models;pressure_velocity;acceptance"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarSourceDecayConfiguration_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarSourceDecayTransportTest.CollectivelyRejectsRankDivergentConfigurationAndSource"
    LABELS "mpi;scalar_source_decay;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarGroupNonOrthogonalConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarSourceDecayTransportTest.EffectiveDiffusivityIncludesExplicitNonOrthogonalCorrection"
    LABELS "mpi;scalar_source_decay"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarGroupAdvectionConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarSourceDecayTransportTest.LiquidAdvectionConservesDistributedInventory:ScalarSourceDecayTransportTest.CollectivelyRejectsRankDivergentAdvanceSelection"
    LABELS "mpi;scalar_source_decay;transport"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME ScalarVoidDiffusionConservation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=ScalarVoidFractionModelTest.DiffusionSmoothsAndConservesGlobalVoid"
    LABELS "mpi;void_fraction"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoilingCollectiveValidation_2procs
    TARGET testPhase13PlusModels
    PROCS 2
    ARGS "--gtest_filter=BoilingSourceModelTest.CollectivelyRejectsRankAsymmetricInvalidCellState"
    LABELS "mpi;boiling;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME FeedbackMap_2procs
    TARGET testFeedbackMapMultiRank
    PROCS 2
    LABELS "mpi;feedback_mapping"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PlanarFreeSurfaceModel_2procs
    TARGET testPlanarFreeSurfaceModelMultiRank
    PROCS 2
    LABELS "mpi;free_surface"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqFreeSurfaceConfig_2procs
    TARGET testBoussinesqFreeSurface
    PROCS 2
    ARGS "--gtest_filter=BoussinesqFreeSurfaceTest.CollectivelyRejectsRankDivergentOptions"
    LABELS "mpi;free_surface;boussinesq;configuration"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqFreeSurfaceStepPreflight_2procs
    TARGET testBoussinesqFreeSurface
    PROCS 2
    ARGS "--gtest_filter=BoussinesqFreeSurfaceTest.CellMassInventoryStepIsPartitionIndependent:BoussinesqFreeSurfaceTest.CollectivelyRejectsRemovalWhileBoilingOwnsSubmergedSteam:BoussinesqFreeSurfaceTest.CollectivelyRejectsRankDivergentExistingStateDuringConfiguration:BoussinesqFreeSurfaceTest.CollectivelyRejectsRankDivergentPresenceBeforeStep:BoussinesqFreeSurfaceTest.CollectivelyRejectsPositiveRankDivergentTimestepBeforeStep"
    LABELS "mpi;free_surface;boussinesq;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME BoussinesqPlanarALE_2procs
    TARGET testBoussinesqPlanarALE
    PROCS 2
    LABELS "mpi;ale;free_surface;boussinesq;conservation;transaction"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME PlanarALEBoundaryValidation_2procs
    TARGET testPlanarALEBoundary
    PROCS 2
    ARGS "--gtest_filter=PlanarALEBoundaryTest.ValidatesGeometryPreservesTangentialSlipAndEnforcesZeroRelativeTopFlux:PlanarALEBoundaryTest.CollectivelyRejectsRankDivergentControlsAndForeignFields"
    LABELS "mpi;ale;free_surface;validation"
    TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME VolumeContinuityModelValidation_2procs
    TARGET testPlanarALEBoundary
    PROCS 2
    ARGS "--gtest_filter=VolumeContinuityModelTest.CollectivelyRejectsRankDivergentGlobalAndOptionalInputs:VolumeContinuityModelTest.MeshDefectCorrectionPreservesPhysicalSourceAndUsesOwnedVolumeWeights"
    LABELS "mpi;ale;continuity;validation"
    TIMEOUT 120)

foreach(_composite_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(
      NAME CoupledBlockOperator_${_composite_procs}procs
      TARGET testCoupledBlockOperator
      PROCS ${_composite_procs}
      LABELS "mpi;pressure_velocity;composite"
      TIMEOUT 180)
endforeach()

foreach(_backend_procs IN ITEMS 2 4)
  simplefluid_add_mpi_test(
      NAME CoupledSolverBackends_${_backend_procs}procs
      TARGET testCoupledSolverBackends
      PROCS ${_backend_procs}
      LABELS "mpi;pressure_velocity;composite;solver_backends"
      TIMEOUT 300)
endforeach()

simplefluid_add_mpi_test(
    NAME CoupledBackendsCombinedPhysics_2procs
    TARGET testTurbulentBoussinesqSolver
    PROCS 2
    ARGS "--gtest_filter=TurbulentBoussinesqSolverTest.CoupledBackendsAdvanceCombinedRansReactingGasAndFeedback:TurbulentBoussinesqSolverTest.AdvancesWallTreatmentClosurePairingsThroughSegregatedAndCoupledSolvers"
    LABELS "mpi;pressure_velocity;composite;turbulence;reacting_gas"
    TIMEOUT 180)

foreach(_region_procs IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME MultiRegionFlow_${_region_procs}procs TARGET testCoupledSolverBackends
        PROCS ${_region_procs} ARGS "--gtest_filter=*MultiRegion*" LABELS "mpi;multi_region;pressure_velocity" TIMEOUT 300)
    simplefluid_add_mpi_test(NAME MultiRegionSolverALE_${_region_procs}procs TARGET testBoussinesqPlanarALE
        PROCS ${_region_procs} ARGS "--gtest_filter=BoussinesqPlanarALESupportMatrixTest.AcceptsCompositeAxialMotion"
        LABELS "mpi;multi_region;ale;pressure_velocity" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME SteadyDiffusionConvergence_${_region_procs}procs TARGET testSteadyDiffusionConvergence
        PROCS ${_region_procs} LABELS "mpi;non_orthogonal;multi_region" TIMEOUT 120)
endforeach()

simplefluid_add_mpi_test(
    NAME SASSupportedPaths_2procs
    TARGET testSASSupportedPaths
    PROCS 2
    LABELS "mpi;turbulence;sas;small_fixture"
    TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASMaterialFeedback_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.MaterialFeedbackPublishesAndRollsBackWithSAS"
    LABELS "mpi;turbulence;sas;feedback;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASNonlinear_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.Nox*"
    LABELS "mpi;turbulence;sas;nonlinear_solver;transaction;small_fixture" TIMEOUT 180)

simplefluid_add_mpi_test(NAME SASScalarVoid_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.ScalarVoidDiffusionCollapseAndFeedbackRollBackWithSAS"
    LABELS "mpi;turbulence;sas;void;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASReactingGas_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.ReactingGasAndSpeciesLedgersRollBackWithSAS"
    LABELS "mpi;turbulence;sas;reacting_gas;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASBoiling_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.BoilingSourcesAndPendingStateRollBackWithSAS"
    LABELS "mpi;turbulence;sas;boiling;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(NAME SASFreeSurface_2procs TARGET testSASSupportedPaths PROCS 2
    ARGS "--gtest_filter=SASSupportedPathsTest.FixedFreeSurface*:SASSupportedPathsTest.LazyFreeSurface*"
    LABELS "mpi;turbulence;sas;free_surface;small_fixture" TIMEOUT 120)

simplefluid_add_mpi_test(
    NAME VolumeContinuityAccess_2procs
    TARGET testVolumeContinuityGeometryAccess
    PROCS 2
    LABELS "mpi;ale;continuity;validation"
    TIMEOUT 60)

foreach(_semi_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME SemiStructuredFlow_${_semi_ranks}procs
        TARGET testCoupledSolverBackends PROCS ${_semi_ranks}
        ARGS "--gtest_filter=*SemiStructuredNonOrthogonal*"
        LABELS "mpi;semi_structured;pressure_velocity;non_orthogonal" TIMEOUT 180)
    simplefluid_add_mpi_test(NAME SemiStructuredSolverALE_${_semi_ranks}procs
        TARGET testBoussinesqPlanarALE PROCS ${_semi_ranks}
        ARGS "--gtest_filter=BoussinesqPlanarALESupportMatrixTest.*SemiStructured*"
        LABELS "mpi;semi_structured;ale;conservation;transaction;empty_rank" TIMEOUT 180)
endforeach()

foreach(_semi_ranks IN ITEMS 2 4)
    simplefluid_add_mpi_test(NAME SemiStructuredVolumeContinuity_${_semi_ranks}procs
        TARGET testVolumeContinuityModel PROCS ${_semi_ranks}
        LABELS "mpi;semi_structured;ale;continuity;transaction;empty_rank" TIMEOUT 120)
endforeach()
