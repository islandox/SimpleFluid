if(NOT DEFINED SIMPLEFLUID_SOURCE_DIR)
    message(FATAL_ERROR "SIMPLEFLUID_SOURCE_DIR is required")
endif()

# Application/native clients select modes through public PLIC interfaces. Backend
# implementations, including rational arithmetic, belong inside PLIC itself.
foreach(component IN ITEMS geometry/mesh FVM solvers)
    file(GLOB_RECURSE sources
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/*.hh"
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/*.cc"
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/*.ipp"
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/*.tcc")
    foreach(source IN LISTS sources)
        if(source MATCHES "/unitTests/")
            continue()
        endif()
        file(READ "${source}" contents)
        if(contents MATCHES "#[ \t]*include[ \t]*[<\"]geometry/PLIC/details/(ordinary|bounded|exact)/")
            message(FATAL_ERROR "PLIC backend implementation included by consumer: ${source}")
        endif()
        if(contents MATCHES "#[ \t]*include[ \t]*[<\"]geometry/(PLIC/Exact|mesh/ExactExtrudedConstruction)")
            message(FATAL_ERROR "Retired geometry backend header included by consumer: ${source}")
        endif()
        if(contents MATCHES "Exact::PolyhedralReconstruction|PolyhedralPhaseSampleFor|PolyhedralReconstruction::[a-z_]+_finite")
            message(FATAL_ERROR "Retired reconstruction API bypasses mode-typed PLIC entry: ${source}")
        endif()
        if(contents MATCHES "cpp_rational|utils/ExactRational[.]hh")
            message(FATAL_ERROR "Rational geometry arithmetic bypasses PLIC: ${source}")
        endif()
        if(contents MATCHES "(Meshes|PLIC)::Exact(PlanarFace|PolyhedralComplex|SweptFace|GeometryMeasure|ConvexFragment|ExtrudedConstruction)")
            message(FATAL_ERROR "Retired backend-specific geometry type used by consumer: ${source}")
        endif()
    endforeach()
endforeach()
# Compiled providers and declaration headers must be included before code.
# Public entry headers use prerequisite declarations to keep this order valid.
function(simplefluid_check_plic_include_order source)
    file(READ "${source}" contents)
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" contents "${contents}")
    string(REGEX REPLACE "//[^\n\r]*" "" contents "${contents}")
    string(REPLACE "\n" ";" lines "${contents}")
    set(seen_code FALSE)
    foreach(line IN LISTS lines)
        string(STRIP "${line}" line)
        if(seen_code AND line MATCHES "^#[ \t]*include[^\n]*[.](hh|tcc)[>\"]")
            message(FATAL_ERROR "Declaration/provider include follows code: ${source}: ${line}")
        endif()
        if(NOT line STREQUAL "" AND NOT line MATCHES "^#")
            set(seen_code TRUE)
        endif()
    endforeach()
endfunction()
file(GLOB_RECURSE plic_sources
    "${SIMPLEFLUID_SOURCE_DIR}/src/geometry/PLIC/*.hh"
    "${SIMPLEFLUID_SOURCE_DIR}/src/geometry/PLIC/*.cc"
    "${SIMPLEFLUID_SOURCE_DIR}/src/geometry/PLIC/*.ipp"
    "${SIMPLEFLUID_SOURCE_DIR}/src/geometry/PLIC/*.tcc")
foreach(source IN LISTS plic_sources)
    simplefluid_check_plic_include_order("${source}")
endforeach()
foreach(source IN ITEMS
    FVM/GeneralVoF.cc FVM/GeneralVoF.hh FVM/details/GeneralVoFState.cc
    FVM/details/FiniteVoFTransport.cc FVM/details/FiniteVoFTransport.tcc
    solvers/GeneralVoFSolver.cc)
    simplefluid_check_plic_include_order("${SIMPLEFLUID_SOURCE_DIR}/src/${source}")
endforeach()
foreach(component IN ITEMS FVM solvers)
    file(GLOB_RECURSE operation_sources
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/details/GeneralVoF/*.hh"
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/details/GeneralVoF/*.cc"
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/details/GeneralVoF/*.ipp"
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/details/GeneralVoF/*.tcc"
        "${SIMPLEFLUID_SOURCE_DIR}/src/${component}/explicit_instantiation/GeneralVoF/*.cc")
    foreach(source IN LISTS operation_sources)
        simplefluid_check_plic_include_order("${source}")
        if(component STREQUAL "solvers" AND NOT source MATCHES "/SolverImpl[.]")
            file(READ "${source}" contents)
            if(contents MATCHES "GeneralVoFSolver::Impl|#[ \t]*include[^\n]*GeneralVoF/SolverImpl[.]")
                message(FATAL_ERROR "Numerical operation depends on solver publication owner: ${source}")
            endif()
            if(source MATCHES "/(Execution|NodeVelocity|Curvature|PhaseStorage|Diagnostics|FaceRates|FaceTransfer|AdaptedFaceTransfer|OwnedFaceTransfer|Reduction)[.]")
                if(contents MATCHES "#[ \t]*include[^\n]*GeneralVoF/(State|Resources)[.]hh")
                    message(FATAL_ERROR "Numerical component requires a whole solver state/resource bundle: ${source}")
                endif()
            endif()
        endif()
    endforeach()
endforeach()
message(STATUS "Native AMR, FVM and solver clients use public PLIC backend boundaries")
