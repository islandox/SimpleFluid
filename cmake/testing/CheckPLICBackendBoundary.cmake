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
    FVM/details/GeneralVoFFiniteTransport.cc FVM/details/GeneralVoFFiniteTransport.tcc
    FVM/details/FiniteVoFTransport.cc FVM/details/FiniteVoFTransport.tcc
    solvers/GeneralVoFSolver.cc solvers/details/GeneralVoFSolverSupport.tcc
    solvers/details/GeneralVoFSolverFinite.tcc solvers/details/GeneralVoFExactSolver.tcc)
    simplefluid_check_plic_include_order("${SIMPLEFLUID_SOURCE_DIR}/src/${source}")
endforeach()
message(STATUS "Native AMR, FVM and solver clients use public PLIC backend boundaries")
