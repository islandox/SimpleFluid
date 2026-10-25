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
        if(contents MATCHES "cpp_rational|utils/ExactRational[.]hh")
            message(FATAL_ERROR "Rational geometry arithmetic bypasses PLIC: ${source}")
        endif()
        if(contents MATCHES "(Meshes|PLIC)::Exact(PlanarFace|PolyhedralComplex|SweptFace|GeometryMeasure|ConvexFragment|ExtrudedConstruction)")
            message(FATAL_ERROR "Retired backend-specific geometry type used by consumer: ${source}")
        endif()
    endforeach()
endforeach()
message(STATUS "Native AMR, FVM and solver clients use public PLIC backend boundaries")
