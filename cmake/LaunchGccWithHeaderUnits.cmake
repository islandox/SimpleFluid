# CMake does not yet add header-unit BMIs to GCC's generated module mapper.
# Add the prebuilt wrapper mappings immediately before invoking the compiler.

if(NOT DEFINED SIMPLEFLUID_HEADER_UNIT_MAP)
    message(FATAL_ERROR "SIMPLEFLUID_HEADER_UNIT_MAP is required")
endif()

set(_command_start -1)
math(EXPR _last_argument "${CMAKE_ARGC} - 1")
foreach(_index RANGE 0 ${_last_argument})
    if(CMAKE_ARGV${_index} STREQUAL "--")
        math(EXPR _command_start "${_index} + 1")
        break()
    endif()
endforeach()

if(_command_start LESS 0)
    message(FATAL_ERROR "Missing compiler command after --")
endif()

file(READ "${SIMPLEFLUID_HEADER_UNIT_MAP}" _header_unit_mappings)
string(REPLACE "\n" ";" _header_unit_mappings "${_header_unit_mappings}")

set(_compiler_command)
set(_uses_augmented_mapper OFF)
set(_expect_dependency_file OFF)
unset(_dependency_file)
foreach(_index RANGE ${_command_start} ${_last_argument})
    set(_argument "${CMAKE_ARGV${_index}}")
    if(_expect_dependency_file)
        set(_dependency_file "${_argument}")
        set(_expect_dependency_file OFF)
    elseif(_argument STREQUAL "-MF")
        set(_expect_dependency_file ON)
    elseif(_argument MATCHES "^-MF(.+)$")
        set(_dependency_file "${CMAKE_MATCH_1}")
    endif()
    if(_argument MATCHES "^-fmodule-mapper=(.+)$")
        set(_module_mapper "${CMAKE_MATCH_1}")
        file(READ "${_module_mapper}" _module_mapper_contents)
        set(_augmented_mapper_contents "${_module_mapper_contents}")
        foreach(_mapping IN LISTS _header_unit_mappings)
            if(_mapping STREQUAL "")
                continue()
            endif()
            string(FIND "${_augmented_mapper_contents}"
                "${_mapping}" _mapping_index)
            if(_mapping_index EQUAL -1)
                string(APPEND _augmented_mapper_contents "${_mapping}\n")
            endif()
        endforeach()

        # The mapper is a CMake dyndep output. Mutating it after generation
        # makes Ninja regenerate it and rebuild every consumer on the next
        # invocation. Compile against a stable sidecar instead.
        set(_augmented_mapper "${_module_mapper}.simplefluid")
        set(_write_augmented_mapper ON)
        if(EXISTS "${_augmented_mapper}")
            file(READ "${_augmented_mapper}" _existing_augmented_contents)
            if("${_existing_augmented_contents}" STREQUAL
               "${_augmented_mapper_contents}")
                set(_write_augmented_mapper OFF)
            endif()
        endif()
        if(_write_augmented_mapper)
            file(WRITE "${_augmented_mapper}"
                "${_augmented_mapper_contents}")
        endif()
        set(_argument "-fmodule-mapper=${_augmented_mapper}")
        set(_uses_augmented_mapper ON)
    endif()
    list(APPEND _compiler_command "${_argument}")
endforeach()

# CMake copies provider compile options into synthetic BMI-only targets. A
# -fmodule-only invocation is never linked or executed and therefore cannot
# emit the .gcda file required by GCC's strict PGO completeness diagnostics.
# The provider's real object compilation retains these diagnostics.
list(FIND _compiler_command "-fmodule-only" _module_only_index)
if(NOT _module_only_index EQUAL -1)
    list(REMOVE_ITEM _compiler_command
        "-Werror=missing-profile"
        "-Werror=coverage-mismatch")
endif()

execute_process(COMMAND ${_compiler_command} RESULT_VARIABLE _compiler_result)
if(NOT _compiler_result EQUAL 0)
    message(FATAL_ERROR "Compiler exited with status ${_compiler_result}")
endif()

# GCC lists the mapper passed to the compiler in its depfile. The sidecar is an
# implementation detail; CMake's original mapper is already an explicit Ninja
# dependency. Keep the sidecar out of Ninja's implicit dependency database so
# creating it during the first compile does not trigger a settling rebuild.
if(_uses_augmented_mapper AND DEFINED _dependency_file
   AND EXISTS "${_dependency_file}")
    file(READ "${_dependency_file}" _dependency_contents)
    string(REPLACE "${_augmented_mapper}" "${_module_mapper}"
        _normalized_dependency_contents "${_dependency_contents}")
    if(NOT "${_normalized_dependency_contents}" STREQUAL
       "${_dependency_contents}")
        file(WRITE "${_dependency_file}"
            "${_normalized_dependency_contents}")
    endif()
endif()
