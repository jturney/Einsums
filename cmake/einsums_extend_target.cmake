#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_extend_target libname modulename)
    set(options CONFIG_FILES)
    set(one_value_args SOURCES_PREFIX SOURCES_PREFIX_FROM_TARGET FEATURE_INFO)
    set(multi_value_args
        CONDITION
        SOURCES
        HEADERS
        OBJECTS
        DEFINES
        PUBLIC_DEFINES
        OPTIONS
        PUBLIC_OPTIONS
        INCLUDES
        PUBLIC_INCLUDES
        DEPENDENCIES
        PUBLIC_DEPENDENCIES
        MODULE_DEPENDENCIES
        CMAKE_SUBDIRS
        EXCLUDE_FROM_GLOBAL_HEADER
    )
    cmake_parse_arguments(${modulename} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    if(${modulename}_UNPARSED_ARGUMENTS)
        message(AUTHOR_WARNING "Arguments were not used by the module: ${${modulename}_UNPARSED_ARGUMENTS}")
    endif()

    if(NOT ${modulename}_CONDITION)
        set(${modulename}_CONDITION ON)
    endif()

    if(${${modulename}_CONDITION})
        set(${modulename}_ENABLED ON)
    else()
        set(${modulename}_ENABLED OFF)
    endif()

    if(NOT ${modulename}_ENABLED)
        einsums_debug("Feature ${${modulename}_FEATURE_INFO} is disabled.")
        return()
    endif()

    # Main directories of the module
    set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set(HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

    if(${modulename}_SOURCES_PREFIX_FROM_TARGET)
        if(NOT TARAGET ${${modulename}_SOURCES_PREFIX_FROM_TARGET})
            einsums_error("SOURCES_PREFIX_FROM_TARGET was given but referenced target does not exist.")
        else()
            get_target_property(${modulename}_SOURCES_PREFIX ${${modulename}_SOURCES_PREFIX_FROM_TARGET} SOURCES_DIR)
        endif()
    endif()

    if(${${modulename}_SOURCES_PREFIX})
        set(SOURCE_ROOT "${SOURCE_ROOT}/${${modulename}_SOURCES_PREFIX}")
    endif()

    einsums_debug("Extend module ${modulename}: SOURCE_ROOT: ${SOURCE_ROOT}")
    einsums_debug("Extend module ${modulename}: HEADER_ROOT: ${HEADER_ROOT}")

    # Write full path for the sources files
    list(TRANSFORM ${modulename}_SOURCES PREPEND ${SOURCE_ROOT}/ OUTPUT_VARIABLE sources)
    list(TRANSFORM ${modulename}_HEADERS PREPEND ${HEADER_ROOT}/ OUTPUT_VARIABLE headers)

    target_sources(einsums_${modulename} PRIVATE ${sources})

    target_compile_definitions(
        einsums_${modulename}
        PRIVATE ${${modulename}_DEFINES}
        PUBLIC ${${modulename}_PUBLIC_DEFINES}
    )

    target_compile_options(
        einsums_${modulename}
        PRIVATE ${${modulename}_OPTIONS}
        PUBLIC ${${modulename}_PUBLIC_OPTIONS}
    )

    target_include_directories(
        einsums_${modulename}
        PRIVATE ${${modulename}_INCLUDES}
        PUBLIC ${${modulename}_PUBLIC_INCLUDES}
    )

    if(EINSUMS_WITH_CHECK_MODULE_DEPENDENCIES)
        # verify that all dependencies are from the same module category
        foreach(dep ${${modulename}_MODULE_DEPENDENCIES})
            # consider only module dependencies, not other targets
            string(FIND ${dep} "einsums_" find_index)
            if(${find_index} EQUAL 0)
                string(SUBSTRING ${dep} 5 -1 dep) # cut off leading "einsums_"
                list(FIND _${libname}_modules ${dep} dep_index)
                if(${dep_index} EQUAL -1)
                    einsums_error(
                        "The module ${dep} should not be be listed in MODULE_DEPENDENCIES "
                        "for module einsums_${modulename}"
                    )
                endif()
            endif()
        endforeach()
    endif()

    target_link_libraries(einsums_${modulename} ${module_public_keyword} ${${modulename}_MODULE_DEPENDENCIES})
    target_link_libraries(einsums_${modulename} ${module_public_keyword} ${${modulename}_PUBLIC_DEPENDENCIES})
    target_link_libraries(einsums_${modulename} PRIVATE ${${modulename}_DEPENDENCIES})

    foreach(dir ${${modulename}_CMAKE_SUBDIRS})
        add_subdirectory(${dir})
    endforeach(dir)

endfunction()
