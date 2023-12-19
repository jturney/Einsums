#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_add_config_define definition)

    # if(ARGN) ignores an argument "0"
    set(Args ${ARGN})
    list(LENGTH Args ArgsLen)
    if(ArgsLen GREATER 0)
        set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_DEFINITIONS "${definition} ${ARGN}")
    else()
        set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_DEFINITIONS "${definition}")
    endif()

endfunction()

function(einsums_add_config_cond_define definition)

    # if(ARGN) ignores an argument "0"
    set(Args ${ARGN})
    list(LENGTH Args ArgsLen)
    if(ArgsLen GREATER 0)
        set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS "${definition} ${ARGN}")
    else()
        set_property(GLOBAL APPEND PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS "${definition}")
    endif()

endfunction()

# ---------------------------------------------------------------------
# Function to add config defines to a list that depends on a namespace variable #defines that match the namespace can
# later be written out to a file
# ---------------------------------------------------------------------
function(einsums_add_config_define_namespace)
    set(options)
    set(one_value_args DEFINE NAMESPACE)
    set(multi_value_args VALUE)
    cmake_parse_arguments(OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(DEF_VAR EINSUMS_CONFIG_DEFINITIONS_${OPTION_NAMESPACE})

    # to avoid extra trailing spaces (no value), use an if check
    if(OPTION_VALUE)
        set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE} ${OPTION_VALUE}")
    else()
        set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE}")
    endif()

endfunction()

function(einsums_add_config_cond_define_namespace)
    set(one_value_args DEFINE NAMESPACE)
    set(multi_value_args VALUE)
    cmake_parse_arguments(OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(DEF_VAR EINSUMS_CONFIG_COND_DEFINITIONS_${OPTION_NAMESPACE})

    # to avoid extra trailing spaces (no value), use an if check
    if(OPTION_VALUE)
        set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE} ${OPTION_VALUE}")
    else()
        set_property(GLOBAL APPEND PROPERTY ${DEF_VAR} "${OPTION_DEFINE}")
    endif()

endfunction()

# ---------------------------------------------------------------------
# Function to write variables out from a global var that was set using the config_define functions into a config file
# ---------------------------------------------------------------------
function(einsums_write_config_defines_file)
    set(options)
    set(one_value_args TEMPLATE NAMESPACE FILENAME)
    set(multi_value_args)
    cmake_parse_arguments(OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(${OPTION_NAMESPACE} STREQUAL "default")
        get_property(DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_DEFINITIONS)
        get_property(COND_DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS)
    else()
        get_property(DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_DEFINITIONS_${OPTION_NAMESPACE})
        get_property(COND_DEFINITIONS_VAR GLOBAL PROPERTY EINSUMS_CONFIG_COND_DEFINITIONS_${OPTION_NAMESPACE})
    endif()

    if(DEFINED DEFINITIONS_VAR)
        list(SORT DEFINITIONS_VAR)
        list(REMOVE_DUPLICATES DEFINITIONS_VAR)
    endif()

    set(einsums_config_defines "\n")
    foreach(def ${DEFINITIONS_VAR})
        set(einsums_config_defines "${einsums_config_defines}#define ${def}\n")
    endforeach()

    if(DEFINED COND_DEFINITIONS_VAR)
        list(SORT COND_DEFINITIONS_VAR)
        list(REMOVE_DUPLICATES COND_DEFINITIONS_VAR)
        set(einsums_config_defines "${einsums_config_defines}\n")
    endif()
    foreach(def ${COND_DEFINITIONS_VAR})
        string(FIND ${def} " " _pos)
        if(NOT ${_pos} EQUAL -1)
            string(SUBSTRING ${def} 0 ${_pos} defname)
        else()
            set(defname ${def})
            string(STRIP ${defname} defname)
        endif()

        # C++20 specific variable
        string(FIND ${def} "HAVE_CXX20" _pos)
        if(NOT ${_pos} EQUAL -1)
            set(einsums_config_defines
                "${einsums_config_defines}#if __cplusplus >= 202002 && !defined(${defname})\n#define ${def}\n#endif\n"
            )
        else()
            # C++17 specific variable
            string(FIND ${def} "HAVE_CXX17" _pos)
            if(NOT ${_pos} EQUAL -1)
                set(einsums_config_defines
                    "${einsums_config_defines}#if __cplusplus >= 201500 && !defined(${defname})\n#define ${def}\n#endif\n"
                )
            else()
                set(einsums_config_defines
                    "${einsums_config_defines}#if !defined(${defname})\n#define ${def}\n#endif\n"
                )
            endif()
        endif()
    endforeach()

    # if the user has not specified a template, generate a proper header file
    if(NOT OPTION_TEMPLATE)
        string(TOUPPER ${OPTION_NAMESPACE} NAMESPACE_UPPER)
        set(PREAMBLE
            "//----------------------------------------------------------------------------------------------\n"
            "// Copyright (c) The Einsums Developers. All rights reserved.\n"
            "// Licensed under the MIT License. See LICENSE.txt in the project root for license information.\n"
            "//----------------------------------------------------------------------------------------------\n"
            "\n"
            "// Do not edit this file! It has been generated by the cmake configuration step.\n"
            "\n"
            "#pragma once"
        )
        set(TEMP_FILENAME "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${NAMESPACE_UPPER}")
        file(WRITE ${TEMP_FILENAME} ${PREAMBLE} ${einsums_config_defines} "\n")
        configure_file("${TEMP_FILENAME}" "${OPTION_FILENAME}" COPYONLY)
        file(REMOVE "${TEMP_FILENAME}")
    else()
        configure_file("${OPTION_TEMPLATE}" "${OPTION_FILENAME}" @ONLY)
    endif()
endfunction()
