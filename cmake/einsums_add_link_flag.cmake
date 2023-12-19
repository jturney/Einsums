#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeParseArguments)
include(CheckCXXCompilerFlag)

macro(einsums_add_link_flag FLAG)
    set(options PUBLIC)
    set(one_value_args)
    set(multi_value_args TARGETS CONFIGURATIONS)
    cmake_parse_arguments(EINSUMS_ADD_LINK_FLAG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(EINSUMS_ADD_LINK_FLAG_PUBLIC)
        set(_dest einsums_public_flags)
    else()
        set(_dest einsums_private_flags)
    endif()

    set(_targets "none")
    if(EINSUMS_ADD_LINK_FLAG_TARGETS)
        set(_targets ${EINSUMS_ADD_LINK_FLAG_TARGETS})
    endif()

    set(_configurations "none")
    if(EINSUMS_ADD_LINK_FLAG_CONFIGURATIONS)
        set(_configurations "${EINSUMS_ADD_LINK_FLAG_CONFIGURATIONS}")
    endif()

    foreach(_config ${_configurations})
        foreach(_target ${_targets})
            if(NOT _config STREQUAL "none" AND NOT _target STREQUAL "none")
                set(_flag "$<$<AND:$<CONFIG:${_config}>,$<STREQUAL:$<TARGET_PROPERTY:TYPE>,${_target}>:${FLAG}>")
            elseif(_config STREQUAL "none" AND NOT _target STREQUAL "none")
                set(_flag "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,${_target}>:${FLAG}>")
            elseif(NOT _config STREQUAL "none" AND _target STREQUAL "none")
                set(_flag "$<$<CONFIG:${_config}>:${FLAG}>")
            else()
                set(_flag "${FLAG}")
            endif()
            target_link_options(${_dest} INTERFACE "${_flag}")
        endforeach()
    endforeach()
endmacro()

macro(einsums_add_link_flag_if_available FLAG)
    set(options PUBLIC)
    set(one_value_args NAME)
    set(multi_value_args TARGETS)
    cmake_parse_arguments(EINSUMS_ADD_LINK_FLAG_IA "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(_public)
    if(EINSUMS_ADD_LINK_FLAG_IA_PUBLIC)
        set(_public PUBLIC)
    endif()

    if(EINSUMS_ADD_LINK_FLAG_IA_NAME)
        string(TOUPPER ${EINSUMS_ADD_LINK_FLAG_IA_NAME} _name)
    else()
        string(TOUPPER ${FLAG} _name)
    endif()

    string(REGEX REPLACE " " "" _name ${_name})
    string(REGEX REPLACE "^-+" "" _name ${_name})
    string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
    string(REGEX REPLACE "," "_" _name ${_name})
    string(REGEX REPLACE "\\+" "X" _name ${_name})

    check_cxx_compiler_flag("${FLAG}" WITH_LINKER_FLAG_${_name})
    if(WITH_LINKER_FLAG_${_name})
        einsums_add_link_flag(${FLAG} TARGETS ${EINSUMS_ADD_LINK_FLAG_IA_TARGETS} ${_public})
    else()
        einsums_info("Linker \"${FLAG}\" not available.")
    endif()

endmacro()
