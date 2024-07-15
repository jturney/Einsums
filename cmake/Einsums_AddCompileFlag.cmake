#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeParseArguments)
include(CheckCXXCompilerFlag)

function(einsums_add_target_compile_option FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(
    target_compile_option "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(target_compile_option_PUBLIC)
    set(_dest einsums_public_flags)
  else()
    set(_dest einsums_private_flags)
  endif()

  set(_configurations "none")
  if(target_compile_option_CONFIGURATIONS)
    set(_configurations ${target_compile_option_CONFIGURATIONS})
  endif()

  set(_languages "CXX")
  if(target_compile_option_LANGUAGES)
    set(_languages ${target_compile_option_LANGUAGES})
  endif()

  foreach(_lang ${_languages})
    foreach(_config ${_configurations})
      if(NOT _config STREQUAL "none")
        set(_config "$<$<AND:$<CONFIG:${_config}>,$<COMPILE_LANGUAGE:${_lang}>>:${FLAG}>")
      else()
        set(_config "$<$<COMPILE_LANGUAGE:${_lang}>:${FLAG}>")
      endif()
      target_compile_options(${_dest} INTERFACE "${_conf}")
    endforeach()
  endforeach()
endfunction()

function(einsums_add_target_compile_option_if_available FLAG)
  set(options PUBLIC)
  set(one_value_args NAME)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(
    target_compile_option_ia "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(target_compile_option_ia_PUBLIC)
    set(_modifier PUBLIC)
  else()
    set(_modifier PRIVATE)
  endif()

  if(target_compile_option_ia_NAME)
    string(TOUPPER ${target_compile_option_ia_NAME} _name)
  else()
    string(TOUPPER ${FLAG} _name)
  endif()

  string(REGEX REPLACE "^-+" "" _name ${_name})
  string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
  string(REGEX REPLACE "," "_" _name ${_name})
  string(REGEX REPLACE "\\+" "X" _name ${_name})

  set(_languages "CXX")
  if(target_compile_option_ia_LANGUAGES)
    set(_languages ${target_compile_option_ia_LANGUAGES})
  endif()

  foreach(_lang ${_languages})
    if(_lang STREQUAL "CXX")
      check_cxx_compiler_flag(${FLAG} EINSUMS_WITH_${_lang}_FLAG_${_name})
    else()
      einsums_error("Unsupported language: ${_lang}.")
    endif()
    if(EINSUMS_WITH_${_lang}_FLAG_${_name})
      einsums_add_target_compile_option(
        ${FLAG} ${_modifier}
        CONFIGURATIONS ${target_compile_option_ia_CONFIGURATIONS}
        LANGUAGES ${_lang}
      )
    else()
      einsums_info("\"${FLAG}\" not available for language ${_lang}.")
    endif()
  endforeach()
endfunction()

function(einsums_remove_target_compile_option FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS)
  cmake_parse_arguments(
    EINSUMS_ADD_TARGET_COMPILE_OPTION "${options}" "${one_value_args}" "${multi_value_args}"
    ${ARGN}
  )

  set(_configurations "none")
  if(EINSUMS_ADD_TARGET_COMPILE_OPTION_CONFIGURATIONS)
    set(_configurations "${EINSUMS_ADD_TARGET_COMPILE_OPTION_CONFIGURATIONS}")
  endif()

  get_property(
    EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC
  )
  get_property(
    EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR GLOBAL
    PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE
  )
  set_property(GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC "")
  set_property(GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE "")

  foreach(_config ${_configurations})
    set(_conf "${FLAG}")
    if(NOT _config STREQUAL "none")
      set(_conf "$<$<CONFIG:${_config}>:${FLAG}>")
    endif()
    foreach(_flag ${EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR})
      if(NOT ${_flag} STREQUAL ${_conf})
        set_property(GLOBAL APPEND PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC "${_flag}")
      endif()
    endforeach()
    foreach(_flag ${EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR})
      if(NOT ${_flag} STREQUAL ${_conf})
        set_property(GLOBAL APPEND PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE "${_flag}")
      endif()
    endforeach()
  endforeach()
endfunction()

function(einsums_add_target_compile_definition FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS)
  cmake_parse_arguments(
    EINSUMS_ADD_TARGET_COMPILE_DEFINITION "${options}" "${one_value_args}" "${multi_value_args}"
    ${ARGN}
  )

  if(EINSUMS_ADD_TARGET_COMPILE_DEFINITION_PUBLIC)
    set(_dest einsums_public_flags)
  else()
    set(_dest einsums_private_flags)
  endif()

  set(_configurations "none")
  if(EINSUMS_ADD_TARGET_COMPILE_DEFINITION_CONFIGURATIONS)
    set(_configurations "${EINSUMS_ADD_TARGET_COMPILE_DEFINITION_CONFIGURATIONS}")
  endif()

  foreach(_config ${_configurations})
    set(_conf "${FLAG}")
    if(NOT _config STREQUAL "none")
      set(_conf "$<$<CONFIG:${_config}>:${FLAG}>")
    endif()
    target_compile_definitions(${_dest} INTERFACE "${_conf}")
  endforeach()
endfunction()

function(einsums_add_compile_flag)
  set(one_value_args)
  einsums_add_target_compile_option(${ARGN})
endfunction()

function(einsums_add_compile_flag_if_available FLAG)
  set(options PUBLIC)
  set(one_value_args NAME)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(
    EINSUMS_ADD_COMPILE_FLAG_IA "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  set(_public)
  if(EINSUMS_ADD_COMPILE_FLAG_IA_PUBLIC)
    set(_public PUBLIC)
  endif()

  if(EINSUMS_ADD_COMPILE_FLAG_IA_NAME)
    string(TOUPPER ${EINSUMS_ADD_COMPILE_FLAG_IA_NAME} _name)
  else()
    string(TOUPPER ${FLAG} _name)
  endif()

  string(REGEX REPLACE "^-+" "" _name ${_name})
  string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
  string(REGEX REPLACE "," "_" _name ${_name})
  string(REGEX REPLACE "\\+" "X" _name ${_name})

  set(_languages "CXX")
  if(EINSUMS_ADD_COMPILE_FLAG_IA_LANGUAGES)
    set(_languages ${EINSUMS_ADD_COMPILE_FLAG_IA_LANGUAGES})
  endif()

  foreach(_lang ${_languages})
    if(_lang STREQUAL "CXX")
      check_cxx_compiler_flag(${FLAG} EINSUMS_WITH_${_lang}_FLAG_${_name})
    else()
      einsums_error("Unsupported language ${_lang}.")
    endif()
    if(EINSUMS_WITH_${_lang}_FLAG_${_name})
      einsums_add_compile_flag(
        ${FLAG} CONFIGURATIONS ${EINSUMS_ADD_COMPILE_FLAG_IA_CONFIGURATIONS} LANGUAGES ${_lang}
        ${_public}
      )
    else()
      einsums_info("\"${FLAG}\" not available for language ${_lang}.")
    endif()
  endforeach()
endfunction()
