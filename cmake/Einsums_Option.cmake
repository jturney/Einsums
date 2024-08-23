#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeDependentOption)
include(CMakeParseArguments)

set(EINSUMS_OPTION_CATEGORIES "Generic" "Build Targets" "Profiling" "Debugging")

macro(einsums_option option type description default)
  set(options ADVANCED)
  set(one_value_args CATEGORY DEPENDS)
  set(multi_value_args STRINGS)
  cmake_parse_arguments(EINSUMS_OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if("${type}" STREQUAL "BOOL")
    # Use regular CMake options for booleans
    if(NOT CALIIBRI_OPTION_DEPENDS)
      option(${option} "${description}" ${default})
    else()
      cmake_dependent_option(${option} "${description}" ${default} "${EINSUMS_OPTION_DEPENDS}" OFF)
    endif()
  else()
    if(EINSUMS_OPTION_DEPENDS)
      message(FATAL_ERROR "einsums_option DEPENDS keyword can only be used with BOOL options")
    endif()
    # Use custom cache variables for other types
    if(NOT DEFINED ${option})
      set(${option}
          ${default}
          CACHE ${type} "${description}" FORCE)
    else()
      get_property(
        _option_is_cache_property
        CACHE "${option}"
        PROPERTY TYPE
        SET)
      if(NOT _option_is_cache_property)
        set(${option}
            ${default}
            CACHE ${type} "${description}" FORCE)
        if(EINSUMS_OPTION_ADVANCED)
          mark_as_advanced(${option})
        endif()
      else()
        set_property(CACHE "${option}" PROPERTY HELPSTRING "${description}")
        set_property(CACHE "${option}" PROPERTY TYPE ${type})
      endif()
    endif()

    if(EINSUMS_OPTION_STRINGS)
      if("${type}" STREQUAL "STRING")
        set_property(CACHE "${option}" PROPERTY STRINGS "${EINSUMS_OPTION_STRINGS}")
      else()
        message(FATAL_ERROR "einsums_option STRINGS keyword can only be used with STRING type")
      endif()
    endif()
  endif()

  if(EINSUMS_OPTION_ADVANCED)
    mark_as_advanced(${option})
  endif()

  set_property(GLOBAL APPEND PROPERTY EINSUMS_MODULE_CONFIG_EINSUMS ${option})

  set(_category "Generic")
  if(EINSUMS_OPTION_CATEGORY)
    set(_category "${EINSUMS_OPTION_CATEGORY}")
  endif()
  set(${option}Category
      ${_category}
      CACHE INTERNAL "")
endmacro()
