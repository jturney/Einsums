#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_add_source_group)
  if(MSVC)
    set(options)
    set(one_value_args NAME CLASS ROOT)
    set(multi_value_args TARGETS)
    cmake_parse_arguments(GROUP "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(name "")
    if(GROUP_NAME)
      set(name ${GROUP_NAME})
    endif()

    if(NOT GROUP_ROOT)
      set(GROUP_ROOT ".")
    endif()
    get_filename_component(root "${GROUP_ROOT}" ABSOLUTE)

    einsums_debug("add_source_group.${name}" "root: ${GROUP_ROOT}")

    foreach(target ${GROUP_TARGETS})
      string(REGEX REPLACE "${root}" "" relpath "${target}")
      set(_target ${relpath})
      string(REGEX REPLACE "[\\\\/][^\\\\/]*$" "" relpath "${relpath}")
      string(REGEX REPLACE "^[\\\\/]" "" relpath "${relpath}")
      string(REGEX REPLACE "/" "\\\\\\\\" relpath "${relpath}")

      if(GROUP_CLASS)
        einsums_debug(
          "add_source_group.${name}"
          "Adding '${target}' to source group '${GROUP_CLASS}', sub-group '${relpath}'"
        )
        source_group("${GROUP_CLASS}\\${relpath}" FILES ${target})
      else()
        einsums_debug("add_source_group.${name}" "Adding ${target} to source group ${relpath}")
        source_group("${relpath}" FILES ${target})
      endif()
    endforeach()
  endif()
endfunction()
