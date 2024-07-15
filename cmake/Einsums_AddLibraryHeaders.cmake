#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_add_library_headers name globtype)
  if(MSVC)
    set(options APPEND)
    set(one_value_args)
    set(multi_value_args EXCLUDE GLOBS)
    cmake_parse_arguments(HEADERS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT HEADERS_APPEND)
      set(${name}_HEADERS
          ""
          CACHE INTERNAL "Headers for lib${name}." FORCE
      )
    endif()

    file(${globtype} headers ${HEADERS_GLOBS})

    foreach(header ${headers})
      get_filename_component(absolute_path ${header} ABSOLUTE)

      set(add_flag ON)

      if(HEADERS_EXCLUDE)
        if(${absolute_path} MATCHES ${HEADERS_EXCLUDE})
          set(add_flag OFF)
        endif()
      endif()

      if(add_flag)
        einsums_debug(
          "add_library_headers.${name}" "Adding ${absolute_path} to header list for lib${name}"
        )
        set(${name}_HEADERS
            ${${name}_HEADERS} ${absolute_path}
            CACHE INTERNAL "Headers for lib${name}." FORCE
        )
      endif()
    endforeach()
  endif()
endfunction()

# ##################################################################################################
function(einsums_add_library_headers_noglob name)
  if(MSVC)
    set(options APPEND)
    set(one_value_args)
    set(multi_value_args EXCLUDE HEADERS)
    cmake_parse_arguments(HEADERS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    einsums_print_list(
      "DEBUG" "einsums_add_library_sources_noglob.${name}" "Sources for ${name}" HEADERS_HEADERS
    )

    set(headers ${HEADERS_HEADERS})

    if(NOT HEADERS_APPEND)
      set(${name}_HEADERS
          ""
          CACHE INTERNAL "Headers for lib${name}." FORCE
      )
    endif()

    foreach(header ${headers})
      get_filename_component(absolute_path ${header} ABSOLUTE)

      set(add_flag ON)

      if(HEADERS_EXCLUDE)
        if(${absolute_path} MATCHES ${HEADERS_EXCLUDE})
          set(add_flag OFF)
        endif()
      endif()

      if(add_flag)
        einsums_debug(
          "einsums_add_library_headers_noglob.${name}"
          "Adding ${absolute_path} to header list for lib${name}"
        )
        set(${name}_HEADERS
            ${${name}_HEADERS} ${absolute_path}
            CACHE INTERNAL "Headers for lib${name}." FORCE
        )
      endif()
    endforeach()
  endif()
endfunction()
