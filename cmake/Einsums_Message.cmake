#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_to_string var)
  set(_var "")

  foreach(_arg ${ARGN})
    string(REPLACE "\\" "/" _arg ${_arg})
    if("${_var}" STREQUAL "")
      set(_var "${_arg}")
    else()
      set(_var "${_var} ${_arg}")
    endif()
  endforeach()

  set(${var}
      ${_var}
      PARENT_SCOPE
  )
endfunction()

function(einsums_info)
  set(msg)
  einsums_to_string(msg ${ARGN})
  message(STATUS "${msg}")
endfunction()

function(einsums_debug)
  if("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(msg "DEBUG:")
    einsums_to_string(msg ${ARGN})
    message(STATUS "${msg}")
  endif()
endfunction()

function(einsums_warn)
  if("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    set(msg "WARNING:")
    einsums_to_string(msg ${ARGN})
    message(STATUS "${msg}")
  endif()
endfunction()

function(einsums_error)
  set(msg "ERROR:")
  einsums_to_string(msg ${ARGN})
  message(FATAL_ERROR "${msg}")
endfunction()

function(einsums_message level)
  if("${level}" MATCHES "ERROR|error|Error")
    einsums_error(${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    einsums_warning(${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    einsums_debug(${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    einsums_info(${ARGN})
  else()
    einsums_error("message" "\"${level}\" is not a einsums configuration logging level.")
  endif()
endfunction()

function(einsums_config_loglevel level return)
  set(${return}
      FALSE
      PARENT_SCOPE
  )

  if("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error" AND "${level}" MATCHES
                                                                 "ERROR|error|Error"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn" AND "${level}" MATCHES
                                                                  "WARN|warn|Warn"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug" AND "${level}" MATCHES
                                                                     "DEBUG|debug|Debug"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info" AND "${level}" MATCHES
                                                                  "INFO|info|Info"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  endif()
endfunction()

function(einsums_print_list level message list)
  einsums_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      einsums_message(${level} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      einsums_message(${level} "${message} is empty.")
    endif()
  endif()
endfunction()
