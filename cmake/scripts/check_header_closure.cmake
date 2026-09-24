#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Fail if a probe translation unit reaches a forbidden header.
#
# The probe is an object target compiled with the compiler's include listing on (-H, or
# /showIncludes for an MSVC-style driver), so every header the compiler opens, through any chain of
# includes and with the build's own flags, appears in the build output. This script rebuilds the
# probe and matches that listing against FORBID.
#
# The object is deleted first. An up-to-date object would build with no output, and a rerun after a
# failure would then pass without having checked anything.
#
# Inputs:
#   OBJECT     the probe's object file, deleted before the build
#   BUILD_DIR  the build tree
#   TARGET     the probe target
#   FORBID     regular expression a header path must not match
#   EXPECT     regular expression the listing must match, proving the compiler reported its headers
#   CONFIG     build configuration, for multi-config generators

foreach(var OBJECT BUILD_DIR TARGET FORBID EXPECT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "check_header_closure: ${var} is required")
  endif()
endforeach()

file(REMOVE "${OBJECT}")

set(config_args)
if(DEFINED CONFIG AND NOT CONFIG STREQUAL "")
  set(config_args --config "${CONFIG}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${TARGET}" ${config_args}
  OUTPUT_VARIABLE build_out
  ERROR_VARIABLE build_err
  RESULT_VARIABLE build_rc
)
set(listing "${build_out}\n${build_err}")

if(NOT build_rc EQUAL 0)
  message(FATAL_ERROR "check_header_closure: building ${TARGET} failed\n${listing}")
endif()

if(NOT listing MATCHES "${EXPECT}")
  message(
    FATAL_ERROR
      "check_header_closure: the build of ${TARGET} listed no headers matching '${EXPECT}', so the compiler did not report "
      "what it included and this check would pass without looking. Is the include listing flag on?\n${listing}"
  )
endif()

string(REGEX MATCHALL "[^\n]*${FORBID}[^\n]*" hits "${listing}")
if(hits)
  list(REMOVE_DUPLICATES hits)
  list(JOIN hits "\n" hit_lines)
  # FORBIDDEN_HEADER_REACHED is what the self-test matches: one token, which CMake's message
  # wrapping cannot split the way it splits a phrase.
  message(FATAL_ERROR "check_header_closure: FORBIDDEN_HEADER_REACHED ${TARGET} reaches headers matching '${FORBID}':\n${hit_lines}")
endif()

message(STATUS "check_header_closure: ${TARGET} reaches no header matching '${FORBID}'")
