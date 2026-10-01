#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Script mode (cmake -P), run by the test einsums_add_simd_rung_objects_test()
# registers. Fails when an object compiled for one SIMD dispatch rung defines a
# weak symbol outside that rung's namespace.
#
# Each per-rung copy of an implementation file is compiled at its own flags. A
# weak definition it emits under a name another copy also emits (a std::vector
# member, an fmt formatter, a logging helper) is merged at link time: the
# linker keeps one copy and every caller, baseline code included, runs it. If
# it keeps the v4 copy, a CPU without AVX-512 dies with SIGILL. A weak symbol is
# safe only when its name is the rung's own, because it is inside the rung's
# arch_<rung> namespace or the SIMD headers' isa_<features> namespace (whose
# names differ wherever the code does), or when it holds no code that depends
# on the flags (the DW.ref personality pointers).
#
# Inputs:
#   NM       the nm to run
#   OBJECTS  the target's object files, joined with '|'

if(NOT NM OR NOT OBJECTS)
  message(FATAL_ERROR "Einsums_CheckRungObjects.cmake needs -DNM=<nm> and -DOBJECTS=<a|b|...>")
endif()

string(REPLACE "|" ";" _objects "${OBJECTS}")

set(_checked 0)
set(_offenders "")
foreach(_object IN LISTS _objects)
  # The dispatch helper names each copy <stem>_<rung>.cpp in a simd_dispatch directory.
  if(NOT _object MATCHES "simd_dispatch/[^/]*_(baseline|v2|v3|v4|sme|native)\\.cpp\\.(o|obj)$")
    continue()
  endif()
  set(_rung "${CMAKE_MATCH_1}")
  math(EXPR _checked "${_checked} + 1")

  execute_process(
    COMMAND "${NM}" --defined-only "${_object}"
    OUTPUT_VARIABLE _symbols
    RESULT_VARIABLE _status
    ERROR_VARIABLE _error
  )
  if(NOT _status EQUAL 0)
    message(FATAL_ERROR "${NM} failed on ${_object}: ${_error}")
  endif()

  string(REPLACE "\n" ";" _lines "${_symbols}")
  foreach(_line IN LISTS _lines)
    # nm prints "<address> <type> <name>"; W and V are weak, u is GNU unique.
    if(NOT _line MATCHES "^[0-9a-fA-F]* *([WVu]) (.+)$")
      continue()
    endif()
    set(_name "${CMAKE_MATCH_2}")
    # Itanium mangling spells a namespace as its length and name, as in 7arch_v4 or 33isa_avx_...
    if(_name MATCHES "[0-9]arch_${_rung}" OR _name MATCHES "[0-9]isa_" OR _name MATCHES "^DW\\.ref\\.")
      continue()
    endif()
    list(APPEND _offenders "${_object}: ${_name}")
  endforeach()
endforeach()

if(_checked EQUAL 0)
  message(FATAL_ERROR "No per-rung objects found among ${OBJECTS}; the check would pass vacuously.")
endif()

if(_offenders)
  list(LENGTH _offenders _count)
  list(JOIN _offenders "\n  " _report)
  message(
    FATAL_ERROR
      "${_count} weak symbol(s) defined by a per-rung object outside its rung's namespace:\n  ${_report}\n"
      "Every rung's copy emits these under the same name, and the linker keeps one for all callers. "
      "Move the code that instantiates them out of the per-rung file (see src/TransposeKernels.hpp in HPTT)."
  )
endif()

message(STATUS "${_checked} per-rung objects define no weak symbols outside their rung")
