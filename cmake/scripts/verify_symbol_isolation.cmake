#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
#
# Assert that the built libEinsums actually carries the symbol isolation
# Einsums_SymbolIsolation.cmake asked for.
#
# Every flag there sits behind check_linker_flag, so the whole thing fails OPEN:
# a toolchain that stops accepting a flag, a linker swap, or a refactor that
# drops the call would each leave the library unprotected with nothing going
# red. This is the test that notices.
#
# Run as: cmake -DLIBRARY=<path> -DABI_VERSION=<n> -DREADELF=<path> -P <this>
#
# Skips (rather than fails) where the check does not apply: non-ELF platforms
# have no flat symbol namespace to protect, and no readelf means no evidence
# either way, which is not the same as evidence of a problem.

if(NOT DEFINED LIBRARY OR NOT EXISTS "${LIBRARY}")
  message(FATAL_ERROR "verify_symbol_isolation: LIBRARY not set or missing: '${LIBRARY}'")
endif()
if(NOT DEFINED ABI_VERSION)
  message(FATAL_ERROR "verify_symbol_isolation: ABI_VERSION not set")
endif()

if(NOT READELF OR NOT EXISTS "${READELF}")
  message(STATUS "SKIP: no readelf available to inspect ${LIBRARY}")
  return()
endif()

file(
  READ "${LIBRARY}" _magic
  LIMIT 4
  HEX
)
if(NOT _magic STREQUAL "7f454c46") # \x7fELF
  message(STATUS "SKIP: ${LIBRARY} is not ELF; Mach-O and PE bind per-module already")
  return()
endif()

execute_process(
  COMMAND "${READELF}" -W --dyn-syms "${LIBRARY}"
  OUTPUT_VARIABLE _syms
  ERROR_VARIABLE _err
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "readelf failed on ${LIBRARY}: ${_err}")
endif()

set(_version_name "EINSUMS_${ABI_VERSION}")

# Symbols that are legitimately unversioned and unmangled: the version definition itself, and the
# linker's own bookkeeping.
set(_allowed_bare "${_version_name}" "_init" "_fini" "_edata" "_end" "__bss_start")

string(REPLACE "\n" ";" _lines "${_syms}")

set(_unversioned "")
set(_unmangled "")
set(_foreign "")
set(_defined_count 0)

foreach(_line IN LISTS _lines)
  # readelf -W --dyn-syms columns: Num: Value Size Type Bind Vis Ndx Name
  if(NOT _line MATCHES
     "^ *[0-9]+: +[0-9A-Fa-f]+ +[0-9]+ +([A-Z]+) +[A-Z]+ +[A-Z]+ +([A-Za-z0-9]+) +([^ ]+)$"
  )
    continue()
  endif()
  set(_type "${CMAKE_MATCH_1}")
  set(_ndx "${CMAKE_MATCH_2}")
  set(_name "${CMAKE_MATCH_3}")

  # UND entries are what we IMPORT. Isolation is about what we EXPORT, and a versioned import is a
  # property of the library we import from.
  if(_ndx STREQUAL "UND")
    continue()
  endif()
  # Only real code and data. FILE/SECTION/TLS entries carry no linkage meaning for this check.
  if(NOT _type STREQUAL "FUNC" AND NOT _type STREQUAL "OBJECT")
    continue()
  endif()

  math(EXPR _defined_count "${_defined_count} + 1")

  string(REGEX REPLACE "@@?.*$" "" _bare "${_name}")
  if(_bare IN_LIST _allowed_bare)
    continue()
  endif()

  # 1. Versioned, so a second copy in the global scope cannot answer for it.
  if(NOT _name MATCHES "@@?${_version_name}$")
    list(APPEND _unversioned "${_bare}")
  endif()

  # 1. Mangled, i.e. it has a C++ namespace to live in. An unmangled export is a C-linkage symbol,
  #   which no namespace scheme can ever protect, so it has to be absent rather than merely
  #   namespaced.
  if(NOT _bare MATCHES "^_Z")
    list(APPEND _unmangled "${_bare}")
  endif()

  # 1. Not from a third-party namespace folded in from a static archive. These carry someone else's
  #   name, so two worlds absorbing different vintages interpose on each other. `--exclude-libs,ALL`
  #   is what drops them.
  #
  # Matched on the MANGLED prefix (`_ZN4hptt` is `hptt::`), because a substring match on the
  # readable name catches our own symbols: after einsums::blas::vendor::sdirprod_kernel was
  # namespaced, a search for "dirprod_kernel" still matched it and reported a leak that was not one.
  foreach(_ns "_ZN4hptt" "_ZN6spdlog" "_ZN3fmt" "_ZN9Catch")
    if(_bare MATCHES "^${_ns}")
      list(APPEND _foreign "${_bare}")
    endif()
  endforeach()
endforeach()

if(_defined_count EQUAL 0)
  message(FATAL_ERROR "verify_symbol_isolation: parsed no defined symbols from ${LIBRARY}; "
                      "the readelf output format is not what this script expects"
  )
endif()

# Built as a plain string, appended to. A list would be joined on ";", which both merges the sample
# names into one word and eats the semicolons in the prose around them.
set(_msg "")

# Report each sample list joined explicitly; CMake list semantics do not survive being dropped into
# a message.
macro(_einsums_report_problem _items _headline _detail)
  list(LENGTH ${_items} _n)
  list(SUBLIST ${_items} 0 5 _sample)
  list(JOIN _sample ", " _sample_str)
  string(APPEND _msg "\n  ${_n} ${_headline}, e.g. ${_sample_str}\n    ${_detail}\n")
endmacro()

if(_unversioned)
  _einsums_report_problem(
    _unversioned
    "exported symbol(s) are not versioned '${_version_name}'"
    "The linker version script was not applied, so another copy of libEinsums in this process can interpose on these."
  )
endif()

if(_unmangled)
  _einsums_report_problem(
    _unmangled "exported symbol(s) have C linkage"
    "A C symbol has no namespace to be isolated by, so it must not be exported at all."
  )
endif()

if(_foreign)
  _einsums_report_problem(
    _foreign "exported symbol(s) come from an absorbed static archive"
    "Expected --exclude-libs,ALL to drop these."
  )
endif()

if(NOT _msg STREQUAL "")
  message(
    FATAL_ERROR
      "${LIBRARY} is not symbol-isolated.${_msg}\n"
      "See cmake/Einsums_SymbolIsolation.cmake and the reproduction in "
      "libs/Einsums/Config/tests/unit/two_worlds."
  )
endif()

message(
  STATUS
    "OK: ${_defined_count} exported symbols, all versioned '${_version_name}', all C++-mangled, "
    "none from an absorbed archive"
)
