#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Name of test file. Include .cpp, .hip, etc. extension.
set(ConfigTests ABI.cpp)

foreach(test ${ConfigTests})
  # Extract base name of test Example1.cpp -> Example1, Example.2.hip -> Example
  get_filename_component(TestName ${test} NAME_WE)
  # Extract extension of the test Example1.cpp -> .cpp, Example.2.hip -> .2.hip
  get_filename_component(Extension ${test} EXT)

  # Remove the leading dot and replace any periods with underscores
  string(SUBSTRING ${Extension} 1 -1 EXT_NO_DOT) # Remove the leading dot
  string(REPLACE "." "_" TestExtension ${EXT_NO_DOT})

  # Check if the cleaned extension is "cu" or "hip"
  if(NOT EINSUMS_WITH_GPU_SUPPORT AND (TestExtension STREQUAL "cu" OR TestExtension STREQUAL "hip"))
    # Skip this iteration
    continue()
  endif()

  set(Sources ${test})

  source_group("Source Files" FILES ${Sources})

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${Sources} ${${TestName}_${TestExtension}_FLAGS}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.Config" ${TestName} ${${TestName}_${TestExtension}_PARAMETERS})
endforeach()

# Two copies of one library in one process: the loader-behaviour reproduction the sealed-worlds
# symbol isolation depends on. Self-contained, links no Einsums except in the test driver.
add_subdirectory(two_worlds)

# Assert the isolation was actually applied to the real libEinsums.
#
# Every flag in Einsums_SymbolIsolation.cmake sits behind check_linker_flag, so the whole thing
# fails OPEN: a toolchain that stops accepting a flag, a linker swap, or a refactor that drops the
# call would each leave the library unprotected with nothing going red. This is the test that
# notices.
#
# Skips itself on non-ELF platforms and where readelf is absent, so it is only evidence where there
# is something to be evidence about.
if(TARGET Einsums)
  find_program(EINSUMS_READELF NAMES readelf llvm-readelf eu-readelf)
  add_test(
    NAME Tests.Unit.Modules.Config.SymbolIsolation
    COMMAND
      ${CMAKE_COMMAND} -DLIBRARY=$<TARGET_FILE:Einsums> -DABI_VERSION=${EINSUMS_ABI_VERSION}
      -DREADELF=${EINSUMS_READELF} -P
      ${PROJECT_SOURCE_DIR}/cmake/scripts/verify_symbol_isolation.cmake
  )
  set_tests_properties(
    Tests.Unit.Modules.Config.SymbolIsolation PROPERTIES LABELS "UNIT_ONLY" SKIP_REGULAR_EXPRESSION
                                                         "SKIP:"
  )
endif()
