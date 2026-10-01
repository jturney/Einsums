#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Einsums' registrations of Stripes' rung tests (see Einsums_SetupStripes). Kernels are compiled per
# rung with stripes_add_dispatch_sources() directly; these wrappers only give the tests Einsums'
# names, command-line flags and per-test properties.

#:
#: .. cmake:command:: einsums_add_simd_rung_tests
#:
#:    Register ``<name>_test`` once more per dispatch rung, forcing the rung its dispatch tables
#:    select:
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_simd_rung_tests("Modules.HPTT" LargeTranspose)
#:
#:    creates ``Tests.Unit.<subcategory>.<name>.simd.<rung>`` (``baseline``, ``v2``, ``v3`` and
#:    ``v4`` on x86, ``baseline`` and ``sme`` on aarch64) through ``stripes_add_rung_tests``, whose
#:    launcher reports a rung the host cannot run as Skipped. No-op when the build is single-TU.
function(einsums_add_simd_rung_tests subcategory name)
  stripes_add_rung_tests(
    TARGET ${name}_test
    NAME Tests.Unit.${subcategory}.${name}.simd
    ARGS --einsums:debug:no-install-signal-handlers --einsums:debug:no-attach-debugger --einsums:profile:no-report
    OUT_TESTS _tests
  )
  foreach(_test IN LISTS _tests)
    # einsums_set_test_properties replaces ENVIRONMENT (for the sanitizer suppressions), which
    # erases the STRIPES_ARCH that pins the rung, so it is appended again afterwards.
    einsums_set_test_properties(${_test} "UNIT_ONLY")
    string(REGEX REPLACE ".*\\." "" _rung "${_test}")
    set_property(
      TEST ${_test}
      APPEND
      PROPERTY ENVIRONMENT "STRIPES_ARCH=${_rung}"
    )
  endforeach()
endfunction()

#:
#: .. cmake:command:: einsums_add_simd_rung_objects_test
#:
#:    Register ``Tests.Unit.<subcategory>.RungObjectsSelfContained``, Stripes' check that every
#:    per-rung object of the OBJECT library ``<target>`` keeps its weak symbols to itself.
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_simd_rung_objects_test(<subcategory> <target>)
function(einsums_add_simd_rung_objects_test subcategory target)
  set(_test Tests.Unit.${subcategory}.RungObjectsSelfContained)
  stripes_add_rung_objects_test(NAME ${_test} TARGET ${target})
  if(TEST ${_test})
    einsums_set_test_properties(${_test} "UNIT_ONLY")
  endif()
endfunction()
