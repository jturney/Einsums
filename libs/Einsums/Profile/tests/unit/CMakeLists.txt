#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Convert ``snake_case`` to ``CamelCase`` so we can derive a test name from
# a ``test_<snake>_python.py`` source filename without keeping a parallel
# name list (mirrors the helper in libs/Einsums/ComputeGraph/tests/unit).
function(_einsums_snake_to_camel out_var snake)
  set(_camel "")
  string(REPLACE "_" ";" _parts "${snake}")
  foreach(_part IN LISTS _parts)
    if(_part STREQUAL "")
      continue()
    endif()
    string(SUBSTRING "${_part}" 0 1 _head)
    string(TOUPPER "${_head}" _head)
    string(SUBSTRING "${_part}" 1 -1 _tail)
    string(APPEND _camel "${_head}${_tail}")
  endforeach()
  set(${out_var} "${_camel}" PARENT_SCOPE)
endfunction()

set(profile_tests
    ProfileRingBuffer.cpp
    ProfileConsumer.cpp
    ProfileAnnotations.cpp
    test_profile_python.py
)

foreach(test ${profile_tests})
  # Extract base name of test Example1.cpp -> Example1, Example.2.hip -> Example
  get_filename_component(TestName ${test} NAME_WE)
  # Extract extension of the test Example1.cpp -> .cpp, Example.2.hip -> .2.hip
  get_filename_component(Extension ${test} EXT)

  if(Extension STREQUAL ".py")
    # Python tests only build when Python bindings are enabled. The
    # finalize step links the per-test pseudo-targets into the package
    # so ctest sees a complete einsums._core to import.
    if(NOT EINSUMS_BUILD_PYTHON OR NOT EINSUMS_PYBIND_AUTOGEN)
      continue()
    endif()
    # Naming convention: ``test_<snake>_python.py`` -> ``<Camel>Python``,
    # so e.g. ``test_profile_python.py`` registers as
    # ``Tests.Unit.Modules.Profile.ProfilePython``.
    string(REGEX REPLACE "^test_"   "" _stem "${TestName}")
    string(REGEX REPLACE "_python$" "" _stem "${_stem}")
    _einsums_snake_to_camel(_camel "${_stem}")
    einsums_add_python_unit_test(
      "Modules.Profile" "${_camel}Python"
      SCRIPT ${test}
    )
    continue()
  endif()

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

  einsums_add_unit_test("Modules.Profile" ${TestName} ${${TestName}_${TestExtension}_PARAMETERS})
endforeach()
