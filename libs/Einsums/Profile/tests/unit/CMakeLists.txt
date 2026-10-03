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

# The C++ suites drive the recording machinery directly (ring buffers, the
# aggregating consumer, the annotation tree), none of which is compiled when the
# profiler is off. The Python suite runs either way: it covers the binding
# surface, which stays in place as no-ops, and skips the assertions that need
# something to have been recorded.
set(profile_tests test_profile_python.py)

if(EINSUMS_WITH_PROFILER)
  list(PREPEND profile_tests ProfileRingBuffer.cpp ProfileConsumer.cpp ProfileAnnotations.cpp
       ProfileStringTable.cpp ProfileProducer.cpp ProfileServer.cpp ProfileSettings.cpp ProfileSites.cpp)
endif()

foreach(test ${profile_tests})
  # Extract base name of test Example1.cpp -> Example1, Example.2.hip -> Example
  get_filename_component(TestName ${test} NAME_WE)
  # Extract extension of the test Example1.cpp -> .cpp, Example.2.hip -> .2.hip
  get_filename_component(Extension ${test} EXT)

  if(Extension STREQUAL ".py")
    # Python tests only build when Python bindings are enabled. The
    # finalize step links the per-test pseudo-targets into the package
    # so ctest sees a complete einsums._core to import.
    if(NOT EINSUMS_BUILD_PYTHON)
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

# Waggle, the profiler, is its own library and must stay buildable without Einsums. Each probe is
# compiled with the compiler's include listing on, and the gate fails if any listed header lies under
# libs/Einsums, through any chain of includes. The probe links Waggle alone, so an Einsums include
# would not even compile; the listing catches what a shared include path would hide. The self-test
# probe links Einsums and includes one of its headers, and must fail, which shows the listing is read.
if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(_waggle_include_listing_flag /showIncludes)
else()
  set(_waggle_include_listing_flag -H)
endif()
foreach(_probe WaggleHeaderClosureProbe WaggleHeaderClosureProbeSelfTest)
  add_library(Profile_${_probe} OBJECT EXCLUDE_FROM_ALL ${_probe}.cpp)
  target_link_libraries(Profile_${_probe} PRIVATE Waggle::waggle)
  if(_probe STREQUAL "WaggleHeaderClosureProbeSelfTest")
    target_link_libraries(Profile_${_probe} PRIVATE Einsums::Einsums)
  endif()
  target_compile_options(Profile_${_probe} PRIVATE ${_waggle_include_listing_flag})
  set_target_properties(Profile_${_probe} PROPERTIES FOLDER "Tests/Unit/Modules/Profile")
endforeach()
foreach(_gate WaggleHeaderClosure WaggleHeaderClosureSelfTest)
  if(_gate STREQUAL "WaggleHeaderClosure")
    set(_probe WaggleHeaderClosureProbe)
  else()
    set(_probe WaggleHeaderClosureProbeSelfTest)
  endif()
  # Only include-listing lines (". path" from -H, "including file: path" from /showIncludes), so
  # the build's own status lines, which name this directory, never match.
  add_test(
    NAME Tests.Unit.Modules.Profile.${_gate}
    COMMAND
      ${CMAKE_COMMAND} "-DOBJECT=$<TARGET_OBJECTS:Profile_${_probe}>" "-DBUILD_DIR=${PROJECT_BINARY_DIR}"
      "-DTARGET=Profile_${_probe}" "-DFORBID=(\\.|including file:) [^\n]*libs[/\\\\]Einsums[/\\\\]"
      "-DEXPECT=waggle[/\\\\]Waggle\\.hpp" "-DCONFIG=$<CONFIG>" -P ${PROJECT_SOURCE_DIR}/cmake/scripts/check_header_closure.cmake
  )
  # A build of the shared tree, like the header compile tests, so it takes their lock.
  set_tests_properties(Tests.Unit.Modules.Profile.${_gate} PROPERTIES LABELS "UNIT_ONLY" RESOURCE_LOCK "einsums_build_tree")
endforeach()
# The self-test passes only on the gate's own verdict: a build failure or a missing listing also
# exits nonzero, and WILL_FAIL would count either as the gate working.
set_tests_properties(
  Tests.Unit.Modules.Profile.WaggleHeaderClosureSelfTest PROPERTIES PASS_REGULAR_EXPRESSION "FORBIDDEN_HEADER_REACHED"
)
