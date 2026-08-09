#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Python-only module: the package shell and the rc -> argv translation that
# starts the runtime. No C++ unit tests here.
if(EINSUMS_BUILD_PYTHON)
  einsums_add_python_unit_test("Modules.Python" CliOptionsPython SCRIPT test_cli_options_python.py)
  einsums_add_python_unit_test("Modules.Python" InteropPsi4Python SCRIPT test_interop_psi4_python.py)
  einsums_add_python_unit_test("Modules.Python" WorldPython SCRIPT test_world_python.py)
  einsums_add_python_unit_test("Modules.Python" StagesPython SCRIPT test_stages_python.py)
  einsums_add_python_unit_test("Modules.Python" PromotePython SCRIPT test_promote_python.py)
  einsums_add_python_unit_test("Modules.Python" ExtractPython SCRIPT test_extract_python.py)
  einsums_add_python_unit_test("Modules.Python" SealedPython SCRIPT test_sealed_python.py)

  # EINSUMS_STAGE_MODULE, exercised by a real compiled stage module rather than
  # by a hand-built stand-in. The macro's job is to publish the fingerprints
  # THIS translation unit's headers computed; an implementation that read them
  # back from the runtime world() record would agree with the library every
  # time and pass a test that only ever sees a correct build. So the probe is
  # compiled twice, at two language levels, and one of the two must be refused.
  #
  # CXX_STANDARD with CXX_STANDARD_REQUIRED, not target_compile_features: the
  # latter requests a floor that an already-C++20 build satisfies without
  # adding a flag, which silently compiles both probes identically and turns
  # "the macro is wrong" into "the input never varied". See BuildModeProbe.cpp
  # in Config, where that happened under icx.
  #
  # Skipped outright, not degraded, where the toolchain has no C++23: a test
  # that cannot discriminate should say so.
  # The shared-graph contract: a C++ stage emitting into the caller's graph.
  # Built once, at the project's own standard, and separate from the handshake
  # probe below - that one is compiled at two language levels and has to stay on
  # a narrow ledge of minimal includes, which ComputeGraph is not.
  pybind11_add_module(einsums_shared_capture_probe MODULE SharedCaptureProbe.cpp)
  set_target_properties(
    einsums_shared_capture_probe
    PROPERTIES FOLDER "Tests/Unit" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  target_link_libraries(einsums_shared_capture_probe PRIVATE Einsums)
  add_dependencies(einsums_shared_capture_probe PyEinsums)

  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    foreach(_std 20 23)
      set(_probe einsums_stage_probe_cxx${_std})
      pybind11_add_module(${_probe} MODULE StageModuleProbe.cpp)
      target_compile_definitions(${_probe} PRIVATE STAGE_PROBE_NAME=${_probe})
      set_target_properties(
        ${_probe}
        PROPERTIES CXX_STANDARD ${_std}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   FOLDER "Tests/Unit"
                   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      )
      target_link_libraries(${_probe} PRIVATE Einsums)
      add_dependencies(${_probe} PyEinsums)
    endforeach()

    einsums_add_python_unit_test(
      "Modules.Python" StageModuleHandshakePython SCRIPT test_stage_module_python.py
    )
    add_dependencies(einsums_stage_probe_cxx20 einsums_shared_capture_probe)
    # APPEND, not set_tests_properties(PROPERTIES ENVIRONMENT ...). ctest's
    # ENVIRONMENT property clobbers rather than merges, so setting it here
    # discards the PYTHONPATH einsums_add_python_unit_test put there and the
    # test fails to import einsums at all. Einsums_AddTest.cmake documents this
    # exact trap; it is still easy to walk into.
    set_property(
      TEST Tests.Unit.Modules.Python.StageModuleHandshakePython
      APPEND
      PROPERTY ENVIRONMENT "EINSUMS_STAGE_PROBE_STD=20" "EINSUMS_STAGE_PROBE_SKEW=23"
    )
  endif()
endif()
