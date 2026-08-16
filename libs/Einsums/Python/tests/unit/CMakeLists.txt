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
  einsums_add_python_unit_test("Modules.Python" RcOptionsPython SCRIPT test_rc_options_python.py)

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
  # Pin the output directory for every configuration, not just the plain
  # property. Einsums_SetOutputPaths sets the PER-CONFIG output variables under
  # MSVC (build/<Config>/lib), and a per-config property beats the plain one, so
  # a bare LIBRARY_OUTPUT_DIRECTORY is silently ignored there: these probes
  # landed in build/RelWithDebInfo/lib while PYTHONPATH pointed at build/lib,
  # and every test that imports one failed with ModuleNotFoundError on Windows
  # alone. Einsums_FinalizePybind.cmake pins _core the same way and says so.
  function(einsums_probe_output_dir _target _dir)
    set_target_properties(${_target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_dir}")
    if(MSVC)
      set(_cfgs ${CMAKE_CONFIGURATION_TYPES})
      if(NOT _cfgs)
        set(_cfgs "${CMAKE_BUILD_TYPE}")
      endif()
      foreach(_cfg IN LISTS _cfgs)
        string(TOUPPER "${_cfg}" _cfg_upper)
        set_target_properties(
          ${_target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${_dir}"
                                RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${_dir}"
        )
      endforeach()
    endif()
  endfunction()

  # Both probe families need a SHARED libEinsums, so ask once, up front.
  #
  # This one emits into the CALLER'S graph, which means the C++ side has to be
  # looking at the same graph registry _core is. Against a static libEinsums it
  # would be looking at its own copy, so the contract it exists to demonstrate
  # cannot hold. It is also a second extension module, which is the shape
  # Einsums_PythonExtensionRegistry refuses outright.
  get_target_property(_einsums_lib_type Einsums TYPE)
  set(_einsums_lib_is_shared FALSE)
  if(_einsums_lib_type STREQUAL "SHARED_LIBRARY")
    set(_einsums_lib_is_shared TRUE)
  endif()

  if(_einsums_lib_is_shared)
    pybind11_add_module(einsums_shared_capture_probe MODULE SharedCaptureProbe.cpp)
    einsums_register_python_extension(einsums_shared_capture_probe)
    set_target_properties(einsums_shared_capture_probe PROPERTIES FOLDER "Tests/Unit")
    einsums_probe_output_dir(einsums_shared_capture_probe "${CMAKE_BINARY_DIR}/lib")
    target_link_libraries(einsums_shared_capture_probe PRIVATE Einsums)
    add_dependencies(einsums_shared_capture_probe PyEinsums)
  endif()

  # ...and skipped outright against a STATIC libEinsums, for the same reason.
  # The handshake asks "did this module resolve to the same copy of the library
  # I did?", which only has an answer when there is a copy to share. A static
  # libEinsums folds a private copy of the sealed-world state into every
  # extension module, so each probe registers into its own table and reports its
  # own identity, and the check refuses a module that is not skewed at all.
  #
  # Coverage builds used to land here, which is how this was found. They no
  # longer do: libs/CMakeLists.txt stopped selecting STATIC for coverage, so the
  # coverage legs run this test again like everyone else. What is left is
  # BUILD_SHARED_LIBS=OFF, someone asking for a static build deliberately, where
  # skipping is the honest answer rather than reporting a mismatch that says
  # nothing about the module under test.
  #
  # The library type was asked for above, once, rather than re-derived here, so
  # this gate follows any later change to the rule.
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES AND _einsums_lib_is_shared)
    foreach(_std 20 23)
      set(_probe einsums_stage_probe_cxx${_std})
      pybind11_add_module(${_probe} MODULE StageModuleProbe.cpp)
      einsums_register_python_extension(${_probe})
      target_compile_definitions(${_probe} PRIVATE STAGE_PROBE_NAME=${_probe})
      set_target_properties(
        ${_probe}
        PROPERTIES CXX_STANDARD ${_std}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   FOLDER "Tests/Unit"
      )
      einsums_probe_output_dir(${_probe} "${CMAKE_BINARY_DIR}/lib")
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
