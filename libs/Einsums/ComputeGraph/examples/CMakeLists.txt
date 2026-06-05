#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.Einsums.ComputeGraph)
  einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.Einsums.ComputeGraph)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.Einsums.ComputeGraph)
    einsums_add_pseudo_dependencies(
      Tests.Examples.Modules Tests.Examples.Modules.Einsums.ComputeGraph
    )
  endif()
else()
  return()
endif()

set(example_programs
    AsyncIO
    BasicCapture
    ControlFlow
    DeferredAllocation
    GraphRebind
    MixedOperations
    OptimizationPasses
    GEMMBatchingDemo
    ParallelExecution
    PipelineWorkflow
    SCFSimulation
    StringEinsum
    TaskPoolIntegration
    FockBuildTaskPool
    IntegralTransform
    ContinuationPipeline
    DistributedGEMM
    GPUOffload
    Views
)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    CG_${example_program} INTERNAL_FLAGS
    SOURCES ${sources}
    FOLDER "Examples/Modules/ComputeGraph"
    NOINSTALL
  )
  target_include_directories(CG_${example_program} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  einsums_add_example_target_dependencies("Modules.Einsums.ComputeGraph" CG_${example_program})

  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_example_test(
      "Modules.Einsums.ComputeGraph" CG_${example_program}
    )
  endif()
endforeach()

# Python-script examples. Registered as plain ctest entries that run the
# script through the build's einsums extension (PYTHONPATH points at
# ``${CMAKE_BINARY_DIR}/lib`` where ``_core.*.so`` sits). Each script also
# enrolls in the global ``EINSUMS_PYTHON_UNIT_TEST_TARGETS`` list so
# ``einsums_finalize_pybind()`` wires the PyEinsums build dependency once
# that target exists — mirrors how ``einsums_add_python_unit_test`` does it.
if(EINSUMS_BUILD_PYTHON
   AND EINSUMS_WITH_TESTS
   AND EINSUMS_WITH_TESTS_EXAMPLES
)
  # Under TSan, `python script.py` (as opposed to `python -m pytest script.py`,
  # which the Python unit tests use) hits libtsan's static-TLS allocation
  # limit when it dlopens our instrumented _core.so:
  #   ImportError: libtsan.so.2: cannot allocate memory in static TLS block
  # The fix is to LD_PRELOAD libtsan so its TLS slot is reserved at process
  # start. Discover the conda env's libtsan once at configure time and only
  # add the preload when the thread sanitizer is active. (Unit tests aren't
  # affected; pytest's startup happens to allocate enough dynamic TLS that
  # einsums._core slots in fine.)
  set(_einsums_python_example_preload "")
  if(EINSUMS_WITH_SANITIZERS AND "${EINSUMS_WITH_SANITIZERS}" MATCHES "thread"
     AND DEFINED ENV{CONDA_PREFIX})
    find_file(EINSUMS_LIBTSAN_PATH
      NAMES libtsan.so.2 libtsan.so
      PATHS "$ENV{CONDA_PREFIX}/lib"
      NO_DEFAULT_PATH
    )
    if(EINSUMS_LIBTSAN_PATH)
      set(_einsums_python_example_preload "LD_PRELOAD=${EINSUMS_LIBTSAN_PATH}")
    endif()
  endif()

  set(python_example_scripts
      mp2_simulation.py
      scf_simulation.py
  )
  foreach(_script ${python_example_scripts})
    get_filename_component(_stem ${_script} NAME_WE)
    # snake_case -> CamelCase, matching the cpp example naming style so
    # ``mp2_simulation`` registers as ``Mp2Simulation``.
    set(_camel "")
    string(REPLACE "_" ";" _parts "${_stem}")
    foreach(_part IN LISTS _parts)
      if(_part STREQUAL "")
        continue()
      endif()
      string(SUBSTRING "${_part}" 0 1 _head)
      string(TOUPPER "${_head}" _head)
      string(SUBSTRING "${_part}" 1 -1 _tail)
      string(APPEND _camel "${_head}${_tail}")
    endforeach()

    set(_test_name "Tests.Examples.Modules.Einsums.ComputeGraph.${_camel}")
    add_test(
      NAME "${_test_name}"
      COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${_script}"
    )
    # Build the test ENVIRONMENT string. PYTHONPATH always; LD_PRELOAD only
    # under TSan (see comment above).
    set(_example_env "PYTHONPATH=${CMAKE_BINARY_DIR}/lib")
    if(_einsums_python_example_preload)
      set(_example_env "${_example_env};${_einsums_python_example_preload}")
    endif()
    set_tests_properties(
      "${_test_name}" PROPERTIES
      ENVIRONMENT "${_example_env}"
      LABELS "EXAMPLES_ONLY;PYTHON"
    )
    einsums_add_pseudo_target("${_test_name}")
    einsums_add_pseudo_dependencies(
      "Tests.Examples.Modules.Einsums.ComputeGraph" "${_test_name}"
    )
    set_property(GLOBAL APPEND PROPERTY EINSUMS_PYTHON_UNIT_TEST_TARGETS "${_test_name}")
  endforeach()
endif()
