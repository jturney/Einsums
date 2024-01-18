# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------


include(einsums_message)

if(EINSUMS_WITH_TESTS)
    if(EINSUMS_WITH_TESTS_UNIT)
      einsums_add_pseudo_target(tests.unit.modules.itt)
      einsums_add_pseudo_dependencies(
        tests.unit.modules tests.unit.modules.itt
      )
      add_subdirectory(unit)
    endif()

    if(EINSUMS_WITH_TESTS_REGRESSIONS)
        einsums_add_pseudo_target(tests.regressions.modules.itt)
        einsums_add_pseudo_dependencies(
          tests.regressions.modules tests.regressions.modules.itt
        )
        add_subdirectory(regressions)
    endif()

    if(EINSUMS_WITH_TESTS_BENCHMARKS)
        einsums_add_pseudo_target(tests.performance.modules.itt)
        einsums_add_pseudo_dependencies(
          tests.performance.modules tests.performance.modules.itt
        )
        add_subdirectory(performance)
    endif()

    if(EINSUMS_WITH_TESTS_HEADERS)
        einsums_add_header_tests(
          modules.itt
          HEADERS ${itt_headers}
          HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
          DEPENDENCIES einsums_itt
        )
    endif()
endif()
