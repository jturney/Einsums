#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_Message)

if(EINSUMS_WITH_TESTS)
  if(EINSUMS_WITH_TESTS_UNIT)
    einsums_add_pseudo_target(Tests.Unit.Modules.Einsums.StringUtil)
    einsums_add_pseudo_dependencies(Tests.Unit.Modules Tests.Unit.Modules.Einsums.StringUtil)
    add_subdirectory(unit)
  endif()

  if(EINSUMS_WITH_TESTS_BENCHMARKS)
    einsums_add_pseudo_target(Tests.Performance.Modules.Einsums.StringUtil)
    einsums_add_pseudo_dependencies(
      Tests.Performance.Modules Tests.Performance.Modules.Einsums.StringUtil
    )
    add_subdirectory(performance)
  endif()

  if(EINSUMS_WITH_TESTS_HEADERS)
    einsums_add_header_tests(
      Modules.Einsums.StringUtil
      HEADERS ${StringUtilHeaders}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      DEPENDENCIES Einsums_StringUtil
    )
  endif()
endif()
