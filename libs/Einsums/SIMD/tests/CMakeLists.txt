#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_Message)

if(EINSUMS_WITH_TESTS)
  if(EINSUMS_WITH_TESTS_UNIT)
    einsums_add_pseudo_target(Tests.Unit.Modules.SIMD)
    einsums_add_pseudo_dependencies(Tests.Unit.Modules Tests.Unit.Modules.SIMD)
    add_subdirectory(unit)
  endif()

  if(EINSUMS_WITH_TESTS_BENCHMARKS)
    einsums_add_pseudo_target(Tests.Performance.Modules.SIMD)
    einsums_add_pseudo_dependencies(Tests.Performance.Modules Tests.Performance.Modules.SIMD)
    add_subdirectory(performance)
  endif()
endif()
