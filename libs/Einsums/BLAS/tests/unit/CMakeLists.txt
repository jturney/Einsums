#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(BLASTests ThreadControl.cpp)

foreach(test ${BLASTests})
  get_filename_component(TestName ${test} NAME_WE)

  set(Sources ${test})

  source_group("Source Files" FILES ${Sources})

  einsums_add_executable(
    ${TestName}_test
    INTERNAL_FLAGS
    SOURCES ${Sources}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.BLAS" ${TestName})
endforeach()
