#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(CommTests
    CommBasic.cpp
)

foreach(test ${CommTests})
  get_filename_component(TestName ${test} NAME_WE)

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${test}
    NOINSTALL
  )

  einsums_add_unit_test(
    "Modules.Comm" ${TestName}
  )
endforeach()
