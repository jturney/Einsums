#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(TensorUtilitiesTests Identity BlockViews)

foreach(test ${TensorUtilitiesTests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    FOLDER ${folder_name}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.TensorUtilities" ${test})
endforeach()

einsums_add_python_unit_test(
  "Modules.TensorUtilities" CreatePython
  SCRIPT test_create_python.py
)
