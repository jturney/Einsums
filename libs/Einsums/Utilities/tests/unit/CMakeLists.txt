#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(UtilitiesTests)

foreach(test ${UtilitiesTests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    FOLDER ${folder_name}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.Utilities" ${test})
endforeach()

# Python coverage for the Apiary-exposed free functions (seed_random,
# make_temp_path). Registers as Tests.Unit.Modules.Utilities.UtilitiesPython.
einsums_add_python_unit_test(
  "Modules.Utilities" "UtilitiesPython"
  SCRIPT test_utilities_python.py
)
