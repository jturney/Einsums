#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(HPTTTests 
#    CreatePlan
)

foreach(test ${HPTTTests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
          FOLDER ${folder_name} NOINSTALL
  )

  einsums_add_unit_test("Modules.HPTT" ${test})
endforeach()
