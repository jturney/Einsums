#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(ini_tests parse)

foreach(test ${ini_tests})
    set(sources ${test}.cpp)

    source_group("Source Files" FILES ${sources})

    einsums_add_executable(
            ${test}_test INTERNAL_FLAGS
            SOURCES ${sources} ${${test}_FLAGS}
            FOLDER ${folder_name}
    )

    einsums_add_unit_test("modules.ini" ${test})
endforeach ()