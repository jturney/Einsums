#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(init_runtime_tests
        config_entry
)

foreach (test ${init_runtime_tests})
    set(sources ${test}.cpp)

    set(${test}_PARAMETERS ${${test}_PARAMETERS} THREADS 4)

    source_group("Source Files" FILES ${sources})
    set(folder_name "Tests/Units/Modules/InitRuntime")

    einsums_add_executable(
            ${test}_test INTERNAL_FLAGS
            SOURCES ${sources} ${${test}_FLAGS}
            FOLDER ${folder_name}
    )

    einsums_add_unit_test("modules.init_runtime" ${test} ${${test_}_PARAMETERS})
endforeach ()