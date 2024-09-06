#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(tensor_tests
        block_tensor
        block_tensor_info
        block_tensor_operations
        tensor tiled_tensor
        tiled_tensor_gemm
        tiled_tensor_gemv
)

foreach (test ${tensor_tests})
    set(sources ${test}.cpp)

    source_group("Source Files" FILES ${sources})

    einsums_add_executable(
            ${test}_test INTERNAL_FLAGS
            SOURCES ${sources} ${${test}_FLAGS}
            FOLDER ${folder_name}
    )

    einsums_add_unit_test("modules.tensor" ${test})
endforeach ()