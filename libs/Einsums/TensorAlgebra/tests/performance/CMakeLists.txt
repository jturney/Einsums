#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(benchmarks Gemm)

foreach(benchmark ${benchmarks})
    set(sources ${benchmark}.cpp)

    source_group("Source Files" FILES ${sources})

    # add benchmark executable
    einsums_add_executable(
            ${benchmark}_test INTERNAL_FLAGS
            SOURCES ${sources}
            ${${benchmark}_FLAGS}
            NOINSTALL
            FOLDER "Benchmarks/Modules/TensorAlgebra"
    )

    einsums_add_performance_test("Modules.TensorAlgebra" ${benchmark} ${${benchmark}_PARAMETERS})
endforeach()
