# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

set(tests tensor_algebra)

foreach(test ${tests})

    set(sources ${test}.cpp)

    source_group("Source Files" FILES ${sources})

    einsums_add_executable(tensor_algebra_test INTERNAL_FLAGS SOURCES tensor_algebra.cpp)

    einsums_add_unit_test("modules.tensor_algebra" ${test} ${${test}_PARAMETERS} VALGRIND)

endforeach()
