#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(HPTTTests
    LargeTranspose
    HalfPrecisionTranspose
    InnerStride
    ParameterValidation
    PlanFile
    StreamingPaths
)

foreach(test ${HPTTTests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    FOLDER ${folder_name}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.HPTT" ${test})
  einsums_add_simd_rung_tests("Modules.HPTT" ${test})
endforeach()

# The per-rung kernel objects must keep their weak symbols to themselves; see
# einsums_add_simd_rung_objects_test in cmake/Einsums_AddSIMDDispatch.cmake.
include(Einsums_AddSIMDDispatch)
einsums_add_simd_rung_objects_test("Modules.HPTT" Einsums_HPTT)
