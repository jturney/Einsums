#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(PackedGemmTests
    BatchedMultiK.cpp
    PackingPlan.cpp
)

foreach(test ${PackedGemmTests})
  get_filename_component(TestName ${test} NAME_WE)

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${test}
    NOINSTALL
  )

  einsums_add_unit_test(
    "Modules.PackedGemm" ${TestName}
  )
endforeach()

# Multi-M/N tests need TensorAlgebra (full einsum dispatch)
einsums_add_executable(
  MultiMN_test INTERNAL_FLAGS
  SOURCES MultiMN.cpp
  NOINSTALL
)
einsums_add_unit_test("Modules.PackedGemm" MultiMN)

# The tile micro-kernel is compiled once per SIMD dispatch rung; run the
# multi-M/N suite (which drives the scatter path through every tile shape)
# once per rung (no-op on single-TU builds).
include(Einsums_AddSIMDDispatch)
einsums_add_simd_rung_tests("Modules.PackedGemm" MultiMN)
