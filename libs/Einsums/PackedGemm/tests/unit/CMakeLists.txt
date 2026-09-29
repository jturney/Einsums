#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(PackedGemmTests
    BatchedMultiK.cpp
    Complex1m.cpp
    EngineOptions.cpp
    FlattenLayouts.cpp
    MicroKernelTile.cpp
    PackTranspose.cpp
    PackingPlan.cpp
    Stream.cpp
    StreamKernel.cpp
    ThreadGrid.cpp
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
# ThreadGrid asserts the teams the engine forms, which depend on the cache sizes and on how many
# cores share an L3: a runner's own would form other teams, or none (macOS reports no sharing). Pin
# the three-core-per-L3 Zen+ part the thresholds were measured on, so every runner models it.
set_property(
  TEST Tests.Unit.Modules.PackedGemm.ThreadGrid APPEND PROPERTY ENVIRONMENT
                                                                "EINSUMS_HARDWARE_L1_CACHE_SIZE=32768"
                                                                "EINSUMS_HARDWARE_L2_CACHE_SIZE=524288"
                                                                "EINSUMS_HARDWARE_L3_CACHE_SIZE=8388608"
                                                                "EINSUMS_PACKED_GEMM_CORES_PER_L3=3"
)

einsums_add_executable(
  MultiMN_test INTERNAL_FLAGS
  SOURCES MultiMN.cpp
  NOINSTALL
)
einsums_add_unit_test("Modules.PackedGemm" MultiMN)

# The direct-BLAS fast paths go through the full einsum dispatch too.
einsums_add_executable(
  DirectBlasPaths_test INTERNAL_FLAGS
  SOURCES DirectBlasPaths.cpp
  NOINSTALL
)
einsums_add_unit_test("Modules.PackedGemm" DirectBlasPaths)

# The tile micro-kernel is compiled once per SIMD dispatch rung; run the
# multi-M/N suite (which drives the scatter path through every tile shape)
# once per rung (no-op on single-TU builds).
include(Einsums_AddSIMDDispatch)
einsums_add_simd_rung_tests("Modules.PackedGemm" MultiMN)
einsums_add_simd_rung_tests("Modules.PackedGemm" MicroKernelTile)
einsums_add_simd_rung_tests("Modules.PackedGemm" PackTranspose)
einsums_add_simd_rung_tests("Modules.PackedGemm" Complex1m)
# The C-temp budget only sizes the block strategy, which some rungs never take for complex.
einsums_add_simd_rung_tests("Modules.PackedGemm" EngineOptions)
# The stream inner loop is compiled once per rung; pin it, and the route that uses it, on each.
einsums_add_simd_rung_tests("Modules.PackedGemm" StreamKernel)
einsums_add_simd_rung_tests("Modules.PackedGemm" Stream)
