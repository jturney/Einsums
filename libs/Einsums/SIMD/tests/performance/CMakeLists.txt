#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# BenchmarkKernels times the SIMD kernels on every dispatch rung the machine supports in one run:
# the kernels (BenchmarkKernelsImpl.cpp) are compiled once per rung, the driver once.
include(Einsums_AddSIMDDispatch)
einsums_add_simd_dispatch_sources(SIMDBenchmarkRungs IMPL BenchmarkKernelsImpl.cpp RUNGS baseline v2 v3 v4)

einsums_add_executable(
  BenchmarkKernels_test INTERNAL_FLAGS
  SOURCES BenchmarkKernels.cpp BenchmarkKernels.hpp ${SIMDBenchmarkRungs}
  NOINSTALL
)
target_compile_definitions(BenchmarkKernels_test PRIVATE ${SIMDBenchmarkRungs_DEFINITIONS})
einsums_add_performance_test("Modules.SIMD" BenchmarkKernels)
