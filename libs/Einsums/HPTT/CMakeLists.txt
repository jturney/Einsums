#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(HPTTHeaders
    Einsums/HPTT/ComputeNode.hpp
    Einsums/HPTT/HPTT.hpp
    Einsums/HPTT/HPTTTypes.hpp
    Einsums/HPTT/Macros.hpp
    Einsums/HPTT/Plan.hpp
    Einsums/HPTT/Transpose.hpp
    Einsums/HPTT/Utils.hpp
    Einsums/HPTT/Files.hpp
)

set(HPTTSources HPTT.cpp Plan.cpp TransposeFactory.cpp Utils.cpp Primes.hpp Files.cpp)

# Transpose.cpp is the arch-generic kernel implementation: it is compiled
# once per SIMD-dispatch rung (each copy in its own namespace) rather than
# as a regular module source. TransposeFactory.cpp bridges the copies at
# plan creation. See Einsums/SIMD/RuntimeFeatures.hpp and
# Einsums_AddSIMDDispatch.cmake.
include(Einsums_AddSIMDDispatch)
einsums_add_simd_dispatch_sources(HPTTTransposeRungs IMPL src/Transpose.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums HPTT
  SOURCES ${HPTTSources}
  HEADERS ${HPTTHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Assertion Einsums_BLASBase Einsums_Errors Einsums_Logging Einsums_SIMD
  CMAKE_SUBDIRS examples tests
)

# The generated rung TUs carry absolute paths, which einsums_add_module's
# src/ prepend cannot express; attach them (and the rung-availability
# definitions TransposeFactory.cpp keys off) directly to the module target.
target_sources(Einsums_HPTT PRIVATE ${HPTTTransposeRungs})
target_compile_definitions(Einsums_HPTT PRIVATE ${HPTTTransposeRungs_DEFINITIONS})
