#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(PackedGemmHeaders
    Einsums/PackedGemm/ContractionKey.hpp
    Einsums/PackedGemm/EinsumPackedGemm.hpp
    Einsums/PackedGemm/InitModule.hpp
    Einsums/PackedGemm/MicroKernel.hpp
    Einsums/PackedGemm/MicroKernelBody.hpp
    Einsums/PackedGemm/Packing.hpp
)

set(PackedGemmSources
    ContractionKey.cpp
    InitModule.cpp
    MicroKernelDispatch.cpp
    Packing.cpp
)

# MicroKernelImpl.cpp is the arch-generic tile-kernel implementation: it is
# compiled once per SIMD-dispatch rung (each copy in its own namespace) rather
# than as a regular module source. MicroKernelDispatch.cpp bridges the copies
# when blis_contraction resolves its tile kernel. See
# Einsums/SIMD/RuntimeFeatures.hpp and Einsums_AddSIMDDispatch.cmake.
include(Einsums_AddSIMDDispatch)
einsums_add_simd_dispatch_sources(
  PackedGemmMicroKernelRungs IMPL src/MicroKernelImpl.cpp RUNGS baseline v2 v3 v4 sme
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums PackedGemm
  SOURCES ${PackedGemmSources}
  HEADERS ${PackedGemmHeaders}
  MODULE_DEPENDENCIES
    Einsums_BLAS
    Einsums_Config
    Einsums_Concepts
    Einsums_Logging
    Einsums_Profile
    Einsums_Runtime
    Einsums_HPTT
    Einsums_SIMD
  CMAKE_SUBDIRS tests
)

# The generated rung TUs carry absolute paths, which einsums_add_module's
# src/ prepend cannot express; attach them (and the rung-availability
# definitions MicroKernelDispatch.cpp keys off) directly to the module target.
target_sources(Einsums_PackedGemm PRIVATE ${PackedGemmMicroKernelRungs})
target_compile_definitions(Einsums_PackedGemm PRIVATE ${PackedGemmMicroKernelRungs_DEFINITIONS})
