#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(PackedGemmHeaders
    Einsums/PackedGemm/ContractionKey.hpp
    Einsums/PackedGemm/InitModule.hpp
    Einsums/PackedGemm/MicroKernel.hpp
    Einsums/PackedGemm/MicroKernelBody.hpp
    Einsums/PackedGemm/Options.hpp
    Einsums/PackedGemm/PackedGemm.hpp
    Einsums/PackedGemm/Packing.hpp
    Einsums/PackedGemm/Stream.hpp
    Einsums/PackedGemm/StreamKernel.hpp
)

set(PackedGemmSources
    ContractionKey.cpp
    InitModule.cpp
    MicroKernelDispatch.cpp
    Options.cpp
    Packing.cpp
    RouteIntrospection.cpp
    Stream.cpp
    StreamKernelDispatch.cpp
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

# StreamKernelImpl.cpp is the inner loop of stream_contract, dispatched the same
# way. No `sme` rung is requested: the kernel is a bandwidth-bound streaming FMA
# that gains nothing from the matrix unit.
einsums_add_simd_dispatch_sources(
  PackedGemmStreamKernelRungs IMPL src/StreamKernelImpl.cpp RUNGS baseline v2 v3 v4
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums PackedGemm
  SOURCES ${PackedGemmSources}
  HEADERS ${PackedGemmHeaders}
  MODULE_DEPENDENCIES
    Einsums_BLAS
    Einsums_Config
    Einsums_Hardware
    Einsums_Concepts
    Einsums_Logging
    Einsums_Options
    Einsums_Profile
    Einsums_Runtime
    Einsums_HPTT
    Einsums_SIMD
  CMAKE_SUBDIRS tests
)

# The generated rung TUs carry absolute paths, which einsums_add_module's
# src/ prepend cannot express; attach them directly to the module target.
target_sources(Einsums_PackedGemm PRIVATE ${PackedGemmMicroKernelRungs} ${PackedGemmStreamKernelRungs})

# Each dispatch TU declares an entry for every rung its definitions name, so
# each gets only its own kernel's set: the micro-kernel builds an `sme` rung and
# the stream kernel does not, and a module-wide definition would have the stream
# dispatch reference an sme copy that was never compiled.
set_source_files_properties(
  src/MicroKernelDispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${PackedGemmMicroKernelRungs_DEFINITIONS}"
)
set_source_files_properties(
  src/StreamKernelDispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${PackedGemmStreamKernelRungs_DEFINITIONS}"
)
