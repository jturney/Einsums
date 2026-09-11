#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TypeSupportHeaders
    Einsums/TypeSupport/AreAllConvertible.hpp
    Einsums/TypeSupport/Arguments.hpp
    Einsums/TypeSupport/CountOfType.hpp
    Einsums/TypeSupport/Observable.hpp
    Einsums/TypeSupport/StringLiteral.hpp
    Einsums/TypeSupport/TypeName.hpp
    Einsums/TypeSupport/Lockable.hpp
    Einsums/TypeSupport/Singleton.hpp
    Einsums/TypeSupport/TypeSwitch.hpp
        Einsums/TypeSupport/InsertionOrderedMap.hpp
    Einsums/TypeSupport/JsonEscape.hpp
)

set(TypeSupportSources)

if(EINSUMS_WITH_GPU_SUPPORT)
  # GPUCast/GPUComplex/GPUTypes are HIP-ONLY headers: they include <hip/hip_common.h>
  # and friends unconditionally and traffic in hipFloatComplex. Listing them for any
  # GPU support meant a CUDA build tried to compile them without ROCm anywhere on the
  # machine, which the header-compile tests catch as
  # "fatal error: hip/hip_common.h: No such file or directory". It went unnoticed
  # because no build had EINSUMS_WITH_GPU_SUPPORT on until now.
  if(EINSUMS_WITH_HIP)
    list(APPEND TypeSupportHeaders Einsums/TypeSupport/GPUCast.hpp Einsums/TypeSupport/GPUComplex.hpp Einsums/TypeSupport/GPUTypes.hpp)
  endif()

  # Vendor-conditional: hip::host / hip::device only exist under ROCm. Requesting
  # them unconditionally on EINSUMS_WITH_GPU_SUPPORT broke CUDA builds, which have
  # no ROCm at all. CUDA needs no extra target here - the GPU*.hpp headers above are
  # header-only type mappings and pull their vendor headers through Einsums_GPU.
  if(EINSUMS_WITH_HIP)
    set(TypeSupportGPUDeps hip::host hip::device)
  else()
    set(TypeSupportGPUDeps)
  endif()
else()
  set(TypeSupportGPUDeps)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums TypeSupport
  SOURCES ${TypeSupportSources}
  HEADERS ${TypeSupportHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES ${TypeSupportGPUDeps}
  CMAKE_SUBDIRS examples tests
)
