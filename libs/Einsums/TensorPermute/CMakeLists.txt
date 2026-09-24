#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


set(TensorPermuteHeaders Einsums/TensorPermute/Permute.hpp Einsums/TensorPermute/Unfold.hpp Einsums/TensorPermute/Detail/HpttPlanCache.hpp)

set(TensorPermuteSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorPermute
  SOURCES ${TensorPermuteSources}
  HEADERS ${TensorPermuteHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    Einsums_BufferAllocator
    Einsums_Config
    Einsums_Errors
    Einsums_Hardware
    Einsums_HPTT
    Einsums_Profile
    Einsums_StringUtil
    Einsums_TensorImpl
  CMAKE_SUBDIRS examples tests
)
