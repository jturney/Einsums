#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TypeSupportHeaders
    Einsums/TypeSupport/AreAllConvertible.hpp
    Einsums/TypeSupport/Arguments.hpp
    Einsums/TypeSupport/Casting.hpp
    Einsums/TypeSupport/CountOfType.hpp
    Einsums/TypeSupport/InsertionOrderedMap.hpp
    Einsums/TypeSupport/Lockable.hpp
    Einsums/TypeSupport/Observable.hpp
    Einsums/TypeSupport/StringLiteral.hpp
    Einsums/TypeSupport/Singleton.hpp
    Einsums/TypeSupport/TypeName.hpp
    Einsums/TypeSupport/TypeSwitch.hpp
    Einsums/TypeSupport/TypeTraits.hpp
)

set(TypeSupportSources)

if(EINSUMS_WITH_GPU_SUPPORT)
  list(APPEND TypeSupportHeaders Einsums/TypeSupport/GPUCast.hpp Einsums/TypeSupport/GPUComplex.hpp)

  set(TypeSupportGPUDeps hip::host hip::device)
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
