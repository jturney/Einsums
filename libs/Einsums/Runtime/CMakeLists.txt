#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(RuntimeHeaders
    Einsums/Runtime.hpp
    Einsums/Runtime/InitRuntime.hpp
    Einsums/Runtime/Runtime.hpp
    Einsums/Runtime/RuntimeForward.hpp
    Einsums/Runtime/StartupFunction.hpp
    Einsums/Runtime/ShutdownFunction.hpp
    Einsums/Runtime/Detail/InitLogging.hpp
)

set(RuntimeSources Finalize.cpp InitLogging.cpp InitRuntime.cpp Runtime.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Runtime
  SOURCES ${RuntimeSources}
  HEADERS ${RuntimeHeaders}
  DEPENDENCIES Einsums::h5cpp
  MODULE_DEPENDENCIES
    Einsums_Assertion
    Einsums_Config
    Einsums_Debugging
    Einsums_Profile
    Einsums_Utilities
    Einsums_RuntimeConfiguration
    Einsums_Errors
    Einsums_Logging
    Einsums_Version
    Einsums_ThreadsBase
  CMAKE_SUBDIRS examples tests
  # BASE_LIBNAME Einsums
)
