#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ThreadsHeaders Einsums/Threads/CallbackNotifier.hpp Einsums/Threads/SchedulerMode.hpp
                   Einsums/Threads/detail/TrackLocks.hpp
)

set(ThreadsSources InitModule.hpp ModuleVars.hpp InitModule.cpp ModuleVars.cpp TrackLocks.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Threads
  SOURCES ${ThreadsSources}
  HEADERS ${ThreadsHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config
                      Einsums_Debugging
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
