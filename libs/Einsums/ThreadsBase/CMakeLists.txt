#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ThreadsBaseHeaders
    Einsums/ThreadsBase/CallbackNotifier.hpp
    Einsums/ThreadsBase/RuntimeState.hpp
    Einsums/ThreadsBase/SchedulerBase.hpp
    Einsums/ThreadsBase/SchedulerMode.hpp
    Einsums/ThreadsBase/TrackLocks.hpp
    Einsums/ThreadsBase/ThreadEnums.hpp
    Einsums/ThreadsBase/ThreadPoolBase.hpp
)

set(ThreadsBaseSources SchedulerBase.cpp SchedulerMode.cpp ThreadEnums.cpp ThreadPoolBase.cpp
                       TrackLocks.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums ThreadsBase
  SOURCES ${ThreadsBaseSources}
  HEADERS ${ThreadsBaseHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Logging Einsums_TypeSupport Einsums_Config Einsums_Hardware
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
