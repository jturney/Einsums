#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(TaskPoolHeaders
    Einsums/TaskPool/TaskPool.hpp
    Einsums/TaskPool/TaskHandle.hpp
    Einsums/TaskPool/TaskGroup.hpp
    Einsums/TaskPool/ExecutionPolicy.hpp
    Einsums/TaskPool/WorkStealingDeque.hpp
    Einsums/TaskPool/WorkerContext.hpp
    Einsums/TaskPool/Dataflow.hpp
)

set(TaskPoolSources
    TaskPool.cpp
    TaskGroup.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TaskPool
  SOURCES ${TaskPoolSources}
  HEADERS ${TaskPoolHeaders}
  MODULE_DEPENDENCIES
    Einsums_Config
    Einsums_Logging
    Einsums_Profile
    Einsums_Runtime
    Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
