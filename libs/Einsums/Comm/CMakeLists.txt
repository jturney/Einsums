#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(CommHeaders
    Einsums/Comm/Platform.hpp
    Einsums/Comm/Runtime.hpp
    Einsums/Comm/Error.hpp
    Einsums/Comm/Communicator.hpp
    Einsums/Comm/Collectives.hpp
    Einsums/Comm/DistributionDescriptor.hpp
    Einsums/Comm/DistributionMap.hpp
    Einsums/Comm/ProcessGrid.hpp
)

set(CommSources
    Runtime.cpp
    Communicator.cpp
    Collectives.cpp
    ProcessGrid.cpp
    InitModule.cpp
)

# MPI is optional — when unavailable, the mock backend compiles and runs serially.
set(COMM_EXTERNAL_DEPS)
if(EINSUMS_WITH_MPI)
  list(APPEND COMM_EXTERNAL_DEPS MPI::MPI_CXX)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Comm
  SOURCES ${CommSources}
  HEADERS ${CommHeaders}
  DEPENDENCIES ${COMM_EXTERNAL_DEPS}
  MODULE_DEPENDENCIES
    Einsums_Config
    Einsums_CXX23
    Einsums_Errors
    Einsums_GPU
    Einsums_Logging
    Einsums_Preprocessor
    Einsums_Runtime
  CMAKE_SUBDIRS tests
)
