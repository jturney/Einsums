#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ProfileHeaders
    Einsums/Profile/Profile.hpp
    Einsums/Profile/Event.hpp
    Einsums/Profile/StringTable.hpp
    Einsums/Profile/RingBuffer.hpp
    Einsums/Profile/Consumer.hpp
    Einsums/Profile/CounterBackend.hpp
    Einsums/Profile/Server.hpp
    Einsums/Profile/LogSink.hpp
)

set(ProfileSources Profile.cpp Consumer.cpp CounterBackend.cpp Server.cpp)

if(EINSUMS_WITH_OMP_TOOLS_H)
  list(APPEND ProfileSource OMPT.cpp)
endif()

set(ProfileDeps)
if(TARGET Tracy::TracyClient)
  list(APPEND ProfileDeps Tracy::TracyClient)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Profile
  PYBIND
  SOURCES ${ProfileSources}
  HEADERS ${ProfileHeaders}
  DEPENDENCIES ${ProfileDeps}
  MODULE_DEPENDENCIES Einsums_Config Einsums_Logging Einsums_Print Einsums_Preprocessor Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
