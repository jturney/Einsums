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
    Einsums/Profile/RequestHandlers.hpp
    Einsums/Profile/Settings.hpp
    Einsums/Profile/LogQueue.hpp
    Einsums/Profile/Diagnostics.hpp
    Einsums/Profile/SpdlogSink.hpp
    Einsums/Profile/Options.hpp
    Einsums/Profile/Recording.hpp
    Einsums/Profile/TickClock.hpp
)

set(ProfileSources Profile.cpp Consumer.cpp CounterBackend.cpp Diagnostics.cpp Options.cpp Server.cpp Settings.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Profile
  PYBIND
  SOURCES ${ProfileSources}
  HEADERS ${ProfileHeaders}
  MODULE_DEPENDENCIES Einsums_Options Einsums_Config Einsums_Logging Einsums_Print Einsums_Preprocessor Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
