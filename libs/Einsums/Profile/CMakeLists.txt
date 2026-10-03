#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# The Einsums side of the profiler: its command-line options, the spdlog sink, the recording check
# and the einsums.profile Python surface. The profiler itself is Waggle (libs/Waggle).
set(ProfileHeaders Einsums/Profile/Profile.hpp Einsums/Profile/Options.hpp Einsums/Profile/Recording.hpp
                   Einsums/Profile/SpdlogSink.hpp
)

set(ProfileSources Options.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Profile
  PYBIND
  SOURCES ${ProfileSources}
  HEADERS ${ProfileHeaders}
  DEPENDENCIES Waggle::waggle
  MODULE_DEPENDENCIES Einsums_Options Einsums_Config Einsums_Logging
  CMAKE_SUBDIRS examples tests
)
