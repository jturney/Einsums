#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ProfileHeaders Einsums/Profile/LabeledSection.hpp Einsums/Profile/Timer.hpp)

set(ProfileSources Section.cpp Timer.cpp)

if (EINSUMS_WITH_OMP_TOOLS_H)
  list(APPEND ProfileSources OMPT.cpp)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Profile
  SOURCES ${ProfileSources}
  HEADERS ${ProfileHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_Logging Einsums_Print Einsums_StringUtil
  CMAKE_SUBDIRS examples tests
)
