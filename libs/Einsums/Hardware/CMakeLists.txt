#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(HardwareHeaders Einsums/Hardware/CpuInfo.hpp Einsums/Hardware/Machine.hpp Einsums/Hardware/Options.hpp
                    Einsums/Hardware/Profile.hpp
)

set(HardwareSources CpuInfo.cpp Machine.cpp Options.cpp Profile.cpp)

# Where an installation keeps the hardware profiles an administrator measured for its users. The
# path is compiled in; a conda package's prefix replacement relocates it with the environment.
include(GNUInstallDirs)
set_source_files_properties(
  src/Profile.cpp PROPERTIES COMPILE_DEFINITIONS EINSUMS_PROFILE_SITE_DIR="${CMAKE_INSTALL_FULL_DATADIR}/einsums/profiles"
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Hardware
  PYBIND
  SOURCES ${HardwareSources}
  HEADERS ${HardwareHeaders}
  DEPENDENCIES 
  MODULE_DEPENDENCIES Einsums_Config Einsums_Json Einsums_Logging Einsums_Options Einsums_SIMD
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)


