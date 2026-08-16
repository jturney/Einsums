#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(OptionsHeaders
    Einsums/Options/Source.hpp
    Einsums/Options/Get.hpp
    Einsums/Options/Declare.hpp
    Einsums/Options/Parse.hpp
    Einsums/Options/Options.hpp
    Einsums/CommandLine.hpp
)

set(OptionsSources Options.cpp Parse.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Options
  SOURCES ${OptionsSources}
  HEADERS ${OptionsHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
