#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ConfigHeaders
    Einsums/Config.hpp
    Einsums/Config/Alias.hpp
    Einsums/Config/BranchHints.hpp
    Einsums/Config/CompilerSpecific.hpp
    Einsums/Config/Debug.hpp
    Einsums/Config/Deleted.hpp
    Einsums/Config/ExportDefinitions.hpp
    Einsums/Config/ForceInline.hpp
    Einsums/Config/Namespace.hpp
    Einsums/Config/Types.hpp
)

set(ConfigSources Version.cpp Types.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Config CONFIG_FILES
  SOURCES ${ConfigSources}
  HEADERS ${ConfigHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Preprocessor Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
