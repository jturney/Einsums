#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ConfigHeaders
    Einsums/Config.hpp
    Einsums/Config/ABI.hpp
    Einsums/Config/Alias.hpp
    Einsums/Config/BranchHints.hpp
    Einsums/Config/CompilerSpecific.hpp
    Einsums/Config/Debug.hpp
    Einsums/Config/ExportDefinitions.hpp
    Einsums/Config/ForceInline.hpp
    Einsums/Config/Namespace.hpp
)

set(ConfigSources ABI.cpp Version.cpp)

# The git identity goes to a private header under `generated/`, never under
# `include/`: anything below `include/` is on the public include path, is
# installed, and is swept by the zombie-header cleanup in einsums_add_module.
# Only Config/src/Version.cpp includes this, which is what keeps a commit from
# invalidating every translation unit's ccache entry. See GitInfo.hpp.in.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/GitInfo.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/GitInfo.hpp" @ONLY
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Config CONFIG_FILES
  SOURCES ${ConfigSources}
  HEADERS ${ConfigHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Preprocessor Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)

target_include_directories(Einsums_Config PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
