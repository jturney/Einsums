#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeFindDependencyMacro)

macro(einsums_find_package)
  if(EINSUMS_FIND_PACKAGE)
    find_dependency(${ARGV})
  else()
    find_package(${ARGV})
  endif()
endmacro()
