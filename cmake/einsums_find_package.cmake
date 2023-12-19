#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeFindDependencyMacro)

macro(einsums_find_package)
    if(EINSUMS_FIND_PACKAGE)
        find_dependency(${ARGN})
    else()
        find_package(${ARGN})
    endif()
endmacro()
