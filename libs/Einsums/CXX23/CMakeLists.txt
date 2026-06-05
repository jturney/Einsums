#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(CXX23Headers
    Einsums/CXX23.hpp
    Einsums/CXX23/Expected.hpp
    Einsums/CXX23/FlatMap.hpp
    Einsums/CXX23/FlatSet.hpp
    Einsums/CXX23/Unreachable.hpp
)

# Header-only module — no source files
include(Einsums_AddModule)
einsums_add_module(
  Einsums CXX23
  HEADERS ${CXX23Headers}
  MODULE_DEPENDENCIES
    Einsums_Config
  CMAKE_SUBDIRS tests
)
