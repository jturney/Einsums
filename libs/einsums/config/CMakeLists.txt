# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(config_headers
    einsums/Config.hpp
    einsums/config/Attributes.hpp
    einsums/config/AutoLink.hpp
    einsums/config/BranchHints.hpp
    einsums/config/CompilerFence.hpp
    einsums/config/CompilerSpecific.hpp
    einsums/config/Constexpr.hpp
    einsums/config/Debug.hpp
    einsums/config/EmulateDeleted.hpp
    einsums/config/export_definitions.hpp
    einsums/config/ForceInline.hpp
    einsums/config/Forward.hpp
    einsums/config/ManualProfiling.hpp
    einsums/config/Move.hpp
    einsums/config/ThreadsStack.hpp
    einsums/config/WarningsPrefix.hpp
    einsums/config/WarningsSuffix.hpp
)

set(config_sources Version.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums config CONFIG_FILES
    GLOBAL_HEADER_GEN OFF
    SOURCES ${config_sources}
    HEADERS ${config_headers}
    MODULE_DEPENDENCIES einsums_preprocessor
    CMAKE_SUBDIRS examples tests
)
