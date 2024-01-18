# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(itt_headers einsums/itt/ThreadName.hpp einsums/modules/itt.hpp)

set(itt_sources IttNotify.cpp ThreadName.cpp)

if(TARGET einsums_internal::amplifier)
    set(itt_notify_optional_dependencies einsums_internal::amplifier)
endif()

include(einsums_add_module)
einsums_add_module(
    einsums itt
    GLOBAL_HEADER_GEN OFF
    SOURCES ${itt_sources}
    HEADERS ${itt_headers}
    MODULE_DEPENDENCIES einsums_config
    DEPENDENCIES ${itt_notify_optional_dependencies}
    CMAKE_SUBDIRS examples tests
)
