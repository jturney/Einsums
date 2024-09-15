#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(timing_headers
        einsums/timing/detail/timestamp.hpp
        einsums/timing/detail/timestamp/bgq.hpp
        einsums/timing/detail/timestamp/cuda.hpp
        einsums/timing/detail/timestamp/linux_generic.hpp
        einsums/timing/detail/timestamp/linux_x86_32.hpp
        einsums/timing/detail/timestamp/linux_x86_64.hpp
        einsums/timing/detail/timestamp/msvc.hpp
        einsums/timing/high_resolution_timer.hpp
        einsums/timing/steady_clock.hpp
        einsums/timing/tick_counter.hpp
)

set(timing_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums timing
  GLOBAL_HEADER_GEN ON
  SOURCES ${timing_sources}
  HEADERS ${timing_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_config
        einsums_type_support
  CMAKE_SUBDIRS examples tests
)
