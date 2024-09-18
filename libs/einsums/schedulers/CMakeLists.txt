#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(schedulers_headers
        einsums/schedulers/deadlock_detection.hpp
        einsums/schedulers/local_priority_queue_scheduler.hpp
        einsums/schedulers/local_queue_scheduler.hpp
        einsums/schedulers/lockfree_queue_backends.hpp
        einsums/schedulers/maintain_queue_wait_times.hpp
        einsums/schedulers/queue_helpers.hpp
        einsums/schedulers/queue_holder_numa.hpp
        einsums/schedulers/queue_holder_thread.hpp
        einsums/schedulers/shared_priority_queue_scheduler.hpp
        einsums/schedulers/static_priority_queue_scheduler.hpp
        einsums/schedulers/static_queue_scheduler.hpp
        einsums/schedulers/thread_queue.hpp
        einsums/schedulers/thread_queue_mc.hpp
        einsums/modules/schedulers.hpp
)

set(schedulers_sources deadlock_detection.cpp maintain_queue_wait_times.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums schedulers
  GLOBAL_HEADER_GEN OFF
  SOURCES ${schedulers_sources}
  HEADERS ${schedulers_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config einsums_assertion einsums_errors einsums_functional einsums_logging
        einsums_threading_base
  CMAKE_SUBDIRS examples tests
)
