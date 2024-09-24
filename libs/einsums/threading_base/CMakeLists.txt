#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_base_headers
    einsums/threading_base/annotated_function.hpp
    einsums/threading_base/callback_notifier.hpp
    einsums/threading_base/create_thread.hpp
    einsums/threading_base/create_work.hpp
    einsums/threading_base/detail/external_timer/apex.hpp
    einsums/threading_base/detail/external_timer/default.hpp
    einsums/threading_base/detail/get_default_pool.hpp
    einsums/threading_base/detail/global_activity_count.hpp
    einsums/threading_base/detail/reset_backtrace.hpp
    einsums/threading_base/detail/reset_lco_description.hpp
        einsums/threading_base/detail/thread_data_stackful.hpp
        einsums/threading_base/detail/thread_data_stackless.hpp
    einsums/threading_base/detail/tracy.hpp
    einsums/threading_base/execution_agent.hpp
    einsums/threading_base/external_timer.hpp
    einsums/threading_base/print.hpp
    einsums/threading_base/register_thread.hpp
    einsums/threading_base/scheduler_base.hpp
    einsums/threading_base/scheduler_mode.hpp
    einsums/threading_base/scheduler_state.hpp
    einsums/threading_base/scoped_annotation.hpp
    einsums/threading_base/set_thread_state.hpp
    einsums/threading_base/set_thread_state_timed.hpp
    einsums/threading_base/thread_data.hpp
        einsums/threading_base/detail/thread_data_stackful.hpp
        einsums/threading_base/detail/thread_data_stackless.hpp
    einsums/threading_base/thread_description.hpp
    einsums/threading_base/thread_helpers.hpp
    einsums/threading_base/thread_init_data.hpp
    einsums/threading_base/thread_num_tss.hpp
    einsums/threading_base/thread_pool_base.hpp
    einsums/threading_base/thread_queue_init_parameters.hpp
    einsums/threading_base/thread_specific_ptr.hpp
        einsums/threading_base/threading_base_fwd.hpp
)

set(threading_base_sources
    annotated_function.cpp
    create_thread.cpp
    create_work.cpp
    execution_agent.cpp
    external_timer_apex.cpp
    get_default_pool.cpp
    global_activity_count.cpp
    print.cpp
    reset_backtrace.cpp
    reset_lco_description.cpp
    scheduler_base.cpp
    scheduler_mode.cpp
    set_thread_state.cpp
    set_thread_state_timed.cpp
    thread_data.cpp
    thread_data_stackful.cpp
    thread_data_stackless.cpp
    thread_description.cpp
    thread_helpers.cpp
    thread_num_tss.cpp
    thread_pool_base.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums threading_base
  GLOBAL_HEADER_GEN ON
        EXCLUDE_FROM_GLOBAL_HEADER "einsums/threading_base/detail/thread_data_stackful.hpp"
        "einsums/threading_base/detail/thread_data_stackless.hpp"
  SOURCES ${threading_base_sources}
  HEADERS ${threading_base_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    einsums_affinity
    einsums_allocator_support
    einsums_assertion
    einsums_execution_base
    einsums_config
    einsums_concurrency
    einsums_coroutines
    einsums_debugging
    einsums_errors
    einsums_functional
    einsums_itt_notify
    einsums_lock_registration
    einsums_logging
    einsums_memory
    einsums_timing
    einsums_type_support
  CMAKE_SUBDIRS examples tests
)
