#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(execution_headers
        einsums/execution/algorithms/bulk.hpp
        einsums/execution/algorithms/continues_on.hpp
        einsums/execution/algorithms/detail/helpers.hpp
        einsums/execution/algorithms/detail/partial_algorithm.hpp
        einsums/execution/algorithms/drop_operation_state.hpp
        einsums/execution/algorithms/drop_value.hpp
        einsums/execution/algorithms/ensure_started.hpp
        einsums/execution/algorithms/execute.hpp
        einsums/execution/algorithms/just.hpp
        einsums/execution/algorithms/let_error.hpp
        einsums/execution/algorithms/let_value.hpp
        einsums/execution/algorithms/require_started.hpp
        einsums/execution/algorithms/schedule_from.hpp
        einsums/execution/algorithms/split.hpp
        einsums/execution/algorithms/split_tuple.hpp
        einsums/execution/algorithms/start_detached.hpp
        einsums/execution/algorithms/sync_wait.hpp
        einsums/execution/algorithms/then.hpp
        einsums/execution/algorithms/transfer_just.hpp
        einsums/execution/algorithms/transfer_when_all.hpp
        einsums/execution/algorithms/unpack.hpp
        einsums/execution/algorithms/when_all.hpp
        einsums/execution/algorithms/when_all_vector.hpp
        einsums/execution/scheduler_queries.hpp
)

set(execution_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums execution
        GLOBAL_HEADER_GEN ON
        SOURCES ${execution_sources}
        HEADERS ${execution_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES
        einsums_config
        einsums_threading
        einsums_errors
        einsums_memory
        einsums_thread_support
        einsums_timing
        einsums_topology
        CMAKE_SUBDIRS examples tests
)
