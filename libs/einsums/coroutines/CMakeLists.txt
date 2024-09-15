#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(coroutines_headers
  einsums/coroutines/coroutine_fwd.hpp
  einsums/coroutines/thread_enums.hpp
  einsums/coroutines/detail/combined_tagged_state.hpp
  einsums/coroutines/detail/coroutine_accessor.hpp
  einsums/coroutines/detail/get_stack_pointer.hpp
  einsums/coroutines/detail/posix_utility.hpp
  einsums/coroutines/detail/swap_context.hpp
)

set(coroutines_sources
  thread_enums.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums coroutines
  GLOBAL_HEADER_GEN ON
  SOURCES ${coroutines_sources}
  HEADERS ${coroutines_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_assertion
        einsums_config
        einsums_errors
        einsums_functional
        einsums_memory
        einsums_thread_support
        einsums_type_support
        einsums_util
  CMAKE_SUBDIRS examples tests
)
