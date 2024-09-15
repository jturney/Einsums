#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(functional_headers
        einsums/functional/bind.hpp
        einsums/functional/bind_back.hpp
        einsums/functional/bind_front.hpp
        einsums/functional/deferred_call.hpp
        einsums/functional/first_argument.hpp
        einsums/functional/function.hpp
        einsums/functional/invoke.hpp
        einsums/functional/one_shot.hpp
        einsums/functional/unique_function.hpp
        einsums/functional/detail/basic_functon.hpp
        einsums/functional/detail/empty_function.hpp
        einsums/functional/detail/invoke_result_plain_function.hpp
        einsums/functional/detail/reset_function.hpp
        einsums/functional/detail/vtable/callable_vtable.hpp
        einsums/functional/detail/vtable/copyable_vtable.hpp
        einsums/functional/detail/vtable/function_vtable.hpp
        einsums/functional/detail/vtable/vtable.hpp
        einsums/functional/traits/get_function_address.hpp
        einsums/functional/traits/get_function_annotation.hpp
        einsums/functional/traits/is_bind_expression.hpp
)

set(functional_sources
        basic_function.cpp
        empty_function.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums functional
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER
        "einsums/functional/traits/get_function_address.hpp"
        "einsums/functional/traits/get_function_annotation.hpp"
        "einsums/functional/traits/is_bind_expression.hpp"
  SOURCES ${functional_sources}
  HEADERS ${functional_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_assertion
        einsums_config
        einsums_debugging
        einsums_errors
        einsums_type_support
  CMAKE_SUBDIRS examples tests
)
