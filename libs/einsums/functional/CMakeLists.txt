# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(functional_headers
    einsums/functional/detail/vtable/copyable_vtable.hpp
    einsums/functional/detail/vtable/vtable.hpp
    einsums/functional/detail/invoke.hpp
    einsums/functional/traits/get_function_address.hpp
    einsums/functional/traits/get_function_annotation.hpp
    einsums/function/traits/is_action.hpp
    einsums/function/traits/is_bind_expression.hpp
)

set(functional_sources basic_function.cpp empty_function.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums functional
    GLOBAL_HEADER_GEN ON
    SOURCES ${functional_sources}
    HEADERS ${functional_headers}
    MODULE_DEPENDENCIES einsums_assertion einsums_config einsums_errors einsums_itt einsums_type_support
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
