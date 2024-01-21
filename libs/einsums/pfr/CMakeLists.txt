# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(pfr_headers
    einsums/pfr.hpp
    einsums/pfr/ops_fields.hpp
    einsums/pfr/functors.hpp
    einsums/pfr/core_name.hpp
    einsums/pfr/traits.hpp
    einsums/pfr/io.hpp
    einsums/pfr/io_fields.hpp
    einsums/pfr/config.hpp
    einsums/pfr/detail/core_name20_static.hpp
    einsums/pfr/detail/sequence_tuple.hpp
    einsums/pfr/detail/core17_generated.hpp
    einsums/pfr/detail/core_name14_disabled.hpp
    einsums/pfr/detail/size_t_.hpp
    einsums/pfr/detail/stdtuple.hpp
    einsums/pfr/detail/offset_based_getter.hpp
    einsums/pfr/detail/stdarray.hpp
    einsums/pfr/detail/core_name.hpp
    einsums/pfr/detail/rvalue_t.hpp
    einsums/pfr/detail/io.hpp
    einsums/pfr/detail/config.hpp
    einsums/pfr/detail/make_flat_tuple_of_references.hpp
    einsums/pfr/detail/unsafe_declval.hpp
    einsums/pfr/detail/cast_to_layout_compatible.hpp
    einsums/pfr/detail/functional.hpp
    einsums/pfr/detail/for_each_field_impl.hpp
    einsums/pfr/detail/core14_loophole.hpp
    einsums/pfr/detail/core17.hpp
    einsums/pfr/detail/core14_classic.hpp
    einsums/pfr/detail/size_array.hpp
    einsums/pfr/detail/fields_count.hpp
    einsums/pfr/detail/fake_object.hpp
    einsums/pfr/detail/core.hpp
    einsums/pfr/detail/make_integer_sequence.hpp
    einsums/pfr/detail/possible_reflectable.hpp
    einsums/pfr/detail/tie_from_structure_tuple.hpp
    einsums/pfr/detail/detectors.hpp
    einsums/pfr/traits_fwd.hpp
    einsums/pfr/functions_for.hpp
    einsums/pfr/ops.hpp
    einsums/pfr/tuple_size.hpp
    einsums/pfr/core.hpp
)

set(pfr_sources)

include(einsums_add_module)
einsums_add_module(
    einsums pfr
    GLOBAL_HEADER_GEN OFF
    SOURCES ${pfr_sources}
    HEADERS ${pfr_headers}
    MODULE_DEPENDENCIES
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
