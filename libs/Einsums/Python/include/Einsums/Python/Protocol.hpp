//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Backend-neutral types that bridge user-side C++ helpers and codegen-emitted
// pybind11/nanobind binding lambdas (Plan C protocol synthesis).
//
// The einsums-pybind codegen tool emits Python protocol bindings (buffer,
// iterator, subscript) from APIARY_*_PROTOCOL_STD directives. The
// emitted lambdas live in the codegen TU (where pybind11 is available) and
// translate Python types (py::tuple, py::slice, py::buffer, ...) into the
// neutral types declared here, then call user-provided pure-C++ helpers.
//
// Goal: user-side helpers — the methods on RuntimeTensor, BlockTensor, etc.
// that the directives point at — see only these neutral types in their
// signatures. They never `#include <pybind11/...>`. Switching the codegen
// target between pybind11 and nanobind requires no changes to user code.
//
// This header has no dependencies beyond the standard library.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace einsums {

/// @brief One slot in a Python-style index expression.
///
/// ``kind`` is the discriminant. Only the members listed for the active
/// ``kind`` carry meaningful values. The others keep their defaults and must
/// not be read. The mapping from Python syntax is:
///
/// - ``t[i]`` becomes ``{Index, index = i}``
/// - ``t[i:j:k]`` becomes ``{Range, start = i, stop = j, step = k}``
/// - ``t[:]`` or ``t[...]`` becomes ``{Full}``
///
/// The codegen-emitted ``__getitem__`` and ``__setitem__`` lambdas convert
/// each slot of the incoming Python tuple, slice, or int into one
/// ``SliceSpec`` and pass the resulting vector to the user-side helper named
/// in ``APIARY_INDEX_PROTOCOL_STD``. Negative indices are normalized against
/// the parent's dim before the struct is built, so user helpers never see
/// negative values.
struct SliceSpec {
    /// Indexer kind for this slot. Acts as the discriminant that says which
    /// of the members below are valid.
    enum class Kind : std::uint8_t {
        Index, ///< Single integer index. Collapses this dimension.
        Range, ///< Slice with a half-open ``[start, stop)`` and a step. May or may not collapse.
        Full,  ///< Whole-dimension selection, from ``:`` or an expanding ``...``.
    };

    /// Active-member discriminant. Defaults to ``Full``.
    Kind kind = Kind::Full;
    /// Single index. Valid only when ``kind == Index``. Normalized to ``[0, dim)``.
    std::int64_t index = 0;
    /// Range start, inclusive. Valid only when ``kind == Range``. Normalized to ``[0, dim]``.
    std::int64_t start = 0;
    /// Range stop, exclusive. Valid only when ``kind == Range``. Normalized to ``[0, dim]``.
    std::int64_t stop = 0;
    /// Range step. Valid only when ``kind == Range``. Defaults to 1 and is never zero.
    std::int64_t step = 1;
};

/// @brief Backend-neutral description of a contiguous or strided buffer.
///
/// This type is used in both directions. On the outgoing path, from a
/// RuntimeTensor to NumPy, the ``data_fn`` named in
/// ``APIARY_BUFFER_PROTOCOL_STD`` returns one of these and the codegen lambda
/// converts it to a ``pybind11::buffer_info`` or ``nb::ndarray`` for the
/// active backend. On the incoming path, from NumPy to a RuntimeTensor for
/// bulk assignment, the codegen lambda receives a ``py::buffer`` or
/// ``nb::ndarray``, fills a ``BufferDescriptor``, and passes it to the user's
/// ``set_buffer`` helper.
///
/// Strides are in element units, matching ``TensorImpl``'s convention. The
/// backend adapter converts them to byte units when building a
/// ``buffer_info``.
struct BufferDescriptor {
    /// Element type identifier. Matches the format-string-style codes that
    /// pybind11 and nanobind use to describe scalar element types. Codegen
    /// lambdas translate to and from the backend's native dtype enum.
    enum class ScalarType : std::uint8_t {
        Unknown = 0,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float16,
        Float32,
        Float64,
        Complex64,  ///< std::complex<float>
        Complex128, ///< std::complex<double>
        Bool,
    };

    void                    *data                = nullptr;
    std::vector<std::size_t> shape               = {};
    std::vector<std::size_t> strides_in_elements = {};
    ScalarType               dtype               = ScalarType::Unknown;
    std::size_t              element_size        = 0;
    bool                     writable            = true;
};

} // namespace einsums
