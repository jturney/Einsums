//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Backend-neutral types between codegen-emitted protocol bindings (buffer, iterator, subscript,
// from APIARY_*_PROTOCOL_STD) and the C++ helpers they call. The emitted lambdas convert Python
// objects into these, so the helpers never include pybind11 and do not care whether the backend is
// pybind11 or nanobind. Standard library only.

#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

/// @brief One slot in a Python-style index expression.
///
/// Only the members for the active ``kind`` are meaningful. From Python syntax:
///
/// - ``t[i]`` becomes ``{Index, index = i}``
/// - ``t[i:j:k]`` becomes ``{Range, start = i, stop = j, step = k}``
/// - ``t[:]`` or ``t[...]`` becomes ``{Full}``
///
/// Negative indices are normalized against the dimension before the helper sees them.
struct SliceSpec {
    /// Which of the members below are valid.
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
/// Used both ways: returned by the ``data_fn`` of ``APIARY_BUFFER_PROTOCOL_STD`` for export to
/// NumPy, and passed to the ``set_buffer`` helper for import. Strides are in elements.
struct BufferDescriptor {
    /// Element type, as the backends' format codes name it.
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

EINSUMS_NAMESPACE_END()
