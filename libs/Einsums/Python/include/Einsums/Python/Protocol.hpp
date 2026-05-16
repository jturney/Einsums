//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Backend-neutral types that bridge user-side C++ helpers and codegen-emitted
// pybind11/nanobind binding lambdas (Plan C protocol synthesis).
//
// The einsums-pybind codegen tool emits Python protocol bindings (buffer,
// iterator, subscript) from EINSUMS_PYBIND_*_PROTOCOL_STD directives. The
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
/// `t[i]`           → ``{Index, index = i}``
/// `t[i:j:k]`       → ``{Range, start = i, stop = j, step = k}``
/// `t[:]` / `t[...]`→ ``{Full}``
///
/// Codegen-emitted ``__getitem__`` / ``__setitem__`` lambdas convert each
/// slot of the incoming Python tuple/slice/int into one ``SliceSpec`` and
/// pass the resulting vector to the user-side helper named in
/// ``EINSUMS_PYBIND_INDEX_PROTOCOL_STD``. Negative indices are normalized
/// against the parent's dim before this struct is built — user helpers
/// never see negative values.
struct SliceSpec {
    /// What kind of indexer this slot represents.
    enum class Kind {
        Index, ///< Single integer index — collapses this dimension.
        Range, ///< Slice with [start, stop) and step. May or may not collapse.
        Full,  ///< Whole-dimension selection (``:`` or ``...`` expanding here).
    };

    Kind kind = Kind::Full;
    /// Valid when kind == Index. Already normalized to [0, dim).
    std::int64_t index = 0;
    /// Valid when kind == Range. Already normalized to [0, dim].
    std::int64_t start = 0;
    /// Valid when kind == Range. Already normalized to [0, dim].
    std::int64_t stop = 0;
    /// Valid when kind == Range. Defaults to 1; never zero.
    std::int64_t step = 1;
};

/// @brief Backend-neutral description of a contiguous-or-strided buffer.
///
/// Used in both directions:
/// - **Outgoing** (RuntimeTensor → NumPy): the ``data_fn`` named in
///   ``EINSUMS_PYBIND_BUFFER_PROTOCOL_STD`` returns one of these; the
///   codegen lambda converts to ``pybind11::buffer_info`` /
///   ``nb::ndarray`` per backend.
/// - **Incoming** (NumPy → RuntimeTensor for bulk-assign): the codegen
///   lambda receives ``py::buffer`` / ``nb::ndarray``, materializes a
///   ``BufferDescriptor``, and passes it to the user's ``set_buffer``
///   helper.
///
/// Strides are in **element units** (matching ``TensorImpl``'s
/// convention). The backend adapter converts to byte units when building
/// ``buffer_info``.
struct BufferDescriptor {
    /// Element type identifier — matches the format-string-style codes
    /// pybind11 / nanobind use to describe scalar element types.
    /// Codegen lambdas translate to/from the backend's native dtype enum.
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
