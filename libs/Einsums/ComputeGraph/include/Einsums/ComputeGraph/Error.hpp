//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Error.hpp
/// @brief Structured error types for the ComputeGraph module.
///
/// Used with einsums::expected for explicit error handling.

#include <Einsums/CXX23/Expected.hpp>

#include <string>

namespace einsums::compute_graph {

/**
 * @brief Structured error for ComputeGraph operations.
 *
 * Carries a category (Kind) and human-readable message.
 * Used as the error type in expected<T, GraphError>.
 */
struct GraphError {
    enum class Kind : std::uint8_t {
        Validation, ///< Tensor validation failure (destroyed, shape mismatch)
        Parse,      ///< Einsum spec or string parsing failure
        IO,         ///< File I/O failure (load_json, save_json)
        Capture,    ///< Capture state violation (record outside capture)
        Type,       ///< Type or rank mismatch in dynamic dispatch
        Dispatch,   ///< BLAS dispatch failure
        Range,      ///< Out-of-range access (missing tensor ID)
    };

    Kind        kind;
    std::string message;

    /// Convenience constructors
    static GraphError io(std::string msg) { return {.kind = Kind::IO, .message = std::move(msg)}; }
    static GraphError parse(std::string msg) { return {.kind = Kind::Parse, .message = std::move(msg)}; }
    static GraphError validation(std::string msg) { return {.kind = Kind::Validation, .message = std::move(msg)}; }
    static GraphError capture(std::string msg) { return {.kind = Kind::Capture, .message = std::move(msg)}; }
    static GraphError type_error(std::string msg) { return {.kind = Kind::Type, .message = std::move(msg)}; }
    static GraphError range(std::string msg) { return {.kind = Kind::Range, .message = std::move(msg)}; }
};

} // namespace einsums::compute_graph
