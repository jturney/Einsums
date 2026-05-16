//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Error.hpp
/// @brief Structured error type for the Comm module.

#include <Einsums/CXX23/Expected.hpp>

#include <string>

namespace einsums::comm {

/**
 * @brief Structured error for MPI/communication operations.
 *
 * Used with expected<T, CommError> for explicit error handling.
 * Carries the MPI error code (when available) and a human-readable message.
 */
struct CommError {
    std::string message;
    int         mpi_error_code{0}; ///< MPI_SUCCESS on mock, MPI_ERR_* on real MPI

    static CommError from_message(std::string msg) { return {.message = std::move(msg), .mpi_error_code = 0}; }
    static CommError from_mpi(std::string msg, int code) { return {.message = std::move(msg), .mpi_error_code = code}; }
};

} // namespace einsums::comm
