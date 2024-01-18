//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>

namespace einsums {

/// \cond NOINTERNAL
// forward declaration
struct error_code;
struct EINSUMS_EXPORT exception;
struct EINSUMS_EXPORT thread_interrupted;
/// \endcond

////////////////////////////////////////////////////////////////////////////////
/// \brief Encode error category for new error_code
enum class throwmode {
    plain       = 0,
    rethrow     = 1,
    lightweight = 0x80, // do not generate an exception for this error_code
    /// \cond NODETAIL
    lightweight_rethrow = lightweight | rethrow
    /// \endcond
};

///////////////////////////////////////////////////////////////////////////
/// \brief Predefined error_code object used as "throw on error" tag.
///
/// The predefined einsums::error_code object \a einsums::throws is supplied for use as
/// a "throw on error" tag.
///
/// Functions that specify an argument in the form 'error_code& ec=throws'
/// (with appropriate namespace qualifiers), have the following error
/// handling semantics:
///
/// If &ec != &throws and an error occurred: ec.value() returns the
/// implementation specific error number for the particular error that
/// occurred and ec.category() returns the error_category for ec.value().
///
/// If &ec != &throws and an error did not occur, ec.clear().
///
/// If an error occurs and &ec == &throws, the function throws an exception
/// of type \a einsums::exception or of a type derived from it. The exception's
/// \a get_errorcode() member function returns a reference to an
/// \a einsums::error_code object with the behavior as specified above.
///
EINSUMS_EXPORT extern error_code throws;

} // namespace einsums

#include <einsums/errors/throw_exception.hpp>
