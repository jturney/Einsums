//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TypeSupport/StringLiteral.hpp>

#include <source_location>
#include <stdexcept>
#include <string>

EINSUMS_NAMESPACE_BEGIN()

namespace detail {

/**
 * Construct a message that contains the type of error being produced, the location that error is being emitted,
 * and the actual message for the error.
 *
 * @param type_name The name of the type producing the error.
 * @param str The message for the error.
 * @param location The source location that the error is being emitted.
 *
 * @return A message with this extra debugging info.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT std::string make_error_message(std::string_view const &type_name, char const *str, std::source_location const &location);

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
template <size_t N>
std::string make_error_message(StringLiteral<N> const type_name, char const *str, std::source_location const &location) {
    return make_error_message(type_name.string_view(), str, location);
}

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
EINSUMS_EXPORT std::string make_error_message(std::string_view const &type_name, std::string const &str,
                                              std::source_location const &location);

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
template <size_t N>
std::string make_error_message(StringLiteral<N> const type_name, std::string const &str, std::source_location const &location) {
    return make_error_message(type_name.string_view(), str, location);
}

} // namespace detail

/**
 * @struct CodedError
 *
 * Distinguishes several throws of the same error type from one function: catch the base class to
 * handle them together, or a specific CodedError to handle one. The TiledTensor gemm, for
 * instance, throws TensorCompatError for both an incompatible output grid and an incompatible
 * inner grid. Both at once:
 *
 * @code
 * TiledTensor<double, 2> A, B, C;
 * try {
 *     gemm<false, false>(1.0, A, B, 0.0, &C);
 * } catch(TensorCompatError &exc) {
 *     // Handle both errors.
 * }
 * @endcode
 *
 * Or each individually:
 *
 * @code
 * try {
 *     gemm<false, false>(1.0, A, B, 0.0, &C);
 * } catch(CodedError<TensorCompatError, 0> &exc) {
 *     // Output doesn't have a compatible grid, so fix that.
 * } catch(CodedError<TensorCompatError, 1> &exc) {
 *     // Input doesn't have a compatible grid, so fix that.
 * }
 * @endcode
 *
 * @tparam ErrorClass The kind of error the object wraps.
 * @tparam ErrorCode The identifier for the error.
 *
 * @versionadded{1.0.0}
 */
template <class ErrorClass, int ErrorCode>
struct CodedError : ErrorClass {
    using ErrorClass::ErrorClass;

    /**
     * Get the error code for this exception
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] constexpr int get_code() const { return ErrorCode; }
};

/**
 * @struct RankError
 *
 * Indicates that the rank of some tensor arguments are not compatible with the given operation.
 *
 * @versionadded{1.1.0}
 */
struct EINSUMS_EXPORT RankError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct DimensionError
 *
 * Indicates that the dimensions of some tensor arguments are not compatible with the given operation.
 * For instance, you can only take the determinant of a square matrix, so passing a matrix that is not
 * square to linear_algebra::det will result in this error.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT DimensionError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct TensorCompatError
 *
 * Indicates that two or more tensors are not compatible with each other for the requested operation,
 * such as a gemm whose inner dimensions differ.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT TensorCompatError : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct NumArgumentError
 *
 * Indicates that a function received the wrong number of arguments, or an array of them of the
 * wrong size, as in RuntimeTensor subscripts. See NotEnoughArgs and TooManyArgs.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT NumArgumentError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct NotEnoughArgs
 *
 * Indicates that a function did not receive enough arguments. Child of NumArgumentError .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT NotEnoughArgs : NumArgumentError {
    using NumArgumentError::NumArgumentError;
};

/**
 * @struct TooManyArgs
 *
 * Indicates that a function received too many arguments. Child of NumArgumentError .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT TooManyArgs : NumArgumentError {
    using NumArgumentError::NumArgumentError;
};

/**
 * @struct AccessDenied
 *
 * Indicates that an operation was stopped due to access restrictions, mostly in the HDF5 code:
 * writing read-only data, or accessing a file without permission.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT AccessDenied : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct TodoError
 *
 * Indicates that a certain code path is not yet finished.
 *
 * This exception, along with
 * NotImplemented , indicates that the action you requested is not yet implemented. If you get
 * this error, come tell us `on our issue tracker <https://github.com/Einsums/Einsums/issues>`_
 * or `our Discord server <https://discord.gg/8GvtkyWZUv>`_, and we will try to focus some energy
 * to filling it out. If you are an experienced C++ programmer, we would appreciate your
 * assistance if you think you have a solution.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT TodoError : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct NotImplemented
 *
 * Indicates that a certain code path is not implemented.
 *
 * This may be because the feature is not
 * yet ready, or it may be that the specific combination of parameters is not acceptable. The
 * message provided should give more information. If you absolutely need that set of features,
 * come tell us `on our discussion page <https://github.com/Einsums/Einsums/discussions>`_ or
 * `our Discord server <https://discord.gg/8GvtkyWZUv>`_, and we will try to work it out. If you
 * are an experienced C++ programmer, we would appreciate your assistance if you think you have
 * a solution.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT NotImplemented : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct BadLogic
 *
 * Means the same as std::logic_error, but can be caught without catching everything derived from it.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT BadLogic : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct UninitializedError
 *
 * Indicates that the code is handling data that is uninitialized. This is usually thrown when
 * Einsums was not initialized.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT UninitializedError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct SystemError
 *
 * Indicates that an error happened when making a system call, or that some system utility
 * failed. For instance, it can be thrown when trying to find a file that doesn't exist.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT SystemError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct EnumError
 *
 * Indicates that an invalid enum value was passed to a function.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT EnumError : std::domain_error {
    using std::domain_error::domain_error;
};

/**
 * @struct ComplexConversionError
 *
 * Thrown when converting a complex number to a real one. Take the magnitude or real part instead,
 * as suits the operation.
 *
 * @versionadded{2.0.0}
 */
struct EINSUMS_EXPORT ComplexConversionError : std::logic_error {
    using std::logic_error::logic_error;
};

EINSUMS_NAMESPACE_END()
