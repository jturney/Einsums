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
 * This error type is used when a function can emit several different instances of the
 * same error.
 *
 * This allows the user to either catch the class the code is based on,
 * or the CodedError with the code specified. This means that the user can
 * handle all errors with a similar cause together, or gain more fine-grained control
 * if needed.
 *
 * As an example, in some specializations of the gemm call, multiple tensor_compat_error s
 * can be thrown. For the TiledTensor version, for instance, a tensor_compat_error can be
 * thrown if either the output tensor's grid doesn't match what the input tensors require,
 * or if the inner input tensor dimension's grid doesn't match what is required. If you
 * wanted to catch both of these at once, you can use something like the following.
 *
 * @code
 * TiledTensor<double, 2> A, B, C;
 * try {
 *     gemm<false, false>(1.0, A, B, 0.0, &C);
 * } catch(tensor_compat_error &exc) {
 *     // Handle both errors.
 * }
 * @endcode
 *
 * Or you can handle each individually, like here.
 *
 * @code
 * try {
 *     gemm<false, false>(1.0, A, B, 0.0, &C);
 * } catch(CodedError<tensor_compat_error, 0> &exc) {
 *     // Output doesn't have a compatible grid, so fix that.
 * } catch(CodedError<tensor_compat_error, 1> &exc) {
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
    constexpr int get_code() const { return ErrorCode; }
};

/**
 * @struct rank_error
 *
 * Indicates that the rank of some tensor arguments are not compatible with the given operation.
 *
 * @versionadded{1.1.0}
 */
struct EINSUMS_EXPORT rank_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct dimension_error
 *
 * Indicates that the dimensions of some tensor arguments are not compatible with the given operation.
 * For instance, you can only take the determinant of a square matrix, so passing a matrix that is not
 * square to linear_algebra::det will result in this error.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT dimension_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct tensor_compat_error
 *
 * Indicates that two or more tensors are not compatible with each other for the requested operation.
 *
 * For instance, matrix multiplication is only allowed for matrices of certain dimensions: for some
 * natural numbers n, m, and k, the only allowed contraction is of the form (n by k) times (k by m)
 * giving (n by m). If you were to pass a 3-by-2 matrix as the first matrix argument, a 1-by-4 matrix
 * as the second matrix argument, and a 3-by-5 matrix as the third matrix argument to
 * linear_algebra::gemm, this error will be thrown because the inner dimensions of the input arguments
 * don't match (2 is not 1), and the outer dimensions of the input matrices and the dimensions of the
 * output matrix don't match either (3 is 3, but 4 is not 5).
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT tensor_compat_error : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct num_argument_error
 *
 * Indicates that a function that can receive a variable number of arguments did not receive the
 * right number of arguments.
 *
 * This is especially used in the RuntimeTensor subscript functions,
 * where the number of indices needed is not known at compile time, so compile-time checks can't
 * be made. This exception has two specializations, not_enough_args and too_many_args , for not
 * enough and too many arguments. It is also thrown when a function takes an array of values that
 * are treated as individual arguments, but the array is too small or too big.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT num_argument_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct not_enough_args
 *
 * Indicates that a function did not receive enough arguments. Child of num_argument_error .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT not_enough_args : num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct too_many_args
 *
 * Indicates that a function received too many arguments. Child of num_argument_error .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT too_many_args : num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct access_denied
 *
 * Indicates that an operation was stopped due to access restrictions. This exception is mostly
 * thrown by the HDF5 compatibility code, where it indicates an illegal action on a file object,
 * such as writing read-only data, or accessing a file without the required permissions.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT access_denied : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct todo_error
 *
 * Indicates that a certain code path is not yet finished.
 *
 * This exception, along with
 * not_implemented , indicates that the action you requested is not yet implemented. If you get
 * this error, come tell us `on our issue tracker <https://github.com/Einsums/Einsums/issues>`_
 * or `our Discord server <https://discord.gg/8GvtkyWZUv>`_, and we will try to focus some energy
 * to filling it out. If you are an experienced C++ programmer, we would appreciate your
 * assistance if you think you have a solution.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT todo_error : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct not_implemented
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
struct EINSUMS_EXPORT not_implemented : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct bad_logic
 *
 * Indicates that an error occurred for some unspecified reason.
 *
 * It means
 * the same as std::logic_error. However, since std::logic_error is the base class for so many
 * exceptions, this specialization is provided so that you can catch it specifically without
 * also matching every exception derived from std::logic_error.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT bad_logic : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct uninitialized_error
 *
 * Indicates that the code is handling data that is uninitialized. This is usually thrown when
 * Einsums was not initialized.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT uninitialized_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct system_error
 *
 * Indicates that an error happened when making a system call, or that some system utility
 * failed. For instance, it can be thrown when trying to find a file that doesn't exist.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT system_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct enum_error
 *
 * Indicates that an invalid enum value was passed to a function.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT enum_error : std::domain_error {
    using std::domain_error::domain_error;
};

/**
 * @struct complex_conversion_error
 *
 * Thrown when trying to convert a complex number to a real number. Instead, the input
 * data should be transformed into a real value in a way that makes sense for the operation
 * being performed. This is often either the magnitude or the real part.
 *
 * @versionadded{2.0.0}
 */
struct EINSUMS_EXPORT complex_conversion_error : std::logic_error {
    using std::logic_error::logic_error;
};

EINSUMS_NAMESPACE_END()
