//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Tensor/Tensor.hpp>

namespace einsums::fft {

namespace detail {

/*************************************
 * Real or complex -> complex        *
 *************************************/
void EINSUMS_EXPORT scfft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT ccfft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result);

void EINSUMS_EXPORT dzfft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result);
void EINSUMS_EXPORT zzfft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result);

/*************************************
 * Real or complex <- complex        *
 *************************************/
void EINSUMS_EXPORT csifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result);
void EINSUMS_EXPORT zdifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result);

void EINSUMS_EXPORT ccifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT zzifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result);

} // namespace detail

/**
 * @brief Gets the frequency corresponding to each position in a frequency tensor.
 *
 * @param[in] n The number of positions in the tensor.
 * @param[in] d The scale factor for the frequency.
 * @return A tensor that contains the frequency at each position in a frequency tensor of the same size.
 *
 * @versionadded{1.0.0}
 */
auto EINSUMS_EXPORT fftfreq(int n, double d = 1.0) -> Tensor<double, 1>;

/**
 * @brief Performs the fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The output needs to have
 * more than half the size of the input.
 *
 * @versionadded{1.0.0}
 */
inline void fft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::scfft(a, result);
}

/**
 * @brief Performs the fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The output needs to be
 * at least as large as the input.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This overload now throws an exception when the inputs have invalid sizes. Previously it did
 *      not throw and was instead prone to buffer overruns.
 * @endversion
 */
inline void fft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccfft(a, result);
}

/**
 * @brief Performs the fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The output needs to have
 * more than half the size of the input.
 *
 * @versionadded{1.0.0}
 */
inline void fft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::dzfft(a, result);
}

/**
 * @brief Performs the fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The output needs to be
 * at least as large as the input.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This overload now throws an exception when the inputs have invalid sizes. Previously it did
 *      not throw and was instead prone to buffer overruns.
 * @endversion
 */
inline void fft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzfft(a, result);
}

/**
 * @brief Performs the inverse fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The input needs to have
 * more than half the size of the output.
 *
 * @versionadded{1.0.0}
 */
inline void ifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result) {
    detail::csifft(a, result);
}

/**
 * @brief Performs the inverse fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The input needs to have
 * more than half the size of the output.
 *
 * @versionadded{1.0.0}
 */
inline void ifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result) {
    detail::zdifft(a, result);
}

/**
 * @brief Performs the inverse fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The input needs to be
 * at least as large as the output.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This overload now throws an exception when the inputs have invalid sizes. Previously it did
 *      not throw and was instead prone to buffer overruns.
 * @endversion
 */
inline void ifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccifft(a, result);
}

/**
 * @brief Performs the inverse fast Fourier transform on the input tensor using the linked FFT library.
 *
 * @param[in] a The input tensor.
 * @param[out] result The output tensor.
 *
 * @throws dimension_error When the sizes of the input and output are invalid. The input needs to be
 * at least as large as the output.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This overload now throws an exception when the inputs have invalid sizes. Previously it did
 *      not throw and was instead prone to buffer overruns.
 * @endversion
 */
inline void ifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzifft(a, result);
}

} // namespace einsums::fft