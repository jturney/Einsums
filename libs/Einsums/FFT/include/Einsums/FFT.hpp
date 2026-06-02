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

auto EINSUMS_EXPORT fftfreq(int n, double d = 1.0) -> Tensor<double, 1>;

inline void fft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::scfft(a, result);
}

inline void fft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccfft(a, result);
}

inline void fft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::dzfft(a, result);
}

inline void fft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzfft(a, result);
}

inline void ifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result) {
    detail::csifft(a, result);
}

inline void ifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result) {
    detail::zdifft(a, result);
}

inline void ifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccifft(a, result);
}

inline void ifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzifft(a, result);
}

} // namespace einsums::fft