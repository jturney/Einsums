//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/FFT.hpp>
#include <Einsums/FFT/Defines.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/ARange.hpp>

#if defined(EINSUMS_HAVE_FFT_LIBRARY_FFTW3)
#    include "Backends/FFTW3/FFT.hpp"
#elif defined(EINSUMS_HAVE_FFT_LIBRARY_MKL)
#    include "Backends/MKL/FFT.hpp"
#endif

namespace einsums::fft {

auto fftfreq(int n, double d) -> Tensor<double, 1> {
    double value   = 1.0 / (n * d);
    auto   results = create_tensor("FFTFreq", n);

    int  N               = (n - 1) / 2 + 1;
    auto p1              = arange<double>(0, N);
    results(Range(0, N)) = p1;

    auto p2              = arange<double>(-(n / 2), 0); // NOLINT
    results(Range(N, n)) = p2;

    linear_algebra::scale(value, &results);

    return results;
}

namespace detail {

namespace {
template <typename T>
void check_size(Tensor<T, 1> const &a, Tensor<std::complex<T>, 1> const *result) {
    if (result->dim(0) >= a.dim(0) / 2 + 1)
        return;

    EINSUMS_THROW_EXCEPTION(dimension_error, "fft called with too small result tensor size\nsize of \"{}\" is {}\nsize of \"{}\" is {}",
                            a.name(), a.dim(0), result->name(), result->dim(0));
}

template <typename T>
void icheck_size(Tensor<std::complex<T>, 1> const &a, Tensor<T, 1> const *result) {
    if (a.dim(0) >= result->dim(0) / 2 + 1)
        return;

    EINSUMS_THROW_EXCEPTION(dimension_error, "fft called with too small result tensor size\nsize of \"{}\" is {}\nsize of \"{}\" is {}",
                            a.name(), a.dim(0), result->name(), result->dim(0));
}

template <typename T>
void check_size(Tensor<std::complex<T>, 1> const &a, Tensor<std::complex<T>, 1> const *result) {
    if (result->dim(0) >= a.dim(0))
        return;

    EINSUMS_THROW_EXCEPTION(dimension_error, "fft called with too small result tensor size\nsize of \"{}\" is {}\nsize of \"{}\" is {}",
                            a.name(), a.dim(0), result->name(), result->dim(0));
}

template <typename T>
void icheck_size(Tensor<std::complex<T>, 1> const &a, Tensor<std::complex<T>, 1> const *result) {
    if (a.dim(0) >= result->dim(0))
        return;

    EINSUMS_THROW_EXCEPTION(dimension_error, "fft called with too small result tensor size\nsize of \"{}\" is {}\nsize of \"{}\" is {}",
                            a.name(), a.dim(0), result->name(), result->dim(0));
}
} // namespace

void scfft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    check_size(a, result);
    // backend::mkl::scfft(a, result);
    backend::FFT_BACKEND::scfft(a, result);
}

void ccfft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    check_size(a, result);
    // backend::mkl::ccfft(a, result);
    backend::FFT_BACKEND::ccfft(a, result);
}

void dzfft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    check_size(a, result);
    // backend::mkl::dzfft(a, result);
    backend::FFT_BACKEND::dzfft(a, result);
}

void zzfft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    check_size(a, result);
    // backend::mkl::zzfft(a, result);
    backend::FFT_BACKEND::zzfft(a, result);
}

void csifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::csifft(a, result);
    backend::FFT_BACKEND::csifft(a, result);
}

void zdifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::zdifft(a, result);
    backend::FFT_BACKEND::zdifft(a, result);
}

void ccifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::ccifft(a, result);
    backend::FFT_BACKEND::ccifft(a, result);
}

void zzifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::zzifft(a, result);
    backend::FFT_BACKEND::zzifft(a, result);
}

} // namespace detail

} // namespace einsums::fft