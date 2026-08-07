//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#define FFT_BACKEND fftw3

EINSUMS_NAMESPACE_BEGIN(fft::backend::fftw3)

void scfft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result);
void ccfft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result);

void dzfft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result);
void zzfft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result);

void csifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result);
void zdifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result);

void ccifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result);
void zzifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result);

EINSUMS_NAMESPACE_END(fft::backend::fftw3)