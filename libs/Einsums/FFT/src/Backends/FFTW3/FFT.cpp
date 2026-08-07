//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "FFT.hpp"

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/FFT/Defines.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#if defined(EINSUMS_HAVE_FFT_LIBRARY_MKL)
#    include <fftw/fftw3.h>
#elif defined(EINSUMS_HAVE_FFT_LIBRARY_FFTW3)
#    include <fftw3.h>
#else
#    error Unable to find FFTW header.
#endif

EINSUMS_NAMESPACE_BEGIN(fft::backend::fftw3)

namespace {
template <typename Plan>
void verify(Plan plan) {
    if (plan == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "einsums::backend::fftw3::XXfft: Unable to create FFTW plan.");
    }
}
} // namespace

/*******************************************************************************
 * Forward transforms                                                          *
 *******************************************************************************/

void scfft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    LabeledSection0();

    fftwf_plan r2c = nullptr;

    verify((r2c = fftwf_plan_dft_r2c_1d(a.dim(0), const_cast<float *>(a.data()), reinterpret_cast<fftwf_complex *>(result->data()),
                                        FFTW_ESTIMATE)));
    fftwf_execute(r2c);
    fftwf_destroy_plan(r2c);
}

void ccfft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    LabeledSection0();

    fftwf_plan c2c = nullptr;

    verify((c2c = fftwf_plan_dft_1d(a.dim(0), reinterpret_cast<fftwf_complex *>(const_cast<std::complex<float> *>(a.data())),
                                    reinterpret_cast<fftwf_complex *>(result->data()), FFTW_FORWARD, FFTW_ESTIMATE)));
    fftwf_execute(c2c);
    fftwf_destroy_plan(c2c);
}

void dzfft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    LabeledSection0();

    fftw_plan r2c = nullptr;

    verify((r2c = fftw_plan_dft_r2c_1d(a.dim(0), const_cast<double *>(a.data()), reinterpret_cast<fftw_complex *>(result->data()),
                                       FFTW_ESTIMATE)));
    fftw_execute(r2c);
    fftw_destroy_plan(r2c);
}

void zzfft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    LabeledSection0();

    fftw_plan c2c = nullptr;

    verify((c2c = fftw_plan_dft_1d(a.dim(0), reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(a.data())),
                                   reinterpret_cast<fftw_complex *>(result->data()), FFTW_FORWARD, FFTW_ESTIMATE)));
    fftw_execute(c2c);
    fftw_destroy_plan(c2c);
}

/*******************************************************************************
 * Backward transforms                                                         *
 *******************************************************************************/
void csifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result) {
    LabeledSection0();

    fftwf_plan c2r = nullptr;

    verify(c2r = fftwf_plan_dft_c2r_1d(result->dim(0), reinterpret_cast<fftwf_complex *>(const_cast<std::complex<float> *>(a.data())),
                                       result->data(), FFTW_ESTIMATE));
    fftwf_execute(c2r);
    fftwf_destroy_plan(c2r);
}

void zdifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result) {
    LabeledSection0();

    fftw_plan c2r = nullptr;

    verify(c2r = fftw_plan_dft_c2r_1d(result->dim(0), reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(a.data())),
                                      result->data(), FFTW_ESTIMATE));
    fftw_execute(c2r);
    fftw_destroy_plan(c2r);
}

void ccifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    LabeledSection0();

    fftwf_plan c2r = nullptr;

    verify(c2r = fftwf_plan_dft_1d(result->dim(0), reinterpret_cast<fftwf_complex *>(const_cast<std::complex<float> *>(a.data())),
                                   reinterpret_cast<fftwf_complex *>(result->data()), FFTW_BACKWARD, FFTW_ESTIMATE));
    fftwf_execute(c2r);
    fftwf_destroy_plan(c2r);
}

void zzifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    LabeledSection0();

    fftw_plan c2c = nullptr;

    verify(c2c = fftw_plan_dft_1d(result->dim(0), reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(a.data())),
                                  reinterpret_cast<fftw_complex *>(result->data()), FFTW_BACKWARD, FFTW_ESTIMATE));
    fftw_execute(c2c);
    fftw_destroy_plan(c2c);
}

EINSUMS_NAMESPACE_END(fft::backend::fftw3)