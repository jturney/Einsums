//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "FFT.hpp"

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/FFT/Defines.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include <mkl_dfti.h>

EINSUMS_NAMESPACE_BEGIN(fft::backend::mkl)

namespace {
inline void verify(MKL_LONG status) {
    if (status == DFTI_NO_ERROR)
        return;

    EINSUMS_THROW_EXCEPTION(std::runtime_error, "MKL DFTI failure: {}", status);
}
} // namespace

/*******************************************************************************
 * Forward transforms                                                          *
 *******************************************************************************/

void scfft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    verify(DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, a.dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_real  = const_cast<float *>(a.data());
    auto *x_cmplx = (MKL_Complex8 *)result->data();

    verify(DftiComputeForward(handle, x_real, x_cmplx));

    DftiFreeDescriptor(&handle);
}

void ccfft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    verify(DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, a.dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_input  = reinterpret_cast<MKL_Complex8 *>(const_cast<std::complex<float> *>(a.data()));
    auto *x_output = reinterpret_cast<MKL_Complex8 *>(result->data());

    verify(DftiComputeForward(handle, x_input, x_output));

    DftiFreeDescriptor(&handle);
}

void dzfft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    verify(DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 1, a.dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_real  = const_cast<double *>(a.data());
    auto *x_cmplx = (MKL_Complex16 *)result->data();

    verify(DftiComputeForward(handle, x_real, x_cmplx));

    DftiFreeDescriptor(&handle);
}

void zzfft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    verify(DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, a.dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_input  = reinterpret_cast<MKL_Complex16 *>(const_cast<std::complex<double> *>(a.data()));
    auto *x_output = reinterpret_cast<MKL_Complex16 *>(result->data());

    verify(DftiComputeForward(handle, x_input, x_output));

    DftiFreeDescriptor(&handle);
}

/*******************************************************************************
 * Backward transforms                                                         *
 *******************************************************************************/

void csifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    // The descriptors are odd.  You create the descriptor as if you're doing
    // a forward transform. In this case, from float -> complex<float> and
    // then you can call the compute backward function.
    verify(DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, result->dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_real  = const_cast<float *>(result->data());
    auto *x_cmplx = (MKL_Complex8 *)a.data();

    verify(DftiComputeBackward(handle, x_cmplx, x_real));

    DftiFreeDescriptor(&handle);
}

void zdifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    verify(DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 1, result->dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_real  = const_cast<double *>(result->data());
    auto *x_cmplx = (MKL_Complex16 *)a.data();

    verify(DftiComputeBackward(handle, x_cmplx, x_real));

    DftiFreeDescriptor(&handle);
}

void ccifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    verify(DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, result->dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_source = (MKL_Complex8 *)a.data();
    auto *x_target = (MKL_Complex8 *)result->data();

    verify(DftiComputeBackward(handle, x_source, x_target));

    DftiFreeDescriptor(&handle);
}

void zzifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    LabeledSection0();

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    verify(DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, result->dim(0)));
    verify(DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    verify(DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    verify(DftiCommitDescriptor(handle));

    auto *x_source = (MKL_Complex16 *)a.data();
    auto *x_target = (MKL_Complex16 *)result->data();

    verify(DftiComputeBackward(handle, x_source, x_target));

    DftiFreeDescriptor(&handle);
}

EINSUMS_NAMESPACE_END(fft::backend::mkl)