//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <complex>

namespace einsums::linear_algebra::detail {
template void impl_direct_product<float, float, float>(float alpha, einsums::detail::TensorImpl<float> const &A,
                                                       einsums::detail::TensorImpl<float> const &B, float beta,
                                                       einsums::detail::TensorImpl<float> *C);
template void impl_direct_product<double, double, double>(double alpha, einsums::detail::TensorImpl<double> const &A,
                                                          einsums::detail::TensorImpl<double> const &B, double beta,
                                                          einsums::detail::TensorImpl<double> *C);
template void impl_direct_product<std::complex<float>, std::complex<float>, std::complex<float>>(
    std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
template void impl_direct_product<std::complex<double>, std::complex<double>, std::complex<double>>(
    std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);

template void impl_direct_division<float, float, float>(float alpha, einsums::detail::TensorImpl<float> const &A,
                                                        einsums::detail::TensorImpl<float> const &B, float beta,
                                                        einsums::detail::TensorImpl<float> *C);
template void impl_direct_division<double, double, double>(double alpha, einsums::detail::TensorImpl<double> const &A,
                                                           einsums::detail::TensorImpl<double> const &B, double beta,
                                                           einsums::detail::TensorImpl<double> *C);
template void impl_direct_division<std::complex<float>, std::complex<float>, std::complex<float>>(
    std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
template void impl_direct_division<std::complex<double>, std::complex<double>, std::complex<double>>(
    std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);

template float  impl_dot<float, float>(einsums::detail::TensorImpl<float> const &a, einsums::detail::TensorImpl<float> const &b);
template double impl_dot<double, double>(einsums::detail::TensorImpl<double> const &a, einsums::detail::TensorImpl<double> const &b);
template std::complex<float> impl_dot<std::complex<float>, std::complex<float>>(einsums::detail::TensorImpl<std::complex<float>> const &a,
                                                                                einsums::detail::TensorImpl<std::complex<float>> const &b);
template std::complex<double>
impl_dot<std::complex<double>, std::complex<double>>(einsums::detail::TensorImpl<std::complex<double>> const &a,
                                                     einsums::detail::TensorImpl<std::complex<double>> const &b);

template float  impl_true_dot<float, float>(einsums::detail::TensorImpl<float> const &a, einsums::detail::TensorImpl<float> const &b);
template double impl_true_dot<double, double>(einsums::detail::TensorImpl<double> const &a, einsums::detail::TensorImpl<double> const &b);
template std::complex<float>
impl_true_dot<std::complex<float>, std::complex<float>>(einsums::detail::TensorImpl<std::complex<float>> const &a,
                                                        einsums::detail::TensorImpl<std::complex<float>> const &b);
template std::complex<double>
              impl_true_dot<std::complex<double>, std::complex<double>>(einsums::detail::TensorImpl<std::complex<double>> const &a,
                                                          einsums::detail::TensorImpl<std::complex<double>> const &b);
template void impl_gemm<float, float, float, float, float>(char transA, char transB, float alpha,
                                                           einsums::detail::TensorImpl<float> const &A,
                                                           einsums::detail::TensorImpl<float> const &B, float beta,
                                                           einsums::detail::TensorImpl<float> *C);
template void impl_gemm<double, double, double, double, double>(char transA, char transB, double alpha,
                                                                einsums::detail::TensorImpl<double> const &A,
                                                                einsums::detail::TensorImpl<double> const &B, double beta,
                                                                einsums::detail::TensorImpl<double> *C);
template void impl_gemm<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>>(
    char transA, char transB, std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
template void impl_gemm<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>>(
    char transA, char transB, std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);
} // namespace einsums::linear_algebra::detail