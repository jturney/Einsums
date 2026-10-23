//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/BLASVendor.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/StridedForEach.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

#include <fmt/format.h>

// Element-wise (Hadamard) division, the counterpart to direct_product:
//
//     c_i := beta * c_i + alpha * a_i / b_i
//
// No BLAS primitive, so plain loops with direct_product's layout dispatch. No zero guard: as in
// numpy, 1/0 is inf.

EINSUMS_NAMESPACE_BEGIN()
namespace linear_algebra {
namespace detail {

template <typename AType, typename BType, typename CType>
void impl_direct_division(CType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B, CType beta,
                          einsums::detail::TensorImpl<CType> *C) {
    WAGGLE_ZONE_FUNC();

    if (A.rank() != B.rank() || A.rank() != C->rank()) {
        EINSUMS_THROW_EXCEPTION(RankError, "Can not combine tensors of different ranks!");
    }

    if (A.dims() != B.dims() || A.dims() != C->dims()) {
        EINSUMS_THROW_EXCEPTION(DimensionError, "Can not combine tensors with different sizes!");
    }

    // A zero beta never reads C, as BLAS never reads an output scaled by zero.
    bool const                         discard = beta == CType{0};
    einsums::detail::ParallelThreshold par;
    using einsums::detail::RunOperand;
    einsums::detail::strided_for_each_run(
        C->dims(),
        [&](size_t n, RunOperand<CType> c, RunOperand<AType const> a, RunOperand<BType const> b) {
            einsums::detail::run_elements(
                n, par,
                [alpha, beta, discard](CType &c_v, AType const &a_v, BType const &b_v) {
                    CType const quotient = alpha * a_v / b_v;
                    c_v                  = discard ? quotient : beta * c_v + quotient;
                },
                c, a, b);
        },
        einsums::detail::strided(*C), einsums::detail::strided(A), einsums::detail::strided(B));
}

extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void
impl_direct_division<float, float, float>(float alpha, einsums::detail::TensorImpl<float> const &A,
                                          einsums::detail::TensorImpl<float> const &B, float beta, einsums::detail::TensorImpl<float> *C);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void
impl_direct_division<double, double, double>(double alpha, einsums::detail::TensorImpl<double> const &A,
                                             einsums::detail::TensorImpl<double> const &B, double beta,
                                             einsums::detail::TensorImpl<double> *C);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void impl_direct_division<std::complex<float>, std::complex<float>, std::complex<float>>(
    std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void impl_direct_division<std::complex<double>, std::complex<double>, std::complex<double>>(
    std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);

} // namespace detail
} // namespace linear_algebra
EINSUMS_NAMESPACE_END()
