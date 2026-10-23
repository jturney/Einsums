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

EINSUMS_NAMESPACE_BEGIN()
namespace linear_algebra {
namespace detail {

template <typename AType, typename BType, typename CType>
void impl_direct_product(CType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B, CType beta,
                         einsums::detail::TensorImpl<CType> *C) {
    WAGGLE_ZONE_FUNC();

    if (A.rank() != B.rank() || A.rank() != C->rank()) {
        EINSUMS_THROW_EXCEPTION(RankError, "Can not combine tensors of different ranks!");
    }

    if (A.dims() != B.dims() || A.dims() != C->dims()) {
        EINSUMS_THROW_EXCEPTION(DimensionError, "Can not combine tensors with different sizes!");
    }

    // An input that is C, element for element, is safe for the element loop, which reads each element
    // before writing it. Any other overlap is read through a copy.
    auto const needs_copy = [&](auto const &in) {
        return einsums::detail::impl_storage_overlaps(in, *C) && !einsums::detail::impl_same_operand(in, *C);
    };
    if (needs_copy(A)) {
        einsums::detail::impl_with_snapshot(A, [&](auto const &a) { impl_direct_product(alpha, a, B, beta, C); });
        return;
    }
    if (needs_copy(B)) {
        einsums::detail::impl_with_snapshot(B, [&](auto const &b) { impl_direct_product(alpha, A, b, beta, C); });
        return;
    }

    // dirprod requires C to overlap neither input. A zero beta discards C, in dirprod and in the
    // loop: what C held, NaN included, does not survive.
    bool const                         c_is_input = einsums::detail::impl_same_operand(A, *C) || einsums::detail::impl_same_operand(B, *C);
    bool const                         discard    = beta == CType{0};
    einsums::detail::ParallelThreshold par;
    size_t const                       total = C->size();
    using einsums::detail::RunOperand;
    einsums::detail::strided_for_each_run(
        C->dims(),
        [&](size_t n, RunOperand<CType> c, RunOperand<AType const> a, RunOperand<BType const> b) {
            if constexpr (std::is_same_v<AType, BType> && std::is_same_v<AType, CType> && blas::IsBlasableV<AType>) {
                if (!c_is_input && einsums::detail::use_vector_routine(n, total, c, a, b)) {
                    blas::dirprod(n, alpha, a.data, a.inc, b.data, b.inc, beta, c.data, c.inc);
                    return;
                }
            }
            einsums::detail::run_elements(
                n, par,
                [alpha, beta, discard](CType &c_v, AType const &a_v, BType const &b_v) {
                    CType const product = alpha * a_v * b_v;
                    c_v                 = discard ? product : beta * c_v + product;
                },
                c, a, b);
        },
        einsums::detail::strided(*C), einsums::detail::strided(A), einsums::detail::strided(B));
}

extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void
impl_direct_product<float, float, float>(float alpha, einsums::detail::TensorImpl<float> const &A,
                                         einsums::detail::TensorImpl<float> const &B, float beta, einsums::detail::TensorImpl<float> *C);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void
impl_direct_product<double, double, double>(double alpha, einsums::detail::TensorImpl<double> const &A,
                                            einsums::detail::TensorImpl<double> const &B, double beta,
                                            einsums::detail::TensorImpl<double> *C);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void impl_direct_product<std::complex<float>, std::complex<float>, std::complex<float>>(
    std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT void impl_direct_product<std::complex<double>, std::complex<double>, std::complex<double>>(
    std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);

} // namespace detail
} // namespace linear_algebra
EINSUMS_NAMESPACE_END()
