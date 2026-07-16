//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/TensorAlgebra/Detail/HpttPlanCache.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>

// HPTT includes <complex> which defined I as a shorthand for complex values.
// This causes issues with einsums since we define I to be a useable index
// for the user. Undefine the one defined in <complex> here.
#if defined(I)
#    undef I
#endif

namespace einsums::tensor_algebra::detail {

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<float>> permute(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                float const beta, float *B, bool /*conjA*/, bool row_major) {
    auto plan = get_or_create_hptt_plan<float>(perm, dim, alpha, A, sizeA, beta, B, row_major);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<double>> permute(int const *perm, int const dim, double const alpha, double const *A, size_t const *sizeA,
                                                 double const beta, double *B, bool /*conjA*/, bool row_major) {
    auto plan = get_or_create_hptt_plan<double>(perm, dim, alpha, A, sizeA, beta, B, row_major);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<std::complex<float>>> permute(int const *perm, int const dim, std::complex<float> const alpha,
                                                              std::complex<float> const *A, size_t const *sizeA,
                                                              std::complex<float> const beta, std::complex<float> *B, bool conjA,
                                                              bool row_major) {
    auto plan = get_or_create_hptt_plan<std::complex<float>>(perm, dim, alpha, A, sizeA, beta, B, row_major);
    plan->set_conj_a(conjA);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<std::complex<double>>> permute(int const *perm, int const dim, std::complex<double> const alpha,
                                                               std::complex<double> const *A, size_t const *sizeA,
                                                               std::complex<double> const beta, std::complex<double> *B, bool conjA,
                                                               bool row_major) {
    auto plan = get_or_create_hptt_plan<std::complex<double>>(perm, dim, alpha, A, sizeA, beta, B, row_major);
    plan->set_conj_a(conjA);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<float>> permute(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                size_t const *offsetA, size_t const *outerSizeA, float const beta, float *B,
                                                size_t const *offsetB, size_t const *outerSizeB, bool /*conjA*/, bool row_major) {
    auto plan = get_or_create_hptt_plan<float>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, row_major);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<double>> permute(int const *perm, int const dim, double const alpha, double const *A, size_t const *sizeA,
                                                 size_t const *offsetA, size_t const *outerSizeA, double const beta, double *B,
                                                 size_t const *offsetB, size_t const *outerSizeB, bool /*conjA*/, bool row_major) {
    auto plan = get_or_create_hptt_plan<double>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB, row_major);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<std::complex<float>>> permute(int const *perm, int const dim, std::complex<float> const alpha,
                                                              std::complex<float> const *A, size_t const *sizeA, size_t const *offsetA,
                                                              size_t const *outerSizeA, std::complex<float> const beta,
                                                              std::complex<float> *B, size_t const *offsetB, size_t const *outerSizeB,
                                                              bool conjA, bool row_major) {
    auto plan = get_or_create_hptt_plan<std::complex<float>>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB,
                                                             row_major);
    plan->set_conj_a(conjA);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<std::complex<double>>> permute(int const *perm, int const dim, std::complex<double> const alpha,
                                                               std::complex<double> const *A, size_t const *sizeA, size_t const *offsetA,
                                                               size_t const *outerSizeA, std::complex<double> const beta,
                                                               std::complex<double> *B, size_t const *offsetB, size_t const *outerSizeB,
                                                               bool conjA, bool row_major) {
    auto plan = get_or_create_hptt_plan<std::complex<double>>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, beta, B, outerSizeB, offsetB,
                                                              row_major);
    plan->set_conj_a(conjA);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<float>> permute(int const *perm, int const dim, float const alpha, float const *A, size_t const *sizeA,
                                                size_t const *offsetA, size_t const *outerSizeA, size_t const innerStrideA,
                                                float const beta, float *B, size_t const *offsetB, size_t const *outerSizeB,
                                                size_t const innerStrideB, bool /*conjA*/, bool row_major) {
    auto plan = get_or_create_hptt_plan<float>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB,
                                               innerStrideB, row_major);
    plan->execute();
    return plan;
}

// NOLINTNEXTLINE
std::shared_ptr<hptt::Transpose<double>> permute(int const *perm, int const dim, double const alpha, double const *A, size_t const *sizeA,
                                                 size_t const *offsetA, size_t const *outerSizeA, size_t const innerStrideA,
                                                 double const beta, double *B, size_t const *offsetB, size_t const *outerSizeB,
                                                 size_t const innerStrideB, bool /*conjA*/, bool row_major) {
    auto plan = get_or_create_hptt_plan<double>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B, outerSizeB, offsetB,
                                                innerStrideB, row_major);
    plan->execute();
    return plan;
}

std::shared_ptr<hptt::Transpose<std::complex<float>>>
// NOLINTNEXTLINE
permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, size_t const *sizeA,
        size_t const *offsetA, size_t const *outerSizeA, size_t const innerStrideA, std::complex<float> const beta, std::complex<float> *B,
        size_t const *offsetB, size_t const *outerSizeB, size_t const innerStrideB, bool conjA, bool row_major) {
    auto plan = get_or_create_hptt_plan<std::complex<float>>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B,
                                                             outerSizeB, offsetB, innerStrideB, row_major);
    plan->set_conj_a(conjA);
    plan->execute();
    return plan;
}

std::shared_ptr<hptt::Transpose<std::complex<double>>>
// NOLINTNEXTLINE
permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, size_t const *sizeA,
        size_t const *offsetA, size_t const *outerSizeA, size_t const innerStrideA, std::complex<double> const beta,
        std::complex<double> *B, size_t const *offsetB, size_t const *outerSizeB, size_t const innerStrideB, bool conjA, bool row_major) {
    auto plan = get_or_create_hptt_plan<std::complex<double>>(perm, dim, alpha, A, sizeA, outerSizeA, offsetA, innerStrideA, beta, B,
                                                              outerSizeB, offsetB, innerStrideB, row_major);
    plan->set_conj_a(conjA);
    plan->execute();
    return plan;
}

} // namespace einsums::tensor_algebra::detail
