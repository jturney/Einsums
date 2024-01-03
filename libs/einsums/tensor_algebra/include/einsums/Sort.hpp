//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/config/ExportDefinitions.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/OpenMP.hpp"
// #include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/concepts/SmartPointerTraits.hpp"
#include "einsums/tensor_algebra/Utilities.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

#if defined(EINSUMS_USE_HPTT)

namespace detail {

void EINSUMS_EXPORT sort(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, float const beta, float *B);
void EINSUMS_EXPORT sort(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, double const beta,
                         double *B);
void EINSUMS_EXPORT sort(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
                         std::complex<float> const beta, std::complex<float> *B);
void EINSUMS_EXPORT sort(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
                         std::complex<double> const beta, std::complex<double> *B);

} // namespace detail

#endif

//
// sort algorithm
//
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename CType, size_t CRank,
          typename... CIndices, typename... AIndices, typename U, typename T = double>
auto sort(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType<T, CRank> *C, U const UA_prefactor,
          std::tuple<AIndices...> const &A_indices, AType<T, ARank> const &A)
    -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, CRank>, CType<T, CRank>> &&
                        std::is_base_of_v<::einsums::detail::TensorBase<T, ARank>, AType<T, ARank>> &&
                        sizeof...(CIndices) == sizeof...(AIndices) && sizeof...(CIndices) == CRank && sizeof...(AIndices) == ARank &&
                        std::is_arithmetic_v<U>> {

    LabeledSection1(FP_ZERO != std::fpclassify(UC_prefactor)
                        ? fmt::format(R"(sort: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor,
                                      A.name(), print_tuple_no_type(A_indices), UC_prefactor, C->name(), print_tuple_no_type(C_indices))
                        : fmt::format(R"(sort: "{}"{} = {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor, A.name(),
                                      print_tuple_no_type(A_indices)));

    T const C_prefactor = UC_prefactor;
    T const A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a sort
    constexpr auto check = difference_t<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    auto target_dims = get_dim_ranges<CRank>(*C);
    auto a_dims      = detail::get_dim_ranges_for(A, target_position_in_A);

    // HPTT interface currently only works for full Tensors and not TensorViews
#if defined(EINSUMS_USE_HPTT)
    if constexpr (std::is_same_v<CType<T, CRank>, Tensor<T, CRank>> && std::is_same_v<AType<T, ARank>, Tensor<T, ARank>>) {
        std::array<int, ARank> perms{};
        std::array<int, ARank> size{};

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0] = get_from_tuple<unsigned long>(target_position_in_A, (2 * i0) + 1);
            size[i0]  = A.dim(i0);
        }

        detail::sort(perms.data(), ARank, A_prefactor, A.data(), size.data(), C_prefactor, C->data());
    } else
#endif
        if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)>) {
        linear_algebra::axpby(A_prefactor, A, C_prefactor, C);
    } else {
        auto view = std::apply(ranges::views::cartesian_product, target_dims);

        EINSUMS_OMP_PARALLEL_FOR
        for (auto it = view.begin(); it < view.end(); it++) {
            auto A_order = detail::construct_indices<AIndices...>(*it, target_position_in_A, *it, target_position_in_A);

            T &target_value = std::apply(*C, *it);
            T  A_value      = std::apply(A, A_order);

            target_value = C_prefactor * target_value + A_prefactor * A_value;
        }
    }
} // namespace einsums::TensorAlgebra

// Sort with default values, no smart pointers
template <NotASmartPointer ObjectA, NotASmartPointer ObjectC, typename... CIndices, typename... AIndices>
void sort(std::tuple<CIndices...> const &C_indices, ObjectC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    sort(0, C_indices, C, 1, A_indices, A);
}

// Sort with default values, two smart pointers
template <SmartPointer SmartPointerA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void sort(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    sort(0, C_indices, C->get(), 1, A_indices, *A);
}

// Sort with default values, one smart pointer (A)
template <SmartPointer SmartPointerA, NotASmartPointer PointerC, typename... CIndices, typename... AIndices>
void sort(std::tuple<CIndices...> const &C_indices, PointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    sort(0, C_indices, C, 1, A_indices, *A);
}

// Sort with default values, one smart pointer (C)
template <NotASmartPointer ObjectA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void sort(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    sort(0, C_indices, C->get(), 1, A_indices, A);
}

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)