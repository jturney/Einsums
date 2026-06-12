//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Concepts/SmartPointer.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>

#if defined(EINSUMS_USE_CATCH2)
#    include <catch2/catch_all.hpp>
#endif

namespace einsums::tensor_algebra {
namespace detail {

// CType has typename to allow for interoperability with scalar types.
template <bool OnlyUseGenericAlgorithm, bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
AlgorithmChoice einsum(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                       BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                       std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B);
} // namespace detail

/*
 * Dispatchers for einsum.
 */
template <bool ConjA = false, bool ConjB = false, TensorConcept AType, TensorConcept BType, typename CType, typename U,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires InSamePlace<AType, CType> || !TensorConcept<CType>;
        requires !SmartPointer<CType>;
    }
void einsum(U const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C, U const UAB_prefactor,
            std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr);

/*
 * Batched einsums calls over collections of tensors.
 */
template <bool ConjA = false, bool ConjB = false, Container CType, Container AType, Container BType, typename CPrefactorType,
          typename ABPrefactorType, typename... AIndices, typename... BIndices, typename... CIndices>
void einsum(CPrefactorType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C_list, ABPrefactorType const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A_list, std::tuple<BIndices...> const &B_indices, BType const &B_list,
            detail::AlgorithmChoice *algorithm_choice = nullptr);

namespace detail {

// Type trait: for a SmartPointer, yields its element_type; otherwise yields T itself.
template <typename T, bool = SmartPointer<T>>
struct DerefTypeHelper {
    using type = T;
};

template <typename T>
struct DerefTypeHelper<T, true> {
    using type = typename T::element_type;
};

template <typename T>
using deref_type_t = typename DerefTypeHelper<T>::type;

// Dereference A or B: smart-pointer → *ptr; anything else → passthrough.
// Note: weak_ptr satisfies SmartPointer but has no operator*, so calling auto_deref
// on one fails to compile, matching the behaviour of the old explicit overloads.
template <typename T>
[[nodiscard]] decltype(auto) auto_deref(T &&t) {
    if constexpr (SmartPointer<std::remove_cvref_t<T>>)
        return (*t);
    else
        return std::forward<T>(t);
}

// Dereference C (passed as pointer-to-wrapper):
//   shared_ptr<Tensor>*  →  Tensor*   (via ->get())
//   Tensor*              →  Tensor*   (passthrough)
template <typename T>
[[nodiscard]] auto auto_deref_c(T *ptr) {
    if constexpr (SmartPointer<T>)
        return ptr->get();
    else
        return ptr;
}

} // namespace detail

// Smart-pointer dispatcher with prefactors.
// Fires when at least one of A, B, or C is a SmartPointer.
template <bool ConjA = false, bool ConjB = false, typename AType, typename BType, typename CType, typename U, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires((SmartPointer<AType> || SmartPointer<BType> || SmartPointer<CType>) && TensorConcept<detail::deref_type_t<AType>> &&
             TensorConcept<detail::deref_type_t<BType>> &&
             requires {
                 requires InSamePlace<detail::deref_type_t<AType>, detail::deref_type_t<BType>>;
                 requires InSamePlace<detail::deref_type_t<AType>, detail::deref_type_t<CType>> ||
                              !TensorConcept<detail::deref_type_t<CType>>;
             })
void einsum(U const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum<ConjA, ConjB>(C_prefactor, C_indices, detail::auto_deref_c(C), AB_prefactor, A_indices, detail::auto_deref(A), B_indices,
                         detail::auto_deref(B), algorithm_choice);
}

// Default-prefactor dispatcher: handles all pointer/non-pointer combinations,
// including containers (batched einsum). There is no TensorConcept constraint
// here; the forwarded with-prefactors call is responsible for type checking.
template <bool ConjA = false, bool ConjB = false, typename AType, typename BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
void einsum(std::tuple<CIndices...> const &C_indices, CType *C, std::tuple<AIndices...> const &A_indices, AType const &A,
            std::tuple<BIndices...> const &B_indices, BType const &B, detail::AlgorithmChoice *algorithm_choice = nullptr) {
    einsum<ConjA, ConjB>(0, C_indices, C, 1, A_indices, A, B_indices, B, algorithm_choice);
}

//
// Element Transform
//

template <SmartPointer SmartPtr, typename UnaryOperator>
void element_transform(SmartPtr *C, UnaryOperator unary_opt) {
    element_transform(C->get(), unary_opt);
}

template <unsigned int N, typename... List>
constexpr auto get_n(std::tuple<List...> const &);

/*
 * Returns the mode-`mode` unfolding of `tensor` with modes startng at `0`
 *
 * @returns unfolded_tensor of shape ``(tensor.dim(mode), -1)``
 */
// template <unsigned int mode, template <typename, size_t> typename CType, size_t CRank, typename T = double>
// Tensor<T, 2> unfold(CType<T, CRank> const &source)
// requires(std::is_same_v<Tensor<T, CRank>, CType<T, CRank>>);

/** Computes the Khatri-Rao product of tensors A and B.
 *
 * Example:
 *    Tensor<2> result = khatri_rao(Indices{I, r}, A, Indices{J, r}, B);
 *
 * Result is described as {(I,J), r}. If multiple common indices are provided they will be collapsed into a single index in the result.
 */
template <bool ConjA = false, bool ConjB = false, TensorConcept AType, TensorConcept BType, typename... AIndices, typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires AType::Rank == sizeof...(AIndices);
        requires BType::Rank == sizeof...(BIndices);
    }
BasicTensorLike<AType, typename AType::ValueType, 2> khatri_rao(std::tuple<AIndices...> const &, AType const &A,
                                                                std::tuple<BIndices...> const &, BType const &B);
} // namespace einsums::tensor_algebra

#include <Einsums/TensorAlgebra/Backends/Dispatch.hpp>
#include <Einsums/TensorAlgebra/Backends/ElementTransform.hpp>
#include <Einsums/TensorAlgebra/Backends/KhatriRao.hpp>
#include <Einsums/TensorAlgebra/Backends/Unfold.hpp>