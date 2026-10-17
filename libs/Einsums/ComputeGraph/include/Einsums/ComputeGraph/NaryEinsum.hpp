//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file NaryEinsum.hpp
/// @brief An einsum over three or more operands, contracted two at a time in the cheapest order.

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief One contraction of an @ref plan_nary_einsum plan.
 *
 * Operands are numbered as the spec names them, from zero, and the result of step @c k is operand
 * ``n + k`` for a spec over @c n operands. The last step writes the einsum's output.
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") NaryEinsumStep {
    /// The left operand's number.
    APIARY_EXPOSE APIARY_READONLY std::size_t left{0};
    /// The right operand's number.
    APIARY_EXPOSE APIARY_READONLY std::size_t right{0};
    /// The two-operand spec this step runs, ``conj(...)`` and permutation operators included.
    APIARY_EXPOSE APIARY_READONLY std::string spec;
    /// The extents of the partial product it forms; the output's own for the last step.
    APIARY_EXPOSE APIARY_READONLY std::vector<std::size_t> dims;
};

/**
 * @brief The cheapest order to contract an einsum over several operands two at a time.
 *
 * The order is the binary tree whose arithmetic, priced at these extents, is least: the same
 * search and the same ranking the graph's ``MultiTermFactorization`` uses on a captured product.
 * Each partial product keeps only the letters a later step or the output still needs, so an index
 * shared by the first two operands alone is summed when they are contracted.
 *
 * @param[in] spec The spec, three or more operands; two are planned too, as one step.
 * @param[in] dims Each operand's extents, in the spec's order.
 * @param[in] max_intermediate_elements The most elements a partial product may hold, or zero for no
 *            limit. When no order keeps every partial product within it, the limit is dropped and
 *            the cheapest order is returned.
 * @return The steps, in the order to run them.
 * @throws std::invalid_argument when the spec is malformed, names a different number of operands
 *         than @p dims holds, gives an operand the wrong rank, or a letter two different extents.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("graph") [[nodiscard]] EINSUMS_EXPORT std::vector<NaryEinsumStep>
plan_nary_einsum(std::string spec, std::vector<std::vector<std::size_t>> dims, std::size_t max_intermediate_elements = 0);

/**
 * @brief The most elements an eager einsum over several operands lets a partial product hold.
 *
 * A captured einsum is not limited: its partial products are graph intermediates, and the graph's
 * ``AxisTiling`` pass builds one a block at a time when it is too large to hold. An eager einsum
 * has no such pass, so its partial products are held to what @c --einsums:max-memory allows.
 *
 * @param[in] element_size The bytes of one element.
 * @return The limit, or zero for none.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("graph") [[nodiscard]] EINSUMS_EXPORT std::size_t eager_einsum_intermediate_limit(std::size_t element_size);

/**
 * @brief An einsum spec over three or more operands, its structure checked at compile time.
 *
 * The same check @ref EinsumFormatString makes, but for two or more ``;`` separators. A runtime
 * string converts unchecked, and is parsed when the einsum runs.
 * @versionadded{2.0.0}
 */
struct NaryEinsumFormatString {
    std::string_view str;

    template <size_t N>
    consteval NaryEinsumFormatString(char const (&s)[N]) : str(s, N - 1) { // NOLINT
        std::size_t semis = 0;
        for (char const ch : str) {
            semis += ch == ';' ? 1 : 0;
        }
        if (semis < 2 || !detail::validate_spec_structure(str, semis)) {
            throw "Invalid einsum format string: must contain exactly one '<-' or '->' and two or more ';'"; // NOLINT
        }
    }

    NaryEinsumFormatString(std::string_view s) : str(s) {} // NOLINT(google-explicit-constructor)

    constexpr operator std::string_view() const { return str; } // NOLINT(google-explicit-constructor)
};

namespace detail {

/// The extents of @p tensor, as a plan takes them.
template <typename TensorType>
std::vector<std::size_t> nary_dims_of(TensorType const &tensor) {
    std::vector<std::size_t> out;
    out.reserve(static_cast<std::size_t>(tensor.rank()));
    for (std::size_t axis = 0; axis < static_cast<std::size_t>(tensor.rank()); ++axis) {
        out.push_back(static_cast<std::size_t>(tensor.dim(axis)));
    }
    return out;
}

/// The name a partial product is spelled with: the tensor's own, or "view" for one that has none.
template <typename TensorType>
std::string nary_name_of(TensorType const &tensor) {
    if constexpr (requires { tensor.name(); }) {
        std::string name{tensor.name()};
        return name.empty() ? std::string{"view"} : name;
    } else {
        return "view";
    }
}

/// Call @p fn with operand @p index: one of @p operands, or a partial product an earlier step formed.
template <typename T, typename Fn, typename Tuple, std::size_t... I>
void with_nary_operand(std::size_t index, Tuple const &operands, std::vector<RuntimeTensor<T> *> const &partials, Fn &&fn,
                       std::index_sequence<I...> /*unused*/) {
    if (index >= sizeof...(I)) {
        fn(*partials[index - sizeof...(I)]);
        return;
    }
    (void)((index == I ? (fn(std::get<I>(operands)), true) : false) || ...);
}

} // namespace detail

/**
 * @brief Graph-aware einsum over three or more operands: ``C = c_pf * C + ab_pf * (A B D ...)``.
 *
 * Contracted two at a time, in the order @ref plan_nary_einsum finds cheapest, through the
 * two-operand einsum. Captured, each partial product is a graph intermediate that later passes
 * may slice or keep whole; eager, each is a temporary held to @ref eager_einsum_intermediate_limit.
 * The prefactors apply to the last contraction alone, so @p c_pf scales C once.
 *
 * @code
 * // The DF particle ladder, written as the equation reads
 * cg::einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", &r2, Bvv, Bvv, tau);
 * @endcode
 *
 * Every operand and C must share one element type, and none may be tiled.
 *
 * @param[in] spec The spec, its operands in the order @p operands are given.
 * @param[in] c_pf The prefactor on C.
 * @param[in,out] C The output.
 * @param[in] ab_pf The prefactor on the product.
 * @param[in] operands Three or more tensors.
 * @versionadded{2.0.0}
 */
template <BasicTensorConcept CType, BasicTensorConcept... Operands>
    requires(sizeof...(Operands) >= 3) && (std::is_same_v<typename CType::ValueType, typename Operands::ValueType> && ...) &&
            (!(TiledTensorConcept<CType> || (TiledTensorConcept<Operands> || ...)))
void einsum(NaryEinsumFormatString spec, typename CType::ValueType c_pf, CType *C, typename CType::ValueType ab_pf,
            Operands const &...operands) {
    using T = typename CType::ValueType;

    auto const &ctx       = CaptureContext::current();
    bool const  capturing = ctx.is_capturing();
    auto const  steps     = plan_nary_einsum(std::string(spec.str), {detail::nary_dims_of(operands)...},
                                             capturing ? 0 : eager_einsum_intermediate_limit(sizeof(T)));
    auto const  tuple     = std::forward_as_tuple(operands...);

    // Each partial product is named for what it multiplies, as "Bvv*tau", so a report that slices
    // one says which.
    std::vector<std::string> names{detail::nary_name_of(operands)...};
    for (std::size_t k = 0; k + 1 < steps.size(); ++k) {
        auto const spell = [&](std::size_t number) { return number < sizeof...(Operands) ? names[number] : "(" + names[number] + ")"; };
        names.push_back(spell(steps[k].left) + "*" + spell(steps[k].right));
    }

    // The partial products: internal graph intermediates when captured, which the passes may build a
    // block at a time and which a graph run without them allocates at its first execute;
    // temporaries when eager.
    std::vector<std::unique_ptr<RuntimeTensor<T>>> owned;
    std::vector<RuntimeTensor<T> *>                partials;
    partials.reserve(steps.size());
    for (std::size_t k = 0; k + 1 < steps.size(); ++k) {
        std::string name = names[sizeof...(Operands) + k];
        if (capturing) {
            partials.push_back(&ctx.graph()->template declare_internal_tensor<T>(std::move(name), steps[k].dims));
        } else {
            owned.push_back(std::make_unique<RuntimeTensor<T>>(std::move(name), steps[k].dims));
            partials.push_back(owned.back().get());
        }
    }

    for (std::size_t k = 0; k < steps.size(); ++k) {
        auto const &step = steps[k];
        bool const  last = k + 1 == steps.size();
        detail::with_nary_operand<T>(
            step.left, tuple, partials,
            [&](auto const &A) {
                detail::with_nary_operand<T>(
                    step.right, tuple, partials,
                    [&](auto const &B) {
                        if (last) {
                            einsum(EinsumFormatString(std::string_view(step.spec)), c_pf, C, ab_pf, A, B);
                        } else {
                            einsum(EinsumFormatString(std::string_view(step.spec)), T{0}, partials[k], T{1}, A, B);
                        }
                    },
                    std::index_sequence_for<Operands...>{});
            },
            std::index_sequence_for<Operands...>{});
    }
}

/// Graph-aware einsum over three or more operands with the default prefactors: C is overwritten.
/// @versionadded{2.0.0}
template <BasicTensorConcept CType, BasicTensorConcept... Operands>
    requires(sizeof...(Operands) >= 3) && (std::is_same_v<typename CType::ValueType, typename Operands::ValueType> && ...) &&
            (!(TiledTensorConcept<CType> || (TiledTensorConcept<Operands> || ...)))
void einsum(NaryEinsumFormatString spec, CType *C, Operands const &...operands) {
    using T = typename CType::ValueType;
    einsum(spec, T{0}, C, T{1}, operands...);
}

EINSUMS_NAMESPACE_END(compute_graph)
