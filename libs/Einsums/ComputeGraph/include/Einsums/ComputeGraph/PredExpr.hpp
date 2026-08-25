//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file PredExpr.hpp
 * @brief A boolean condition a node can hold as DATA rather than as a closure.
 *
 * @par Why this exists
 * @ref BoundExpr answered "what is this runtime integer" for view bounds:
 * a literal, a named @ref ParamTable entry, or a callback, with only the last
 * unserializable. Three descriptors held the boolean analogue as a bare
 * ``std::function`` instead -- ``ConditionalDescriptor::predicate``,
 * ``LoopDescriptor::condition`` and, through @ref BoundExpr, the source of a
 * ``write_param``. A closure cannot be written to a file, so those three were
 * the remaining reason a control-flow node could not be saved.
 *
 * ``PredExpr`` is the same variant discipline one level up. A convergence test
 * is a comparison against pipeline parameters and a gate is a load from a flag
 * array; both land in serializable arms without contortion, and the callback
 * arm stays fully legal for everything else. Only that arm blocks a save, and
 * @ref reconstruction_blocker names it per node.
 *
 * @par The iteration counter
 * A loop condition needs to see which iteration just finished. The counter
 * lives in its own arm (@ref PredExpr::Iteration) rather than as a reserved
 * @ref ParamTable name such as ``"loop:iteration"``, and the reason is that a
 * ``ParamTable`` is a flat, graph-wide, user-visible namespace whose entries
 * are ordered against their consumers by @ref param_writes / @ref param_reads.
 * Injecting a loop-private counter into it would collide between nested loops,
 * would let any view bound read a value no node writes as dataflow, and would
 * need a synthetic writer node before the scheduler could order anything
 * against it. The counter is loop-private state, so it is spelled as loop-private
 * state.
 *
 * @see DESIGN-algebraic-optimizer.md, Part 3.3 and Part 9 item 2
 */

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/GateFlags.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/// @brief The comparison a @ref PredExpr::Compare or @ref PredExpr::Iteration arm performs.
enum class CmpOp : std::uint8_t {
    Eq, ///< ``lhs == rhs``
    Ne, ///< ``lhs != rhs``
    Lt, ///< ``lhs < rhs``
    Le, ///< ``lhs <= rhs``
    Gt, ///< ``lhs > rhs``
    Ge, ///< ``lhs >= rhs``
};

/**
 * @brief Name of a @ref CmpOp, for diagnostics and for the IR's by-name rule.
 * @param[in] op The operator to name.
 * @return One of "eq", "ne", "lt", "le", "gt", "ge".
 */
[[nodiscard]] inline std::string_view cmp_op_name(CmpOp op) noexcept {
    switch (op) {
    case CmpOp::Eq:
        return "eq";
    case CmpOp::Ne:
        return "ne";
    case CmpOp::Lt:
        return "lt";
    case CmpOp::Le:
        return "le";
    case CmpOp::Gt:
        return "gt";
    case CmpOp::Ge:
        return "ge";
    }
    return "eq";
}

/**
 * @brief The @ref CmpOp spelled @p name, if there is one.
 * @param[in] name A spelling @ref cmp_op_name produces.
 * @return The operator, or an empty optional when nothing is spelled that way.
 *
 * The reverse of @ref cmp_op_name, for the IR's by-name rule: a saved predicate
 * writes "lt", never the enumerator's numeric value.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::optional<CmpOp> cmp_op_from_name(std::string_view name) noexcept {
    for (auto const op : {CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge}) {
        if (cmp_op_name(op) == name) {
            return op;
        }
    }
    return std::nullopt;
}

/**
 * @brief Apply @p op to two integers.
 * @param[in] lhs Left operand.
 * @param[in] op  The comparison.
 * @param[in] rhs Right operand.
 * @return The result of the comparison.
 */
[[nodiscard]] inline bool apply_cmp(std::int64_t lhs, CmpOp op, std::int64_t rhs) noexcept {
    switch (op) {
    case CmpOp::Eq:
        return lhs == rhs;
    case CmpOp::Ne:
        return lhs != rhs;
    case CmpOp::Lt:
        return lhs < rhs;
    case CmpOp::Le:
        return lhs <= rhs;
    case CmpOp::Gt:
        return lhs > rhs;
    case CmpOp::Ge:
        return lhs >= rhs;
    }
    return false;
}

/**
 * @brief A boolean expression a conditional or a loop evaluates at execute time.
 *
 * One of five arms, four of which are pure data:
 *
 *   - **Const**:     a literal ``true`` or ``false``.
 *   - **Compare**:   ``lhs <op> rhs`` over two @ref BoundExpr values, so either
 *                    side may itself be a literal, a named @ref ParamTable
 *                    entry, or (unserializably) a callback.
 *   - **Iteration**: ``iteration <op> rhs``, where ``iteration`` is the 0-based
 *                    index of the loop iteration that just finished. Meaningful
 *                    only inside a loop; @ref resolve throws elsewhere.
 *   - **FlagTest**:  one slot of a @ref GateFlags array, read through the same
 *                    shared buffer @ref Graph::add_conditional_flag has always
 *                    used. An index past the end reads false, which is the
 *                    conservative answer a replay can give without erroring.
 *   - **Callback**:  a ``std::function``, maximally flexible and the ONLY arm a
 *                    saved graph cannot hold.
 *
 * A default-constructed ``PredExpr`` is ``Const{true}``: for a loop that is
 * "run to ``max_iterations``", which is exactly what an absent
 * ``std::function`` condition has always meant.
 *
 * @par Construction
 * ``bool`` converts implicitly. Callables do NOT: the conversion is
 * ``explicit`` so that an overload set offering both ``std::function`` and
 * ``PredExpr`` spellings stays unambiguous for a caller passing a lambda. Use
 * @ref callback, @ref compare, @ref iteration or @ref flag to say which arm is
 * meant.
 *
 * @code
 * // "keep iterating while the pipeline still says we have work"
 * auto pred = cg::PredExpr::compare(cg::BoundExpr{"remaining"}, cg::CmpOp::Gt, cg::BoundExpr{0});
 * auto &body = graph.add_loop("iter", 100, pred);
 * @endcode
 */
class PredExpr {
  public:
    /// A literal truth value.
    struct Const {
        bool value{true};
    };

    /// ``lhs <op> rhs`` over two runtime integers.
    struct Compare {
        BoundExpr lhs;           ///< Left operand.
        CmpOp     op{CmpOp::Lt}; ///< The comparison.
        BoundExpr rhs;           ///< Right operand.
    };

    /// ``iteration <op> rhs``, against the enclosing loop's iteration index.
    struct Iteration {
        CmpOp     op{CmpOp::Lt}; ///< The comparison.
        BoundExpr rhs;           ///< Right operand.
    };

    /// One slot of a @ref GateFlags array, held as the shared buffer itself.
    struct FlagTest {
        std::shared_ptr<std::vector<std::uint8_t>> flags;    ///< The shared array. Null reads false.
        std::size_t                                index{0}; ///< Which slot selects this predicate.
    };

    /// An opaque predicate. The iteration index is passed through for a loop.
    struct Callback {
        std::function<bool(std::size_t)> fn;
    };

    using Storage = std::variant<Const, Compare, Iteration, FlagTest, Callback>;

    PredExpr() = default;

    /// A literal predicate.
    PredExpr(bool value) : _storage(Const{value}) {} // NOLINT(google-explicit-constructor)

    /// Wrap a nullary predicate. Explicit; see the construction note on the class.
    template <typename F>
        requires(!std::same_as<std::remove_cvref_t<F>, PredExpr> && !std::same_as<std::remove_cvref_t<F>, bool> && std::invocable<F &> &&
                 std::convertible_to<std::invoke_result_t<F &>, bool>)
    explicit PredExpr(F fn) : _storage(Callback{[f = std::move(fn)](std::size_t) mutable { return static_cast<bool>(f()); }}) {}

    /// Wrap a predicate that reads the iteration index. Explicit; see the class note.
    template <typename F>
        requires(!std::same_as<std::remove_cvref_t<F>, PredExpr> && !std::invocable<F &> && std::invocable<F &, std::size_t> &&
                 std::convertible_to<std::invoke_result_t<F &, std::size_t>, bool>)
    explicit PredExpr(F fn) : _storage(Callback{[f = std::move(fn)](std::size_t it) mutable { return static_cast<bool>(f(it)); }}) {}

    /// @brief A literal predicate.
    /// @param[in] value The value to return.
    /// @return The Const arm.
    [[nodiscard]] static PredExpr always(bool value) { return PredExpr{value}; }

    /// @brief ``lhs <op> rhs``.
    /// @param[in] lhs Left operand.
    /// @param[in] op The comparison.
    /// @param[in] rhs Right operand.
    /// @return The Compare arm.
    [[nodiscard]] static PredExpr compare(BoundExpr lhs, CmpOp op, BoundExpr rhs) {
        PredExpr out;
        out._storage = Compare{.lhs = std::move(lhs), .op = op, .rhs = std::move(rhs)};
        return out;
    }

    /// @brief ``iteration <op> rhs``, against the enclosing loop's iteration index.
    /// @param[in] op The comparison.
    /// @param[in] rhs Right operand.
    /// @return The Iteration arm.
    [[nodiscard]] static PredExpr iteration(CmpOp op, BoundExpr rhs) {
        PredExpr out;
        out._storage = Iteration{.op = op, .rhs = std::move(rhs)};
        return out;
    }

    /// @brief One slot of a gate-flag array.
    /// @param[in] flags The array to read. Shared, not copied.
    /// @param[in] index Which slot selects this predicate.
    /// @return The FlagTest arm.
    [[nodiscard]] static PredExpr flag(GateFlags const &flags, std::size_t index) { return flag(flags.buffer(), index); }

    /// @brief One slot of a gate-flag array, from the shared buffer directly.
    /// @param[in] buffer The array to read.
    /// @param[in] index Which slot selects this predicate.
    /// @return The FlagTest arm.
    [[nodiscard]] static PredExpr flag(std::shared_ptr<std::vector<std::uint8_t>> buffer, std::size_t index) {
        PredExpr out;
        out._storage = FlagTest{.flags = std::move(buffer), .index = index};
        return out;
    }

    /// @brief An opaque nullary predicate.
    /// @param[in] fn The predicate. An EMPTY function is legal and throws when evaluated,
    ///            which is what an empty ``std::function`` predicate has always done.
    /// @return The Callback arm.
    [[nodiscard]] static PredExpr callback(std::function<bool()> fn) {
        PredExpr out;
        out._storage = Callback{[f = std::move(fn)](std::size_t) { return f(); }};
        return out;
    }

    /// @brief An opaque predicate that reads the iteration index.
    /// @param[in] fn The predicate.
    /// @return The Callback arm.
    [[nodiscard]] static PredExpr callback(std::function<bool(std::size_t)> fn) {
        PredExpr out;
        out._storage = Callback{std::move(fn)};
        return out;
    }

    /// The arm this predicate holds.
    [[nodiscard]] Storage const &storage() const noexcept { return _storage; }

    [[nodiscard]] bool is_const() const noexcept { return std::holds_alternative<Const>(_storage); }
    [[nodiscard]] bool is_compare() const noexcept { return std::holds_alternative<Compare>(_storage); }
    [[nodiscard]] bool is_iteration() const noexcept { return std::holds_alternative<Iteration>(_storage); }
    [[nodiscard]] bool is_flag_test() const noexcept { return std::holds_alternative<FlagTest>(_storage); }

    /// True for the one arm a saved graph cannot hold. Note that a Compare arm
    /// whose OPERAND is a @ref BoundExpr callback is not this: ask
    /// @ref names_a_closure for the question a save actually needs answered.
    [[nodiscard]] bool is_callback() const noexcept { return std::holds_alternative<Callback>(_storage); }

    /// @brief Whether ANY part of this predicate is a closure.
    /// @return True when the predicate itself is a callback, or when a comparison
    ///         operand is a @ref BoundExpr callback.
    [[nodiscard]] bool names_a_closure() const noexcept {
        if (auto const *cmp = std::get_if<Compare>(&_storage)) {
            return cmp->lhs.is_callback() || cmp->rhs.is_callback();
        }
        if (auto const *it = std::get_if<Iteration>(&_storage)) {
            return it->rhs.is_callback();
        }
        return std::holds_alternative<Callback>(_storage);
    }

    /// @brief Append the @ref ParamTable names this predicate reads to @p names.
    /// @param[in,out] names Destination.
    ///
    /// Callback arms name nothing and are deliberately not reported, which is
    /// the same rule @ref param_reads applies to a callback-valued view bound:
    /// no name means no derivable edge.
    void collect_param_names(std::vector<std::string> &names) const {
        auto const add = [&names](BoundExpr const &bound) {
            if (bound.is_param()) {
                names.push_back(bound.param_name());
            }
        };
        if (auto const *cmp = std::get_if<Compare>(&_storage)) {
            add(cmp->lhs);
            add(cmp->rhs);
        } else if (auto const *it = std::get_if<Iteration>(&_storage)) {
            add(it->rhs);
        }
    }

    /**
     * @brief Evaluate the predicate.
     *
     * @param[in] params    The active parameter table. May be null when no arm
     *                      needs it; a @ref BoundExpr::Param operand without one
     *                      is an error rather than a silent default.
     * @param[in] iteration The 0-based index of the loop iteration that just
     *                      finished. Ignored by every arm but @ref Iteration and
     *                      @ref Callback.
     * @return The predicate's value.
     * @throws std::runtime_error When a named parameter is unset, or when a
     *         parameter is named with no table bound.
     */
    [[nodiscard]] bool resolve(ParamTable const *params, std::size_t iteration = 0) const {
        // A table to resolve Const bounds against when the caller has none.
        // Const-qualified and function-local, so it is initialized once and
        // never written.
        static ParamTable const empty;

        auto const value = [&](BoundExpr const &bound) -> std::int64_t {
            if (bound.is_param() && params == nullptr) {
                throw std::runtime_error("PredExpr: parameter '" + bound.param_name() + "' is named with no ParamTable bound");
            }
            return bound.resolve(params != nullptr ? *params : empty);
        };

        return std::visit(
            [&](auto const &arm) -> bool {
                using T = std::decay_t<decltype(arm)>;
                if constexpr (std::is_same_v<T, Const>) {
                    return arm.value;
                } else if constexpr (std::is_same_v<T, Compare>) {
                    return apply_cmp(value(arm.lhs), arm.op, value(arm.rhs));
                } else if constexpr (std::is_same_v<T, Iteration>) {
                    return apply_cmp(static_cast<std::int64_t>(iteration), arm.op, value(arm.rhs));
                } else if constexpr (std::is_same_v<T, FlagTest>) {
                    // Past the end, or an array that was never bound, reads
                    // false. A conditional cannot report an error usefully from
                    // inside a replay, and skipping a branch is conservative.
                    return arm.flags != nullptr && arm.index < arm.flags->size() && (*arm.flags)[arm.index] != 0;
                } else {
                    return arm.fn(iteration);
                }
            },
            _storage);
    }

  private:
    Storage _storage{Const{true}};
};

EINSUMS_NAMESPACE_END(compute_graph)
