//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file EscapeAnalysis.hpp
 * @brief Who writes a tensor, who reads it, and whether anything outside a set of nodes can see it.
 *
 * @par Why this is one component and not three
 * Two passes had already built most of this, each for its own question and each
 * in its own translation unit. `SymmetryPropagation` counts value-writers per
 * tensor and collects the pointers a descendant sub-graph references, because a
 * symmetry tag is only sound on a tensor nothing can overwrite behind its back.
 * `LoopInvariantHoisting` counts value-writers by POINTER across a loop's whole
 * subtree, because a producer may only leave the loop when its outputs have
 * exactly one writer in there. The region rewrite framework needs a third
 * phrasing of the same facts, and writing a third copy is how a
 * module ends up with three derivations of one relation that disagree in the
 * corner nobody tested. That is not a hypothetical here: the full-cover alias
 * bug and the 32-hop `resolve_alias` cap were both an incomplete alias relation,
 * and both surfaced as a race or a wrong number rather than as an error.
 *
 * So the facts are computed once, in one place, and the three questions are three
 * accessors over them.
 *
 * @par What "escape" means
 * A tensor escapes a set of nodes when anything outside that set can observe or
 * change its contents. A region rewrite may dissolve an intermediate only if it
 * does NOT escape: nothing outside reads it, nothing outside writes it, no view
 * of the same buffer is used outside, no descendant sub-graph touches it, and it
 * is not a tensor the user owns. Every one of those is a way for a rewrite that
 * looks locally correct to change a number somewhere else.
 *
 * @par Aliasing is resolved, never assumed
 * Every buffer question here goes through @ref Graph::resolve_alias first. A
 * write through a view of @c T is a write to @c T, and a component that counted
 * the view object instead would report a tensor as single-writer while a
 * different node overwrote it every iteration.
 *
 * @par Lifetime
 * An instance holds a non-owning reference to the graph it was built over and a
 * snapshot of that graph's node set. It is valid only until the graph's
 * structure changes, which for a pass means "for the duration of one `run()`
 * before it starts rewriting". Rebuild after a rewrite rather than updating in
 * place; the analysis is a linear walk and the arithmetic is not worth the class
 * of bug an incrementally maintained one invites.
 *
 * @see TensorExpr.hpp for the region rewriting this was factored out to serve
 */

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/**
 * @brief Why a tensor may or may not be dissolved into a region rewrite.
 *
 * Exactly one reason is reported, and the order the checks run in is fixed
 * (see @ref EscapeAnalysis::classify) so a tensor with two problems always
 * reports the same one. A varying reason would make a skip tally vary between
 * runs, and the tally is what a declining pass is read through.
 */
enum class Escape : std::uint8_t {
    /// Nothing outside the node set can observe this tensor. Safe to dissolve.
    Dissolvable,
    /// The tensor is not an intermediate: the user holds it and expects a value in it.
    UserOwned,
    /// A node outside the set reads it, so its value is observable.
    ReadOutside,
    /// A node outside the set writes it, so its value is not this region's to decide.
    WrittenOutside,
    /// Another tensor over the same buffer is used outside, or is itself user-owned.
    AliasedFromOutside,
    /// A descendant loop body or conditional branch references the buffer, and a
    /// sub-graph's writes do not appear in this graph's node list.
    TouchedBySubgraph,
    /// The graph does not know this tensor. Declines rather than guesses.
    Unknown,
};

/**
 * @brief A short, shape-independent phrase for @p reason, suitable for @c note_skip.
 *
 * Shape-independent on purpose: `OptimizerPass::note_skip` aggregates by reason
 * string, so a phrase carrying a tensor name would produce one line per
 * candidate instead of one line with a count.
 *
 * @param[in] reason The verdict to name.
 * @return The phrase, e.g. ``"a node outside the region reads it"``.
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view escape_reason(Escape reason);

/**
 * @brief Writers, readers and sub-graph reach for every tensor of one graph.
 *
 * Built once per graph with @ref EscapeAnalysis::over, then queried. See the
 * file comment for what it is for and how long it stays valid.
 */
class EINSUMS_EXPORT EscapeAnalysis {
  public:
    /**
     * @brief Analyse @p graph.
     * @param[in] graph The graph to walk. Must outlive the returned analysis, and
     *                  must not have its node set changed while it is held.
     * @return The analysis.
     */
    [[nodiscard]] static EscapeAnalysis over(Graph const &graph);

    /// @brief The graph this was built over.
    /// @return The graph.
    [[nodiscard]] Graph const &graph() const { return *_graph; }

    /**
     * @brief How many nodes of THIS graph write a value into @p id's buffer.
     *
     * Lifecycle nodes (Alloc, Free, Materialize, Initialize) do not count: they
     * list the tensor as an output without writing a value that could invalidate
     * anything inferred about it, and a freshly created or zeroed tensor is then
     * filled by exactly one real op. Descendant sub-graphs are not counted here;
     * see @ref subtree_writer_count.
     *
     * A @c View node DOES count, and that is a known conservatism rather than an
     * oversight. It lists its slice as an output, which resolves onto the
     * parent's buffer, so a tensor with two views of it reports three writers
     * where a person would say one. The effect is to decline: a symmetry tag is
     * not applied, a hoist is refused. Both passes this component was factored
     * out of already behaved this way, and the factoring's job was to keep their
     * behaviour rather than to improve it in the same step, so changing it is a
     * separate change with its own numeric tests.
     *
     * @param[in] id The tensor. Resolved through its alias chain first.
     * @return The count, zero for a tensor nothing writes.
     */
    [[nodiscard]] int writer_count(TensorId id) const;

    /**
     * @brief How many nodes write a value into @p id's buffer, in this graph and
     *        every descendant sub-graph.
     *
     * Keyed by the underlying tensor POINTER, which is what stays stable across
     * graphs: a loop body has its own tensor table and its own ids for the same
     * buffers. A tensor with no pointer (a deferred shell whose storage is not
     * attached yet) is not counted, which makes the answer conservative in the
     * direction that declines.
     *
     * @param[in] id The tensor. Resolved through its alias chain first.
     * @return The count.
     */
    [[nodiscard]] int subtree_writer_count(TensorId id) const;

    /**
     * @brief Does any descendant sub-graph reference @p id's buffer at all?
     *
     * A ``Loop`` node does not list its body's writes, so a tensor a body touches
     * looks untouched from the parent's node list. Anything that reasons from
     * that list has to ask this before concluding it knows every access.
     *
     * @param[in] id The tensor.
     * @return True when a descendant references the buffer, and true for a tensor
     *         with no pointer to compare (nothing can be proved about it).
     */
    [[nodiscard]] bool touched_by_subtree(TensorId id) const;

    /**
     * @brief Exactly one value-writer here, and no descendant touches it.
     *
     * The soundness guard `SymmetryPropagation` needs, in one call: an inferred
     * property of a tensor stays true only while nothing else can write it.
     *
     * @param[in] id The tensor.
     * @return True when the tensor's contents are settled by one node of this graph.
     */
    [[nodiscard]] bool stable(TensorId id) const;

    /**
     * @brief Can @p id be dissolved into a rewrite of @p region, and if not, why?
     *
     * Checks run in a fixed order so a tensor failing
     * two of them always reports the first:
     *
     *  1. Does the graph know it? Otherwise @ref Escape::Unknown.
     *  2. Is it an intermediate? Otherwise @ref Escape::UserOwned.
     *  3. Is any tensor over the same buffer user-owned? @ref Escape::AliasedFromOutside.
     *  4. Does a node outside the region write a VALUE into the buffer? @ref Escape::WrittenOutside.
     *  5. Does a node outside the region read its value? @ref Escape::ReadOutside.
     *  6. Does a descendant sub-graph touch it? @ref Escape::TouchedBySubgraph.
     *
     * Writes are checked before reads deliberately. Both decline, but the two
     * call for different responses from whoever reads the tally: an outside
     * writer means the region does not own the value at all, while an outside
     * reader often means the region simply needs to grow.
     *
     * Lifecycle mentions do NOT count on either side. `create_*` and `declare_*`
     * put an Alloc ahead of the first real write, Alloc is not raisable and so is
     * never inside a region, and counting its mention would make every
     * graph-owned intermediate undissolvable - which is every intermediate a
     * rewrite exists to dissolve. The caller carries the consequence: dissolving
     * an intermediate leaves its Alloc and Free naming a tensor nothing writes
     * any more, which is dead rather than wrong and is what
     * `DeadNodeElimination` is for.
     *
     * @param[in] id     The tensor to classify.
     * @param[in] region The node ids the rewrite covers.
     * @return The verdict.
     */
    [[nodiscard]] Escape classify(TensorId id, std::unordered_set<NodeId> const &region) const;

    /**
     * @brief Every tensor id of this graph that resolves to the same buffer as @p id.
     *
     * Includes @p id itself. This is the set a buffer question has to range over:
     * asking about one view and not its siblings is how a region convinces itself
     * a buffer is private while a sibling view of it is read three nodes later.
     *
     * @param[in] id The tensor.
     * @return The ids sharing its buffer, sorted so the result does not vary between runs.
     */
    [[nodiscard]] std::vector<TensorId> aliases_of(TensorId id) const;

  private:
    Graph const *_graph{nullptr};

    /// Value-writers and readers of this graph, keyed by RESOLVED tensor id.
    std::unordered_map<TensorId, std::vector<NodeId>> _value_writers;
    std::unordered_map<TensorId, std::vector<NodeId>> _any_writers;
    std::unordered_map<TensorId, std::vector<NodeId>> _value_readers;
    std::unordered_map<TensorId, std::vector<NodeId>> _any_readers;

    /// Resolved id -> every id of this graph that resolves to it.
    std::unordered_map<TensorId, std::vector<TensorId>> _by_root;

    /// Value-writer counts across the whole subtree, keyed by tensor pointer.
    std::unordered_map<void const *, int> _subtree_writers;

    /// Pointers any descendant sub-graph mentions.
    std::unordered_set<void const *> _subtree_ptrs;
};

/**
 * @brief The node of @p graph that updates @p tensor, when @p graph is a solver's loop body.
 *
 * A tagged tensor some node writes is normally refused outright by a factorization: its factors
 * would go stale whenever it changed, and a fitting runs once per bound problem. An AMPLITUDE is
 * written every iteration by design, and the way out is not to relax the refusal but to
 * recognize the one writer that makes a re-fit meaningful.
 *
 * What counts as the update statement, and nothing else does:
 *
 *  - a @c DirectDivision whose destination is @p tensor, which is a residual divided by an
 *    energy denominator into the amplitude, the plain update with the extrapolation left to the
 *    host predicate; or
 *  - an @c Axpby accumulating into @p tensor whose source is itself produced in this graph by a
 *    @c DirectDivision, which is the same update written as a step the host's DIIS can read.
 *
 * There must be exactly ONE value-writer of the buffer here and none in any descendant, because
 * two writers mean the value the fit is about is not settled by the statement being recognized.
 *
 * @param[in] graph The loop body.
 * @param[in] tensor The tagged tensor.
 * @return The updating node's id, or the reason this is not an update.
 */
[[nodiscard]] EINSUMS_EXPORT expected<NodeId, std::string> amplitude_update_writer(Graph const &graph, TensorId tensor);

EINSUMS_NAMESPACE_END(compute_graph)
