//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief The symbolic cost attributed to one graph node.
 *
 * A value type, so a caller may keep it after the pass and the graph are gone.
 */
struct NodeCost {
    std::string  graph_name; ///< Graph the node belongs to. Node ids are graph-local.
    NodeId       node_id{0}; ///< The node's id within that graph.
    std::string  label;      ///< The node's human-readable label.
    SymbolicCost cost;       ///< Its flops, traffic and resident polynomials.

    /// @brief Compare two node costs field by field.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when graph, id, label and all three polynomials match.
    [[nodiscard]] friend bool operator==(NodeCost const &lhs, NodeCost const &rhs) = default;
};

/**
 * @brief The symbolic element count of one graph-owned intermediate.
 */
struct IntermediateSize {
    std::string  graph_name;   ///< Graph the tensor belongs to. Tensor ids are graph-local.
    TensorId     tensor_id{0}; ///< The tensor's id within that graph.
    std::string  name;         ///< The tensor's name.
    SymbolicPoly size;         ///< Its element count as a polynomial in index-space scales.

    /// @brief Compare two intermediate sizes field by field.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when graph, id, name and polynomial match.
    [[nodiscard]] friend bool operator==(IntermediateSize const &lhs, IntermediateSize const &rhs) = default;
};

/**
 * @brief Report how a program scales: every contraction's cost polynomial, and what limits it.
 *
 * The user-facing face of the symbolic cost model. It walks the graph, asks
 * @ref symbolic_cost_for for each contraction's cost, adds up the flops, sizes the intermediates,
 * and names the node whose polynomial dominates. Nothing is rewritten and nothing is decided; the
 * point is to answer "what is the rate-limiting term" beside the question "how fast is this", and
 * to be the first consumer that proves a program's space annotations are the ones its author
 * meant.
 *
 * Three surfaces carry the same numbers:
 * - @ref explain, the two or three lines that join ``graph.explain()``,
 * - @ref print_report, the full per-node table,
 * - the accessors (@ref node_costs, @ref total_flops, @ref rate_limiting, ...) for tests and
 *   tooling.
 *
 * @par What is analysed
 * Contraction nodes only. Every other kind is counted in @ref skip_reasons and contributes
 * nothing: inventing a cost formula for an op whose cost model has not been written would put
 * numbers in the report that no pass agrees with.
 *
 * @par Where the letters come from
 * A node's cost is derived from the spaces its operands' handles carry AT THE TIME THE PASS RUNS,
 * not from the ``letter_spaces`` map frozen into the node at capture. That is what makes the
 * report reflect a declaration that arrived after capture, and everything ``SpacePropagation``
 * inferred immediately before this pass in the default pipeline. Letters that no annotated slot
 * reaches, and letters whose slots contradict each other, fall back to the node's captured map and
 * then to an anonymous per-letter variable, so an unannotated program still yields a complete
 * report, just one written in ``?i`` rather than in ``o``.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("scaling");
 * graph.set_space_registry(registry);
 * graph.annotate_spaces(T2, {occ, occ, virt, virt});
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ijab <- ijcd ; cdab", &R2, T2, Vvvvv);
 * }
 * auto [modified, scaling] = graph.apply<cg::passes::ScalingAnalysis>();
 * // scaling.total_flops().to_string(&registry) == "2*o^2*v^4"
 * scaling.print_report(std::cout);
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * scaling = cg.ScalingAnalysis()
 * pm = cg.PassManager()
 * pm.add(scaling)
 * pm.run(g)
 * assert scaling.total_flops_str() == "2*o^2*v^4"
 * print(scaling.report_string())
 * @endcode
 * Python sees the polynomials as their renderings rather than as @ref SymbolicPoly objects; see
 * the rendered surface on this class.
 *
 * @par Limitations
 * - Analysis only. @ref run always reports "not modified".
 * - @ref memory_bound is the SUM of the intermediate sizes, which is an upper bound on the
 *   high-water mark and not the high-water mark itself. A liveness-aware figure needs the interval
 *   analysis ``MemoryPlanning`` already does for bytes, applied to polynomials, and that is a later
 *   task. Read it as "no more than this", never as "this much".
 * - A loop body is analysed once, not once per iteration: a trip count is a runtime quantity and
 *   multiplying a polynomial by it would put a number in the report that the symbolic layer cannot
 *   defend. Node and tensor entries carry their graph's name so a body's contribution stays
 *   attributable.
 * - An intermediate is sized from the index letters of the contraction that writes it. An
 *   intermediate no contraction writes gets no entry, and is counted in @ref skip_reasons.
 * - The rate-limiting verdict ranks flops only, through the same total order the structural passes
 *   use. Two nodes tie only when their flop polynomials are literally identical, so a tie is a
 *   statement about the program rather than an artifact of the comparison.
 *
 * @par Future improvements
 * - A liveness-aware memory high-water polynomial, once intervals and polynomials meet.
 * - Cost formulas for the non-contraction kinds that have one, starting with Gemm and the
 *   element-wise family.
 * - A Python surface for the polynomials themselves, once @ref SymbolicPoly is worth binding as a
 *   type. Today Python reads their renderings.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ScalingAnalysis : public OptimizerPass {
  public:
    APIARY_EXPOSE ScalingAnalysis() = default;

    /// @copydoc OptimizerPass::name
    [[nodiscard]] std::string name() const override { return "ScalingAnalysis"; }

    /// @copydoc OptimizerPass::run
    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::reset_stats
    void reset_stats() override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

    /// Recurse into loop bodies and conditional branches.
    ///
    /// Safe: the pass reads op structure and writes nothing, so visiting a body costs a body's
    /// worth of analysis and changes nothing at any level. See the class documentation for what a
    /// body's cost means once it is in the totals.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /**
     * @brief Write the full per-node report to a stream.
     * @param[in,out] os The stream to write to.
     *
     * Renders every node's three polynomials, every intermediate's size, the totals and the
     * rate-limiting nodes. Space variables resolve to their scale symbols through the registry the
     * analysed graph used, which the pass borrows: rendering after that registry has been
     * destroyed is undefined, exactly as it is for @ref SymbolicPoly::to_string.
     */
    void print_report(std::ostream &os) const;

    /// Per-node costs, in the order the nodes were visited.
    /// @return The costs. Empty when the graph holds no contraction.
    [[nodiscard]] std::vector<NodeCost> const &node_costs() const noexcept { return _node_costs; }

    /// Per-intermediate symbolic sizes, in tensor-id order within each graph visited.
    /// @return The sizes.
    [[nodiscard]] std::vector<IntermediateSize> const &intermediate_sizes() const noexcept { return _intermediate_sizes; }

    /// The sum of every analysed node's flops polynomial.
    /// @return The total. The zero polynomial when nothing was analysed.
    [[nodiscard]] SymbolicPoly const &total_flops() const noexcept { return _total_flops; }

    /// The sum of every analysed node's traffic polynomial.
    /// @return The total. The zero polynomial when nothing was analysed.
    [[nodiscard]] SymbolicPoly const &total_traffic() const noexcept { return _total_traffic; }

    /// The sum of every intermediate's size, an UPPER BOUND on the memory high-water mark.
    /// @return The bound, in elements. See the class documentation for what it does not say.
    [[nodiscard]] SymbolicPoly const &memory_bound() const noexcept { return _memory_bound; }

    /// The node or nodes whose flops polynomial is maximal under the registry's scale order.
    /// @return The rate-limiting nodes. Empty when nothing was analysed; more than one entry only
    ///         when their flop polynomials are identical.
    [[nodiscard]] std::vector<NodeCost> const &rate_limiting() const noexcept { return _rate_limiting; }

    /// How many analysed nodes carry at least one unannotated letter.
    /// @return The count. Nonzero means the report is weaker than annotation could make it.
    APIARY_EXPOSE APIARY_GETTER("num_unannotated_nodes") [[nodiscard]] std::size_t num_unannotated_nodes() const noexcept {
        return _num_unannotated_nodes;
    }

    /// How many nodes were analysed.
    /// @return The number of contraction nodes the pass costed.
    APIARY_EXPOSE APIARY_GETTER("num_analyzed") [[nodiscard]] std::size_t num_analyzed() const noexcept { return _node_costs.size(); }

    // ── Rendered surface ────────────────────────────────────────────────────
    //
    // Everything above hands out polynomials, which is what a C++ pass wants and what no other
    // language binding can use: a SymbolicPoly is a canonical-form container whose only readable
    // form is its rendering, and rendering needs the registry the pass borrowed. So the same
    // numbers leave through these, as strings the pass has already resolved symbols in. A caller
    // that wants to compute with a polynomial is a C++ caller and uses the accessors above.

    /**
     * @brief The same report @ref print_report writes, returned as a string.
     * @return The report. Never empty; it says so when nothing was analysed.
     *
     * The form a caller that is not holding a stream wants, which is every Python caller and every
     * test that asserts on what the report says. Streams are not bound, and a returned string
     * needs no flushing to be readable.
     */
    APIARY_EXPOSE [[nodiscard]] std::string report_string() const;

    /// The rendering of @ref total_flops.
    /// @return The polynomial as text, in the analysed graph's scale symbols.
    APIARY_EXPOSE [[nodiscard]] std::string total_flops_str() const;

    /// The rendering of @ref total_traffic.
    /// @return The polynomial as text, in the analysed graph's scale symbols.
    APIARY_EXPOSE [[nodiscard]] std::string total_traffic_str() const;

    /// The rendering of @ref memory_bound.
    /// @return The polynomial as text. See the class documentation for what the bound does not say.
    APIARY_EXPOSE [[nodiscard]] std::string memory_bound_str() const;

    /// The label of every analysed node, in the order they were visited.
    /// @return The labels, parallel to @ref node_flops.
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> node_labels() const;

    /// The rendered flops polynomial of every analysed node, in the order they were visited.
    /// @return The renderings, parallel to @ref node_labels.
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> node_flops() const;

    /// The labels of the rate-limiting nodes.
    /// @return The labels. More than one entry only when their flop polynomials are identical.
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> rate_limiting_labels() const;

    /// The name of every sized intermediate, in the order they were visited.
    /// @return The names, parallel to @ref intermediate_sizes_str.
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> intermediate_names() const;

    /// The rendered element-count polynomial of every sized intermediate.
    /// @return The renderings, parallel to @ref intermediate_names.
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> intermediate_sizes_str() const;

  private:
    /// @brief Recompute @ref _rate_limiting from @ref _node_costs.
    /// @param[in] registry Registry backing the comparison. May be null.
    void rank_nodes(SpaceRegistry const *registry);

    std::vector<NodeCost>         _node_costs;
    std::vector<IntermediateSize> _intermediate_sizes;
    std::vector<NodeCost>         _rate_limiting;
    SymbolicPoly                  _total_flops;
    SymbolicPoly                  _total_traffic;
    SymbolicPoly                  _memory_bound;
    std::size_t                   _num_unannotated_nodes{0};

    /// Registry of the last graph visited, borrowed for rendering. Null until @ref run.
    SpaceRegistry const *_registry{nullptr};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
