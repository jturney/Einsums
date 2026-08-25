//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/// @brief How much a @ref CrossSpaceFinding is worth acting on.
// With the annotation macros in front, clang-format reads the enum base as a braced initializer
// and glues the brace to it.
// clang-format off
enum class APIARY_EXPOSE APIARY_MODULE("graph") CrossSpaceSeverity : std::uint8_t {
    Note,    ///< Legitimate, listed for the record: a restriction of one space to a subspace.
    Warning, ///< Nothing declared settles whether the two spaces overlap.
    Error,   ///< The two spaces are declared disjoint, so the contraction is identically zero.
};
// clang-format on

/**
 * @brief Name of a @ref CrossSpaceSeverity value.
 * @param[in] severity The value to name.
 * @return "note", "warning" or "error".
 */
[[nodiscard]] inline std::string_view cross_space_severity_name(CrossSpaceSeverity severity) noexcept {
    switch (severity) {
    case CrossSpaceSeverity::Note:
        return "note";
    case CrossSpaceSeverity::Warning:
        return "warning";
    case CrossSpaceSeverity::Error:
        return "error";
    }
    return "note";
}

/**
 * @brief One index letter of one contraction binding two different index spaces.
 *
 * A value type, so a caller may keep it after the pass and the graph are gone. The two sides are
 * named "first" and "second" in operand order (A before B before C), and the message repeats
 * everything the fields carry so a report needs nothing else.
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") CrossSpaceFinding {
    // Every field is bound read-only: a finding is a verdict the pass reached, and a caller that
    // edits one is only lying to whatever reads it next.
    APIARY_EXPOSE APIARY_READONLY CrossSpaceSeverity severity{CrossSpaceSeverity::Note}; ///< How much to worry.
    APIARY_EXPOSE APIARY_READONLY std::string graph_name;                                ///< Graph the node belongs to.
    APIARY_EXPOSE APIARY_READONLY NodeId      node_id{0};                                ///< The node's id within that graph.
    APIARY_EXPOSE APIARY_READONLY std::string node_label;                                ///< The node's human-readable label.
    APIARY_EXPOSE APIARY_READONLY std::string letter;                                    ///< The index letter that binds both spaces.

    APIARY_EXPOSE APIARY_READONLY SpaceId first_space;            ///< Space bound on the earlier operand.
    APIARY_EXPOSE APIARY_READONLY std::string first_space_name;   ///< That space's registered name.
    APIARY_EXPOSE APIARY_READONLY char const *first_operand{"A"}; ///< Which operand it sits on: "A", "B" or "C".
    APIARY_EXPOSE APIARY_READONLY TensorId    first_tensor{0};    ///< That operand's tensor id.
    APIARY_EXPOSE APIARY_READONLY std::string first_tensor_name;  ///< That operand's tensor name.

    APIARY_EXPOSE APIARY_READONLY SpaceId second_space;            ///< Space bound on the later operand.
    APIARY_EXPOSE APIARY_READONLY std::string second_space_name;   ///< That space's registered name.
    APIARY_EXPOSE APIARY_READONLY char const *second_operand{"B"}; ///< Which operand it sits on.
    APIARY_EXPOSE APIARY_READONLY TensorId    second_tensor{0};    ///< That operand's tensor id.
    APIARY_EXPOSE APIARY_READONLY std::string second_tensor_name;  ///< That operand's tensor name.

    /// Whether either side's annotation was INFERRED rather than declared. A finding that rests on
    /// an inference is reported one severity level lower than the same finding on two declarations.
    APIARY_EXPOSE APIARY_READONLY bool rests_on_inferred{false};

    APIARY_EXPOSE APIARY_READONLY std::string message; ///< The full sentence, ready to print.
};

/**
 * @brief Flag contraction letters that bind one operand's slot against a slot of another space.
 *
 * The bug this exists to catch is a letter binding an ``occ`` slot of one operand against a
 * ``virt`` slot of another. In a coupled-cluster transcription that is almost always a mistake,
 * and it is one the differential fuzzers cannot catch, because a hand-derived reference and its
 * implementation are self-consistently wrong together.
 *
 * @par What capture already catches, and why this pass is still needed
 * Capture raises when a letter binds two different DECLARED spaces within one contraction, so a
 * program annotated before it is captured is already policed. This pass covers everything capture
 * cannot see:
 *
 * - annotations that arrive AFTER capture, through ``Graph::annotate_spaces``, which capture never
 *   revisits,
 * - annotations ``SpacePropagation`` inferred, which that pass declines to argue about by design
 *   and leaves for diagnosis here,
 * - relation-aware verdicts. Capture asks only whether two ids are equal. This pass asks the
 *   registry whether the two spaces are disjoint, whether one contains the other, or whether
 *   nothing declared relates them, and reports three different things.
 *
 * @par The verdicts
 * A letter is re-derived from the operands' CURRENT handle annotations, never from the map frozen
 * into the node at capture, and each slot binding it is compared against the first:
 *
 * - Same space: nothing to report.
 * - Declared DISJOINT: an error. The contraction as written sums over an empty intersection, so it
 *   is identically zero, which is essentially never what the author meant.
 * - One CONTAINED in the other: a note. Contracting a ``pno`` slot against a ``virt`` slot is a
 *   restriction of the parent space to a subspace, which the design's containment reasoning treats
 *   as legitimate. Listed because a restriction that was not intended looks exactly like one that
 *   was.
 * - Nothing declared either way: a warning. The registry holds only what was declared, and
 *   "unknown" is a first-class answer that must be treated as carefully as "no".
 *
 * A finding involving a slot whose annotation was INFERRED is reported one level lower and says so.
 * An inference is a derivation from someone else's declaration, and an authoritative verdict resting
 * on a derived premise is worse than a weak verdict resting on a firm one.
 *
 * @par The pass never fails a pipeline
 * It never throws, never mutates, and @ref run always reports "not modified". A diagnosis is
 * delivered through @ref findings, @ref explain and @ref print_report, and what to do about it is
 * the caller's decision.
 *
 * @par Example (C++)
 * @code
 * graph.annotate_spaces(A, {occ, virt});
 * graph.annotate_spaces(B, {occ, aux});     // 'a' meets virt on A and occ on B
 * auto [modified, check] = graph.apply<cg::passes::CrossSpaceValidation>();
 * for (auto const &finding : check.findings()) {
 *     std::cerr << finding.message << '\n';
 * }
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * check = cg.CrossSpaceValidation()
 * pm = cg.PassManager()
 * pm.add(check)
 * pm.run(g)
 * for finding in check.findings:
 *     print(finding.severity, finding.message)
 * print(check.report_string())
 * @endcode
 *
 * @par Limitations
 * - Contraction nodes only. Every other kind is counted in @ref skip_reasons; a linear combination
 *   whose operands disagree slot by slot is a real cross-space bug and is not yet reported here.
 * - One finding per pair of distinct spaces on one letter, compared against the letter's first
 *   binding, so a letter binding three spaces yields two findings rather than three.
 * - An unannotated program yields nothing, which is the honest answer: with no premises the
 *   registry can prove nothing. The way to make the pass useful is to annotate.
 * - The verdicts read only DECLARED relations. Two spaces that genuinely never overlap but were
 *   never declared disjoint produce a warning, not an error.
 *
 * @par Future improvements
 * - Cover ``Axpby`` and the element-wise kinds, whose operands must agree slot by slot.
 * - An option to promote errors to an exception, for a build that wants a wrong transcription to
 *   stop the run rather than to be reported.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT CrossSpaceValidation : public OptimizerPass {
  public:
    APIARY_EXPOSE CrossSpaceValidation() = default;

    /// @copydoc OptimizerPass::name
    [[nodiscard]] std::string name() const override { return "CrossSpaceValidation"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::Diagnostic; }

    /// @copydoc OptimizerPass::run
    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::reset_stats
    void reset_stats() override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

    /// Recurse into loop bodies and conditional branches.
    ///
    /// Safe: the pass reads op structure and writes nothing, and a body's letters are as worth
    /// checking as the top level's. Findings carry their graph's name so they stay attributable.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /**
     * @brief Write every finding to a stream, most severe first.
     * @param[in,out] os The stream to write to.
     */
    void print_report(std::ostream &os) const;

    /**
     * @brief The same report @ref print_report writes, returned as a string.
     * @return The report, most severe finding first. Empty for a clean or an unannotated graph.
     *
     * The form a caller that is not holding a stream wants, which is every Python caller and every
     * test that asserts on what the report says. Streams are not bound, and a returned string
     * needs no flushing to be readable.
     */
    APIARY_EXPOSE [[nodiscard]] std::string report_string() const;

    /// Every finding, in the order the nodes were visited and letters sorted within a node.
    /// @return The findings. Empty for a clean or an unannotated graph.
    APIARY_EXPOSE APIARY_GETTER("findings") [[nodiscard]] std::vector<CrossSpaceFinding> const &findings() const noexcept {
        return _findings;
    }

    /// How many findings are errors.
    /// @return The count of @ref CrossSpaceSeverity::Error findings.
    APIARY_EXPOSE APIARY_GETTER("num_errors") [[nodiscard]] std::size_t num_errors() const noexcept { return _num_errors; }

    /// How many findings are warnings.
    /// @return The count of @ref CrossSpaceSeverity::Warning findings.
    APIARY_EXPOSE APIARY_GETTER("num_warnings") [[nodiscard]] std::size_t num_warnings() const noexcept { return _num_warnings; }

    /// How many findings are notes.
    /// @return The count of @ref CrossSpaceSeverity::Note findings.
    APIARY_EXPOSE APIARY_GETTER("num_notes") [[nodiscard]] std::size_t num_notes() const noexcept { return _num_notes; }

  private:
    std::vector<CrossSpaceFinding> _findings;
    std::size_t                    _num_errors{0};
    std::size_t                    _num_warnings{0};
    std::size_t                    _num_notes{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
