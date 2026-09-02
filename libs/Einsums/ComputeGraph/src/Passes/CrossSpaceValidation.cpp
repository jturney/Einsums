//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CrossSpaceValidation.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "LetterBindings.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Whether an id can be handed to the registry at all. A handle annotated against a DIFFERENT
/// registry than the graph now uses would otherwise make a query throw, and a diagnostic pass that
/// throws takes the pipeline down with it.
bool resolvable(SpaceRegistry const &registry, SpaceId id) {
    return id.valid() && id.value() < registry.size();
}

/// Name a space for a diagnostic. Qualified because this file's own ``detail`` namespace (the
/// letter-binding helpers) would otherwise hide the one that holds the labeller.
std::string name_of(SpaceRegistry const &registry, SpaceId id) {
    return compute_graph::detail::space_label(&registry, id);
}

/// A relation query that answers Unknown rather than throwing on an id the registry cannot resolve.
Tristate safe_disjoint(SpaceRegistry const &registry, SpaceId first, SpaceId second) {
    if (!resolvable(registry, first) || !resolvable(registry, second)) {
        return Tristate::Unknown;
    }
    return registry.is_disjoint(first, second);
}

/// @copydoc safe_disjoint
Tristate safe_contained(SpaceRegistry const &registry, SpaceId inner, SpaceId outer) {
    if (!resolvable(registry, inner) || !resolvable(registry, outer)) {
        return Tristate::Unknown;
    }
    return registry.is_contained(inner, outer);
}

/// One severity level down, floored at Note. An inference is a derivation from someone else's
/// declaration, so a verdict resting on one is reported weaker than the same verdict resting on
/// two declarations.
CrossSpaceSeverity downgrade(CrossSpaceSeverity severity) {
    switch (severity) {
    case CrossSpaceSeverity::Error:
        return CrossSpaceSeverity::Warning;
    case CrossSpaceSeverity::Warning:
    case CrossSpaceSeverity::Note:
        return CrossSpaceSeverity::Note;
    }
    return CrossSpaceSeverity::Note;
}

/// Sort key that puts the most severe finding first.
int severity_rank(CrossSpaceSeverity severity) {
    switch (severity) {
    case CrossSpaceSeverity::Error:
        return 0;
    case CrossSpaceSeverity::Warning:
        return 1;
    case CrossSpaceSeverity::Note:
        return 2;
    }
    return 2;
}

} // namespace

void CrossSpaceValidation::reset_stats() {
    _findings.clear();
    _num_errors   = 0;
    _num_warnings = 0;
    _num_notes    = 0;
}

bool CrossSpaceValidation::run(Graph &graph) {
    // The accumulate-across-subgraphs rule @ref PassCounter states, on a COLLECTION rather than a
    // counter: what this run() added is the tail past this mark, and the mark is the index the
    // report below starts reading from, so the snapshot is needed as a value and not only as a
    // "did it move".
    std::size_t const findings_at_entry = _findings.size();

    SpaceRegistry const &registry = graph.space_registry();

    for (auto const &node : graph.nodes()) {
        if (node.kind != OpKind::Einsum) {
            note_skip("not a contraction node", fmt::format("node {} is a {}", node.id, op_kind_name(node.kind)));
            continue;
        }
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr) {
            note_skip("contraction node carries no descriptor", fmt::format("node {} ('{}')", node.id, node.label));
            continue;
        }

        auto const bindings = detail::letter_bindings(graph, node, *desc);
        if (bindings.empty()) {
            note_skip("no operand carries an index-space annotation", fmt::format("node {} ('{}')", node.id, node.label));
            continue;
        }

        for (auto const &binding : bindings) {
            if (binding.slots.size() < 2) {
                continue;
            }
            auto const &first = binding.slots.front();

            // One finding per DISTINCT second space, compared against the letter's first binding:
            // a letter bound to three spaces is two mistakes, not three, and repeating the same
            // pair once per slot would bury the report in duplicates.
            std::vector<SpaceId> reported;
            for (auto const &second : binding.slots) {
                if (second.space == first.space) {
                    continue;
                }
                if (std::ranges::find(reported, second.space) != reported.end()) {
                    continue;
                }
                reported.push_back(second.space);

                CrossSpaceSeverity severity = CrossSpaceSeverity::Warning;
                std::string        verdict  = "no declared relation settles whether they share an element";
                if (safe_disjoint(registry, first.space, second.space) == Tristate::Yes) {
                    severity = CrossSpaceSeverity::Error;
                    verdict  = "the two spaces are declared disjoint, so the contraction sums over an empty "
                               "intersection and is identically zero as written";
                } else if (safe_contained(registry, first.space, second.space) == Tristate::Yes) {
                    severity = CrossSpaceSeverity::Note;
                    verdict  = fmt::format("'{}' is contained in '{}', so this contraction restricts the parent space",
                                           name_of(registry, first.space), name_of(registry, second.space));
                } else if (safe_contained(registry, second.space, first.space) == Tristate::Yes) {
                    severity = CrossSpaceSeverity::Note;
                    verdict  = fmt::format("'{}' is contained in '{}', so this contraction restricts the parent space",
                                           name_of(registry, second.space), name_of(registry, first.space));
                }

                bool const rests_on_inferred = first.inferred || second.inferred;
                if (rests_on_inferred) {
                    severity = downgrade(severity);
                }

                std::string const first_name  = name_of(registry, first.space);
                std::string const second_name = name_of(registry, second.space);

                std::string message = fmt::format(
                    "{}: index letter '{}' of node {} ('{}') binds space '{}' on operand {} ('{}') and space '{}' on operand {} ('{}'); {}",
                    cross_space_severity_name(severity), binding.letter, node.id, node.label, first_name, first.operand, first.tensor_name,
                    second_name, second.operand, second.tensor_name, verdict);
                if (rests_on_inferred) {
                    message += "; the verdict rests on an INFERRED annotation and is reported one level weaker for it";
                }

                _findings.push_back(CrossSpaceFinding{.severity           = severity,
                                                      .graph_name         = graph.name(),
                                                      .node_id            = node.id,
                                                      .node_label         = node.label,
                                                      .letter             = binding.letter,
                                                      .first_space        = first.space,
                                                      .first_space_name   = first_name,
                                                      .first_operand      = first.operand,
                                                      .first_tensor       = first.tensor,
                                                      .first_tensor_name  = first.tensor_name,
                                                      .second_space       = second.space,
                                                      .second_space_name  = second_name,
                                                      .second_operand     = second.operand,
                                                      .second_tensor      = second.tensor,
                                                      .second_tensor_name = second.tensor_name,
                                                      .rests_on_inferred  = rests_on_inferred,
                                                      .message            = std::move(message)});
                switch (severity) {
                case CrossSpaceSeverity::Error:
                    ++_num_errors;
                    break;
                case CrossSpaceSeverity::Warning:
                    ++_num_warnings;
                    break;
                case CrossSpaceSeverity::Note:
                    ++_num_notes;
                    break;
                }
            }
        }
    }

    if (_findings.size() > findings_at_entry) {
        EINSUMS_LOG_WARN("CrossSpaceValidation: {} error(s), {} warning(s), {} note(s)", _num_errors, _num_warnings, _num_notes);
        report(1, fmt::format("{} error(s), {} warning(s), {} note(s)", _num_errors, _num_warnings, _num_notes));
        for (std::size_t i = findings_at_entry; i < _findings.size(); ++i) {
            report(2, _findings[i].message);
        }
    }

    // Diagnostic pass, never changes the node list and never raises.
    return false;
}

std::vector<std::string> CrossSpaceValidation::explain() const {
    if (_findings.empty()) {
        return {};
    }

    std::vector<std::string> lines;
    lines.push_back(fmt::format("CrossSpaceValidation: {} error(s), {} warning(s), {} note(s) across {} letter binding(s)", _num_errors,
                                _num_warnings, _num_notes, _findings.size()));
    for (auto const &finding : _findings) {
        if (finding.severity == CrossSpaceSeverity::Error) {
            lines.push_back(fmt::format("CrossSpaceValidation: {}", finding.message));
        }
    }
    return lines;
}

void CrossSpaceValidation::print_report(std::ostream &os) const {
    os << "=== CrossSpaceValidation ===\n";
    os << fmt::format("  {} error(s), {} warning(s), {} note(s)\n", _num_errors, _num_warnings, _num_notes);

    if (_findings.empty()) {
        os << "  no cross-space conflicts found\n";
        return;
    }

    std::vector<CrossSpaceFinding> ordered = _findings;
    std::ranges::stable_sort(ordered, [](CrossSpaceFinding const &lhs, CrossSpaceFinding const &rhs) {
        return severity_rank(lhs.severity) < severity_rank(rhs.severity);
    });

    for (auto const &finding : ordered) {
        os << fmt::format("  [{}#{}] {}\n", finding.graph_name, finding.node_id, finding.message);
    }
}

std::string CrossSpaceValidation::report_string() const {
    std::ostringstream out;
    print_report(out);
    return std::move(out).str();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
