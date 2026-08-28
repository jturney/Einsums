//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

std::string_view term_kind_name(TermKind kind) {
    switch (kind) {
    case TermKind::Leaf:
        return "leaf";
    case TermKind::Contraction:
        return "contraction";
    case TermKind::Elementwise:
        return "elementwise";
    case TermKind::Scale:
        return "scale";
    case TermKind::Sum:
        return "sum";
    }
    return "unknown";
}

bool is_raisable(OpKind kind) {
    switch (kind) {
    case OpKind::Einsum:
    case OpKind::Permute:
    case OpKind::Transpose:
    case OpKind::Scale:
    case OpKind::Axpby:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::ElementTransform:
    case OpKind::Dot:
    case OpKind::Trace:
        return true;
    default:
        return false;
    }
}

TermId TensorExpr::add(ExprTerm term) {
    terms.push_back(std::move(term));
    return static_cast<TermId>(terms.size() - 1);
}

SymbolicCost TensorExpr::total_cost() const {
    SymbolicCost out;
    for (auto const &term : terms) {
        out.flops += term.cost.flops;
        out.traffic += term.cost.traffic;
        out.resident += term.cost.resident;
    }
    return out;
}

namespace {

/// ``name[i,j]`` for a leaf, or ``name`` when it has no indices. Spaces are not
/// printed: a rendering meant to be diffed wants one line per statement, and the
/// spaces are recoverable from the graph when a reader wants them.
std::string render_indices(std::vector<ExprIndex> const &indices) {
    if (indices.empty()) {
        return {};
    }
    std::string out = "[";
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        out += indices[i].letter;
    }
    out += ']';
    return out;
}

/// A prefactor, omitted entirely when it is one, because ``1 * A`` reads worse
/// than ``A`` and the whole point of the rendering is to be read.
std::string render_factor(PrefactorScalar const &factor) {
    if (auto const *real = std::get_if<double>(&factor); real != nullptr && *real == 1.0) {
        return {};
    }
    return to_string(factor) + " * ";
}

// NOLINTNEXTLINE(misc-no-recursion): the arena is a DAG and terms name children by index.
std::string render_term(TensorExpr const &expr, TermId id) {
    if (id == invalid_term || id >= expr.terms.size()) {
        return "<none>";
    }
    auto const &term = expr.at(id);
    switch (term.kind) {
    case TermKind::Leaf:
        return term.name + render_indices(term.indices);
    case TermKind::Contraction: {
        std::string out = render_factor(term.factor);
        for (std::size_t i = 0; i < term.operands.size(); ++i) {
            if (i != 0) {
                out += ' ';
            }
            auto const operand = render_term(expr, term.operands[i]);
            // The operand's own index list is printed, not the leaf's: the same
            // tensor may enter two contractions under different letters, and the
            // contraction's list is the one that says what THIS term does.
            out += operand.substr(0, operand.find('['));
            out += render_indices(i < term.operand_indices.size() ? term.operand_indices[i] : std::vector<ExprIndex>{});
            if (i < term.conjugate.size() && term.conjugate[i]) {
                out += '*';
            }
        }
        return out;
    }
    case TermKind::Elementwise: {
        std::string out = fmt::format("{}(", term.name.empty() ? std::string(op_kind_name(term.element_kind)) : term.name);
        for (std::size_t i = 0; i < term.operands.size(); ++i) {
            if (i != 0) {
                out += ", ";
            }
            auto const operand = render_term(expr, term.operands[i]);
            if (i < term.operand_indices.size()) {
                // The letters this TERM reads the operand by, which is what a rewrite that
                // renamed one has to be able to show. A leaf's own indices are positional axis
                // names (#0, #1) and would render a substitution as though nothing happened,
                // which is the one thing the dump exists to make visible.
                out += operand.substr(0, operand.find('['));
                out += render_indices(term.operand_indices[i]);
            } else {
                out += operand;
            }
        }
        out += ')';
        return out;
    }
    case TermKind::Scale:
        return render_factor(term.factor) + (term.operands.empty() ? "<none>" : render_term(expr, term.operands[0]));
    case TermKind::Sum: {
        std::string out;
        for (std::size_t i = 0; i < term.operands.size(); ++i) {
            if (i != 0) {
                out += " + ";
            }
            out += render_term(expr, term.operands[i]);
        }
        return out;
    }
    }
    return "<unknown>";
}

} // namespace

std::string TensorExpr::to_string(SpaceRegistry const *registry) const {
    std::string out;
    for (auto const &statement : statements) {
        out += statement.target_name;
        out += render_indices(statement.target_indices);
        out += is_zero(statement.target_prefactor) ? " = " : " += ";
        if (!is_zero(statement.target_prefactor)) {
            if (auto const *real = std::get_if<double>(&statement.target_prefactor); real == nullptr || *real != 1.0) {
                // An accumulation whose destination prefactor is not one scales
                // what is already there, and a rendering that hid that would
                // describe different arithmetic.
                out += fmt::format("({} * self) + ", compute_graph::to_string(statement.target_prefactor));
            }
        }
        out += render_term(*this, statement.value);
        auto const &term = statement.value < terms.size() ? at(statement.value) : ExprTerm{};
        if (!term.cost.flops.is_zero()) {
            out += fmt::format("    # {}", term.cost.flops.to_string(registry));
        }
        out += '\n';
    }
    return out;
}

// ── Region formation ───────────────────────────────────────────────────────

namespace {

/// An ElementTransform whose kernel has no registered name cannot be rebuilt
/// from the algebra, which is the same reason it cannot be saved. Treating it as
/// a barrier is the honest answer: raising it would produce a term that lowers
/// into a different kernel or none.
bool raisable_here(Node const &node) {
    if (!is_raisable(node.kind)) {
        return false;
    }
    if (node.kind == OpKind::ElementTransform) {
        auto const *desc = std::get_if<ElementTransformDescriptor>(&node.op_data);
        return desc != nullptr && !desc->op_name.empty();
    }
    return true;
}

} // namespace

std::vector<Region> form_regions(Graph const &graph, EscapeAnalysis const &escapes, RegionOptions const &options) {
    std::vector<Region> regions;
    auto const         &nodes = graph.nodes();

    std::size_t position = 0;
    while (position < nodes.size()) {
        if (!raisable_here(nodes[position])) {
            ++position;
            continue;
        }
        std::size_t const first = position;
        while (position < nodes.size() && raisable_here(nodes[position])) {
            ++position;
        }

        Region region;
        region.first = first;
        region.last  = position;
        for (std::size_t i = first; i < position; ++i) {
            region.nodes.push_back(nodes[i].id);
        }
        if (region.nodes.size() < options.min_nodes) {
            continue;
        }

        std::unordered_set<NodeId> const members(region.nodes.begin(), region.nodes.end());

        // Classify every tensor the run touches, in FIRST-MENTION order over the
        // run: inputs in node order then outputs in node order, which is the same
        // walk GraphIR uses to make two captures agree. An unordered walk here
        // would give the region's operand list a different order on every run and
        // every rewrite built from it would inherit that.
        std::vector<TensorId> mentioned;
        auto const            mention = [&mentioned](TensorId id) {
            if (std::ranges::find(mentioned, id) == mentioned.end()) {
                mentioned.push_back(id);
            }
        };
        for (std::size_t i = first; i < position; ++i) {
            for (auto const tid : nodes[i].inputs) {
                mention(tid);
            }
            for (auto const tid : nodes[i].outputs) {
                mention(tid);
            }
        }

        for (auto const tid : mentioned) {
            bool written_inside = false;
            for (std::size_t i = first; i < position && !written_inside; ++i) {
                written_inside = std::ranges::find(nodes[i].outputs, tid) != nodes[i].outputs.end();
            }
            if (!written_inside) {
                region.inputs.push_back(tid);
                continue;
            }
            if (escapes.classify(tid, members) == Escape::Dissolvable) {
                region.internal.push_back(tid);
            } else {
                region.outputs.push_back(tid);
            }
        }
        regions.push_back(std::move(region));
    }
    return regions;
}

// ── Raise ──────────────────────────────────────────────────────────────────

namespace {

/// The letters of one operand, paired with the space the node bound to each.
std::vector<ExprIndex> indices_from(std::vector<std::string> const &letters, EinsumDescriptor const &desc) {
    std::vector<ExprIndex> out;
    out.reserve(letters.size());
    for (auto const &letter : letters) {
        ExprIndex index;
        index.letter = letter;
        if (auto const space = desc.space_for_letter(letter); space.has_value()) {
            index.space = *space;
        }
        out.push_back(std::move(index));
    }
    return out;
}

/// The axes of a tensor with no index letters of its own, named positionally.
/// An elementwise op has no letters in the descriptor, and the algebra still has
/// to say how wide the value is, so the axes are named after the tensor's own
/// spaces where it has them and left anonymous where it does not.
std::vector<ExprIndex> axes_of(Graph const &graph, TensorId id) {
    std::vector<ExprIndex> out;
    TensorHandle const    *handle = graph.find_tensor(id);
    if (handle == nullptr) {
        return out;
    }
    out.reserve(handle->rank);
    for (std::size_t axis = 0; axis < handle->rank; ++axis) {
        ExprIndex index;
        index.letter = fmt::format("#{}", axis);
        if (axis < handle->spaces.size()) {
            index.space = handle->spaces[axis];
        }
        out.push_back(std::move(index));
    }
    return out;
}

std::string name_of(Graph const &graph, TensorId id) {
    TensorHandle const *handle = graph.find_tensor(id);
    return handle == nullptr ? fmt::format("t{}", static_cast<std::uint64_t>(id)) : handle->name;
}

/// One leaf per tensor per expression, so a tensor entering two statements is
/// one arena entry and a rewrite that rewrites the leaf rewrites both uses.
TermId leaf_for(TensorExpr &expr, std::unordered_map<TensorId, TermId> &cache, Graph const &graph, TensorId id) {
    if (auto const hit = cache.find(id); hit != cache.end()) {
        return hit->second;
    }
    ExprTerm leaf;
    leaf.kind         = TermKind::Leaf;
    leaf.name         = name_of(graph, id);
    leaf.tensor       = id;
    leaf.indices      = axes_of(graph, id);
    TermId const term = expr.add(std::move(leaf));
    cache.emplace(id, term);
    return term;
}

} // namespace

expected<TensorExpr, RaiseFailure> raise_region(Graph const &graph, Region const &region) {
    TensorExpr                           expr;
    std::unordered_map<TensorId, TermId> leaves;

    for (std::size_t offset = 0; offset < region.nodes.size(); ++offset) {
        std::size_t const position = region.first + offset;
        // By position, and then checked against the id the region recorded. A
        // region is formed and raised inside one pass run with nothing mutating
        // in between, so the positions hold; asserting it here is what turns a
        // future caller who breaks that assumption into a decline rather than
        // into a rewrite of the wrong nodes.
        if (position >= graph.nodes().size() || graph.nodes()[position].id != region.nodes[offset]) {
            return unexpected(RaiseFailure{.reason = "the graph moved under the region",
                                           .detail = fmt::format("position {} no longer holds node id {}", position,
                                                                 static_cast<std::uint64_t>(region.nodes[offset]))});
        }
        Node const *node = &graph.nodes()[position];
        if (node->outputs.empty()) {
            return unexpected(RaiseFailure{.reason = "a region node writes nothing", .detail = fmt::format("node '{}'", node->label)});
        }

        ExprStatement statement;
        statement.target       = node->outputs[0];
        statement.target_name  = name_of(graph, statement.target);
        statement.origin       = node->id;
        statement.origin_kind  = node->kind;
        statement.origin_label = node->label;

        if (node->kind == OpKind::Einsum) {
            auto const *desc = std::get_if<EinsumDescriptor>(&node->op_data);
            if (desc == nullptr) {
                return unexpected(
                    RaiseFailure{.reason = "a contraction carries no einsum descriptor", .detail = fmt::format("node '{}'", node->label)});
            }
            // The LIVE index lists, not the capture snapshot. A pass that rewrote
            // an index list wrote it through the shared block, and raising the
            // snapshot would raise the algebra as it was before that pass. The
            // two are different TYPES rather than two copies of one: the live
            // block holds a ParsedEinsumSpec and the snapshot a
            // packed_gemm::ContractionSpec, so the letters are taken from
            // whichever is present rather than the object.
            bool const                      live   = desc->indices != nullptr;
            std::vector<std::string> const &a_list = live ? desc->indices->spec.a_indices : desc->spec.a_indices;
            std::vector<std::string> const &b_list = live ? desc->indices->spec.b_indices : desc->spec.b_indices;
            std::vector<std::string> const &c_list = live ? desc->indices->spec.c_indices : desc->spec.c_indices;
            if (a_list.empty() && b_list.empty()) {
                return unexpected(
                    RaiseFailure{.reason = "a contraction carries no index lists", .detail = fmt::format("node '{}'", node->label)});
            }
            if (node->inputs.size() < 2) {
                return unexpected(
                    RaiseFailure{.reason = "a contraction names fewer than two operands", .detail = fmt::format("node '{}'", node->label)});
            }
            auto const &params = desc->params;

            ExprTerm term;
            term.kind    = TermKind::Contraction;
            term.factor  = params != nullptr ? params->ab_pf : desc->ab_prefactor;
            term.indices = indices_from(c_list, *desc);
            term.operands.push_back(leaf_for(expr, leaves, graph, node->inputs[0]));
            term.operands.push_back(leaf_for(expr, leaves, graph, node->inputs[1]));
            term.operand_indices.push_back(indices_from(a_list, *desc));
            term.operand_indices.push_back(indices_from(b_list, *desc));
            term.conjugate.push_back(params != nullptr ? params->conj_a : desc->conj_a);
            term.conjugate.push_back(params != nullptr ? params->conj_b : desc->conj_b);
            term.cost = symbolic_cost_for(*desc);

            statement.target_prefactor = params != nullptr ? params->c_pf : desc->c_prefactor;
            statement.target_indices   = term.indices;
            statement.value            = expr.add(std::move(term));
        } else {
            // Named, carried, not interpreted. See the header note: an algebraic
            // pass may move or delete one of these, and does not reach inside it.
            ExprTerm term;
            term.kind         = TermKind::Elementwise;
            term.element_kind = node->kind;
            term.descriptor   = node->op_data;
            term.indices      = axes_of(graph, statement.target);
            if (auto const *element = std::get_if<ElementTransformDescriptor>(&node->op_data); element != nullptr) {
                term.name = element->op_name;
            }
            for (auto const input : node->inputs) {
                term.operands.push_back(leaf_for(expr, leaves, graph, input));
            }
            statement.target_indices = term.indices;
            // A dense elementwise node lists its destination among its inputs
            // when it reads it, which is the RMW convention the schedulers rely
            // on, so that is also how the algebra learns it accumulates.
            statement.target_prefactor = std::ranges::find(node->inputs, statement.target) != node->inputs.end()
                                             ? PrefactorScalar{double{1}}
                                             : PrefactorScalar{double{0}};
            statement.value            = expr.add(std::move(term));
        }
        expr.statements.push_back(std::move(statement));
    }
    return expr;
}

// ── Lower ──────────────────────────────────────────────────────────────────

namespace {

/// Rank and dtype of a node's destination, which is the key @ref build_executor
/// dispatches on. Same rule as the serializer's, deliberately: two derivations
/// of one key is one too many.
std::pair<packed_gemm::ScalarType, std::size_t> destination_key(Graph const &graph, TensorId id) {
    if (TensorHandle const *handle = graph.find_tensor(id); handle != nullptr) {
        return {handle->dtype, handle->rank};
    }
    return {packed_gemm::ScalarType::Unknown, 0};
}

} // namespace

expected<void, RaiseFailure> lower_region(Graph &graph, Region const &region, TensorExpr const &expr) {
    std::vector<Node> emitted;
    emitted.reserve(expr.statements.size());

    for (auto const &statement : expr.statements) {
        if (statement.value == invalid_term || statement.value >= expr.terms.size()) {
            return unexpected(RaiseFailure{.reason = "a statement has no value term", .detail = statement.target_name});
        }
        auto const &term = expr.at(statement.value);

        if (term.kind == TermKind::Contraction) {
            if (term.operands.size() != 2 || term.operand_indices.size() != 2) {
                // Multi-operand contractions are representable in the IR and are
                // what a factorization pass produces; nothing lowers them yet,
                // because the node set has no multi-operand contraction to lower
                // them TO. A pass that emits one has to pair it with a binary
                // decomposition, and declining here says so rather than
                // silently dropping operands.
                return unexpected(
                    RaiseFailure{.reason = "a contraction with other than two operands has no node form",
                                 .detail = fmt::format("target '{}' has {} operands", statement.target_name, term.operands.size())});
            }
            ParsedEinsumSpec spec;
            auto const       letters = [](std::vector<ExprIndex> const &indices) {
                std::vector<std::string> out;
                out.reserve(indices.size());
                for (auto const &index : indices) {
                    out.push_back(index.letter);
                }
                return out;
            };
            spec.a_indices = letters(term.operand_indices[0]);
            spec.b_indices = letters(term.operand_indices[1]);
            spec.c_indices = letters(statement.target_indices);
            spec.raw       = fmt::format("{} <- {} ; {}", fmt::join(spec.c_indices, ","), fmt::join(spec.a_indices, ","),
                                         fmt::join(spec.b_indices, ","));

            auto const &a_leaf = expr.at(term.operands[0]);
            auto const &b_leaf = expr.at(term.operands[1]);
            try {
                emitted.push_back(graph.make_einsum_node(a_leaf.tensor, b_leaf.tensor, statement.target, spec, statement.target_prefactor,
                                                         term.factor, !term.conjugate.empty() && term.conjugate[0],
                                                         term.conjugate.size() > 1 && term.conjugate[1], statement.origin_label));
            } catch (std::exception const &error) {
                return unexpected(RaiseFailure{.reason = "a contraction could not be rebuilt from the algebra",
                                               .detail = fmt::format("target '{}': {}", statement.target_name, error.what())});
            }
            continue;
        }

        if (term.kind != TermKind::Elementwise) {
            return unexpected(RaiseFailure{.reason = "a term kind has no node form yet",
                                           .detail = fmt::format("target '{}' is a {}", statement.target_name, term_kind_name(term.kind))});
        }

        Node node;
        node.id    = graph.reserve_node_id();
        node.kind  = term.element_kind;
        node.label = statement.origin_label.empty() ? fmt::format("{}({})", op_kind_name(term.element_kind), statement.target_name)
                                                    : statement.origin_label;
        node.outputs.push_back(statement.target);
        for (auto const operand : term.operands) {
            node.inputs.push_back(expr.at(operand).tensor);
        }
        node.op_data = term.descriptor;

        auto const [dtype, rank] = destination_key(graph, statement.target);
        try {
            node.execute = build_executor(node.kind, dtype, rank, node.op_data, graph, std::span<TensorId const>{node.inputs},
                                          std::span<TensorId const>{node.outputs});
        } catch (std::exception const &error) {
            return unexpected(RaiseFailure{.reason = "an elementwise term could not be rebuilt from the algebra",
                                           .detail = fmt::format("target '{}': {}", statement.target_name, error.what())});
        }
        emitted.push_back(std::move(node));
    }

    // Nothing was mutated until here, so a refusal above left the graph exactly
    // as it was. Erase and splice together: the replacement lands at the region's
    // first position, so no writer can end up behind a reader that survived.
    std::vector<bool> remove(region.last, false);
    for (std::size_t i = region.first; i < region.last; ++i) {
        remove[i] = true;
    }
    graph.erase_nodes(remove);
    graph.insert_node_groups({{region.first, std::move(emitted)}});
    graph.note_structural_change();
    graph.topological_sort();
    return {};
}

EINSUMS_NAMESPACE_END(compute_graph)
