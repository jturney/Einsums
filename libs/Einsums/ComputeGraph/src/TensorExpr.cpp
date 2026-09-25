//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <complex>
#include <map>
#include <set>
#include <span>
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

bool is_grouped_raisable(OpKind kind) {
    switch (kind) {
    case OpKind::GroupedBatchedGemm:
    case OpKind::GroupedDot:
    case OpKind::GroupedAxpby:
    case OpKind::GroupedPermute:
    case OpKind::GroupedDirectProduct:
    case OpKind::GroupedDirectDivision:
        return true;
    default:
        return false;
    }
}

std::size_t RaggedFamily::typical_extent(std::string_view letter) const {
    for (auto const &[name, values] : extents) {
        if (name != letter) {
            continue;
        }
        if (values.empty()) {
            return 0;
        }
        std::size_t total = 0;
        for (auto const value : values) {
            total += value;
        }
        // Rounded to nearest rather than truncated, so a family whose members
        // are all 3 does not read as 2 through an off-by-one in the division.
        return (total + values.size() / 2) / values.size();
    }
    return 0;
}

TermId TensorExpr::add(ExprTerm term) {
    terms.push_back(std::move(term));
    return static_cast<TermId>(terms.size() - 1);
}

namespace {

/// Mark @p id and everything below it, so a cost sums what the expression COMPUTES.
// NOLINTNEXTLINE(misc-no-recursion): the arena is a DAG and terms name children by index.
void mark_reachable(std::vector<ExprTerm> const &terms, TermId id, std::vector<bool> &seen) {
    if (id == invalid_term || id >= terms.size() || seen[id]) {
        return;
    }
    seen[id] = true;
    for (auto const child : terms[id].operands) {
        mark_reachable(terms, child, seen);
    }
}

} // namespace

SymbolicCost TensorExpr::total_cost() const {
    // Only the terms the STATEMENTS reach. A rewrite replaces a statement's value and leaves the
    // term it replaced in the arena, since nothing renumbers indices mid-rewrite; summing the
    // arena would therefore count the arithmetic a rewrite just removed and report a
    // before-and-after that never changes. That is not a cosmetic difference: this number is
    // what a report offers as evidence the rewrite was worth making.
    std::vector<bool> reachable(terms.size(), false);
    for (auto const &statement : statements) {
        mark_reachable(terms, statement.value, reachable);
    }

    SymbolicCost out;
    for (std::size_t id = 0; id < terms.size(); ++id) {
        if (!reachable[id]) {
            continue;
        }
        out.flops += terms[id].cost.flops;
        out.traffic += terms[id].cost.traffic;
        out.resident += terms[id].cost.resident;
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
bool raisable_here(Node const &node, RegionOptions const &options) {
    if (options.grouped && is_grouped_raisable(node.kind)) {
        // A blocked grouped batch writes column ranges of shared bases and
        // declares only the DISTINCT bases as outputs, so its member list of
        // destinations is not on the node at all. It stays a barrier rather
        // than raising into an algebra that would name the wrong destinations.
        auto const *gemm = node.op_data.get_if<GroupedBatchedGemmDescriptor>();
        return gemm == nullptr || !gemm->blocked;
    }
    if (!is_raisable(node.kind)) {
        return false;
    }
    if (node.kind == OpKind::ElementTransform) {
        auto const *desc = node.op_data.get_if<ElementTransformDescriptor>();
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
        if (!raisable_here(nodes[position], options)) {
            ++position;
            continue;
        }
        std::size_t const first = position;
        while (position < nodes.size() && raisable_here(nodes[position], options)) {
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

        // A grouped node's destinations are ONE value. A rewrite that dissolved
        // half a family would leave the other half being written by a node it
        // had just removed, so a family whose every member is dissolvable is
        // dissolvable and one with a single escaping member is not.
        std::unordered_set<TensorId> pinned_by_family;
        for (std::size_t i = first; i < position; ++i) {
            if (!is_grouped_raisable(nodes[i].kind)) {
                continue;
            }
            bool const whole =
                std::ranges::all_of(nodes[i].outputs, [&](TensorId tid) { return escapes.classify(tid, members) == Escape::Dissolvable; });
            if (!whole) {
                pinned_by_family.insert(nodes[i].outputs.begin(), nodes[i].outputs.end());
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
            if (pinned_by_family.count(tid) == 0 && escapes.classify(tid, members) == Escape::Dissolvable) {
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

// ── Grouped families ───────────────────────────────────────────────────────

/// The live shape of one member, read through the handle's impl rather than off
/// the registration snapshot, which is what the handle's own note asks for: the
/// snapshot is null for a tensor that was deferred when it was registered.
struct MemberShape {
    std::vector<std::size_t> dims;
    std::size_t              leading{0};
    bool                     ok{false};
};

MemberShape member_shape(Graph const &graph, TensorId id) {
    MemberShape         out;
    TensorHandle const *handle = graph.find_tensor(id);
    if (handle == nullptr || handle->impl_fn == nullptr) {
        return out;
    }
    void *raw = handle->impl_fn();
    if (raw == nullptr) {
        return out;
    }
    detail::dispatch_scalar_type(handle->dtype, [&]<typename T>(T /*tag*/) {
        auto const *impl = static_cast<einsums::detail::TensorImpl<T> const *>(raw);
        out.dims.reserve(impl->rank());
        for (std::size_t axis = 0; axis < impl->rank(); ++axis) {
            out.dims.push_back(static_cast<std::size_t>(impl->dim(axis)));
        }
        out.leading = impl->rank() == 2 ? static_cast<std::size_t>(impl->get_lda()) : 0;
        out.ok      = true;
    });
    return out;
}

/// The member letter of family @p id. ``#`` so it cannot collide with an
/// author's letter, and a dump reads it as the synthetic axis it is.
std::string member_letter(FamilyId id) {
    return fmt::format("#m{}", static_cast<std::uint32_t>(id));
}

/// A letter family @p id introduces for one of its ragged axes.
std::string family_letter(FamilyId id, std::string_view tag) {
    return fmt::format("#{}{}", static_cast<std::uint32_t>(id), tag);
}

/// The space a family's member letter ranges over.
///
/// Keyed on the MEMBER COUNT rather than on the node, for two reasons that pull
/// the same way. Registration has no inverse, so a name minted per node would
/// grow the process registry by one entry per optimize call; and two grouped
/// nodes over one pair list have the same count by construction, so a
/// count-keyed space is the one they should share. Re-registering an identical
/// space is a lookup, which is what makes raising the same region twice free.
SpaceId family_space(Graph const &graph, std::size_t members) {
    return graph.space_registry().register_space(IndexSpace{.name           = fmt::format("grouped-members-{}", members),
                                                            .scale_symbol   = fmt::format("gm{}", members),
                                                            .dim_symbol     = fmt::format("ngm{}", members),
                                                            .typical_extent = static_cast<double>(members),
                                                            .growth         = GrowthClass::linear()});
}

/// One leaf per MEMBER LIST, so two statements naming the same ordered list
/// read one value. Identity is the list rather than any one member's buffer,
/// which is the whole of what makes a rewrite able to see that a grouped
/// statement consumes what another grouped statement produced.
TermId ragged_leaf_for(TensorExpr &expr, std::map<std::vector<TensorId>, TermId> &cache, Graph const &graph,
                       std::vector<TensorId> const &members, std::vector<ExprIndex> indices) {
    if (auto const hit = cache.find(members); hit != cache.end()) {
        return hit->second;
    }
    ExprTerm leaf;
    leaf.kind         = TermKind::Leaf;
    leaf.name         = fmt::format("{{{} x{}}}", name_of(graph, members.front()), members.size());
    leaf.members      = members;
    leaf.indices      = std::move(indices);
    TermId const term = expr.add(std::move(leaf));
    cache.emplace(members, term);
    return term;
}

/// An index over @p space, spelled @p letter.
ExprIndex indexed(std::string letter, SpaceId space) {
    ExprIndex out;
    out.letter = std::move(letter);
    out.space  = space;
    return out;
}

/// Record one letter's per-member extents on @p family, appending when the
/// letter is new and leaving the first statement of it alone when it is not.
void note_extent(RaggedFamily &family, std::string const &letter, std::size_t member, std::size_t value) {
    for (auto &[name, values] : family.extents) {
        if (name == letter) {
            if (values.size() <= member) {
                values.resize(member + 1, 0);
            }
            values[member] = value;
            return;
        }
    }
    std::vector<std::size_t> values(member + 1, 0);
    values[member] = value;
    family.extents.emplace_back(letter, std::move(values));
}

/// The variable a raised grouped letter contributes to a cost polynomial.
///
/// The member letter is the family's registered space, so a dump names it and
/// two families over one member count compare by scale order. Every other
/// letter is ragged and therefore ANONYMOUS: it has one extent per member and
/// no space says which, so the number that prices it is the typical extent a
/// client feeds the comparison as a bound extent, below the scale rung, which
/// is where a bound extent for an ordinary letter already sits.
SymbolicVar grouped_variable(RaggedFamily const &family, std::string const &letter) {
    return letter == family.letter ? SymbolicVar::space(family.space) : SymbolicVar::anonymous(letter);
}

/// The product of the distinct letters of @p indices, coefficient one.
SymbolicPoly grouped_poly(RaggedFamily const &family, std::vector<ExprIndex> const &indices) {
    SymbolicPoly          poly = SymbolicPoly::constant(1.0);
    std::set<std::string> seen;
    for (auto const &index : indices) {
        if (seen.insert(index.letter).second) {
            poly *= SymbolicPoly::variable(grouped_variable(family, index.letter));
        }
    }
    return poly;
}

/// The cost of one grouped contraction, in the conventions @ref symbolic_cost_for
/// uses: flops is twice the loop space, traffic is the three operand sizes, and
/// resident equals traffic. Deliberately the same conventions rather than a
/// second opinion, so a region's before-and-after prices both sides one way.
SymbolicCost grouped_cost(RaggedFamily const &family, std::vector<ExprIndex> const &a, std::vector<ExprIndex> const &b,
                          std::vector<ExprIndex> const &c) {
    std::vector<ExprIndex> loop = a;
    loop.insert(loop.end(), b.begin(), b.end());

    SymbolicCost cost;
    cost.flops    = grouped_poly(family, loop) * 2.0;
    cost.traffic  = grouped_poly(family, c) + grouped_poly(family, a) + grouped_poly(family, b);
    cost.resident = cost.traffic;
    return cost;
}

RaiseFailure grouped_refusal(Node const &node, std::string reason, std::string detail = {}) {
    return RaiseFailure{.reason = std::move(reason),
                        .detail = detail.empty() ? fmt::format("node '{}'", node.label) : fmt::format("node '{}': {}", node.label, detail)};
}

/// The per-member source lists of a grouped element-wise node, recovered from
/// the node's flat input list and the per-member destination prefactors.
///
/// The capture sites append the destination to the inputs only for a member
/// whose beta is non-zero, so the list is variable-length per member and the
/// betas are what say where each member's sources start.
expected<std::vector<std::vector<TensorId>>, std::string> grouped_sources(Node const &node, std::vector<PrefactorScalar> const &betas,
                                                                          std::size_t sources_per_member) {
    std::vector<std::vector<TensorId>> out(sources_per_member);
    for (auto &list : out) {
        list.reserve(betas.size());
    }
    std::size_t cursor = 0;
    for (std::size_t member = 0; member < betas.size(); ++member) {
        for (std::size_t slot = 0; slot < sources_per_member; ++slot) {
            if (cursor >= node.inputs.size()) {
                return unexpected(std::string{"the input list is shorter than the member count says"});
            }
            out[slot].push_back(node.inputs[cursor++]);
        }
        if (!is_zero(betas[member])) {
            if (cursor >= node.inputs.size() || node.inputs[cursor] != node.outputs[member]) {
                return unexpected(fmt::format("member {} accumulates but does not read its own destination", member));
            }
            ++cursor;
        }
    }
    if (cursor != node.inputs.size()) {
        return unexpected(std::string{"the input list holds more operands than the members account for"});
    }
    return out;
}

/// Raise one grouped node into a statement over a member letter.
///
/// Every arm declines rather than approximating, and the reasons are
/// shape-independent so the skip tally folds them into one counted line. A
/// family whose members disagree on a shape contract beyond their extents - a
/// transpose flag, a prefactor, a rank - is left whole with that reason,
/// because the algebra has one term per family and a term cannot say that one
/// member transposes and another does not.
expected<ExprStatement, RaiseFailure> raise_grouped(Graph const &graph, Node const &node, TensorExpr &expr,
                                                    std::map<std::vector<TensorId>, TermId> &ragged) {
    auto const family_id = static_cast<FamilyId>(expr.families.size());

    RaggedFamily family;
    family.kind       = node.kind;
    family.descriptor = node.op_data;
    family.letter     = member_letter(family_id);

    ExprStatement statement;
    statement.origin       = node.id;
    statement.origin_kind  = node.kind;
    statement.origin_label = node.label;
    statement.family       = family_id;
    statement.targets      = node.outputs;

    if (node.kind == OpKind::GroupedBatchedGemm) {
        auto const *desc = node.op_data.get_if<GroupedBatchedGemmDescriptor>();
        if (desc == nullptr || desc->groups.empty()) {
            return unexpected(grouped_refusal(node, "a grouped batch carries no group table"));
        }
        auto const count = static_cast<std::size_t>(desc->total);
        if (node.outputs.size() != count || node.inputs.size() < 2 * count) {
            return unexpected(grouped_refusal(node, "a grouped batch's operand lists do not match its member count"));
        }
        // One transpose pair and one prefactor pair for the whole call is what
        // the capture API records; a group table that disagrees came from
        // somewhere else and the algebra has no way to say so.
        auto const &first = desc->groups.front();
        for (auto const &group : desc->groups) {
            if (group.trans_a != first.trans_a || group.trans_b != first.trans_b) {
                return unexpected(grouped_refusal(node, "a grouped family's members disagree on a shape contract",
                                                  "the groups carry different transpose flags"));
            }
            if (group.alpha != first.alpha || group.beta != first.beta) {
                return unexpected(grouped_refusal(node, "a grouped family's members disagree on a shape contract",
                                                  "the groups carry different prefactors"));
            }
        }
        auto const trans = [](char flag) { return flag == 'T' || flag == 't' || flag == 'C' || flag == 'c'; };
        auto const conj  = [](char flag) { return flag == 'C' || flag == 'c'; };

        family.members      = count;
        family.space        = family_space(graph, count);
        SpaceId const space = family.space;

        std::string const mem = family.letter;
        std::string const li  = family_letter(family_id, "m");
        std::string const lk  = family_letter(family_id, "k");
        std::string const lj  = family_letter(family_id, "n");

        std::vector<TensorId> a_members, b_members;
        a_members.reserve(count);
        b_members.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            a_members.push_back(node.inputs[2 * i]);
            b_members.push_back(node.inputs[2 * i + 1]);
        }
        // Every group's members are contiguous in the flattened lists, so the
        // shape a member runs at is its group's.
        for (auto const &group : desc->groups) {
            for (int slot = 0; slot < group.count; ++slot) {
                auto const member = static_cast<std::size_t>(group.first) + static_cast<std::size_t>(slot);
                note_extent(family, li, member, static_cast<std::size_t>(group.m));
                note_extent(family, lj, member, static_cast<std::size_t>(group.n));
                note_extent(family, lk, member, static_cast<std::size_t>(group.k));
            }
        }

        std::vector<ExprIndex> a_idx{indexed(mem, space)};
        std::vector<ExprIndex> b_idx{indexed(mem, space)};
        std::vector<ExprIndex> c_idx{indexed(mem, space), indexed(li, SpaceId{}), indexed(lj, SpaceId{})};
        if (trans(first.trans_a)) {
            a_idx.push_back(indexed(lk, SpaceId{}));
            a_idx.push_back(indexed(li, SpaceId{}));
        } else {
            a_idx.push_back(indexed(li, SpaceId{}));
            a_idx.push_back(indexed(lk, SpaceId{}));
        }
        if (trans(first.trans_b)) {
            b_idx.push_back(indexed(lj, SpaceId{}));
            b_idx.push_back(indexed(lk, SpaceId{}));
        } else {
            b_idx.push_back(indexed(lk, SpaceId{}));
            b_idx.push_back(indexed(lj, SpaceId{}));
        }

        ExprTerm term;
        term.kind    = TermKind::Contraction;
        term.family  = family_id;
        term.factor  = PrefactorScalar{first.alpha};
        term.indices = c_idx;
        term.operands.push_back(ragged_leaf_for(expr, ragged, graph, a_members, a_idx));
        term.operands.push_back(ragged_leaf_for(expr, ragged, graph, b_members, b_idx));
        term.operand_indices.push_back(a_idx);
        term.operand_indices.push_back(b_idx);
        term.conjugate.push_back(conj(first.trans_a));
        term.conjugate.push_back(conj(first.trans_b));
        term.cost = grouped_cost(family, a_idx, b_idx, c_idx);

        statement.target_indices   = std::move(c_idx);
        statement.target_prefactor = PrefactorScalar{first.beta};
        statement.target_name      = fmt::format("{{{} x{}}}", name_of(graph, node.outputs.front()), count);
        statement.value            = expr.add(std::move(term));
        expr.families.push_back(std::move(family));
        return statement;
    }

    if (node.kind == OpKind::GroupedDot) {
        auto const *desc = node.op_data.get_if<GroupedDotDescriptor>();
        if (desc == nullptr) {
            return unexpected(grouped_refusal(node, "a grouped reduction carries no descriptor"));
        }
        auto const count = static_cast<std::size_t>(desc->total);
        if (node.outputs.size() != count || node.inputs.size() != 2 * count) {
            return unexpected(grouped_refusal(node, "a grouped reduction's operand lists do not match its member count"));
        }
        std::vector<TensorId> a_members, b_members;
        a_members.reserve(count);
        b_members.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            a_members.push_back(node.inputs[2 * i]);
            b_members.push_back(node.inputs[2 * i + 1]);
        }
        auto const zeroth = member_shape(graph, a_members.front());
        if (!zeroth.ok || zeroth.dims.empty()) {
            return unexpected(grouped_refusal(node, "a grouped family's members have no readable shape"));
        }
        std::size_t const rank = zeroth.dims.size();

        family.members      = count;
        family.space        = family_space(graph, count);
        SpaceId const space = family.space;

        std::vector<ExprIndex> operand_idx{indexed(family.letter, space)};
        for (std::size_t axis = 0; axis < rank; ++axis) {
            operand_idx.push_back(indexed(family_letter(family_id, fmt::format("a{}", axis)), SpaceId{}));
        }
        for (std::size_t i = 0; i < count; ++i) {
            auto const shape = member_shape(graph, a_members[i]);
            if (!shape.ok || shape.dims.size() != rank) {
                return unexpected(grouped_refusal(node, "a grouped family's members disagree on a shape contract",
                                                  fmt::format("member {} has a different rank", i)));
            }
            for (std::size_t axis = 0; axis < rank; ++axis) {
                note_extent(family, operand_idx[axis + 1].letter, i, shape.dims[axis]);
            }
        }

        ExprTerm term;
        term.kind    = TermKind::Contraction;
        term.family  = family_id;
        term.indices = {indexed(family.letter, space)};
        term.operands.push_back(ragged_leaf_for(expr, ragged, graph, a_members, operand_idx));
        term.operands.push_back(ragged_leaf_for(expr, ragged, graph, b_members, operand_idx));
        term.operand_indices.push_back(operand_idx);
        term.operand_indices.push_back(operand_idx);
        term.conjugate.assign({false, false});
        term.cost = grouped_cost(family, operand_idx, operand_idx, term.indices);

        statement.target_indices   = {indexed(family.letter, space)};
        statement.target_prefactor = PrefactorScalar{double{0}};
        statement.target_name      = fmt::format("{{{} x{}}}", name_of(graph, node.outputs.front()), count);
        statement.value            = expr.add(std::move(term));
        expr.families.push_back(std::move(family));
        return statement;
    }

    // The grouped element-wise kinds. Named and carried, exactly as their dense
    // counterparts are: a rewrite may move one of these or delete it, and the
    // per-member prefactors it holds are not a rewrite surface.
    std::vector<PrefactorScalar> betas;
    std::size_t                  sources = 0;
    if (node.kind == OpKind::GroupedAxpby) {
        auto const *desc = node.op_data.get_if<GroupedAxpbyDescriptor>();
        if (desc == nullptr) {
            return unexpected(grouped_refusal(node, "a grouped accumulation carries no descriptor"));
        }
        betas   = desc->betas;
        sources = 1;
    } else {
        auto const *desc = node.op_data.get_if<GroupedElementwiseDescriptor>();
        if (desc == nullptr) {
            return unexpected(grouped_refusal(node, "a grouped element-wise node carries no descriptor"));
        }
        if (node.kind == OpKind::GroupedPermute && desc->c_indices.empty()) {
            return unexpected(grouped_refusal(node, "a grouped permute carries no index lists"));
        }
        betas   = desc->betas;
        sources = node.kind == OpKind::GroupedPermute ? 1 : 2;
    }
    auto const count = betas.size();
    if (count == 0 || node.outputs.size() != count) {
        return unexpected(grouped_refusal(node, "a grouped family's operand lists do not match its member count"));
    }
    auto const lists = grouped_sources(node, betas, sources);
    if (!lists) {
        return unexpected(grouped_refusal(node, "a grouped family's operand lists do not match its member count", lists.error()));
    }

    auto const zeroth = member_shape(graph, node.outputs.front());
    if (!zeroth.ok) {
        return unexpected(grouped_refusal(node, "a grouped family's members have no readable shape"));
    }
    std::size_t const rank = zeroth.dims.size();

    family.members      = count;
    family.space        = family_space(graph, count);
    SpaceId const space = family.space;

    // A permute's own letters, prefixed so two families cannot collide on one.
    // Everything else names its axes positionally, which is what a dense
    // element-wise term already does.
    auto const *elementwise = node.op_data.get_if<GroupedElementwiseDescriptor>();
    auto const  axis_name   = [&](std::size_t axis, bool source) {
        if (node.kind == OpKind::GroupedPermute && elementwise != nullptr) {
            auto const &names = source ? elementwise->a_indices : elementwise->c_indices;
            if (axis < names.size()) {
                return family_letter(family_id, names[axis]);
            }
        }
        return family_letter(family_id, fmt::format("a{}", axis));
    };

    std::vector<ExprIndex> out_idx{indexed(family.letter, space)};
    std::vector<ExprIndex> src_idx{indexed(family.letter, space)};
    for (std::size_t axis = 0; axis < rank; ++axis) {
        out_idx.push_back(indexed(axis_name(axis, false), SpaceId{}));
    }
    std::size_t const source_rank = node.kind == OpKind::GroupedPermute && elementwise != nullptr ? elementwise->a_indices.size() : rank;
    for (std::size_t axis = 0; axis < source_rank; ++axis) {
        src_idx.push_back(indexed(axis_name(axis, true), SpaceId{}));
    }

    for (std::size_t i = 0; i < count; ++i) {
        auto const shape = member_shape(graph, node.outputs[i]);
        if (!shape.ok || shape.dims.size() != rank) {
            return unexpected(grouped_refusal(node, "a grouped family's members disagree on a shape contract",
                                              fmt::format("member {} has a different rank", i)));
        }
        for (std::size_t axis = 0; axis < rank; ++axis) {
            note_extent(family, out_idx[axis + 1].letter, i, shape.dims[axis]);
        }
        auto const source = member_shape(graph, lists->front()[i]);
        if (!source.ok || source.dims.size() != source_rank) {
            return unexpected(grouped_refusal(node, "a grouped family's members disagree on a shape contract",
                                              fmt::format("member {}'s source has a different rank", i)));
        }
        for (std::size_t axis = 0; axis < source_rank; ++axis) {
            note_extent(family, src_idx[axis + 1].letter, i, source.dims[axis]);
        }
    }

    ExprTerm term;
    term.kind         = TermKind::Elementwise;
    term.family       = family_id;
    term.element_kind = node.kind;
    term.descriptor   = node.op_data;
    term.indices      = out_idx;
    for (auto const &list : *lists) {
        term.operands.push_back(ragged_leaf_for(expr, ragged, graph, list, src_idx));
        term.operand_indices.push_back(&list == &lists->front() ? src_idx : out_idx);
    }

    statement.target_indices = std::move(out_idx);
    // One destination prefactor for a family whose members each have their own.
    // The descriptor is what the executor reads and what a rewrite must not
    // reinterpret; the statement's own prefactor exists to say whether the
    // value READS its destination, which is the question the hazard edges and
    // the escape rule ask, and any accumulating member makes the answer yes.
    statement.target_prefactor = std::ranges::any_of(betas, [](PrefactorScalar const &beta) { return !is_zero(beta); })
                                     ? PrefactorScalar{double{1}}
                                     : PrefactorScalar{double{0}};
    statement.target_name      = fmt::format("{{{} x{}}}", name_of(graph, node.outputs.front()), count);
    statement.value            = expr.add(std::move(term));
    expr.families.push_back(std::move(family));
    return statement;
}

} // namespace

expected<TensorExpr, RaiseFailure> raise_region(Graph const &graph, Region const &region) {
    TensorExpr                              expr;
    std::unordered_map<TensorId, TermId>    leaves;
    std::map<std::vector<TensorId>, TermId> ragged;

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

        if (is_grouped_raisable(node->kind)) {
            auto grouped = raise_grouped(graph, *node, expr, ragged);
            if (!grouped) {
                return unexpected(std::move(grouped.error()));
            }
            expr.statements.push_back(std::move(*grouped));
            continue;
        }

        ExprStatement statement;
        statement.target       = node->outputs[0];
        statement.target_name  = name_of(graph, statement.target);
        statement.origin       = node->id;
        statement.origin_kind  = node->kind;
        statement.origin_label = node->label;

        if (node->kind == OpKind::Einsum) {
            auto const *desc = node->op_data.get_if<EinsumDescriptor>();
            if (desc == nullptr) {
                return unexpected(
                    RaiseFailure{.reason = "a contraction carries no einsum descriptor", .detail = fmt::format("node '{}'", node->label)});
            }
            // The algebra has one element type, and lowering rebuilds every node from the
            // destination's.
            if (!passes::einsum_is_uniform(graph, *node)) {
                return unexpected(RaiseFailure{.reason = "a contraction's operands hold different element types",
                                               .detail = fmt::format("node '{}'", node->label)});
            }
            // The LIVE index lists, not the capture snapshot: raising the snapshot
            // would raise the algebra as it was before a pass rewrote it.
            auto const                      lists  = live_index_lists(*desc);
            std::vector<std::string> const &a_list = lists.a;
            std::vector<std::string> const &b_list = lists.b;
            std::vector<std::string> const &c_list = lists.c;
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
            term.factor  = live_ab_prefactor(*desc);
            term.indices = indices_from(c_list, *desc);
            term.operands.push_back(leaf_for(expr, leaves, graph, node->inputs[0]));
            term.operands.push_back(leaf_for(expr, leaves, graph, node->inputs[1]));
            term.operand_indices.push_back(indices_from(a_list, *desc));
            term.operand_indices.push_back(indices_from(b_list, *desc));
            term.conjugate.push_back(live_conj_a(*desc));
            term.conjugate.push_back(live_conj_b(*desc));
            term.cost = symbolic_cost_for(*desc);

            statement.target_prefactor = live_c_prefactor(*desc);
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
            if (auto const *element = node->op_data.get_if<ElementTransformDescriptor>(); element != nullptr) {
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

/// What makes two members of a grouped batch belong to one uniform group.
///
/// The same six numbers the capture API groups on, restated here rather than
/// shared with it: the capture site's copy lives in a header this source has no
/// reason to include, and the tuple is the BLAS call's own signature rather
/// than either site's invention.
struct GemmShapeKey {
    int m, n, k, lda, ldb, ldc;

    auto operator<=>(GemmShapeKey const &) const = default;
};

RaiseFailure lower_refusal(ExprStatement const &statement, std::string reason, std::string detail = {}) {
    return RaiseFailure{.reason = std::move(reason),
                        .detail = detail.empty() ? statement.target_name : fmt::format("'{}': {}", statement.target_name, detail)};
}

/// Rebuild a grouped batched GEMM from the algebra.
///
/// Every shape parameter is re-derived: the transpose flags from the index
/// lists, the extents and leading dimensions from the live operands, and the
/// grouping by first appearance the way the capture API groups it. Nothing is
/// carried over from the descriptor the raise saw, which is what makes a
/// rewrite that changed a member's extents emit a group table that describes
/// what it actually wrote rather than what the capture happened to hold.
expected<Node, RaiseFailure> lower_grouped_gemm(Graph &graph, TensorExpr const &expr, ExprStatement const &statement) {
    auto const &term  = expr.at(statement.value);
    auto const  count = statement.targets.size();
    if (expr.at(term.operands[0]).members.size() != count || expr.at(term.operands[1]).members.size() != count) {
        return unexpected(lower_refusal(statement, "a grouped term's operand lists disagree on their member count"));
    }
    if (term.operand_indices.size() != 2 || term.operand_indices[0].size() != 3 || term.operand_indices[1].size() != 3) {
        return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind",
                                        "a batched matrix product wants two rank-two operands"));
    }

    std::string const &mem = statement.target_indices[0].letter;
    std::string const &row = statement.target_indices[1].letter;
    std::string const &col = statement.target_indices[2].letter;
    // A contraction's operands are unordered and a GEMM's are not: the first is
    // what contributes the destination's ROWS. A rewrite hands them over in
    // whatever order its search settled on, so the roles are read off the
    // letters here rather than assumed from the positions. Reversing the pair
    // is the whole of what that costs, since C = A B and C = B A over the same
    // letters are the same arithmetic read from the other side.
    bool const  flip   = term.operand_indices[0][1].letter != row && term.operand_indices[0][2].letter != row;
    auto const &a      = term.operand_indices[flip ? 1 : 0];
    auto const &b      = term.operand_indices[flip ? 0 : 1];
    auto const &a_leaf = expr.at(term.operands[flip ? 1 : 0]);
    auto const &b_leaf = expr.at(term.operands[flip ? 0 : 1]);
    if (a[0].letter != mem || b[0].letter != mem) {
        return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind",
                                        "the member letter is not outermost on every operand"));
    }
    // The link letter is the one both operands carry and the output does not.
    std::string link;
    for (std::size_t slot = 1; slot < 3; ++slot) {
        if (a[slot].letter != row && a[slot].letter != col) {
            link = a[slot].letter;
        }
    }
    if (link.empty() || (a[1].letter != row && a[2].letter != row)) {
        return unexpected(
            lower_refusal(statement, "a grouped term's shape maps onto no grouped kind", "the operands do not spell a matrix product"));
    }
    if ((b[1].letter != link || b[2].letter != col) && (b[1].letter != col || b[2].letter != link)) {
        return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind",
                                        "the second operand does not carry the link and the column"));
    }
    bool const trans_a = a[1].letter == link;
    bool const trans_b = b[1].letter == col;
    bool const conj_a  = term.conjugate.size() > (flip ? 1U : 0U) && term.conjugate[flip ? 1 : 0];
    bool const conj_b  = term.conjugate.size() > (flip ? 0U : 1U) && term.conjugate[flip ? 0 : 1];

    // Shape keys read off the live operands, then grouped by first appearance.
    // Applied to a list the capture already flattened this is the identity,
    // which is what keeps an unchanged region's node identical to the one it
    // was raised from.
    std::vector<GemmShapeKey> keys(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto const a_shape = member_shape(graph, a_leaf.members[i]);
        auto const b_shape = member_shape(graph, b_leaf.members[i]);
        auto const c_shape = member_shape(graph, statement.targets[i]);
        if (!a_shape.ok || !b_shape.ok || !c_shape.ok || a_shape.dims.size() != 2 || b_shape.dims.size() != 2 || c_shape.dims.size() != 2) {
            return unexpected(lower_refusal(statement, "a grouped batch's members are not rank-two matrices", fmt::format("member {}", i)));
        }
        keys[i] = GemmShapeKey{.m   = static_cast<int>(c_shape.dims[0]),
                               .n   = static_cast<int>(c_shape.dims[1]),
                               .k   = static_cast<int>(trans_a ? a_shape.dims[0] : a_shape.dims[1]),
                               .lda = static_cast<int>(a_shape.leading),
                               .ldb = static_cast<int>(b_shape.leading),
                               .ldc = static_cast<int>(c_shape.leading)};
    }

    auto const [dtype, rank] = destination_key(graph, statement.targets.front());
    auto const alpha         = as<std::complex<double>>(term.factor);
    auto const beta          = as<std::complex<double>>(statement.target_prefactor);

    std::map<GemmShapeKey, std::size_t>   index_of;
    std::vector<GemmShapeKey>             order;
    std::vector<std::vector<std::size_t>> grouped;
    for (std::size_t i = 0; i < count; ++i) {
        auto const [it, fresh] = index_of.try_emplace(keys[i], order.size());
        if (fresh) {
            order.push_back(keys[i]);
            grouped.emplace_back();
        }
        grouped[it->second].push_back(i);
    }

    GroupedBatchedGemmDescriptor desc;
    desc.total  = static_cast<int>(count);
    desc.scalar = detail::blas_scalar_from(dtype);
    desc.groups.reserve(order.size());
    desc.labels.reserve(order.size());
    char const ta    = conj_a ? 'C' : (trans_a ? 'T' : 'N');
    char const tb    = conj_b ? 'C' : (trans_b ? 'T' : 'N');
    int        first = 0;
    for (std::size_t g = 0; g < order.size(); ++g) {
        auto const &key = order[g];
        desc.groups.push_back(GemmGroup{.m       = key.m,
                                        .n       = key.n,
                                        .k       = key.k,
                                        .lda     = key.lda,
                                        .ldb     = key.ldb,
                                        .ldc     = key.ldc,
                                        .trans_a = ta,
                                        .trans_b = tb,
                                        .alpha   = alpha,
                                        .beta    = beta,
                                        .count   = static_cast<int>(grouped[g].size()),
                                        .first   = first});
        desc.labels.push_back(fmt::format("gemm {}x{}x{} trans={}{} x{}", key.m, key.k, key.n, ta, tb, grouped[g].size()));
        first += static_cast<int>(grouped[g].size());
    }

    Node node;
    node.id    = graph.reserve_node_id();
    node.kind  = OpKind::GroupedBatchedGemm;
    node.label = statement.origin_label.empty()
                     ? fmt::format("gemm_batch_grouped x{} in {} shapes (trans={}{})", count, order.size(), ta, tb)
                     : statement.origin_label;
    node.inputs.reserve(3 * count);
    node.outputs.reserve(count);
    for (auto const &members : grouped) {
        for (auto const member : members) {
            node.inputs.push_back(a_leaf.members[member]);
            node.inputs.push_back(b_leaf.members[member]);
            node.outputs.push_back(statement.targets[member]);
        }
    }
    // A non-zero destination prefactor reads every destination before writing
    // it, so the RAW edge from whoever produced each one has to survive.
    if (!is_zero(statement.target_prefactor)) {
        node.inputs.insert(node.inputs.end(), node.outputs.begin(), node.outputs.end());
    }
    node.op_data = std::move(desc);
    try {
        node.execute = build_executor(node.kind, dtype, rank, node.op_data, graph, std::span<TensorId const>{node.inputs},
                                      std::span<TensorId const>{node.outputs});
    } catch (std::exception const &error) {
        return unexpected(lower_refusal(statement, "a grouped batch could not be rebuilt from the algebra", error.what()));
    }
    return node;
}

/// Rebuild a grouped reduction from the algebra: one contraction per member
/// summed over every letter but the member one.
expected<Node, RaiseFailure> lower_grouped_dot(Graph &graph, TensorExpr const &expr, ExprStatement const &statement) {
    auto const &term   = expr.at(statement.value);
    auto const &a_leaf = expr.at(term.operands[0]);
    auto const &b_leaf = expr.at(term.operands[1]);
    auto const  count  = statement.targets.size();
    if (a_leaf.members.size() != count || b_leaf.members.size() != count) {
        return unexpected(lower_refusal(statement, "a grouped term's operand lists disagree on their member count"));
    }
    if (term.operand_indices.size() != 2 || term.operand_indices[0] != term.operand_indices[1]) {
        return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind",
                                        "a batched reduction wants two operands over one index list"));
    }
    if (term.operand_indices[0].empty() || term.operand_indices[0][0].letter != statement.target_indices[0].letter) {
        return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind",
                                        "the member letter is not outermost on every operand"));
    }
    if (!is_zero(statement.target_prefactor)) {
        return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind",
                                        "a batched reduction writes its destinations and cannot accumulate"));
    }

    GroupedDotDescriptor desc;
    desc.total = static_cast<int>(count);

    Node node;
    node.id    = graph.reserve_node_id();
    node.kind  = OpKind::GroupedDot;
    node.label = statement.origin_label.empty() ? fmt::format("dot x{}", count) : statement.origin_label;
    node.inputs.reserve(2 * count);
    node.outputs = statement.targets;
    for (std::size_t i = 0; i < count; ++i) {
        node.inputs.push_back(a_leaf.members[i]);
        node.inputs.push_back(b_leaf.members[i]);
    }
    node.op_data             = desc;
    auto const [dtype, rank] = destination_key(graph, statement.targets.front());
    try {
        node.execute = build_executor(node.kind, dtype, rank, node.op_data, graph, std::span<TensorId const>{node.inputs},
                                      std::span<TensorId const>{node.outputs});
    } catch (std::exception const &error) {
        return unexpected(lower_refusal(statement, "a grouped reduction could not be rebuilt from the algebra", error.what()));
    }
    return node;
}

/// Rebuild a grouped element-wise node from the algebra. The descriptor is
/// carried rather than re-derived, exactly as a dense element-wise term's is;
/// what the algebra rebuilds is the operand list, from the member lists and the
/// per-member destination prefactors.
expected<Node, RaiseFailure> lower_grouped_elementwise(Graph &graph, TensorExpr const &expr, ExprStatement const &statement) {
    auto const &term  = expr.at(statement.value);
    auto const  count = statement.targets.size();

    std::vector<PrefactorScalar> betas;
    if (term.element_kind == OpKind::GroupedAxpby) {
        auto const *desc = term.descriptor.get_if<GroupedAxpbyDescriptor>();
        if (desc == nullptr || desc->betas.size() != count) {
            return unexpected(lower_refusal(statement, "a grouped accumulation's descriptor does not match its member count"));
        }
        betas = desc->betas;
    } else {
        auto const *desc = term.descriptor.get_if<GroupedElementwiseDescriptor>();
        if (desc == nullptr || desc->betas.size() != count) {
            return unexpected(lower_refusal(statement, "a grouped element-wise node's descriptor does not match its member count"));
        }
        betas = desc->betas;
    }
    for (auto const operand : term.operands) {
        if (expr.at(operand).members.size() != count) {
            return unexpected(lower_refusal(statement, "a grouped term's operand lists disagree on their member count"));
        }
    }

    Node node;
    node.id      = graph.reserve_node_id();
    node.kind    = term.element_kind;
    node.label   = statement.origin_label.empty() ? fmt::format("{} x{}", term.element_kind, count) : statement.origin_label;
    node.outputs = statement.targets;
    node.inputs.reserve((term.operands.size() + 1) * count);
    for (std::size_t i = 0; i < count; ++i) {
        for (auto const operand : term.operands) {
            node.inputs.push_back(expr.at(operand).members[i]);
        }
        if (!is_zero(betas[i])) {
            node.inputs.push_back(statement.targets[i]);
        }
    }
    node.op_data             = term.descriptor;
    auto const [dtype, rank] = destination_key(graph, statement.targets.front());
    try {
        node.execute = build_executor(node.kind, dtype, rank, node.op_data, graph, std::span<TensorId const>{node.inputs},
                                      std::span<TensorId const>{node.outputs});
    } catch (std::exception const &error) {
        return unexpected(lower_refusal(statement, "a grouped element-wise node could not be rebuilt from the algebra", error.what()));
    }
    return node;
}

/// The grouped kind a statement carrying a member letter lowers onto.
///
/// A shape that maps onto none of them is DECLINED. It is never emitted as a
/// per-member loop of ordinary nodes: that loop is what the grouped family
/// exists to avoid, and a rewrite that quietly produced one would trade a
/// factor of nearly three in dispatch for whatever the rewrite saved.
expected<Node, RaiseFailure> lower_grouped(Graph &graph, TensorExpr const &expr, ExprStatement const &statement) {
    auto const &term = expr.at(statement.value);
    if (statement.targets.empty()) {
        return unexpected(lower_refusal(statement, "a grouped statement names no destinations"));
    }
    if (term.kind == TermKind::Contraction) {
        if (term.operands.size() != 2 || !expr.at(term.operands[0]).ragged() || !expr.at(term.operands[1]).ragged()) {
            return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind",
                                            "a grouped contraction wants exactly two ragged operands"));
        }
        if (statement.target_indices.size() == 3) {
            return lower_grouped_gemm(graph, expr, statement);
        }
        if (statement.target_indices.size() == 1) {
            return lower_grouped_dot(graph, expr, statement);
        }
        return unexpected(lower_refusal(
            statement, "a grouped term's shape maps onto no grouped kind",
            fmt::format("a batched output of rank {} is neither a matrix nor a scalar", statement.target_indices.size() - 1)));
    }
    if (term.kind == TermKind::Elementwise && is_grouped_raisable(term.element_kind)) {
        return lower_grouped_elementwise(graph, expr, statement);
    }
    return unexpected(lower_refusal(statement, "a grouped term's shape maps onto no grouped kind", fmt::format("a {} term", term.kind)));
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

        if (statement.family != invalid_family) {
            auto grouped = lower_grouped(graph, expr, statement);
            if (!grouped) {
                return unexpected(std::move(grouped.error()));
            }
            emitted.push_back(std::move(*grouped));
            continue;
        }

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
            spec.raw       = spec.render();

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
                                           .detail = fmt::format("target '{}' is a {}", statement.target_name, term.kind)});
        }

        Node node;
        node.id   = graph.reserve_node_id();
        node.kind = term.element_kind;
        node.label =
            statement.origin_label.empty() ? fmt::format("{}({})", term.element_kind, statement.target_name) : statement.origin_label;
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
