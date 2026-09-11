//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Write.cpp
/// @brief A graph, rendered as an ``einsums_graph_ir`` document.
///
/// One of the three passes described in `Common.hpp`. Renumbering, canonical
/// emission, and the refusals a save makes rather than writing something a
/// reader could only half rebuild. What travels and what deliberately does not
/// is the member walk, written down in `GraphIR.cpp`.
///
/// Everything here is file-local except @ref write_structure, which is the hash
/// domain `Graph::content_hash` digests, and @ref write_document, which is what
/// `save_graph` emits. `Frame`, the dense per-fragment numbering the whole pass
/// threads through itself, never leaves this translation unit.

#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/ABI.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Version.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::graph_ir)

namespace {

/// @ref Graph::manifest mutates (it links aliases and builds usage analysis) and
/// is documented as non-const "not by choice". Writing a graph is a read-only
/// operation from the caller's point of view, so the const is kept at the API
/// boundary and dropped exactly here, once, with the reason written down.
InterfaceManifest manifest_of(Graph const &graph) {
    return const_cast<Graph &>(graph).manifest(); // NOLINT(cppcoreguidelines-pro-type-const-cast)
}

/// The elements of @p range, each mapped through @p project, as a JSON array.
///
/// The writer had fourteen spelled-out copies of this loop, and a spelled-out
/// loop hides which of them meant to project the element and which meant to
/// pass it through.
template <typename Range, typename Project>
Value to_array(Range const &range, Project project) {
    Array out;
    for (auto const &item : range) {
        out.emplace_back(project(item));
    }
    return Value{std::move(out)};
}

/// The elements of @p range as a JSON array, converted by @ref Value's own
/// constructors. One of those takes a ``std::size_t``, so an extent or a dense
/// id needs no cast at the call site.
template <typename Range>
Value to_array(Range const &range) {
    return to_array(range, [](auto const &item) { return Value{item}; });
}

/// A typed scalar: a discriminated union by dtype NAME, with the real part
/// always present and the imaginary part present exactly for the complex arms.
///
/// Discriminated rather than bare, because a @ref PrefactorScalar's alternative
/// is part of what it means: a scale by `float{1}` and one by
/// `complex<double>{1,0}` reach different BLAS arms, and a schema that wrote
/// only the number would silently merge them.
Value write_prefactor(PrefactorScalar const &scalar) {
    Object out;
    out.set("dtype",
            Value{std::string(scalar_type_name(std::visit([](auto x) { return packed_gemm::get_scalar_type<decltype(x)>(); }, scalar)))});
    std::visit(
        [&out](auto x) {
            using T = decltype(x);
            if constexpr (std::is_arithmetic_v<T>) {
                out.set("re", number_or_tag(static_cast<double>(x)));
            } else {
                out.set("re", number_or_tag(static_cast<double>(x.real())));
                out.set("im", number_or_tag(static_cast<double>(x.imag())));
            }
        },
        scalar);
    return Value{std::move(out)};
}

/// @copydoc write_prefactor
Value write_complex(std::complex<double> const &value) {
    return write_prefactor(PrefactorScalar{value});
}

/// One frame's dense numbering: this graph's ids in first-mention order, plus
/// the enclosing frames' tensor identities so a body can name a parent buffer.
class Frame {
  public:
    Frame(Graph const &graph, Frame const *parent) : _graph(graph), _parent(parent) {}

    /// Fold one slot's other ids onto the id that represents it. See ManifestEntry::also: a
    /// name is a slot, and two capture identities over one buffer are one entry, so the file
    /// holds ONE tensor and every reference to either handle points at it. Without this the
    /// folded handle would be written as a tensor record of its own, under the same name, and
    /// the loaded graph would hold the two entries the manifest just refused.
    void set_canonical(std::unordered_map<TensorId, TensorId> canonical) { _canonical = std::move(canonical); }

    /// The dense id for @p id, minting one on first mention.
    std::size_t intern(TensorId id) {
        if (auto const folded = _canonical.find(id); folded != _canonical.end()) {
            id = folded->second;
        }
        auto const it = _dense.find(id);
        if (it != _dense.end()) {
            return it->second;
        }
        std::size_t const dense = _order.size();
        _dense.emplace(id, dense);
        _order.push_back(id);
        return dense;
    }

    [[nodiscard]] std::vector<TensorId> const &order() const noexcept { return _order; }
    [[nodiscard]] Graph const                 &graph() const noexcept { return _graph; }

    /// The enclosing frame's dense id for the buffer @p id names, if any frame
    /// above this one names it. Identity is the tensor OBJECT's address, which
    /// is what survives a body registering its own handle for a parent's buffer.
    [[nodiscard]] std::optional<std::size_t> outer_of(void const *tensor_ptr) const {
        for (Frame const *frame = _parent; frame != nullptr; frame = frame->_parent) {
            for (std::size_t dense = 0; dense < frame->_order.size(); ++dense) {
                TensorHandle const *handle = frame->_graph.find_tensor(frame->_order[dense]);
                if (handle != nullptr && handle->tensor_ptr == tensor_ptr) {
                    return dense;
                }
            }
            // A handle this frame FOLDED onto another one is not in its order any more, and a
            // body whose own view of the buffer is the folded identity would otherwise find no
            // match and be loaded as a buffer of its own, under the same name, which is the two
            // entries the fold exists to remove.
            TensorId const local = frame->_graph.find_tensor_id_by_ptr(tensor_ptr);
            if (local != 0) {
                if (auto const folded = frame->_canonical.find(local); folded != frame->_canonical.end()) {
                    if (auto const dense = frame->_dense.find(folded->second); dense != frame->_dense.end()) {
                        return dense->second;
                    }
                }
            }
        }
        return std::nullopt;
    }

  private:
    Graph const                              &_graph;
    Frame const                              *_parent;
    std::unordered_map<TensorId, std::size_t> _dense;
    std::vector<TensorId>                     _order;
    std::unordered_map<TensorId, TensorId>    _canonical;
};

/// Refuse, naming the node and the field, which is the shape every save refusal
/// takes: a user can only act on a message that says which node and which arm.
[[noreturn]] void refuse(Node const &node, std::string_view field, std::string_view why) {
    throw SaveRefusal(fmt::format("node {} ('{}', {}): {} {}", node.id, node.label, op_kind_name(node.kind), field, why));
}

Value write_bound_expr(BoundExpr const &expr, Node const &node, std::string_view field) {
    Object out;
    if (expr.is_const()) {
        out.set("const", Value{expr.const_value()});
        return Value{std::move(out)};
    }
    if (expr.is_param()) {
        out.set("param", Value{expr.param_name()});
        return Value{std::move(out)};
    }
    refuse(node, field,
           "is a BoundExpr callback, a std::function a file cannot hold; use a literal or a named parameter "
           "(cg::BoundExpr{\"n_occ\"}) instead");
}

Value write_pred_expr(Graph const &root, PredExpr const &pred, Node const &node, std::string_view field) {
    Object out;
    if (auto const *arm = std::get_if<PredExpr::Const>(&pred.storage())) {
        out.set("const", Value{arm->value});
        return Value{std::move(out)};
    }
    if (auto const *arm = std::get_if<PredExpr::Compare>(&pred.storage())) {
        Object compare;
        compare.set("lhs", write_bound_expr(arm->lhs, node, field));
        compare.set("op", Value{std::string(cmp_op_name(arm->op))});
        compare.set("rhs", write_bound_expr(arm->rhs, node, field));
        out.set("compare", Value{std::move(compare)});
        return Value{std::move(out)};
    }
    if (auto const *arm = std::get_if<PredExpr::Iteration>(&pred.storage())) {
        Object iteration;
        iteration.set("op", Value{std::string(cmp_op_name(arm->op))});
        iteration.set("rhs", write_bound_expr(arm->rhs, node, field));
        out.set("iteration", Value{std::move(iteration)});
        return Value{std::move(out)};
    }
    if (auto const *arm = std::get_if<PredExpr::FlagTest>(&pred.storage())) {
        std::string const name = root.gate_flag_name(arm->flags);
        if (name.empty()) {
            refuse(node, field,
                   "tests an UNNAMED gate-flag array; a shared_ptr has no identity that survives a restart, so name the "
                   "array with Graph::name_gate_flags before saving");
        }
        Object flag;
        flag.set("name", Value{name});
        flag.set("index", Value{arm->index});
        out.set("flag", Value{std::move(flag)});
        return Value{std::move(out)};
    }
    refuse(node, field,
           "is a PredExpr callback, a std::function a file cannot hold; a comparison over parameters, over the iteration "
           "index, or a named gate flag is the data-shaped spelling");
}

Value write_fragment(Graph const &graph, Graph const &root, Frame const *parent, std::string name);

/// Everything a node's descriptor holds, per kind. The coverage here is exactly
/// the reconstructible set: anything else was already refused by the
/// serializability report before the writer ran.
// NOLINTNEXTLINE(misc-no-recursion): control-flow descriptors hold fragments.
Value write_descriptor(Node const &node, Graph const &graph, Graph const &root, Frame &frame) {
    Object out;
    switch (node.kind) {
    case OpKind::Transpose:
        // Deliberately empty: a transpose is fully described by its kind, dtype,
        // rank and operands, and TraceDescriptor's note explains why the empty
        // object is still written rather than the key omitted.
        return Value{std::move(out)};
    case OpKind::Scale: {
        auto const &desc = std::get<ScaleDescriptor>(node.op_data);
        // The LIVE scalar when the node carries one, because that is what the
        // executor reads; the snapshot beside it can be a pass's stale copy.
        out.set("factor", write_prefactor(live_factor(desc)));
        return Value{std::move(out)};
    }
    case OpKind::Permute: {
        auto const &desc = std::get<PermuteDescriptor>(node.op_data);
        out.set("alpha", desc.params != nullptr ? write_prefactor(desc.params->alpha) : write_complex(desc.alpha));
        out.set("beta", desc.params != nullptr ? write_prefactor(desc.params->beta) : write_complex(desc.beta));
        out.set("c_indices", to_array(desc.c_indices));
        out.set("a_indices", to_array(desc.a_indices));
        return Value{std::move(out)};
    }
    case OpKind::Axpby: {
        auto const &desc = std::get<AxpbyDescriptor>(node.op_data);
        // The LIVE scalars, for the reason the Scale case above states.
        out.set("alpha", write_prefactor(live_alpha(desc)));
        out.set("beta", write_prefactor(live_beta(desc)));
        return Value{std::move(out)};
    }
    case OpKind::DirectProduct:
    case OpKind::DirectDivision: {
        auto const &desc = std::get<ElementwiseBinaryDescriptor>(node.op_data);
        // The LIVE scalars, for the reason the Scale case above states.
        out.set("alpha", write_prefactor(live_alpha(desc)));
        out.set("beta", write_prefactor(live_beta(desc)));
        return Value{std::move(out)};
    }
    case OpKind::Einsum: {
        auto const &desc = std::get<EinsumDescriptor>(node.op_data);
        // The LIVE index lists when the node shares them, for the same reason
        // the live scalars win above: the executor reads
        // ``indices->spec``, and the descriptor's own ContractionSpec beside it
        // is the at-capture snapshot a rewriting pass may have left behind.
        // ``ParsedEinsumSpec::raw`` is deliberately not written: it is a display
        // string the loader regenerates exactly as @ref build_executor does.
        bool const live = desc.indices != nullptr;
        out.set("c_indices", to_array(live ? desc.indices->spec.c_indices : desc.spec.c_indices));
        out.set("a_indices", to_array(live ? desc.indices->spec.a_indices : desc.spec.a_indices));
        out.set("b_indices", to_array(live ? desc.indices->spec.b_indices : desc.spec.b_indices));
        out.set("link_indices", to_array(live ? desc.indices->link_indices : desc.spec.link_indices));
        out.set("target_indices", to_array(desc.spec.target_indices));
        out.set("all_indices", to_array(desc.spec.all_indices));
        out.set("scalar_output", Value{desc.spec.scalar_output});
        out.set("conj_a", Value{live_conj_a(desc)});
        out.set("conj_b", Value{live_conj_b(desc)});
        out.set("c_prefactor", write_prefactor(live_c_prefactor(desc)));
        out.set("ab_prefactor", write_prefactor(live_ab_prefactor(desc)));

        out.set("letter_spaces", to_array(desc.letter_spaces, [&graph](auto const &pair) {
                    Object entry;
                    entry.set("letter", Value{pair.first});
                    entry.set("space", Value{graph.space_registry().space(pair.second).name});
                    return Value{std::move(entry)};
                }));

        if (desc.gemm_hint != nullptr) {
            // A hint is a PLANNING snapshot the GEMMBatching pass reads, and it
            // is written verbatim rather than re-derived: a loaded graph has no
            // geometry to derive one from until a bind, and a rebind of a LIVE
            // graph does not refresh it either, so restoring the recorded one
            // leaves a loaded graph in exactly the state a rebound captured one
            // is in. The batched executor re-derives every leading dimension
            // from the live impl at execute, so a stale one is never read.
            Object hint;
            hint.set("m", Value{static_cast<std::int64_t>(desc.gemm_hint->m)});
            hint.set("n", Value{static_cast<std::int64_t>(desc.gemm_hint->n)});
            hint.set("k", Value{static_cast<std::int64_t>(desc.gemm_hint->k)});
            hint.set("trans_a", Value{std::string(1, desc.gemm_hint->trans_a)});
            hint.set("trans_b", Value{std::string(1, desc.gemm_hint->trans_b)});
            auto const operand = [&frame](GemmOperand const &op) {
                Object entry;
                entry.set("id", Value{frame.intern(op.id)});
                entry.set("leading_dim", Value{static_cast<std::int64_t>(op.leading_dim)});
                return Value{std::move(entry)};
            };
            hint.set("a", operand(desc.gemm_hint->a));
            hint.set("b", operand(desc.gemm_hint->b));
            hint.set("c", operand(desc.gemm_hint->c));
            out.set("gemm_hint", Value{std::move(hint)});
        }
        return Value{std::move(out)};
    }
    case OpKind::Dot: {
        out.set("conjugated", Value{std::get<DotDescriptor>(node.op_data).conjugated});
        return Value{std::move(out)};
    }
    case OpKind::Trace:
        return Value{std::move(out)};
    case OpKind::Gemm: {
        auto const &desc = std::get<GemmDescriptor>(node.op_data);
        out.set("alpha", write_prefactor(desc.alpha));
        out.set("beta", write_prefactor(desc.beta));
        out.set("trans_a", Value{std::string(1, desc.trans_a)});
        out.set("trans_b", Value{std::string(1, desc.trans_b)});
        return Value{std::move(out)};
    }
    case OpKind::Syev: {
        // The LAPACK job, and nothing else. It is a template argument at the capture site, so
        // it is the one part of a syev a file has to carry; the operand roles, the triangle
        // read and the absence of a prefactor are all fixed by the operation.
        out.set("compute_eigenvectors", Value{std::get<SyevDescriptor>(node.op_data).compute_eigenvectors});
        return Value{std::move(out)};
    }
    case OpKind::ElementTransform: {
        auto const &desc = std::get<ElementTransformDescriptor>(node.op_data);
        out.set("op", Value{desc.op_name});
        // Written only when the capture site chose one. An absent key is not
        // "no parameter" but "the default this op documents", which is what
        // makes every file written before the key existed still mean what it
        // meant: the ops that grew a parameter kept their old behavior as that
        // default. See ElementOpSignature::default_param.
        if (desc.param.has_value()) {
            out.set("param", number_or_tag(*desc.param));
        }
        return Value{std::move(out)};
    }
    case OpKind::WriteParam: {
        auto const &desc = std::get<WriteParamDescriptor>(node.op_data);
        out.set("param", Value{desc.name});
        out.set("source_type", Value{std::string(param_source_type_name(desc.source_type))});
        if (desc.source_expr.has_value()) {
            out.set("source_expr", write_bound_expr(*desc.source_expr, node, "source_expr"));
        }
        return Value{std::move(out)};
    }
    case OpKind::Conditional: {
        auto const &desc = std::get<ConditionalDescriptor>(node.op_data);
        out.set("predicate", write_pred_expr(root, desc.predicate, node, "predicate"));
        if (desc.then_branch == nullptr) {
            refuse(node, "then_branch", "is null; a conditional without a then-branch has nothing to record");
        }
        out.set("then", write_fragment(*desc.then_branch, root, &frame, fmt::format("then({})", node.label)));
        out.set("else", desc.else_branch != nullptr ? write_fragment(*desc.else_branch, root, &frame, fmt::format("else({})", node.label))
                                                    : Value{nullptr});
        return Value{std::move(out)};
    }
    case OpKind::Loop: {
        auto const &desc = std::get<LoopDescriptor>(node.op_data);
        out.set("max_iterations", Value{desc.max_iterations});
        out.set("condition", write_pred_expr(root, desc.condition, node, "condition"));
        if (desc.body == nullptr) {
            refuse(node, "body", "is null; a loop without a body has nothing to record");
        }
        out.set("body", write_fragment(*desc.body, root, &frame, fmt::format("loop({})", node.label)));
        return Value{std::move(out)};
    }
    case OpKind::LaplaceQuadrature: {
        auto const &desc = std::get<LaplaceQuadratureDescriptor>(node.op_data);
        // All three, and all three REQUIRED on the way back in. A rule read at a different
        // tolerance, a different point count or a different sign per axis is a different
        // approximation, and none of the three has a default that could stand in for a
        // caller's choice.
        out.set("epsilon", number_or_tag(desc.epsilon));
        out.set("points", Value{desc.points});
        out.set("signs", to_array(desc.signs, [](std::int8_t sign) { return Value{static_cast<std::int64_t>(sign)}; }));
        return Value{std::move(out)};
    }
    case OpKind::Setup: {
        auto const &desc = std::get<SetupDescriptor>(node.op_data);
        if (desc.body == nullptr) {
            refuse(node, "body", "is null; a setup node without a body has nothing to record");
        }
        // The body and nothing else. SetupState is what a replay COMPUTED, which is a fact
        // about one process holding one bound problem, and a loaded graph has neither: it
        // arrives having fitted nothing, which is what a default-constructed state says.
        out.set("body", write_fragment(*desc.body, root, &frame, fmt::format("setup({})", node.label)));
        return Value{std::move(out)};
    }
    default:
        refuse(node, "kind", "has no descriptor encoding; the serializability report should have caught this first");
    }
}

/// A slot whose element type is not one of the four BLAS ones has no storage a
/// loader could allocate for it.
///
/// The case that reaches here is an INTEGRAL scalar: ``cg::write_param(name, n)``
/// over an ``int`` registers a rank-0 handle whose dtype is @c Unknown, because
/// @ref TensorHandle::dtype names BLAS element types and nothing else. The
/// descriptor still records the C++ type (@ref WriteParamDescriptor::source_type),
/// but the tensor record cannot, so the file would describe storage the reader
/// has to guess at. Refuse instead, and name the two spellings that work.
void require_storable_dtype(std::string_view name, packed_gemm::ScalarType dtype) {
    if (dtype != packed_gemm::ScalarType::Unknown) {
        return;
    }
    throw SaveRefusal(fmt::format("tensor '{}' has no BLAS element type, which an integral scalar operand does not: a file cannot name "
                                  "storage for it. Write the parameter from a double scalar, or from a BoundExpr "
                                  "(cg::write_param(name, cg::BoundExpr{{...}}))",
                                  name));
}

/// Write a tensor's provenance tag into @p out, when it carries one.
///
/// Shared by the two records that describe a tensor - its manifest entry and its tensor record -
/// because a field added to one and not the other is the drift this module keeps being bitten by,
/// and it was: the first version of this wrote the tag only into the tensor record, so every
/// tensor a caller actually binds, which is every tensor that has a manifest entry, saved without
/// one.
///
/// Written only when the tag says something, so an untagged tensor costs the file nothing and
/// the golden corpus is unchanged by tags existing.
void write_provenance_tag(Object &out, ProvenanceTag const &tag) {
    if (!tag.valid()) {
        return;
    }
    Object record;
    record.set("name", Value{tag.name});
    if (!tag.attributes.empty()) {
        Object attributes;
        for (auto const &[key, entry] : tag.attributes) {
            attributes.set(key, Value{entry});
        }
        record.set("attributes", Value{std::move(attributes)});
    }
    out.set("tag", Value{std::move(record)});
}

/// The seven keys that describe a tensor's SHAPE, in the one order both records
/// that carry them use.
///
/// A tensor is described twice in a file, once by its manifest entry and once by
/// its tensor record, and the two emitted these keys from two copies of the same
/// code. @p spaces arrives already built because the two sides name spaces
/// differently: a handle holds ids to resolve against the registry, a manifest
/// entry holds the names themselves.
///
/// KEY ORDER IS THE FILE FORMAT here. A saved graph is compared byte for byte
/// against a golden corpus and its content hash is taken over these bytes, so
/// reordering this is a format change, not a cleanup.
void write_shape(Object &out, packed_gemm::ScalarType dtype, std::size_t rank, std::vector<std::size_t> const &dims,
                 std::vector<std::string> const &dim_symbols, Value spaces, bool spaces_inferred, ProvenanceTag const *tag) {
    out.set("dtype", Value{std::string(scalar_type_name(dtype))});
    out.set("rank", Value{rank});
    out.set("dims", to_array(dims));
    out.set("dim_symbols", to_array(dim_symbols));
    out.set("spaces", std::move(spaces));
    out.set("spaces_inferred", Value{spaces_inferred});
    // Provenance is SAVED structure rather than a re-derivable annotation: it is a statement
    // about the mathematics that no machine can recover, and a rewrite justified by it (a delta
    // eliminated, an integral factorized) is exactly what a saved graph keeps.
    if (tag != nullptr) {
        write_provenance_tag(out, *tag);
    }
}

/// One tensor's storage-side record.
Value write_tensor(Graph const &graph, TensorHandle const &handle, std::size_t dense, Frame const &frame) {
    require_storable_dtype(handle.name, handle.dtype);
    Object out;
    out.set("id", Value{dense});
    out.set("name", Value{handle.name});
    write_shape(out, handle.dtype, handle.rank, handle.dims, handle.dim_symbols,
                // name_of and not space(), because an axis nobody has named carries an invalid id
                // and renders as the empty string. That is the deliberate hole annotate_space_axis
                // makes, and a writer that threw on it could not save a graph the public API can
                // build. The reader turns an empty name back into an unannotated axis.
                to_array(handle.spaces, [&graph](SpaceId space) { return Value{graph.space_registry().name_of(space)}; }),
                handle.spaces_inferred, &handle.tag);
    out.set("intermediate", Value{handle.is_intermediate});
    out.set("scope", Value{std::string(tensor_ownership_name(handle.ownership))});
    out.set("init", Value{std::string(init_kind_name(handle.init_kind))});
    // Whether the tensor has storage yet, which is NOT what "init" says: that one is what to
    // fill it with once it does. Dropping this turned every loaded intermediate into a
    // materialized one, and a materialized intermediate refuses the extent-changing bind that
    // cross-problem reuse is made of.
    out.set("alloc", Value{std::string(alloc_state_name(handle.alloc_state))});

    if (auto const outer = frame.outer_of(handle.tensor_ptr); outer.has_value()) {
        out.set("outer", Value{*outer});
    }
    return Value{std::move(out)};
}

/// One node's record.
// NOLINTNEXTLINE(misc-no-recursion): see write_descriptor.
Value write_node(Node const &node, std::size_t dense_id, Graph const &graph, Graph const &root, Frame &frame) {
    Object out;
    out.set("id", Value{dense_id});
    out.set("kind", Value{std::string(op_kind_name(node.kind))});
    out.set("label", Value{node.label});

    auto const dense_of = [&frame](TensorId id) { return Value{frame.intern(id)}; };
    out.set("inputs", to_array(node.inputs, dense_of));
    out.set("outputs", to_array(node.outputs, dense_of));

    // Rank is keyed on the DESTINATION, which is the rule build_executor states;
    // a kind with no tensor destination (write_param's expression arm, the
    // control-flow kinds) records rank 0 and an unknown dtype, neither of which
    // that builder dispatches on.
    //
    // Asked of the KIND and not of the operand lists, because a Setup node carries lists of
    // its own: its body's writes are its outputs. Reading a dtype off the first of them would
    // give a control-flow node a destination it does not have and would make one node's record
    // depend on which tensor a set happened to order first.
    packed_gemm::ScalarType dtype = packed_gemm::ScalarType::Unknown;
    std::size_t             rank  = 0;
    if (is_control_flow(node.kind)) {
        // Nothing to derive.
    } else if (!node.outputs.empty()) {
        if (TensorHandle const *handle = graph.find_tensor(node.outputs[0]); handle != nullptr) {
            dtype = handle->dtype;
            rank  = handle->rank;
        }
    } else if (!node.inputs.empty()) {
        if (TensorHandle const *handle = graph.find_tensor(node.inputs[0]); handle != nullptr) {
            dtype = handle->dtype;
        }
    }
    out.set("dtype", Value{std::string(scalar_type_name(dtype))});
    out.set("rank", Value{rank});

    // A Loop or Conditional node's own operand lists are EMPTY: its body is captured after the
    // node exists, so what it touches is known only through the subtree. A Setup node's lists
    // are refreshed from its body and name the same buffers, but the walk below is what interns
    // the ones only a DESCENDANT of the body names, so it runs for all three. The parent frame
    // has to be told about the buffers the body names BEFORE the fragment is written, because
    // a fragment tensor is matched to an enclosing frame by address and a frame that has not
    // interned the buffer yet reports no match. The body then writes its own tensor record,
    // the load allocates a second buffer for it, and the two ends of one boundary quietly
    // become two tensors.
    //
    // It went unnoticed while every such buffer happened to be mentioned by an EARLIER node.
    // A setup node holding a fitting is the case where that cannot be true: it is spliced at
    // the front precisely so it runs before the consumers that mention what it produces.
    //
    // Walked in program order over the subtree's nodes rather than over its tensor map, which
    // is unordered: the dense numbering is what makes two captures of one program produce
    // byte-identical files, so anything that feeds it has to be ordered.
    if (is_control_flow(node.kind)) {
        // NOLINTNEXTLINE(misc-no-recursion): control-flow bodies nest, so the walk over them does too.
        std::function<void(Graph const &)> intern_boundary = [&](Graph const &sub) {
            for (auto const &inner : sub.nodes()) {
                auto const note = [&](TensorId tid) {
                    TensorHandle const *handle = sub.find_tensor(tid);
                    if (handle == nullptr || handle->tensor_ptr == nullptr) {
                        return;
                    }
                    if (TensorId const outer = graph.find_tensor_id_by_ptr(handle->tensor_ptr); outer != 0) {
                        frame.intern(outer);
                    }
                };
                for (auto const tid : inner.inputs) {
                    note(tid);
                }
                for (auto const tid : inner.outputs) {
                    note(tid);
                }
            }
            sub.for_each_subgraph(intern_boundary);
        };
        // This node's own bodies, taken from its descriptor: for_each_subgraph would hand over
        // every sub-graph in the parent, and interning another node's boundary here would put
        // tensors into the dense order in a position nothing mentions them at.
        for_each_child_graph(node, intern_boundary);
    }

    out.set("descriptor", write_descriptor(node, graph, root, frame));
    return Value{std::move(out)};
}

/// A fragment: local tensors, local nodes, and the boundary references that tie
/// them to the enclosing frame. The same shape at every level, which is the
/// point; see the file note in GraphIR.hpp.
// NOLINTNEXTLINE(misc-no-recursion): fragments nest.
Value write_fragment(Graph const &graph, Graph const &root, Frame const *parent, std::string name) {
    Frame frame(graph, parent);

    // Nodes are walked first so the tensor numbering is first-mention order over
    // the node list, which is what makes two captures of one program agree.
    Array nodes;
    for (std::size_t position = 0; position < graph.nodes().size(); ++position) {
        nodes.emplace_back(write_node(graph.nodes()[position], position, graph, root, frame));
    }

    Array tensors;
    for (std::size_t dense = 0; dense < frame.order().size(); ++dense) {
        TensorHandle const *handle = graph.find_tensor(frame.order()[dense]);
        if (handle == nullptr) {
            throw SaveRefusal(fmt::format("fragment '{}' references tensor id {} that the graph does not define", name,
                                          static_cast<std::uint64_t>(frame.order()[dense])));
        }
        tensors.emplace_back(write_tensor(graph, *handle, dense, frame));
    }

    Object out;
    out.set("name", Value{std::move(name)});
    out.set("tensors", Value{std::move(tensors)});
    out.set("nodes", Value{std::move(nodes)});
    return Value{std::move(out)};
}

/// The provenance block. Data, never instructions; see GraphIR.hpp.
///
/// The pass list comes from the GRAPH, which records each structural-algebraic pass that
/// rewrote it, unless the caller supplied one. It used to come only from the caller, which meant
/// it was accurate exactly as often as someone remembered to fill it in - and the block exists
/// to answer "what shaped this file" for a graph whose numbers turn out wrong, which is the
/// moment nobody has that information to hand.
///
/// A caller who does supply a list still wins, because a tool assembling a file from pieces
/// knows things the graph does not.
Value write_provenance(Graph const &graph, SaveOptions const &options) {
    Object out;
    out.set("library_version", Value{full_version_as_string()});
    out.set("config_fingerprint", Value{fmt::format("0x{:016x}", sealed::config_fingerprint())});
    out.set("structural_passes", to_array(options.structural_passes.empty() ? graph.structural_passes() : options.structural_passes));
    return Value{std::move(out)};
}

} // namespace

/// The structure sections, in their fixed order. This object IS the hash domain;
/// @ref Graph::content_hash digests its canonical bytes and the writer adds only
/// the provenance key beside it.
Object write_structure(Graph const &graph) {
    if (auto const blockers = graph.serializability_report(); !blockers.empty()) {
        std::string report;
        for (auto const &blocker : blockers) {
            report += fmt::format("\n  {}node {} ('{}', {}): {}", blocker.subgraph_path.empty() ? "" : blocker.subgraph_path + "/",
                                  blocker.node_id, blocker.label, blocker.kind_name, blocker.reason);
        }
        throw SaveRefusal(
            fmt::format("graph '{}' holds {} node(s) that cannot be rebuilt from data:{}", graph.name(), blockers.size(), report));
    }

    InterfaceManifest const contract = manifest_of(graph);

    Frame frame(graph, nullptr);
    {
        std::unordered_map<TensorId, TensorId> canonical;
        for (auto const &entry : contract.entries()) {
            for (TensorId const other : entry.also) {
                canonical.emplace(other, entry.id);
            }
        }
        frame.set_canonical(std::move(canonical));
    }
    // The manifest is interned FIRST, in manifest order, so a graph's interface
    // occupies the low dense ids whatever order capture happened to register in.
    for (auto const &entry : contract.entries()) {
        frame.intern(entry.id);
    }

    Array nodes;
    for (std::size_t position = 0; position < graph.nodes().size(); ++position) {
        nodes.emplace_back(write_node(graph.nodes()[position], position, graph, graph, frame));
    }

    Array manifest;
    for (auto const &entry : contract.entries()) {
        require_storable_dtype(entry.name, entry.dtype);
        Object record;
        record.set("id", Value{frame.intern(entry.id)});
        record.set("name", Value{entry.name});
        record.set("direction", Value{std::string(manifest_direction_name(entry.direction))});
        // The tag comes from the HANDLE rather than from the manifest entry, which carries none:
        // a tag is a statement about the tensor, and the manifest entry is a statement about the
        // interface slot it fills.
        TensorHandle const *handle = graph.find_tensor(entry.id);
        write_shape(record, entry.dtype, entry.rank, entry.dims, entry.dim_symbols, to_array(entry.spaces), entry.spaces_inferred,
                    handle != nullptr ? &handle->tag : nullptr);
        record.set("scope", Value{std::string(tensor_ownership_name(entry.scope))});

        // By NAME, never by id: an alias declaration is part of the interface
        // contract and a caller binds by name.
        std::string alias_name;
        if (entry.aliases_input != 0) {
            if (ManifestEntry const *parent = contract.find_by_id(entry.aliases_input); parent != nullptr) {
                alias_name = parent->name;
            }
        }
        record.set("aliases_input", alias_name.empty() ? Value{nullptr} : Value{alias_name});
        manifest.emplace_back(std::move(record));
    }

    Array tensors;
    for (std::size_t dense = 0; dense < frame.order().size(); ++dense) {
        TensorId const      id     = frame.order()[dense];
        TensorHandle const *handle = graph.find_tensor(id);
        if (handle == nullptr) {
            throw SaveRefusal(
                fmt::format("graph '{}' references tensor id {} that it does not define", graph.name(), static_cast<std::uint64_t>(id)));
        }
        if (contract.find_by_id(id) != nullptr) {
            continue; // already fully described by its manifest entry
        }
        tensors.emplace_back(write_tensor(graph, *handle, dense, frame));
    }

    // Spaces: every name any slot or any contraction letter mentions, sorted so
    // the section is a function of the graph rather than of a hash table.
    std::vector<std::string> space_names;
    for (auto const &[id, handle] : graph.tensors_map()) {
        for (auto const space : handle.spaces) {
            // An unannotated axis of a partial annotation names no space, so it contributes
            // nothing to the section that declares which spaces the file mentions.
            if (std::string name = graph.space_registry().name_of(space); !name.empty()) {
                space_names.push_back(std::move(name));
            }
        }
    }
    for (auto const &[symbol, space] : graph.symbol_spaces()) {
        space_names.push_back(graph.space_registry().space(space).name);
    }
    std::ranges::sort(space_names);
    space_names.erase(std::ranges::unique(space_names).begin(), space_names.end());

    std::vector<std::pair<std::string, std::string>> ties;
    for (auto const &[symbol, space] : graph.symbol_spaces()) {
        ties.emplace_back(symbol, graph.space_registry().space(space).name);
    }
    std::ranges::sort(ties);
    Object spaces;
    spaces.set("names", to_array(space_names));
    spaces.set("symbol_ties", to_array(ties, [](auto const &pair) {
                   Object tie;
                   tie.set("symbol", Value{pair.first});
                   tie.set("space", Value{pair.second});
                   return Value{std::move(tie)};
               }));

    // Parameters, sorted by name: ParamTable is an unordered_map.
    std::vector<std::pair<std::string, std::int64_t>> params;
    if (graph.params_ptr() != nullptr) {
        for (auto const &[name, value] : graph.params_ptr()->entries()) {
            params.emplace_back(name, value);
        }
    }
    std::ranges::sort(params);
    Value const param_array = to_array(params, [](auto const &pair) {
        Object param;
        param.set("name", Value{pair.first});
        param.set("value", Value{pair.second});
        return Value{std::move(param)};
    });

    Value const gate_flags = to_array(graph.named_gate_flags(), [](auto const &pair) {
        Object flags;
        flags.set("name", Value{pair.first});
        flags.set("size", Value{pair.second != nullptr ? pair.second->size() : std::size_t{0}});
        return Value{std::move(flags)};
    });

    // Slot redirects, sorted by dense id. Only a pair whose two ends the walk
    // already reached can be written; a redirect naming a tensor no node
    // mentions has nothing to act on and is dropped.
    std::vector<std::pair<std::size_t, std::size_t>> redirects;
    for (auto const &[from, to] : graph.slot_redirects()) {
        auto const dense_from = std::ranges::find(frame.order(), from);
        auto const dense_to   = std::ranges::find(frame.order(), to);
        if (dense_from == frame.order().end() || dense_to == frame.order().end()) {
            continue;
        }
        redirects.emplace_back(static_cast<std::size_t>(dense_from - frame.order().begin()),
                               static_cast<std::size_t>(dense_to - frame.order().begin()));
    }
    std::ranges::sort(redirects);
    Value const redirect_array = to_array(redirects, [](auto const &pair) {
        Object redirect;
        redirect.set("from", Value{pair.first});
        redirect.set("to", Value{pair.second});
        return Value{std::move(redirect)};
    });

    // Approximation records, in the order they were applied. Order is content here rather
    // than presentation: composition is not commutative for a relative effect, so a reader
    // re-deriving the composed bound has to see them in the order the passes ran.
    Value const approximations = to_array(graph.approximations(), [](auto const &record) {
        Object entry;
        entry.set("pass_name", Value{record.pass_name});
        entry.set("effect", Value{std::string(approximation_effect_name(record.effect))});
        entry.set("tolerance", Value{record.tolerance});
        entry.set("bound", Value{record.bound});
        entry.set("origin", Value{std::string(approximation_origin_name(record.origin))});
        entry.set("outputs", to_array(record.outputs));
        entry.set("spaces", to_array(record.spaces));
        entry.set("setup", Value{record.setup});
        entry.set("measurement", Value{record.measurement});
        return Value{std::move(entry)};
    });

    Object out;
    out.set(std::string(key_version), Value{std::string(graph_ir_schema_version)});
    out.set("name", Value{graph.name()});
    out.set("manifest", Value{std::move(manifest)});
    out.set("spaces", Value{std::move(spaces)});
    out.set("approximations", approximations);
    out.set("params", param_array);
    out.set("gate_flags", gate_flags);
    out.set("tensors", Value{std::move(tensors)});
    out.set("slot_redirects", redirect_array);
    out.set("nodes", Value{std::move(nodes)});
    return out;
}

/// The whole document: the structure sections with provenance spliced in at the
/// fixed second position.
Value write_document(Graph const &graph, SaveOptions const &options) {
    Object const structure = write_structure(graph);
    Object       out;
    out.set(std::string(key_version), Value{std::string(graph_ir_schema_version)});
    out.set(std::string(key_provenance), write_provenance(graph, options));
    for (std::size_t i = 0; i < structure.size(); ++i) {
        if (structure.key_at(i) == key_version) {
            continue;
        }
        out.set(structure.key_at(i), structure.value_at(i));
    }
    return Value{std::move(out)};
}

EINSUMS_NAMESPACE_END(compute_graph::graph_ir)
