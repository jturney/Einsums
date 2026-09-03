//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GraphIR.cpp
/// @brief The ``einsums_graph_ir`` writer, reader and validator.
///
/// The schema itself is documented once, in ``GraphIR.hpp``; this file is the
/// implementation of it. Three pieces, in order:
///
///  1. **The member walk.** Which of `Graph`'s members are STRUCTURE and travel
///     in a file, and which are tuning artifacts that deliberately do not. Kept
///     beside `Graph::move_members_from`'s "don't forget a member" discipline;
///     that function's comment points here and this one points back.
///  2. **The writer.** Renumbering, canonical emission, and the refusals a save
///     makes rather than writing something a reader could only half rebuild.
///  3. **The reader.** A document is read into a plain intermediate form first,
///     collecting EVERY problem, and only a clean read is turned into a graph.
///     That split is what lets `validate_graph_ir` report three seeded errors as
///     three messages while `load_graph` stops at the first.
///
/// @par The member walk (see also Graph::move_members_from)
/// Saved, because it is structure:
///   `_name`, `_nodes` (kind, label, inputs, outputs, descriptor), `_tensors`
///   (name, dtype, rank, dims, dim symbols, spaces, intermediate flag, scope,
///   init kind, alias parent), `_symbol_spaces`, `_params`' entries,
///   `_named_gate_flags`, `_slot_redirects`.
///
/// NOT saved, because it is machine-dependent tuning that a load re-derives:
///   `Node::thread_width`, `Node::admission_priority`, `Node::stream_id`,
///   `Node::estimated_flops`, `Node::estimated_bytes`, `_planned_thread_count`,
///   `_plan_trial` and the two plan snapshots, `_timing_samples`,
///   `_timing_report`, `_last_optimize_report`.
///
/// NOT saved, because it is a live-process resource with no meaning in a file:
///   every `std::function` on a `TensorHandle`, `_owned_tensors`,
///   `_adopted_cleanups`, `_slot_map`, `_ptr_index`, `_device_shadows`,
///   `_executor`, `_params_store`, `_indices_store`, `_content_mutex`,
///   `_deps` / `_usage` and the version counters that guard them,
///   `_profile_strings`, `_bound_operands`, `_ragged_extents`, `_scope_maps`.
///   `_ragged_extents` and `_bound_operands` are bind-time state about the
///   CALLER's tensors, and a loaded graph is bound afresh.
///
/// `_space_registry` is a non-owning pointer into a registry the caller owns, so
/// what travels is the space NAMES; a loaded graph resolves them against the
/// registry it is given.

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
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

using json::Array;
using json::Object;
using json::Value;

// ── Small shared helpers ───────────────────────────────────────────────────

/// The section keys, in the order the schema fixes them.
constexpr std::string_view key_version    = "einsums_graph_ir";
constexpr std::string_view key_provenance = "provenance";

/// @ref Graph::manifest mutates (it links aliases and builds usage analysis) and
/// is documented as non-const "not by choice". Writing a graph is a read-only
/// operation from the caller's point of view, so the const is kept at the API
/// boundary and dropped exactly here, once, with the reason written down.
InterfaceManifest manifest_of(Graph const &graph) {
    return const_cast<Graph &>(graph).manifest(); // NOLINT(cppcoreguidelines-pro-type-const-cast)
}

/// Render a double, or the tagged string that stands for a value JSON cannot
/// spell. The three tags are the whole special-value policy: JSON has no NaN and
/// no infinity, and a schema that wrote `null` for them would lose the
/// distinction between "not a number" and "no value here".
Value number_or_tag(double value) {
    if (std::isnan(value)) {
        return Value{"nan"};
    }
    if (std::isinf(value)) {
        return Value{value > 0 ? "inf" : "-inf"};
    }
    return Value{value};
}

/// The inverse of @ref number_or_tag.
std::optional<double> tagged_number(Value const &value) {
    if (value.is_number()) {
        return value.as_double();
    }
    if (!value.is_string()) {
        return std::nullopt;
    }
    if (value.as_string() == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (value.as_string() == "inf") {
        return std::numeric_limits<double>::infinity();
    }
    if (value.as_string() == "-inf") {
        return -std::numeric_limits<double>::infinity();
    }
    return std::nullopt;
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

// ── The intermediate form ──────────────────────────────────────────────────
//
// A document is read into these first. Building a Graph is a separate step that
// only runs on a clean read, which is what lets the reader keep going after a
// problem and report every one of them.

struct IrTensor {
    std::size_t              id{0};
    std::string              name;
    packed_gemm::ScalarType  dtype{packed_gemm::ScalarType::Unknown};
    std::size_t              rank{0};
    std::vector<std::size_t> dims;
    std::vector<std::string> dim_symbols;
    std::vector<std::string> spaces;
    bool                     spaces_inferred{false};

    /// What the tensor is declared to BE. Optional, because the vast majority of tensors are
    /// untagged and writing an empty object for each of them would cost every file bytes for
    /// nothing; absent reads as untagged.
    ProvenanceTag tag;

    bool            intermediate{false};
    TensorOwnership scope{TensorOwnership::Graph};
    InitKind        init{InitKind::None};

    /// Whether the tensor still needs storage. Absent in files written before this was
    /// recorded, and Materialized is the right reading of those: a graph that never said
    /// otherwise had its intermediates allocated at capture.
    AllocState alloc{AllocState::Materialized};

    /// Set on a FRAGMENT tensor that denotes a buffer the enclosing frame
    /// already defines; the value is that frame's id for it.
    std::optional<std::size_t> outer;

    /// Manifest-only: the entry whose storage this one is part of, by name.
    std::string aliases_input;

    /// Manifest-only: what the file claims the direction is, checked against
    /// what the loaded nodes actually do.
    ManifestDirection direction{ManifestDirection::Input};
};

struct IrFragment;

struct IrNode {
    std::size_t              id{0};
    OpKind                   kind{OpKind::Custom};
    std::string              label;
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;
    packed_gemm::ScalarType  dtype{packed_gemm::ScalarType::Unknown};
    std::size_t              rank{0};
    OpData                   descriptor;

    /// Dense operand ids a descriptor names, remapped at build time. Only a GEMM
    /// hint has any, and they are kept out of the descriptor until then because a
    /// descriptor holds real @ref TensorId values.
    std::vector<std::size_t> hint_ids;

    std::shared_ptr<IrFragment> then_branch;
    std::shared_ptr<IrFragment> else_branch;
    std::shared_ptr<IrFragment> body;
};

struct IrFragment {
    std::string           name;
    std::vector<IrTensor> tensors;
    std::vector<IrNode>   nodes;
};

/// Gate-flag buffers, by name. Made while READING, because the predicates read
/// out of the document already hold them; the graph then adopts exactly these
/// rather than allocating a second set nothing points at.
using GateFlagTable = std::unordered_map<std::string, std::shared_ptr<std::vector<std::uint8_t>>>;

struct IrDocument {
    std::string version;
    std::string name;
    /// The provenance block's pass list. CARRIED, never acted on: a load does not re-run one
    /// of these, and does not refuse a graph because of what is in the list. It is put back on
    /// the loaded graph so a load, further optimization and a re-save do not silently drop the
    /// history of everything that shaped the file in the first place.
    std::vector<std::string>                          structural_passes;
    GateFlagTable                                     gate_buffers;
    std::vector<IrTensor>                             manifest;
    std::vector<std::string>                          space_names;
    std::vector<std::pair<std::string, std::string>>  symbol_ties;
    std::vector<ApproximationRecord>                  approximations;
    std::vector<std::pair<std::string, std::int64_t>> params;
    std::vector<std::pair<std::string, std::size_t>>  gate_flags;
    std::vector<IrTensor>                             tensors;
    std::vector<std::pair<std::size_t, std::size_t>>  slot_redirects;
    std::vector<IrNode>                               nodes;
};

// ── Writer ─────────────────────────────────────────────────────────────────

/// Thrown by the writer when a graph holds something a file cannot carry.
/// Caught at the API boundary and turned into a @ref GraphError.
struct SaveRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// One frame's dense numbering: this graph's ids in first-mention order, plus
/// the enclosing frames' tensor identities so a body can name a parent buffer.
class Frame {
  public:
    Frame(Graph const &graph, Frame const *parent) : _graph(graph), _parent(parent) {}

    /// The dense id for @p id, minting one on first mention.
    std::size_t intern(TensorId id) {
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
        }
        return std::nullopt;
    }

  private:
    Graph const                              &_graph;
    Frame const                              *_parent;
    std::unordered_map<TensorId, std::size_t> _dense;
    std::vector<TensorId>                     _order;
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
                to_array(handle.spaces, [&graph](SpaceId space) { return Value{graph.space_registry().space(space).name}; }),
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
    // a kind with no tensor destination (write_param's expression arm, the two
    // control-flow kinds) records rank 0 and an unknown dtype, neither of which
    // that builder dispatches on.
    packed_gemm::ScalarType dtype = packed_gemm::ScalarType::Unknown;
    std::size_t             rank  = 0;
    if (!node.outputs.empty()) {
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

    // A control-flow node's own operand lists are EMPTY: its body is captured after the node
    // exists, so what it touches is known only through the subtree. The parent frame therefore
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
            space_names.push_back(graph.space_registry().space(space).name);
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

// ── Reader ─────────────────────────────────────────────────────────────────
//
// Every helper here records a problem and returns a harmless default rather
// than bailing out, which is what lets one pass over a document report every
// problem it has. `load_graph` then reports the first and `validate_graph_ir`
// all of them, from the same code.

/// Problems found while reading, in document order.
using Problems = std::vector<std::string>;

void note(Problems &problems, std::string const &path, json::Position where, std::string message) {
    problems.push_back(fmt::format("line {} column {}: {}: {}", where.line, where.column, path, std::move(message)));
}

Object const *as_object(Value const &value, std::string const &path, Problems &problems) {
    if (!value.is_object()) {
        note(problems, path, value.position, fmt::format("expected an object, found {}", value.type_name()));
        return nullptr;
    }
    return &value.as_object();
}

Array const *as_array(Value const &value, std::string const &path, Problems &problems) {
    if (!value.is_array()) {
        note(problems, path, value.position, fmt::format("expected an array, found {}", value.type_name()));
        return nullptr;
    }
    return &value.as_array();
}

/// Consume @p key, reporting its absence.
Value const *field(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    Value const *value = object.take(key);
    if (value == nullptr) {
        note(problems, path, where, fmt::format("required key '{}' is missing", key));
    }
    return value;
}

/// What @ref read_scalar needs to know about one leaf type: how to recognise it,
/// how to read it, and how a diagnostic names it. Three near-identical readers
/// differing only in that triple is what this replaces.
template <typename T>
struct ScalarLeaf;

template <>
struct ScalarLeaf<std::string> {
    static constexpr std::string_view noun = "a string";
    static bool                       matches(Value const &value) { return value.is_string(); }
    static std::string                read(Value const &value) { return value.as_string(); }
};

template <>
struct ScalarLeaf<std::int64_t> {
    static constexpr std::string_view noun = "an integer";
    static bool                       matches(Value const &value) { return value.is_int(); }
    static std::int64_t               read(Value const &value) { return value.as_int(); }
};

template <>
struct ScalarLeaf<bool> {
    static constexpr std::string_view noun = "a bool";
    static bool                       matches(Value const &value) { return value.is_bool(); }
    static bool                       read(Value const &value) { return value.as_bool(); }
};

/// Consume @p key and read it as a @c T, reporting an absent key and a wrong type.
/// A failure of either kind yields a value-initialized @c T and a problem, so a
/// caller reads on and the load collects every fault rather than the first.
template <typename T>
T read_scalar(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    Value const *value = field(object, key, path, problems, where);
    if (value == nullptr) {
        return T{};
    }
    if (!ScalarLeaf<T>::matches(*value)) {
        note(problems, fmt::format("{}.{}", path, key), value->position,
             fmt::format("expected {}, found {}", ScalarLeaf<T>::noun, value->type_name()));
        return T{};
    }
    return ScalarLeaf<T>::read(*value);
}

std::string read_string(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    return read_scalar<std::string>(object, key, path, problems, where);
}

std::int64_t read_int(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    return read_scalar<std::int64_t>(object, key, path, problems, where);
}

bool read_bool(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    return read_scalar<bool>(object, key, path, problems, where);
}

/// A tensor's provenance tag, or an empty one when the record carries none.
///
/// OPTIONAL by the compatibility policy: a file written before tags existed has no ``tag`` key
/// and its tensors are untagged, which is exactly what they were. Reading it as anything else
/// would break the golden corpus, and the corpus is right that an added field takes a documented
/// default.
///
/// Every key of the tag object is CONSUMED, including the attribute keys, because an unconsumed
/// key is a load-time error by design: a file carrying a field this build does not understand is
/// a file written by something newer, and reading it silently would be the drift the strict
/// document model exists to prevent.
ProvenanceTag read_provenance_tag(Object const &object, std::string const &path, Problems &problems) {
    ProvenanceTag out;
    Value const  *value = object.take("tag");
    if (value == nullptr || value->is_null()) {
        return out;
    }
    if (!value->is_object()) {
        note(problems, fmt::format("{}.tag", path), value->position, fmt::format("expected an object, found {}", value->type_name()));
        return out;
    }

    Object const     &tag      = value->as_object();
    std::string const tag_path = fmt::format("{}.tag", path);
    out.name                   = read_string(tag, "name", tag_path, problems, value->position);

    if (Value const *attributes = tag.take("attributes"); attributes != nullptr && !attributes->is_null()) {
        if (!attributes->is_object()) {
            note(problems, fmt::format("{}.attributes", tag_path), attributes->position,
                 fmt::format("expected an object, found {}", attributes->type_name()));
        } else {
            Object const &entries = attributes->as_object();
            for (auto const &key : entries.keys()) {
                Value const *entry = entries.take(key);
                if (entry == nullptr) {
                    continue;
                }
                if (!entry->is_string()) {
                    note(problems, fmt::format("{}.attributes.{}", tag_path, key), entry->position,
                         fmt::format("expected a string, found {}", entry->type_name()));
                    continue;
                }
                out.attributes.emplace_back(key, entry->as_string());
            }
            // Sorted, matching what the writer's own sort produced, so two loads of one file
            // compare equal and a tag round-trips to the same bytes.
            std::ranges::sort(out.attributes, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
        }
    }

    if (out.name.empty()) {
        note(problems, tag_path, value->position, "a provenance tag with an empty name says nothing; omit the key instead");
    }
    return out;
}

std::vector<std::string> read_string_array(Object const &object, std::string_view key, std::string const &path, Problems &problems,
                                           json::Position where) {
    std::vector<std::string> out;
    Value const             *value = field(object, key, path, problems, where);
    if (value == nullptr) {
        return out;
    }
    std::string const child = fmt::format("{}.{}", path, key);
    Array const      *items = as_array(*value, child, problems);
    if (items == nullptr) {
        return out;
    }
    for (std::size_t i = 0; i < items->size(); ++i) {
        if (!(*items)[i].is_string()) {
            note(problems, fmt::format("{}[{}]", child, i), (*items)[i].position,
                 fmt::format("expected a string, found {}", (*items)[i].type_name()));
            continue;
        }
        out.push_back((*items)[i].as_string());
    }
    return out;
}

std::vector<std::size_t> read_extent_array(Object const &object, std::string_view key, std::string const &path, Problems &problems,
                                           json::Position where) {
    std::vector<std::size_t> out;
    Value const             *value = field(object, key, path, problems, where);
    if (value == nullptr) {
        return out;
    }
    std::string const child = fmt::format("{}.{}", path, key);
    Array const      *items = as_array(*value, child, problems);
    if (items == nullptr) {
        return out;
    }
    for (std::size_t i = 0; i < items->size(); ++i) {
        if (!(*items)[i].is_int() || (*items)[i].as_int() < 0) {
            note(problems, fmt::format("{}[{}]", child, i), (*items)[i].position, "expected a non-negative integer");
            out.push_back(0);
            continue;
        }
        out.push_back(static_cast<std::size_t>((*items)[i].as_int()));
    }
    return out;
}

/// Resolve an already-fetched value as a by-name enumerator, reporting the string
/// that did not resolve and what the alternatives are. This shape - the name plus
/// the known set - is what makes an unresolvable name actionable rather than
/// merely fatal.
///
/// Takes the value rather than the key so that the required and the optional
/// spelling below differ only in how they FETCH it; a null @p value is already
/// either reported (required) or allowed (optional), so it is a silent fallback
/// here either way.
template <typename T, typename Fn>
T resolve_named(Value const *value, std::string_view key, std::string const &path, Problems &problems, Fn &&resolve, std::string_view what,
                T fallback) {
    if (value == nullptr) {
        return fallback;
    }
    if (!value->is_string()) {
        note(problems, fmt::format("{}.{}", path, key), value->position, fmt::format("expected a string, found {}", value->type_name()));
        return fallback;
    }
    if (auto const resolved = resolve(value->as_string()); resolved.has_value()) {
        return *resolved;
    }
    note(problems, fmt::format("{}.{}", path, key), value->position, fmt::format("'{}' is not a known {}", value->as_string(), what));
    return fallback;
}

/// A by-name enumerator under a key the file must carry.
template <typename T, typename Fn>
T read_named(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where, Fn &&resolve,
             std::string_view what, T fallback) {
    return resolve_named(field(object, key, path, problems, where), key, path, problems, std::forward<Fn>(resolve), what, fallback);
}

/// As @ref read_named, but for a key a file is allowed not to have.
///
/// The compatibility policy lets the schema GAIN fields, with an absent one taking its
/// documented default, so a key added after a golden was written must not be demanded of it.
/// ``take`` rather than @ref field: it marks the key consumed for the strict unconsumed-key
/// check without reporting a missing one as a problem.
template <typename T, typename Fn>
T read_named_optional(Object const &object, std::string_view key, std::string const &path, Problems &problems, Fn &&resolve,
                      std::string_view what, T fallback) {
    return resolve_named(object.take(key), key, path, problems, std::forward<Fn>(resolve), what, fallback);
}

/// A typed scalar, back to a @ref PrefactorScalar.
PrefactorScalar read_prefactor(Value const &value, std::string const &path, Problems &problems) {
    PrefactorScalar out{double{0}};
    Object const   *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    auto const dtype = read_named<packed_gemm::ScalarType>(*object, "dtype", path, problems, value.position, scalar_type_from_name, "dtype",
                                                           packed_gemm::ScalarType::Float64);

    auto const component = [&](std::string_view key, bool required) -> double {
        Value const *component_value = object->take(key);
        if (component_value == nullptr) {
            if (required) {
                note(problems, path, value.position, fmt::format("typed scalar is missing '{}'", key));
            }
            return 0.0;
        }
        if (auto const number = tagged_number(*component_value); number.has_value()) {
            return *number;
        }
        note(problems, fmt::format("{}.{}", path, key), component_value->position,
             R"(expected a number or one of the special tags "nan", "inf", "-inf")");
        return 0.0;
    };

    double const re = component("re", true);
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        return PrefactorScalar{static_cast<float>(re)};
    case packed_gemm::ScalarType::Float64:
        return PrefactorScalar{re};
    case packed_gemm::ScalarType::Complex64:
        return PrefactorScalar{std::complex<float>{static_cast<float>(re), static_cast<float>(component("im", true))}};
    case packed_gemm::ScalarType::Complex128:
        return PrefactorScalar{std::complex<double>{re, component("im", true)}};
    default:
        note(problems, path, value.position, "a typed scalar cannot have dtype 'unknown'");
        return out;
    }
}

BoundExpr read_bound_expr(Value const &value, std::string const &path, Problems &problems) {
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return BoundExpr{std::int64_t{0}};
    }
    if (Value const *literal = object->take("const"); literal != nullptr) {
        if (!literal->is_int()) {
            note(problems, fmt::format("{}.const", path), literal->position, "expected an integer");
            return BoundExpr{std::int64_t{0}};
        }
        return BoundExpr{literal->as_int()};
    }
    if (Value const *param = object->take("param"); param != nullptr) {
        if (!param->is_string()) {
            note(problems, fmt::format("{}.param", path), param->position, "expected a string");
            return BoundExpr{std::int64_t{0}};
        }
        return BoundExpr{param->as_string()};
    }
    note(problems, path, value.position, R"(a BoundExpr must be {"const": <integer>} or {"param": "<name>"})");
    return BoundExpr{std::int64_t{0}};
}

PredExpr read_pred_expr(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates) {
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return PredExpr{true};
    }
    if (Value const *literal = object->take("const"); literal != nullptr) {
        if (!literal->is_bool()) {
            note(problems, fmt::format("{}.const", path), literal->position, "expected a bool");
            return PredExpr{true};
        }
        return PredExpr{literal->as_bool()};
    }
    if (Value const *compare = object->take("compare"); compare != nullptr) {
        std::string const child = fmt::format("{}.compare", path);
        Object const     *body  = as_object(*compare, child, problems);
        if (body == nullptr) {
            return PredExpr{true};
        }
        Value const *lhs = field(*body, "lhs", child, problems, compare->position);
        auto const   op =
            read_named<CmpOp>(*body, "op", child, problems, compare->position, cmp_op_from_name, "comparison operator", CmpOp::Lt);
        Value const *rhs = field(*body, "rhs", child, problems, compare->position);
        if (lhs == nullptr || rhs == nullptr) {
            return PredExpr{true};
        }
        return PredExpr::compare(read_bound_expr(*lhs, child + ".lhs", problems), op, read_bound_expr(*rhs, child + ".rhs", problems));
    }
    if (Value const *iteration = object->take("iteration"); iteration != nullptr) {
        std::string const child = fmt::format("{}.iteration", path);
        Object const     *body  = as_object(*iteration, child, problems);
        if (body == nullptr) {
            return PredExpr{true};
        }
        auto const op =
            read_named<CmpOp>(*body, "op", child, problems, iteration->position, cmp_op_from_name, "comparison operator", CmpOp::Lt);
        Value const *rhs = field(*body, "rhs", child, problems, iteration->position);
        if (rhs == nullptr) {
            return PredExpr{true};
        }
        return PredExpr::iteration(op, read_bound_expr(*rhs, child + ".rhs", problems));
    }
    if (Value const *flag = object->take("flag"); flag != nullptr) {
        std::string const child = fmt::format("{}.flag", path);
        Object const     *body  = as_object(*flag, child, problems);
        if (body == nullptr) {
            return PredExpr{true};
        }
        std::string const name  = read_string(*body, "name", child, problems, flag->position);
        auto const        index = read_int(*body, "index", child, problems, flag->position);
        auto const        found = gates.find(name);
        if (found == gates.end()) {
            std::vector<std::string> known;
            known.reserve(gates.size());
            for (auto const &[gate_name, buffer] : gates) {
                known.push_back(gate_name);
            }
            std::ranges::sort(known);
            note(problems, child, flag->position,
                 fmt::format("gate-flag array '{}' is not declared by this file. Declared: [{}]", name, fmt::join(known, ", ")));
            return PredExpr{true};
        }
        if (index < 0) {
            note(problems, child, flag->position, "a gate-flag index must not be negative");
            return PredExpr{true};
        }
        return PredExpr::flag(found->second, static_cast<std::size_t>(index));
    }
    note(problems, path, value.position,
         R"(a PredExpr must be one of {"const": <bool>}, {"compare": ...}, {"iteration": ...} or {"flag": ...})");
    return PredExpr{true};
}

/// One tensor record. A MANIFEST entry and an intermediate share most of their
/// shape and differ in the two directions the schema keeps apart: a manifest
/// entry declares a direction and an alias parent, an intermediate declares
/// whether it is graph-owned and how it is initialized.
IrTensor read_tensor(Value const &value, std::string const &path, Problems &problems, bool manifest_entry) {
    IrTensor      out;
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    auto const id = read_int(*object, "id", path, problems, value.position);
    if (id < 0) {
        note(problems, path, value.position, "a tensor id must not be negative");
    }
    out.id          = static_cast<std::size_t>(std::max<std::int64_t>(id, 0));
    out.name        = read_string(*object, "name", path, problems, value.position);
    out.dtype       = read_named<packed_gemm::ScalarType>(*object, "dtype", path, problems, value.position, scalar_type_from_name, "dtype",
                                                    packed_gemm::ScalarType::Float64);
    out.rank        = static_cast<std::size_t>(std::max<std::int64_t>(read_int(*object, "rank", path, problems, value.position), 0));
    out.dims        = read_extent_array(*object, "dims", path, problems, value.position);
    out.dim_symbols = read_string_array(*object, "dim_symbols", path, problems, value.position);
    out.spaces      = read_string_array(*object, "spaces", path, problems, value.position);
    out.spaces_inferred = read_bool(*object, "spaces_inferred", path, problems, value.position);
    out.tag             = read_provenance_tag(*object, path, problems);
    out.scope = read_named<TensorOwnership>(*object, "scope", path, problems, value.position, tensor_ownership_from_name, "ownership scope",
                                            TensorOwnership::Graph);
    if (manifest_entry) {
        out.direction = read_named<ManifestDirection>(*object, "direction", path, problems, value.position, manifest_direction_from_name,
                                                      "manifest direction", ManifestDirection::Input);
        if (Value const *alias = field(*object, "aliases_input", path, problems, value.position); alias != nullptr) {
            if (alias->is_string()) {
                out.aliases_input = alias->as_string();
            } else if (!alias->is_null()) {
                note(problems, fmt::format("{}.aliases_input", path), alias->position, "expected a manifest name or null");
            }
        }
    } else {
        out.intermediate = read_bool(*object, "intermediate", path, problems, value.position);
        out.init         = read_named<InitKind>(*object, "init", path, problems, value.position, init_kind_from_name, "initialization kind",
                                        InitKind::None);
        // Optional: a file written before this key existed means Materialized, which is what
        // its graph's intermediates were.
        out.alloc = read_named_optional<AllocState>(*object, "alloc", path, problems, alloc_state_from_name, "allocation state",
                                                    AllocState::Materialized);
    }

    if (out.dims.size() != out.rank) {
        note(problems, path, value.position, fmt::format("rank is {} but {} dims are given", out.rank, out.dims.size()));
    }
    if (!out.dim_symbols.empty() && out.dim_symbols.size() != out.rank) {
        note(problems, path, value.position,
             fmt::format("rank is {} but {} dim symbols are given; the annotation is all axes or none", out.rank, out.dim_symbols.size()));
    }
    if (!out.spaces.empty() && out.spaces.size() != out.rank) {
        note(problems, path, value.position,
             fmt::format("rank is {} but {} index spaces are given; the annotation is all axes or none", out.rank, out.spaces.size()));
    }

    if (Value const *outer = object->take("outer"); outer != nullptr) {
        if (!outer->is_int() || outer->as_int() < 0) {
            note(problems, fmt::format("{}.outer", path), outer->position, "expected a non-negative integer");
        } else {
            out.outer = static_cast<std::size_t>(outer->as_int());
        }
    }
    return out;
}

IrFragment read_fragment(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                         SpaceRegistry const &registry);

/// One node's descriptor, per kind. The coverage is exactly the reconstructible
/// set, which is what makes an unknown kind here a file problem rather than a
/// gap: the writer could not have produced one.
// NOLINTNEXTLINE(misc-no-recursion): control-flow descriptors hold fragments.
void read_descriptor(IrNode &node, Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                     SpaceRegistry const &registry) {
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return;
    }
    auto const scalar = [&](std::string_view key, PrefactorScalar fallback) {
        Value const *entry = field(*object, key, path, problems, value.position);
        return entry != nullptr ? read_prefactor(*entry, fmt::format("{}.{}", path, key), problems) : fallback;
    };
    auto const transpose_char = [&](std::string_view key, char fallback) {
        std::string const text = read_string(*object, key, path, problems, value.position);
        if (text.size() != 1) {
            if (!text.empty()) {
                note(problems, fmt::format("{}.{}", path, key), value.position, "a BLAS transpose flag is one character");
            }
            return fallback;
        }
        return text[0];
    };

    switch (node.kind) {
    case OpKind::Transpose:
        node.descriptor = std::monostate{};
        return;
    case OpKind::Scale: {
        ScaleDescriptor desc;
        desc.factor     = scalar("factor", PrefactorScalar{double{1}});
        desc.params     = make_elementwise_params(desc.factor);
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Permute: {
        PermuteDescriptor desc;
        auto const        alpha = scalar("alpha", PrefactorScalar{double{1}});
        auto const        beta  = scalar("beta", PrefactorScalar{double{0}});
        desc.alpha              = as<std::complex<double>>(alpha);
        desc.beta               = as<std::complex<double>>(beta);
        desc.c_indices          = read_string_array(*object, "c_indices", path, problems, value.position);
        desc.a_indices          = read_string_array(*object, "a_indices", path, problems, value.position);
        desc.params             = make_elementwise_params(alpha, beta);
        node.descriptor         = std::move(desc);
        return;
    }
    case OpKind::Axpby: {
        AxpbyDescriptor desc;
        desc.alpha      = scalar("alpha", PrefactorScalar{double{1}});
        desc.beta       = scalar("beta", PrefactorScalar{double{0}});
        desc.params     = make_elementwise_params(desc.alpha, desc.beta);
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::DirectProduct:
    case OpKind::DirectDivision: {
        ElementwiseBinaryDescriptor desc;
        desc.alpha      = scalar("alpha", PrefactorScalar{double{1}});
        desc.beta       = scalar("beta", PrefactorScalar{double{0}});
        desc.params     = make_elementwise_params(desc.alpha, desc.beta);
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Einsum: {
        EinsumDescriptor desc;
        desc.spec.c_indices      = read_string_array(*object, "c_indices", path, problems, value.position);
        desc.spec.a_indices      = read_string_array(*object, "a_indices", path, problems, value.position);
        desc.spec.b_indices      = read_string_array(*object, "b_indices", path, problems, value.position);
        desc.spec.link_indices   = read_string_array(*object, "link_indices", path, problems, value.position);
        desc.spec.target_indices = read_string_array(*object, "target_indices", path, problems, value.position);
        desc.spec.all_indices    = read_string_array(*object, "all_indices", path, problems, value.position);
        desc.spec.scalar_output  = read_bool(*object, "scalar_output", path, problems, value.position);
        desc.conj_a              = read_bool(*object, "conj_a", path, problems, value.position);
        desc.conj_b              = read_bool(*object, "conj_b", path, problems, value.position);
        desc.spec.conj_a         = desc.conj_a;
        desc.spec.conj_b         = desc.conj_b;
        desc.spec.scalar_type    = node.dtype;
        desc.c_prefactor         = scalar("c_prefactor", PrefactorScalar{double{0}});
        desc.ab_prefactor        = scalar("ab_prefactor", PrefactorScalar{double{1}});

        if (Value const *letters = field(*object, "letter_spaces", path, problems, value.position); letters != nullptr) {
            std::string const child = fmt::format("{}.letter_spaces", path);
            if (Array const *items = as_array(*letters, child, problems); items != nullptr) {
                for (std::size_t i = 0; i < items->size(); ++i) {
                    std::string const entry_path = fmt::format("{}[{}]", child, i);
                    Object const     *entry      = as_object((*items)[i], entry_path, problems);
                    if (entry == nullptr) {
                        continue;
                    }
                    std::string const letter = read_string(*entry, "letter", entry_path, problems, (*items)[i].position);
                    std::string const space  = read_string(*entry, "space", entry_path, problems, (*items)[i].position);
                    auto const        id     = registry.find(space);
                    if (!id.has_value()) {
                        note(problems, entry_path, (*items)[i].position,
                             fmt::format("index space '{}' is not registered in this process", space));
                        continue;
                    }
                    desc.letter_spaces.emplace_back(letter, *id);
                }
            }
        }

        if (Value const *hint = object->take("gemm_hint"); hint != nullptr) {
            std::string const child = fmt::format("{}.gemm_hint", path);
            if (Object const *body = as_object(*hint, child, problems); body != nullptr) {
                auto record          = std::make_shared<GemmHint>();
                record->m            = static_cast<int>(read_int(*body, "m", child, problems, hint->position));
                record->n            = static_cast<int>(read_int(*body, "n", child, problems, hint->position));
                record->k            = static_cast<int>(read_int(*body, "k", child, problems, hint->position));
                std::string const ta = read_string(*body, "trans_a", child, problems, hint->position);
                std::string const tb = read_string(*body, "trans_b", child, problems, hint->position);
                record->trans_a      = ta.size() == 1 ? ta[0] : 'N';
                record->trans_b      = tb.size() == 1 ? tb[0] : 'N';
                auto const operand   = [&](std::string_view key, GemmOperand &slot) {
                    Value const *entry = field(*body, key, child, problems, hint->position);
                    if (entry == nullptr) {
                        node.hint_ids.push_back(0);
                        return;
                    }
                    std::string const operand_path = fmt::format("{}.{}", child, key);
                    Object const     *fields       = as_object(*entry, operand_path, problems);
                    if (fields == nullptr) {
                        node.hint_ids.push_back(0);
                        return;
                    }
                    node.hint_ids.push_back(static_cast<std::size_t>(
                        std::max<std::int64_t>(read_int(*fields, "id", operand_path, problems, entry->position), 0)));
                    slot.leading_dim = static_cast<int>(read_int(*fields, "leading_dim", operand_path, problems, entry->position));
                };
                operand("a", record->a);
                operand("b", record->b);
                operand("c", record->c);
                desc.gemm_hint = std::move(record);
            }
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Dot: {
        DotDescriptor desc;
        desc.conjugated = read_bool(*object, "conjugated", path, problems, value.position);
        node.descriptor = desc;
        return;
    }
    case OpKind::Trace:
        node.descriptor = TraceDescriptor{};
        return;
    case OpKind::Gemm: {
        GemmDescriptor desc;
        desc.alpha      = scalar("alpha", PrefactorScalar{double{1}});
        desc.beta       = scalar("beta", PrefactorScalar{double{0}});
        desc.trans_a    = transpose_char("trans_a", 'n');
        desc.trans_b    = transpose_char("trans_b", 'n');
        node.descriptor = desc;
        return;
    }
    case OpKind::Syev: {
        SyevDescriptor desc;
        // REQUIRED, not defaulted, even though the field arrived with this kind. No file
        // predates it: no earlier build could write a Syev node at all, so an absent key is a
        // malformed file rather than an older one, and the two LAPACK jobs leave A holding
        // different matrices. Guessing the default would silently change what a graph computes.
        desc.compute_eigenvectors = read_bool(*object, "compute_eigenvectors", path, problems, value.position);
        node.descriptor           = desc;
        return;
    }
    case OpKind::ElementTransform: {
        ElementTransformDescriptor desc;
        desc.op_name = read_string(*object, "op", path, problems, value.position);
        if (!desc.op_name.empty() && !element_ops::global_element_op_registry().contains(desc.op_name)) {
            note(problems, fmt::format("{}.op", path), value.position,
                 fmt::format("element op '{}' is not registered in this process. Registered: [{}]", desc.op_name,
                             fmt::join(element_ops::global_element_op_registry().names(), ", ")));
        }
        // OPTIONAL, and absent means the op's documented default rather than
        // zero: an older file predates the key entirely, and a node written by
        // this build omits it whenever the capture site named no number.
        if (Value const *param = object->take("param"); param != nullptr) {
            if (auto const parsed = tagged_number(*param); parsed.has_value()) {
                desc.param = *parsed;
            } else {
                note(problems, fmt::format("{}.param", path), param->position, "expected a number");
            }
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::WriteParam: {
        WriteParamDescriptor desc;
        desc.name        = read_string(*object, "param", path, problems, value.position);
        desc.source_type = read_named<ParamSourceType>(*object, "source_type", path, problems, value.position, param_source_type_from_name,
                                                       "parameter source type", ParamSourceType::Int64);
        if (Value const *expr = object->take("source_expr"); expr != nullptr) {
            desc.source_expr = read_bound_expr(*expr, fmt::format("{}.source_expr", path), problems);
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Conditional: {
        ConditionalDescriptor desc;
        if (Value const *predicate = field(*object, "predicate", path, problems, value.position); predicate != nullptr) {
            desc.predicate = read_pred_expr(*predicate, fmt::format("{}.predicate", path), problems, gates);
        }
        if (Value const *then_branch = field(*object, "then", path, problems, value.position); then_branch != nullptr) {
            node.then_branch =
                std::make_shared<IrFragment>(read_fragment(*then_branch, fmt::format("{}.then", path), problems, gates, registry));
        }
        if (Value const *else_branch = field(*object, "else", path, problems, value.position);
            else_branch != nullptr && !else_branch->is_null()) {
            node.else_branch =
                std::make_shared<IrFragment>(read_fragment(*else_branch, fmt::format("{}.else", path), problems, gates, registry));
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Loop: {
        LoopDescriptor desc;
        auto const     limit = read_int(*object, "max_iterations", path, problems, value.position);
        if (limit < 0) {
            note(problems, path, value.position, "max_iterations must not be negative");
        }
        desc.max_iterations = static_cast<std::size_t>(std::max<std::int64_t>(limit, 0));
        if (Value const *condition = field(*object, "condition", path, problems, value.position); condition != nullptr) {
            desc.condition = read_pred_expr(*condition, fmt::format("{}.condition", path), problems, gates);
        }
        if (Value const *body = field(*object, "body", path, problems, value.position); body != nullptr) {
            node.body = std::make_shared<IrFragment>(read_fragment(*body, fmt::format("{}.body", path), problems, gates, registry));
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Setup: {
        SetupDescriptor desc;
        if (Value const *body = field(*object, "body", path, problems, value.position); body != nullptr) {
            node.body = std::make_shared<IrFragment>(read_fragment(*body, fmt::format("{}.body", path), problems, gates, registry));
        }
        node.descriptor = std::move(desc);
        return;
    }
    default:
        note(problems, path, value.position,
             fmt::format("op kind '{}' is not one this schema can describe; the reconstructible set is what a file may contain",
                         op_kind_name(node.kind)));
        return;
    }
}

// NOLINTNEXTLINE(misc-no-recursion): see read_descriptor.
IrNode read_node(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                 SpaceRegistry const &registry) {
    IrNode        out;
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    out.id    = static_cast<std::size_t>(std::max<std::int64_t>(read_int(*object, "id", path, problems, value.position), 0));
    out.kind  = read_named<OpKind>(*object, "kind", path, problems, value.position, op_kind_from_name, "op kind", OpKind::Custom);
    out.label = read_string(*object, "label", path, problems, value.position);

    auto const ids = [&](std::string_view key) {
        std::vector<std::size_t> list;
        Value const             *entry = field(*object, key, path, problems, value.position);
        if (entry == nullptr) {
            return list;
        }
        std::string const child = fmt::format("{}.{}", path, key);
        Array const      *items = as_array(*entry, child, problems);
        if (items == nullptr) {
            return list;
        }
        for (std::size_t i = 0; i < items->size(); ++i) {
            if (!(*items)[i].is_int() || (*items)[i].as_int() < 0) {
                note(problems, fmt::format("{}[{}]", child, i), (*items)[i].position, "expected a non-negative tensor id");
                list.push_back(0);
                continue;
            }
            list.push_back(static_cast<std::size_t>((*items)[i].as_int()));
        }
        return list;
    };
    out.inputs  = ids("inputs");
    out.outputs = ids("outputs");
    out.dtype   = read_named<packed_gemm::ScalarType>(*object, "dtype", path, problems, value.position, scalar_type_from_name, "dtype",
                                                    packed_gemm::ScalarType::Unknown);
    out.rank    = static_cast<std::size_t>(std::max<std::int64_t>(read_int(*object, "rank", path, problems, value.position), 0));

    if (Value const *descriptor = field(*object, "descriptor", path, problems, value.position); descriptor != nullptr) {
        read_descriptor(out, *descriptor, fmt::format("{}.descriptor", path), problems, gates, registry);
    }
    return out;
}

// NOLINTNEXTLINE(misc-no-recursion): fragments nest.
IrFragment read_fragment(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                         SpaceRegistry const &registry) {
    IrFragment    out;
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    out.name = read_string(*object, "name", path, problems, value.position);
    if (Value const *tensors = field(*object, "tensors", path, problems, value.position); tensors != nullptr) {
        std::string const child = fmt::format("{}.tensors", path);
        if (Array const *items = as_array(*tensors, child, problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.tensors.push_back(read_tensor((*items)[i], fmt::format("{}[{}]", child, i), problems, /*manifest_entry=*/false));
            }
        }
    }
    if (Value const *nodes = field(*object, "nodes", path, problems, value.position); nodes != nullptr) {
        std::string const child = fmt::format("{}.nodes", path);
        if (Array const *items = as_array(*nodes, child, problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.nodes.push_back(read_node((*items)[i], fmt::format("{}[{}]", child, i), problems, gates, registry));
            }
        }
    }
    return out;
}

/// Compare two ``major.minor.patch`` strings.
/// @return -1, 0 or 1, or nullopt when either is not a semver triple.
std::optional<int> compare_semver(std::string_view lhs, std::string_view rhs) {
    auto const split = [](std::string_view text) -> std::optional<std::array<long, 3>> {
        std::array<long, 3> parts{};
        std::size_t         start = 0;
        for (std::size_t part = 0; part < 3; ++part) {
            std::size_t const dot = part < 2 ? text.find('.', start) : text.size();
            if (dot == std::string_view::npos || dot == start) {
                return std::nullopt;
            }
            std::string_view const field_text = text.substr(start, dot - start);
            long                   value      = 0;
            for (char const digit : field_text) {
                if (digit < '0' || digit > '9') {
                    return std::nullopt;
                }
                value = value * 10 + (digit - '0');
            }
            parts[part] = value;
            start       = dot + 1;
        }
        return parts;
    };
    auto const left  = split(lhs);
    auto const right = split(rhs);
    if (!left.has_value() || !right.has_value()) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < 3; ++i) {
        if ((*left)[i] != (*right)[i]) {
            return (*left)[i] < (*right)[i] ? -1 : 1;
        }
    }
    return 0;
}

/// Read the whole document. Reports every problem it can see; a document that
/// reports none is one the builder may run on.
IrDocument read_document(Value const &root, Problems &problems, SpaceRegistry const &registry) {
    IrDocument    out;
    Object const *object = as_object(root, "$", problems);
    if (object == nullptr) {
        return out;
    }

    // The version gate runs FIRST and, when it refuses, nothing else is read:
    // every message a newer schema would produce would be about fields this
    // build does not understand, which buries the one message that matters.
    out.version = read_string(*object, key_version, "$", problems, root.position);
    if (out.version.empty()) {
        return out;
    }
    auto const order = compare_semver(out.version, graph_ir_schema_version);
    if (!order.has_value()) {
        note(problems, "$", root.position, fmt::format("'{}' is not a major.minor.patch schema version", out.version));
        return out;
    }
    if (*order > 0) {
        note(problems, "$", root.position,
             fmt::format("this file is einsums_graph_ir {} and this build understands up to {}; a newer build reads an older IR, "
                         "never the reverse",
                         out.version, graph_ir_schema_version));
        return out;
    }

    // Provenance is DATA. Every field is consumed so the strict audit passes,
    // and not one of them is acted on.
    if (Value const *provenance = field(*object, key_provenance, "$", problems, root.position); provenance != nullptr) {
        if (Object const *body = as_object(*provenance, "$.provenance", problems); body != nullptr) {
            body->mark_consumed("library_version");
            body->mark_consumed("config_fingerprint");
            // Read rather than merely consumed, and reading is not acting: nothing behaves
            // differently for what is in here. A malformed entry is skipped rather than
            // failing the load, because provenance is not structure and a file whose history
            // is unreadable still describes a perfectly good graph.
            if (Value const *passes = body->take("structural_passes"); passes != nullptr && passes->is_array()) {
                for (auto const &entry : passes->as_array()) {
                    if (entry.is_string()) {
                        out.structural_passes.push_back(entry.as_string());
                    }
                }
            }
        }
    }

    out.name = read_string(*object, "name", "$", problems, root.position);

    // Gate-flag declarations are read before anything that can reference one.
    if (Value const *gates = field(*object, "gate_flags", "$", problems, root.position); gates != nullptr) {
        if (Array const *items = as_array(*gates, "$.gate_flags", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.gate_flags[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                std::string const name = read_string(*entry, "name", path, problems, (*items)[i].position);
                auto const        size = read_int(*entry, "size", path, problems, (*items)[i].position);
                if (size < 0) {
                    note(problems, path, (*items)[i].position, "a gate-flag array size must not be negative");
                    continue;
                }
                out.gate_flags.emplace_back(name, static_cast<std::size_t>(size));
            }
        }
    }

    for (auto const &[name, size] : out.gate_flags) {
        out.gate_buffers.emplace(name, std::make_shared<std::vector<std::uint8_t>>(size, std::uint8_t{0}));
    }
    GateFlagTable const &gates = out.gate_buffers;

    if (Value const *manifest = field(*object, "manifest", "$", problems, root.position); manifest != nullptr) {
        if (Array const *items = as_array(*manifest, "$.manifest", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.manifest.push_back(read_tensor((*items)[i], fmt::format("$.manifest[{}]", i), problems, /*manifest_entry=*/true));
            }
        }
    }

    if (Value const *spaces = field(*object, "spaces", "$", problems, root.position); spaces != nullptr) {
        if (Object const *body = as_object(*spaces, "$.spaces", problems); body != nullptr) {
            out.space_names = read_string_array(*body, "names", "$.spaces", problems, spaces->position);
            for (auto const &name : out.space_names) {
                if (!registry.find(name).has_value()) {
                    std::vector<std::string> known;
                    for (SpaceId const id : registry.ids()) {
                        known.push_back(registry.space(id).name);
                    }
                    note(problems, "$.spaces.names", spaces->position,
                         fmt::format("index space '{}' is not registered in this process. Registered: [{}]", name, fmt::join(known, ", ")));
                }
            }
            if (Value const *ties = field(*body, "symbol_ties", "$.spaces", problems, spaces->position); ties != nullptr) {
                if (Array const *items = as_array(*ties, "$.spaces.symbol_ties", problems); items != nullptr) {
                    for (std::size_t i = 0; i < items->size(); ++i) {
                        std::string const path  = fmt::format("$.spaces.symbol_ties[{}]", i);
                        Object const     *entry = as_object((*items)[i], path, problems);
                        if (entry == nullptr) {
                            continue;
                        }
                        out.symbol_ties.emplace_back(read_string(*entry, "symbol", path, problems, (*items)[i].position),
                                                     read_string(*entry, "space", path, problems, (*items)[i].position));
                    }
                }
            }
        }
    }

    // OPTIONAL, and the compatibility policy says an added field takes a documented default:
    // a file written before this section existed describes a graph nothing approximated,
    // which is an empty list and is exactly right.
    if (Value const *approximations = object->take("approximations"); approximations != nullptr) {
        if (Array const *items = as_array(*approximations, "$.approximations", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.approximations[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                ApproximationRecord record;
                record.pass_name = read_string(*entry, "pass_name", path, problems, (*items)[i].position);
                // The effect has no safe default: a bound whose units are unreadable cannot be
                // composed or compared, and guessing one would silently produce a number in
                // the wrong scale. read_named reports the unresolvable name and the load fails.
                record.effect =
                    read_named<ApproximationEffect>(*entry, "effect", path, problems, (*items)[i].position, approximation_effect_from_name,
                                                    "approximation effect", ApproximationEffect::NormRelative);
                if (Value const *tolerance = field(*entry, "tolerance", path, problems, (*items)[i].position); tolerance != nullptr) {
                    if (auto const parsed = tagged_number(*tolerance); parsed.has_value()) {
                        record.tolerance = *parsed;
                    } else {
                        note(problems, fmt::format("{}.tolerance", path), tolerance->position, "expected a number");
                    }
                }
                if (Value const *bound = field(*entry, "bound", path, problems, (*items)[i].position); bound != nullptr) {
                    if (auto const parsed = tagged_number(*bound); parsed.has_value()) {
                        record.bound = *parsed;
                    } else {
                        note(problems, fmt::format("{}.bound", path), bound->position, "expected a number");
                    }
                }
                // OPTIONAL, and defaulted to asserted rather than measured. A file written
                // before this key existed carries a number whose provenance nobody recorded,
                // and reading it as evidence would promote a guess by nothing more than a
                // newer build having opened it.
                record.origin  = read_named_optional<ApproximationOrigin>(*entry, "origin", path, problems, approximation_origin_from_name,
                                                                         "approximation origin", ApproximationOrigin::Asserted);
                record.outputs = read_string_array(*entry, "outputs", path, problems, (*items)[i].position);
                record.spaces  = read_string_array(*entry, "spaces", path, problems, (*items)[i].position);
                record.setup   = read_string(*entry, "setup", path, problems, (*items)[i].position);
                out.approximations.push_back(std::move(record));
            }
        }
    }

    if (Value const *params = field(*object, "params", "$", problems, root.position); params != nullptr) {
        if (Array const *items = as_array(*params, "$.params", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.params[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                out.params.emplace_back(read_string(*entry, "name", path, problems, (*items)[i].position),
                                        read_int(*entry, "value", path, problems, (*items)[i].position));
            }
        }
    }

    if (Value const *tensors = field(*object, "tensors", "$", problems, root.position); tensors != nullptr) {
        if (Array const *items = as_array(*tensors, "$.tensors", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.tensors.push_back(read_tensor((*items)[i], fmt::format("$.tensors[{}]", i), problems, /*manifest_entry=*/false));
            }
        }
    }

    if (Value const *redirects = field(*object, "slot_redirects", "$", problems, root.position); redirects != nullptr) {
        if (Array const *items = as_array(*redirects, "$.slot_redirects", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.slot_redirects[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                out.slot_redirects.emplace_back(
                    static_cast<std::size_t>(std::max<std::int64_t>(read_int(*entry, "from", path, problems, (*items)[i].position), 0)),
                    static_cast<std::size_t>(std::max<std::int64_t>(read_int(*entry, "to", path, problems, (*items)[i].position), 0)));
            }
        }
    }

    if (Value const *nodes = field(*object, "nodes", "$", problems, root.position); nodes != nullptr) {
        if (Array const *items = as_array(*nodes, "$.nodes", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.nodes.push_back(read_node((*items)[i], fmt::format("$.nodes[{}]", i), problems, gates, registry));
            }
        }
    }
    return out;
}

// ── Builder ────────────────────────────────────────────────────────────────

/// Thrown when a structurally valid document still cannot be turned into a
/// graph. Caught at the API boundary.
struct BuildFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// One frame's loaded storage: the dense id's tensor object, its id in this
/// frame's graph, and whether it is a scalar.
struct LoadedTensor {
    void    *object{nullptr};
    TensorId id{0};
    bool     scalar{false};
};

/// Allocate placeholder storage for one tensor and register it.
///
/// The storage is the graph's own, and every deleter is adopted by the ROOT
/// graph so a body's tensors outlive the fragment that named them. A caller
/// replaces it slot by slot with @ref Graph::bind; a slot nobody binds still has
/// something to execute against, which is what keeps an unbound loaded graph
/// from dereferencing nothing.
template <typename T>
LoadedTensor allocate_tensor(Graph &root, Graph &graph, IrTensor const &spec) {
    if (spec.rank == 0) {
        auto *scalar = new T{};
        root.adopt([scalar]() { delete scalar; });
        TensorId const id = graph.register_tensor(make_scalar_handle(scalar, 0, spec.name));
        return LoadedTensor{.object = static_cast<void *>(scalar), .id = id, .scalar = true};
    }
    using TensorType = GeneralRuntimeTensor<T, std::allocator<T>>;

    // A tensor the file says is DEFERRED comes back deferred, with the lifecycle hooks its
    // declaration installed. Allocating it here instead would look harmless (the pass would
    // just find it already materialized) and would quietly cost the graph the one thing the
    // save exists to enable: only a deferred intermediate can be resized by a bind, so a
    // materialized one refuses the move to a different-sized problem.
    bool const deferred = spec.alloc == AllocState::Deferred;
    auto      *tensor =
        deferred ? new TensorType(typename TensorType::DeferredAlloc{}, spec.name, spec.dims) : new TensorType(spec.name, spec.dims);
    if (!deferred) {
        tensor->zero();
    }
    root.adopt([tensor]() { delete tensor; });
    auto handle            = make_handle(*tensor, 0);
    handle.is_intermediate = spec.intermediate;
    handle.ownership       = spec.scope;
    handle.init_kind       = spec.init;
    if (deferred) {
        handle.alloc_state        = AllocState::Deferred;
        handle.materialize_fn     = [tensor]() { tensor->materialize(); };
        handle.release_fn         = [tensor]() { tensor->release(); };
        handle.is_materialized_fn = [tensor]() { return tensor->is_materialized(); };
        handle.resize_deferred_fn = [tensor](std::vector<size_t> const &new_dims) { tensor->resize_deferred(new_dims); };
        handle.zero_fn            = make_zero_fn(tensor);
    }
    TensorId const id = graph.register_tensor(std::move(handle));
    graph.get_or_create_slot(*tensor, id);
    return LoadedTensor{.object = static_cast<void *>(tensor), .id = id, .scalar = false};
}

/// Register a handle in @p graph over storage an enclosing frame already owns.
template <typename T>
LoadedTensor adopt_outer_tensor(Graph &graph, IrTensor const &spec, LoadedTensor const &outer) {
    if (outer.scalar) {
        TensorId const id = graph.register_tensor(make_scalar_handle(static_cast<T *>(outer.object), 0, spec.name));
        return LoadedTensor{.object = outer.object, .id = id, .scalar = true};
    }
    using TensorType       = GeneralRuntimeTensor<T, std::allocator<T>>;
    auto *tensor           = static_cast<TensorType *>(outer.object);
    auto  handle           = make_handle(*tensor, 0);
    handle.is_intermediate = spec.intermediate;
    handle.ownership       = spec.scope;
    handle.init_kind       = spec.init;
    TensorId const id      = graph.register_tensor(std::move(handle));
    graph.get_or_create_slot(*tensor, id);
    return LoadedTensor{.object = outer.object, .id = id, .scalar = false};
}

/// Dispatch @p spec's dtype and allocate or adopt accordingly.
LoadedTensor materialize_tensor(Graph &root, Graph &graph, IrTensor const &spec, LoadedTensor const *outer) {
    auto const apply = [&]<typename T>(T /*tag*/) {
        return outer != nullptr ? adopt_outer_tensor<T>(graph, spec, *outer) : allocate_tensor<T>(root, graph, spec);
    };
    switch (spec.dtype) {
    case packed_gemm::ScalarType::Float32:
        return apply(float{});
    case packed_gemm::ScalarType::Float64:
        return apply(double{});
    case packed_gemm::ScalarType::Complex64:
        return apply(std::complex<float>{});
    case packed_gemm::ScalarType::Complex128:
        return apply(std::complex<double>{});
    default:
        throw BuildFailure(fmt::format("tensor '{}' has dtype 'unknown', which names no storage the loader can allocate", spec.name));
    }
}

std::vector<LoadedTensor> build_frame(Graph &root, Graph &graph, std::vector<IrTensor> const &tensors, std::vector<IrNode> const &nodes,
                                      std::vector<LoadedTensor> const *parent, GateFlagTable const &gates, SpaceRegistry const &registry);

/// Turn one fragment into a live sub-graph.
///
/// The body shares the root's @ref ParamTable, which is what a captured body
/// does: a ``BoundExpr::Param`` inside a loop resolves against the pipeline's
/// parameters, not against a table private to the iteration.
// NOLINTNEXTLINE(misc-no-recursion): fragments nest.
std::shared_ptr<Graph> build_fragment(Graph &root, IrFragment const &fragment, std::vector<LoadedTensor> const &parent,
                                      GateFlagTable const &gates, SpaceRegistry const &registry) {
    auto body = std::make_shared<Graph>(fragment.name);
    body->set_params_ptr(root.params_ptr());
    build_frame(root, *body, fragment.tensors, fragment.nodes, &parent, gates, registry);
    return body;
}

/// Materialize one frame's tensors, rebuild its nodes, and recurse into any
/// fragment a control-flow node names.
// NOLINTNEXTLINE(misc-no-recursion): see build_fragment.
std::vector<LoadedTensor> build_frame(Graph &root, Graph &graph, std::vector<IrTensor> const &tensors, std::vector<IrNode> const &nodes,
                                      std::vector<LoadedTensor> const *parent, GateFlagTable const &gates, SpaceRegistry const &registry) {
    std::vector<LoadedTensor> loaded(tensors.size());
    for (auto const &spec : tensors) {
        if (spec.id >= tensors.size()) {
            throw BuildFailure(
                fmt::format("tensor id {} is outside the frame's {} tensors; ids must be dense from 0", spec.id, tensors.size()));
        }
        if (loaded[spec.id].id != 0) {
            throw BuildFailure(fmt::format("two tensors both claim id {}", spec.id));
        }
        LoadedTensor const *outer = nullptr;
        if (spec.outer.has_value()) {
            if (parent == nullptr) {
                throw BuildFailure(
                    fmt::format("tensor '{}' declares an outer reference, but the top-level frame has no enclosing one", spec.name));
            }
            if (*spec.outer >= parent->size()) {
                throw BuildFailure(
                    fmt::format("tensor '{}' references outer id {}, which the enclosing frame does not define", spec.name, *spec.outer));
            }
            outer = &(*parent)[*spec.outer];
        }
        loaded[spec.id] = materialize_tensor(root, graph, spec, outer);
    }

    // Annotations, spaces before symbols so a (symbol, space) tie is recorded
    // from whichever side names it first.
    for (auto const &spec : tensors) {
        if (spec.spaces.empty()) {
            continue;
        }
        std::vector<SpaceId> ids;
        ids.reserve(spec.spaces.size());
        for (auto const &name : spec.spaces) {
            auto const id = registry.find(name);
            if (!id.has_value()) {
                throw BuildFailure(fmt::format("tensor '{}' names index space '{}', which is not registered", spec.name, name));
            }
            ids.push_back(*id);
        }
        graph.annotate_spaces(loaded[spec.id].id, std::move(ids));
        // annotate_spaces records a DECLARATION; the file says whether the
        // original was one, and a derived annotation must read back as derived
        // so a validation pass still reports the weaker verdict on it.
        graph.tensor(loaded[spec.id].id).spaces_inferred = spec.spaces_inferred;
    }
    for (auto const &spec : tensors) {
        if (!spec.dim_symbols.empty()) {
            graph.annotate_dims(loaded[spec.id].id, spec.dim_symbols);
        }
    }

    // Provenance last, and independent of the two above: a tag is a statement about what the
    // tensor IS, and it neither constrains nor is constrained by extents or spaces.
    for (auto const &spec : tensors) {
        if (spec.tag.valid()) {
            graph.annotate_tag(loaded[spec.id].id, spec.tag);
        }
    }

    for (auto const &spec : nodes) {
        Node node;
        node.id    = graph.reserve_node_id();
        node.kind  = spec.kind;
        node.label = spec.label;

        auto const remap = [&](std::vector<std::size_t> const &dense, char const *role) {
            std::vector<TensorId> out;
            out.reserve(dense.size());
            for (auto const id : dense) {
                if (id >= loaded.size()) {
                    throw BuildFailure(
                        fmt::format("node '{}' names {} tensor id {}, which the frame does not define", spec.label, role, id));
                }
                out.push_back(loaded[id].id);
            }
            return out;
        };
        node.inputs  = remap(spec.inputs, "input");
        node.outputs = remap(spec.outputs, "output");
        node.op_data = spec.descriptor;

        if (auto *einsum = std::get_if<EinsumDescriptor>(&node.op_data)) {
            // The live blocks a captured node carries. Restoring them is not
            // cosmetic: a loaded graph is optimized after loading, and a pass
            // that rewrites a prefactor writes through this handle.
            einsum->params         = graph.create_params(einsum->c_prefactor, einsum->ab_prefactor);
            einsum->params->conj_a = einsum->conj_a;
            einsum->params->conj_b = einsum->conj_b;
            einsum->indices =
                graph.create_indices(einsum->spec.a_indices, einsum->spec.b_indices, einsum->spec.c_indices, einsum->spec.link_indices);
            einsum->indices->spec.conj_a = einsum->conj_a;
            einsum->indices->spec.conj_b = einsum->conj_b;
            // Regenerated exactly as build_executor regenerates it for a node
            // with no live block, so a loaded node's diagnostics read the same.
            einsum->indices->spec.raw = einsum->indices->spec.render();
            einsum->site              = std::make_shared<packed_gemm::ContractionSite>();
            if (einsum->gemm_hint != nullptr) {
                if (spec.hint_ids.size() != 3) {
                    throw BuildFailure(
                        fmt::format("node '{}' carries a GEMM hint with {} operands, not three", spec.label, spec.hint_ids.size()));
                }
                auto const hint_id = [&](std::size_t dense) {
                    if (dense >= loaded.size()) {
                        throw BuildFailure(
                            fmt::format("node '{}' GEMM hint names tensor id {}, which the frame does not define", spec.label, dense));
                    }
                    return loaded[dense].id;
                };
                einsum->gemm_hint->a.id = hint_id(spec.hint_ids[0]);
                einsum->gemm_hint->b.id = hint_id(spec.hint_ids[1]);
                einsum->gemm_hint->c.id = hint_id(spec.hint_ids[2]);
                switch (spec.dtype) {
                case packed_gemm::ScalarType::Float32:
                    einsum->gemm_hint->scalar = BlasScalar::Float;
                    break;
                case packed_gemm::ScalarType::Float64:
                    einsum->gemm_hint->scalar = BlasScalar::Double;
                    break;
                case packed_gemm::ScalarType::Complex64:
                    einsum->gemm_hint->scalar = BlasScalar::ComplexFloat;
                    break;
                case packed_gemm::ScalarType::Complex128:
                    einsum->gemm_hint->scalar = BlasScalar::ComplexDouble;
                    break;
                default:
                    throw BuildFailure(fmt::format("node '{}' carries a GEMM hint with dtype 'unknown'", spec.label));
                }
            }
        }
        if (auto *conditional = std::get_if<ConditionalDescriptor>(&node.op_data)) {
            if (spec.then_branch == nullptr) {
                throw BuildFailure(fmt::format("conditional node '{}' has no then-branch", spec.label));
            }
            conditional->then_branch = build_fragment(root, *spec.then_branch, loaded, gates, registry);
            conditional->else_branch = spec.else_branch != nullptr ? build_fragment(root, *spec.else_branch, loaded, gates, registry)
                                                                   : std::make_shared<Graph>("else");
        }
        if (auto *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
            if (spec.body == nullptr) {
                throw BuildFailure(fmt::format("loop node '{}' has no body", spec.label));
            }
            loop->body  = build_fragment(root, *spec.body, loaded, gates, registry);
            loop->state = std::make_shared<LoopState>();
        }
        if (auto *setup = std::get_if<SetupDescriptor>(&node.op_data)) {
            if (spec.body == nullptr) {
                throw BuildFailure(fmt::format("setup node '{}' has no body", spec.label));
            }
            setup->body = build_fragment(root, *spec.body, loaded, gates, registry);
            // Fresh, so a loaded graph knows it has fitted nothing. Anything else would be
            // a file claiming the factors are on hand in a process that has never bound.
            setup->state = std::make_shared<SetupState>();
        }

        try {
            node.execute = build_executor(spec.kind, spec.dtype, spec.rank, node.op_data, graph, node.inputs, node.outputs);
        } catch (std::exception const &error) {
            throw BuildFailure(fmt::format("node '{}' ({}): {}", spec.label, op_kind_name(spec.kind), error.what()));
        }
        graph.add_node(std::move(node));
    }
    return loaded;
}

/// Turn a clean document into a graph.
Graph build_graph(IrDocument const &document, SpaceRegistry &registry) {
    Graph graph(document.name);

    // Before anything is annotated. The ids about to be resolved come from THIS registry, and
    // a graph reading its spaces back through a different one would resolve them to whatever
    // happens to sit at those indices there.
    graph.set_space_registry(registry);

    // Exactly the buffers the predicates read out of the document already hold.
    for (auto const &[name, size] : document.gate_flags) {
        auto const found = document.gate_buffers.find(name);
        if (found == document.gate_buffers.end()) {
            throw BuildFailure(fmt::format("gate-flag array '{}' was declared but no buffer was made for it", name));
        }
        graph.name_gate_flags(name, found->second);
    }

    for (auto const &[name, value] : document.params) {
        graph.params_ptr()->set(name, value);
    }

    // The frame's tensor list is the manifest entries and the intermediates,
    // placed at their dense ids.
    std::vector<IrTensor> tensors(document.manifest.size() + document.tensors.size());
    auto const            place = [&tensors](IrTensor const &entry) {
        if (entry.id >= tensors.size()) {
            throw BuildFailure(
                fmt::format("tensor '{}' claims id {}, past the {} tensors this file defines", entry.name, entry.id, tensors.size()));
        }
        tensors[entry.id] = entry;
    };
    for (auto const &entry : document.manifest) {
        place(entry);
    }
    for (auto const &entry : document.tensors) {
        place(entry);
    }

    std::vector<LoadedTensor> const loaded = build_frame(graph, graph, tensors, document.nodes, nullptr, document.gate_buffers, registry);

    // The history the file carries, put back on the graph. Nothing acts on it; it is here so a
    // load, further optimization and a re-save produce a file that still names everything that
    // shaped this graph rather than only what the second pipeline did.
    for (auto const &pass_name : document.structural_passes) {
        graph.note_structural_pass(pass_name);
    }

    // The accuracy history, which unlike the pass list IS acted on: a lossy pass applied to
    // the loaded graph composes against these, and a differential comparison widens by them.
    // Installed directly rather than through note_approximation, which would re-run
    // can_approximate against a budget this process has not set and could refuse a record
    // that was perfectly legal in the process that wrote it. A file is a statement of what
    // WAS applied, not a request to apply it.
    graph.restore_approximations(document.approximations);

    // Slot redirects, once every slot exists. Not a derived cache: a node whose
    // dataflow still names a merged-away duplicate reads the survivor's buffer
    // only because of these.
    for (auto const &[from, to] : document.slot_redirects) {
        if (from >= loaded.size() || to >= loaded.size()) {
            throw BuildFailure(fmt::format("slot redirect {} -> {} names a tensor this file does not define", from, to));
        }
        graph.redirect_slot(loaded[from].id, loaded[to].id);
    }

    // Manifest-declared aliases, by name. This is the only aliasing relation
    // that survives a save at all - a pair that shares storage with no ``View``
    // node recording it exists solely as two addresses that happen to coincide,
    // and a file has neither.
    std::unordered_map<std::string, TensorId> by_name;
    for (auto const &entry : document.manifest) {
        by_name.emplace(entry.name, loaded[entry.id].id);
    }
    for (auto const &entry : document.manifest) {
        if (entry.aliases_input.empty()) {
            continue;
        }
        auto const parent = by_name.find(entry.aliases_input);
        if (parent == by_name.end()) {
            throw BuildFailure(fmt::format("manifest entry '{}' declares itself an alias of '{}', which is not a manifest entry",
                                           entry.name, entry.aliases_input));
        }
        graph.declare_alias(loaded[entry.id].id, parent->second);
    }
    graph.link_alias_storage();

    // The manifest is DERIVED from what the nodes do, so the file's copy is a
    // declaration to check rather than state to install. A disagreement means
    // the node list and the interface describe different programs, which is
    // exactly the corruption a strict reader exists to catch.
    InterfaceManifest const contract = graph.manifest();
    for (auto const &entry : document.manifest) {
        ManifestEntry const *derived = contract.find(entry.name);
        if (derived == nullptr) {
            throw BuildFailure(
                fmt::format("the file declares interface tensor '{}', but no node in the rebuilt graph reads or writes it", entry.name));
        }
        if (derived->direction != entry.direction) {
            throw BuildFailure(fmt::format("interface tensor '{}' is declared '{}' but the rebuilt nodes make it '{}'", entry.name,
                                           manifest_direction_name(entry.direction), manifest_direction_name(derived->direction)));
        }
    }
    if (contract.size() != document.manifest.size()) {
        // Naming the difference, not just counting it. A mismatch here means the node list and
        // the interface describe different programs, and the only actionable part of that is
        // WHICH tensor appeared or vanished; a bare count sends the reader back to diff two
        // manifests by hand, which is what this check already did for them.
        std::vector<std::string> declared;
        for (auto const &entry : document.manifest) {
            declared.push_back(entry.name);
        }
        std::vector<std::string> derived = contract.names();
        std::ranges::sort(declared);
        std::ranges::sort(derived);
        std::vector<std::string> only_derived;
        std::ranges::set_difference(derived, declared, std::back_inserter(only_derived));
        std::vector<std::string> only_declared;
        std::ranges::set_difference(declared, derived, std::back_inserter(only_declared));
        throw BuildFailure(fmt::format(
            "the rebuilt graph has {} interface tensors but the file declares {}. Only in the rebuilt graph: [{}]. Only in the file: [{}]",
            contract.size(), document.manifest.size(), fmt::join(only_derived, ", "), fmt::join(only_declared, ", ")));
    }

    return graph;
}

} // namespace

// ── content_hash ───────────────────────────────────────────────────────────

std::uint64_t Graph::content_hash() const {
    std::string canonical;
    try {
        canonical = json::emit(Value{write_structure(*this)}, json::EmitOptions{.style = json::EmitStyle::Canonical});
    } catch (SaveRefusal const &refusal) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph::content_hash: {}", refusal.what());
    }
    // FNV-1a over the canonical bytes. Not a cryptographic digest and not meant
    // to be: the property wanted is that a structural change moves it, which is
    // what a differential test and a cache key need.
    std::uint64_t hash = 1469598103934665603ULL;
    for (char const byte : canonical) {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(byte));
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ── save ───────────────────────────────────────────────────────────────────

expected<std::string, GraphError> save_graph_string(Graph const &graph, SaveOptions const &options) {
    try {
        Value const document = write_document(graph, options);
        return json::emit(document, json::EmitOptions{.style = options.pretty ? json::EmitStyle::Pretty : json::EmitStyle::Canonical});
    } catch (SaveRefusal const &refusal) {
        return unexpected(GraphError::validation(fmt::format("save_graph: {}", refusal.what())));
    } catch (std::exception const &error) {
        return unexpected(GraphError::validation(fmt::format("save_graph: {}", error.what())));
    }
}

expected<void, GraphError> save_graph(Graph const &graph, std::string const &path, SaveOptions const &options) {
    auto text = save_graph_string(graph, options);
    if (!text) {
        return unexpected(text.error());
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return unexpected(GraphError::io(fmt::format("save_graph: cannot open '{}' for writing", path)));
    }
    file << *text;
    if (!file) {
        return unexpected(GraphError::io(fmt::format("save_graph: writing '{}' failed", path)));
    }
    return {};
}

// ── load and validate ──────────────────────────────────────────────────────

namespace {

/// Read @p path whole, or say why not.
expected<std::string, GraphError> read_file(std::string const &path, std::string_view who) {
    std::ifstream const file(path, std::ios::binary);
    if (!file) {
        return unexpected(GraphError::io(fmt::format("{}: cannot open '{}'", who, path)));
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (file.bad()) {
        return unexpected(GraphError::io(fmt::format("{}: reading '{}' failed", who, path)));
    }
    return buffer.str();
}

/// Parse and read @p text, returning the intermediate form and every problem.
///
/// The unconsumed-key audit runs LAST and over the whole document, because a
/// leftover key is only leftover once every reader has had its turn.
IrDocument inspect(std::string_view text, Problems &problems, json::Value &document, SpaceRegistry const &registry) {
    auto parsed = json::parse(text);
    if (!parsed) {
        problems.push_back(parsed.error().to_string());
        return {};
    }
    document       = std::move(*parsed);
    IrDocument out = read_document(document, problems, registry);

    std::vector<std::string> unconsumed;
    json::collect_unconsumed(document, "$", unconsumed);
    for (auto const &key : unconsumed) {
        problems.push_back(fmt::format("{}: this build does not understand that key. A field may be ADDED to the schema and a newer "
                                       "build then reads an older file, but a reader never silently ignores content",
                                       key));
    }
    return out;
}

} // namespace

expected<Graph, GraphError> load_graph_string(std::string_view text) {
    return load_graph_string(text, global_space_registry());
}

expected<Graph, GraphError> load_graph_string(std::string_view text, SpaceRegistry &registry) {
    Problems         problems;
    json::Value      document;
    IrDocument const ir = inspect(text, problems, document, registry);
    if (!problems.empty()) {
        return unexpected(GraphError::parse(fmt::format("load_graph: {}", problems.front())));
    }
    try {
        return build_graph(ir, registry);
    } catch (BuildFailure const &failure) {
        return unexpected(GraphError::validation(fmt::format("load_graph: {}", failure.what())));
    } catch (std::exception const &error) {
        return unexpected(GraphError::validation(fmt::format("load_graph: {}", error.what())));
    }
}

expected<Graph, GraphError> load_graph(std::string const &path) {
    return load_graph(path, global_space_registry());
}

expected<Graph, GraphError> load_graph(std::string const &path, SpaceRegistry &registry) {
    auto text = read_file(path, "load_graph");
    if (!text) {
        return unexpected(text.error());
    }
    auto graph = load_graph_string(*text, registry);
    if (!graph) {
        return unexpected(GraphError{.kind = graph.error().kind, .message = fmt::format("{} (reading '{}')", graph.error().message, path)});
    }
    return graph;
}

expected<void, GraphError> validate_graph_ir_string(std::string_view text) {
    return validate_graph_ir_string(text, global_space_registry());
}

expected<void, GraphError> validate_graph_ir_string(std::string_view text, SpaceRegistry &registry) {
    Problems         problems;
    json::Value      document;
    IrDocument const ir = inspect(text, problems, document, registry);
    if (problems.empty()) {
        // A document that reads clean still has to BUILD, and a build failure is
        // one more problem to report rather than a separate outcome.
        try {
            Graph const graph = build_graph(ir, registry);
            (void)graph;
        } catch (std::exception const &error) {
            problems.emplace_back(error.what());
        }
    }
    if (problems.empty()) {
        return {};
    }
    std::string message = fmt::format("validate_graph_ir: {} problem(s)", problems.size());
    for (auto const &problem : problems) {
        message += "\n  " + problem;
    }
    return unexpected(GraphError::parse(std::move(message)));
}

expected<void, GraphError> validate_graph_ir(std::string const &path) {
    auto text = read_file(path, "validate_graph_ir");
    if (!text) {
        return unexpected(text.error());
    }
    return validate_graph_ir_string(*text);
}

// ── The Python spelling ────────────────────────────────────────────────────

void save_graph_file(Graph const &graph, std::string const &path) {
    if (auto const done = save_graph(graph, path); !done) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", done.error().message);
    }
}

Graph *load_graph_file(std::string const &path) {
    auto graph = load_graph(path);
    if (!graph) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", graph.error().message);
    }
    // A raw pointer because the Python binding takes ownership of it; there is
    // no C++ caller for this overload.
    return new Graph(std::move(*graph)); // NOLINT(cppcoreguidelines-owning-memory)
}

Graph *load_graph_file_into(std::string const &path, SpaceRegistry &registry) {
    auto graph = load_graph(path, registry);
    if (!graph) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", graph.error().message);
    }
    // As load_graph_file: a raw pointer because the Python binding takes ownership.
    return new Graph(std::move(*graph)); // NOLINT(cppcoreguidelines-owning-memory)
}

void validate_graph_ir_file(std::string const &path) {
    if (auto const done = validate_graph_ir(path); !done) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", done.error().message);
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
