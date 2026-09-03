//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CSE.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

// The live scalar accessors this file used to define for itself (live_c_prefactor and friends)
// now live beside the descriptors they read, in Node.hpp: CSE was not the only caller that
// needed them, and two copies of "prefer the shared params over the snapshot" is one too many.

/// Do two einsum descriptors describe the same contraction topology?
///
/// Checks the at-capture ContractionSpec snapshot AND the live ParsedEinsumSpec
/// index lists the executor reads. An index rewriter (PermuteFusion) is
/// required to update both; comparing both means one that updated only one
/// cannot make two different contractions look alike here.
bool einsum_indices_equal(EinsumDescriptor const &a, EinsumDescriptor const &b) {
    if (!(a.spec == b.spec)) {
        return false;
    }
    if ((a.indices == nullptr) != (b.indices == nullptr)) {
        return false;
    }
    if (a.indices == nullptr) {
        return true;
    }
    auto const &sa = a.indices->spec;
    auto const &sb = b.indices->spec;
    return sa.c_indices == sb.c_indices && sa.a_indices == sb.a_indices && sa.b_indices == sb.b_indices && sa.conj_a == sb.conj_a &&
           sa.conj_b == sb.conj_b;
}

// ── Proportional matching ───────────────────────────────────────────────────

/// Whether @p v is exactly a power of two, sign included.
///
/// The r with `b == r * a`, when there is one that scales exactly.
/// r == 1 is the ordinary identical case and falls out of the same test.
std::optional<double> exact_ratio(double a, double b) {
    if (a == b) {
        return 1.0;
    }
    if (a == 0.0) {
        return std::nullopt;
    }
    double const r = b / a;
    if (!is_exact_power_of_two(r) || a * r != b) {
        return std::nullopt;
    }
    return r;
}

/// Same, for type-erased prefactors. Identical values match whatever their
/// alternative; a real ratio is only sought between two real-valued scalars,
/// so a complex pair still merges when it is an exact duplicate.
std::optional<double> exact_ratio(PrefactorScalar const &a, PrefactorScalar const &b) {
    if (a == b) {
        return 1.0;
    }
    if (!is_real_valued(a) || !is_real_valued(b)) {
        return std::nullopt;
    }
    return exact_ratio(as_real<double>(a), as_real<double>(b));
}

std::optional<double> exact_ratio(std::complex<double> a, std::complex<double> b) {
    if (a == b) {
        return 1.0;
    }
    if (a.imag() != 0.0 || b.imag() != 0.0) {
        return std::nullopt;
    }
    return exact_ratio(a.real(), b.real());
}

/// Every BatchedGemm field except the source prefactor. The strided flag and
/// its per-operand batch strides are part of the identity: pointer-array and
/// strided batches have different executor semantics even when the BLAS key
/// looks identical.
bool batched_gemm_shape_equal(BatchedGemmDescriptor const &a, BatchedGemmDescriptor const &b) {
    return a.m == b.m && a.n == b.n && a.k == b.k && a.lda == b.lda && a.ldb == b.ldb && a.ldc == b.ldc && a.trans_a == b.trans_a &&
           a.trans_b == b.trans_b && a.beta == b.beta && a.batch_count == b.batch_count && a.scalar == b.scalar && a.strided == b.strided &&
           a.batch_stride_a == b.batch_stride_a && a.batch_stride_b == b.batch_stride_b && a.batch_stride_c == b.batch_stride_c;
}

/// The factor r for which node @p b computes `r` times what node @p a computes,
/// or null when they are not the same computation up to a real scalar.
///
/// Everything except the one prefactor the result is linear in must match
/// exactly, the DESTINATION prefactor included: the caller checks
/// @ref pure_overwrite on the survivor alone, which only covers the duplicate
/// because the pair is required to agree there.
///
/// Descriptor kinds with no arm here (monostate, the tiled and view
/// descriptors, control flow) never match. Those nodes are not pure-overwrite
/// producers either, so this only restates the caller's gate.
std::optional<double> op_data_ratio(OpData const &a, OpData const &b) {
    if (a.index() != b.index()) {
        return std::nullopt;
    }

    if (auto const *ea = std::get_if<EinsumDescriptor>(&a)) {
        auto const &eb = std::get<EinsumDescriptor>(b);
        if (!(einsum_indices_equal(*ea, eb) && live_c_prefactor(*ea) == live_c_prefactor(eb) && live_conj_a(*ea) == live_conj_a(eb) &&
              live_conj_b(*ea) == live_conj_b(eb))) {
            return std::nullopt;
        }
        return exact_ratio(live_ab_prefactor(*ea), live_ab_prefactor(eb));
    }
    if (auto const *aa = std::get_if<AxpbyDescriptor>(&a)) {
        auto const &ab = std::get<AxpbyDescriptor>(b);
        if (live_beta(*aa) != live_beta(ab)) {
            return std::nullopt;
        }
        return exact_ratio(live_alpha(*aa), live_alpha(ab));
    }
    if (auto const *pa = std::get_if<PermuteDescriptor>(&a)) {
        auto const &pb = std::get<PermuteDescriptor>(b);
        // The index orders are half the operation: `C[j,i,k] = A[i,j,k]` and
        // `C[i,k,j] = A[i,j,k]` read the same source with the same scalars and
        // write the same shape, yet compute different transposes.
        if (!(pa->beta == pb.beta && pa->c_indices == pb.c_indices && pa->a_indices == pb.a_indices)) {
            return std::nullopt;
        }
        return exact_ratio(pa->alpha, pb.alpha);
    }
    if (auto const *ba = std::get_if<BatchedGemmDescriptor>(&a)) {
        auto const &bb = std::get<BatchedGemmDescriptor>(b);
        if (!batched_gemm_shape_equal(*ba, bb)) {
            return std::nullopt;
        }
        return exact_ratio(ba->alpha, bb.alpha);
    }
    if (auto const *sa = std::get_if<ScaleDescriptor>(&a)) {
        // In place, so never a pure-overwrite producer and never reached; the
        // exact comparison is kept so the variant is covered explicitly.
        return sa->factor == std::get<ScaleDescriptor>(b).factor ? std::optional<double>{1.0} : std::nullopt;
    }
    return std::nullopt;
}

// ── Moving the factor onto the readers ──────────────────────────────────────

/// Can @p nd absorb a real scalar into its read of @p tensor?
///
/// Only ops whose executor takes the factor from LIVE shared params qualify: an
/// einsum's ab_pf and an axpby's alpha. Permute and BatchedGemm bake their
/// scalars into the executor closure, so editing their descriptor would leave
/// the closure applying the old value - the same rule ScaleAbsorption follows.
///
/// The tensor must be exactly one operand and must not be the destination:
/// reading it twice would need r squared, and a destination read is an
/// accumulation the factor does not distribute over.
bool foldable_reader(Node const &nd, TensorId tensor) {
    if (std::ranges::find(nd.outputs, tensor) != nd.outputs.end()) {
        return false;
    }
    if (std::ranges::count(nd.inputs, tensor) != 1) {
        return false;
    }
    if (nd.kind == OpKind::Einsum) {
        auto const *d = std::get_if<EinsumDescriptor>(&nd.op_data);
        return d != nullptr && d->params != nullptr;
    }
    if (nd.kind == OpKind::Axpby) {
        auto const *d = std::get_if<AxpbyDescriptor>(&nd.op_data);
        return d != nullptr && d->params != nullptr;
    }
    return false;
}

/// Multiply @p nd's read of its operand by @p r. Caller guarantees
/// @ref foldable_reader. Writes the live params the executor reads and the
/// snapshot beside them, so later analysis sees the same value.
void fold_reader(Node &nd, double r) {
    if (auto *d = std::get_if<EinsumDescriptor>(&nd.op_data)) {
        d->params->ab_pf = scale_prefactor(d->params->ab_pf, r);
        d->ab_prefactor  = d->params->ab_pf;
        return;
    }
    auto *d          = std::get_if<AxpbyDescriptor>(&nd.op_data);
    d->params->alpha = scale_prefactor(d->params->alpha, r);
    d->alpha         = d->params->alpha;
}

} // namespace

namespace {

/// What a merge inside one graph of the tree needs to know about the rest of it.
///
/// CSE declines the pass manager's auto-recursion and walks the tree itself
/// (see CSE::run) precisely so this exists. Two facts are only answerable from
/// the root:
///
///   - **Is a buffer visible outside the graph being rewritten?** A merge
///     redirects readers with @ref Graph::redirect_slot, which repoints only
///     the slot table of the graph it is called on. A node in the parent, in a
///     sibling branch, or in a nested body reading the eliminated duplicate's
///     output would keep reading a buffer nothing writes any more.
///   - **Is a buffer a graph-owned intermediate?** A body registers its OWN
///     handle for a tensor its parent created, and that handle reports
///     `is_intermediate == false` even for parent-created scratch. Asking the
///     body would reject every candidate; the root's answer is the real one.
struct TreeContext {
    /// Buffer -> how many nodes touch its VALUE anywhere in the tree. Counted
    /// from RAW node I/O: every sub-graph's nodes are visited in their own
    /// right, so a control-flow node's raw (empty) I/O double-counts nothing.
    ///
    /// Lifecycle nodes are excluded, as they are from the writer count: an
    /// Alloc or Materialize in the parent for scratch only a body uses manages
    /// storage without observing the value, and counting it would make every
    /// body-local buffer look externally visible.
    std::unordered_map<void const *, size_t> touches_in_tree;
    /// Buffers that any graph in the tree calls a non-view graph-owned
    /// intermediate. Membership is what Guard C tests.
    std::unordered_set<void const *> intermediate_buffers;
};

void collect_tree_context(Graph const &graph, TreeContext &ctx) {
    auto const &tensors = graph.tensors_map();
    auto const  ptr_of  = [&](TensorId tid) -> void const  *{
        auto it = tensors.find(tid);
        return (it != tensors.end()) ? it->second.tensor_ptr : nullptr;
    };

    for (auto const &nd : graph.nodes()) {
        if (is_lifecycle(nd.kind))
            continue;
        for (auto const tid : nd.inputs) {
            if (auto const *p = ptr_of(tid))
                ctx.touches_in_tree[p]++;
        }
        for (auto const tid : nd.outputs) {
            if (auto const *p = ptr_of(tid))
                ctx.touches_in_tree[p]++;
        }
    }
    for (auto const &[tid, handle] : tensors) {
        if (handle.tensor_ptr != nullptr && handle.is_intermediate && handle.aliases == 0) {
            ctx.intermediate_buffers.insert(handle.tensor_ptr);
        }
    }

    graph.for_each_subgraph([&ctx](Graph const &sub) { collect_tree_context(sub, ctx); });
}

} // namespace

bool CSE::run_on_graph(Graph &graph, void const *tree_context, bool is_subgraph) {
    auto const &ctx = *static_cast<TreeContext const *>(tree_context);

    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        return false;
    }

    bool              modified = false;
    std::vector<bool> remove(nodes.size(), false);

    // For each pair, check if they compute the same expression.
    // Build a remapping of tensor IDs for eliminated nodes.
    std::unordered_map<TensorId, TensorId> tensor_redirect;

    // Resolve a TensorId in this graph to its underlying buffer pointer
    // (stable identity for a tensor; null when unresolved).
    auto ptr_of = [&](TensorId tid) -> void const * {
        auto it = graph.tensors_map().find(tid);
        return (it != graph.tensors_map().end()) ? it->second.tensor_ptr : nullptr;
    };

    // Count real (non-lifecycle) writers of each buffer across the graph.
    // CSE eliminates node j by redirecting readers of its output to node i's
    // output. That is only sound when the surviving buffer holds a *stable*
    // value: if anything writes node i's output again (e.g. a later in-place
    // scale), the redirected readers would observe the mutated value instead
    // of the common subexpression. So every output buffer involved in a merge
    // must have exactly one writer, the producing node itself.
    std::unordered_map<void const *, int> writer_count;
    for (auto const &nd : nodes) {
        if (is_lifecycle(nd.kind))
            continue;
        for (auto out : nd.outputs) {
            if (auto const *p = ptr_of(out))
                writer_count[p]++;
        }
    }

    // Buffers reached from inside a control-flow node's sub-graphs (Guard D).
    //
    // A Loop/Conditional node's own inputs/outputs say nothing about what its
    // body touches; Graph::effective_io is what reconstructs that. Neither the
    // writer count above nor the reader scan below sees a body, and
    // Graph::redirect_slot repoints only THIS graph's slot table, so a merge
    // involving a buffer a body reads leaves that body reading a never-written
    // buffer, and one a body writes hands redirected readers a mutated value.
    std::unordered_set<void const *> subgraph_touched;
    for (auto const &nd : nodes) {
        if (!is_control_flow(nd.kind))
            continue;
        auto const [sub_in, sub_out] = graph.effective_io(nd);
        for (auto tid : sub_in) {
            if (auto const *p = ptr_of(tid))
                subgraph_touched.insert(p);
        }
        for (auto tid : sub_out) {
            if (auto const *p = ptr_of(tid))
                subgraph_touched.insert(p);
        }
    }

    // How many of a buffer's tree-wide touches come from THIS graph's nodes.
    // A buffer touched more often elsewhere is visible outside, and
    // Graph::redirect_slot cannot reach those readers (Guard F).
    std::unordered_map<void const *, size_t> local_touches;
    for (auto const &nd : nodes) {
        if (is_lifecycle(nd.kind))
            continue;
        for (auto const tid : nd.inputs) {
            if (auto const *p = ptr_of(tid))
                local_touches[p]++;
        }
        for (auto const tid : nd.outputs) {
            if (auto const *p = ptr_of(tid))
                local_touches[p]++;
        }
    }
    auto const visible_outside_this_graph = [&](void const *p) {
        auto const total = ctx.touches_in_tree.find(p);
        if (total == ctx.touches_in_tree.end()) {
            return true; // unknown: assume the worst
        }
        auto const local = local_touches.find(p);
        return total->second > (local == local_touches.end() ? 0 : local->second);
    };

    // Candidates bucketed by what cheaply distinguishes them, so the scan does
    // not compare every pair.
    //
    // Walking the nodes once and matching each against the earlier SURVIVORS
    // that share a key is the same answer the pairwise scan gave - the earliest
    // equivalent node still wins - at a fraction of the work. It matters
    // because TiledExpansion emits thousands of nodes (its default budget is
    // 4096) where almost nothing matches, and the quadratic term dominated.
    //
    // The key deliberately excludes op_data: two nodes differing only in a
    // prefactor must land in the same bucket for the proportional merge below
    // to find them.
    //
    // Keying on the REDIRECTED inputs is well-defined even though the redirect
    // map grows as the scan runs. A redirect key is always the output of an
    // already-removed node, and a node's inputs only ever name outputs of nodes
    // before it, so by the time any node is keyed every redirect that could
    // affect it has been recorded. (This also fixes an asymmetry in the old
    // scan, which redirected the candidate's inputs but compared them against
    // the survivor's RAW inputs, and so missed matches whose shared operand had
    // itself been redirected.)
    struct CandidateKey {
        OpKind                kind{};
        std::vector<TensorId> inputs;
        size_t                num_outputs{0};

        bool operator==(CandidateKey const &o) const { return kind == o.kind && num_outputs == o.num_outputs && inputs == o.inputs; }
    };
    struct CandidateKeyHash {
        size_t operator()(CandidateKey const &k) const {
            size_t h = 0;
            hash_combine(h, static_cast<std::uint8_t>(k.kind));
            hash_combine(h, k.num_outputs);
            hash_range(h, k.inputs);
            return h;
        }
    };

    std::unordered_map<CandidateKey, std::vector<size_t>, CandidateKeyHash> buckets;

    for (size_t j = 0; j < nodes.size(); j++) {
        if (remove[j])
            continue;

        // Only pure-overwrite producers may be a CSE survivor/candidate. (Since
        // a matched pair must have equal op_data, checking one covers the other.)
        if (!pure_overwrite(nodes[j]))
            continue;

        CandidateKey key;
        key.kind        = nodes[j].kind;
        key.num_outputs = nodes[j].outputs.size();
        key.inputs      = nodes[j].inputs;
        for (auto &tid : key.inputs) {
            auto it = tensor_redirect.find(tid);
            if (it != tensor_redirect.end()) {
                tid = it->second;
            }
        }

        auto &bucket = buckets[key];
        bool  merged = false;

        for (size_t const i : bucket) {
            // The two nodes must compute the same thing up to one real scalar;
            // `ratio` is 1 for an outright duplicate.
            auto const ratio = op_data_ratio(nodes[i].op_data, nodes[j].op_data);
            if (!ratio)
                continue;

            // Guard C: the duplicate's outputs must be graph-owned
            // intermediates. A user-visible output is a contract - the user
            // reads that tensor directly, not through an executor slot, so
            // eliding its producer leaves it unwritten no matter how graph
            // consumers are redirected. (Folding such duplicates behind an
            // inserted copy node is possible future work.)
            //
            // Answered from the TREE, not from this graph: a loop body
            // registers its own handle for a tensor its parent created, and
            // that handle says `is_intermediate == false` even for
            // parent-created scratch. Asking the body would reject everything.
            bool duplicate_user_visible = false;
            for (auto out : nodes[j].outputs) {
                auto const *p = ptr_of(out);
                if (p == nullptr || ctx.intermediate_buffers.count(p) == 0) {
                    duplicate_user_visible = true;
                    break;
                }
            }
            // Inside a sub-graph the SURVIVOR must be graph-owned too. A loop's
            // predicate or DIIS callback runs between iterations and can write
            // user tensors from Python; nothing in the node list describes that,
            // so a user-visible survivor is a buffer this pass cannot reason
            // about. Graph-owned scratch is never what such a callback touches.
            if (!duplicate_user_visible && is_subgraph) {
                for (auto out : nodes[i].outputs) {
                    auto const *p = ptr_of(out);
                    if (p == nullptr || ctx.intermediate_buffers.count(p) == 0) {
                        duplicate_user_visible = true;
                        break;
                    }
                }
            }
            if (duplicate_user_visible)
                continue;

            // Guard F: neither output may be visible outside THIS graph.
            // Graph::redirect_slot repoints only this graph's slot table, so a
            // reader in the parent, a sibling branch, or a nested body would
            // keep reading the eliminated duplicate's now never-written buffer.
            bool escapes = false;
            for (auto const *outs : {&nodes[i].outputs, &nodes[j].outputs}) {
                for (auto out : *outs) {
                    auto const *p = ptr_of(out);
                    if (p == nullptr || visible_outside_this_graph(p)) {
                        escapes = true;
                        break;
                    }
                }
                if (escapes)
                    break;
            }
            if (escapes)
                continue;

            // Guard B: both producers' output buffers must be written exactly
            // once (by themselves). Otherwise redirecting readers onto a buffer
            // that gets mutated again would hand them the wrong value.
            bool single_writer = true;
            for (auto out : nodes[i].outputs) {
                auto const *p = ptr_of(out);
                if (p == nullptr || writer_count[p] != 1) {
                    single_writer = false;
                    break;
                }
            }
            if (single_writer) {
                for (auto out : nodes[j].outputs) {
                    auto const *p = ptr_of(out);
                    if (p == nullptr || writer_count[p] != 1) {
                        single_writer = false;
                        break;
                    }
                }
            }
            if (!single_writer)
                continue;

            // Guard D: neither output may be a buffer some control-flow node's
            // sub-graph touches (see subgraph_touched above).
            bool subgraph_reachable = false;
            for (auto out : nodes[i].outputs) {
                if (auto const *p = ptr_of(out); p != nullptr && subgraph_touched.count(p) > 0) {
                    subgraph_reachable = true;
                    break;
                }
            }
            if (!subgraph_reachable) {
                for (auto out : nodes[j].outputs) {
                    if (auto const *p = ptr_of(out); p != nullptr && subgraph_touched.count(p) > 0) {
                        subgraph_reachable = true;
                        break;
                    }
                }
            }
            if (subgraph_reachable)
                continue;

            // Guard A: the shared inputs must not be overwritten between i and
            // j. If some intervening node writes one of node i's inputs, then
            // node i (at its position) and node j (at its position) actually
            // read different values, so they are not the *same* computation at
            // runtime and the merge would reuse a stale result.
            std::unordered_set<void const *> input_ptrs;
            for (auto in : nodes[i].inputs) {
                if (auto const *p = ptr_of(in))
                    input_ptrs.insert(p);
            }
            bool inputs_stable = true;
            for (size_t k = i + 1; k < j && inputs_stable; k++) {
                if (remove[k] || is_lifecycle(nodes[k].kind))
                    continue;
                for (auto out : nodes[k].outputs) {
                    auto const *p = ptr_of(out);
                    if (p != nullptr && input_ptrs.count(p) > 0) {
                        inputs_stable = false;
                        break;
                    }
                }
            }
            if (!inputs_stable)
                continue;

            // Guard E, proportional duplicates only: node j's value is `r` times
            // node i's, so handing its readers node i's buffer also has to
            // multiply `r` into each of them. Every reader must be able to take
            // the factor, and the pair must have the single output the factor
            // attaches to. (A reader that cannot take it could instead be fed a
            // scaled copy of the survivor, but a copy is two sweeps over the
            // result and an outer product is barely more than one, so that trade
            // needs a cost model - see the note in the header.)
            std::vector<size_t> folds;
            if (*ratio != 1.0) {
                if (nodes[j].outputs.size() != 1)
                    continue;
                TensorId const dup_out      = nodes[j].outputs[0];
                bool           all_foldable = true;
                for (size_t k = 0; k < nodes.size(); k++) {
                    if (k == j || remove[k] || is_lifecycle(nodes[k].kind))
                        continue;
                    if (std::ranges::find(nodes[k].inputs, dup_out) == nodes[k].inputs.end())
                        continue;
                    if (!foldable_reader(nodes[k], dup_out)) {
                        all_foldable = false;
                        break;
                    }
                    folds.push_back(k);
                }
                if (!all_foldable)
                    continue;
            }

            for (size_t const k : folds) {
                fold_reader(nodes[k], *ratio);
            }

            // Equivalent! Redirect j's outputs to i's outputs.
            //
            // Two redirects are needed, for two different consumers:
            //   1. tensor_redirect + the Node::inputs rewrite below, keeps the
            //      TensorId metadata correct so liveness-based passes
            //      (MemoryPlanning, FreeInsertion) see node i's output as the
            //      live buffer and node j's as dead.
            //   2. Graph::redirect_slot: repoints node j's output slot at node
            //      i's buffer so any *already-baked* executor lambda that
            //      captured j's slot reads i's result at execute time. Without
            //      this the metadata rewrite is invisible at runtime and a
            //      surviving consumer of node j's output reads a never-written
            //      buffer (silent wrong results). Node i is a survivor (never
            //      itself removed in this pass), so its slot ptr is stable and
            //      copying it here is durable across re-execution.
            for (size_t k = 0; k < nodes[i].outputs.size(); k++) {
                tensor_redirect[nodes[j].outputs[k]] = nodes[i].outputs[k];
                graph.redirect_slot(nodes[j].outputs[k], nodes[i].outputs[k]);
            }

            remove[j] = true;
            modified  = true;
            merged    = true;
            ++_num_eliminated;

            if (*ratio == 1.0) {
                EINSUMS_LOG_INFO("CSE: eliminated node {} (duplicate of node {})", nodes[j].id, nodes[i].id);
                report(2, fmt::format("eliminate node {} '{}' — duplicate of node {}", nodes[j].id, nodes[j].label, nodes[i].id));
            } else {
                EINSUMS_LOG_INFO("CSE: eliminated node {} ({}x node {}, factor folded into {} reader(s))", nodes[j].id, *ratio, nodes[i].id,
                                 folds.size());
                report(2, fmt::format("eliminate node {} '{}' — {}x node {}, factor folded into {} reader(s)", nodes[j].id, nodes[j].label,
                                      *ratio, nodes[i].id, folds.size()));
            }
            break; // j is gone; nothing later in this bucket can match it
        }

        if (!merged) {
            // A survivor: later nodes with this key match against it.
            bucket.push_back(j);
        }
    }

    if (!modified)
        return false;

    // Apply redirections to all remaining nodes' inputs
    for (size_t i = 0; i < nodes.size(); i++) {
        if (remove[i])
            continue;
        for (auto &tid : nodes[i].inputs) {
            auto it = tensor_redirect.find(tid);
            if (it != tensor_redirect.end()) {
                tid = it->second;
            }
        }
    }

    // Remove eliminated nodes
    graph.erase_nodes(remove);
    graph.mark_sorted();

    return true;
}

bool CSE::run(Graph &graph) {
    // CSE walks the tree itself rather than letting PassManager recurse (see
    // recurse_into_subgraphs). The driver hands a pass nothing but the
    // sub-graph, and two of the guards below cannot be decided from inside one:
    // whether a buffer escapes the graph being rewritten, and whether a body's
    // handle for a parent-created tensor really describes a user tensor.
    // Collecting the whole tree once here answers both.
    _num_eliminated = 0;

    TreeContext ctx;
    collect_tree_context(graph, ctx);

    bool modified = run_on_graph(graph, &ctx, /*is_subgraph=*/false);

    // Post-order, so a body is rewritten before the parent it sits in. The
    // context stays valid across rewrites: erasing nodes only ever REMOVES
    // touches, so a buffer can look visible-outside when it no longer is, which
    // costs a missed merge and never an unsound one.
    std::function<void(Graph &)> descend = [&](Graph &sub) {
        sub.for_each_subgraph(descend);
        if (run_on_graph(sub, &ctx, /*is_subgraph=*/true)) {
            modified = true;
        }
    };
    graph.for_each_subgraph(descend);

    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
