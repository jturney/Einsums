//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/DistributiveFactoring.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Key for grouping factorable einsum nodes.
struct FactorKey {
    TensorId                 output_id;
    TensorId                 shared_input_id;
    bool                     shared_is_first;
    std::vector<std::string> non_shared_indices;

    bool operator==(FactorKey const &o) const {
        return output_id == o.output_id && shared_input_id == o.shared_input_id && shared_is_first == o.shared_is_first &&
               non_shared_indices == o.non_shared_indices;
    }
};

struct FactorKeyHash {
    size_t operator()(FactorKey const &k) const {
        size_t h = std::hash<TensorId>{}(k.output_id);
        h ^= std::hash<TensorId>{}(k.shared_input_id) * 2654435761ULL;
        h ^= std::hash<bool>{}(k.shared_is_first) * 40503ULL;
        for (auto const &s : k.non_shared_indices) {
            h ^= std::hash<std::string>{}(s)*16777619ULL;
        }
        return h;
    }
};

struct FactorCandidate {
    size_t   node_index;
    TensorId non_shared_input;
    double   ab_prefactor;
};

/// A real prefactor exactly equal to one.
///
/// The factored form applies the output prefactor ONCE, so every member after
/// the first has to be a pure accumulation. A member with ``c_pf != 1`` rescales
/// the partial sum its predecessors already wrote, which the single combined
/// contraction cannot reproduce: `W = 2*(W + A*B0) + A*B1` is not
/// `W = 2*W + A*(B0 + B1)`.
bool is_unit_real(PrefactorScalar const &p) {
    return is_real_valued(p) && as_real<double>(p) == 1.0;
}

/// Canonical identity of a summed intermediate: which operands, scaled by what.
///
/// Sorted, so two groups that happen to list the same terms in a different order
/// still agree. Prefactors are part of the identity: CCSD's tau and tau-tilde sum
/// the same three operands and differ only in a coefficient, and they are
/// genuinely different tensors.
std::vector<std::pair<TensorId, double>> sum_identity(std::vector<FactorCandidate> const &members) {
    std::vector<std::pair<TensorId, double>> key;
    key.reserve(members.size());
    for (auto const &c : members) {
        key.emplace_back(c.non_shared_input, c.ab_prefactor);
    }
    std::ranges::sort(key);
    return key;
}

/// Whether @p v is exactly a power of two, sign included.
///
/// Multiplying by such a value only shifts an exponent, so scaling a sum gives
/// bit-for-bit what scaling every term before summing would: `r*(a+b)` equals
/// `r*a + r*b`. That identity is what lets one built sum stand in for a
/// proportional one without changing the arithmetic.
bool is_exact_power_of_two(double v) {
    if (!std::isfinite(v) || v == 0.0) {
        return false;
    }
    int          exp = 0;
    double const m   = std::frexp(v, &exp);
    return m == 0.5 || m == -0.5;
}

/// The factor r for which @p candidate is r times @p built, term by term, when
/// there is one that is safe to use.
///
/// r == 1 (an exact match) is the common case and falls out of the same test.
/// Ratios that are not powers of two are declined even when they are exactly
/// representable: the scale would be applied to the assembled sum rather than to
/// each term, and only a power of two makes those agree. Lifting that is safe up
/// to rounding, but it would make the result depend on how the pass chose to
/// share, which is a bad property for a pass that runs by default.
std::optional<double> proportional_scale(std::vector<std::pair<TensorId, double>> const &built,
                                         std::vector<std::pair<TensorId, double>> const &candidate) {
    if (built.empty() || built.size() != candidate.size()) {
        return std::nullopt;
    }
    // Both lists are sorted and every operand in a group is distinct, so equal
    // sums line up index by index.
    for (size_t i = 0; i < built.size(); i++) {
        if (built[i].first != candidate[i].first) {
            return std::nullopt;
        }
    }
    if (built[0].second == 0.0) {
        return std::nullopt;
    }
    double const r = candidate[0].second / built[0].second;
    if (!is_exact_power_of_two(r)) {
        return std::nullopt;
    }
    for (size_t i = 0; i < built.size(); i++) {
        if (built[i].second * r != candidate[i].second) {
            return std::nullopt;
        }
    }
    return r;
}

/// GEMM-shaped extents of a contraction, for pricing it.
struct ContractionShape {
    size_t m{1};     ///< target extents carried only by A
    size_t n{1};     ///< target extents carried only by B
    size_t k{1};     ///< link extents: in both operands, not in the output
    size_t batch{1}; ///< target extents in both operands and the output
    bool   ok{false};
};

/// Classify a contraction's indices the way PackedGemm does, so the cost model
/// can be asked for the time of an equivalent batched GEMM.
ContractionShape contraction_shape(std::vector<std::string> const &a_idx, std::vector<size_t> const &a_dims,
                                   std::vector<std::string> const &b_idx, std::vector<size_t> const &b_dims,
                                   std::vector<std::string> const &c_idx) {
    ContractionShape s;
    if (a_idx.size() != a_dims.size() || b_idx.size() != b_dims.size()) {
        return s;
    }
    std::unordered_map<std::string, size_t> extent;
    for (size_t i = 0; i < a_idx.size(); i++) {
        extent[a_idx[i]] = a_dims[i];
    }
    for (size_t i = 0; i < b_idx.size(); i++) {
        extent[b_idx[i]] = b_dims[i];
    }
    auto const has = [](std::vector<std::string> const &v, std::string const &x) { return std::ranges::find(v, x) != v.end(); };
    for (auto const &[name, ext] : extent) {
        bool const in_a = has(a_idx, name);
        bool const in_b = has(b_idx, name);
        bool const in_c = has(c_idx, name);
        if (in_a && in_b && in_c) {
            s.batch *= ext;
        } else if (in_a && in_b) {
            s.k *= ext;
        } else if (in_a && in_c) {
            s.m *= ext;
        } else if (in_b && in_c) {
            s.n *= ext;
        } else {
            s.k *= ext; // summed over but present in one operand only: still reduced
        }
    }
    s.ok = true;
    return s;
}

/// A summed intermediate already built at this level, offered for reuse.
struct BuiltSum {
    std::vector<std::pair<TensorId, double>> identity;
    TensorId                                 t_id{0};
    std::size_t                              build_pos{0}; ///< pre-erase position its nodes are spliced at
    std::unordered_set<TensorId>             operands;     ///< what it summed, for staleness checks
};

} // namespace

DistributiveFactoring::DistributiveFactoring() : DistributiveFactoring(CostModel::detect_default(), Factor::Auto) {
}

DistributiveFactoring::DistributiveFactoring(Factor factor) : DistributiveFactoring(CostModel::detect_default(), factor) {
}

DistributiveFactoring::DistributiveFactoring(CostModel cost_model, Factor factor) : _cost_model(std::move(cost_model)), _factor(factor) {
}

void DistributiveFactoring::reset_stats() {
    _num_groups       = 0;
    _num_eliminated   = 0;
    _num_unprofitable = 0;
}

bool DistributiveFactoring::run(Graph &graph) {
    // Own the recursion (like LoopInvariantHoisting): reset the counters once
    // here at the root, then descend ourselves. If we instead opted into
    // PassManager auto-recursion, run() would be re-invoked per subgraph and
    // reset the top-level tally each time. recurse_into_subgraphs()
    // returns false so the PassManager does not double-walk.
    _groups.clear();
    return run_recursive(graph);
}

bool DistributiveFactoring::run_recursive(Graph &graph) {
    bool modified = factor_one_level(graph);
    // Factor loop bodies and conditional branches too; counters accumulate
    // across the whole tree (no per-level reset).
    graph.for_each_subgraph([&](Graph &sub) {
        if (run_recursive(sub)) {
            modified = true;
        }
    });
    return modified;
}

bool DistributiveFactoring::factor_one_level(Graph &graph) {
    graph.topological_sort();

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    if (nodes.size() < 2) {
        return false;
    }

    // --- Phase 1: Collect candidates ---
    std::unordered_map<FactorKey, std::vector<FactorCandidate>, FactorKeyHash> candidate_groups;

    for (size_t ni = 0; ni < nodes.size(); ni++) {
        auto const &node = nodes[ni];
        if (node.kind != OpKind::Einsum)
            continue;

        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (!desc)
            continue;
        if (desc->conj_a || desc->conj_b)
            continue; // conjugated contractions aren't factored (conj not threaded through the rewrite)
        if (is_zero(desc->c_prefactor))
            continue;
        if (node.inputs.size() != 2 || node.outputs.size() != 1)
            continue;

        TensorId const out_id = node.outputs[0];
        TensorId const in_a   = node.inputs[0];
        TensorId const in_b   = node.inputs[1];
        auto const    &spec   = desc->spec;

        // The factoring math below is real-valued; a prefactor with nonzero
        // imaginary part would silently lose it, so skip those nodes.
        if (!is_real_valued(desc->ab_prefactor)) {
            continue;
        }
        auto const ab_pf_d = as_real<double>(desc->ab_prefactor);

        // Try first input as shared
        {
            FactorKey const key{
                .output_id = out_id, .shared_input_id = in_a, .shared_is_first = true, .non_shared_indices = spec.b_indices};
            candidate_groups[key].push_back({.node_index = ni, .non_shared_input = in_b, .ab_prefactor = ab_pf_d});
        }
        // Try second input as shared
        {
            FactorKey const key{
                .output_id = out_id, .shared_input_id = in_b, .shared_is_first = false, .non_shared_indices = spec.a_indices};
            candidate_groups[key].push_back({.node_index = ni, .non_shared_input = in_a, .ab_prefactor = ab_pf_d});
        }
    }

    // --- Phase 2: Filter to valid groups ---
    struct ValidGroup {
        FactorKey                    key;
        std::vector<FactorCandidate> candidates;
    };
    std::vector<ValidGroup> valid_groups;

    for (auto &[key, candidates] : candidate_groups) {
        if (candidates.size() < 2)
            continue;

        // All non-shared inputs must have same shape and dtype
        auto it0 = tensors.find(candidates[0].non_shared_input);
        if (it0 == tensors.end())
            continue;
        auto const &ref_dims  = it0->second.dims;
        auto const  ref_dtype = it0->second.dtype;

        bool all_compatible = true;
        for (size_t ci = 1; ci < candidates.size(); ci++) {
            auto it = tensors.find(candidates[ci].non_shared_input);
            if (it == tensors.end() || it->second.dims != ref_dims || it->second.dtype != ref_dtype) {
                all_compatible = false;
                break;
            }
        }
        if (!all_compatible)
            continue;

        // All non-shared inputs must be different tensors
        bool has_duplicate = false;
        for (size_t ci = 0; ci < candidates.size() && !has_duplicate; ci++) {
            for (size_t cj = ci + 1; cj < candidates.size(); cj++) {
                if (candidates[ci].non_shared_input == candidates[cj].non_shared_input) {
                    has_duplicate = true;
                    break;
                }
            }
        }
        if (has_duplicate)
            continue;

        // The rewrite redirects the non-shared operand's SLOT (keyed by tensor id)
        // to the summed intermediate T. If the shared operand is the SAME tensor as
        // a summed operand, they share that slot, so a self-contraction like A*A
        // would read T for BOTH factors (T*T instead of A*T). Reject aliased groups;
        // the slot-redirect trick cannot separate the two reads.
        bool shared_aliases_nonshared = false;
        for (auto const &c : candidates) {
            if (c.non_shared_input == key.shared_input_id) {
                shared_aliases_nonshared = true;
                break;
            }
        }
        if (shared_aliases_nonshared)
            continue;

        valid_groups.push_back({.key = key, .candidates = std::move(candidates)});
    }

    if (valid_groups.empty()) {
        return false;
    }

    // --- Phase 3: Deduplicate (largest group first, greedy) ---
    // Ties break on position, earliest first, so that when several groups sum the
    // same operands the EARLIEST one builds the intermediate and the later ones
    // can reuse it. Sorting on size alone leaves the winner to unordered_map
    // iteration order, and an owner positioned after its peers cannot be shared.
    std::ranges::sort(valid_groups, [](ValidGroup const &a, ValidGroup const &b) {
        if (a.candidates.size() != b.candidates.size()) {
            return a.candidates.size() > b.candidates.size();
        }
        return a.candidates.front().node_index < b.candidates.front().node_index;
    });

    // create_*_tensor_dynamic appends Alloc nodes (index >= orig_count) that are
    // always kept; the removal / used sets cover only the original nodes.
    size_t const      orig_count = nodes.size();
    std::vector<bool> node_used(orig_count, false);
    std::vector<bool> remove(orig_count, false);
    bool              modified = false;

    // Replacement nodes, keyed by the pre-erase position they belong at.
    std::vector<std::pair<std::size_t, std::vector<Node>>> inserts;

    // Sums built so far, so several consumers of one quantity share one tensor.
    std::vector<BuiltSum> built_sums;

    for (auto &vg : valid_groups) {
        std::vector<FactorCandidate> available;
        for (auto const &c : vg.candidates) {
            if (!node_used[c.node_index]) {
                available.push_back(c);
            }
        }
        if (available.size() < 2)
            continue;

        // Candidates are collected in ascending node-index (execution) order and
        // that order survives filtering, so available.front() is the earliest
        // member and available.back() the latest.
        size_t const first_pos = available.front().node_index;
        size_t const last_pos  = available.back().node_index;

        // --- Pure-accumulation gate ---
        // Only the first member's output prefactor survives into the combined
        // contraction, so the rest must accumulate with exactly 1. See
        // @ref is_unit_real for why anything else changes the result.
        {
            bool unit_accumulate = true;
            for (size_t ci = 1; ci < available.size(); ci++) {
                auto const *d = std::get_if<EinsumDescriptor>(&nodes[available[ci].node_index].op_data);
                if (d == nullptr || !is_unit_real(d->c_prefactor)) {
                    unit_accumulate = false;
                    break;
                }
            }
            if (!unit_accumulate) {
                auto on = tensors.find(vg.key.output_id);
                report(3, fmt::format("skip factoring into '{}': a member accumulates with a prefactor other than 1, which would rescale "
                                      "the partial sum",
                                      on != tensors.end() ? on->second.name : "?"));
                continue;
            }
        }

        // --- Placement / interference gate (mirrors GEMMBatching / LCCF) ---
        // The combined node takes the FIRST member's slot so it stays scan-before
        // any consumer of the factored output. Position IS program order in this
        // IR, so appending the writer would let a later reader of the output run
        // first and observe a stale buffer, and the PassManager's program-order
        // verifier (check_observed_writes) would reject the rewrite.
        // The slot move is only sound when no OTHER node between the first and last
        // member reads/writes the output (would observe/clobber the partial sum)
        // or writes the shared / non-shared operands (would change a factor
        // mid-fold). Control-flow nodes hide their I/O in sub-graphs, so their
        // presence in the span disqualifies the group outright.
        {
            std::vector<bool>            is_member(nodes.size(), false);
            std::unordered_set<TensorId> operand_ids;
            operand_ids.insert(vg.key.shared_input_id);
            for (auto const &c : available) {
                is_member[c.node_index] = true;
                operand_ids.insert(c.non_shared_input);
            }
            bool const interference =
                span_interferes(nodes, first_pos, last_pos, is_member, vg.key.output_id, operand_ids, /*reject_control_flow=*/true);
            if (interference) {
                auto on = tensors.find(vg.key.output_id);
                report(3, fmt::format("skip factoring into '{}': an intervening node reads/writes the output or a factor operand",
                                      on != tensors.end() ? on->second.name : "?"));
                continue;
            }
        }

        // --- Phase 4: Rewrite the graph ---

        // Get the reference tensor handle for the non-shared operands
        auto ref_it = tensors.find(available[0].non_shared_input);
        if (ref_it == tensors.end())
            continue;
        auto const &ref_handle = ref_it->second;

        // The combined executor redirects the einsum's non-shared operand slot to
        // the accumulator T, so T must be the SAME tensor kind the operands are: a
        // runtime graph reads a GeneralRuntimeTensor at that slot, a compile-time
        // graph a Tensor<T, Rank>. Require every summed operand to share that kind
        // and build a matching accumulator, so make_zero/make_axpy dispatch
        // correctly - a compile-time accumulator fed runtime operands rank-errors
        // at execute on any Python-captured graph.
        bool const operands_runtime = ref_handle.is_runtime;
        bool       kinds_uniform    = true;
        for (auto const &c : available) {
            auto it = tensors.find(c.non_shared_input);
            if (it == tensors.end() || it->second.is_runtime != operands_runtime) {
                kinds_uniform = false;
                break;
            }
        }
        if (!kinds_uniform) {
            // The accumulator this pass builds has to dispatch like its
            // operands; a compile-time accumulator fed runtime operands
            // rank-errors at execute.
            note_skip("summed operands mix runtime and statically-typed tensors",
                      fmt::format("group of {} into tensor {}", available.size(), vg.key.output_id));
            continue;
        }

        // --- Reuse a sum already built at this level ---
        // A quantity a chemist names once and consumes several times (CCSD's tau
        // feeds W_mnij, W_abef and the T2 equation) arrives here as several groups
        // summing the same operands with the same prefactors. Reusing the first
        // build is what makes it one tensor rather than one per consumer. CSE
        // cannot do it for us: an accumulation buffer has several writers, and its
        // single-writer guard is there to stop readers being redirected onto a
        // buffer that is mutated again.
        // A consumer that wants the same sum scaled reuses it too: the scale rides
        // on the contraction's ab_pf, which costs nothing, instead of paying for a
        // second buffer and a second axpy chain. CCSD consumes tau with 1/4 in
        // W_mnij and W_abef and with 1/2 in the T2 equation, so this is the common
        // case rather than a corner one.
        auto const              identity = sum_identity(available);
        std::optional<TensorId> reuse_id;
        double                  reuse_scale = 1.0;
        for (auto const &bs : built_sums) {
            auto const scale = proportional_scale(bs.identity, identity);
            if (!scale) {
                continue;
            }
            // The build has to run before this contraction reads it.
            if (bs.build_pos >= first_pos) {
                continue;
            }
            // And nothing in between may rewrite a summed operand, or T holds a
            // stale value by the time we get here. Control flow hides its writes
            // in a subgraph, so its presence in the span forfeits the reuse.
            bool usable = true;
            for (size_t k = bs.build_pos; k < first_pos && usable; k++) {
                if (remove[k]) {
                    continue; // subsumed by an earlier group; it will not exist
                }
                if (nodes[k].kind == OpKind::Loop || nodes[k].kind == OpKind::Conditional) {
                    usable = false;
                    break;
                }
                for (auto out : nodes[k].outputs) {
                    if (bs.operands.contains(out)) {
                        usable = false;
                        break;
                    }
                }
            }
            if (!usable) {
                continue;
            }
            reuse_id    = bs.t_id;
            reuse_scale = *scale;
            break;
        }

        // --- Profitability ---
        // The trade is (N-1) contractions saved against one axpy chain over the
        // summed operands, plus a buffer. It pays when the contraction is
        // flop-bound and loses when it is bandwidth-bound: a cheap contraction
        // over large operands spends more assembling the sum than it saves. Price
        // both in microseconds against the shared machine profile, the same way
        // TiledExpansion decides between its two lowerings, rather than guessing
        // from a term count. A reused sum costs nothing to build, so those groups
        // are essentially always profitable. Factor::Always skips the question.
        if (_factor == Factor::Auto) {
            auto const *fd    = std::get_if<EinsumDescriptor>(&nodes[first_pos].op_data);
            auto        sh_h  = tensors.find(vg.key.shared_input_id);
            auto        out_h = tensors.find(vg.key.output_id);
            if (fd == nullptr || sh_h == tensors.end() || out_h == tensors.end()) {
                continue;
            }
            auto const &a_dims = vg.key.shared_is_first ? sh_h->second.dims : ref_handle.dims;
            auto const &b_dims = vg.key.shared_is_first ? ref_handle.dims : sh_h->second.dims;
            auto const  shape  = contraction_shape(fd->spec.a_indices, a_dims, fd->spec.b_indices, b_dims, fd->spec.c_indices);
            if (!shape.ok) {
                continue;
            }

            size_t const elem    = ref_handle.element_size != 0 ? ref_handle.element_size : sizeof(double);
            size_t       b_elems = 1;
            for (auto d : ref_handle.dims) {
                b_elems *= d;
            }
            size_t const b_bytes = b_elems * elem;

            // estimate_total_gemm_time_us already carries the launch and allocation
            // cost of being a node, so node_overhead_us is added only to the axpy
            // chain, which has no such built-in estimate. Adding it to both put
            // small shapes on a knife edge between the two totals.
            double const overhead = _cost_model.node_overhead_us(Target::CPU);
            double const einsum_us =
                static_cast<double>(shape.batch) * _cost_model.estimate_total_gemm_time_us(shape.m, shape.n, shape.k, elem, Target::CPU);
            double const n_terms    = static_cast<double>(available.size());
            double const unfactored = n_terms * einsum_us;

            // Zero writes T once; each axpy reads its operand and read-modify-writes
            // T, so three buffer-sized touches per term.
            double const build_us = reuse_id ? 0.0
                                             : _cost_model.estimate_memory_time_us(b_bytes, Target::CPU) +
                                                   n_terms * _cost_model.estimate_memory_time_us(3 * b_bytes, Target::CPU) +
                                                   (n_terms + 1.0) * overhead;
            double const factored = build_us + einsum_us;

            if (factored >= unfactored) {
                _num_unprofitable++;
                note_skip("cost model says the summed accumulator costs more than the contractions it saves",
                          fmt::format("group of {} into '{}': {:.1f} us unfactored vs {:.1f} us factored", available.size(),
                                      out_h->second.name, unfactored, factored));
                continue;
            }
        }

        // Create intermediate tensor T = sum of non-shared operands, of the same
        // kind as the operands. This APPENDS an Alloc node (kept below).
        TensorId    t_id{};
        std::string t_name;
        if (reuse_id) {
            t_id     = *reuse_id;
            auto tit = tensors.find(t_id);
            t_name   = tit != tensors.end() ? tit->second.name : "?";
        } else {
            t_name             = fmt::format("_df_sum_{}", _num_groups);
            auto create_result = operands_runtime ? graph.create_zero_runtime_tensor_dynamic(t_name, ref_handle.dtype, ref_handle.dims)
                                                  : graph.create_tensor_dynamic(t_name, ref_handle.dtype, ref_handle.dims);
            if (!create_result)
                continue; // Skip this factoring group if tensor creation fails
            t_id = create_result.value().first;
        }

        // Emit the factorization as ORDINARY nodes: zero T, accumulate each
        // non-shared operand into it, contract once against the shared operand.
        //
        // An earlier version fused all of that into one OpKind::Custom node whose
        // executor swapped a slot pointer so the first member's baked einsum
        // executor would read T. That worked, but it made the factorization opaque:
        // T's construction was invisible to every other pass, so two consumers of
        // the same sum each built their own copy (CSE has no nodes to match) and a
        // sum that is loop-invariant could not be hoisted out of a CC iteration.
        // Explicit nodes cost nothing at execute and let those passes do their job.
        auto const *first_desc = std::get_if<EinsumDescriptor>(&nodes[first_pos].op_data);
        if (first_desc == nullptr) {
            continue;
        }
        // make_einsum_node wants the parsed form; the descriptor keeps the
        // PackedGemm topology. Only the raw index lists carry over, which is all
        // that changes here: T takes the non-shared operand's place, so the
        // contraction's indices are exactly the ones the members already used.
        ParsedEinsumSpec spec;
        spec.c_indices = first_desc->spec.c_indices;
        spec.a_indices = first_desc->spec.a_indices;
        spec.b_indices = first_desc->spec.b_indices;
        spec.raw =
            fmt::format("{} <- {} ; {}", fmt::join(spec.c_indices, ","), fmt::join(spec.a_indices, ","), fmt::join(spec.b_indices, ","));
        auto const c_pf = first_desc->c_prefactor;

        std::vector<Node> emitted;
        emitted.reserve(available.size() + 2);

        // Reusing an existing sum emits only the contraction; the build already
        // stands earlier in the graph.
        if (!reuse_id) {
            // T = 0. Recorded as a Scale by zero, which is what it means, so the
            // algebraic passes can reason about it; the executor is a true zero-fill
            // rather than a multiply, so a freshly allocated T holding garbage cannot
            // turn into a NaN here.
            {
                Node nd;
                nd.id      = graph.reserve_node_id();
                nd.kind    = OpKind::Scale;
                nd.label   = fmt::format("zero({})", t_name);
                nd.inputs  = {t_id};
                nd.outputs = {t_id};
                nd.op_data = ScaleDescriptor{.factor = 0.0};
                nd.execute = graph.make_zero_executor(t_id);
                emitted.push_back(std::move(nd));
            }

            // T += ab_i * B_i, carrying each member's own product prefactor, so the
            // combined contraction needs no prefactor of its own.
            for (auto const &c : available) {
                auto nit = tensors.find(c.non_shared_input);
                Node nd;
                nd.id      = graph.reserve_node_id();
                nd.kind    = OpKind::Axpby;
                nd.label   = fmt::format("axpy({} -> {}, alpha={})", nit != tensors.end() ? nit->second.name : "?", t_name, c.ab_prefactor);
                nd.inputs  = {c.non_shared_input, t_id};
                nd.outputs = {t_id};

                // Live scalars shared with the executor, so the accumulator this
                // pass builds is as readable to later passes as a captured one.
                auto params   = std::make_shared<AxpbyParams>();
                params->alpha = PrefactorScalar{c.ab_prefactor};
                params->beta  = PrefactorScalar{1.0};

                AxpbyDescriptor desc;
                desc.alpha  = params->alpha;
                desc.beta   = params->beta;
                desc.params = params;
                nd.op_data  = desc;

                nd.execute = graph.make_axpby_executor(params, c.non_shared_input, t_id);
                emitted.push_back(std::move(nd));
            }
        }

        // One contraction, with T in the position the non-shared operand held, so
        // the spec is unchanged. ab_pf carries only the ratio to a reused sum; for
        // a sum built here it is 1, the members' prefactors having moved into the
        // axpys.
        {
            TensorId const a_id  = vg.key.shared_is_first ? vg.key.shared_input_id : t_id;
            TensorId const b_id  = vg.key.shared_is_first ? t_id : vg.key.shared_input_id;
            auto           label = reuse_scale == 1.0
                                       ? fmt::format("df_factored({} terms via {})", available.size(), t_name)
                                       : fmt::format("df_factored({} terms via {} scaled by {})", available.size(), t_name, reuse_scale);
            emitted.push_back(graph.make_einsum_node(a_id, b_id, vg.key.output_id, spec, c_pf, PrefactorScalar{reuse_scale},
                                                     /*conj_a=*/false, /*conj_b=*/false, std::move(label)));
        }

        // Offer this sum to the groups still to come.
        if (!reuse_id) {
            BuiltSum bs;
            bs.identity  = identity;
            bs.t_id      = t_id;
            bs.build_pos = first_pos;
            for (auto const &c : available) {
                bs.operands.insert(c.non_shared_input);
            }
            built_sums.push_back(std::move(bs));
        }

        // Splice in at the first member's position so the combined writer stays
        // scan-before every consumer of the output, and drop all the members; see
        // the placement gate above for why a later position is unsound.
        inserts.emplace_back(first_pos, std::move(emitted));
        for (auto const &c : available) {
            remove[c.node_index]    = true;
            node_used[c.node_index] = true;
        }
        _num_eliminated += available.size() - 1;

        // Record the group for reporting
        FactoringGroup fg;
        auto           sh_it = tensors.find(vg.key.shared_input_id);
        fg.shared_tensor     = sh_it != tensors.end() ? sh_it->second.name : "?";
        auto out_it          = tensors.find(vg.key.output_id);
        fg.output_tensor     = out_it != tensors.end() ? out_it->second.name : "?";
        fg.num_terms         = available.size();
        for (auto const &c : available) {
            auto nit = tensors.find(c.non_shared_input);
            fg.summed_tensors.push_back(nit != tensors.end() ? nit->second.name : "?");
        }
        report(2, fmt::format("factor {} contractions into '{}' sharing operand '{}' (sum operands, contract once)", available.size(),
                              out_it != tensors.end() ? out_it->second.name : "?", sh_it != tensors.end() ? sh_it->second.name : "?"));
        _groups.push_back(std::move(fg));
        _num_groups++;
        modified = true;
    }

    if (!modified)
        return false;

    // create_*_tensor_dynamic appended Alloc nodes past orig_count; pad the mask so
    // it covers them (always kept) and so the shift table below is complete.
    remove.resize(nodes.size(), false);

    // Keep the appended Alloc nodes; drop the subsumed originals.
    graph.erase_nodes(remove);

    // Positions were recorded pre-erase; shift each down by the number of erased
    // nodes below it. A group's own first-member slot IS erased, so the shifted
    // position lands exactly where it used to be and the replacements take its
    // place (same accounting as TiledExpansion).
    std::vector<size_t> erased_below(remove.size() + 1, 0);
    for (size_t i = 0; i < remove.size(); i++) {
        erased_below[i + 1] = erased_below[i] + (remove[i] ? 1 : 0);
    }
    for (auto &[position, group] : inserts) {
        position -= erased_below[position];
    }
    graph.insert_node_groups(std::move(inserts));

    // Re-sort since we added/removed nodes
    graph.topological_sort();

    EINSUMS_LOG_INFO("DistributiveFactoring: rewrote {} groups, eliminated {} nodes", _num_groups, _num_eliminated);
    report(1, fmt::format("factored {} group(s), eliminated {} contraction node(s)", _num_groups, _num_eliminated));
    return true;
}

std::vector<std::string> DistributiveFactoring::explain() const {
    if (_num_groups == 0 && _num_unprofitable == 0) {
        return {};
    }
    return {fmt::format("DistributiveFactoring: factored {} group(s), eliminated {} contraction(s); {} declined as unprofitable",
                        _num_groups, _num_eliminated, _num_unprofitable)};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
