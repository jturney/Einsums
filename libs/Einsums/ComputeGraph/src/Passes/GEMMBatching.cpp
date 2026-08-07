//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/Detail/BatchedGemm.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/GEMMBatching.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

// ── Grouping key ────────────────────────────────────────────────────────────
//
// Two einsums can be merged iff every field here matches AND they live at
// the same dependency level (level handled separately in the outer map).

struct BatchKey {
    int        m, n, k;
    char       trans_a, trans_b;
    BlasScalar scalar;
    // Alpha/beta bit-equal so we don't accidentally batch 1.0 with
    // 0.9999… (common precision drift would break the semantic match).
    std::uint64_t alpha_bits;
    std::uint64_t beta_bits;

    bool operator<(BatchKey const &o) const {
        return std::tie(m, n, k, trans_a, trans_b, scalar, alpha_bits, beta_bits) <
               std::tie(o.m, o.n, o.k, o.trans_a, o.trans_b, o.scalar, o.alpha_bits, o.beta_bits);
    }
};

std::uint64_t bits_of(double v) {
    std::uint64_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

} // namespace

void GEMMBatching::reset_stats() {
    _num_batches      = 0;
    _total_batched    = 0;
    _num_gate_skipped = 0;
}

bool GEMMBatching::run(Graph &graph) {
    // Per-apply counters: compare against entry values, not zero. The
    // recursive driver calls run() once per subgraph and reset_stats() runs
    // only once per apply, so `_num_x > 0` would report this graph as
    // modified whenever ANY earlier subgraph changed something.
    size_t const num_batches_at_entry = _num_batches;
    graph.topological_sort();

    auto        &nodes   = graph.nodes();
    auto const  &deps    = graph.dependencies();
    size_t const n_nodes = nodes.size();

    if (n_nodes < 2)
        return false;

    // Dependency levels: two nodes at the same level have no path between
    // them; safe to batch. DependencyInfo already computed these (the
    // executors consume them) - invert its level->positions lists into a
    // per-position lookup instead of re-deriving from predecessors.
    std::vector<size_t> level(n_nodes, 0);
    for (size_t lvl = 0; lvl < deps.levels.size(); ++lvl) {
        for (size_t const pos : deps.levels[lvl]) {
            if (pos < n_nodes)
                level[pos] = lvl;
        }
    }

    // Group candidate indices by (level, BatchKey).
    std::map<std::pair<size_t, BatchKey>, std::vector<size_t>> groups;

    for (size_t nd = 0; nd < n_nodes; ++nd) {
        if (nodes[nd].kind != OpKind::Einsum)
            continue;
        auto *desc = std::get_if<EinsumDescriptor>(&nodes[nd].op_data);
        if (!desc || !desc->gemm_hint)
            continue; // non-GEMM-pattern einsums skipped by capture
        if (desc->conj_a || desc->conj_b)
            continue; // conjugated einsums aren't batched (conj not threaded through the batch rewrite)

        BatchKey key;
        key.m       = desc->gemm_hint->m;
        key.n       = desc->gemm_hint->n;
        key.k       = desc->gemm_hint->k;
        key.trans_a = desc->gemm_hint->trans_a;
        key.trans_b = desc->gemm_hint->trans_b;
        key.scalar  = desc->gemm_hint->scalar;
        // PrefactorScalar carries dtype info too; fold both index + bytes
        // into the batching key so we never group differently-typed prefactors.
        key.alpha_bits = static_cast<std::uint64_t>(hash(desc->ab_prefactor));
        key.beta_bits  = static_cast<std::uint64_t>(hash(desc->c_prefactor));
        groups[{level[nd], key}].push_back(nd);
    }

    std::vector<bool> remove(n_nodes, false);

    for (auto const &[keyed_level, group] : groups) {
        if (group.size() < 2)
            continue;

        auto const &[lvl, key] = keyed_level;

        // Profitability gate: a batch executes as one node, and TaskPool
        // workers run BLAS single-threaded, so batching serializes work the
        // Dataflow executor would spread across workers. Only worth it when
        // each member is small enough that per-node scheduling overhead is a
        // meaningful fraction of its runtime.
        if (_has_cost_model) {
            size_t const elem_size = key.scalar == BlasScalar::Float          ? sizeof(float)
                                     : key.scalar == BlasScalar::Double       ? sizeof(double)
                                     : key.scalar == BlasScalar::ComplexFloat ? sizeof(std::complex<float>)
                                                                              : sizeof(std::complex<double>);
            double const gemm_us   = _cost_model.estimate_total_gemm_time_us(static_cast<size_t>(key.m), static_cast<size_t>(key.n),
                                                                             static_cast<size_t>(key.k), elem_size, Target::CPU);
            if (gemm_us > _max_gemm_us) {
                _num_gate_skipped++;
                EINSUMS_LOG_INFO("GEMMBatching: group of {} GEMMs ({}x{}x{}, ~{:.1f}us each) exceeds the {:.0f}us batching "
                                 "threshold — leaving them as independent nodes",
                                 group.size(), key.m, key.k, key.n, gemm_us, _max_gemm_us);
                note_skip("each GEMM in the group is already large enough to run better as its own parallel node",
                          fmt::format("group of {} at level {}", group.size(), lvl));
                report(2, fmt::format("skip group of {} GEMMs (~{:.1f}us each > {:.0f}us gate) — better as parallel nodes", group.size(),
                                      gemm_us, _max_gemm_us));
                continue;
            }
        }

        // Probe lda/ldb/ldc on the first member; reject the group if any
        // other member disagrees. Non-uniform strides can't share one
        // gemm_batch call.
        auto     *first_desc = std::get_if<EinsumDescriptor>(&nodes[group.front()].op_data);
        auto      first_a    = first_desc->gemm_hint->extract_a();
        auto      first_b    = first_desc->gemm_hint->extract_b();
        auto      first_c    = first_desc->gemm_hint->extract_c();
        int const lda = first_a.second, ldb = first_b.second, ldc = first_c.second;

        bool uniform = true;
        for (size_t idx = 1; idx < group.size(); ++idx) {
            auto *d       = std::get_if<EinsumDescriptor>(&nodes[group[idx]].op_data);
            auto [ap, la] = d->gemm_hint->extract_a();
            auto [bp, lb] = d->gemm_hint->extract_b();
            auto [cp, lc] = d->gemm_hint->extract_c();
            (void)ap;
            (void)bp;
            (void)cp;
            if (la != lda || lb != ldb || lc != ldc) {
                uniform = false;
                break;
            }
        }
        if (!uniform) {
            EINSUMS_LOG_INFO("GEMMBatching: group of {} einsums at level {} ({}×{}×{}) has mismatched strides — not batching", group.size(),
                             lvl, key.m, key.k, key.n);
            note_skip("group members' operand strides differ, so one batched call cannot address them",
                      fmt::format("group of {} at level {}", group.size(), lvl));
            report(3, fmt::format("skip group of {} einsums at level {} ({}x{}x{}) — mismatched strides", group.size(), lvl, key.m, key.k,
                                  key.n));
            continue;
        }

        // Placement/interference gate. The topological sort derives RAW/WAR/WAW
        // edges from scan order (position IS program order in this IR), so the
        // batched node must occupy the FIRST member's slot to stay ahead of any
        // consumer of a member's output (appending would legally schedule the
        // batch after such a consumer, which then reads a stale buffer). That
        // placement is only sound if no outside node between the first and last
        // member touches what the batch reads or writes; skip the group when
        // one does. Control-flow nodes hide their I/O in sub-graphs, so their
        // presence in the span disqualifies the group outright.
        size_t const first_pos = *std::min_element(group.begin(), group.end());
        size_t const last_pos  = *std::max_element(group.begin(), group.end());
        {
            std::unordered_set<size_t>   member_set(group.begin(), group.end());
            std::unordered_set<TensorId> batch_reads, batch_writes;
            for (size_t const idx : group) {
                for (auto tid : nodes[idx].inputs)
                    batch_reads.insert(tid);
                for (auto tid : nodes[idx].outputs)
                    batch_writes.insert(tid);
            }
            bool interference = false;
            for (size_t i = first_pos + 1; i < last_pos && !interference; i++) {
                if (member_set.count(i))
                    continue;
                Node const &other = nodes[i];
                if (other.kind == OpKind::Loop || other.kind == OpKind::Conditional) {
                    interference = true;
                    break;
                }
                for (auto tid : other.outputs) {
                    if (batch_reads.count(tid) || batch_writes.count(tid)) {
                        interference = true;
                        break;
                    }
                }
                for (auto tid : other.inputs) {
                    if (batch_writes.count(tid)) {
                        interference = true;
                        break;
                    }
                }
            }
            if (interference) {
                EINSUMS_LOG_INFO("GEMMBatching: group of {} einsums at level {} has an interfering node between members — not batching",
                                 group.size(), lvl);
                note_skip("a node between the group members reads or writes one of their operands",
                          fmt::format("group of {} at level {}", group.size(), lvl));
                report(3, fmt::format("skip group of {} einsums at level {} — interfering node between members", group.size(), lvl));
                continue;
            }
        }

        // Build the batched descriptor.
        BatchedGemmDescriptor d;
        d.m       = key.m;
        d.n       = key.n;
        d.k       = key.k;
        d.lda     = lda;
        d.ldb     = ldb;
        d.ldc     = ldc;
        d.trans_a = key.trans_a;
        d.trans_b = key.trans_b;
        // The descriptor carries the full complex prefactor; the batch key
        // (alpha_bits/beta_bits) already hashes the full value, so only einsums
        // with bit-identical prefactors (real and imaginary) are grouped here.
        d.alpha       = as<std::complex<double>>(first_desc->ab_prefactor);
        d.beta        = as<std::complex<double>>(first_desc->c_prefactor);
        d.batch_count = static_cast<int>(group.size());
        d.scalar      = key.scalar;

        // Collect the per-member extractors (ordered so a_array[i],
        // b_array[i], c_array[i] all reference the same original
        // contraction: preserves semantics when alpha*A*B+beta*C).
        std::vector<std::function<std::pair<void const *, int>()>> a_exs;
        std::vector<std::function<std::pair<void const *, int>()>> b_exs;
        std::vector<std::function<std::pair<void *, int>()>>       c_exs;
        a_exs.reserve(group.size());
        b_exs.reserve(group.size());
        c_exs.reserve(group.size());
        std::vector<TensorId> batched_inputs;  // [A_0, B_0, A_1, B_1, …]
        std::vector<TensorId> batched_outputs; // [C_0, C_1, …]
        batched_inputs.reserve(2 * group.size());
        batched_outputs.reserve(group.size());

        for (size_t const idx : group) {
            auto *g_desc = std::get_if<EinsumDescriptor>(&nodes[idx].op_data);
            a_exs.push_back(g_desc->gemm_hint->extract_a);
            b_exs.push_back(g_desc->gemm_hint->extract_b);
            c_exs.push_back(g_desc->gemm_hint->extract_c);
            batched_inputs.push_back(nodes[idx].inputs[0]);
            batched_inputs.push_back(nodes[idx].inputs[1]);
            batched_outputs.push_back(nodes[idx].outputs[0]);
        }

        // beta != 0 means gemm_batch READS every destination before writing it.
        // Only A and B are copied above, so without this the batched node claims
        // to overwrite each C without reading it, and the RAW edge from whoever
        // produced that C is gone. The members carried those reads themselves --
        // an accumulating einsum lists its output among its inputs (bug-1009) --
        // and collapsing them must not drop the convention. The failure is a
        // scheduling race, so it shows up intermittently as a destination read
        // before it is written.
        if (d.beta != std::complex<double>{0.0, 0.0}) {
            batched_inputs.insert(batched_inputs.end(), batched_outputs.begin(), batched_outputs.end());
        }

        // Shared with cg::batched_gemm so the two producers of a BatchedGemm
        // node cannot drift; see ComputeGraph/Detail/BatchedGemm.hpp.
        auto executor = detail::make_batched_gemm_executor(d, std::move(a_exs), std::move(b_exs), std::move(c_exs));

        // Construct the new node and mark originals for removal.
        Node batched;
        batched.kind    = OpKind::BatchedGemm;
        batched.label   = fmt::format("gemm_batch x{} ({}x{}x{}, trans={}{})", d.batch_count, d.m, d.k, d.n, d.trans_a, d.trans_b);
        batched.execute = std::move(executor);
        batched.inputs  = std::move(batched_inputs);
        batched.outputs = std::move(batched_outputs);
        batched.op_data = d;
        // Occupy the first member's slot (and reuse its id) so the batch stays
        // scan-before every consumer of a member's output; see the placement
        // gate above for why appending is unsound.
        batched.id       = nodes[first_pos].id;
        nodes[first_pos] = std::move(batched);

        for (size_t const idx : group)
            if (idx != first_pos)
                remove[idx] = true;

        _num_batches++;
        _total_batched += group.size();
        EINSUMS_LOG_INFO("GEMMBatching: batched {} einsums at level {} ({}×{}×{}) into gemm_batch", group.size(), lvl, key.m, key.k, key.n);
        report(2, fmt::format("batch {} independent einsums ({}x{}x{}) into one gemm_batch", group.size(), key.m, key.k, key.n));
    }

    if (_num_batches == num_batches_at_entry)
        return false;
    report(1, fmt::format("batched {} GEMM(s) into {} gemm_batch node(s)", _total_batched, _num_batches));

    // Compact: each batch already sits at its first member's slot, so
    // dropping the other members preserves a valid program order; only the
    // position-keyed dependency lists are stale. topological_sort() sees the
    // node-count change and rebuilds them without re-deriving the order.
    graph.erase_nodes(remove);
    graph.topological_sort();
    return true;
}

std::vector<std::string> GEMMBatching::explain() const {
    if (num_batches() == 0 && num_gate_skipped() == 0) {
        return {};
    }
    return {fmt::format("GEMMBatching: {} batch(es) absorbing {} GEMM(s); {} group(s) left parallel by the profitability gate",
                        num_batches(), total_batched(), num_gate_skipped())};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
