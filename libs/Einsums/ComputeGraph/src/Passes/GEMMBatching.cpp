//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/GEMMBatching.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

// ── Typed batched executor dispatch ─────────────────────────────────────────
//
// At pass time we've type-erased all tensors to void*; the scalar tag on
// each group drives the final specialization.

template <typename T>
void run_batched_gemm(BatchedGemmDescriptor const &d, std::vector<void const *> const &a_vs, std::vector<void const *> const &b_vs,
                      std::vector<void *> const &c_vs) {
    std::vector<T const *> a_arr(d.batch_count);
    std::vector<T const *> b_arr(d.batch_count);
    std::vector<T *>       c_arr(d.batch_count);
    for (int i = 0; i < d.batch_count; ++i) {
        a_arr[i] = static_cast<T const *>(a_vs[i]);
        b_arr[i] = static_cast<T const *>(b_vs[i]);
        c_arr[i] = static_cast<T *>(c_vs[i]);
    }
    blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, static_cast<T>(d.alpha.real()), a_arr.data(), d.lda, b_arr.data(), d.ldb,
                        static_cast<T>(d.beta.real()), c_arr.data(), d.ldc, d.batch_count);
}

// The descriptor carries the full complex prefactor; preserve both parts when
// dispatching the complex batched GEMM (a complex einsum prefactor like a phase
// factor must not be truncated to its real part).
template <typename Complex>
void run_batched_gemm_complex(BatchedGemmDescriptor const &d, std::vector<void const *> const &a_vs, std::vector<void const *> const &b_vs,
                              std::vector<void *> const &c_vs) {
    using T = typename Complex::value_type;
    std::vector<Complex const *> a_arr(d.batch_count);
    std::vector<Complex const *> b_arr(d.batch_count);
    std::vector<Complex *>       c_arr(d.batch_count);
    for (int i = 0; i < d.batch_count; ++i) {
        a_arr[i] = static_cast<Complex const *>(a_vs[i]);
        b_arr[i] = static_cast<Complex const *>(b_vs[i]);
        c_arr[i] = static_cast<Complex *>(c_vs[i]);
    }
    Complex alpha{static_cast<T>(d.alpha.real()), static_cast<T>(d.alpha.imag())};
    Complex beta{static_cast<T>(d.beta.real()), static_cast<T>(d.beta.imag())};
    blas::gemm_batch<Complex>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, a_arr.data(), d.lda, b_arr.data(), d.ldb, beta, c_arr.data(),
                              d.ldc, d.batch_count);
}

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

bool GEMMBatching::run(Graph &graph) {
    graph.topological_sort();

    auto        &nodes   = graph.nodes();
    auto const  &deps    = graph.dependencies();
    size_t const n_nodes = nodes.size();

    _num_batches      = 0;
    _total_batched    = 0;
    _num_gate_skipped = 0;

    if (n_nodes < 2)
        return false;

    // Dependency levels (same computation OpenMPExecutor uses). Two
    // nodes at the same level have no path between them; safe to batch.
    std::vector<size_t> level(n_nodes, 0);
    for (size_t nd = 0; nd < n_nodes; ++nd) {
        for (size_t const pred : deps.predecessors[nd]) {
            if (level[pred] + 1 > level[nd])
                level[nd] = level[pred] + 1;
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
    std::vector<Node> new_nodes_to_append;

    for (auto const &[keyed_level, group] : groups) {
        if (group.size() < 2)
            continue;

        auto const &[lvl, key] = keyed_level;

        // Profitability gate: a batch executes as one node, and TaskPool
        // workers run BLAS single-threaded, so batching serializes work the
        // Dataflow executor would spread across workers. Only worth it when
        // each member is small enough that per-node scheduling overhead is a
        // meaningful fraction of its runtime.
        if (_has_profile) {
            size_t const elem_size = key.scalar == BlasScalar::Float          ? sizeof(float)
                                     : key.scalar == BlasScalar::Double       ? sizeof(double)
                                     : key.scalar == BlasScalar::ComplexFloat ? sizeof(std::complex<float>)
                                                                              : sizeof(std::complex<double>);
            double const gemm_us   = _profile.estimate_total_gemm_time_us(static_cast<size_t>(key.m), static_cast<size_t>(key.n),
                                                                          static_cast<size_t>(key.k), elem_size, Target::CPU);
            if (gemm_us > _max_gemm_us) {
                _num_gate_skipped++;
                EINSUMS_LOG_INFO("GEMMBatching: group of {} GEMMs ({}x{}x{}, ~{:.1f}us each) exceeds the {:.0f}us batching "
                                 "threshold — leaving them as independent nodes",
                                 group.size(), key.m, key.k, key.n, gemm_us, _max_gemm_us);
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
            report(3, fmt::format("skip group of {} einsums at level {} ({}x{}x{}) — mismatched strides", group.size(), lvl, key.m, key.k,
                                  key.n));
            continue;
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

        // Batched executor: pack pointers on every call (rebind may
        // have changed them), then dispatch to the typed gemm_batch.
        auto executor = [d, a_exs = std::move(a_exs), b_exs = std::move(b_exs), c_exs = std::move(c_exs)]() {
            std::vector<void const *> a_vs(d.batch_count);
            std::vector<void const *> b_vs(d.batch_count);
            std::vector<void *>       c_vs(d.batch_count);
            for (int i = 0; i < d.batch_count; ++i) {
                a_vs[i] = a_exs[i]().first;
                b_vs[i] = b_exs[i]().first;
                c_vs[i] = c_exs[i]().first;
            }
            switch (d.scalar) {
            case BlasScalar::Float:
                run_batched_gemm<float>(d, a_vs, b_vs, c_vs);
                break;
            case BlasScalar::Double:
                run_batched_gemm<double>(d, a_vs, b_vs, c_vs);
                break;
            case BlasScalar::ComplexFloat:
                run_batched_gemm_complex<std::complex<float>>(d, a_vs, b_vs, c_vs);
                break;
            case BlasScalar::ComplexDouble:
                run_batched_gemm_complex<std::complex<double>>(d, a_vs, b_vs, c_vs);
                break;
            }
        };

        // Construct the new node and mark originals for removal.
        Node batched;
        batched.kind    = OpKind::BatchedGemm;
        batched.label   = fmt::format("gemm_batch x{} ({}x{}x{}, trans={}{})", d.batch_count, d.m, d.k, d.n, d.trans_a, d.trans_b);
        batched.execute = std::move(executor);
        batched.inputs  = std::move(batched_inputs);
        batched.outputs = std::move(batched_outputs);
        batched.op_data = d;
        new_nodes_to_append.push_back(std::move(batched));

        for (size_t const idx : group)
            remove[idx] = true;

        _num_batches++;
        _total_batched += group.size();
        EINSUMS_LOG_INFO("GEMMBatching: batched {} einsums at level {} ({}×{}×{}) into gemm_batch", group.size(), lvl, key.m, key.k, key.n);
        report(2, fmt::format("batch {} independent einsums ({}x{}x{}) into one gemm_batch", group.size(), key.m, key.k, key.n));
    }

    if (_num_batches == 0)
        return false;
    report(1, fmt::format("batched {} GEMM(s) into {} gemm_batch node(s)", _total_batched, _num_batches));

    // Compact: drop originals in place, then hand the new BatchedGemm
    // nodes to Graph::add_node() so it assigns ids and invalidates the
    // topological sort. Ordering between the new nodes and survivors
    // is settled on the next sort, dependency edges are intact.
    std::vector<Node> filtered;
    filtered.reserve(nodes.size() - _total_batched);
    for (size_t i = 0; i < nodes.size(); ++i)
        if (!remove[i])
            filtered.push_back(std::move(nodes[i]));
    nodes = std::move(filtered);

    for (auto &nn : new_nodes_to_append)
        graph.add_node(std::move(nn));
    // add_node() clears the sort flag for us.
    return true;
}

} // namespace einsums::compute_graph::passes
