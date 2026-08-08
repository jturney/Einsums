//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <cstddef>
#include <dlpno/PnoOverlaps.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cg = einsums::compute_graph;

namespace dlpno {

namespace {

/// `std::vector<std::int64_t>` to the `size_t` vector the cg ops take.
///
/// The contract says `int64_t` because that is what a Python `int` maps to and
/// a negative index is a bug worth being able to represent; the ops index
/// memory and take `size_t`. The conversion is here, once, rather than at nine
/// call sites.
std::vector<std::size_t> as_sizes(std::vector<std::int64_t> const &v) {
    return {v.begin(), v.end()};
}

/// `0, 1, ... n-1`, which several of the gathers need for their untouched axes.
std::vector<std::size_t> iota(std::size_t n) {
    std::vector<std::size_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = i;
    }
    return out;
}

/// Replay a setup graph as an OpenMP team over its independent nodes.
///
/// Mirrors DLPNOBase._run on the Python side, including the executor choice.
/// Never a thread pool: these graphs run BLAS, and the OpenMP-built OpenBLAS
/// indexes its scratch by omp_get_thread_num(), which is 0 on every
/// caller-created std::thread. An OpenMP region is a real team.
void run_setup_graph(cg::Graph &graph) {
    graph.set_executor(std::make_shared<cg::OpenMPExecutor>());
    graph.execute();
}

} // namespace

PnoOverlaps compute_pno_overlaps(std::vector<einsums::RuntimeTensor<double>> const &X_pno, einsums::RuntimeTensor<double> const &S_pao,
                                 std::vector<std::vector<std::int64_t>> const &lmopair_to_paos, std::vector<std::int64_t> const &n_pno,
                                 std::vector<std::int64_t> const &bucket_of, std::vector<std::int64_t> const &bucket_dims,
                                 CouplingPlan const &plan) {
    auto const npao        = S_pao.dim(0);
    auto const n_lmo_pairs = n_pno.size();
    auto const n_cls       = plan.classes.size();

    auto const pair_dim = [&](std::int64_t ij) { return static_cast<std::size_t>(bucket_dims[bucket_of[ij]]); };

    // Each pair's PNO transform on the full PAO axis, padded to its bucket. The
    // scatters are captured with the GEMMs that consume them rather than run
    // here, so they parallelize too and the graph orders them itself.
    //
    // Held by value in a vector rather than by pointer: the graph keeps
    // operands alive through capture (M0), but these have to outlive THIS
    // function only until execute(), and a vector of RuntimeTensor is the
    // plainest thing that does it.
    std::vector<einsums::RuntimeTensor<double>> X_pad;
    X_pad.reserve(n_lmo_pairs);
    for (std::size_t ij = 0; ij < n_lmo_pairs; ++ij) {
        X_pad.emplace_back(n_pno[ij] ? einsums::create_zero_tensor<double>("X (padded) " + std::to_string(ij), npao,
                                                                           pair_dim(static_cast<std::int64_t>(ij)))
                                     : einsums::RuntimeTensor<double>{});
    }

    // One overlap tensor per shape class, in pair order, rank 2 so that a
    // coupling, a (pair, class) run, and the whole class are all column ranges
    // of it. The extra slot at the end is a reserved zero that the transposed
    // copy's padding reads from.
    PnoOverlaps out;
    out.S_cls.reserve(n_cls);
    for (std::size_t ci = 0; ci < n_cls; ++ci) {
        auto const M_b = static_cast<std::size_t>(bucket_dims[plan.classes[ci].first]);
        auto const M_p = static_cast<std::size_t>(bucket_dims[plan.classes[ci].second]);
        out.S_cls.emplace_back(
            einsums::create_zero_tensor<double>("S (class " + std::to_string(ci) + ")", M_b, M_p * (plan.slots_pair[ci] + 1)));
    }

    // One half per coupled pair, keyed by pair index. A map rather than a dense
    // vector because dead pairs have no entry and the lookup is not hot.
    std::map<std::int64_t, einsums::RuntimeTensor<double>> half;
    for (auto const ij : plan.coupled_pairs) {
        half.emplace(ij, einsums::create_zero_tensor<double>("half " + std::to_string(ij), pair_dim(ij), npao));
    }

    // The batches. Both GEMMs go out as batches over buckets rather than one
    // node per pair, which is what lets the partner concatenation go: gemm_batch
    // takes a single m/n/k and a single leading dimension for the whole batch,
    // and bucketing already guarantees every member of a group agrees on all of
    // them.
    //
    // This is the loop the whole promotion is about. On the Python side each
    // entry costs a pybind round trip building a view object; here the operands
    // are pointers into tensors that already exist, and the destination is
    // described by an offset rather than constructed at all.
    std::map<std::int64_t, std::vector<std::int64_t>> by_bucket;
    for (auto const ij : plan.coupled_pairs) {
        by_bucket[bucket_of[ij]].push_back(ij);
    }

    struct ShapeBatch {
        std::vector<einsums::RuntimeTensor<double> const *> a;
        std::vector<einsums::RuntimeTensor<double> const *> b;
        std::vector<std::size_t>                            offsets;
    };
    std::map<std::int64_t, ShapeBatch> by_shape;
    for (std::size_t n = 0; n < plan.pair.size(); ++n) {
        auto const ci  = plan.cls[n];
        auto const M_b = static_cast<std::size_t>(bucket_dims[plan.classes[ci].first]);
        auto const M_p = static_cast<std::size_t>(bucket_dims[plan.classes[ci].second]);
        auto      &sb  = by_shape[ci];
        sb.a.push_back(&half.at(plan.pair[n]));
        sb.b.push_back(&X_pad[plan.partner[n]]);
        // Column block `slot` of a (M_b, M_p * N) column-major tensor starts at
        // element slot * M_p * M_b and inherits the parent's leading dimension,
        // which is exactly what batched_gemm_blocked wants.
        sb.offsets.push_back(static_cast<std::size_t>(plan.dest_slot[n]) * M_p * M_b);
    }

    cg::Graph g("pno overlaps");
    {
        cg::CaptureGuard const guard(g);

        for (std::size_t ij = 0; ij < n_lmo_pairs; ++ij) {
            if (n_pno[ij] == 0) {
                continue;
            }
            cg::scatter(&X_pad[ij], X_pno[ij], {as_sizes(lmopair_to_paos[ij]), iota(static_cast<std::size_t>(n_pno[ij]))});
        }

        for (auto const &[bucket, members] : by_bucket) {
            std::vector<einsums::RuntimeTensor<double> const *> a, b;
            std::vector<einsums::RuntimeTensor<double> *>       c;
            a.reserve(members.size());
            b.reserve(members.size());
            c.reserve(members.size());
            for (auto const ij : members) {
                a.push_back(&X_pad[ij]);
                b.push_back(&S_pao);
                c.push_back(&half.at(ij));
            }
            cg::batched_gemm(1.0, a, b, 0.0, c, /*trans_a=*/true);
        }

        for (auto const &[ci, sb] : by_shape) {
            auto const M_b = static_cast<std::size_t>(bucket_dims[plan.classes[ci].first]);
            auto const M_p = static_cast<std::size_t>(bucket_dims[plan.classes[ci].second]);
            cg::batched_gemm_blocked(1.0, sb.a, sb.b, 0.0, &out.S_cls[ci], sb.offsets, M_b, M_p);
        }
        // Guard released here: the graph is complete.
    }
    // Parallelism across the batch, each member serial underneath. The OpenMP
    // executor rather than Dataflow because Dataflow runs nodes on std::thread
    // workers, and the OpenMP-built OpenBLAS conda resolves by default indexes
    // its internal scratch by omp_get_thread_num(), which is 0 on every
    // caller-created thread. An OpenMP region is a real team, so the same BLAS
    // is safe underneath it.
    //
    // No pass pipeline, for the reason the Python side documents: this graph is
    // already in the form the passes would produce, so applying them is pure
    // cost.
    run_setup_graph(g);

    // One vectorized scaling per class: every coupling owns a column range.
    for (std::size_t ci = 0; ci < n_cls; ++ci) {
        auto const          M_p = static_cast<std::size_t>(bucket_dims[plan.classes[ci].second]);
        std::vector<double> factors(M_p * (plan.slots_pair[ci] + 1), 1.0);
        for (std::size_t n = 0; n < plan.pair.size(); ++n) {
            if (plan.cls[n] != static_cast<std::int64_t>(ci)) {
                continue;
            }
            double const s = std::sqrt(std::abs(plan.factor[n]));
            auto const   d = static_cast<std::size_t>(plan.dest_slot[n]) * M_p;
            for (std::size_t k = 0; k < M_p; ++k) {
                factors[d + k] = s;
            }
        }
        // Column-major and contiguous, so element (r, c) is data[r + c * M_b]
        // and a whole column shares one factor.
        //
        // Through data() rather than operator(). The runtime-rank accessor
        // recomputes an offset from the stride vector on every element, and
        // this loop touches 18.8M of them at ethanol/cc-pVTZ; the first version
        // did exactly that and made the C++ backend 8.6x SLOWER than the Python
        // one it replaced. The Python it replaced was not doing scalar work at
        // all - it was one vectorized numpy statement running at memory
        // bandwidth. See the note on this function.
        auto      &S                  = out.S_cls[ci];
        auto const M_b                = S.dim(0);
        auto const cols               = S.dim(1);
        double *__restrict const data = S.data();
        for (std::size_t c = 0; c < cols; ++c) {
            double const f = factors[c];
            if (f == 1.0) {
                continue;
            }
            double *const column = data + c * M_b;
            for (std::size_t r = 0; r < M_b; ++r) {
                column[r] *= f;
            }
        }
    }

    // The partner-major transposed copy the first half consumes, lifted from
    // the pair-major one by one permuting gather per class. It reads S AFTER
    // the scaling, so it inherits sqrt|f| and only the sign is left to apply.
    out.S_T.reserve(n_cls);
    std::vector<einsums::RuntimeTensorView<double>> src_views;
    src_views.reserve(n_cls);
    for (std::size_t ci = 0; ci < n_cls; ++ci) {
        auto const M_b = static_cast<std::size_t>(bucket_dims[plan.classes[ci].first]);
        auto const M_p = static_cast<std::size_t>(bucket_dims[plan.classes[ci].second]);
        out.S_T.emplace_back(einsums::create_zero_tensor<double>("S^T (class " + std::to_string(ci) + ")", M_p, M_b,
                                                                 static_cast<std::size_t>(plan.slots_partner[ci])));
        src_views.emplace_back(out.S_cls[ci].reshape_view({M_b, M_p, static_cast<std::size_t>(plan.slots_pair[ci] + 1)}));
    }

    cg::Graph g2("pno overlaps transposed");
    {
        cg::CaptureGuard const guard(g2);
        for (std::size_t ci = 0; ci < n_cls; ++ci) {
            auto const M_b = static_cast<std::size_t>(bucket_dims[plan.classes[ci].first]);
            auto const M_p = static_cast<std::size_t>(bucket_dims[plan.classes[ci].second]);
            cg::gather(&out.S_T[ci], src_views[ci], {iota(M_b), iota(M_p), as_sizes(plan.inv_perm[ci])}, {1, 0, 2});
        }
    }
    run_setup_graph(g2);

    // The sign rides on this copy only.
    for (std::size_t n = 0; n < plan.pair.size(); ++n) {
        if (plan.sign[n] >= 0.0) {
            continue;
        }
        // One coupling owns slot s entirely, and (p, b, s) is contiguous in
        // p then b, so the slot is one contiguous M_p * M_b block. Same reason
        // as the scaling above for going through data().
        auto      &T                   = out.S_T[plan.cls[n]];
        auto const slot                = static_cast<std::size_t>(plan.src_slot[n]);
        auto const block               = T.dim(0) * T.dim(1);
        double *__restrict const first = T.data() + slot * block;
        for (std::size_t k = 0; k < block; ++k) {
            first[k] = -first[k];
        }
    }

    return out;
}

} // namespace dlpno
