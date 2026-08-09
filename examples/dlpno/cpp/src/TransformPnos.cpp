//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Scaffolded by 'python -m einsums.stages promote' from dlpno.stages.
// Written once and never overwritten: this file is yours.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <dlpno/TransformPnos.hpp>
#include <string>
#include <vector>

namespace cg = einsums::compute_graph;
using einsums::linear_algebra::Transpose;

namespace dlpno {

namespace {

/// `std::vector<std::int64_t>` to the `size_t` vector the cg ops take. Same
/// rationale as ComputePnoOverlaps.cpp: the contract says `int64_t`, the ops
/// index memory.
std::vector<std::size_t> as_sizes(std::vector<std::int64_t> const &v) {
    return {v.begin(), v.end()};
}

/// Replay a setup graph as an OpenMP team over its independent nodes. Mirrors
/// DLPNOBase._run, including the executor choice; see ComputePnoOverlaps.cpp
/// for why it is never a thread pool.
void run_setup_graph(cg::Graph &graph) {
    graph.set_executor(std::make_shared<cg::OpenMPExecutor>());
    graph.execute();
}

einsums::SliceSpec spec_full() {
    return {};
}

einsums::SliceSpec spec_index(std::int64_t i) {
    einsums::SliceSpec s;
    s.kind  = einsums::SliceSpec::Kind::Index;
    s.index = i;
    return s;
}

einsums::SliceSpec spec_range(std::int64_t lo, std::int64_t hi) {
    einsums::SliceSpec s;
    s.kind  = einsums::SliceSpec::Kind::Range;
    s.start = lo;
    s.stop  = hi;
    return s;
}

/// A fresh tensor holding a copy of `src`'s values: `ten.from_numpy` without
/// the numpy. `src` is dense and freshly built here, so a flat copy is exact.
einsums::RuntimeTensor<double> copy_of(std::string const &name, einsums::RuntimeTensor<double> const &src) {
    std::vector<std::size_t> dims(src.rank());
    for (std::size_t d = 0; d < src.rank(); ++d) {
        dims[d] = src.dim(d);
    }
    einsums::RuntimeTensor<double> out(name, dims);
    out.zero();
    if (src.size() != 0) {
        std::memcpy(out.data(), src.data(), src.size() * sizeof(double));
    }
    return out;
}

/// Diagonalize many independent matrices in one graph, eigenpairs descending:
/// `dlpno.base.batched_eigh`, including the host-side reorder. The
/// parallelism is across matrices; each `syev` stays serial underneath.
std::pair<std::vector<einsums::RuntimeTensor<double>>, std::vector<einsums::RuntimeTensor<double>>>
batched_eigh(std::vector<einsums::RuntimeTensor<double> const *> const &mats, std::string const &label) {
    std::vector<einsums::RuntimeTensor<double>> evecs, evals;
    evecs.reserve(mats.size());
    evals.reserve(mats.size());
    for (auto const *A : mats) {
        evecs.emplace_back(copy_of(label + " vectors", *A));
        evals.emplace_back(einsums::create_zero_tensor<double>(label + " values", A->dim(0)));
    }
    if (!evecs.empty()) {
        cg::Graph g(label);
        {
            cg::CaptureGuard const guard(g);
            for (std::size_t k = 0; k < evecs.size(); ++k) {
                cg::syev<true>(&evecs[k], &evals[k]);
            }
        }
        run_setup_graph(g);
    }
    // Descending: reverse the eigenvalues and the eigenvector COLUMNS.
    // Column major, so a column is one contiguous block.
    for (std::size_t k = 0; k < evecs.size(); ++k) {
        auto const n = evals[k].dim(0);
        double    *w = evals[k].data();
        for (std::size_t a = 0; a < n / 2; ++a) {
            std::swap(w[a], w[n - 1 - a]);
        }
        double             *v = evecs[k].data();
        std::vector<double> tmp(n);
        for (std::size_t a = 0; a < n / 2; ++a) {
            double *lo = v + a * n;
            double *hi = v + (n - 1 - a) * n;
            std::memcpy(tmp.data(), lo, n * sizeof(double));
            std::memcpy(lo, hi, n * sizeof(double));
            std::memcpy(hi, tmp.data(), n * sizeof(double));
        }
    }
    return {std::move(evals), std::move(evecs)};
}

} // namespace

PnoTransform transform_pnos(einsums::RuntimeTensor<double> const &q_ia, std::vector<einsums::RuntimeTensor<double>> const &fits,
                            std::vector<std::int64_t> const &fit_of, std::vector<std::int64_t> const &fit_pos,
                            std::vector<einsums::RuntimeTensor<double>> const &dom_X,
                            std::vector<einsums::RuntimeTensor<double>> const &dom_e,
                            std::vector<einsums::RuntimeTensor<double>> const &dom_F, std::vector<std::int64_t> const &dom_of,
                            std::vector<std::vector<std::int64_t>> const &ribfs, std::vector<std::vector<std::int64_t>> const &paos,
                            std::vector<std::int64_t> const &lmo_j, std::vector<double> const &shift, std::vector<double> const &pno_scale,
                            std::int64_t min_pnos, double t_cut_pno) {
    auto const n_upper = fit_of.size();

    // The canonical dimension of each pair's domain: shape(K_pao)[0] on the
    // Python side, after the rotation into the canonical PAO basis.
    auto const ncan_of = [&](std::size_t u) { return dom_X[dom_of[u]].dim(1); };

    // ── Stage 1: everything up to the pair density ─────────────────────────
    //
    // Every operand is created before the capture so its address is stable:
    // the graph holds operands by pointer, and a vector that reallocates
    // mid-loop would silently retarget every earlier node.
    std::vector<einsums::RuntimeTensor<double>> blk, K_raw, K_half, K_can, D_pao, T_pao, Tt_pao, D_ij, D_ij_t, e_init, e_os_init;
    for (auto *v : {&blk, &K_raw, &K_half, &K_can, &D_pao, &T_pao, &Tt_pao, &D_ij, &D_ij_t, &e_init, &e_os_init}) {
        v->reserve(n_upper);
    }
    for (std::size_t u = 0; u < n_upper; ++u) {
        auto const nq   = ribfs[u].size();
        auto const nu   = paos[u].size();
        auto const ncan = ncan_of(u);
        blk.emplace_back(einsums::create_zero_tensor<double>("(Q|j a)", nq, std::size_t{1}, nu));
        K_raw.emplace_back(einsums::create_zero_tensor<double>("K (ia|jb)", nu, nu));
        K_half.emplace_back(einsums::create_zero_tensor<double>("doublet", ncan, nu));
        K_can.emplace_back(einsums::create_zero_tensor<double>("K (can PAO)", ncan, ncan));
        D_pao.emplace_back(einsums::create_zero_tensor<double>("denominator", ncan, ncan));
        T_pao.emplace_back(einsums::create_zero_tensor<double>("T (can PAO)", ncan, ncan));
        Tt_pao.emplace_back(einsums::create_zero_tensor<double>("Tt", ncan, ncan));
        D_ij.emplace_back(einsums::create_zero_tensor<double>("D (pair)", ncan, ncan));
        D_ij_t.emplace_back(einsums::create_zero_tensor<double>("doublet", ncan, ncan));
        e_init.emplace_back(einsums::create_zero_tensor<double>("e_ij", std::size_t{1}));
        e_os_init.emplace_back(einsums::create_zero_tensor<double>("e_ij os", std::size_t{1}));
    }

    cg::Graph g1("PNO stage 1");
    {
        cg::CaptureGuard const guard(g1);
        for (std::size_t u = 0; u < n_upper; ++u) {
            auto const &X = dom_X[dom_of[u]];
            auto const &e = dom_e[dom_of[u]];

            // The exchange operator over the pair's PAO domain: one gather
            // into a rank-3 block, one GEMM through its rank-2 view. The
            // length-1 middle axis exists so no host copy lands mid-loop;
            // (nq, 1, nu) column major has exactly the layout of (nq, nu).
            auto const fit2d = fits[fit_of[u]].at_view({spec_full(), spec_index(fit_pos[u]), spec_full()});
            cg::gather(&blk[u], q_ia, {as_sizes(ribfs[u]), {static_cast<std::size_t>(lmo_j[u])}, as_sizes(paos[u])});
            auto const blk2d = blk[u].at_view({spec_full(), spec_index(0), spec_full()});
            cg::gemm(1.0, fit2d, blk2d, 0.0, &K_raw[u], Transpose::T, Transpose::N);

            // Rotate into the canonical PAO basis (triplet, trans_a).
            cg::gemm(1.0, X, K_raw[u], 0.0, &K_half[u], Transpose::T, Transpose::N);
            cg::gemm(1.0, K_half[u], X, 0.0, &K_can[u]);

            // One semicanonical MP2 step: D[a,b] = shift - e_a - e_b, T = K / D.
            cg::outer_sum(&D_pao[u], std::vector<einsums::RuntimeTensor<double> const *>{&e, &e}, {-1.0, -1.0});
            cg::shift(shift[u], &D_pao[u]);
            cg::direct_division(1.0, K_can[u], D_pao[u], 0.0, &T_pao[u]);

            // Tt = 2 T - T^T.
            cg::permute("ab <- ba", &Tt_pao[u], T_pao[u]);
            cg::axpby(2.0, T_pao[u], -1.0, &Tt_pao[u]);

            // Pair density; eigenvectors are the PNOs, eigenvalues the
            // occupation numbers.
            cg::gemm(1.0, Tt_pao[u], T_pao[u], 0.0, &D_ij[u], Transpose::N, Transpose::T);
            cg::gemm(1.0, Tt_pao[u], T_pao[u], 0.0, &D_ij_t[u], Transpose::T, Transpose::N);
            cg::axpby(1.0, D_ij_t[u], 1.0, &D_ij[u]);

            // Pointer-writer dots: there is no value to return until the
            // graph runs.
            cg::dot_python(&e_init[u], K_can[u], Tt_pao[u]);
            cg::dot_python(&e_os_init[u], K_can[u], T_pao[u]);
        }
        // Guard released here: the graph is complete.
    }
    run_setup_graph(g1);

    std::vector<einsums::RuntimeTensor<double> const *> density;
    density.reserve(n_upper);
    for (auto const &D : D_ij) {
        density.push_back(&D);
    }
    auto [occs, pno_vecs] = batched_eigh(density, "PNO");

    // ── Stage 2: the truncation decision (data dependent, host code), then
    //    the Fock matrix in each surviving subspace ──────────────────────────
    std::vector<std::size_t> keep(n_upper, 0);
    for (std::size_t u = 0; u < n_upper; ++u) {
        auto const    nvir  = ncan_of(u);
        auto const    start = std::min<std::size_t>(static_cast<std::size_t>(min_pnos), nvir);
        double const  cut   = pno_scale[u] * t_cut_pno;
        std::size_t   k     = start;
        double const *occ   = occs[u].data();
        for (std::size_t a = start; a < nvir; ++a) {
            if (std::abs(occ[a]) >= cut) {
                ++k;
            }
        }
        keep[u] = k;
    }

    // The survivors are the LEADING columns (descending occupation), so the
    // truncation is a view rather than a gather. Held for the life of the
    // remaining graphs.
    std::vector<einsums::RuntimeTensorView<double>> X_trunc;
    X_trunc.reserve(n_upper);
    for (std::size_t u = 0; u < n_upper; ++u) {
        X_trunc.emplace_back(pno_vecs[u].at_view({spec_full(), spec_range(0, static_cast<std::int64_t>(keep[u]))}));
    }

    std::vector<einsums::RuntimeTensor<double>> F_half, Fmo;
    F_half.reserve(n_upper);
    Fmo.reserve(n_upper);
    for (std::size_t u = 0; u < n_upper; ++u) {
        F_half.emplace_back(einsums::create_zero_tensor<double>("doublet", keep[u], ncan_of(u)));
        Fmo.emplace_back(einsums::create_zero_tensor<double>("C^T F C", keep[u], keep[u]));
    }
    cg::Graph g2("PNO stage 2");
    {
        cg::CaptureGuard const guard(g2);
        for (std::size_t u = 0; u < n_upper; ++u) {
            // Orthonormal but not canonical yet; rotate so F is diagonal.
            cg::gemm(1.0, X_trunc[u], dom_F[dom_of[u]], 0.0, &F_half[u], Transpose::T, Transpose::N);
            cg::gemm(1.0, F_half[u], X_trunc[u], 0.0, &Fmo[u]);
        }
    }
    run_setup_graph(g2);

    std::vector<einsums::RuntimeTensor<double> const *> fock;
    fock.reserve(n_upper);
    for (auto const &F : Fmo) {
        fock.push_back(&F);
    }
    auto [e_pnos, canons] = batched_eigh(fock, "PNO canon");

    // ── Stage 3: rotate into the canonical PNO basis and record the pair ───
    //
    // The lower-triangle mirror (K_ji = K_ij^T) is the caller's: a host
    // transpose at scatter time, not arithmetic, so it stays outside the
    // contract.
    PnoTransform                                out;
    std::vector<einsums::RuntimeTensor<double>> K_h2, T_h2, Tt_h2, Tt_pno;
    for (auto *v : {&out.X_pno, &out.K_pno, &out.T_pno, &out.e_trunc, &out.e_os_trunc, &K_h2, &T_h2, &Tt_h2, &Tt_pno}) {
        v->reserve(n_upper);
    }
    for (std::size_t u = 0; u < n_upper; ++u) {
        auto const nu   = paos[u].size();
        auto const ncan = ncan_of(u);
        auto const k    = keep[u];
        out.X_pno.emplace_back(einsums::create_zero_tensor<double>("X (PAO->PNO)", nu, k));
        out.K_pno.emplace_back(einsums::create_zero_tensor<double>("K (PNO)", k, k));
        out.T_pno.emplace_back(einsums::create_zero_tensor<double>("T (PNO)", k, k));
        out.e_trunc.emplace_back(einsums::create_zero_tensor<double>("e trunc", std::size_t{1}));
        out.e_os_trunc.emplace_back(einsums::create_zero_tensor<double>("e os trunc", std::size_t{1}));
        K_h2.emplace_back(einsums::create_zero_tensor<double>("doublet", k, ncan));
        T_h2.emplace_back(einsums::create_zero_tensor<double>("doublet", k, ncan));
        Tt_h2.emplace_back(einsums::create_zero_tensor<double>("doublet", k, ncan));
        Tt_pno.emplace_back(einsums::create_zero_tensor<double>("Tt (PNO)", k, k));
    }
    // X (PNO) = X_trunc . canon, the per-pair rotation stage 3 shares.
    std::vector<einsums::RuntimeTensor<double>> X_rot;
    X_rot.reserve(n_upper);
    for (std::size_t u = 0; u < n_upper; ++u) {
        X_rot.emplace_back(einsums::create_zero_tensor<double>("X (PNO)", ncan_of(u), keep[u]));
    }
    cg::Graph g3("PNO stage 3");
    {
        cg::CaptureGuard const guard(g3);
        for (std::size_t u = 0; u < n_upper; ++u) {
            cg::gemm(1.0, X_trunc[u], canons[u], 0.0, &X_rot[u]);

            cg::gemm(1.0, X_rot[u], K_can[u], 0.0, &K_h2[u], Transpose::T, Transpose::N);
            cg::gemm(1.0, K_h2[u], X_rot[u], 0.0, &out.K_pno[u]);
            cg::gemm(1.0, X_rot[u], T_pao[u], 0.0, &T_h2[u], Transpose::T, Transpose::N);
            cg::gemm(1.0, T_h2[u], X_rot[u], 0.0, &out.T_pno[u]);
            cg::gemm(1.0, X_rot[u], Tt_pao[u], 0.0, &Tt_h2[u], Transpose::T, Transpose::N);
            cg::gemm(1.0, Tt_h2[u], X_rot[u], 0.0, &Tt_pno[u]);

            cg::dot_python(&out.e_trunc[u], out.K_pno[u], Tt_pno[u]);
            cg::dot_python(&out.e_os_trunc[u], out.K_pno[u], out.T_pno[u]);

            // PAO domain -> canonical PNO, in one transform.
            cg::gemm(1.0, dom_X[dom_of[u]], X_rot[u], 0.0, &out.X_pno[u]);
        }
    }
    run_setup_graph(g3);

    out.e_pno        = std::move(e_pnos);
    out.e_initial    = std::move(e_init);
    out.e_os_initial = std::move(e_os_init);
    out.n_pno.reserve(n_upper);
    for (std::size_t u = 0; u < n_upper; ++u) {
        out.n_pno.push_back(static_cast<std::int64_t>(keep[u]));
    }
    return out;
}

} // namespace dlpno
