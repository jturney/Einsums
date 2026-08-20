//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Scaffolded by 'python -m einsums.stages promote' from dlpno.stages.
// Written once and never overwritten: this file is yours.

#include <Einsums/ComputeGraph.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <dlpno/ComputePnoIntegrals.hpp>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if !defined(_OPENMP)
// A silently serial loop is a performance regression that looks exactly like a
// correct port, and threading the per-pair loop below is what this file is for.
// Fail the build instead of shipping one.
#    error "dlpno stage module: OpenMP is required (the per-pair loop below is the point of the port)."
#endif

namespace cg = einsums::compute_graph;
using einsums::linear_algebra::Transpose;

namespace dlpno {

namespace {

/// `std::vector<std::int64_t>` to the `size_t` vector the cg ops take. Same
/// rationale as the other two ports: the contract says `int64_t`, the ops index
/// memory.
std::vector<std::size_t> as_sizes(std::vector<std::int64_t> const &v) {
    return {v.begin(), v.end()};
}

einsums::RuntimeTensor<double> zeros(std::string const &name, std::vector<std::size_t> const &dims) {
    einsums::RuntimeTensor<double> out(name, dims);
    out.zero();
    return out;
}

/// The placeholder for a block a pair does not have: `dlpno.cc_integrals._absent`.
einsums::RuntimeTensor<double> absent() {
    return zeros("absent", {0, 0});
}

/// `metric[np.ix_(idx, idx)]` as a fresh matrix, which is what
/// `sparse.submatrix_rows_and_cols` gathers. A copy, so no arithmetic to match.
einsums::RuntimeTensor<double> submatrix(einsums::RuntimeTensor<double> const &metric, std::vector<std::int64_t> const &idx) {
    auto const                     n = idx.size();
    einsums::RuntimeTensor<double> out("(P|Q) domain", {n, n});
    double                        *dst = out.data();
    double const                  *src = metric.data();
    auto const                     ld  = metric.dim(0);
    for (std::size_t c = 0; c < n; ++c) {
        for (std::size_t r = 0; r < n; ++r) {
            dst[r + c * n] = src[static_cast<std::size_t>(idx[r]) + static_cast<std::size_t>(idx[c]) * ld];
        }
    }
    return out;
}

/// Every three-index block one pair is built from, plus the four
/// full-inverse-fitted copies the non-projected families read.
///
/// `q_ov` and `q_vv` are POINTERS, and to the returned `Qma`/`Qab` themselves
/// wherever the pair is strong. They are the two largest blocks the phase keeps
/// - `nq * na^2` for `Qab` - and `RuntimeTensor` has no move assignment, so
/// handing them over at the end would deep-copy them and double the phase's
/// peak. Written into their final home instead, which is also what the Python
/// does by sharing the object.
///
/// They are allocated rank 3 rather than rank 2 and reshaped where a GEMM wants
/// a matrix, which is the mirror image of the Python. Same memory either way.
struct Raw {
    std::size_t                     nq{}, nu{}, nk{}, na{};
    einsums::RuntimeTensor<double>  q_pair, q_io, q_jo, q_iv, q_jv;
    einsums::RuntimeTensor<double> *q_ov{nullptr}, *q_vv{nullptr};
    einsums::RuntimeTensor<double>  q_io_inv, q_jo_inv, q_iv_inv, q_jv_inv;
};

/// Copy `w` columns of a column-major `(nq, ...)` block into or out of a stacked
/// right-hand side. Both sides are contiguous, so this is the memcpy that the
/// Python's `view[:, at:at + w] = ...` compiles down to.
void copy_columns(double *dst, double const *src, std::size_t nq, std::size_t w) {
    if (w != 0) {
        std::memcpy(dst, src, nq * w * sizeof(double));
    }
}

} // namespace

PnoIntegralBlocks
compute_pno_integrals(einsums::RuntimeTensor<double> const &q_ij, einsums::RuntimeTensor<double> const &q_ia,
                      einsums::RuntimeTensor<double> const &q_ab, einsums::RuntimeTensor<double> const &metric,
                      std::vector<einsums::RuntimeTensor<double>> const &X_pno, std::vector<std::int64_t> const &i_lmo,
                      std::vector<std::int64_t> const &j_lmo, std::vector<std::int64_t> const &n_pno, std::vector<bool> const &strong,
                      std::vector<std::vector<std::int64_t>> const &ribfs, std::vector<std::vector<std::int64_t>> const &paos,
                      std::vector<std::vector<std::int64_t>> const &lmos, std::vector<std::vector<std::int64_t>> const &extended,
                      std::vector<einsums::RuntimeTensor<double>> const &rot_X, std::vector<std::vector<std::int64_t>> const &rot_paos,
                      std::vector<std::vector<std::int64_t>> const &nb_ij, std::vector<std::vector<std::int64_t>> const &nb_ji) {
    auto const n_pairs = i_lmo.size();

    // ── The shape of this port: one fused pass over pairs ──────────────────
    //
    // The first port of this stage ran as five phases - a gather graph, a
    // half-transform graph, a fit loop, a contraction graph, a non-projected
    // loop - with a barrier between each. Measured at ethanol/cc-pVTZ on 10
    // threads that shape spent 3.1 s of its 5.6 s in the first four phases,
    // against 2.0 s for psi4's WHOLE phase, and the difference was traffic,
    // not kernels: the raw gathers are the widest thing the phase touches
    // (`uv_blk` alone is `nq * nu^2` per pair, gigabytes over a chunk), and
    // the staged shape wrote all of them to memory in one phase and read them
    // all back in the next.
    //
    // So the port now has psi4's own shape (dlpno/ccsd.cc,
    // DLPNOCCSD::compute_pno_integrals): every pair runs its whole chain -
    // gather, half transform, fit, contract, non-projected - inside one
    // `omp parallel for schedule(dynamic, 1)` over pairs sorted
    // most-expensive-first. A pair's gathers are locals of its iteration,
    // consumed while cache-warm and dropped at scope end, so a team's worth of
    // them is live rather than a chunk's worth. The barriers go with them: no
    // pair waits for every other pair's gathers.
    //
    // The arithmetic is untouched: each pair issues the SAME einsums
    // operations on the SAME values in the same per-pair order as the staged
    // shape and as the Python backend - eagerly instead of via a captured
    // graph, which dispatches to the same kernels. Pairs share no data (every
    // cross-pair input, `rot_X`/`rot_paos`, is a read-only argument), so the
    // fusion moves no bit; the differential test asserts that against the
    // Python backend.

    // ── Outputs and per-pair homes, allocated up front ─────────────────────
    //
    // Every output tensor is created before the parallel loop and every vector
    // is reserved rather than grown: the loop writes through `Raw`'s pointers
    // into `Qma`/`Qab`, so a vector that reallocated mid-loop would retarget
    // them - and a concurrent emplace_back is a data race besides.
    PnoIntegralBlocks out;
    out.Qma.reserve(n_pairs);
    out.Qab.reserve(n_pairs);
    std::vector<einsums::RuntimeTensor<double>> spare_ov, spare_vv;
    spare_ov.reserve(n_pairs);
    spare_vv.reserve(n_pairs);

    std::vector<Raw> raw(n_pairs);
    for (std::size_t p = 0; p < n_pairs; ++p) {
        auto &r  = raw[p];
        r.nq     = ribfs[p].size();
        r.nu     = paos[p].size();
        r.nk     = lmos[p].size();
        r.na     = static_cast<std::size_t>(n_pno[p]);
        r.q_pair = zeros("(Q|i j)", {r.nq, 1});
        r.q_io   = zeros("(Q|i m)", {r.nq, r.nk});
        r.q_jo   = zeros("(Q|j m)", {r.nq, r.nk});
        r.q_iv   = zeros("(Q|i a)", {r.nq, r.na});
        r.q_jv   = zeros("(Q|j a)", {r.nq, r.na});
        // A weak pair returns no DF factor, so its two big blocks are scratch
        // and a strong pair's are the output. Either way they are allocated
        // exactly once and never handed over.
        bool const keep = strong[p];
        out.Qma.emplace_back(keep ? zeros("(Q|m a)", {r.nq, r.nk, r.na}) : absent());
        out.Qab.emplace_back(keep ? zeros("(Q|a b)", {r.nq, r.na, r.na}) : absent());
        spare_ov.emplace_back(keep ? absent() : zeros("(Q|m a)", {r.nq, r.nk, r.na}));
        spare_vv.emplace_back(keep ? absent() : zeros("(Q|a b)", {r.nq, r.na, r.na}));
        r.q_ov = keep ? &out.Qma[p] : &spare_ov[p];
        r.q_vv = keep ? &out.Qab[p] : &spare_vv[p];
    }

    for (auto *v : {&out.K_mibj, &out.J_ijmb, &out.K_ivvv, &out.K_mjai, &out.K_jvvv, &out.i_Qk, &out.i_Qa, &out.j_Qk, &out.j_Qa}) {
        v->reserve(n_pairs);
    }
    for (std::size_t p = 0; p < n_pairs; ++p) {
        auto const &r        = raw[p];
        bool const  off_diag = i_lmo[p] != j_lmo[p];
        out.K_mibj.emplace_back(zeros("K (m i|b j)", {r.nk, r.na}));
        out.J_ijmb.emplace_back(zeros("J (i j|m b)", {r.nk, r.na}));
        out.K_ivvv.emplace_back(zeros("K (i e|a f)", {r.na, r.na, r.na}));
        out.K_mjai.emplace_back(off_diag ? zeros("K (m j|a i)", {r.nk, r.na}) : absent());
        out.K_jvvv.emplace_back(off_diag ? zeros("K (j e|a f)", {r.na, r.na, r.na}) : absent());
    }

    // The non-projected output slabs are placed, never appended: slot s of
    // pair p lands at offset(p) + s, the prefix sum over the neighbour-slot
    // counts, which is the walk the caller replays to put the ``None``s back.
    // Placement is also what lets the fused loop fill them concurrently.
    std::vector<std::size_t> base_ij(n_pairs, 0), base_ji(n_pairs, 0);
    std::size_t              slots_ij = 0, slots_ji = 0;
    for (std::size_t p = 0; p < n_pairs; ++p) {
        base_ij[p] = slots_ij;
        base_ji[p] = slots_ji;
        slots_ij += nb_ij[p].size();
        slots_ji += nb_ji[p].size();
    }
    for (auto const &[vec, n] : {std::pair{&out.J_ikac, slots_ij}, std::pair{&out.K_iakc, slots_ij}, std::pair{&out.J_jkac, slots_ji},
                                 std::pair{&out.K_jakc, slots_ji}}) {
        vec->reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            vec->emplace_back(absent());
        }
    }

    // ── Pairs, most expensive first ────────────────────────────────────────
    //
    // psi4's own load-balancing: under dynamic scheduling the tail of the loop
    // is whatever was claimed last, so the fat pairs have to go first or one
    // of them IS the tail. The cost model is psi4's too - the flops of the
    // half transforms plus, for a strong pair, the extended-domain contraction
    // that dominates the non-projected families.
    std::vector<std::size_t> cost(n_pairs, 0);
    for (std::size_t p = 0; p < n_pairs; ++p) {
        auto const &r = raw[p];
        cost[p]       = r.nq * r.nu * r.nu * r.na + r.nq * r.nu * r.na * r.na + r.nq * r.nu * r.nk * r.na;
        if (strong[p]) {
            cost[p] += r.nq * extended[p].size() * r.nu * r.na;
        }
    }
    std::vector<std::size_t> order(n_pairs);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return cost[a] > cost[b]; });

    // One neighbour's slab, rotated into that neighbour's PNOs.
    //
    // `block` is `(nlmo, na * ne)`: one row per neighbour, holding a
    // (PNO, extended PAO) slab flattened into the second axis. Which of the two
    // runs fastest there is not a convention to pick but a consequence of
    // column major, and the two families disagree: a (nq, na, ne) block
    // flattened to (nq, na*ne) has W = a + na*e, while a (nk*ne, na) result
    // reinterpreted as (nk, ne*na) has W = e + ne*a. Reading either with the
    // other's stride silently transposes every slab.
    //
    // Both branches reproduce the numpy expression the Python backend writes,
    // down to which matrix the BLAS gets as its left operand: the Python's
    // `(X^T slab)^T` is `slab^T X` on column-major operands, and its
    // `slab X` is `(X^T slab^T)^T`. Verified bit-identical rather than assumed.
    auto slice_neighbours = [&](std::vector<std::int64_t> const &slots, std::vector<einsums::RuntimeTensor<double>> *into, std::size_t base,
                                std::vector<std::int64_t> const &ext, einsums::RuntimeTensor<double> const &block, std::size_t na,
                                std::size_t ne, bool pno_fastest, std::string const &name) {
        std::unordered_map<std::int64_t, std::size_t> where;
        where.reserve(ext.size() * 2);
        for (std::size_t pos = 0; pos < ext.size(); ++pos) {
            where.emplace(ext[pos], pos);
        }
        auto const    nk  = block.dim(0);
        double const *src = block.data();
        for (std::size_t k = 0; k < slots.size(); ++k) {
            auto const slot = slots[k];
            if (slot < 0) {
                continue;
            }
            auto const &X_kj  = rot_X[static_cast<std::size_t>(slot)];
            auto const &dom   = rot_paos[static_cast<std::size_t>(slot)];
            auto const  nkeep = dom.size();
            auto const  nakj  = X_kj.dim(1);

            // Written into its slot rather than assigned into it afterwards:
            // RuntimeTensor has no move assignment, so a local result would be
            // deep-copied on the way in.
            auto &dest = (*into)[base + k];
            dest       = zeros(name, {na, nakj});

            if (pno_fastest) {
                // W = a + na*e. The transposed slab (a_ij, u_kj) is what the
                // GEMM wants, and it is what numpy's C-order (ne, na)[keep, :]
                // slice hands the BLAS.
                auto    slab = zeros("slab", {na, nkeep});
                double *dst  = slab.data();
                for (std::size_t c = 0; c < nkeep; ++c) {
                    auto const e = where.at(dom[c]);
                    for (std::size_t a = 0; a < na; ++a) {
                        dst[a + c * na] = src[k + nk * (a + na * e)];
                    }
                }
                cg::gemm(1.0, slab, X_kj, 0.0, &dest);
            } else {
                // W = e + ne*a. Here the slab arrives (a_ij, u_kj) and numpy
                // multiplies it on the LEFT, which in column-major terms is
                // X^T slab^T with the result transposed on the way out.
                auto    slab = zeros("slab", {nkeep, na});
                double *dst  = slab.data();
                for (std::size_t a = 0; a < na; ++a) {
                    for (std::size_t r = 0; r < nkeep; ++r) {
                        dst[r + a * nkeep] = src[k + nk * (where.at(dom[r]) + ne * a)];
                    }
                }
                auto product = zeros("rotated^T", {nakj, na});
                cg::gemm(1.0, X_kj, slab, 0.0, &product, Transpose::T, Transpose::N);
                cg::permute("ab <- ba", &dest, product);
            }
        }
    };

    // ── The fused pass ─────────────────────────────────────────────────────
    //
    // Every pair is independent, its gathers and right-hand sides are
    // transient, and the BLAS and LAPACK underneath run serial inside the team
    // - which is why this is an OpenMP team and never a caller-created thread
    // pool (trap 7: the OpenMP-built OpenBLAS indexes its scratch by
    // omp_get_thread_num, which is 0 on every non-OpenMP thread, and returns
    // silently wrong numbers).
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t idx = 0; idx < n_pairs; ++idx) {
        auto const p        = order[idx];
        auto      &r        = raw[p];
        bool const off_diag = i_lmo[p] != j_lmo[p];
        auto const nq       = r.nq;
        auto const nu       = r.nu;
        auto const nk       = r.nk;
        auto const na       = r.na;
        auto const qs       = as_sizes(ribfs[p]);
        auto const us       = as_sizes(paos[p]);
        auto const ms       = as_sizes(lmos[p]);

        // ── Gather and half-transform, inside one scope ────────────────────
        //
        // The gathers land in rank-3 blocks whose views the transforms read; a
        // single-LMO selection leaves a length-1 axis, which costs nothing in
        // column major and saves a host copy. They are locals of this scope so
        // that `uv_blk` - `nq * nu^2`, an order above any block the phase
        // produces - is dropped the moment its last transform has run, while
        // still cache-warm from being written.
        {
            std::vector<std::size_t> const i{static_cast<std::size_t>(i_lmo[p])};
            std::vector<std::size_t> const j{static_cast<std::size_t>(j_lmo[p])};

            auto ij_blk = zeros("(Q|i j) raw", {nq, 1, 1});
            cg::gather(&ij_blk, q_ij, {qs, i, j});
            auto io_blk = zeros("(Q|i m) raw", {nq, 1, nk});
            cg::gather(&io_blk, q_ij, {qs, i, ms});
            auto jo_blk = zeros("(Q|j m) raw", {nq, 1, nk});
            cg::gather(&jo_blk, q_ij, {qs, j, ms});
            auto iu_blk = zeros("(Q|i u) raw", {nq, 1, nu});
            cg::gather(&iu_blk, q_ia, {qs, i, us});
            auto ju_blk = zeros("(Q|j u) raw", {nq, 1, nu});
            cg::gather(&ju_blk, q_ia, {qs, j, us});
            auto mu_blk = zeros("(Q|m u) raw", {nq, nk, nu});
            cg::gather(&mu_blk, q_ia, {qs, ms, us});
            auto uv_blk = zeros("(Q|u v) raw", {nq, nu, nu});
            cg::gather(&uv_blk, q_ab, {qs, us, us});

            auto const &X = X_pno[p];

            // (Q|i j) and (Q|i m) need no transform, only a reshape of the
            // length-1 axis away, which a view already is.
            cg::axpby(1.0, ij_blk.reshape_view({nq, 1}), 0.0, &r.q_pair);
            cg::axpby(1.0, io_blk.reshape_view({nq, nk}), 0.0, &r.q_io);
            cg::axpby(1.0, jo_blk.reshape_view({nq, nk}), 0.0, &r.q_jo);
            // (Q|i u) X -> (Q|i a), and the same for j.
            cg::gemm(1.0, iu_blk.reshape_view({nq, nu}), X, 0.0, &r.q_iv);
            cg::gemm(1.0, ju_blk.reshape_view({nq, nu}), X, 0.0, &r.q_jv);
            // (Q m | u) X -> (Q m | a). The (Q, m) axes are adjacent and both
            // lead, so the whole block is one GEMM through a merged view.
            auto q_ov_2d = r.q_ov->reshape_view({nq * nk, na});
            cg::gemm(1.0, mu_blk.reshape_view({nq * nk, nu}), X, 0.0, &q_ov_2d);
            // X^T (Q | u v) X, as two einsums rather than a GEMM pair per
            // auxiliary function - see the Python for why that shape is a trap.
            auto half = zeros("(Q|a v) half", {nq, na, nu});
            cg::einsum("Qav <- Quv ; ua", &half, uv_blk, X);
            cg::einsum("Qab <- Qav ; vb", r.q_vv, half, X);
        }

        // ── The fit, at the two metric powers psi4 uses ────────────────────
        auto const A = submatrix(metric, ribfs[p]);

        // The full-inverse copies, kept: they are the fitted side of the two
        // non-projected families, whose other side is unfitted. Applying the
        // symmetric power to both would double-count the metric. Only a strong
        // pair HAS non-projected families to read these, so only a strong pair
        // pays the solve; no output moves, because the copies are not output.
        if (strong[p]) {
            auto    rhs = zeros("full-inverse rhs", {nq, 2 * nk + 2 * na});
            double *at  = rhs.data();
            copy_columns(at, r.q_io.data(), nq, nk);
            copy_columns(at + nq * nk, r.q_jo.data(), nq, nk);
            copy_columns(at + nq * 2 * nk, r.q_iv.data(), nq, na);
            copy_columns(at + nq * (2 * nk + na), r.q_jv.data(), nq, na);
            auto Acopy  = A; // gesv overwrites its left-hand side
            std::ignore = cg::gesv(&Acopy, &rhs);

            double const *got = rhs.data();
            r.q_io_inv        = zeros("J^-1 (Q|i m)", {nq, nk});
            r.q_jo_inv        = zeros("J^-1 (Q|j m)", {nq, nk});
            r.q_iv_inv        = zeros("J^-1 (Q|i a)", {nq, na});
            r.q_jv_inv        = zeros("J^-1 (Q|j a)", {nq, na});
            copy_columns(r.q_io_inv.data(), got, nq, nk);
            copy_columns(r.q_jo_inv.data(), got + nq * nk, nq, nk);
            copy_columns(r.q_iv_inv.data(), got + nq * 2 * nk, nq, na);
            copy_columns(r.q_jv_inv.data(), got + nq * (2 * nk + na), nq, na);
        }

        // Then the symmetric fit, which leaves every primary block a DF factor:
        // one solve per pair over all of them at once, since they share the
        // metric and gesv factorizes once for all columns.
        {
            std::pair<double *, std::size_t> const widths[] = {
                {r.q_pair.data(), 1}, {r.q_io.data(), nk},       {r.q_jo.data(), nk},       {r.q_iv.data(), na},
                {r.q_jv.data(), na},  {r.q_ov->data(), nk * na}, {r.q_vv->data(), na * na},
            };
            std::size_t total = 0;
            for (auto const &[data, w] : widths) {
                total += w;
            }
            auto        rhs = zeros("symmetric fit rhs", {nq, total});
            std::size_t at  = 0;
            for (auto const &[data, w] : widths) {
                copy_columns(rhs.data() + nq * at, data, nq, w);
                at += w;
            }
            // power(0.5) is a matrix square root through an eigendecomposition;
            // the cutoff is its eigenvalue floor and matters only for a
            // near-singular auxiliary basis.
            auto A_half = cg::pow(A, 0.5);
            std::ignore = cg::gesv(&A_half, &rhs);
            at          = 0;
            for (auto const &[data, w] : widths) {
                copy_columns(data, rhs.data() + nq * at, nq, w);
                at += w;
            }
        }

        // ── The contracted integrals, B^T B over the pair's auxiliary domain ─

        // (m i | b j) and its ji partner (m j | a i).
        cg::gemm(1.0, r.q_io, r.q_jv, 0.0, &out.K_mibj[p], Transpose::T, Transpose::N);
        if (off_diag) {
            cg::gemm(1.0, r.q_jo, r.q_iv, 0.0, &out.K_mjai[p], Transpose::T, Transpose::N);
        }

        // (i j | m b), built as a COLUMN and reshaped rather than as a row.
        // The pair factor is (Q, 1), so contracting it on the left would
        // make this a GEMM with one row, which is the degenerate shape that
        // cost the iteration a factor of two in Eq. 84c. The reshape is
        // exact rather than a reinterpretation: a column-major (nk*na, 1)
        // and a (nk, na) put element (k, a) at the same offset k + nk*a.
        auto J_col = out.J_ijmb[p].reshape_view({nk * na, 1});
        cg::gemm(1.0, r.q_ov->reshape_view({r.nq, nk * na}), r.q_pair, 0.0, &J_col, Transpose::T, Transpose::N);

        // (i e | a f), as (e, a, f).
        auto const q_vv_2d = r.q_vv->reshape_view({r.nq, na * na});
        auto       K_i_2d  = out.K_ivvv[p].reshape_view({na, na * na});
        cg::gemm(1.0, r.q_iv, q_vv_2d, 0.0, &K_i_2d, Transpose::T, Transpose::N);
        if (off_diag) {
            auto K_j_2d = out.K_jvvv[p].reshape_view({na, na * na});
            cg::gemm(1.0, r.q_jv, q_vv_2d, 0.0, &K_j_2d, Transpose::T, Transpose::N);
        }

        // ── The non-projected families ─────────────────────────────────────
        //
        // The extended-domain gathers are the widest thing this phase touches,
        // wider even than `uv_blk`. As iteration locals they are live for one
        // pair per thread, which is what lets the caller's memory budget charge
        // a chunk for the largest of them ONCE rather than for the sum over
        // its members.
        if (strong[p]) {
            auto const es = as_sizes(extended[p]);
            auto const ne = es.size();

            // (Q | a_ij v_ext): half in the pair's PNOs, half in the extended PAOs.
            auto uv_ext = zeros("(Q|u v_ext) raw", {nq, r.nu, ne});
            cg::gather(&uv_ext, q_ab, {qs, us, es});
            auto q_av = zeros("(Q|a v_ext)", {nq, na, ne});
            cg::einsum("Qae <- Que ; ua", &q_av, uv_ext, X_pno[p]);
            // (Q | m u_ext), raw and unfitted: the K_iakc side.
            auto mu_ext = zeros("(Q|m u_ext) raw", {nq, nk, ne});
            cg::gather(&mu_ext, q_ia, {qs, ms, es});

            auto const q_av_2d   = q_av.reshape_view({nq, na * ne});
            auto const mu_ext_2d = mu_ext.reshape_view({nq, nk * ne});

            // J: (P|Q)^-1 (Q|i m) contracted with (Q | a v_ext).
            auto K_iovv = zeros("(i m|a v_ext)", {nk, na * ne});
            cg::gemm(1.0, r.q_io_inv, q_av_2d, 0.0, &K_iovv, Transpose::T, Transpose::N);
            // K: (Q | m u_ext) contracted with (P|Q)^-1 (Q|i a).
            auto K_oviv    = zeros("(m u_ext|i a)", {nk, ne * na});
            auto K_oviv_2d = K_oviv.reshape_view({nk * ne, na});
            cg::gemm(1.0, mu_ext_2d, r.q_iv_inv, 0.0, &K_oviv_2d, Transpose::T, Transpose::N);

            slice_neighbours(nb_ij[p], &out.J_ikac, base_ij[p], extended[p], K_iovv, na, ne, true, "(i k|a c)");
            slice_neighbours(nb_ij[p], &out.K_iakc, base_ij[p], extended[p], K_oviv, na, ne, false, "(i a|k c)");
            if (off_diag) {
                auto K_jovv = zeros("(j m|a v_ext)", {nk, na * ne});
                cg::gemm(1.0, r.q_jo_inv, q_av_2d, 0.0, &K_jovv, Transpose::T, Transpose::N);
                auto K_ovjv    = zeros("(m u_ext|j a)", {nk, ne * na});
                auto K_ovjv_2d = K_ovjv.reshape_view({nk * ne, na});
                cg::gemm(1.0, mu_ext_2d, r.q_jv_inv, 0.0, &K_ovjv_2d, Transpose::T, Transpose::N);
                slice_neighbours(nb_ji[p], &out.J_jkac, base_ji[p], extended[p], K_jovv, na, ne, true, "(j k|a c)");
                slice_neighbours(nb_ji[p], &out.K_jakc, base_ji[p], extended[p], K_ovjv, na, ne, false, "(j a|k c)");
            }
        }
    }

    // ── The density-fitted factors, strong pairs only ──────────────────────
    //
    // `Qma` and `Qab` were written in place; these four are the small ones, a
    // few hundred kilobytes a pair against the megabytes of `Qab`, so a copy is
    // cheaper than another layer of indirection.
    for (std::size_t p = 0; p < n_pairs; ++p) {
        auto const &r        = raw[p];
        bool const  off_diag = i_lmo[p] != j_lmo[p];
        bool const  keep     = strong[p];
        out.i_Qk.emplace_back(keep ? r.q_io : absent());
        out.i_Qa.emplace_back(keep ? r.q_iv : absent());
        out.j_Qk.emplace_back(keep && off_diag ? r.q_jo : absent());
        out.j_Qa.emplace_back(keep && off_diag ? r.q_jv : absent());
    }
    return out;
}

} // namespace dlpno
