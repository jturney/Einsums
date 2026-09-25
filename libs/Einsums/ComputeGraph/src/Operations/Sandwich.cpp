//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/Detail/GroupedMembers.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "Record.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

// ── grouped_sandwich: q-tiled dressed sandwich accumulations as ONE node ──────

/// One member of a grouped sandwich: ``C += sum_q B_q S B_q^T`` with the
/// dressed slice ``B_q = A[q] - P^T M[q]`` built in cache, never in memory.
///
/// This is psi4's own shape for the Eq. 76/93 residual term (dlpno/ccsd.cc,
/// "the T1-dressing ... on the fly, as this intermediate is only used once"),
/// adapted to a q-fastest layout: slices of ``A`` along the auxiliary axis are
/// strided here, so a BLOCK of them is transposed into slice-major scratch
/// sized to stay cache-resident, and the dress plus both sandwich GEMMs run
/// out of that scratch. ``A`` streams from memory exactly once; the dressed
/// factor and the half product exist only as one cache-resident slice.
///
/// Deterministic by construction: the q blocks and the slices inside them run
/// in ascending order on one thread, so the accumulation order into ``C`` is a
/// function of the extents alone. What this DOES change, relative to the pair
/// of whole-q contractions it replaces, is that ``C`` accumulates per q slice
/// rather than in one GEMM reduction - the same operand values sum in a
/// different order, so results agree to accumulation roundoff, not bitwise.
template <typename T>
void sandwich_member(T const *A, std::size_t saq, std::size_t saa, std::size_t sab, T const *M, std::size_t smq, std::size_t smk,
                     std::size_t smb, T const *P, std::size_t ldp, T const *S, std::size_t lds, T *C, std::size_t ldc, std::size_t nq,
                     std::size_t nk, std::size_t na) {
    if (nq == 0 || na == 0) {
        return; // C += nothing: the accumulate form of the zero-extent contract
    }
    std::size_t const slice   = na * na;
    std::size_t const m_slice = nk * na;
    // Slices per block: enough to amortize the strided gather, small enough
    // that the staged block plus its M companion stay comfortably inside a
    // per-core L2 share.
    std::size_t tq = std::max<std::size_t>(4, (512UL * 1024) / (sizeof(T) * (slice + std::max<std::size_t>(1, m_slice))));
    tq             = std::min(tq, nq);

    // Plain heap scratch, deliberately not BufferVector: the metered buffer
    // pool is sized for contraction workspace (4 MiB by default), and a team
    // of these members would exhaust it. This scratch is bounded at a few
    // hundred kilobytes per running member by the block sizing above.
    std::vector<T> bblk(tq * slice);
    std::vector<T> mblk(std::max<std::size_t>(1, tq * m_slice));
    std::vector<T> w(slice);

    for (std::size_t q0 = 0; q0 < nq; q0 += tq) {
        std::size_t const tt = std::min(tq, nq - q0);

        // Stage the A block slice-major. The source walks are contiguous in q
        // (the fastest axis), so memory is read in order; the scattered writes
        // land in the cache-resident block.
        for (std::size_t b = 0; b < na; b++) {
            for (std::size_t a = 0; a < na; a++) {
                T const *from = A + q0 * saq + a * saa + b * sab;
                T       *to   = bblk.data() + a + na * b;
                for (std::size_t t = 0; t < tt; t++) {
                    to[t * slice] = from[t * saq];
                }
            }
        }
        if (nk != 0) {
            for (std::size_t b = 0; b < na; b++) {
                for (std::size_t k = 0; k < nk; k++) {
                    T const *from = M + q0 * smq + k * smk + b * smb;
                    T       *to   = mblk.data() + k + nk * b;
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * m_slice] = from[t * smq];
                    }
                }
            }
        }

        for (std::size_t t = 0; t < tt; t++) {
            T *Bs = bblk.data() + t * slice;
            // Dress: B_q -= P^T M_q. Skipped when there is nothing to dress
            // with, which is also what keeps a (1, na) placeholder P legal for
            // an nk == 0 member.
            if (nk != 0) {
                blas::gemm('T', 'N', static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), static_cast<blas::int_t>(nk), T{-1}, P,
                           static_cast<blas::int_t>(ldp), mblk.data() + t * m_slice, static_cast<blas::int_t>(nk), T{1}, Bs,
                           static_cast<blas::int_t>(na));
            }
            // W = B_q S, then C += W B_q^T.
            blas::gemm('N', 'N', static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), T{1}, Bs,
                       static_cast<blas::int_t>(na), S, static_cast<blas::int_t>(lds), T{0}, w.data(), static_cast<blas::int_t>(na));
            blas::gemm('N', 'T', static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), T{1}, w.data(),
                       static_cast<blas::int_t>(na), Bs, static_cast<blas::int_t>(na), T{1}, C, static_cast<blas::int_t>(ldc));
        }
    }
}

// ── grouped_gather_rotate: q-tiled gather plus two-sided rotation as ONE node ─

/// One member of a grouped gather-rotate:
/// ``C[q, a, b] = sum_uv src[Q[q], U[u], U[v]] X[u, a] X[v, b]``, with the
/// gathered ``(q, u, v)`` block built one cache-resident q tile at a time and
/// never materialized whole.
///
/// The emission this replaces is a gather of the whole ``(Q|u v)`` domain block
/// followed by two contractions through a half-transformed block of the same
/// leading extent, which is four full streams of the largest thing the phase
/// holds where one suffices. Here the source block streams once, and the half
/// product exists only for the tile in flight.
///
/// Constraints, all checked by the caller and relied on here:
/// - ``src`` is rank 3 and shared by every member; ``Q`` selects its axis 0,
///   ``U`` selects axes 1 and 2 SYMMETRICALLY (one list, because one ``X``
///   rotates both).
/// - ``X`` is ``(nu, nt)`` and column-contiguous, so it is a legal GEMM operand
///   at ``lda = ldx``.
/// - ``C`` is ``(nq, nt, nt)`` and ASSIGNED, not accumulated: every element is
///   written, so the destination may be uninitialized on entry.
/// - Offsets arrive premultiplied by the source strides, so the staging loop
///   adds rather than multiplies.
///
/// Bitwise reproducible, and independent of the tiling: q indexes no sum, so
/// each output element is one GEMM's reduction over the FULL ``u`` (then ``v``)
/// range whatever the tile size. That is a stronger contract than the grouped
/// sandwich's - there is no accumulation to reassociate - and it is why the
/// node needs no fixed-order argument. It is not bitwise agreement with the
/// gather-plus-two-einsums form it replaces, which blocks its reductions
/// differently; that agreement is to roundoff.
template <typename T>
void gather_rotate_member(T const *src, std::size_t sq, std::size_t const *qoff, std::size_t nq, std::size_t const *uoff,
                          std::size_t const *voff, std::size_t nu, T const *X, std::size_t ldx, std::size_t nt, T *C, std::size_t scq,
                          std::size_t sca, std::size_t scb) {
    if (nq == 0 || nt == 0) {
        return; // nothing to write
    }
    if (nu == 0) {
        // An empty sum, and the operation ASSIGNS, so the zeros are the answer
        // and have to be written rather than left as whatever was allocated.
        for (std::size_t b = 0; b < nt; b++) {
            for (std::size_t a = 0; a < nt; a++) {
                T *to = C + a * sca + b * scb;
                for (std::size_t q = 0; q < nq; q++) {
                    to[q * scq] = T{0};
                }
            }
        }
        return;
    }

    std::size_t const slice  = nu * nu;
    std::size_t const hslice = nt * nu;
    std::size_t const oslice = nt * nt;
    // Slices per tile: enough to amortize the gather and to give the first GEMM
    // a wide right-hand side, small enough that the staged block, the half
    // product and the result stay inside a per-core L2 share.
    std::size_t tq = std::max<std::size_t>(4, (512UL * 1024) / (sizeof(T) * (slice + hslice + oslice)));
    tq             = std::min(tq, nq);

    // Plain heap scratch, deliberately not BufferVector: the metered buffer pool
    // is sized for contraction workspace, and a team of these members running at
    // once would exhaust it - and a throw out of the pool inside this node's
    // OpenMP region is not recoverable.
    std::vector<T> blk(tq * slice);
    std::vector<T> half(tq * hslice);
    std::vector<T> out(tq * oslice);

    for (std::size_t q0 = 0; q0 < nq; q0 += tq) {
        std::size_t const tt = std::min(tq, nq - q0);

        // Whether this tile's auxiliary selection is one ascending run, which is
        // the common case: a domain is a sorted list of functions and the shells
        // behind it are contiguous. A run turns the gather into a strided walk
        // of the source, which is what the fastest axis wants.
        bool run = true;
        for (std::size_t t = 1; run && t < tt; t++) {
            run = qoff[q0 + t] == qoff[q0] + t * sq;
        }

        // Stage the tile slice-major, u fastest. The source walks are along the
        // FASTEST source axis, so memory is read in order; the strided writes
        // land inside the cache-resident tile. The layout is exactly the
        // ``(nu) x (nu * tt)`` matrix the first GEMM wants, which is why the
        // rotation below is one call and not one per slice.
        for (std::size_t b = 0; b < nu; b++) {
            for (std::size_t a = 0; a < nu; a++) {
                T const *from = src + uoff[a] + voff[b];
                T       *to   = blk.data() + a + nu * b;
                if (run) {
                    T const *base = from + qoff[q0];
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * slice] = base[t * sq];
                    }
                } else {
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * slice] = from[qoff[q0 + t]];
                    }
                }
            }
        }

        // half[a, v, t] = sum_u X[u, a] blk[u, v, t]: one GEMM for the whole
        // tile, because (v, t) is a single merged axis in this layout.
        blas::gemm('T', 'N', static_cast<blas::int_t>(nt), static_cast<blas::int_t>(nu * tt), static_cast<blas::int_t>(nu), T{1}, X,
                   static_cast<blas::int_t>(ldx), blk.data(), static_cast<blas::int_t>(nu), T{0}, half.data(),
                   static_cast<blas::int_t>(nt));
        // out[a, b, t] = sum_v half[a, v, t] X[v, b]: per slice, because the
        // contracted index is interior once the tile is laid out this way.
        for (std::size_t t = 0; t < tt; t++) {
            blas::gemm('N', 'N', static_cast<blas::int_t>(nt), static_cast<blas::int_t>(nt), static_cast<blas::int_t>(nu), T{1},
                       half.data() + t * hslice, static_cast<blas::int_t>(nt), X, static_cast<blas::int_t>(ldx), T{0},
                       out.data() + t * oslice, static_cast<blas::int_t>(nt));
        }

        // Scatter the tile out. The destination's auxiliary axis is the one
        // being walked, so a unit q stride - what an owning (nq, nt, nt) store
        // has - makes every one of these a contiguous run.
        for (std::size_t b = 0; b < nt; b++) {
            for (std::size_t a = 0; a < nt; a++) {
                T const *from = out.data() + a + nt * b;
                T       *to   = C + q0 * scq + a * sca + b * scb;
                if (scq == 1) {
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t] = from[t * oslice];
                    }
                } else {
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * scq] = from[t * oslice];
                    }
                }
            }
        }
    }
}

namespace {

template <typename T>
void run_sandwich_member(Impl<T> &ci, Impl<T> const &ai, Impl<T> const &mi, Impl<T> const &pi, Impl<T> const &si) {
    size_t const nk = mi.dim(1);
    sandwich_member(ai.data(), ai.stride(0), ai.stride(1), ai.stride(2), mi.data(), mi.stride(0), mi.stride(1), mi.stride(2), pi.data(),
                    nk != 0 ? pi.stride(1) : 1, si.data(), si.stride(1), ci.data(), ci.stride(1), ai.dim(0), nk, ai.dim(1));
}

template <typename T>
void run_gather_rotate_member(Impl<T> &ci, Impl<T> const &si, Impl<T> const &xi, std::vector<size_t> const &qs,
                              std::vector<size_t> const &us) {
    size_t const nq = qs.size(), nu = us.size(), nt = xi.dim(1);

    // Strides are read here rather than baked at capture: a slot may point
    // at a different tensor object on a later replay.
    size_t const        sq = si.stride(0);
    std::vector<size_t> qoff(nq), uoff(nu), voff(nu);
    for (size_t t = 0; t < nq; t++) {
        qoff[t] = qs[t] * sq;
    }
    for (size_t a = 0; a < nu; a++) {
        uoff[a] = us[a] * si.stride(1);
        voff[a] = us[a] * si.stride(2);
    }
    gather_rotate_member(si.data(), sq, qoff.data(), nq, uoff.data(), voff.data(), nu, xi.data(), nu != 0 ? xi.stride(1) : 1, nt, ci.data(),
                         ci.stride(0), ci.stride(1), ci.stride(2));
}
} // namespace

template <typename T>
void eager_grouped_sandwich(std::vector<Impl<T> *> const &c, std::vector<Impl<T> const *> const &a, std::vector<Impl<T> const *> const &m,
                            std::vector<Impl<T> const *> const &p, std::vector<Impl<T> const *> const &s) {
    LabeledSection("grouped_sandwich eager");
    // Members are independent (distinct destinations, each accumulated
    // serially by one thread), and the whole run is one parallel region -
    // an OpenMP team, never a caller-created thread pool (trap 7). An
    // exception may not cross the region boundary (that terminates), so
    // the first one is carried out by hand.
    run_grouped_members(c.size(), [&](size_t i) { run_sandwich_member<T>(*c[i], *a[i], *m[i], *p[i], *s[i]); });
}

template <typename T>
void capture_grouped_sandwich(CaptureContext &ctx, std::vector<SlotRef> const &c, std::vector<SlotRef> const &a,
                              std::vector<SlotRef> const &m, std::vector<SlotRef> const &p, std::vector<SlotRef> const &s) {
    LabeledSection("grouped_sandwich capture");
    size_t const          count = c.size();
    std::vector<TensorId> inputs, outputs;
    inputs.reserve(5 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        inputs.push_back(a[i].first);
        inputs.push_back(m[i].first);
        inputs.push_back(p[i].first);
        inputs.push_back(s[i].first);
        // Accumulation reads the destination, so the RAW edge from whoever
        // produced C must survive - the same rule grouped_axpby records for a
        // non-zero beta.
        inputs.push_back(c[i].first);
        outputs.push_back(c[i].first);
    }

    auto c_access = accessors<T>(c), a_access = accessors<T>(a), m_access = accessors<T>(m), p_access = accessors<T>(p),
         s_access = accessors<T>(s);

    // The extents as the operands have them now, at capture.
    GroupedSandwichDescriptor d;
    d.total = static_cast<int>(count);
    d.nq.reserve(count);
    d.nk.reserve(count);
    d.na.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.nq.push_back(static_cast<std::int64_t>(a_access[i].template impl<T>()->dim(0)));
        d.nk.push_back(static_cast<std::int64_t>(m_access[i].template impl<T>()->dim(1)));
        d.na.push_back(static_cast<std::int64_t>(a_access[i].template impl<T>()->dim(1)));
    }

    auto executor = [c_access = std::move(c_access), a_access = std::move(a_access), m_access = std::move(m_access),
                     p_access = std::move(p_access), s_access = std::move(s_access)]() {
        LabeledSection("grouped_sandwich execute");
        run_grouped_members(c_access.size(), [&](size_t i) {
            run_sandwich_member<T>(*c_access[i].template impl<T>(), *a_access[i].template impl<T>(), *m_access[i].template impl<T>(),
                                   *p_access[i].template impl<T>(), *s_access[i].template impl<T>());
        });
    };
    ctx.record(OpKind::GroupedSandwich, fmt::format("sandwich x{}", count), std::move(inputs), std::move(outputs), std::move(executor),
               std::move(d));
}

template <typename T>
void eager_grouped_gather_rotate(std::vector<Impl<T> *> const &c, Impl<T> const &src, std::vector<Impl<T> const *> const &x,
                                 std::vector<std::vector<size_t>> const &q_list, std::vector<std::vector<size_t>> const &u_list) {
    LabeledSection("grouped_gather_rotate eager");
    // Members are independent (distinct destinations, each assigned by one
    // thread), and the whole run is one parallel region - an OpenMP team,
    // never a caller-created thread pool. An exception may not cross the
    // region boundary (that terminates, and takes a libomp worker with it),
    // so the first one is carried out by hand.
    run_grouped_members(c.size(), [&](size_t i) { run_gather_rotate_member<T>(*c[i], src, *x[i], q_list[i], u_list[i]); });
}

template <typename T>
void capture_grouped_gather_rotate(CaptureContext &ctx, std::vector<SlotRef> const &c, SlotRef src, std::vector<SlotRef> const &x,
                                   std::vector<std::vector<size_t>> q_list, std::vector<std::vector<size_t>> u_list) {
    LabeledSection("grouped_gather_rotate capture");
    size_t const          count = c.size();
    std::vector<TensorId> inputs, outputs;
    inputs.reserve(count + 1);
    outputs.reserve(count);
    inputs.push_back(src.first);
    for (size_t i = 0; i < count; i++) {
        inputs.push_back(x[i].first);
        // The destination is ASSIGNED, not accumulated, so it is an output only:
        // unlike the grouped sandwich there is no read of C to keep an edge for.
        outputs.push_back(c[i].first);
    }

    auto                  c_access = accessors<T>(c), x_access = accessors<T>(x);
    OperandAccessor const s_access(src.second, packed_gemm::get_scalar_type<T>());

    GroupedGatherRotateDescriptor d;
    d.total      = static_cast<int>(count);
    d.elem_bytes = static_cast<std::int64_t>(sizeof(T));
    d.nq.reserve(count);
    d.nu.reserve(count);
    d.nt.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.nq.push_back(static_cast<std::int64_t>(q_list[i].size()));
        d.nu.push_back(static_cast<std::int64_t>(u_list[i].size()));
        d.nt.push_back(static_cast<std::int64_t>(x_access[i].template impl<T>()->dim(1)));
    }

    // The index lists are copied into the executor rather than referenced: a
    // captured node outlives the call, and every replay reads them again.
    auto executor = [s_access, c_access = std::move(c_access), x_access = std::move(x_access), q_list = std::move(q_list),
                     u_list = std::move(u_list)]() {
        LabeledSection("grouped_gather_rotate execute");
        run_grouped_members(c_access.size(), [&](size_t i) {
            run_gather_rotate_member<T>(*c_access[i].template impl<T>(), *s_access.template impl<T>(), *x_access[i].template impl<T>(),
                                        q_list[i], u_list[i]);
        });
    };
    ctx.record(OpKind::GroupedGatherRotate, fmt::format("gather_rotate x{}", count), std::move(inputs), std::move(outputs),
               std::move(executor), std::move(d));
}

#define EINSUMS_SANDWICH_OPERATIONS(T)                                                                                                     \
    template EINSUMS_EXPORT void eager_grouped_sandwich<T>(std::vector<Impl<T> *> const &, std::vector<Impl<T> const *> const &,           \
                                                           std::vector<Impl<T> const *> const &, std::vector<Impl<T> const *> const &,     \
                                                           std::vector<Impl<T> const *> const &);                                          \
    template EINSUMS_EXPORT void capture_grouped_sandwich<T>(CaptureContext &, std::vector<SlotRef> const &, std::vector<SlotRef> const &, \
                                                             std::vector<SlotRef> const &, std::vector<SlotRef> const &,                   \
                                                             std::vector<SlotRef> const &);                                                \
    template EINSUMS_EXPORT void eager_grouped_gather_rotate<T>(                                                                           \
        std::vector<Impl<T> *> const &, Impl<T> const &, std::vector<Impl<T> const *> const &, std::vector<std::vector<size_t>> const &,   \
        std::vector<std::vector<size_t>> const &);                                                                                         \
    template EINSUMS_EXPORT void capture_grouped_gather_rotate<T>(CaptureContext &, std::vector<SlotRef> const &, SlotRef,                 \
                                                                  std::vector<SlotRef> const &, std::vector<std::vector<size_t>>,          \
                                                                  std::vector<std::vector<size_t>>);

// The sandwich and the gather-rotate are real-only, as their wrappers require.
EINSUMS_CG_REAL_ELEMENT_TYPES(EINSUMS_SANDWICH_OPERATIONS)
#undef EINSUMS_SANDWICH_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
