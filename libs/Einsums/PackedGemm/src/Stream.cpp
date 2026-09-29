//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/Stream.hpp>
#include <Einsums/PackedGemm/StreamKernel.hpp>

#include <algorithm>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

namespace {

int64_t elems_of(StreamLayout const &x) {
    int64_t n = 1;
    for (int64_t const d : x.dims) {
        n *= d;
    }
    return n;
}

// Column-major packing of an output's own axis order: stride 1 on axis 0, each
// later axis the product of the extents before it. Private accumulators are
// indexed this way, so every term writing one output agrees on the mapping
// whatever order its axes are named in, and a strided view output costs its
// element count rather than its offset span.
std::vector<int64_t> dense_strides(StreamLayout const &x) {
    std::vector<int64_t> ds(x.dims.size(), 1);
    for (size_t d = 1; d < x.dims.size(); d++) {
        ds[d] = ds[d - 1] * x.dims[d - 1];
    }
    return ds;
}

// Visit an output's real element offsets in the order its dense packing uses
// (axis 0 fastest), so the n-th visit pairs with dense index n.
template <typename F>
void for_each_element(StreamLayout const &x, F &&body) {
    size_t const         rank = x.dims.size();
    int64_t const        n    = elems_of(x);
    std::vector<int64_t> cc(rank, 0);
    int64_t              off = 0;
    for (int64_t e = 0; e < n; e++) {
        body(off);
        for (size_t d = 0; d < rank; d++) {
            cc[d]++;
            off += x.strides[d];
            if (cc[d] < x.dims[d]) {
                break;
            }
            off -= cc[d] * x.strides[d];
            cc[d] = 0;
        }
    }
}

// Prefix blocks each thread must have for the prefix split to balance.
constexpr int64_t kBlocksPerThread = 4;

// Elements of S each thread must have before another thread is worth forking.
// Measured on the M4 over the exchange shape and a large-output shape: 16384
// forked too early and was slower at every mid size (K at n = 20: 66 us
// against 35 us at 65536).
constexpr int64_t kElemsPerThread = 65536;

} // namespace

namespace {

/// Merge pairs of axes of S that are contiguous in S and in every term's output
/// and weight: axis a merges into axis b when a steps exactly one run of b in S,
/// and, in each of C and W, both are summed or a is the axis right after b and
/// steps one run of it. The merged axis keeps b's position; its extent is the
/// product. A term's operand axes merge the same way, so an element's offset and
/// its place in the operand's dense order are unchanged. Partition axes follow
/// their merged axis.
template <typename T>
void merge_contiguous_axes(StreamLayout &sl, std::vector<StreamTerm<T>> &tv, std::vector<int> &pax) {
    auto const erase_axis = [](StreamLayout &l, int axis) {
        l.dims.erase(l.dims.begin() + axis);
        l.strides.erase(l.strides.begin() + axis);
    };
    auto const fits = [](std::vector<int> const &map, StreamLayout const &l, int a, int b) {
        int const oa = map[static_cast<size_t>(a)], ob = map[static_cast<size_t>(b)];
        if ((oa < 0) != (ob < 0)) {
            return false;
        }
        return oa < 0 ||
               (oa == ob + 1 && l.strides[static_cast<size_t>(oa)] == l.dims[static_cast<size_t>(ob)] * l.strides[static_cast<size_t>(ob)]);
    };
    auto const merge_operand = [&](std::vector<int> &map, StreamLayout &l, int a, int b) {
        int const oa = map[static_cast<size_t>(a)], ob = map[static_cast<size_t>(b)];
        if (oa >= 0) {
            l.dims[static_cast<size_t>(ob)] *= l.dims[static_cast<size_t>(oa)];
            erase_axis(l, oa);
            for (auto &x : map) {
                if (x > oa) {
                    --x;
                }
            }
        }
        map.erase(map.begin() + a);
    };

    for (bool merged = true; merged;) {
        merged      = false;
        int const r = static_cast<int>(sl.dims.size());
        for (int a = 0; a < r && !merged; ++a) {
            for (int b = 0; b < r && !merged; ++b) {
                if (a == b || sl.strides[static_cast<size_t>(a)] != sl.dims[static_cast<size_t>(b)] * sl.strides[static_cast<size_t>(b)]) {
                    continue;
                }
                bool const ok = std::ranges::all_of(
                    tv, [&](StreamTerm<T> const &t) { return fits(t.c_axis, t.c_layout, a, b) && fits(t.w_axis, t.w_layout, a, b); });
                if (!ok) {
                    continue;
                }
                sl.dims[static_cast<size_t>(b)] *= sl.dims[static_cast<size_t>(a)];
                erase_axis(sl, a);
                for (auto &t : tv) {
                    merge_operand(t.c_axis, t.c_layout, a, b);
                    merge_operand(t.w_axis, t.w_layout, a, b);
                }
                int const        b_new = b > a ? b - 1 : b;
                std::vector<int> moved;
                for (int const x : pax) {
                    int const y = x == a ? b_new : (x > a ? x - 1 : x);
                    if (std::ranges::find(moved, y) == moved.end()) {
                        moved.push_back(y);
                    }
                }
                pax    = std::move(moved);
                merged = true;
            }
        }
    }
}

/// The storage-order walk of stream_contract, after its checks, the output
/// prefactors and the axis merge. The reduction reads the outputs through
/// @p out_layouts, their layouts before the merge, which index the same elements
/// in the same dense order.
template <typename T>
void stream_walk(T const *s, StreamLayout const &s_layout, std::vector<StreamTerm<T>> const &terms, std::vector<int> const &partition_axes,
                 std::vector<size_t> const &slot_of, std::vector<T *> const &outs, std::vector<StreamLayout const *> const &out_layouts) {
    int const rank = static_cast<int>(s_layout.dims.size());
    // Storage-order axes: descending stride, so the last walks stride 1.
    std::vector<int> axes(static_cast<size_t>(rank));
    std::iota(axes.begin(), axes.end(), 0);
    std::ranges::stable_sort(axes, [&](int x, int y) { return s_layout.strides[x] > s_layout.strides[y]; });

    std::vector<int64_t> dims(static_cast<size_t>(rank)), s_delta(static_cast<size_t>(rank));
    for (int d = 0; d < rank; d++) {
        dims[d]    = s_layout.dims[axes[d]];
        s_delta[d] = s_layout.strides[axes[d]];
    }

    // Team size: enough elements per thread to pay for the region. Below that
    // the region's fork and the per-thread buffers cost more than the stream
    // (about 30 us for ten threads on the M4, against 1-2 us for the serial
    // walk of a few thousand elements), so a small call runs on this thread.
    int64_t const s_elems = elems_of(s_layout);
#ifdef _OPENMP
    int const requested = static_cast<int>(std::clamp<int64_t>(s_elems / kElemsPerThread, 1, omp_get_max_threads()));
#else
    int const requested = 1;
#endif

    // Partition axis: prefer one with at least a block per thread, then the
    // largest stride. A low-stride partition turns each thread's read into a
    // strided comb through S (measured about 5x slower); a high-stride one keeps
    // contiguous slabs. A serial walk needs no partition: it writes every output
    // directly because nothing can race.
    // Outer prefix: the outermost storage axes that every output carries. Split
    // across threads, it hands each thread a contiguous slab of S and a disjoint
    // slice of every output, so all of them are written in place with no
    // private copy and no reduction - and the blocks are the product of the
    // prefix extents, fine enough to balance where one axis is not (an extent
    // of 32 over ten threads leaves four threads a third more work).
    int     prefix       = 0;
    int64_t prefix_total = 1;
    while (prefix < rank - 1 &&
           std::ranges::all_of(terms, [&](StreamTerm<T> const &t) { return t.c_axis[static_cast<size_t>(axes[prefix])] >= 0; })) {
        prefix_total *= dims[prefix];
        prefix++;
    }
    bool const prefix_split = requested > 1 && prefix_total >= kBlocksPerThread * requested;

    int part_axis = -1;
    if (requested > 1 && !prefix_split) {
        for (int const axis : partition_axes) {
            if (axis < 0 || axis >= rank) {
                continue;
            }
            if (part_axis < 0) {
                part_axis = axis;
                continue;
            }
            bool const fills      = s_layout.dims[axis] >= requested;
            bool const best_fills = s_layout.dims[part_axis] >= requested;
            if (fills != best_fills
                    ? fills
                    : (fills ? s_layout.strides[axis] > s_layout.strides[part_axis] : s_layout.dims[axis] > s_layout.dims[part_axis])) {
                part_axis = axis;
            }
        }
    }

    // Per-term affine deltas along each storage-order axis of S.
    struct Plan {
        T                    alpha;
        T const             *w;
        T                   *out;
        bool                 direct;
        std::vector<int64_t> cdelta, wdelta;
        size_t               slot;
        int64_t              out_elems;
    };
    std::vector<Plan> plans;
    plans.reserve(terms.size());
    for (size_t k = 0; k < terms.size(); k++) {
        auto const &t = terms[k];
        Plan        p;
        p.alpha  = t.alpha;
        p.w      = t.w;
        p.out    = t.c;
        p.direct = requested == 1 || prefix_split || (part_axis >= 0 && t.c_axis[static_cast<size_t>(part_axis)] >= 0);
        p.cdelta.assign(static_cast<size_t>(rank), 0);
        p.wdelta.assign(static_cast<size_t>(rank), 0);
        // A direct writer addresses C itself; a privatized term addresses its
        // dense accumulator.
        std::vector<int64_t> const c_strides = p.direct ? t.c_layout.strides : dense_strides(t.c_layout);
        for (int d = 0; d < rank; d++) {
            if (int const ca = t.c_axis[static_cast<size_t>(axes[d])]; ca >= 0) {
                p.cdelta[d] = c_strides[static_cast<size_t>(ca)];
            }
            if (int const wa = t.w_axis[static_cast<size_t>(axes[d])]; wa >= 0) {
                p.wdelta[d] = t.w_layout.strides[static_cast<size_t>(wa)];
            }
        }
        p.slot      = slot_of[k];
        p.out_elems = elems_of(t.c_layout);
        plans.push_back(std::move(p));
    }

    int pd = -1;
    for (int d = 0; d < rank && part_axis >= 0; d++) {
        if (axes[d] == part_axis) {
            pd = d;
        }
    }

    StreamTileFn<T> const stream_tile = stream_tile_entry<T>();

    // Work units, handed out dynamically: the cores of one machine need not run
    // at one speed (the M4's efficiency cores take about twice as long per
    // unit), and a static equal split leaves the fast ones waiting. A unit is
    // one prefix coordinate, one index of the partition axis, or one chunk of
    // the flat outer range.
    int64_t outer_total = 1;
    for (int d = 0; d < rank - 1; d++) {
        outer_total *= dims[d];
    }
    int64_t prefix_block = 1; // outer iterations per prefix coordinate
    for (int d = prefix; d < rank - 1; d++) {
        prefix_block *= dims[d];
    }
    int64_t const units = prefix_split ? prefix_total : (pd >= 0 ? dims[pd] : outer_total);
    // Consecutive units are handed out together, about eight chunks per thread:
    // enough to rebalance, few enough that the hand-out and each walk's setup
    // stay off the profile. Consecutive units of any kind form one box.
    int64_t const chunk  = std::max<int64_t>(1, units / (int64_t{8} * requested));
    int64_t const chunks = (units + chunk - 1) / chunk;

    // Walk outer iterations [q0, q1) of the box whose extents are ldims and whose
    // origin sits at the given offsets.
    auto const walk = [&](std::vector<int64_t> const &ldims, int64_t s_base, std::vector<int64_t> const &c_base,
                          std::vector<int64_t> const &w_base, int64_t q0, int64_t q1, std::vector<std::vector<T>> &priv) {
        int64_t const        inner_n = ldims[rank - 1];
        std::vector<int64_t> coord(static_cast<size_t>(rank), 0);
        {
            int64_t rem = q0;
            for (int d = rank - 2; d >= 0; d--) {
                coord[static_cast<size_t>(d)] = rem % ldims[d];
                rem /= ldims[d];
            }
        }
        int64_t              s_off = s_base;
        std::vector<int64_t> c_off = c_base, w_off = w_base;
        for (int d = 0; d < rank - 1; d++) {
            s_off += coord[static_cast<size_t>(d)] * s_delta[d];
            for (size_t k = 0; k < plans.size(); k++) {
                c_off[k] += coord[static_cast<size_t>(d)] * plans[k].cdelta[d];
                w_off[k] += coord[static_cast<size_t>(d)] * plans[k].wdelta[d];
            }
        }

        // Rows along the second-innermost axis go to the tile kernel together,
        // as many as remain on that axis and in this range; the odometer then
        // steps the outer axes once per tile rather than once per row.
        int64_t const ds  = s_delta[rank - 1];
        int const     row = rank - 2; // the tile's row axis, or -1 for a rank-1 stream
        int64_t const ds2 = row >= 0 ? s_delta[row] : 0;
        for (int64_t q = q0; q < q1;) {
            int64_t const m = row >= 0 ? std::min(ldims[row] - coord[static_cast<size_t>(row)], q1 - q) : 1;
            for (size_t k = 0; k < plans.size(); k++) {
                auto const   &p   = plans[k];
                T            *cb  = p.direct ? p.out : priv[p.slot].data();
                int64_t const dc2 = row >= 0 ? p.cdelta[row] : 0;
                int64_t const dw2 = row >= 0 ? p.wdelta[row] : 0;
                stream_tile(cb, s, p.w, p.alpha, m, inner_n, c_off[k], s_off, w_off[k], ds, p.cdelta[rank - 1], p.wdelta[rank - 1], ds2,
                            dc2, dw2);
            }
            q += m;
            if (row < 0 || q >= q1) {
                break;
            }

            // Advance the row axis past the tile, then carry outward like an
            // odometer; the origin stays put because a wrap subtracts only the
            // box-local contribution.
            for (int d = row; d >= 0; d--) {
                int64_t const step = d == row ? m : 1;
                coord[static_cast<size_t>(d)] += step;
                s_off += step * s_delta[d];
                for (size_t k = 0; k < plans.size(); k++) {
                    c_off[k] += step * plans[k].cdelta[d];
                    w_off[k] += step * plans[k].wdelta[d];
                }
                if (coord[static_cast<size_t>(d)] < ldims[d]) {
                    break;
                }
                s_off -= coord[static_cast<size_t>(d)] * s_delta[d];
                for (size_t k = 0; k < plans.size(); k++) {
                    c_off[k] -= coord[static_cast<size_t>(d)] * plans[k].cdelta[d];
                    w_off[k] -= coord[static_cast<size_t>(d)] * plans[k].wdelta[d];
                }
                coord[static_cast<size_t>(d)] = 0;
            }
        }
    };

    // One slot per thread the region could get; a thread that takes no unit
    // leaves its slot empty and the reduction skips it. The slot is indexed by
    // the thread number of the region actually entered, which inside another
    // parallel region is a team of one.
    std::vector<std::vector<std::vector<T>>> thread_priv(static_cast<size_t>(requested));
    std::vector<int64_t> const               zero_bases(plans.size(), 0);

#ifdef _OPENMP
#    pragma omp parallel num_threads(requested) if (requested > 1)
#endif
    {
#ifdef _OPENMP
        int const tid = omp_get_thread_num();
#else
        int const tid = 0;
#endif
        auto &priv = thread_priv[static_cast<size_t>(tid)];

#ifdef _OPENMP
#    pragma omp for schedule(dynamic, 1)
#endif
        for (int64_t c = 0; c < chunks; c++) {
            int64_t const u0 = c * chunk;
            int64_t const u1 = std::min(units, u0 + chunk);
            if (priv.empty()) {
                priv.resize(outs.size());
                for (size_t o = 0; o < outs.size(); o++) {
                    for (auto const &p : plans) {
                        if (p.slot == o && !p.direct) {
                            priv[o].assign(static_cast<size_t>(p.out_elems), T{0});
                            break;
                        }
                    }
                }
            }
            if (prefix_split) {
                walk(dims, 0, zero_bases, zero_bases, u0 * prefix_block, u1 * prefix_block, priv);
            } else if (pd >= 0) {
                // A block of the partition axis: a box of that extent on it.
                std::vector<int64_t> ldims = dims;
                ldims[pd]                  = u1 - u0;
                std::vector<int64_t> c_base(plans.size()), w_base(plans.size());
                for (size_t k = 0; k < plans.size(); k++) {
                    c_base[k] = u0 * plans[k].cdelta[pd];
                    w_base[k] = u0 * plans[k].wdelta[pd];
                }
                int64_t box = 1;
                for (int d = 0; d < rank - 1; d++) {
                    box *= ldims[d];
                }
                walk(ldims, u0 * s_delta[pd], c_base, w_base, 0, box, priv);
            } else {
                walk(dims, 0, zero_bases, zero_bases, u0, u1, priv);
            }
        }
    }

    // Reduce the private accumulators. Dense index e pairs with the e-th element
    // the output's own odometer visits, which keeps a strided output's gaps
    // untouched: a view output shares them with the rest of its parent.
    for (size_t u = 0; u < outs.size(); u++) {
        T *const cd = outs[u];
        for (auto const &tp : thread_priv) {
            if (u >= tp.size() || tp[u].empty()) {
                continue;
            }
            auto const &buf = tp[u];
            size_t      e   = 0;
            for_each_element(*out_layouts[u], [&](int64_t off) { cd[off] += buf[e++]; });
        }
    }
}

} // namespace

template <typename T>
void stream_contract(T const *s, StreamLayout const &s_layout, std::vector<StreamTerm<T>> const &terms,
                     std::vector<int> const &partition_axes) {
    int const rank = static_cast<int>(s_layout.dims.size());
    for (auto const &t : terms) {
        if (t.c_axis.size() != static_cast<size_t>(rank) || t.w_axis.size() != static_cast<size_t>(rank)) {
            throw std::invalid_argument("stream_contract: a term's axis maps must have one entry per axis of the streamed tensor");
        }
    }

    // Distinct outputs, in first-appearance order. Terms name an output by its
    // data pointer.
    std::vector<T *>                  outs;
    std::vector<StreamLayout const *> out_layouts;
    std::vector<size_t>               slot_of(terms.size());
    for (size_t k = 0; k < terms.size(); k++) {
        auto it = std::ranges::find(outs, terms[k].c);
        if (it == outs.end()) {
            outs.push_back(terms[k].c);
            out_layouts.push_back(&terms[k].c_layout);
            slot_of[k] = outs.size() - 1;
        } else {
            slot_of[k] = static_cast<size_t>(it - outs.begin());
            if (terms[k].c_pf != T{1}) {
                throw std::invalid_argument(
                    "stream_contract: only the first term on an output may scale it; a later term's c_pf must be 1");
            }
        }
    }

    // Apply each output's prefactor once, before anything accumulates, and
    // before the empty-stream return: an empty contraction still scales C.
    // Zero overwrites rather than multiplies, so a NaN already in C does not
    // survive a c_pf of 0.
    for (size_t k = 0; k < terms.size(); k++) {
        bool const first = std::ranges::find(slot_of.begin(), slot_of.begin() + static_cast<std::ptrdiff_t>(k), slot_of[k]) ==
                           slot_of.begin() + static_cast<std::ptrdiff_t>(k);
        if (!first || terms[k].c_pf == T{1}) {
            continue;
        }
        T *const cd = terms[k].c;
        T const  f  = terms[k].c_pf;
        if (f == T{0}) {
            for_each_element(terms[k].c_layout, [cd](int64_t off) { cd[off] = T{0}; });
        } else {
            for_each_element(terms[k].c_layout, [cd, f](int64_t off) { cd[off] *= f; });
        }
    }

    if (terms.empty() || elems_of(s_layout) == 0 || rank == 0) {
        return;
    }

    // Work on copies with contiguous axes merged: a longer innermost run and a
    // tile the kernels can block. A shared output is left alone, since two terms
    // could map the same axes of S onto different axes of it.
    StreamLayout               sl  = s_layout;
    std::vector<StreamTerm<T>> tv  = terms;
    std::vector<int>           pax = partition_axes;
    if (outs.size() == terms.size()) {
        merge_contiguous_axes(sl, tv, pax);
    }
    stream_walk<T>(s, sl, tv, pax, slot_of, outs, out_layouts);
}

#define EINSUMS_STREAM_CONTRACT_INSTANTIATE(T)                                                                                             \
    template EINSUMS_EXPORT void stream_contract<T>(T const *, StreamLayout const &, std::vector<StreamTerm<T>> const &,                   \
                                                    std::vector<int> const &);

EINSUMS_STREAM_CONTRACT_INSTANTIATE(float)
EINSUMS_STREAM_CONTRACT_INSTANTIATE(double)
EINSUMS_STREAM_CONTRACT_INSTANTIATE(std::complex<float>)
EINSUMS_STREAM_CONTRACT_INSTANTIATE(std::complex<double>)

#undef EINSUMS_STREAM_CONTRACT_INSTANTIATE

EINSUMS_NAMESPACE_END(packed_gemm)
