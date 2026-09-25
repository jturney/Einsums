//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <algorithm>
#include <complex>
#include <vector>

#include "Record.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {

template <typename T>
void run_block_copy(Impl<T> &d, Impl<T> const &s, std::vector<size_t> const &dst_offsets, std::vector<size_t> const &src_offsets,
                    std::vector<size_t> const &extents) {
    size_t const        N = extents.size();
    std::vector<size_t> d_str(N), s_str(N);
    for (size_t k = 0; k < N; ++k) {
        d_str[k] = d.stride(k);
        s_str[k] = s.stride(k);
    }
    T       *d_data = d.data();
    T const *s_data = s.data();
    // Axis 0 fastest, correctness-only, not cache-aware.
    for_each_index(extents, [&](std::vector<size_t> const &idx) {
        size_t d_off = 0, s_off = 0;
        for (size_t k = 0; k < N; ++k) {
            d_off += (dst_offsets[k] + idx[k]) * d_str[k];
            s_off += (src_offsets[k] + idx[k]) * s_str[k];
        }
        d_data[d_off] = s_data[s_off];
    });
}

template <typename T>
void run_gather(Impl<T> &d, Impl<T> const &s, std::vector<std::vector<size_t>> const &indices, std::vector<size_t> const &extents,
                std::vector<size_t> const &dst_axis) {
    size_t const        N = indices.size();
    std::vector<size_t> d_str(N), s_str(N);
    for (size_t k = 0; k < N; ++k) {
        // Indexed by SOURCE axis, so the traversal below is unchanged: it
        // walks the destination linearly through whatever strides it is
        // handed, and a permutation is just a different set of strides. The
        // contiguous-run fast path keys on d_str[0] == 1, so it switches
        // itself off exactly when the permutation breaks that.
        d_str[k] = d.stride(dst_axis[k]);
        s_str[k] = s.stride(k);
    }
    T       *d_data = d.data();
    T const *s_data = s.data();

    // The source is the indexed side; the destination is walked linearly -
    // which is what makes the parallel opt-in sound here: every op call
    // writes a disjoint destination run no matter what the index lists
    // hold. scatter and scatter_add stay serial; see the walker's contract.
    for_each_selection_run(
        indices, extents, s_str, d_str, [&](size_t s_off, size_t d_off, size_t n) { std::copy_n(s_data + s_off, n, d_data + d_off); },
        /*parallel=*/true);
}

template <typename T>
void run_scatter(bool accumulate, Impl<T> &d, Impl<T> const &s, std::vector<std::vector<size_t>> const &indices,
                 std::vector<size_t> const &extents) {
    size_t const        N = indices.size();
    std::vector<size_t> d_str(N), s_str(N);
    for (size_t k = 0; k < N; ++k) {
        d_str[k] = d.stride(k);
        s_str[k] = s.stride(k);
    }
    T       *d_data = d.data();
    T const *s_data = s.data();

    // The destination is the indexed side here, which is what makes this
    // the inverse of gather; the source is walked linearly.
    if (accumulate) {
        // A repeated index only collapses into a run when it repeats the
        // PREVIOUS index plus one, which it cannot, so the run path never merges
        // two writes to the same element.
        for_each_selection_run(indices, extents, d_str, s_str, [&](size_t d_off, size_t s_off, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                d_data[d_off + i] += s_data[s_off + i];
            }
        });
    } else {
        for_each_selection_run(indices, extents, d_str, s_str,
                               [&](size_t d_off, size_t s_off, size_t n) { std::copy_n(s_data + s_off, n, d_data + d_off); });
    }
}
} // namespace

template <typename T>
void eager_block_copy(Impl<T> &dst, Impl<T> const &src, std::vector<size_t> const &dst_offsets, std::vector<size_t> const &src_offsets,
                      std::vector<size_t> const &extents) {
    LabeledSection("block_copy eager");
    run_block_copy<T>(dst, src, dst_offsets, src_offsets, extents);
}

template <typename T>
void capture_block_copy(CaptureContext &ctx, SlotRef dst, SlotRef src, std::vector<size_t> dst_offsets, std::vector<size_t> src_offsets,
                        std::vector<size_t> extents) {
    LabeledSection("block_copy capture");
    record_unary<T, T>(ctx, "block_copy", "block_copy execute", dst, src,
                       [dst_offsets = std::move(dst_offsets), src_offsets = std::move(src_offsets), extents = std::move(extents)](
                           Impl<T> &d, Impl<T> const &s) { run_block_copy<T>(d, s, dst_offsets, src_offsets, extents); });
}

template <typename T>
void eager_gather(Impl<T> &dst, Impl<T> const &src, std::vector<std::vector<size_t>> const &indices, std::vector<size_t> const &extents,
                  std::vector<size_t> const &dst_axis) {
    LabeledSection("gather eager");
    run_gather<T>(dst, src, indices, extents, dst_axis);
}

template <typename T>
void capture_gather(CaptureContext &ctx, SlotRef dst, SlotRef src, std::vector<std::vector<size_t>> indices, std::vector<size_t> extents,
                    std::vector<size_t> dst_axis) {
    LabeledSection("gather capture");
    record_unary<T, T>(ctx, "gather", "gather execute", dst, src,
                       [indices = std::move(indices), extents = std::move(extents),
                        dst_axis = std::move(dst_axis)](Impl<T> &d, Impl<T> const &s) { run_gather<T>(d, s, indices, extents, dst_axis); });
}

template <typename T>
void eager_scatter(bool accumulate, Impl<T> &dst, Impl<T> const &src, std::vector<std::vector<size_t>> const &indices,
                   std::vector<size_t> const &extents) {
    LabeledSection(accumulate ? "scatter_add eager" : "scatter eager");
    run_scatter<T>(accumulate, dst, src, indices, extents);
}

template <typename T>
void capture_scatter(CaptureContext &ctx, bool accumulate, SlotRef dst, SlotRef src, std::vector<std::vector<size_t>> indices,
                     std::vector<size_t> extents) {
    LabeledSection(accumulate ? "scatter_add capture" : "scatter capture");
    // dst is BOTH an input and an output: a scatter leaves everything outside
    // the selection untouched, so whatever wrote those elements has to be
    // ordered before this node; a scatter_add accumulates onto what is there.
    record_unary<T, T>(
        ctx, accumulate ? "scatter_add" : "scatter", accumulate ? "scatter_add execute" : "scatter execute", dst, src,
        [accumulate, indices = std::move(indices), extents = std::move(extents)](Impl<T> &d, Impl<T> const &s) {
            run_scatter<T>(accumulate, d, s, indices, extents);
        },
        /*reads_dst=*/true);
}

#define EINSUMS_SELECTION_OPERATIONS(T)                                                                                                    \
    template EINSUMS_EXPORT void eager_block_copy<T>(Impl<T> &, Impl<T> const &, std::vector<size_t> const &, std::vector<size_t> const &, \
                                                     std::vector<size_t> const &);                                                         \
    template EINSUMS_EXPORT void capture_block_copy<T>(CaptureContext &, SlotRef, SlotRef, std::vector<size_t>, std::vector<size_t>,       \
                                                       std::vector<size_t>);                                                               \
    template EINSUMS_EXPORT void eager_gather<T>(Impl<T> &, Impl<T> const &, std::vector<std::vector<size_t>> const &,                     \
                                                 std::vector<size_t> const &, std::vector<size_t> const &);                                \
    template EINSUMS_EXPORT void capture_gather<T>(CaptureContext &, SlotRef, SlotRef, std::vector<std::vector<size_t>>,                   \
                                                   std::vector<size_t>, std::vector<size_t>);                                              \
    template EINSUMS_EXPORT void eager_scatter<T>(bool, Impl<T> &, Impl<T> const &, std::vector<std::vector<size_t>> const &,              \
                                                  std::vector<size_t> const &);                                                            \
    template EINSUMS_EXPORT void capture_scatter<T>(CaptureContext &, bool, SlotRef, SlotRef, std::vector<std::vector<size_t>>,            \
                                                    std::vector<size_t>);

EINSUMS_CG_ELEMENT_TYPES(EINSUMS_SELECTION_OPERATIONS)
#undef EINSUMS_SELECTION_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
