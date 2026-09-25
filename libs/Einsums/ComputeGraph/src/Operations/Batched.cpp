//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/BatchedGemm.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/Detail/GroupedBatchedGemm.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <complex>
#include <cstddef>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

// ── batched gemm ──────────────────────────────────────────────────────────────

template <typename T>
void eager_batched_gemm(BatchedGemmDescriptor const &d, std::vector<void const *> const &a, std::vector<void const *> const &b,
                        std::vector<void *> const &c) {
    LabeledSection("batched_gemm eager");
    if constexpr (IsComplexV<T>) {
        run_batched_gemm_complex<T>(d, a, b, c);
    } else {
        run_batched_gemm<T>(d, a, b, c);
    }
}

void capture_batched_gemm(CaptureContext &ctx, BatchedGemmDescriptor const &d, bool reads_c, std::vector<SlotRef> const &a,
                          std::vector<SlotRef> const &b, std::vector<SlotRef> const &c) {
    LabeledSection("batched_gemm capture");
    size_t const        count = a.size();
    BatchedGemmOperands a_ops, b_ops, c_ops;
    a_ops.reserve(count);
    b_ops.reserve(count);
    c_ops.reserve(count);
    // Node I/O keeps the pass's convention: inputs interleaved A_0, B_0, A_1,
    // B_1, ... and outputs C_0, C_1, ... in batch order.
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        inputs.push_back(a[i].first);
        inputs.push_back(b[i].first);
        outputs.push_back(c[i].first);
        // Read through the slot, not the captured pointer: rebind() and the
        // MemoryPlanning arena can both move a tensor's storage.
        a_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{a[i].second}, .offset = 0});
        b_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{b[i].second}, .offset = 0});
        c_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{c[i].second}, .offset = 0});
    }
    // beta != 0 means gemm_batch reads every destination before writing it, so
    // the RAW edge from whoever produced each C must survive (bug-1009).
    if (reads_c) {
        inputs.insert(inputs.end(), outputs.begin(), outputs.end());
    }

    auto executor = make_batched_gemm_executor(d, std::move(a_ops), std::move(b_ops), std::move(c_ops));
    ctx.record(OpKind::BatchedGemm,
               fmt::format("gemm_batch x{} ({}x{}x{}, trans={}{})", d.batch_count, d.m, d.k, d.n, d.trans_a, d.trans_b), std::move(inputs),
               std::move(outputs), std::move(executor), d);
}

void capture_batched_gemm_blocked(CaptureContext &ctx, BatchedGemmDescriptor const &d, bool reads_c, std::vector<SlotRef> const &a,
                                  std::vector<SlotRef> const &b, SlotRef c_base, std::vector<std::size_t> const &c_offsets) {
    LabeledSection("batched_gemm_blocked capture");
    size_t const        count = a.size();
    BatchedGemmOperands a_ops, b_ops, c_ops;
    a_ops.reserve(count);
    b_ops.reserve(count);
    c_ops.reserve(count);
    std::vector<TensorId> inputs;
    inputs.reserve(2 * count);

    for (size_t i = 0; i < count; ++i) {
        inputs.push_back(a[i].first);
        inputs.push_back(b[i].first);
        a_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{a[i].second}, .offset = 0});
        b_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{b[i].second}, .offset = 0});
        // Read the base through its slot and offset at execute time: rebind()
        // and the MemoryPlanning arena can both move the storage after capture.
        c_ops.push_back(
            BatchedGemmOperand{.accessor = OperandAccessor{c_base.second}, .offset = static_cast<std::ptrdiff_t>(c_offsets[i])});
    }

    std::vector<TensorId> outputs{c_base.first};
    // beta != 0 means gemm_batch reads every destination before writing it, so
    // the RAW edge from whoever produced c_base must survive (bug-1009).
    if (reads_c) {
        inputs.push_back(c_base.first);
    }

    auto executor = make_batched_gemm_executor(d, std::move(a_ops), std::move(b_ops), std::move(c_ops));
    ctx.record(OpKind::BatchedGemm,
               fmt::format("gemm_batch x{} into blocks ({}x{}x{}, trans={}{})", d.batch_count, d.m, d.k, d.n, d.trans_a, d.trans_b),
               std::move(inputs), std::move(outputs), std::move(executor), d);
}

template <typename T>
void eager_grouped_batched_gemm(GroupedBatchedGemmDescriptor const &d, std::vector<void const *> const &a,
                                std::vector<void const *> const &b, std::vector<void *> const &c) {
    LabeledSection("grouped_batched_gemm eager");
    run_grouped_batched_gemm<T>(d, a, b, c);
}

void capture_grouped_batched_gemm(CaptureContext &ctx, GroupedBatchedGemmDescriptor d, bool reads_c, bool trans_a, bool trans_b,
                                  std::vector<SlotRef> const &a, std::vector<SlotRef> const &b, std::vector<SlotRef> const &c,
                                  std::vector<SlotRef> const &c_bases, std::vector<std::size_t> const &c_offsets) {
    bool const blocked = !c_bases.empty();
    LabeledSection(blocked ? "grouped_batched_gemm_blocked capture" : "grouped_batched_gemm capture");
    size_t const        count = a.size();
    BatchedGemmOperands a_ops, b_ops, c_ops;
    a_ops.reserve(count);
    b_ops.reserve(count);
    c_ops.reserve(count);
    // Node I/O keeps the batched form's convention, inputs interleaved
    // A_0, B_0, A_1, B_1, ... and outputs C_0, C_1, ..., in the FLATTENED
    // order, so a group's offset indexes the extractors and the node lists
    // alike. The blocked form's outputs are its distinct bases.
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    inputs.reserve(2 * count);
    for (SlotRef const &base : c_bases) {
        outputs.push_back(base.first);
    }
    for (size_t i = 0; i < count; ++i) {
        inputs.push_back(a[i].first);
        inputs.push_back(b[i].first);
        if (!blocked) {
            outputs.push_back(c[i].first);
        }
        // Read through the slot, not the captured pointer: rebind() and the
        // MemoryPlanning arena can both move a tensor's storage.
        a_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{a[i].second}, .offset = 0});
        b_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{b[i].second}, .offset = 0});
        c_ops.push_back(BatchedGemmOperand{.accessor = OperandAccessor{c[i].second},
                                           .offset   = blocked ? static_cast<std::ptrdiff_t>(c_offsets[i]) : 0});
    }
    // beta != 0 means every destination is read before it is written, so the
    // RAW edge from whoever produced each C must survive (bug-1009).
    if (reads_c) {
        inputs.insert(inputs.end(), outputs.begin(), outputs.end());
    }

    auto executor = make_grouped_batched_gemm_executor(d, std::move(a_ops), std::move(b_ops), std::move(c_ops));
    auto label    = blocked ? fmt::format("gemm_batch_grouped x{} in {} shapes into blocks of {} (trans={}{})", d.total, d.groups.size(),
                                          outputs.size(), trans_a ? 'T' : 'N', trans_b ? 'T' : 'N')
                            : fmt::format("gemm_batch_grouped x{} in {} shapes (trans={}{})", d.total, d.groups.size(), trans_a ? 'T' : 'N',
                                          trans_b ? 'T' : 'N');
    ctx.record(OpKind::GroupedBatchedGemm, std::move(label), std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

#define EINSUMS_GROUPED_BATCHED(T)                                                                                                         \
    template EINSUMS_EXPORT void eager_grouped_batched_gemm<T>(GroupedBatchedGemmDescriptor const &, std::vector<void const *> const &,    \
                                                               std::vector<void const *> const &, std::vector<void *> const &);
EINSUMS_GROUPED_BATCHED(float)
EINSUMS_GROUPED_BATCHED(double)
EINSUMS_GROUPED_BATCHED(std::complex<float>)
EINSUMS_GROUPED_BATCHED(std::complex<double>)
#undef EINSUMS_GROUPED_BATCHED

template EINSUMS_EXPORT void eager_batched_gemm<float>(BatchedGemmDescriptor const &, std::vector<void const *> const &,
                                                       std::vector<void const *> const &, std::vector<void *> const &);
template EINSUMS_EXPORT void eager_batched_gemm<double>(BatchedGemmDescriptor const &, std::vector<void const *> const &,
                                                        std::vector<void const *> const &, std::vector<void *> const &);
template EINSUMS_EXPORT void eager_batched_gemm<std::complex<float>>(BatchedGemmDescriptor const &, std::vector<void const *> const &,
                                                                     std::vector<void const *> const &, std::vector<void *> const &);
template EINSUMS_EXPORT void eager_batched_gemm<std::complex<double>>(BatchedGemmDescriptor const &, std::vector<void const *> const &,
                                                                      std::vector<void const *> const &, std::vector<void *> const &);

EINSUMS_NAMESPACE_END(compute_graph::detail)
