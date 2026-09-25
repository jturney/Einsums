//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/Detail/GroupedMembers.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <complex>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

namespace {
template <typename T>
std::vector<OperandAccessor> accessors(std::vector<SlotRef> const &refs) {
    std::vector<OperandAccessor> out;
    out.reserve(refs.size());
    for (SlotRef const &r : refs) {
        // Read through the slot, not a captured pointer: rebind() and the
        // MemoryPlanning arena can both move a tensor's storage.
        out.emplace_back(r.second, packed_gemm::get_scalar_type<T>());
    }
    return out;
}
} // namespace

// ── grouped_dot ───────────────────────────────────────────────────────────────

template <typename T>
void eager_grouped_dot(std::vector<Impl<T> *> const &results, std::vector<Impl<T> const *> const &a,
                       std::vector<Impl<T> const *> const &b) {
    LabeledSection("grouped_dot eager");
    blas::SerialVendorScope const serial;
    for (size_t i = 0; i < results.size(); i++) {
        results[i]->data()[0] = linear_algebra::detail::dot(*a[i], *b[i]);
    }
}

template <typename T>
void capture_grouped_dot(CaptureContext &ctx, std::vector<SlotRef> const &results, std::vector<SlotRef> const &a,
                         std::vector<SlotRef> const &b) {
    LabeledSection("grouped_dot capture");
    size_t const count = results.size();
    // Inputs interleaved A_0, B_0, A_1, B_1, ... and outputs in entry order, so
    // entry i indexes both. The same convention the batched nodes use.
    std::vector<TensorId> inputs, outputs;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        inputs.push_back(a[i].first);
        inputs.push_back(b[i].first);
        outputs.push_back(results[i].first);
    }

    auto executor = [r_access = accessors<T>(results), a_access = accessors<T>(a), b_access = accessors<T>(b)]() {
        LabeledSection("grouped_dot execute");
        blas::SerialVendorScope const serial;
        for (size_t i = 0; i < r_access.size(); i++) {
            r_access[i].template impl<T>()->data()[0] =
                linear_algebra::detail::dot(*a_access[i].template impl<T>(), *b_access[i].template impl<T>());
        }
    };

    GroupedDotDescriptor d;
    d.total = static_cast<int>(count);
    ctx.record(OpKind::GroupedDot, fmt::format("dot x{}", count), std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

// ── grouped_axpby ─────────────────────────────────────────────────────────────

template <typename T>
void eager_grouped_axpby(std::vector<T> const &alphas, std::vector<Impl<T> const *> const &x, std::vector<T> const &betas,
                         std::vector<Impl<T> *> const &y) {
    LabeledSection("grouped_axpby eager");
    for (size_t i = 0; i < x.size(); i++) {
        linear_algebra::detail::axpby(alphas[i], *x[i], betas[i], y[i]);
    }
}

template <typename T>
void capture_grouped_axpby(CaptureContext &ctx, std::vector<T> alphas, std::vector<T> betas, std::vector<SlotRef> const &x,
                           std::vector<SlotRef> const &y) {
    LabeledSection("grouped_axpby capture");
    size_t const          count = x.size();
    std::vector<TensorId> inputs, outputs;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        inputs.push_back(x[i].first);
        // beta != 0 means this entry reads its destination before writing it, so
        // the RAW edge from whoever produced Y must survive (bug-1009).
        if (betas[i] != T{0}) {
            inputs.push_back(y[i].first);
        }
        outputs.push_back(y[i].first);
    }

    GroupedAxpbyDescriptor d;
    d.total = static_cast<int>(count);
    d.alphas.reserve(count);
    d.betas.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.alphas.emplace_back(alphas[i]);
        d.betas.emplace_back(betas[i]);
    }

    auto executor = [alphas = std::move(alphas), betas = std::move(betas), x_access = accessors<T>(x), y_access = accessors<T>(y)]() {
        LabeledSection("grouped_axpby execute");
        for (size_t i = 0; i < x_access.size(); i++) {
            linear_algebra::detail::axpby(alphas[i], *x_access[i].template impl<T>(), betas[i], y_access[i].template impl<T>());
        }
    };
    ctx.record(OpKind::GroupedAxpby, fmt::format("axpby x{}", count), std::move(inputs), std::move(outputs), std::move(executor),
               std::move(d));
}

// ── grouped_permute ───────────────────────────────────────────────────────────

template <typename T>
void eager_grouped_permute(ParsedPermuteSpec const &parsed, std::vector<T> const &c_pfs, std::vector<Impl<T> *> const &c,
                           std::vector<T> const &a_pfs, std::vector<Impl<T> const *> const &a) {
    LabeledSection("grouped_permute eager");
    run_grouped_members(c.size(), [&](size_t i) { dispatch::string_permute_impl<T>(parsed, c_pfs[i], c[i], a_pfs[i], *a[i]); });
}

template <typename T>
void capture_grouped_permute(CaptureContext &ctx, ParsedPermuteSpec parsed, std::vector<T> c_pfs, std::vector<T> a_pfs,
                             std::vector<SlotRef> const &a, std::vector<SlotRef> const &c) {
    LabeledSection("grouped_permute capture");
    size_t const          count = c.size();
    std::vector<TensorId> inputs, outputs;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        inputs.push_back(a[i].first);
        // A non-zero c_pf reads the destination before writing it, so the RAW
        // edge from whoever produced C must survive.
        if (c_pfs[i] != T{0}) {
            inputs.push_back(c[i].first);
        }
        outputs.push_back(c[i].first);
    }

    GroupedElementwiseDescriptor d;
    d.total     = static_cast<int>(count);
    d.c_indices = parsed.c_indices;
    d.a_indices = parsed.a_indices;
    d.alphas.reserve(count);
    d.betas.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.alphas.emplace_back(a_pfs[i]);
        d.betas.emplace_back(c_pfs[i]);
    }
    auto label = fmt::format("permute x{}: C[{}] = A[{}]", count, fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","));

    auto executor = [parsed = std::move(parsed), c_pfs = std::move(c_pfs), a_pfs = std::move(a_pfs), c_access = accessors<T>(c),
                     a_access = accessors<T>(a)]() {
        LabeledSection("grouped_permute execute");
        run_grouped_members(c_access.size(), [&](size_t i) {
            dispatch::string_permute_impl<T>(parsed, c_pfs[i], c_access[i].template impl<T>(), a_pfs[i], *a_access[i].template impl<T>());
        });
    };
    ctx.record(OpKind::GroupedPermute, std::move(label), std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

// ── grouped direct product / division ─────────────────────────────────────────

namespace {
template <typename T>
void run_binary(bool division, T alpha, Impl<T> const &a, Impl<T> const &b, T beta, Impl<T> *c) {
    if (division) {
        linear_algebra::detail::direct_division(alpha, a, b, beta, c);
    } else {
        linear_algebra::detail::direct_product(alpha, a, b, beta, c);
    }
}
} // namespace

template <typename T>
void eager_grouped_binary(OpKind kind, std::vector<T> const &alphas, std::vector<Impl<T> const *> const &a,
                          std::vector<Impl<T> const *> const &b, std::vector<T> const &betas, std::vector<Impl<T> *> const &c) {
    bool const division = kind == OpKind::GroupedDirectDivision;
    LabeledSection(division ? "grouped_direct_division eager" : "grouped_direct_product eager");
    run_grouped_members(c.size(), [&](size_t i) { run_binary<T>(division, alphas[i], *a[i], *b[i], betas[i], c[i]); });
}

template <typename T>
void capture_grouped_binary(CaptureContext &ctx, OpKind kind, char const *label, std::vector<T> const &alphas, std::vector<T> const &betas,
                            std::vector<SlotRef> const &a, std::vector<SlotRef> const &b, std::vector<SlotRef> const &c) {
    bool const division = kind == OpKind::GroupedDirectDivision;
    LabeledSection(division ? "grouped_direct_division capture" : "grouped_direct_product capture");
    size_t const          count = c.size();
    std::vector<TensorId> inputs, outputs;
    inputs.reserve(3 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        inputs.push_back(a[i].first);
        inputs.push_back(b[i].first);
        // A non-zero beta reads the destination before writing it.
        if (betas[i] != T{0}) {
            inputs.push_back(c[i].first);
        }
        outputs.push_back(c[i].first);
    }

    auto executor = [division, alphas, betas, a_access = accessors<T>(a), b_access = accessors<T>(b), c_access = accessors<T>(c)]() {
        run_grouped_members(c_access.size(), [&](size_t i) {
            run_binary<T>(division, alphas[i], *a_access[i].template impl<T>(), *b_access[i].template impl<T>(), betas[i],
                          c_access[i].template impl<T>());
        });
    };

    GroupedElementwiseDescriptor d;
    d.total = static_cast<int>(count);
    d.alphas.reserve(count);
    d.betas.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.alphas.emplace_back(alphas[i]);
        d.betas.emplace_back(betas[i]);
    }
    ctx.record(kind, fmt::format("{} x{}", label, count), std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

#define EINSUMS_GROUPED_OPERATIONS(T)                                                                                                      \
    template EINSUMS_EXPORT void eager_grouped_dot<T>(std::vector<Impl<T> *> const &, std::vector<Impl<T> const *> const &,                \
                                                      std::vector<Impl<T> const *> const &);                                               \
    template EINSUMS_EXPORT void capture_grouped_dot<T>(CaptureContext &, std::vector<SlotRef> const &, std::vector<SlotRef> const &,      \
                                                        std::vector<SlotRef> const &);                                                     \
    template EINSUMS_EXPORT void eager_grouped_axpby<T>(std::vector<T> const &, std::vector<Impl<T> const *> const &,                      \
                                                        std::vector<T> const &, std::vector<Impl<T> *> const &);                           \
    template EINSUMS_EXPORT void capture_grouped_axpby<T>(CaptureContext &, std::vector<T>, std::vector<T>, std::vector<SlotRef> const &,  \
                                                          std::vector<SlotRef> const &);                                                   \
    template EINSUMS_EXPORT void eager_grouped_permute<T>(ParsedPermuteSpec const &, std::vector<T> const &,                               \
                                                          std::vector<Impl<T> *> const &, std::vector<T> const &,                          \
                                                          std::vector<Impl<T> const *> const &);                                           \
    template EINSUMS_EXPORT void capture_grouped_permute<T>(CaptureContext &, ParsedPermuteSpec, std::vector<T>, std::vector<T>,           \
                                                            std::vector<SlotRef> const &, std::vector<SlotRef> const &);                   \
    template EINSUMS_EXPORT void eager_grouped_binary<T>(OpKind, std::vector<T> const &, std::vector<Impl<T> const *> const &,             \
                                                         std::vector<Impl<T> const *> const &, std::vector<T> const &,                     \
                                                         std::vector<Impl<T> *> const &);                                                  \
    template EINSUMS_EXPORT void capture_grouped_binary<T>(CaptureContext &, OpKind, char const *, std::vector<T> const &,                 \
                                                           std::vector<T> const &, std::vector<SlotRef> const &,                           \
                                                           std::vector<SlotRef> const &, std::vector<SlotRef> const &);

EINSUMS_GROUPED_OPERATIONS(float)
EINSUMS_GROUPED_OPERATIONS(double)
EINSUMS_GROUPED_OPERATIONS(std::complex<float>)
EINSUMS_GROUPED_OPERATIONS(std::complex<double>)
#undef EINSUMS_GROUPED_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
