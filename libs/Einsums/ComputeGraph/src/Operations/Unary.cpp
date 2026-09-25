//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

namespace {
/// Record a one-in one-out Custom node that re-reads both operands through their slots on every
/// run, so the node follows rebind() and the memory planner moving a tensor's storage.
/// @p apply is called as ``apply(dst, src)``, as the eager entry calls it.
template <typename TD, typename TS, typename Fn>
void record_unary(CaptureContext &ctx, char const *name, char const *execute_label, SlotRef dst, SlotRef src, Fn apply,
                  bool reads_dst = false) {
    OperandAccessor const d_access(dst.second, packed_gemm::get_scalar_type<TD>());
    OperandAccessor const s_access(src.second, packed_gemm::get_scalar_type<TS>());
    auto                  executor = [d_access, s_access, apply, execute_label]() {
        LabeledSection(execute_label);
        apply(*d_access.impl<TD>(), *s_access.impl<TS>());
    };
    std::vector<TensorId> inputs{src.first};
    if (reads_dst) {
        inputs.push_back(dst.first);
    }
    ctx.record(OpKind::Custom, name, std::move(inputs), {dst.first}, std::move(executor));
}
} // namespace

// ── conj ──────────────────────────────────────────────────────────────────────

template <typename T>
void eager_conj(Impl<T> &A) {
    LabeledSection("conj eager");
    einsums::detail::impl_conj(A);
}

template <typename T>
void capture_conj(CaptureContext &ctx, SlotRef a, std::string_view name) {
    LabeledSection("conj capture");
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<T>());
    auto                  executor = [a_access]() {
        LabeledSection("conj execute");
        einsums::detail::impl_conj(*a_access.impl<T>());
    };
    ctx.record(OpKind::Custom, fmt::format("conj({})", name), {a.first}, {a.first}, std::move(executor));
}

// ── real / imag / abs ─────────────────────────────────────────────────────────

namespace {
template <typename T>
void run_complex_part(ComplexPart part, Impl<T> const &a, Impl<RemoveComplexT<T>> &o) {
    switch (part) {
    case ComplexPart::Real:
        if constexpr (IsComplexV<T>) {
            einsums::detail::impl_real(a, o);
        } else {
            einsums::detail::impl_copy(a, o); // Re(x) == x for real x
        }
        break;
    case ComplexPart::Imag:
        if constexpr (IsComplexV<T>) {
            einsums::detail::impl_imag(a, o);
        } else {
            // Im(x) == 0 for real x: copy then scale by zero (avoids reading
            // uninitialized output the way a bare scal(0) would).
            einsums::detail::impl_copy(a, o);
            einsums::detail::impl_scal(RemoveComplexT<T>{0}, o);
        }
        break;
    case ComplexPart::Abs:
        einsums::detail::impl_abs(a, o);
        break;
    }
}

constexpr char const *part_name(ComplexPart part) {
    return part == ComplexPart::Real ? "real" : part == ComplexPart::Imag ? "imag" : "abs";
}
constexpr char const *part_execute(ComplexPart part) {
    return part == ComplexPart::Real ? "real execute" : part == ComplexPart::Imag ? "imag execute" : "abs execute";
}
} // namespace

template <typename T>
void eager_complex_part(ComplexPart part, Impl<T> const &A, Impl<RemoveComplexT<T>> &out) {
    LabeledSection(part == ComplexPart::Real ? "real eager" : part == ComplexPart::Imag ? "imag eager" : "abs eager");
    run_complex_part<T>(part, A, out);
}

template <typename T>
void capture_complex_part(CaptureContext &ctx, ComplexPart part, SlotRef out, SlotRef a) {
    LabeledSection(part == ComplexPart::Real ? "real capture" : part == ComplexPart::Imag ? "imag capture" : "abs capture");
    record_unary<RemoveComplexT<T>, T>(ctx, part_name(part), part_execute(part), out, a,
                                       [part](Impl<RemoveComplexT<T>> &o, Impl<T> const &src) { run_complex_part<T>(part, src, o); });
}

// ── sqrt ──────────────────────────────────────────────────────────────────────

namespace {
template <typename T>
void run_sqrt(Impl<T> &o, Impl<T> const &a) {
    size_t const N     = a.rank();
    size_t       total = 1;
    for (size_t k = 0; k < N; ++k)
        total *= a.dim(k);
    if (total == 0)
        return;
    std::vector<size_t> idx(N, 0), o_str(N), a_str(N), dims(N);
    for (size_t k = 0; k < N; ++k) {
        o_str[k] = o.stride(k);
        a_str[k] = a.stride(k);
        dims[k]  = a.dim(k);
    }
    T       *o_data = o.data();
    T const *a_data = a.data();
    for (size_t count = 0; count < total; ++count) {
        size_t o_off = 0, a_off = 0;
        for (size_t k = 0; k < N; ++k) {
            o_off += idx[k] * o_str[k];
            a_off += idx[k] * a_str[k];
        }
        T const v = a_data[a_off];
        if (v < T{0}) {
            EINSUMS_THROW_EXCEPTION(std::domain_error, "cg::sqrt: negative input {}; use abs() first if that is intended",
                                    static_cast<double>(v));
        }
        o_data[o_off] = std::sqrt(v);
        for (size_t k = 0; k < N; ++k) {
            if (++idx[k] < dims[k])
                break;
            idx[k] = 0;
        }
    }
}
} // namespace

template <typename T>
void eager_sqrt(Impl<T> const &A, Impl<T> &out) {
    LabeledSection("sqrt eager");
    run_sqrt<T>(out, A);
}

template <typename T>
void capture_sqrt(CaptureContext &ctx, SlotRef out, SlotRef a) {
    LabeledSection("sqrt capture");
    record_unary<T, T>(ctx, "sqrt", "sqrt execute", out, a, [](Impl<T> &o, Impl<T> const &src) { run_sqrt<T>(o, src); });
}

// ── diagonal ──────────────────────────────────────────────────────────────────

namespace {
template <typename T>
void run_diagonal(Impl<T> &o, Impl<T> const &a) {
    size_t const n      = std::min(a.dim(0), a.dim(1));
    T           *o_data = o.data();
    T const     *a_data = a.data();
    size_t const s0 = a.stride(0), s1 = a.stride(1), os = o.stride(0);
    for (size_t i = 0; i < n; ++i)
        o_data[i * os] = a_data[i * s0 + i * s1];
}
} // namespace

template <typename T>
void eager_diagonal(Impl<T> const &A, Impl<T> &out) {
    LabeledSection("diagonal eager");
    run_diagonal<T>(out, A);
}

template <typename T>
void capture_diagonal(CaptureContext &ctx, SlotRef out, SlotRef a) {
    LabeledSection("diagonal capture");
    record_unary<T, T>(ctx, "diagonal", "diagonal execute", out, a, [](Impl<T> &o, Impl<T> const &src) { run_diagonal<T>(o, src); });
}

// ── shift ─────────────────────────────────────────────────────────────────────

namespace {
template <typename T>
void run_shift(T beta, Impl<T> &target) {
    T           *data = target.data();
    size_t const n    = target.size();
    for (size_t i = 0; i < n; ++i) {
        data[i] += beta;
    }
}
} // namespace

template <typename T>
void eager_shift(T beta, Impl<T> &A) {
    LabeledSection("shift eager");
    run_shift<T>(beta, A);
}

template <typename T>
void capture_shift(CaptureContext &ctx, T beta, SlotRef a, std::string_view name) {
    LabeledSection("shift capture");
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<T>());
    auto                  executor = [beta, a_access]() {
        LabeledSection("shift execute");
        run_shift<T>(beta, *a_access.impl<T>());
    };
    ctx.record(OpKind::Custom, fmt::format("shift({})", name), {a.first}, {a.first}, std::move(executor));
}

#define EINSUMS_UNARY_OPERATIONS(T)                                                                                                        \
    template EINSUMS_EXPORT void eager_conj<T>(Impl<T> &);                                                                                 \
    template EINSUMS_EXPORT void capture_conj<T>(CaptureContext &, SlotRef, std::string_view);                                             \
    template EINSUMS_EXPORT void eager_complex_part<T>(ComplexPart, Impl<T> const &, Impl<RemoveComplexT<T>> &);                           \
    template EINSUMS_EXPORT void capture_complex_part<T>(CaptureContext &, ComplexPart, SlotRef, SlotRef);                                 \
    template EINSUMS_EXPORT void eager_diagonal<T>(Impl<T> const &, Impl<T> &);                                                            \
    template EINSUMS_EXPORT void capture_diagonal<T>(CaptureContext &, SlotRef, SlotRef);                                                  \
    template EINSUMS_EXPORT void eager_shift<T>(T, Impl<T> &);                                                                             \
    template EINSUMS_EXPORT void capture_shift<T>(CaptureContext &, T, SlotRef, std::string_view);

EINSUMS_UNARY_OPERATIONS(float)
EINSUMS_UNARY_OPERATIONS(double)
EINSUMS_UNARY_OPERATIONS(std::complex<float>)
EINSUMS_UNARY_OPERATIONS(std::complex<double>)
#undef EINSUMS_UNARY_OPERATIONS

// sqrt is real-only: a complex square root needs a branch choice the caller must make.
#define EINSUMS_REAL_UNARY_OPERATIONS(T)                                                                                                   \
    template EINSUMS_EXPORT void eager_sqrt<T>(Impl<T> const &, Impl<T> &);                                                                \
    template EINSUMS_EXPORT void capture_sqrt<T>(CaptureContext &, SlotRef, SlotRef);

EINSUMS_REAL_UNARY_OPERATIONS(float)
EINSUMS_REAL_UNARY_OPERATIONS(double)
#undef EINSUMS_REAL_UNARY_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
