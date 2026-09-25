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

#include <complex>
#include <stdexcept>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

namespace {
/// A one-in one-out Custom node that re-reads both operands through their slots on every run, so
/// the node follows rebind() and the memory planner moving a tensor's storage.
template <typename T, typename Fn>
void record_unary(CaptureContext &ctx, char const *name, char const *execute_label, SlotRef dst, SlotRef src, Fn apply) {
    constexpr auto        dtype = packed_gemm::get_scalar_type<T>();
    OperandAccessor const d_access(dst.second, dtype);
    OperandAccessor const s_access(src.second, dtype);
    auto                  executor = [d_access, s_access, apply, execute_label]() {
        LabeledSection(execute_label);
        apply(*d_access.impl<T>(), *s_access.impl<T>());
    };
    ctx.record(OpKind::Custom, name, {src.first}, {dst.first}, std::move(executor));
}

template <typename T>
void run_sum_axes(Impl<T> &o, Impl<T> const &a, std::vector<size_t> const &kept) {
    size_t const        N     = a.rank();
    size_t              total = 1;
    std::vector<size_t> dims(N), a_str(N);
    for (size_t k = 0; k < N; ++k) {
        dims[k]  = a.dim(k);
        a_str[k] = a.stride(k);
        total *= dims[k];
    }
    size_t              out_total = 1;
    std::vector<size_t> o_str(kept.size());
    for (size_t k = 0; k < kept.size(); ++k) {
        o_str[k] = o.stride(k);
        out_total *= o.dim(k);
    }
    T *o_data = o.data();
    // Assign, not accumulate: zero first so a replay does not add to the
    // previous execution's result.
    for (size_t k = 0; k < out_total; ++k) {
        size_t off = 0, rem = k;
        for (size_t d = 0; d < kept.size(); ++d) {
            off += (rem % o.dim(d)) * o_str[d];
            rem /= o.dim(d);
        }
        o_data[off] = T{0};
    }
    if (total == 0)
        return;
    T const            *a_data = a.data();
    std::vector<size_t> idx(N, 0);
    for (size_t count = 0; count < total; ++count) {
        size_t a_off = 0, o_off = 0;
        for (size_t k = 0; k < N; ++k)
            a_off += idx[k] * a_str[k];
        for (size_t k = 0; k < kept.size(); ++k)
            o_off += idx[kept[k]] * o_str[k];
        o_data[o_off] += a_data[a_off];
        for (size_t k = 0; k < N; ++k) {
            if (++idx[k] < dims[k])
                break;
            idx[k] = 0;
        }
    }
}

template <typename T>
void run_reshape(Impl<T> &o, Impl<T> const &a, bool row_major) {
    size_t const a_rank = a.rank(), o_rank = o.rank();
    size_t       a_total = 1;
    for (size_t k = 0; k < a_rank; ++k)
        a_total *= a.dim(k);
    if (a_total == 0)
        return;
    std::vector<size_t> a_dims(a_rank), a_str(a_rank), o_dims(o_rank), o_str(o_rank);
    for (size_t k = 0; k < a_rank; ++k) {
        a_dims[k] = a.dim(k);
        a_str[k]  = a.stride(k);
    }
    for (size_t k = 0; k < o_rank; ++k) {
        o_dims[k] = o.dim(k);
        o_str[k]  = o.stride(k);
    }
    T       *o_data = o.data();
    T const *a_data = a.data();
    // Walk the shared linear index and decompose it into each shape.
    for (size_t lin = 0; lin < a_total; ++lin) {
        size_t a_off = 0, o_off = 0, rem = lin;
        if (row_major) {
            for (size_t k = a_rank; k-- > 0;) {
                a_off += (rem % a_dims[k]) * a_str[k];
                rem /= a_dims[k];
            }
            rem = lin;
            for (size_t k = o_rank; k-- > 0;) {
                o_off += (rem % o_dims[k]) * o_str[k];
                rem /= o_dims[k];
            }
        } else {
            for (size_t k = 0; k < a_rank; ++k) {
                a_off += (rem % a_dims[k]) * a_str[k];
                rem /= a_dims[k];
            }
            rem = lin;
            for (size_t k = 0; k < o_rank; ++k) {
                o_off += (rem % o_dims[k]) * o_str[k];
                rem /= o_dims[k];
            }
        }
        o_data[o_off] = a_data[a_off];
    }
}

template <typename T>
void run_outer_sum(Impl<T> &r, std::vector<Impl<T> const *> const &vecs, std::vector<T> const &coeffs) {
    size_t const N = vecs.size();
    // Dim check (deferred from capture time so view operands can resolve).
    for (size_t k = 0; k < N; ++k) {
        if (vecs[k]->dim(0) != r.dim(k)) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: vector[{}] length ({}) doesn't match result dim {} ({})", k,
                                    vecs[k]->dim(0), k, r.dim(k));
        }
    }
    size_t const        total = r.size();
    std::vector<size_t> idx(N, 0);
    std::vector<size_t> dims(N), strides(N);
    for (size_t k = 0; k < N; ++k) {
        dims[k]    = r.dim(k);
        strides[k] = r.stride(k);
    }
    T *out = r.data();
    for (size_t count = 0; count < total; ++count) {
        T sum{};
        for (size_t k = 0; k < N; ++k) {
            sum += coeffs[k] * vecs[k]->data()[idx[k]];
        }
        size_t offset = 0;
        for (size_t k = 0; k < N; ++k)
            offset += idx[k] * strides[k];
        out[offset] = sum;
        // Increment multi-index (axis 0 fastest, direction is irrelevant for correctness).
        for (size_t k = 0; k < N; ++k) {
            if (++idx[k] < dims[k])
                break;
            idx[k] = 0;
        }
    }
}
} // namespace

template <typename T>
void eager_sum_axes(Impl<T> &out, Impl<T> const &A, std::vector<size_t> const &kept) {
    LabeledSection("sum_axes eager");
    run_sum_axes<T>(out, A, kept);
}

template <typename T>
void capture_sum_axes(CaptureContext &ctx, SlotRef out, SlotRef a, std::vector<size_t> kept) {
    LabeledSection("sum_axes capture");
    record_unary<T>(ctx, "sum_axes", "sum_axes execute", out, a,
                    [kept = std::move(kept)](Impl<T> &o, Impl<T> const &src) { run_sum_axes<T>(o, src, kept); });
}

template <typename T>
void eager_reshape(Impl<T> &out, Impl<T> const &A, bool row_major) {
    LabeledSection("reshape eager");
    run_reshape<T>(out, A, row_major);
}

template <typename T>
void capture_reshape(CaptureContext &ctx, SlotRef out, SlotRef a, bool row_major) {
    LabeledSection("reshape capture");
    record_unary<T>(ctx, "reshape", "reshape execute", out, a,
                    [row_major](Impl<T> &o, Impl<T> const &src) { run_reshape<T>(o, src, row_major); });
}

template <typename T>
void eager_outer_sum(Impl<T> &result, std::vector<Impl<T> const *> const &vectors, std::vector<T> const &coeffs) {
    LabeledSection("outer_sum eager");
    run_outer_sum<T>(result, vectors, coeffs);
}

template <typename T>
void capture_outer_sum(CaptureContext &ctx, SlotRef result, std::vector<SlotRef> vectors, std::vector<T> coeffs,
                       std::vector<double> coefficients) {
    LabeledSection("outer_sum capture");
    constexpr auto dtype = packed_gemm::get_scalar_type<T>();
    size_t const   N     = vectors.size();

    std::vector<TensorId>        in_ids;
    std::vector<OperandAccessor> v_access;
    in_ids.reserve(N);
    v_access.reserve(N);
    for (auto const &v : vectors) {
        in_ids.push_back(v.first);
        v_access.emplace_back(v.second, dtype);
    }
    OperandAccessor const r_access(result.second, dtype);

    // The vector operands are re-read through their slots on every run, so the
    // node follows a rebind, as the eager path reads the caller's originals.
    auto executor = [v_access, r_access, coeffs = std::move(coeffs), N]() {
        LabeledSection("outer_sum execute");
        std::vector<Impl<T> const *> rebound(N);
        for (size_t k = 0; k < N; ++k)
            rebound[k] = v_access[k].impl<T>();
        run_outer_sum<T>(*r_access.impl<T>(), rebound, coeffs);
    };

    // The coefficients ride on the node as well as inside the executor, so a pass can read what
    // this computes. `LaplaceTransform` accepts a denominator its graph writes only when it can
    // VERIFY the recipe, and one signed coefficient per axis is the whole of the recipe; without
    // the numbers here they are only inside a closure.
    OuterSumDescriptor desc;
    desc.coefficients = coefficients.empty() ? std::vector<double>(N, 1.0) : std::move(coefficients);
    ctx.record(OpKind::Custom, "outer_sum", std::move(in_ids), {result.first}, std::move(executor), std::move(desc));
}

#define EINSUMS_SHAPE_OPERATIONS(T)                                                                                                        \
    template EINSUMS_EXPORT void eager_sum_axes<T>(Impl<T> &, Impl<T> const &, std::vector<size_t> const &);                               \
    template EINSUMS_EXPORT void capture_sum_axes<T>(CaptureContext &, SlotRef, SlotRef, std::vector<size_t>);                             \
    template EINSUMS_EXPORT void eager_reshape<T>(Impl<T> &, Impl<T> const &, bool);                                                       \
    template EINSUMS_EXPORT void capture_reshape<T>(CaptureContext &, SlotRef, SlotRef, bool);                                             \
    template EINSUMS_EXPORT void eager_outer_sum<T>(Impl<T> &, std::vector<Impl<T> const *> const &, std::vector<T> const &);              \
    template EINSUMS_EXPORT void capture_outer_sum<T>(CaptureContext &, SlotRef, std::vector<SlotRef>, std::vector<T>, std::vector<double>);

EINSUMS_SHAPE_OPERATIONS(float)
EINSUMS_SHAPE_OPERATIONS(double)
EINSUMS_SHAPE_OPERATIONS(std::complex<float>)
EINSUMS_SHAPE_OPERATIONS(std::complex<double>)
#undef EINSUMS_SHAPE_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
