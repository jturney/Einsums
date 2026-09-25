//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <array>
#include <complex>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

// ── scale ─────────────────────────────────────────────────────────────────────

template <typename T>
void eager_scale(T factor, Impl<T> &A) {
    LabeledSection("scale eager");
    linear_algebra::detail::scale(factor, &A);
}

template <typename T>
void capture_scale(CaptureContext &ctx, T factor, TensorId a_id, std::string_view name, std::size_t rank) {
    LabeledSection("scale capture");

    // The factor is recorded in the tensor's own scalar type rather than
    // projected onto a double, so a complex scale reads back exactly.
    ScaleDescriptor desc;
    desc.factor        = PrefactorScalar{factor};
    desc.params        = std::make_shared<ElementwiseParams>();
    desc.params->alpha = desc.factor;

    auto label = fmt::format("scale({}, {})", to_string(desc.factor), name);

    ctx.record_built(OpKind::Scale, std::move(label), packed_gemm::get_scalar_type<T>(), rank, std::move(desc), {},
                     std::span<TensorId const>{&a_id, 1}, {a_id}, {a_id});
}

// ── axpy / axpby ──────────────────────────────────────────────────────────────

template <typename T>
void eager_axpy(T alpha, Impl<T> const &X, Impl<T> &Y) {
    LabeledSection("axpy eager");
    linear_algebra::detail::axpy(alpha, X, &Y);
}

template <typename T>
void capture_axpy(CaptureContext &ctx, T alpha, TensorId x_id, TensorId y_id, std::string_view x_name, std::string_view y_name,
                  std::size_t rank) {
    LabeledSection("axpy capture");

    // axpy IS an axpby with beta == 1, so it records as one. It used to be
    // its own OpKind carrying no op_data at all, which made the op opaque:
    // a pass could see "something accumulates into Y" but not by how much,
    // so every scalar-aware rewrite (ScaleAbsorption, CSE, ElementWiseFusion,
    // SymmetrizedAccumulation, ...) gated on OpKind::Axpby and skipped it.
    // Since `Y += X` is how this operation is spelled in every other
    // library, the most natural spelling was the one the optimizer could not
    // see.
    //
    // Recording the same kind rather than a parallel one is what makes those
    // passes work on it, with no per-pass special-casing. The kernel choice
    // is the executor's, not the kind's: it still calls the BLAS axpy fast
    // path whenever beta == 1, which is every capture from here.
    auto params   = std::make_shared<AxpbyParams>();
    params->alpha = PrefactorScalar{alpha};
    params->beta  = PrefactorScalar{T{1}};

    auto label = fmt::format("axpy(alpha={}, {}, {})", alpha, x_name, y_name);

    AxpbyDescriptor desc;
    desc.alpha  = params->alpha;
    desc.beta   = params->beta;
    desc.params = params;

    // The built executor reads the scalars through the shared params, so
    // the descriptor is the single source of truth rather than a snapshot
    // the executor can silently disagree with. A pass that rewrites beta
    // away from 1 turns this into a genuine axpby, and the executor honors
    // that rather than ignoring the write.
    std::array<TensorId, 2> const inputs{x_id, y_id};

    // Y += alpha*X reads its destination unconditionally (beta == 1); list it
    // as an input so dependency passes see the read (matches gemm's and
    // direct_product's out-tensor-as-input convention - without it,
    // LoopInvariantHoisting's reads-its-output guard is blind to the
    // accumulation and Reorder misses the WAR hazard on Y's old value).
    ctx.record_built(OpKind::Axpby, std::move(label), packed_gemm::get_scalar_type<T>(), rank, std::move(desc), inputs,
                     std::span<TensorId const>{&y_id, 1}, {x_id, y_id}, {y_id});
}

template <typename T>
void eager_axpby(T alpha, Impl<T> const &X, T beta, Impl<T> &Y) {
    LabeledSection("axpby eager");
    linear_algebra::detail::axpby(alpha, X, beta, &Y);
}

template <typename T>
void capture_axpby(CaptureContext &ctx, T alpha, T beta, TensorId x_id, TensorId y_id, std::size_t rank) {
    LabeledSection("axpby capture");

    // Live-mutable scalars shared with the executor (single source of truth:
    // a pass that folds a scale into this axpby writes beta through params and
    // the executor honors it on replay). The descriptor keeps the at-capture
    // snapshot for analysis passes.
    auto params   = std::make_shared<AxpbyParams>();
    params->alpha = PrefactorScalar{alpha};
    params->beta  = PrefactorScalar{beta};

    auto label = fmt::format("axpby(alpha={}, beta={})", alpha, beta);

    AxpbyDescriptor desc;
    desc.alpha  = params->alpha;
    desc.beta   = params->beta;
    desc.params = params;

    // Y = alpha*X + beta*Y reads its destination when beta != 0; same
    // out-tensor-as-input convention as gemm/direct_product (see axpy).
    std::vector<TensorId> axpby_inputs = (beta != T{0}) ? std::vector<TensorId>{x_id, y_id} : std::vector<TensorId>{x_id};

    ctx.record_built(OpKind::Axpby, std::move(label), packed_gemm::get_scalar_type<T>(), rank, std::move(desc), axpby_inputs,
                     std::span<TensorId const>{&y_id, 1}, axpby_inputs, {y_id});
}

// ── direct_product / direct_division ──────────────────────────────────────────

template <typename T>
void eager_direct_product(T alpha, Impl<T> const &A, Impl<T> const &B, T beta, Impl<T> &C) {
    LabeledSection("direct_product eager");
    linear_algebra::detail::direct_product(alpha, A, B, beta, &C);
}

template <typename T>
void eager_direct_division(T alpha, Impl<T> const &A, Impl<T> const &B, T beta, Impl<T> &C) {
    LabeledSection("direct_division eager");
    linear_algebra::detail::direct_division(alpha, A, B, beta, &C);
}

void capture_elementwise_binary(CaptureContext &ctx, OpKind kind, PrefactorScalar alpha, PrefactorScalar beta, bool reads_c, TensorId a_id,
                                TensorId b_id, TensorId c_id, packed_gemm::ScalarType dtype, std::size_t rank) {
    bool const product = kind == OpKind::DirectProduct;
    LabeledSection(product ? "direct_product capture" : "direct_division capture");

    // The scalars used to be baked into the executor with no descriptor at all,
    // so the node was opaque to every pass that reasons about prefactors. The
    // product and the division share the descriptor; the kind distinguishes them.
    ElementwiseBinaryDescriptor desc;
    desc.alpha         = alpha;
    desc.beta          = beta;
    desc.params        = std::make_shared<ElementwiseParams>();
    desc.params->alpha = desc.alpha;
    desc.params->beta  = desc.beta;

    // When beta != 0 the op reads its destination (C = alpha*A*B + beta*C), so C
    // is an input as well as the output. List it -- otherwise dependency-based
    // passes (LoopInvariantHoisting, Reorder, ...) don't see the read and may
    // hoist the accumulation out of a loop or reorder it past another writer of C.
    // (gemm already does this; matches the out-tensor-as-input convention.)
    std::vector<TensorId> inputs = reads_c ? std::vector<TensorId>{a_id, b_id, c_id} : std::vector<TensorId>{a_id, b_id};

    ctx.record_built(kind, product ? "direct_product" : "direct_division", dtype, rank, std::move(desc), inputs,
                     std::span<TensorId const>{&c_id, 1}, inputs, {c_id});
}

// ── element_transform by name ─────────────────────────────────────────────────

template <typename T>
void eager_element_transform(Impl<T> &C, std::string_view op_name, std::optional<double> param) {
    auto kernel = element_ops::global_element_op_registry().kernel<T>(op_name, param);
    LabeledSection("element_transform eager");
    element_ops::detail::apply_element_op<T>(kernel, &C);
}

template <typename T>
void capture_element_transform(CaptureContext &ctx, TensorId c_id, std::size_t rank, std::string_view op_name,
                               std::optional<double> param) {
    // Looked up only to validate: an unknown name or a parameter the op does not
    // take fails here, at capture, rather than when the graph replays.
    (void)element_ops::global_element_op_registry().kernel<T>(op_name, param);
    LabeledSection("element_transform capture");

    ElementTransformDescriptor desc;
    desc.op_name = std::string(op_name);
    desc.param   = param;

    // Both lists name C: the transform reads every element and writes it back,
    // which is the read-modify-write convention scale and the closure overload
    // already use.
    ctx.record_built(OpKind::ElementTransform, "element_transform", packed_gemm::get_scalar_type<T>(), rank, std::move(desc),
                     std::span<TensorId const>{&c_id, 1}, std::span<TensorId const>{&c_id, 1}, {c_id}, {c_id});
}

#define EINSUMS_ELEMENTWISE_OPERATIONS(T)                                                                                                  \
    template EINSUMS_EXPORT void eager_scale<T>(T, Impl<T> &);                                                                             \
    template EINSUMS_EXPORT void capture_scale<T>(CaptureContext &, T, TensorId, std::string_view, std::size_t);                           \
    template EINSUMS_EXPORT void eager_axpy<T>(T, Impl<T> const &, Impl<T> &);                                                             \
    template EINSUMS_EXPORT void capture_axpy<T>(CaptureContext &, T, TensorId, TensorId, std::string_view, std::string_view,              \
                                                 std::size_t);                                                                             \
    template EINSUMS_EXPORT void eager_axpby<T>(T, Impl<T> const &, T, Impl<T> &);                                                         \
    template EINSUMS_EXPORT void capture_axpby<T>(CaptureContext &, T, T, TensorId, TensorId, std::size_t);                                \
    template EINSUMS_EXPORT void eager_direct_product<T>(T, Impl<T> const &, Impl<T> const &, T, Impl<T> &);                               \
    template EINSUMS_EXPORT void eager_direct_division<T>(T, Impl<T> const &, Impl<T> const &, T, Impl<T> &);                              \
    template EINSUMS_EXPORT void eager_element_transform<T>(Impl<T> &, std::string_view, std::optional<double>);                           \
    template EINSUMS_EXPORT void capture_element_transform<T>(CaptureContext &, TensorId, std::size_t, std::string_view,                   \
                                                              std::optional<double>);

EINSUMS_ELEMENTWISE_OPERATIONS(float)
EINSUMS_ELEMENTWISE_OPERATIONS(double)
EINSUMS_ELEMENTWISE_OPERATIONS(std::complex<float>)
EINSUMS_ELEMENTWISE_OPERATIONS(std::complex<double>)
#undef EINSUMS_ELEMENTWISE_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
