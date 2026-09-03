//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/LaplaceQuadrature.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// @ref build_executor's own diagnostic wording for @p kind.
std::string scalar_context(OpKind kind) {
    return fmt::format("build_executor({})", op_kind_name(kind));
}

/// Resolve one operand under @ref build_executor's own diagnostic wording.
OperandAccessor resolve_operand(Graph &graph, TensorId id, OpKind kind, char const *role) {
    return resolve_operand(graph, id, scalar_context(kind), role);
}

// ── Per-kind builders ───────────────────────────────────────────────────────
//
// Every one of these takes the RANK-ERASED route: the kernel is the
// ``TensorImpl<T>`` overload the typed entry points reduce to, so the built
// executor runs the same code the capture-baked lambda ran and one dtype
// dispatch covers every rank. See ExecutorBuilder.hpp for why `rank` is
// nonetheless part of the key.

/// ``A *= factor``. Rank-erased; the kernel is ``linear_algebra::detail::scale``.
std::function<void()> build_scale(packed_gemm::ScalarType dtype, ScaleDescriptor const &desc, OperandAccessor const &a) {
    auto params = live_or_private_params(desc.params, desc.factor, PrefactorScalar{double{0}});
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        return [params, a]() {
            LabeledSection("scale execute");
            linear_algebra::detail::scale(as<T>(params->alpha), a.impl<T>());
        };
    });
}

/// ``C = beta*C + alpha*permute(A)``. Rank-erased; the kernel is
/// @ref dispatch::string_permute_impl, which is the same body the tensor-object
/// overload forwards to, so the eager path and this one cannot diverge.
std::function<void()> build_permute(packed_gemm::ScalarType dtype, PermuteDescriptor const &desc, OperandAccessor const &a,
                                    OperandAccessor const &c) {
    auto params = live_or_private_params(desc.params, PrefactorScalar{desc.alpha}, PrefactorScalar{desc.beta});

    // Built once. The index lists are a structural field: a pass that rewrites
    // them rebuilds the executor rather than mutating it under a replay.
    ParsedPermuteSpec parsed;
    parsed.c_indices = desc.c_indices;
    parsed.a_indices = desc.a_indices;
    parsed.raw       = parsed.render();

    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        return [params, parsed, a, c]() {
            LabeledSection("permute execute");
            dispatch::string_permute_impl<T>(parsed, as<T>(params->beta), c.impl<T>(), as<T>(params->alpha), *a.impl<T>());
        };
    });
}

/// ``C = A^T``. Rank-erased; the kernel is the same ``tensor_algebra::detail::permute``
/// the typed ``tensor_algebra::transpose`` reduces to, reached with the index
/// letters that spell a rank-2 axis swap.
///
/// A transpose carries NO descriptor, deliberately. It is a fixed permutation
/// with no scalars, so (kind, dtype, rank, operand ids) is its complete
/// content; an empty descriptor alternative would record nothing and reusing
/// @ref PermuteDescriptor would invent index letters capture never had, which
/// PermuteFusion and CSE would then reason about as a user-written permute.
std::function<void()> build_transpose(packed_gemm::ScalarType dtype, OperandAccessor const &a, OperandAccessor const &c) {
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        return [a, c]() {
            LabeledSection("transpose execute");
            auto const *source = a.impl<T>();
            auto       *target = c.impl<T>();
            // The dimension guard the typed entry point applies. Kept because a
            // rebind validates dims against the slot, not against the operand
            // this node is paired with.
            if (target->dim(0) < source->dim(1) || target->dim(1) < source->dim(0)) {
                EINSUMS_THROW_EXCEPTION(DimensionError, "transpose: the output tensor is smaller than the transposed input");
            }
            static std::string const c_indices = "ab";
            static std::string const a_indices = "ba";
            tensor_algebra::detail::permute<false, T>(T{0}, c_indices, target, T{1}, a_indices, *source);
        };
    });
}

/// ``Y = alpha*X + beta*Y``. Rank-erased; the kernels are
/// ``linear_algebra::detail::axpy`` / ``axpby``.
///
/// The ``beta == 1`` arm is the BLAS axpy fast path, matching the capture-time
/// axpy executor and ``Graph::make_axpby_executor``: a pass may rewrite beta
/// away from 1, so the choice belongs to the executor rather than to the kind.
std::function<void()> build_axpby(packed_gemm::ScalarType dtype, AxpbyDescriptor const &desc, OperandAccessor const &x,
                                  OperandAccessor const &y) {
    auto params = live_or_private_params(desc.params, desc.alpha, desc.beta);
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        return [params, x, y]() {
            LabeledSection("axpby execute");
            auto const  alpha  = as<T>(params->alpha);
            auto const  beta   = as<T>(params->beta);
            auto const *source = x.impl<T>();
            auto       *target = y.impl<T>();
            if (beta == T{1}) {
                linear_algebra::detail::axpy(alpha, *source, target);
            } else {
                linear_algebra::detail::axpby(alpha, *source, beta, target);
            }
        };
    });
}

/// ``C = alpha*(A*B) + beta*C`` and ``C = alpha*(A/B) + beta*C``, element-wise.
/// Rank-erased; the kernels are ``linear_algebra::detail::direct_product`` /
/// ``direct_division``. One builder for the two kinds, matching the one
/// descriptor they share.
std::function<void()> build_elementwise_binary(OpKind kind, packed_gemm::ScalarType dtype, ElementwiseBinaryDescriptor const &desc,
                                               OperandAccessor const &a, OperandAccessor const &b, OperandAccessor const &c) {
    auto       params = live_or_private_params(desc.params, desc.alpha, desc.beta);
    bool const divide = kind == OpKind::DirectDivision;
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        if (divide) {
            return [params, a, b, c]() {
                LabeledSection("direct_division execute");
                auto *out = c.impl<T>();
                ProfileAnnotate("size", static_cast<std::int64_t>(out->size()));
                linear_algebra::detail::direct_division<T, T, T>(as<T>(params->alpha), *a.impl<T>(), *b.impl<T>(), as<T>(params->beta),
                                                                 out);
            };
        }
        return [params, a, b, c]() {
            LabeledSection("direct_product execute");
            auto *out = c.impl<T>();
            ProfileAnnotate("size", static_cast<std::int64_t>(out->size()));
            linear_algebra::detail::direct_product<T, T, T>(as<T>(params->alpha), *a.impl<T>(), *b.impl<T>(), as<T>(params->beta), out);
        };
    });
}

/**
 * @brief ``C = c_pf*C + ab_pf*contract(A, B)``, at any rank.
 *
 * Rank-erased through @ref RuntimeTensorView, which is the rank-erased ladder
 * @ref dispatch::string_einsum already carries for runtime-rank operands: one
 * dtype dispatch reaches every BLAS fast path, PackedGemm and the generic loop,
 * where a static-rank route would need an instantiation per (rank_a, rank_b,
 * rank_c) triple of the whole cascade.
 *
 * The three views are built HERE, once, and only re-seated on each call.
 * ``TensorImpl``'s copy assignment reuses the ``std::vector`` capacity its dims
 * and strides already hold, so a replay allocates nothing; constructing the
 * views per call - which is what the pass-built executor did - is six small
 * allocations a tiled expansion pays thousands of times.
 *
 * A node never runs concurrently with itself, which is what makes the scratch
 * views safe to hold; it is the same invariant the strided-batched fast path's
 * pointer-table memo relies on.
 */
std::function<void()> build_einsum(packed_gemm::ScalarType dtype, EinsumDescriptor const &desc, OperandAccessor const &a,
                                   OperandAccessor const &b, OperandAccessor const &c) {
    // Live handles when the node has them (capture, and any node a pass built
    // through Graph::make_einsum_node), private ones seeded from the snapshot
    // scalars when it does not (the loader case). A private block still runs
    // correctly; it simply is not shared with anything that could rewrite it.
    std::shared_ptr<EinsumParams> params = desc.params;
    if (params == nullptr) {
        params         = std::make_shared<EinsumParams>();
        params->c_pf   = desc.c_prefactor;
        params->ab_pf  = desc.ab_prefactor;
        params->conj_a = desc.conj_a;
        params->conj_b = desc.conj_b;
    }

    std::shared_ptr<EinsumIndices> indices = desc.indices;
    if (indices == nullptr) {
        indices                 = std::make_shared<EinsumIndices>();
        indices->spec.a_indices = desc.spec.a_indices;
        indices->spec.b_indices = desc.spec.b_indices;
        indices->spec.c_indices = desc.spec.c_indices;
        indices->spec.raw       = indices->spec.render();
        indices->link_indices   = desc.spec.link_indices;
    }

    // The packed-GEMM memo. One per node, so per-tile nodes from a tiled
    // expansion never share one and a parallel executor needs no
    // synchronization around it.
    std::shared_ptr<packed_gemm::ContractionSite> site = desc.site;
    if (site == nullptr) {
        site = std::make_shared<packed_gemm::ContractionSite>();
    }

    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        // Scratch views, re-seated per call. Shared rather than captured by
        // value so the executor stays copyable without copying the geometry.
        //
        // Constructed, never assigned: RuntimeTensorView's assignment operators
        // copy the ELEMENTS through the view, which is the opposite of what is
        // wanted here. Re-seating goes through the impl below, whose assignment
        // really is "take these dims, strides and pointer".
        using Impl = ::einsums::detail::TensorImpl<T>;
        struct Views {
            Views(Impl const &ai, Impl const &bi, Impl const &ci) : a(ai), b(bi), c(ci) {}
            RuntimeTensorView<T> a, b, c;
        };
        auto views = std::make_shared<Views>(*a.impl<T>(), *b.impl<T>(), *c.impl<T>());

        return [params, indices, site, views, a, b, c]() {
            LabeledSection("einsum execute");
            // Re-seat each view on its operand's LIVE impl: aliasing, so writes
            // to C land in the real tensor, and current, so rebind(),
            // Materialization and the MemoryPlanning arena are all honored.
            views->a.impl() = *a.impl<T>();
            views->b.impl() = *b.impl<T>();
            views->c.impl() = *c.impl<T>();
            ProfileAnnotate("a_size", static_cast<std::int64_t>(views->a.size()));
            ProfileAnnotate("b_size", static_cast<std::int64_t>(views->b.size()));
            ProfileAnnotate("c_size", static_cast<std::int64_t>(views->c.size()));
            // The spec goes BY REFERENCE, so an index rewrite (PermuteFusion)
            // is honored without rebuilding it: the pass writes into this very
            // object. link_indices was computed once, so a replay spares three
            // set builds.
            dispatch::string_einsum(indices->spec, as<T>(params->c_pf), &views->c, as<T>(params->ab_pf), views->a, views->b, params->conj_a,
                                    params->conj_b, &indices->link_indices, site.get());
        };
    });
}

/**
 * @brief ``result = sum_i A_i * B_i``, or ``sum_i conj(A_i) * B_i`` for a dotc.
 *
 * Rank-erased; the kernels are ``linear_algebra::detail::dot`` and
 * ``true_dot``, which are exactly what the typed entry points reduce to.
 *
 * The serial fence is not an optimization detail, it is the operation's
 * definition. A reduction's summation order is its thread count's, so an
 * unfenced dot is a function of the machine as well as of the operands, and
 * every capture site this builder replaces held the same scope for the same
 * reason. Dropping it here would make a replay stop matching its own capture
 * on a machine whose BLAS width changed between them.
 */
std::function<void()> build_dot(packed_gemm::ScalarType dtype, DotDescriptor const &desc, OperandAccessor const &a,
                                OperandAccessor const &b, ScalarAccessor const &result) {
    bool const conjugated = desc.conjugated;
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        if (conjugated) {
            return [a, b, result]() {
                LabeledSection("dotc execute");
                blas::SerialVendorScope const serial;
                *result.address<T>() = linear_algebra::detail::true_dot(*a.impl<T>(), *b.impl<T>());
            };
        }
        return [a, b, result]() {
            LabeledSection("dot execute");
            blas::SerialVendorScope const serial;
            *result.address<T>() = linear_algebra::detail::dot(*a.impl<T>(), *b.impl<T>());
        };
    });
}

/**
 * @brief ``result = sum_i A(i, i)``, over a square rank-2 operand.
 *
 * Rank-erased through ``TensorImpl::subscript``, which is what the typed
 * ``A(i, i)`` of the capture sites resolves to. The sum is sequential and in
 * index order, as it was there: a trace has no BLAS kernel and no fence to
 * take, so identity here is a matter of not reordering the additions.
 *
 * The square check travels with the executor rather than being settled at
 * capture, because a rebind validates dims against the SLOT and can seat a
 * differently shaped tensor under this node between replays.
 */
std::function<void()> build_trace(packed_gemm::ScalarType dtype, OperandAccessor const &a, ScalarAccessor const &result) {
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        return [a, result]() {
            LabeledSection("trace execute");
            auto const *source = a.impl<T>();
            if (source->rank() != 2) {
                EINSUMS_THROW_EXCEPTION(RankError, "cg::trace: input must be rank-2; got rank {}.", source->rank());
            }
            if (source->dim(0) != source->dim(1)) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
            }
            T sum{};
            for (size_t i = 0; i < source->dim(0); ++i) {
                sum += source->subscript(i, i);
            }
            *result.address<T>() = sum;
        };
    });
}

/// Narrow one arithmetic scalar at @p source to the @c std::int64_t a
/// @ref ParamTable holds, decoding the storage by @p type.
std::int64_t read_param_source(ParamSourceType type, void const *source) {
    auto const load = [source]<typename T>(T /*tag*/) { return static_cast<std::int64_t>(*static_cast<T const *>(source)); };
    switch (type) {
    case ParamSourceType::Bool:
        return load(bool{});
    case ParamSourceType::Char:
        return load(char{});
    case ParamSourceType::SChar:
        return load(static_cast<signed char>(0));
    case ParamSourceType::UChar:
        return load(static_cast<unsigned char>(0));
    case ParamSourceType::Short:
        return load(short{});
    case ParamSourceType::UShort:
        return load(static_cast<unsigned short>(0));
    case ParamSourceType::Int:
        return load(int{});
    case ParamSourceType::UInt:
        return load(static_cast<unsigned int>(0));
    case ParamSourceType::Long:
        return load(long{});
    case ParamSourceType::ULong:
        return load(static_cast<unsigned long>(0));
    case ParamSourceType::LongLong:
        return load(static_cast<long long>(0));
    case ParamSourceType::ULongLong:
        return load(static_cast<unsigned long long>(0));
    case ParamSourceType::Float:
        return load(float{});
    case ParamSourceType::Double:
        return load(double{});
    case ParamSourceType::LongDouble:
        return load(static_cast<long double>(0));
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor(WriteParam): unknown ParamSourceType {}", static_cast<int>(type));
}

/**
 * @brief ``C = op(C)``, element-wise, with the kernel looked up by NAME.
 *
 * Rank-erased through @ref element_ops::detail::apply_element_op, which walks
 * the destination's live @ref einsums::detail::TensorImpl and so covers every
 * rank and every layout with one dtype dispatch.
 *
 * The registry is consulted ONCE, here. That is the whole point of the name
 * being in the descriptor: an op this process has not registered is refused at
 * BUILD time -- which is capture time today and load time when a loader exists
 * -- with the op named, instead of failing somewhere in the middle of a replay.
 *
 * A parameterized op's policy number is bound here too, from the descriptor or,
 * where the node carries none, from the default its registration documents. So
 * the built kernel is a plain unary map either way and the replay loop is the
 * same one an unparameterized op runs.
 */
std::function<void()> build_element_transform(packed_gemm::ScalarType dtype, ElementTransformDescriptor const &desc,
                                              OperandAccessor const &c) {
    if (desc.op_name.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor(ElementTransform): the descriptor names no element op");
    }
    auto const &registry = element_ops::global_element_op_registry();
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        auto kernel = registry.kernel<T>(desc.op_name, desc.param);
        return [kernel = std::move(kernel), c]() {
            LabeledSection("element_transform execute");
            element_ops::detail::apply_element_op<T>(kernel, c.impl<T>());
        };
    });
}

/**
 * @brief ``if (predicate) then_branch else else_branch``.
 *
 * The predicate is @ref PredExpr data rather than a closure the capture site
 * baked, so a pass that rewrites it -- turning a callback into a flag test, say,
 * which is what the design's gate-flag materialization does at load -- changes
 * what the next replay evaluates.
 *
 * The @ref ParamTable is held by ``shared_ptr``, matching what the WriteParam
 * builder does and for the same reason: reading ``Graph::params_ptr()`` per call
 * would need a graph pointer that a graph move invalidates.
 */
std::function<void()> build_conditional(ConditionalDescriptor const &desc, std::shared_ptr<ParamTable> params) {
    if (!desc.then_branch) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor(Conditional): the descriptor carries no then branch");
    }
    return [predicate = desc.predicate, then_branch = desc.then_branch, else_branch = desc.else_branch, params = std::move(params)]() {
        if (predicate.resolve(params.get())) {
            then_branch->execute();
        } else if (else_branch && else_branch->num_nodes() > 0) {
            else_branch->execute();
        }
    };
}

/**
 * @brief ``do { body } while (condition)``, bounded by ``max_iterations``.
 *
 * The condition is evaluated AFTER each iteration, so the body always runs at
 * least once, and it sees the index of the iteration that just finished. A
 * default @ref PredExpr is an unconditional true, which reproduces exactly what
 * an absent ``std::function`` condition always did: run to the safety limit.
 *
 * The iteration count is written through the descriptor's shared
 * @ref LoopState. It used to be written into the executor lambda's own COPY of
 * the descriptor, so the count on the node stayed 0 forever and nothing could
 * observe how many iterations a loop actually ran.
 */
std::function<void()> build_loop(LoopDescriptor const &desc, std::shared_ptr<ParamTable> params) {
    if (!desc.body) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor(Loop): the descriptor carries no body");
    }
    // A node a pass assembled by hand carries no live state; a private block
    // keeps it running, and it simply reports no iteration count. Callers that
    // want the count populate `state` before building, as capture does.
    std::shared_ptr<LoopState> state = desc.state != nullptr ? desc.state : std::make_shared<LoopState>();
    return [condition = desc.condition, body = desc.body, max_iterations = desc.max_iterations, state = std::move(state),
            params = std::move(params)]() {
        state->last_iteration_count = 0;
        for (size_t iter = 0; iter < max_iterations; iter++) {
            body->execute();
            state->last_iteration_count = iter + 1;
            if (!condition.resolve(params.get(), iter)) {
                break;
            }
        }
    };
}

/**
 * @brief The body, once per bound problem, and a boolean on every replay after that.
 *
 * The whole of what makes a setup node different from a loop node with one iteration is
 * here: the two guards below. A replay that finds the body already computed returns without
 * touching it, which is what "executes once per bind" means once it is expressed as a node
 * in the ordinary schedule rather than as a phase beside it.
 *
 * The key guard is the cache of Part 6.3. @ref Graph::invalidate_setup clears
 * @ref SetupState::computed on every bind, because a bind is the only event that can change
 * what the body would produce; a caller that KNOWS the new bind is the same problem says so
 * with @ref Graph::set_setup_key, and the two keys agreeing is what lets the body be skipped
 * anyway. Nothing here validates the claim, which is the trade the design records: a
 * host-provided key is cheap and trusts the host, and the alternative is hashing every bound
 * input on every bind.
 */
std::function<void()> build_setup(SetupDescriptor const &desc) {
    if (!desc.body) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor(Setup): the descriptor carries no body");
    }
    // As build_loop does: a node a pass assembled by hand carries no live state, and a
    // private block keeps it running rather than making the omission a crash.
    std::shared_ptr<SetupState> state = desc.state != nullptr ? desc.state : std::make_shared<SetupState>();
    return [body = desc.body, state = std::move(state)]() {
        if (state->computed) {
            return;
        }
        if (!state->pending_key.empty() && state->pending_key == state->computed_key) {
            state->computed = true;
            return;
        }
        body->execute();
        state->computed     = true;
        state->computed_key = state->pending_key;
    };
}

/**
 * @brief The Laplace quadrature, refitted from whatever energies are bound now.
 *
 * Everything the node needs is either in the descriptor or in the operands, which is what
 * makes the kind saveable: the rule is a closed-form function of the spectral range, so a
 * loaded graph rebuilds exactly the rule the capture would have.
 *
 * The exponentials are written rather than the points alone because the points are useless on
 * their own. What the rewrite consumes is one matrix per axis, and forming those from the
 * points would need an outer product and an elementwise exponential per axis, which is four
 * more nodes per axis to compute what this already has in registers.
 */
std::function<void()> build_laplace_quadrature(packed_gemm::ScalarType dtype, LaplaceQuadratureDescriptor const &desc, Graph &graph,
                                               std::span<TensorId const> inputs, std::span<TensorId const> outputs) {
    std::size_t const axes = inputs.size();
    if (axes == 0 || desc.signs.size() != axes || outputs.size() != axes + 3) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "build_executor(LaplaceQuadrature): {} energy vector(s), {} sign(s) and {} output(s); the node wants one "
                                "energy vector and one sign per axis, and points, weights, the measured error and one exponential "
                                "matrix per axis as outputs",
                                axes, desc.signs.size(), outputs.size());
    }
    if (desc.points < 2) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "build_executor(LaplaceQuadrature): the descriptor asks for {} point(s); a "
                                "trapezoidal rule needs at least two",
                                desc.points);
    }

    std::vector<OperandAccessor> energies;
    std::vector<OperandAccessor> exponentials;
    energies.reserve(axes);
    exponentials.reserve(axes);
    for (std::size_t axis = 0; axis < axes; ++axis) {
        energies.push_back(resolve_operand(graph, inputs[axis], OpKind::LaplaceQuadrature, "energy"));
        exponentials.push_back(resolve_operand(graph, outputs[axis + 3], OpKind::LaplaceQuadrature, "exponentials"));
    }
    OperandAccessor const points  = resolve_operand(graph, outputs[0], OpKind::LaplaceQuadrature, "points");
    OperandAccessor const weights = resolve_operand(graph, outputs[1], OpKind::LaplaceQuadrature, "weights");
    OperandAccessor const error   = resolve_operand(graph, outputs[2], OpKind::LaplaceQuadrature, "error");

    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        if constexpr (!std::is_floating_point_v<T>) {
            // A complex denominator has no spectral range to bound and no exponential integral
            // representing its reciprocal. The refusal is at the BUILD rather than in the loop
            // so that a file describing one is rejected when it is read.
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "build_executor(LaplaceQuadrature): a complex element type reached the real-only quadrature");
        } else {
            return [desc, energies, exponentials, points, weights, error]() {
                LabeledSection("laplace_quadrature execute");

                std::vector<double> low;
                std::vector<double> high;
                std::vector<int>    signs;
                low.reserve(energies.size());
                high.reserve(energies.size());
                signs.reserve(energies.size());
                for (std::size_t axis = 0; axis < energies.size(); ++axis) {
                    auto const *impl = energies[axis].impl<T>();
                    if (impl == nullptr || impl->rank() != 1 || impl->dim(0) == 0) {
                        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                                "laplace_quadrature: axis {}'s energy operand is not a non-empty vector", axis);
                    }
                    T const          *data     = impl->data();
                    std::size_t const inc      = impl->stride(0);
                    T                 smallest = data[0];
                    T                 largest  = data[0];
                    for (std::size_t i = 1; i < impl->dim(0); ++i) {
                        smallest = std::min(smallest, data[i * inc]);
                        largest  = std::max(largest, data[i * inc]);
                    }
                    low.push_back(static_cast<double>(smallest));
                    high.push_back(static_cast<double>(largest));
                    signs.push_back(desc.signs[axis] >= 0 ? 1 : -1);
                }

                laplace::SpectralRange const  range = laplace::spectral_range(low, high, signs);
                laplace::QuadratureGrid const grid  = laplace::build_quadrature(range, desc.epsilon, static_cast<std::size_t>(desc.points));

                auto *point_impl  = points.impl<T>();
                auto *weight_impl = weights.impl<T>();
                auto *error_impl  = error.impl<T>();
                if (point_impl == nullptr || weight_impl == nullptr || error_impl == nullptr) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "laplace_quadrature: an output tensor has no storage");
                }
                if (point_impl->size() != grid.points.size() || weight_impl->size() != grid.points.size() || error_impl->size() < 1) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "laplace_quadrature: the rule has {} point(s) and the outputs hold {} and {}",
                                            grid.points.size(), point_impl->size(), weight_impl->size());
                }
                for (std::size_t j = 0; j < grid.points.size(); ++j) {
                    point_impl->data()[j * point_impl->stride(0)]   = static_cast<T>(grid.points[j]);
                    weight_impl->data()[j * weight_impl->stride(0)] = static_cast<T>(grid.weights[j]);
                }
                error_impl->data()[0] = static_cast<T>(laplace::measure_quadrature_error(grid));

                for (std::size_t axis = 0; axis < energies.size(); ++axis) {
                    auto const *energy = energies[axis].impl<T>();
                    auto       *target = exponentials[axis].impl<T>();
                    if (target == nullptr || target->rank() != 2 || target->dim(0) != grid.points.size() ||
                        target->dim(1) != energy->dim(0)) {
                        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "laplace_quadrature: axis {}'s exponential matrix is not [{} x {}]",
                                                axis, grid.points.size(), energy == nullptr ? 0 : energy->dim(0));
                    }
                    // The sign the exponent carries is the one the rule was built with. A
                    // uniformly negative denominator is represented by the rule for its
                    // negation, so every axis flips with it and the minus sign rides on the
                    // weights, which is what keeps the caller's own sign convention.
                    double const      sign = (desc.signs[axis] >= 0 ? 1.0 : -1.0) * (grid.negated ? -1.0 : 1.0);
                    T const          *e    = energy->data();
                    std::size_t const einc = energy->stride(0);
                    for (std::size_t j = 0; j < grid.points.size(); ++j) {
                        // Only the FIRST axis carries the weights, so the product over axes
                        // applies each weight exactly once.
                        double const scale = axis == 0 ? grid.weights[j] : 1.0;
                        for (std::size_t i = 0; i < energy->dim(0); ++i) {
                            double const exponent = -sign * grid.points[j] * static_cast<double>(e[i * einc]);
                            target->data()[j * target->stride(0) + i * target->stride(1)] = static_cast<T>(scale * std::exp(exponent));
                        }
                    }
                }
            };
        }
    });
}

/**
 * @brief ``params[name] = int64(source)``, the TENSOR arm of ``cg::write_param``.
 *
 * The source address is read on every call rather than closed over, which is
 * the whole conversion: the captured executor held a ``T &`` to the caller's
 * variable, so repointing the graph's handle for that scalar moved what every
 * analysis believed and nothing about what the write actually read.
 *
 * The @ref ParamTable is held by ``shared_ptr``, matching what capture did.
 * Reading ``Graph::params_ptr()`` per call instead would follow a later
 * ``set_params_ptr``, which is a behavior change rather than a conversion, and
 * the executor would then need a graph pointer that a graph move invalidates.
 */
std::function<void()> build_write_param(WriteParamDescriptor const &desc, std::shared_ptr<ParamTable> params,
                                        ScalarAccessor const &source) {
    return [name = desc.name, type = desc.source_type, params = std::move(params), source]() {
        LabeledSection("write_param execute");
        if (!params) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param executor: no ParamTable bound to graph");
        }
        // Any T decodes through the same address; the accessor's element type
        // only picks a pointer cast, and read_param_source performs the real one.
        params->set(name, read_param_source(type, source.address<char>()));
    };
}

/**
 * @brief ``params[name] = expr``, the EXPRESSION arm of ``cg::write_param``.
 *
 * The three @ref BoundExpr arms in one builder, because they are one operation
 * reading its value three ways: a literal, another parameter, or a callback.
 * Only the last blocks a save, and @ref reconstruction_blocker says which arm a
 * node holds. The callback case reproduces the capture-baked lambda this
 * replaces exactly -- same table, same ordering, same diagnostic when no table
 * is bound.
 */
std::function<void()> build_write_param_expr(std::string name, BoundExpr expr, std::shared_ptr<ParamTable> params) {
    return [name = std::move(name), expr = std::move(expr), params = std::move(params)]() {
        LabeledSection("write_param execute");
        if (!params) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param executor: no ParamTable bound to graph");
        }
        params->set(name, expr.resolve(*params));
    };
}

/**
 * @brief ``C = alpha*op(A)*op(B) + beta*C``.
 *
 * Rank-erased; the kernel is ``linear_algebra::detail::gemm`` at the
 * ``TensorImpl`` level, which is where all three ``cg::gemm`` capture overloads
 * already landed. See the dispatch-routes note in ExecutorBuilder.hpp for why
 * the higher, symmetry-aware entry is not the one to reach for.
 *
 * The prefactors are read from the descriptor's snapshots, not from a live
 * params block, because no pass rewrites a gemm's prefactors; @ref
 * GemmDescriptor states the rule and what changes when one does.
 */
std::function<void()> build_gemm(packed_gemm::ScalarType dtype, GemmDescriptor const &desc, OperandAccessor const &a,
                                 OperandAccessor const &b, OperandAccessor const &c) {
    char const ta = desc.trans_a;
    char const tb = desc.trans_b;
    // The two annotation strings the capture-baked executors emitted, chosen
    // once here rather than per call.
    bool const  transposed_a = std::tolower(static_cast<unsigned char>(ta)) != 'n';
    bool const  transposed_b = std::tolower(static_cast<unsigned char>(tb)) != 'n';
    char const *trans_label  = transposed_a ? (transposed_b ? "TT" : "TN") : (transposed_b ? "NT" : "NN");

    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        auto const alpha = as<T>(desc.alpha);
        auto const beta  = as<T>(desc.beta);
        return [alpha, beta, ta, tb, transposed_a, trans_label, a, b, c]() {
            LabeledSection("gemm execute");
            auto const *a_impl = a.impl<T>();
            auto       *c_impl = c.impl<T>();
            ProfileAnnotate("trans", trans_label);
            ProfileAnnotate("m", static_cast<std::int64_t>(c_impl->dim(0)));
            ProfileAnnotate("n", static_cast<std::int64_t>(c_impl->dim(1)));
            ProfileAnnotate("k", static_cast<std::int64_t>(transposed_a ? a_impl->dim(0) : a_impl->dim(1)));
            linear_algebra::detail::gemm(ta, tb, alpha, *a_impl, *b.impl<T>(), beta, c_impl);
        };
    });
}

/**
 * @brief ``A = V diag(W) V^T``, in place. Rank-erased; the kernel is
 *        ``linear_algebra::detail::syev``, which is what the typed and
 *        runtime-rank capture overloads both reduce to.
 *
 * Real ONLY, and refused here rather than inside the dtype dispatch. The capture
 * sites require a non-complex operand and record the complex case as
 * @ref OpKind::Heev, so a complex Syev node is a file describing an operation no
 * capture in this library can produce. Naming Heev is the only thing the reader
 * of that message can act on; letting the dispatch instantiate the complex arm
 * would reach a kernel whose eigenvalue output is a real vector the node's dtype
 * does not describe.
 *
 * ``compute_eigenvectors`` is read ONCE, at build time, because it selects
 * between two LAPACK jobs and is a structural field like a permute's index
 * lists: a pass that changes it rebuilds the executor rather than mutating it
 * under a replay.
 *
 * No vendor fence, deliberately. The capture-baked executors took none, and a
 * dot's argument for one does not carry: LAPACK's answer is not a summation
 * order the thread count reorders, and fencing here would make a rebuilt node
 * slower than the capture it replaces without making it more reproducible.
 */
std::function<void()> build_syev(packed_gemm::ScalarType dtype, SyevDescriptor const &desc, OperandAccessor const &a,
                                 OperandAccessor const &w) {
    if (dtype == packed_gemm::ScalarType::Complex64 || dtype == packed_gemm::ScalarType::Complex128) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "build_executor(Syev): a symmetric eigendecomposition is real, and this node names a complex operand. "
                                "The complex counterpart is Hermitian and records as OpKind::Heev, which has no builder entry yet");
    }

    bool const compute_eigenvectors = desc.compute_eigenvectors;
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::function<void()> {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            if (compute_eigenvectors) {
                return [a, w]() {
                    LabeledSection("syev execute");
                    auto *matrix = a.impl<T>();
                    ProfileAnnotate("n", static_cast<std::int64_t>(matrix->dim(0)));
                    linear_algebra::detail::syev<true>(matrix, w.impl<T>());
                };
            }
            return [a, w]() {
                LabeledSection("syev execute");
                auto *matrix = a.impl<T>();
                ProfileAnnotate("n", static_cast<std::int64_t>(matrix->dim(0)));
                linear_algebra::detail::syev<false>(matrix, w.impl<T>());
            };
        } else {
            // Unreachable: the guard above returned for both complex arms. The
            // branch exists because the dispatch instantiates all four.
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor(Syev): a complex dtype reached the real-only builder");
        }
    });
}

/// One name per @ref OpData alternative, in the variant's own declaration order.
///
/// Indexed by ``index()`` rather than matched by a chain of
/// ``holds_alternative``: the chain silently answered "descriptor alternative
/// #N" for every alternative nobody had added to it, which was most of them,
/// and adding one to the variant left it that way with nothing complaining. The
/// static_assert below is what complains now, at compile time.
constexpr auto kDescriptorNames = std::to_array<std::string_view>({
    "no descriptor",
    "EinsumDescriptor",
    "ScaleDescriptor",
    "PermuteDescriptor",
    "ConditionalDescriptor",
    "LoopDescriptor",
    "AllocDescriptor",
    "TransferDescriptor",
    "DiskIODescriptor",
    "CommDescriptor",
    "InitializeDescriptor",
    "BatchedGemmDescriptor",
    "GroupedBatchedGemmDescriptor",
    "ViewDescriptor",
    "WriteParamDescriptor",
    "AxpbyDescriptor",
    "GroupedDotDescriptor",
    "GroupedAxpbyDescriptor",
    "GroupedElementwiseDescriptor",
    "GroupedSandwichDescriptor",
    "GroupedGatherRotateDescriptor",
    "TiledEinsumDescriptor",
    "TiledElementwiseDescriptor",
    "TiledPermuteDescriptor",
    "TiledDotDescriptor",
    "ElementwiseBinaryDescriptor",
    "DotDescriptor",
    "TraceDescriptor",
    "GemmDescriptor",
    "ElementTransformDescriptor",
    "SetupDescriptor",
    "SyevDescriptor",
    "LaplaceQuadratureDescriptor",
});

static_assert(kDescriptorNames.size() == std::variant_size_v<OpData>,
              "OpData gained or lost an alternative: add or remove its name in kDescriptorNames, in the variant's own order");

/// Where @c D sits in @ref OpData's alternative list.
template <typename D, std::size_t Index = 0>
constexpr std::size_t alternative_index() {
    if constexpr (std::is_same_v<D, std::variant_alternative_t<Index, OpData>>) {
        return Index;
    } else {
        return alternative_index<D, Index + 1>();
    }
}

/// The table is POSITIONAL, so a name in the wrong slot names the wrong
/// descriptor and the size check above would not notice. These pin the slots the
/// refusal messages are written against, at compile time.
static_assert(kDescriptorNames[alternative_index<std::monostate>()] == "no descriptor");
static_assert(kDescriptorNames[alternative_index<EinsumDescriptor>()] == "EinsumDescriptor");
static_assert(kDescriptorNames[alternative_index<ScaleDescriptor>()] == "ScaleDescriptor");
static_assert(kDescriptorNames[alternative_index<AxpbyDescriptor>()] == "AxpbyDescriptor");
static_assert(kDescriptorNames[alternative_index<GemmDescriptor>()] == "GemmDescriptor");
static_assert(kDescriptorNames[alternative_index<SyevDescriptor>()] == "SyevDescriptor");
static_assert(kDescriptorNames[alternative_index<LaplaceQuadratureDescriptor>()] == "LaplaceQuadratureDescriptor");

/// The descriptor alternative @p data holds, for a diagnostic.
std::string_view descriptor_name(OpData const &data) {
    return kDescriptorNames[data.index()];
}

/// Complain that a kind with a builder entry carries the wrong descriptor.
[[noreturn]] void wrong_descriptor(OpKind kind, OpData const &data, char const *expected) {
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor({}): expected a {}, found {}", op_kind_name(kind), expected,
                            descriptor_name(data));
}

/// Complain that an operand list is shorter than the kind's convention needs.
[[noreturn]] void missing_operand(OpKind kind, char const *list, std::size_t needed, std::size_t have) {
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor({}): needs at least {} {}, node lists {}", op_kind_name(kind), needed,
                            list, have);
}

/// @p desc's @c D alternative, or a refusal naming what it holds instead.
///
/// Fourteen copies of the get_if-then-null-check idiom is what this replaces,
/// and the refusal is unchanged: @ref wrong_descriptor never returns, so the
/// reference handed back is always to a live alternative.
///
/// Callers bind the result to a named reference rather than calling it inside
/// the builder's argument list, and deliberately: argument evaluation order is
/// unspecified, so an inline call could report an unresolvable operand where
/// the hand-written checks reported the wrong descriptor.
template <typename D>
D const &expect(OpKind kind, OpData const &desc, char const *expected) {
    D const *held = std::get_if<D>(&desc);
    if (held == nullptr) {
        wrong_descriptor(kind, desc, expected);
    }
    return *held;
}

/// Refuse when @p operands holds fewer than @p needed entries.
///
/// @p list is the list's name as the message spells it, "inputs" or "outputs".
void need(OpKind kind, char const *list, std::span<TensorId const> operands, std::size_t needed) {
    if (operands.size() < needed) {
        missing_operand(kind, list, needed, operands.size());
    }
}

/**
 * @brief Whether @p impl's strides are monotone in its own declared storage order.
 *
 * A permute_view keeps the storage-order FLAG of its parent but presents
 * reordered strides, so ``is_row_major()``/``is_column_major()`` alone cannot
 * prove the canonical layout a GEMM assumes. Found by the large-rank
 * differential fuzzer: a view with the slice axes swapped passed the flag gate
 * and produced wrong results. Extent-1 axes are never traversed, so their
 * (possibly inflated) strides are ignored.
 */
template <typename T>
bool layout_matches_flag(::einsums::detail::TensorImpl<T> const &impl) {
    bool const   row_major = impl.is_row_major();
    size_t const rank      = impl.rank();
    size_t       prev      = 0;
    bool         first     = true;
    for (size_t n = 0; n < rank; ++n) {
        size_t const d = row_major ? rank - 1 - n : n;
        if (impl.dim(d) <= 1) {
            continue;
        }
        size_t const st = impl.stride(d);
        if (!first && st < prev) {
            return false;
        }
        prev  = st;
        first = false;
    }
    return true;
}

} // namespace

OperandAccessor try_resolve_operand(Graph &graph, TensorId id) {
    if (TensorSlot *slot = graph.find_slot(id); slot != nullptr && slot->ptr != nullptr && slot->impl_of != nullptr) {
        return OperandAccessor{slot};
    }
    if (TensorHandle const *handle = graph.find_tensor(id); handle != nullptr && handle->impl_fn) {
        return OperandAccessor{handle->impl_fn};
    }
    return OperandAccessor{};
}

OperandAccessor resolve_operand(Graph &graph, TensorId id, std::string_view context, char const *role) {
    OperandAccessor accessor = try_resolve_operand(graph, id);
    if (!accessor.valid()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "{}: operand {} (tensor {}) exposes no rank-erased geometry; a tile-wise sparse tensor has no single "
                                "buffer to build a dense executor over",
                                context, role, id);
    }
    return accessor;
}

ScalarAccessor resolve_scalar_operand(Graph &graph, TensorId id, std::string_view context, char const *role) {
    // The tensor route first, so a destination that has a slot follows rebind()
    // and redirect_slot(); a registered raw scalar has neither and falls
    // through to its handle.
    if (OperandAccessor const accessor = try_resolve_operand(graph, id); accessor.valid()) {
        return ScalarAccessor{accessor};
    }
    if (TensorHandle const *handle = graph.find_tensor(id); handle != nullptr && handle->tensor_ptr != nullptr) {
        return ScalarAccessor{handle};
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                            "{}: scalar operand {} (tensor {}) has neither rank-erased geometry nor a registered address, so there is "
                            "nowhere to write the result",
                            context, role, id);
}

std::shared_ptr<GemmHint> derive_gemm_hint(packed_gemm::ScalarType dtype, packed_gemm::ContractionSpec const &spec, Graph &graph,
                                           TensorId a_id, TensorId b_id, TensorId c_id) {
    // Index-shape gate first: it costs three size comparisons and rejects every
    // contraction that is not a plain matrix product, so nothing below has to
    // resolve an operand for a shape it will refuse anyway.
    if (spec.a_indices.size() != 2 || spec.b_indices.size() != 2 || spec.c_indices.size() != 2 || spec.link_indices.size() != 1) {
        return nullptr;
    }

    // Roles: C's first index must be the one A contributes, its second the one
    // B contributes. See the header note for why a wrong hint is worse than none.
    std::string const &link = spec.link_indices[0];
    auto const         free = [&link](std::vector<std::string> const &idx) { return idx[0] == link ? idx[1] : idx[0]; };
    if (spec.c_indices[0] != free(spec.a_indices) || spec.c_indices[1] != free(spec.b_indices)) {
        return nullptr;
    }

    OperandAccessor const a = try_resolve_operand(graph, a_id);
    OperandAccessor const b = try_resolve_operand(graph, b_id);
    OperandAccessor const c = try_resolve_operand(graph, c_id);
    if (!a.valid() || !b.valid() || !c.valid()) {
        return nullptr;
    }

    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> std::shared_ptr<GemmHint> {
        auto const *a_impl = a.impl<T>();
        auto const *b_impl = b.impl<T>();
        auto const *c_impl = c.impl<T>();
        if (a_impl == nullptr || b_impl == nullptr || c_impl == nullptr) {
            return nullptr;
        }
        if (a_impl->rank() != 2 || b_impl->rank() != 2 || c_impl->rank() != 2) {
            return nullptr;
        }
        if (!layout_matches_flag(*a_impl) || !layout_matches_flag(*b_impl) || !layout_matches_flag(*c_impl)) {
            return nullptr;
        }

        auto hint     = std::make_shared<GemmHint>();
        hint->scalar  = blas_scalar_of<T>();
        hint->trans_a = (spec.a_indices[0] == link) ? 'T' : 'N';
        hint->trans_b = (spec.b_indices[1] == link) ? 'T' : 'N';
        hint->m       = static_cast<int>(c_impl->dim(0));
        hint->n       = static_cast<int>(c_impl->dim(1));
        hint->k       = static_cast<int>(hint->trans_a == 'N' ? a_impl->dim(1) : a_impl->dim(0));
        hint->a       = GemmOperand{.id = a_id, .leading_dim = static_cast<int>(a_impl->get_lda())};
        hint->b       = GemmOperand{.id = b_id, .leading_dim = static_cast<int>(b_impl->get_lda())};
        hint->c       = GemmOperand{.id = c_id, .leading_dim = static_cast<int>(c_impl->get_lda())};
        return hint;
    });
}

std::string reconstruction_blocker(Node const &node) {
    if (!is_reconstructible(node.kind)) {
        // The batched family is held back for a reason worth stating, because "not yet" reads
        // as an oversight and this one is a decision. A grouped batch partitions its members
        // by SHAPE at capture and reorders the operand lists to match, so its descriptor is a
        // function of one problem's extents; saving it would freeze that problem into the
        // file, which is exactly what the structure/tuning split forbids (batching
        // is never saved, and a grouping chosen on one machine is wrong on another). Making it
        // reconstructible therefore means saving the ALGEBRAIC form and re-grouping on load,
        // not teaching the descriptor to serialize itself.
        switch (node.kind) {
        case OpKind::BatchedGemm:
        case OpKind::GroupedBatchedGemm:
            return "batched forms are a resource decision and are deliberately not saved; capture the "
                   "per-member operations and let the resource phase group them";
        default:
            break;
        }
        return "kind not yet reconstructible";
    }
    switch (node.kind) {
    case OpKind::Scale:
        return std::holds_alternative<ScaleDescriptor>(node.op_data)
                   ? std::string{}
                   : fmt::format("Scale: expected a ScaleDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::Permute:
        return std::holds_alternative<PermuteDescriptor>(node.op_data)
                   ? std::string{}
                   : fmt::format("Permute: expected a PermuteDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::Transpose:
        // A transpose is fully described by its kind, dtype, rank and operands;
        // anything else on op_data means the node is not the transpose it claims.
        return std::holds_alternative<std::monostate>(node.op_data)
                   ? std::string{}
                   : fmt::format("Transpose: expected no descriptor, found {}", descriptor_name(node.op_data));
    case OpKind::Axpby:
        return std::holds_alternative<AxpbyDescriptor>(node.op_data)
                   ? std::string{}
                   : fmt::format("Axpby: expected an AxpbyDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
        if (std::holds_alternative<ElementwiseBinaryDescriptor>(node.op_data)) {
            return {};
        }
        if (std::holds_alternative<TiledElementwiseDescriptor>(node.op_data)) {
            return fmt::format("{}: tiled variant, whose per-tile kernel has no builder entry", op_kind_name(node.kind));
        }
        return fmt::format("{}: expected an ElementwiseBinaryDescriptor, found {}", op_kind_name(node.kind), descriptor_name(node.op_data));
    case OpKind::Einsum:
        return std::holds_alternative<EinsumDescriptor>(node.op_data)
                   ? std::string{}
                   : fmt::format("Einsum: expected an EinsumDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::Dot:
        if (std::holds_alternative<DotDescriptor>(node.op_data)) {
            return {};
        }
        if (std::holds_alternative<TiledDotDescriptor>(node.op_data)) {
            return "Dot: tiled variant, whose per-tile reduction has no builder entry";
        }
        return fmt::format("Dot: expected a DotDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::Trace:
        // The empty TraceDescriptor records nothing and is required anyway: it
        // is what separates the dense trace from the tiled one, which shares
        // the kind and reduces over a grid. See TraceDescriptor.
        return std::holds_alternative<TraceDescriptor>(node.op_data)
                   ? std::string{}
                   : fmt::format("Trace: expected a TraceDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::WriteParam: {
        auto const *desc = std::get_if<WriteParamDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return fmt::format("WriteParam: expected a WriteParamDescriptor, found {}", descriptor_name(node.op_data));
        }
        // The kind's bit is true and this ARM is still not saveable. See the
        // note on is_reconstructible: a callback is one of the four closures a
        // node can carry, and only the Const and Param arms of the
        // BoundExpr that replaced it are content a file can carry.
        if (desc->source_expr.has_value() && desc->source_expr->is_callback()) {
            return "WriteParam: callback arm, whose source is a std::function a saved graph cannot hold";
        }
        return {};
    }
    case OpKind::Gemm:
        return std::holds_alternative<GemmDescriptor>(node.op_data)
                   ? std::string{}
                   : fmt::format("Gemm: expected a GemmDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::ElementTransform:
        // The lambda-taking capture overloads record no descriptor, which is
        // not an oversight: a closure has no name a file could carry. The
        // message names the fix rather than the field, because "the field is
        // missing" is not something a user can act on.
        if (std::holds_alternative<ElementTransformDescriptor>(node.op_data)) {
            return {};
        }
        if (std::holds_alternative<std::monostate>(node.op_data)) {
            return "ElementTransform: the kernel is an anonymous closure; register a named op with element_ops::register_op and "
                   "capture with cg::element_transform(C, name)";
        }
        return fmt::format("ElementTransform: expected an ElementTransformDescriptor, found {}", descriptor_name(node.op_data));
    case OpKind::Conditional: {
        auto const *desc = std::get_if<ConditionalDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return fmt::format("Conditional: expected a ConditionalDescriptor, found {}", descriptor_name(node.op_data));
        }
        // Holding a SUBGRAPH is deliberately not a blocker. Whether that
        // subgraph can be written is a question about the subgraph, and
        // reporting it here would name the wrong node.
        if (desc->predicate.names_a_closure()) {
            return "Conditional: callback predicate, which is a std::function a saved graph cannot hold; a comparison over "
                   "parameters or a gate flag is the data-shaped spelling";
        }
        return {};
    }
    case OpKind::Loop: {
        auto const *desc = std::get_if<LoopDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return fmt::format("Loop: expected a LoopDescriptor, found {}", descriptor_name(node.op_data));
        }
        if (desc->condition.names_a_closure()) {
            return "Loop: callback condition, which is a std::function a saved graph cannot hold; a comparison over parameters or "
                   "over the iteration index is the data-shaped spelling";
        }
        return {};
    }
    case OpKind::Syev:
        // The kind covers a dense decomposition and a tiled one, and the tiled capture site
        // records NO descriptor, so "no descriptor" here is what a per-block syev looks like
        // rather than a field someone forgot to fill in. The message says both, because
        // "expected a SyevDescriptor" alone sends the reader hunting for a writer bug.
        return std::holds_alternative<SyevDescriptor>(node.op_data)
                   ? std::string{}
                   : fmt::format("Syev: expected a SyevDescriptor, found {}; the tiled variant records none, and its per-block "
                                 "decomposition has no builder entry",
                                 descriptor_name(node.op_data));
    case OpKind::LaplaceQuadrature: {
        auto const *desc = std::get_if<LaplaceQuadratureDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return fmt::format("LaplaceQuadrature: expected a LaplaceQuadratureDescriptor, found {}", descriptor_name(node.op_data));
        }
        // The whole node is the descriptor plus its operands, and every field of it is a
        // number. That is the point of the kind existing: the rule it refits is a closed-form
        // function of the bound energies, so nothing about it has to be remembered by the
        // process that reads the file.
        return {};
    }
    case OpKind::Setup: {
        auto const *desc = std::get_if<SetupDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return fmt::format("Setup: expected a SetupDescriptor, found {}", descriptor_name(node.op_data));
        }
        // Nothing else to check. A setup node's only data is its body, and the body's own
        // nodes are reported separately by the recursive blocker walk; the live state is
        // deliberately not written, because a loaded graph has not fitted anything yet.
        return {};
    }
    default:
        break;
    }
    return "kind not yet reconstructible";
}

namespace {

/// Append the blockers of @p graph and, recursively, of every Loop body,
/// Conditional branch and Setup body below it, tagging each with the path that
/// reaches it.
///
/// Recursion is written out here rather than delegated to
/// ``Graph::for_each_subgraph``, which visits one level and hands the visitor
/// only the sub-graph: the path needs the control-flow NODE that owns the body,
/// and that is exactly what the visitor cannot see.
// NOLINTNEXTLINE(misc-no-recursion): control-flow bodies nest, so the walk over them does too.
void collect_blockers(Graph const &graph, std::string const &path, std::vector<SerializabilityBlocker> &out) {
    // NOLINTNEXTLINE(misc-no-recursion): see above.
    auto descend = [&](Graph const *sub, std::string const &step) {
        if (sub != nullptr) {
            collect_blockers(*sub, path.empty() ? step : path + "/" + step, out);
        }
    };

    for (auto const &node : graph.nodes()) {
        if (std::string reason = reconstruction_blocker(node); !reason.empty()) {
            out.push_back(SerializabilityBlocker{.node_id       = node.id,
                                                 .label         = node.label,
                                                 .kind_name     = std::string{op_kind_name(node.kind)},
                                                 .reason        = std::move(reason),
                                                 .subgraph_path = path});
        }

        if (auto const *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
            descend(loop->body.get(), fmt::format("loop({})", node.label));
        } else if (auto const *cond = std::get_if<ConditionalDescriptor>(&node.op_data)) {
            descend(cond->then_branch.get(), fmt::format("then({})", node.label));
            descend(cond->else_branch.get(), fmt::format("else({})", node.label));
        } else if (auto const *setup = std::get_if<SetupDescriptor>(&node.op_data)) {
            descend(setup->body.get(), fmt::format("setup({})", node.label));
        }
    }
}

} // namespace

std::vector<SerializabilityBlocker> Graph::serializability_report() const {
    std::vector<SerializabilityBlocker> blockers;
    collect_blockers(*this, std::string{}, blockers);
    return blockers;
}

std::function<void()> build_executor(OpKind kind, packed_gemm::ScalarType dtype, std::size_t rank, OpData const &desc, Graph &graph,
                                     std::span<TensorId const> inputs, std::span<TensorId const> outputs) {
    // Rank is keyed on the DESTINATION, and several kinds legitimately have a
    // rank-0 one: an einsum whose spec has no C indices ("<- i ; i"), a dot and
    // a trace writing a registered scalar, a write_param, whose destination is a
    // ParamTable entry that no tensor names at all, and the two control-flow
    // kinds, which have no tensor destination of any rank. Rank is not
    // dispatched on anyway; see the header.
    bool const scalar_destination_ok =
        kind == OpKind::Einsum || kind == OpKind::Dot || kind == OpKind::Trace || kind == OpKind::WriteParam || is_control_flow(kind);
    if (rank == 0 && !scalar_destination_ok) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor({}): rank must be at least 1", op_kind_name(kind));
    }

    switch (kind) {
    case OpKind::Scale: {
        need(kind, "outputs", outputs, 1);
        auto const &d = expect<ScaleDescriptor>(kind, desc, "ScaleDescriptor");
        return build_scale(dtype, d, resolve_operand(graph, outputs[0], kind, "A"));
    }
    case OpKind::Permute: {
        need(kind, "inputs", inputs, 1);
        need(kind, "outputs", outputs, 1);
        auto const &d = expect<PermuteDescriptor>(kind, desc, "PermuteDescriptor");
        return build_permute(dtype, d, resolve_operand(graph, inputs[0], kind, "A"), resolve_operand(graph, outputs[0], kind, "C"));
    }
    case OpKind::Transpose: {
        need(kind, "inputs", inputs, 1);
        need(kind, "outputs", outputs, 1);
        if (rank != 2) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "build_executor(Transpose): rank must be 2, got {}", rank);
        }
        return build_transpose(dtype, resolve_operand(graph, inputs[0], kind, "A"), resolve_operand(graph, outputs[0], kind, "C"));
    }
    case OpKind::Axpby: {
        need(kind, "inputs", inputs, 1);
        need(kind, "outputs", outputs, 1);
        auto const &d = expect<AxpbyDescriptor>(kind, desc, "AxpbyDescriptor");
        return build_axpby(dtype, d, resolve_operand(graph, inputs[0], kind, "X"), resolve_operand(graph, outputs[0], kind, "Y"));
    }
    case OpKind::DirectProduct:
    case OpKind::DirectDivision: {
        need(kind, "inputs", inputs, 2);
        need(kind, "outputs", outputs, 1);
        auto const &d = expect<ElementwiseBinaryDescriptor>(kind, desc, "ElementwiseBinaryDescriptor");
        return build_elementwise_binary(kind, dtype, d, resolve_operand(graph, inputs[0], kind, "A"),
                                        resolve_operand(graph, inputs[1], kind, "B"), resolve_operand(graph, outputs[0], kind, "C"));
    }
    case OpKind::Einsum: {
        // A repeated destination in inputs[2] is the RMW convention, not a
        // fourth operand, so only the leading two inputs are read here.
        need(kind, "inputs", inputs, 2);
        need(kind, "outputs", outputs, 1);
        auto const &d = expect<EinsumDescriptor>(kind, desc, "EinsumDescriptor");
        return build_einsum(dtype, d, resolve_operand(graph, inputs[0], kind, "A"), resolve_operand(graph, inputs[1], kind, "B"),
                            resolve_operand(graph, outputs[0], kind, "C"));
    }
    case OpKind::Dot: {
        need(kind, "inputs", inputs, 2);
        need(kind, "outputs", outputs, 1);
        auto const &d = expect<DotDescriptor>(kind, desc, "DotDescriptor");
        return build_dot(dtype, d, resolve_operand(graph, inputs[0], kind, "A"), resolve_operand(graph, inputs[1], kind, "B"),
                         resolve_scalar_operand(graph, outputs[0], scalar_context(kind), "result"));
    }
    case OpKind::Trace: {
        need(kind, "inputs", inputs, 1);
        need(kind, "outputs", outputs, 1);
        // The descriptor carries nothing a trace reads; it is checked so that a
        // node holding the wrong one is refused rather than run.
        (void)expect<TraceDescriptor>(kind, desc, "TraceDescriptor");
        return build_trace(dtype, resolve_operand(graph, inputs[0], kind, "A"),
                           resolve_scalar_operand(graph, outputs[0], scalar_context(kind), "result"));
    }
    case OpKind::WriteParam: {
        auto const &d = expect<WriteParamDescriptor>(kind, desc, "WriteParamDescriptor");
        // The expression arm carries its own value and names no operand; the
        // tensor arm reads inputs[0]. Both rebuild here, and only the Callback
        // arm of the expression blocks a SAVE - see reconstruction_blocker.
        if (d.source_expr.has_value()) {
            return build_write_param_expr(d.name, d.source_expr.value(), graph.params_ptr());
        }
        need(kind, "inputs", inputs, 1);
        return build_write_param(d, graph.params_ptr(), resolve_scalar_operand(graph, inputs[0], scalar_context(kind), "source"));
    }
    case OpKind::Gemm: {
        // C repeated in inputs[2] is the RMW convention of an accumulating
        // gemm, not a fourth operand; only the leading two inputs are read.
        need(kind, "inputs", inputs, 2);
        need(kind, "outputs", outputs, 1);
        auto const &d = expect<GemmDescriptor>(kind, desc, "GemmDescriptor");
        return build_gemm(dtype, d, resolve_operand(graph, inputs[0], kind, "A"), resolve_operand(graph, inputs[1], kind, "B"),
                          resolve_operand(graph, outputs[0], kind, "C"));
    }
    case OpKind::ElementTransform: {
        auto const &d = expect<ElementTransformDescriptor>(kind, desc, "ElementTransformDescriptor");
        need(kind, "outputs", outputs, 1);
        return build_element_transform(dtype, d, resolve_operand(graph, outputs[0], kind, "C"));
    }
    case OpKind::Syev: {
        // A is listed in BOTH lists because it is decomposed in place, so the matrix operand
        // is read from outputs[0] and never from inputs[0]: a pass that rewrote the input
        // list would otherwise silently move which buffer gets overwritten. W is outputs[1].
        need(kind, "outputs", outputs, 2);
        auto const &d = expect<SyevDescriptor>(kind, desc, "SyevDescriptor");
        return build_syev(dtype, d, resolve_operand(graph, outputs[0], kind, "A"), resolve_operand(graph, outputs[1], kind, "W"));
    }
    case OpKind::Conditional:
        return build_conditional(expect<ConditionalDescriptor>(kind, desc, "ConditionalDescriptor"), graph.params_ptr());
    case OpKind::Loop:
        return build_loop(expect<LoopDescriptor>(kind, desc, "LoopDescriptor"), graph.params_ptr());
    case OpKind::Setup:
        return build_setup(expect<SetupDescriptor>(kind, desc, "SetupDescriptor"));
    case OpKind::LaplaceQuadrature:
        return build_laplace_quadrature(dtype, expect<LaplaceQuadratureDescriptor>(kind, desc, "LaplaceQuadratureDescriptor"), graph,
                                        inputs, outputs);
    default:
        break;
    }

    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                            "build_executor: op kind '{}' has no builder entry yet; its executor is still hand-baked at capture, so "
                            "the node cannot be rebuilt from data alone",
                            op_kind_name(kind));
}

EINSUMS_NAMESPACE_END(compute_graph)
