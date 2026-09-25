//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ErasedOperations.hpp
/// @brief The library halves of the dense ``cg::`` operations: declared here, compiled once in src/Operations/.
///
/// Each dense operation in Operations.hpp is a thin template that hands its operands' TensorImpls
/// to an eager entry, or, when capturing, their registered ids to a capture entry. The entries are
/// templated only on the element type, declared without a body, and explicitly instantiated in the
/// library for float, double, complex<float> and complex<double>, so a caller compiles neither the
/// kernel dispatch nor the node recording, and cannot change them.

#include <Einsums/ComputeGraph/LuPivots.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class CaptureContext;
struct ParsedPermuteSpec;
struct TensorSlot;

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// A tensor whose storage is a single @c TensorImpl of its element type: the dense tensors, views
/// and runtime-rank tensors, which are what the library's operation entries take.
template <typename A>
concept ImplBackedTensor = requires(std::remove_cvref_t<A> &t) {
    { t.impl() } -> std::same_as<einsums::detail::TensorImpl<typename std::remove_cvref_t<A>::ValueType> &>;
};

// ── scale ─────────────────────────────────────────────────────────────────────

/// ``A := factor * A``, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_scale(T factor, einsums::detail::TensorImpl<T> &A);

/// Record ``A := factor * A`` into the capturing graph. @p a_id is A's registered id.
template <typename T>
EINSUMS_EXPORT void capture_scale(CaptureContext &ctx, T factor, TensorId a_id, std::string_view name, std::size_t rank);

// ── axpy / axpby ──────────────────────────────────────────────────────────────

/// ``Y := alpha * X + Y``, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_axpy(T alpha, einsums::detail::TensorImpl<T> const &X, einsums::detail::TensorImpl<T> &Y);

/// Record ``Y := alpha * X + Y``. It records as an axpby with beta = 1.
template <typename T>
EINSUMS_EXPORT void capture_axpy(CaptureContext &ctx, T alpha, TensorId x_id, TensorId y_id, std::string_view x_name,
                                 std::string_view y_name, std::size_t rank);

/// ``Y := alpha * X + beta * Y``, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_axpby(T alpha, einsums::detail::TensorImpl<T> const &X, T beta, einsums::detail::TensorImpl<T> &Y);

/// Record ``Y := alpha * X + beta * Y``.
template <typename T>
EINSUMS_EXPORT void capture_axpby(CaptureContext &ctx, T alpha, T beta, TensorId x_id, TensorId y_id, std::size_t rank);

// ── direct_product / direct_division ──────────────────────────────────────────

/// ``C := alpha * A * B + beta * C`` element-wise, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_direct_product(T alpha, einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &B, T beta,
                                         einsums::detail::TensorImpl<T> &C);

/// ``C := alpha * A / B + beta * C`` element-wise, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_direct_division(T alpha, einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &B, T beta,
                                          einsums::detail::TensorImpl<T> &C);

/**
 * @brief Record an element-wise binary operation, @c OpKind::DirectProduct or
 *        @c OpKind::DirectDivision, into the capturing graph.
 *
 * The prefactors keep the caller's scalar type. @p reads_c says whether beta is nonzero, in
 * which case the node reads its destination and lists C as an input.
 */
EINSUMS_EXPORT void capture_elementwise_binary(CaptureContext &ctx, OpKind kind, PrefactorScalar alpha, PrefactorScalar beta, bool reads_c,
                                               TensorId a_id, TensorId b_id, TensorId c_id, packed_gemm::ScalarType dtype,
                                               std::size_t rank);

// ── element_transform by name ─────────────────────────────────────────────────

/// Apply the registered element op @p op_name to every element of @p C, eagerly.
/// @throws std::invalid_argument as @c cg::element_transform documents.
template <typename T>
EINSUMS_EXPORT void eager_element_transform(einsums::detail::TensorImpl<T> &C, std::string_view op_name, std::optional<double> param);

/// Record the registered element op @p op_name on C. The op is looked up here, so an unknown
/// name or a misplaced parameter fails at capture rather than at replay.
template <typename T>
EINSUMS_EXPORT void capture_element_transform(CaptureContext &ctx, TensorId c_id, std::size_t rank, std::string_view op_name,
                                              std::optional<double> param);

// ── permute / transpose ───────────────────────────────────────────────────────

/// ``C := beta * C + alpha * permute(A)`` on a parsed spec, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_permute(ParsedPermuteSpec const &parsed, T beta, einsums::detail::TensorImpl<T> &C, T alpha,
                                  einsums::detail::TensorImpl<T> const &A);

/// Record ``C := beta * C + alpha * permute(A)``.
template <typename T>
EINSUMS_EXPORT void capture_permute(CaptureContext &ctx, ParsedPermuteSpec const &parsed, T beta, T alpha, TensorId a_id, TensorId c_id,
                                    std::size_t rank);

/// ``C := A^T`` for rank-2 operands, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_transpose(einsums::detail::TensorImpl<T> &C, einsums::detail::TensorImpl<T> const &A);

/// Record ``C := A^T``.
template <typename T>
EINSUMS_EXPORT void capture_transpose(CaptureContext &ctx, TensorId a_id, TensorId c_id, std::size_t rank);

// ── gemm ──────────────────────────────────────────────────────────────────────

/// A tensor's declared symmetry, or null for a type that carries none (a view, a runtime tensor).
template <typename A>
SymmetryDescriptor const *symmetry_of(A const &tensor) noexcept {
    if constexpr (requires { tensor.symmetry(); }) {
        return tensor.symmetry();
    } else {
        return nullptr;
    }
}

/**
 * @brief ``C := alpha * op(A) op(B) + beta * C`` for rank-2 operands, eagerly.
 *
 * @p ta and @p tb are ``'n'``, ``'t'`` or ``'c'``. When either symmetry descriptor is non-null
 * and describes a symmetric or Hermitian matrix, this dispatches to ``symm`` / ``hemm`` instead,
 * as the typed @c linear_algebra::gemm does.
 */
template <typename T>
EINSUMS_EXPORT void eager_gemm(char ta, char tb, T alpha, einsums::detail::TensorImpl<T> const &A, SymmetryDescriptor const *desc_a,
                               einsums::detail::TensorImpl<T> const &B, SymmetryDescriptor const *desc_b, T beta,
                               einsums::detail::TensorImpl<T> &C);

/**
 * @brief Record ``C := alpha * op(A) op(B) + beta * C``.
 *
 * @p from_flags names the overload for the node's label: ``gemm<N,T>`` for the compile-time flags,
 * ``gemm(n,t)`` for the runtime ones. @p reads_c says whether beta is nonzero.
 */
template <typename T>
EINSUMS_EXPORT void capture_gemm(CaptureContext &ctx, char ta, char tb, bool from_flags, T alpha, T beta, bool reads_c, TensorId a_id,
                                 TensorId b_id, TensorId c_id);

// ── dot / trace ───────────────────────────────────────────────────────────────

/// ``sum A * B`` over every element, or ``sum conj(A) * B`` when @p conjugated, eagerly and with
/// the vendor BLAS held to one thread, so the summation order depends on the operands alone.
template <typename T>
EINSUMS_EXPORT T eager_dot(einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &B, bool conjugated);

/// Record a dot product into the destination @p r_id, keyed on @p rank: 0 for a registered scalar,
/// the destination tensor's rank otherwise.
template <typename T>
EINSUMS_EXPORT void capture_dot(CaptureContext &ctx, bool conjugated, TensorId a_id, TensorId b_id, TensorId r_id, std::size_t rank);

/// The sum of a square rank-2 operand's diagonal, in index order, eagerly.
/// @throws RankError for an operand that is not rank-2; std::invalid_argument for one that is not square.
template <typename T>
EINSUMS_EXPORT T eager_trace(einsums::detail::TensorImpl<T> const &A);

/// Record a trace into the destination @p r_id, keyed on @p rank as for @ref capture_dot.
template <typename T>
EINSUMS_EXPORT void capture_trace(CaptureContext &ctx, TensorId a_id, TensorId r_id, std::size_t rank);

// ── gemv / ger / gerc ─────────────────────────────────────────────────────────

/// An operand as @c CaptureContext::get_slot returns it: its id and the slot a replay resolves it through.
using SlotRef = std::pair<TensorId, TensorSlot *>;

/// ``y := alpha * op(A) x + beta * y`` for a rank-2 A, eagerly. @p ta is ``'n'``, ``'t'`` or ``'c'``.
template <typename T>
EINSUMS_EXPORT void eager_gemv(char ta, T alpha, einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &x, T beta,
                               einsums::detail::TensorImpl<T> &y);

/// Record ``y := alpha * op(A) x + beta * y``. @p from_flags names the overload for the label, as
/// for @ref capture_gemm; @p reads_y says whether beta is nonzero.
template <typename T>
EINSUMS_EXPORT void capture_gemv(CaptureContext &ctx, char ta, bool from_flags, T alpha, T beta, bool reads_y, SlotRef a, SlotRef x,
                                 SlotRef y);

/// ``A := A + alpha * x y^T``, or ``A + alpha * x y^H`` when @p conjugated, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_ger(bool conjugated, T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
                              einsums::detail::TensorImpl<T> &A);

/// Record the rank-1 update ``A := A + alpha * x y^T`` (``y^H`` when @p conjugated).
template <typename T>
EINSUMS_EXPORT void capture_ger(CaptureContext &ctx, bool conjugated, T alpha, SlotRef x, SlotRef y, SlotRef a);

// ── LAPACK ────────────────────────────────────────────────────────────────────

/// Symmetric (real @p T) or Hermitian (complex @p T) eigendecomposition of A, eagerly: the
/// eigenvalues into W, and the eigenvectors over A when @p compute_eigenvectors.
template <typename T>
EINSUMS_EXPORT void eager_syev(bool compute_eigenvectors, einsums::detail::TensorImpl<T> &A,
                               einsums::detail::TensorImpl<RemoveComplexT<T>> &W);

/// Record the eigendecomposition: an @c OpKind::Syev node carrying a @c SyevDescriptor for real
/// @p T, an @c OpKind::Heev node for complex @p T. A is both an input and an output.
template <typename T>
EINSUMS_EXPORT void capture_syev(CaptureContext &ctx, bool compute_eigenvectors, SlotRef a, SlotRef w);

/// Solve ``A X = B`` in place, eagerly: B becomes X and A its LU factors. Returns LAPACK's info.
template <typename T>
EINSUMS_EXPORT int eager_gesv(einsums::detail::TensorImpl<T> &A, einsums::detail::TensorImpl<T> &B);

/// Record ``A X = B``, solved in place.
template <typename T>
EINSUMS_EXPORT void capture_gesv(CaptureContext &ctx, SlotRef a, SlotRef b);

/// Invert A in place, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_invert(einsums::detail::TensorImpl<T> &A);

/// Record the in-place inversion of A.
template <typename T>
EINSUMS_EXPORT void capture_invert(CaptureContext &ctx, SlotRef a);

/// LU-factor A in place, eagerly, growing @p pivots to ``min(m, n)`` first. Returns LAPACK's info.
template <typename T>
EINSUMS_EXPORT int eager_getrf(einsums::detail::TensorImpl<T> &A, LuPivots &pivots);

/// Record the in-place LU factorization of A. The node shares @p pivots' buffer, so the pivots a
/// replay writes are the ones a later getrs on the same @c LuPivots reads.
template <typename T>
EINSUMS_EXPORT void capture_getrf(CaptureContext &ctx, SlotRef a, LuPivots const &pivots);

/// Solve ``A X = B`` in place from A's LU factors and @p pivots, eagerly. Returns LAPACK's info.
template <typename T>
EINSUMS_EXPORT int eager_getrs(einsums::detail::TensorImpl<T> const &A, LuPivots const &pivots, einsums::detail::TensorImpl<T> &B);

/// Record the solve from LU factors.
template <typename T>
EINSUMS_EXPORT void capture_getrs(CaptureContext &ctx, SlotRef a, LuPivots const &pivots, SlotRef b);

// ── unary element-wise ────────────────────────────────────────────────────────

/// ``A := conj(A)``, eagerly; the identity on a real type.
template <typename T>
EINSUMS_EXPORT void eager_conj(einsums::detail::TensorImpl<T> &A);

/// Record ``A := conj(A)`` as a Custom node labelled with @p name.
template <typename T>
EINSUMS_EXPORT void capture_conj(CaptureContext &ctx, SlotRef a, std::string_view name);

/// Which real-valued part of a tensor @ref eager_complex_part takes.
enum class ComplexPart : std::uint8_t { Real, Imag, Abs };

/// ``out := Re(A)``, ``Im(A)`` or ``|A|`` element-wise, eagerly. For a real @p T these are a copy,
/// zero and the absolute value.
template <typename T>
EINSUMS_EXPORT void eager_complex_part(ComplexPart part, einsums::detail::TensorImpl<T> const &A,
                                       einsums::detail::TensorImpl<RemoveComplexT<T>> &out);

/// Record ``out := Re(A)``, ``Im(A)`` or ``|A|`` as a Custom node.
template <typename T>
EINSUMS_EXPORT void capture_complex_part(CaptureContext &ctx, ComplexPart part, SlotRef out, SlotRef a);

/// ``out := sqrt(A)`` element-wise for a real @p T, eagerly.
/// @throws std::domain_error on a negative element.
template <typename T>
EINSUMS_EXPORT void eager_sqrt(einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> &out);

/// Record ``out := sqrt(A)`` as a Custom node.
template <typename T>
EINSUMS_EXPORT void capture_sqrt(CaptureContext &ctx, SlotRef out, SlotRef a);

/// ``out(i) := A(i, i)`` for ``i < min(m, n)``, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_diagonal(einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> &out);

/// Record the diagonal extraction as a Custom node.
template <typename T>
EINSUMS_EXPORT void capture_diagonal(CaptureContext &ctx, SlotRef out, SlotRef a);

/// ``A := A + beta`` element-wise over A's storage, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_shift(T beta, einsums::detail::TensorImpl<T> &A);

/// Record ``A := A + beta`` as a Custom node labelled from @p name.
template <typename T>
EINSUMS_EXPORT void capture_shift(CaptureContext &ctx, T beta, SlotRef a, std::string_view name);

// ── reductions ────────────────────────────────────────────────────────────────

/// The norm of a rank-1 or rank-2 operand, eagerly, with the vendor BLAS held to one thread.
/// @p norm_type is the @c linear_algebra::Norm code.
template <typename T>
EINSUMS_EXPORT RemoveComplexT<T> eager_norm(char norm_type, einsums::detail::TensorImpl<T> const &A);

/// Record a norm written to the registered scalar @p result (id @p r_id).
template <typename T>
EINSUMS_EXPORT void capture_norm(CaptureContext &ctx, char norm_type, SlotRef a, TensorId r_id, RemoveComplexT<T> *result);

/// Record a norm written to the first element of the tensor @p r.
template <typename T>
EINSUMS_EXPORT void capture_norm_into(CaptureContext &ctx, char norm_type, SlotRef a, SlotRef r);

/// The sum of every element, walked stride-correctly so a view reduces correctly, eagerly.
template <typename T>
EINSUMS_EXPORT T eager_sum(einsums::detail::TensorImpl<T> const &A);

/// Record the sum of every element of A into the first element of @p r.
template <typename T>
EINSUMS_EXPORT void capture_sum(CaptureContext &ctx, SlotRef r, SlotRef a);

/// The largest element of a real tensor, NaN-propagating as @c numpy.max is, eagerly.
template <typename T>
EINSUMS_EXPORT T eager_max(einsums::detail::TensorImpl<T> const &A);

/// Record the largest element of A into the first element of @p r.
template <typename T>
EINSUMS_EXPORT void capture_max(CaptureContext &ctx, SlotRef r, SlotRef a);

// ── block copy / gather / scatter ─────────────────────────────────────────────
// The caller validates the arguments; these take them as validated.

/// Copy the block of @p extents at @p src_offsets in src to @p dst_offsets in dst, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_block_copy(einsums::detail::TensorImpl<T> &dst, einsums::detail::TensorImpl<T> const &src,
                                     std::vector<std::size_t> const &dst_offsets, std::vector<std::size_t> const &src_offsets,
                                     std::vector<std::size_t> const &extents);

/// Record the block copy as a Custom node.
template <typename T>
EINSUMS_EXPORT void capture_block_copy(CaptureContext &ctx, SlotRef dst, SlotRef src, std::vector<std::size_t> dst_offsets,
                                       std::vector<std::size_t> src_offsets, std::vector<std::size_t> extents);

/// Gather the elements of src selected by @p indices (one list per source axis) into dst, source
/// axis k landing on destination axis @p dst_axis [k], eagerly.
template <typename T>
EINSUMS_EXPORT void eager_gather(einsums::detail::TensorImpl<T> &dst, einsums::detail::TensorImpl<T> const &src,
                                 std::vector<std::vector<std::size_t>> const &indices, std::vector<std::size_t> const &extents,
                                 std::vector<std::size_t> const &dst_axis);

/// Record the gather as a Custom node.
template <typename T>
EINSUMS_EXPORT void capture_gather(CaptureContext &ctx, SlotRef dst, SlotRef src, std::vector<std::vector<std::size_t>> indices,
                                   std::vector<std::size_t> extents, std::vector<std::size_t> dst_axis);

/// Scatter src into the elements of dst selected by @p indices, eagerly: assigning, or adding
/// when @p accumulate.
template <typename T>
EINSUMS_EXPORT void eager_scatter(bool accumulate, einsums::detail::TensorImpl<T> &dst, einsums::detail::TensorImpl<T> const &src,
                                  std::vector<std::vector<std::size_t>> const &indices, std::vector<std::size_t> const &extents);

/// Record the scatter (or scatter_add) as a Custom node. dst is both an input and an output.
template <typename T>
EINSUMS_EXPORT void capture_scatter(CaptureContext &ctx, bool accumulate, SlotRef dst, SlotRef src,
                                    std::vector<std::vector<std::size_t>> indices, std::vector<std::size_t> extents);

// ── sum_axes / reshape / outer_sum ────────────────────────────────────────────
// The caller validates the arguments; these take them as validated.

/// ``out := A`` summed over every axis not in @p kept (A's axes that survive, in order), eagerly.
/// Assigns rather than accumulates.
template <typename T>
EINSUMS_EXPORT void eager_sum_axes(einsums::detail::TensorImpl<T> &out, einsums::detail::TensorImpl<T> const &A,
                                   std::vector<std::size_t> const &kept);

/// Record the axis sum as a Custom node.
template <typename T>
EINSUMS_EXPORT void capture_sum_axes(CaptureContext &ctx, SlotRef out, SlotRef a, std::vector<std::size_t> kept);

/// Copy A's elements into out's shape through a shared linear index, row- or column-major, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_reshape(einsums::detail::TensorImpl<T> &out, einsums::detail::TensorImpl<T> const &A, bool row_major);

/// Record the reshape as a Custom node.
template <typename T>
EINSUMS_EXPORT void capture_reshape(CaptureContext &ctx, SlotRef out, SlotRef a, bool row_major);

/// ``result(i0, i1, ...) := sum_k coeffs[k] * vectors[k](ik)``, eagerly.
/// @throws std::invalid_argument when a vector's length is not its result axis's extent.
template <typename T>
EINSUMS_EXPORT void eager_outer_sum(einsums::detail::TensorImpl<T>                            &result,
                                    std::vector<einsums::detail::TensorImpl<T> const *> const &vectors, std::vector<T> const &coeffs);

/// Record the outer sum as a Custom node carrying an @c OuterSumDescriptor with @p coefficients
/// (empty meaning all ones), so a pass can read what it computes.
template <typename T>
EINSUMS_EXPORT void capture_outer_sum(CaptureContext &ctx, SlotRef result, std::vector<SlotRef> vectors, std::vector<T> coeffs,
                                      std::vector<double> coefficients);

// ── batched gemm ──────────────────────────────────────────────────────────────
// The caller validates the batch and builds its descriptor; these take both as validated.

/// Run the uniform batch @p d over the members' data pointers, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_batched_gemm(BatchedGemmDescriptor const &d, std::vector<void const *> const &a,
                                       std::vector<void const *> const &b, std::vector<void *> const &c);

/// Record the batch with one destination per member. The node's inputs are A_0, B_0, A_1, B_1, ...
/// (then every C when @p reads_c) and its outputs C_0, C_1, ..., in batch order.
EINSUMS_EXPORT void capture_batched_gemm(CaptureContext &ctx, BatchedGemmDescriptor const &d, bool reads_c, std::vector<SlotRef> const &a,
                                         std::vector<SlotRef> const &b, std::vector<SlotRef> const &c);

/// Record the batch into blocks of one destination, member i at element offset @p c_offsets [i].
EINSUMS_EXPORT void capture_batched_gemm_blocked(CaptureContext &ctx, BatchedGemmDescriptor const &d, bool reads_c,
                                                 std::vector<SlotRef> const &a, std::vector<SlotRef> const &b, SlotRef c_base,
                                                 std::vector<std::size_t> const &c_offsets);

/// Run the grouped batch @p d over the members' data pointers, in its flattened order, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_grouped_batched_gemm(GroupedBatchedGemmDescriptor const &d, std::vector<void const *> const &a,
                                               std::vector<void const *> const &b, std::vector<void *> const &c);

/**
 * @brief Record the grouped batch, members in @p d 's flattened order.
 *
 * With @p c_bases empty, each member has its own destination @p c [i] and the node's outputs are
 * those. Otherwise the members write blocks of the distinct destinations @p c_bases, which are the
 * node's outputs: member i into @p c [i] (one of them) at element offset @p c_offsets [i].
 */
EINSUMS_EXPORT void capture_grouped_batched_gemm(CaptureContext &ctx, GroupedBatchedGemmDescriptor d, bool reads_c, bool trans_a,
                                                 bool trans_b, std::vector<SlotRef> const &a, std::vector<SlotRef> const &b,
                                                 std::vector<SlotRef> const &c, std::vector<SlotRef> const &c_bases,
                                                 std::vector<std::size_t> const &c_offsets = {});

// ── grouped element-wise and reductions ───────────────────────────────────────
// The caller validates each member; these take the lists as validated, in member order.

/// ``results[i][0] := sum a[i] * b[i]`` for each member, eagerly, with the vendor BLAS held to one thread.
template <typename T>
EINSUMS_EXPORT void eager_grouped_dot(std::vector<einsums::detail::TensorImpl<T> *> const       &results,
                                      std::vector<einsums::detail::TensorImpl<T> const *> const &a,
                                      std::vector<einsums::detail::TensorImpl<T> const *> const &b);

/// Record the grouped dot. Inputs A_0, B_0, A_1, B_1, ...; outputs the results, in member order.
template <typename T>
EINSUMS_EXPORT void capture_grouped_dot(CaptureContext &ctx, std::vector<SlotRef> const &results, std::vector<SlotRef> const &a,
                                        std::vector<SlotRef> const &b);

/// ``y[i] := alphas[i] * x[i] + betas[i] * y[i]`` for each member, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_grouped_axpby(std::vector<T> const &alphas, std::vector<einsums::detail::TensorImpl<T> const *> const &x,
                                        std::vector<T> const &betas, std::vector<einsums::detail::TensorImpl<T> *> const &y);

/// Record the grouped axpby. A member with a nonzero beta also lists its Y as an input.
template <typename T>
EINSUMS_EXPORT void capture_grouped_axpby(CaptureContext &ctx, std::vector<T> alphas, std::vector<T> betas, std::vector<SlotRef> const &x,
                                          std::vector<SlotRef> const &y);

/// ``c[i] := c_pfs[i] * c[i] + a_pfs[i] * permute(a[i])`` on one parsed spec for each member, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_grouped_permute(ParsedPermuteSpec const &parsed, std::vector<T> const &c_pfs,
                                          std::vector<einsums::detail::TensorImpl<T> *> const &c, std::vector<T> const &a_pfs,
                                          std::vector<einsums::detail::TensorImpl<T> const *> const &a);

/// Record the grouped permute. A member with a nonzero c_pf also lists its C as an input.
template <typename T>
EINSUMS_EXPORT void capture_grouped_permute(CaptureContext &ctx, ParsedPermuteSpec parsed, std::vector<T> c_pfs, std::vector<T> a_pfs,
                                            std::vector<SlotRef> const &a, std::vector<SlotRef> const &c);

/// ``c[i] := alphas[i] * a[i] (*|/) b[i] + betas[i] * c[i]`` element-wise for each member, eagerly:
/// a product for @c OpKind::GroupedDirectProduct, a division for @c OpKind::GroupedDirectDivision.
template <typename T>
EINSUMS_EXPORT void eager_grouped_binary(OpKind kind, std::vector<T> const &alphas,
                                         std::vector<einsums::detail::TensorImpl<T> const *> const &a,
                                         std::vector<einsums::detail::TensorImpl<T> const *> const &b, std::vector<T> const &betas,
                                         std::vector<einsums::detail::TensorImpl<T> *> const &c);

/// Record the grouped product or division as a @p kind node labelled ``"<label> x<count>"``.
template <typename T>
EINSUMS_EXPORT void capture_grouped_binary(CaptureContext &ctx, OpKind kind, char const *label, std::vector<T> const &alphas,
                                           std::vector<T> const &betas, std::vector<SlotRef> const &a, std::vector<SlotRef> const &b,
                                           std::vector<SlotRef> const &c);

// ── grouped sandwich / gather-rotate (real element types) ─────────────────────

/// ``c[i] += sum_q B_q s[i] B_q^T`` with ``B_q = a[i][q] - p[i]^T m[i][q]``, for each member, eagerly.
template <typename T>
EINSUMS_EXPORT void eager_grouped_sandwich(std::vector<einsums::detail::TensorImpl<T> *> const       &c,
                                           std::vector<einsums::detail::TensorImpl<T> const *> const &a,
                                           std::vector<einsums::detail::TensorImpl<T> const *> const &m,
                                           std::vector<einsums::detail::TensorImpl<T> const *> const &p,
                                           std::vector<einsums::detail::TensorImpl<T> const *> const &s);

/// Record the grouped sandwich. Each member lists A, M, P, S and C (which it accumulates into) as inputs.
template <typename T>
EINSUMS_EXPORT void capture_grouped_sandwich(CaptureContext &ctx, std::vector<SlotRef> const &c, std::vector<SlotRef> const &a,
                                             std::vector<SlotRef> const &m, std::vector<SlotRef> const &p, std::vector<SlotRef> const &s);

/// ``c[i](t, u, v) := sum x[i]^T src(q_list[i], u_list[i], u_list[i]) x[i]``, the gather of one
/// member's indices from @p src and its two-sided rotation by x[i], for each member, eagerly.
template <typename T>
EINSUMS_EXPORT void
eager_grouped_gather_rotate(std::vector<einsums::detail::TensorImpl<T> *> const &c, einsums::detail::TensorImpl<T> const &src,
                            std::vector<einsums::detail::TensorImpl<T> const *> const &x,
                            std::vector<std::vector<std::size_t>> const &q_list, std::vector<std::vector<std::size_t>> const &u_list);

/// Record the grouped gather-rotate. The destinations are assigned, so they are outputs only.
template <typename T>
EINSUMS_EXPORT void capture_grouped_gather_rotate(CaptureContext &ctx, std::vector<SlotRef> const &c, SlotRef src,
                                                  std::vector<SlotRef> const &x, std::vector<std::vector<std::size_t>> q_list,
                                                  std::vector<std::vector<std::size_t>> u_list);

EINSUMS_NAMESPACE_END(compute_graph::detail)
