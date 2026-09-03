//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/PredExpr.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/TensorSlot.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph; // Forward declaration for ConditionalDescriptor/LoopDescriptor

// NodeId, OpKind, Target, and simple descriptor types are now defined in
// <Einsums/ComputeGraphTypes/Enums.hpp>, <Einsums/ComputeGraphTypes/Ids.hpp>,
// and <Einsums/ComputeGraphTypes/Descriptors.hpp>.

/**
 * @brief One operand of a GEMM-shaped einsum, recorded as data.
 *
 * @par Why the leading dimension is only a hint
 * @ref leading_dim is a SNAPSHOT, taken when the hint was derived. It exists
 * so a planning pass can compare operand layouts across candidate nodes
 * *before* anything executes, which is what @ref passes::GEMMBatching does to
 * decide whether one ``gemm_batch`` call can address a whole group.
 *
 * It is never what the batch actually runs on. ``Graph::rebind`` accepts any
 * tensor of matching rank and dims, so a rebind to a column slice of a larger
 * store legitimately changes the leading dimension without changing anything
 * the hint records; the batched executor therefore re-derives every leading
 * dimension from the live @c TensorImpl at execute time and the snapshot is
 * not consulted again. A rebind that changes an lda *after* a batch was formed
 * consequently still computes correctly, because the group's descriptor is
 * rebuilt from the live values on the next replay.
 *
 * The identity that survives a rebind is @ref id, which is why the operand is
 * recorded as an id rather than as a pointer or a closure over one.
 */
struct GemmOperand {
    TensorId id{0};          ///< The operand's tensor id. Resolution happens at execute, through the graph's slot.
    int      leading_dim{0}; ///< Capture-time leading dimension. A PLANNING HINT; see the class note.
};

/**
 * @brief BLAS-level batching hint for 2D×2D→2D einsums.
 *
 * Populated when a contraction matches the GEMM pattern (two rank-2 inputs,
 * one rank-2 output, one link index, and strides agreeing with each operand's
 * declared layout). The GEMMBatching pass reads this to decide which einsums
 * can be collapsed into a single `blas::gemm_batch` call, and ThreadPlanning
 * reads m/n/k to size a node. Non-GEMM contractions leave
 * ``gemm_hint == nullptr``.
 *
 * Every field is DATA. The operands used to be three ``std::function`` members that
 * resolved a live pointer plus leading dimension at call time; a closure
 * cannot be written to a file, and it was the last thing standing between an
 * einsum node and being reconstructible from its descriptor
 * (@ref is_reconstructible). Resolution moved into the batched executor, which
 * reaches the same live geometry through the graph's slots and therefore
 * honors ``rebind`` and ``redirect_slot`` by construction rather than by each
 * closure remembering to.
 *
 * @see derive_gemm_hint, the single derivation both capture and
 *      @ref Graph::make_einsum_node use.
 */
struct GemmHint {
    BlasScalar  scalar;       ///< Element type for gemm_batch dispatch.
    int         m{0};         ///< Rows of C (and A if trans_a=='N').
    int         n{0};         ///< Cols of C (and B if trans_b=='N').
    int         k{0};         ///< Link dimension.
    char        trans_a{'N'}; ///< Transpose flag for A derived from its index order.
    char        trans_b{'N'}; ///< Transpose flag for B.
    GemmOperand a;            ///< Left input.
    GemmOperand b;            ///< Right input.
    GemmOperand c;            ///< Destination.
};

/**
 * @brief Metadata for Einsum nodes, enabling optimization passes.
 *
 * Stores the contraction pattern (which indices belong to A, B, C, which are
 * link/target indices) and the scalar prefactors. This metadata is used by:
 * - ScaleAbsorption to spot dead scales (einsum with c_prefactor == 0 overwrites them)
 * - CSE to detect duplicate computations
 * - ContractionPlanning to detect and restructure GEMM chains
 *
 * @see packed_gemm::ContractionSpec for the contraction topology format
 */
struct EinsumDescriptor {
    packed_gemm::ContractionSpec spec;                    ///< Contraction topology: index lists, link/target classification
    PrefactorScalar              c_prefactor{double{0}};  ///< C prefactor (snapshot of EinsumParams::c_pf at capture time)
    PrefactorScalar              ab_prefactor{double{1}}; ///< AB prefactor (snapshot of EinsumParams::ab_pf at capture time)
    bool                         conj_a{false};           ///< Whether to conjugate A (for complex types)
    bool                         conj_b{false};           ///< Whether to conjugate B (for complex types)

    /// Live-mutable index state shared with the executor lambda.
    ///
    /// Optimization passes (PermuteFusion, future index rewriters) mutate
    /// this in place; the executor dereferences it on every call, so
    /// rewrites take effect on the next `graph.execute()`. The
    /// `spec` field above is the at-capture snapshot. Analysis passes
    /// can read either, but rewriters must update both to keep
    /// downstream analysis consistent.
    std::shared_ptr<EinsumIndices> indices;

    /// Live-mutable scalar state shared with the executor lambda, same
    /// pattern as `indices`.
    ///
    /// CPU executors read prefactors from here on
    /// every call; `c_prefactor`/`ab_prefactor` above are the at-capture
    /// snapshots (still read by GPU dispatch). Graph::update_prefactors
    /// writes both through this handle so the node stays self-contained
    /// under passes that reorder or remove nodes.
    std::shared_ptr<EinsumParams> params;

    /// BLAS-level batching hint; non-null only for 2D×2D→2D contractions
    /// with one link index. Read by the GEMMBatching pass.
    std::shared_ptr<GemmHint> gemm_hint;

    /// This node's packed-GEMM memo, the same object the executor lambda holds -
    /// the `indices`/`params` pattern again, and shared for the same reason.
    ///
    /// A plan-time pass needs somewhere to write the node's kernel route (@ref
    /// packed_gemm::KernelRoute), and the site is where the dispatch already
    /// looks. One site per node, so writing a route through this handle settles
    /// it for this node and no other.
    std::shared_ptr<packed_gemm::ContractionSite> site;

    /// Index space bound to each index letter of THIS contraction, sorted by letter.
    ///
    /// Derived at capture (and rebuilt whenever a pass reconstructs the node) by walking every
    /// operand's index letters against that operand's @ref TensorHandle::spaces. A space is a
    /// property of a tensor SLOT, not of a letter globally, so the same letter may legitimately
    /// range over different spaces in different contractions of one program; this field is the
    /// per-node resolution of that, which is why it sits beside the index lists rather than in
    /// @ref spec.
    ///
    /// Empty when no operand of the node carries an annotation, which is every node of an
    /// unannotated program. A letter that only ever met unannotated slots gets no entry, so a
    /// partially annotated program yields a partial map rather than a wrong one.
    ///
    /// Deliberately NOT part of @ref packed_gemm::ContractionSpec: that type's ``operator==`` is
    /// a plan-cache key, and a semantic annotation must never split a cache entry.
    ///
    /// @see space_for_letter
    std::vector<std::pair<std::string, SpaceId>> letter_spaces;

    /**
     * @brief The space bound to one index letter of this contraction.
     * @param[in] letter The index letter to look up.
     * @return The bound space, or an empty optional when the letter carries no annotation.
     */
    [[nodiscard]] std::optional<SpaceId> space_for_letter(std::string_view letter) const {
        for (auto const &entry : letter_spaces) {
            if (entry.first == letter) {
                return entry.second;
            }
        }
        return std::nullopt;
    }
};

/// Live-mutable scalars for every kind whose operation is ``dst = alpha*<source> + beta*dst``,
/// shared with the executor lambda - the same snapshot + shared-params pattern as
/// @ref EinsumParams.
///
/// The executor reads its prefactors from here on every call, so a pass that rewrites one
/// writes it through this handle and the change takes effect on the next ``graph.execute()``.
/// That is the whole point: the executors these kinds used to carry baked their scalars into a
/// closure, so a descriptor rewrite was silently ignored at replay.
///
/// A descriptor also keeps an at-capture SNAPSHOT of the same scalars, which is what analysis
/// passes read; a rewriter must write both, and the ``live_*`` accessors below are how a reader
/// stays right about which one matters.
///
/// @par One type, three names
/// Axpby, the dense element-wise kinds and the tiled element-wise kinds each used to declare
/// their own field-for-field identical struct. They are NOT variant alternatives - each is
/// reached through its own descriptor's ``params`` member - so nothing ever discriminated
/// between them, and three definitions were only three things to keep in step. The aliases
/// below keep each descriptor reading in its own vocabulary.
///
/// Not every kind uses both halves: a Scale is an in-place multiply with no destination
/// prefactor, and a tiled Scale or Axpy is the same, so @ref beta stays at its default there.
struct ElementwiseParams {
    PrefactorScalar alpha{double{1}}; ///< Prefactor on the source operand(s).
    PrefactorScalar beta{double{0}};  ///< Prefactor on the destination (0 = pure overwrite).
};

/// @copydoc ElementwiseParams
/// Spelled for axpby's ``Y = alpha*X + beta*Y``.
using AxpbyParams = ElementwiseParams;

/// Metadata for Axpby nodes (Y = alpha*X + beta*Y). Prefactors are type-erased
/// (PrefactorScalar) so complex axpby folds exactly, matching EinsumDescriptor.
struct AxpbyDescriptor {
    PrefactorScalar              alpha{double{1}}; ///< alpha snapshot (at-capture value)
    PrefactorScalar              beta{double{0}};  ///< beta snapshot (at-capture value)
    std::shared_ptr<AxpbyParams> params;           ///< live values the executor reads each call
};

/**
 * @brief Metadata for @ref OpKind::Scale nodes: ``A *= factor``.
 *
 * Used by ScaleAbsorption to detect scales a following overwrite makes dead,
 * and by ElementWiseFusion to merge consecutive scales of one tensor.
 *
 * @ref factor is a @ref PrefactorScalar rather than the plain @c double it was
 * until the executor builder landed. Capture filled the old field from
 * ``factor.real()``, so a complex scale read back as its real part: CSE merged
 * scales differing only in the imaginary part, and ScaleAbsorption folded the
 * truncated value into the following op. The variant records exactly what the
 * executor applies.
 *
 * @ref params is null on nodes a pass built by hand rather than through
 * @ref build_executor; readers must gate on it.
 */
struct ScaleDescriptor {
    PrefactorScalar                    factor{double{1}}; ///< The scaling factor (at-capture snapshot).
    std::shared_ptr<ElementwiseParams> params;            ///< Live value the executor reads each call (@ref ElementwiseParams::alpha).
};

/**
 * @brief Metadata for @ref OpKind::Permute nodes.
 *
 * Stores the alpha/beta prefactors: C = beta * C + alpha * permute(A).
 *
 * The snapshot scalars are @c std::complex<double> (the widest concrete scalar,
 * as in @ref BatchedGemmDescriptor) so a complex permute records exactly what
 * its executor will apply. They were @c double, filled from @c alpha.real() at
 * capture, which made `alpha = 1+3i` read back as a plain `1.0`: PermuteFusion
 * saw a pure axis reorder and fused the permute away, dropping the imaginary
 * part, and CSE merged permutes that differ only in it.
 *
 * The INDEX LISTS are deliberately not live-mutable. No pass rewrites them
 * today, and a pass that wants to has a better move than mutating them under a
 * running executor: rewrite the descriptor and call @ref build_executor again.
 * The scalars are live because passes genuinely do rewrite those between
 * replays.
 */
struct PermuteDescriptor {
    std::complex<double>               alpha{1.0, 0.0}; ///< Prefactor for the source tensor (at-capture snapshot).
    std::complex<double>               beta{0.0, 0.0};  ///< Prefactor for the destination tensor (0 = overwrite).
    std::vector<std::string>           c_indices;       ///< Output index names (e.g., {"j","i"}).
    std::vector<std::string>           a_indices;       ///< Input index names (e.g., {"i","j"}).
    std::shared_ptr<ElementwiseParams> params;          ///< Live scalars the executor reads each call.
};

/**
 * @brief Metadata for the DENSE element-wise binary kinds,
 *        @ref OpKind::DirectProduct and @ref OpKind::DirectDivision.
 *
 * Both compute ``C = alpha * (A op B) + beta * C`` over three operands of one
 * shape, differing only in the per-element operation, which @ref Node::kind
 * already names. One descriptor for the two rather than a near-duplicate pair,
 * for the same reason @ref GroupedElementwiseDescriptor serves three grouped
 * kinds: there is nothing to record beyond the scalars, and a second type
 * would only be a second thing to keep in step.
 *
 * The cost of sharing is the usual one: a pass that probes
 * ``get_if<ElementwiseBinaryDescriptor>`` without also checking
 * @ref Node::kind cannot tell a product from a quotient. Check the kind.
 *
 * The TILED direct division is a different node despite sharing the kind: it
 * carries a @ref TiledElementwiseDescriptor, because its operands have no
 * single contiguous buffer. @ref build_executor rejects it, and
 * @ref Graph::serializability_report names it.
 */
struct ElementwiseBinaryDescriptor {
    PrefactorScalar                    alpha{double{1}}; ///< Prefactor on the A-op-B product (at-capture snapshot).
    PrefactorScalar                    beta{double{0}};  ///< Prefactor on the destination (0 = overwrite).
    std::shared_ptr<ElementwiseParams> params;           ///< Live scalars the executor reads each call.
};

/**
 * @name Live scalar accessors
 *
 * Every prefactor-bearing descriptor keeps TWO records of one scalar: the at-capture SNAPSHOT on
 * the descriptor itself, and - when the node carries a shared params block - the value the
 * executor actually reads on every call. A pass that folds a scale writes the LIVE one, so a
 * reader that wants to know what the next ``graph.execute()`` will compute must prefer it, and
 * fall back to the snapshot only for a node that has no block: one a pass assembled by hand, or
 * one a loader rebuilt.
 *
 * That rule was written out by hand wherever it was needed - twenty-odd ternaries across the IR
 * writer, the expression raiser, the GPU dispatch and three passes - and getting it wrong is
 * invisible in the worst way, because the value read is merely STALE rather than absent. These
 * are the rule, in one place.
 *
 * The inverse question - "what did capture record" - is still the plain member, and a few callers
 * genuinely want that one: the IR writer's cache keys and anything comparing a node against the
 * program as written. Reach for the member deliberately, not by forgetting these exist.
 *
 * @note @ref PermuteDescriptor deliberately has no accessor here. Its snapshots are
 *       @c std::complex<double> while its params block holds @ref PrefactorScalar, so the two
 *       halves do not share a return type - and @ref GraphIR encodes them differently depending
 *       on which one it took. Unifying that is a change to the saved form, not a refactor.
 * @{
 */

/// @brief The destination prefactor an einsum will actually apply.
/// @param[in] desc The descriptor to read.
/// @return The live value, or the snapshot when the node carries no params block.
[[nodiscard]] inline PrefactorScalar const &live_c_prefactor(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->c_pf : desc.c_prefactor;
}

/// @brief The product prefactor an einsum will actually apply.
/// @param[in] desc The descriptor to read.
/// @return The live value, or the snapshot when the node carries no params block.
[[nodiscard]] inline PrefactorScalar const &live_ab_prefactor(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->ab_pf : desc.ab_prefactor;
}

/// @brief Whether an einsum will actually conjugate its first operand.
/// @param[in] desc The descriptor to read.
/// @return The live flag, or the snapshot when the node carries no params block.
[[nodiscard]] inline bool live_conj_a(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->conj_a : desc.conj_a;
}

/// @brief Whether an einsum will actually conjugate its second operand.
/// @param[in] desc The descriptor to read.
/// @return The live flag, or the snapshot when the node carries no params block.
[[nodiscard]] inline bool live_conj_b(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->conj_b : desc.conj_b;
}

/// @brief The source prefactor an axpby will actually apply.
/// @param[in] desc The descriptor to read.
/// @return The live value, or the snapshot when the node carries no params block.
[[nodiscard]] inline PrefactorScalar const &live_alpha(AxpbyDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->alpha : desc.alpha;
}

/// @brief The destination prefactor an axpby will actually apply.
/// @param[in] desc The descriptor to read.
/// @return The live value, or the snapshot when the node carries no params block.
[[nodiscard]] inline PrefactorScalar const &live_beta(AxpbyDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->beta : desc.beta;
}

/// @brief The source prefactor a direct product / division will actually apply.
/// @param[in] desc The descriptor to read.
/// @return The live value, or the snapshot when the node carries no params block.
[[nodiscard]] inline PrefactorScalar const &live_alpha(ElementwiseBinaryDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->alpha : desc.alpha;
}

/// @brief The destination prefactor a direct product / division will actually apply.
/// @param[in] desc The descriptor to read.
/// @return The live value, or the snapshot when the node carries no params block.
[[nodiscard]] inline PrefactorScalar const &live_beta(ElementwiseBinaryDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->beta : desc.beta;
}

/// @brief The factor a scale will actually apply.
/// @param[in] desc The descriptor to read.
/// @return The live value, or the snapshot when the node carries no params block.
///
/// A scale uses @ref ElementwiseParams::alpha only; its ``beta`` is meaningless for an in-place
/// multiply and is never read.
[[nodiscard]] inline PrefactorScalar const &live_factor(ScaleDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->alpha : desc.factor;
}

/// @}

/**
 * @brief Metadata for the DENSE @ref OpKind::Dot nodes.
 *
 * A dot reduces two same-shaped operands to one scalar. There is exactly one
 * choice to record, and @ref conjugated is it: ``dot`` sums @f$A_i B_i@f$ while
 * ``dotc`` sums @f$\overline{A_i} B_i@f$, which for a complex operand are
 * different numbers and for a real one are the same. Neither form carries a
 * prefactor -- there is no scalar to scale, since the destination is written
 * rather than accumulated into -- so recording one would invent an operation
 * the kernel does not perform.
 *
 * The TILED dot is a different node despite sharing the kind, exactly as the
 * tiled direct division is: it carries a @ref TiledDotDescriptor, whose
 * per-tile reduction has no builder entry. Check @ref Node::kind AND the
 * alternative, or ask @ref reconstruction_blocker.
 */
struct DotDescriptor {
    bool conjugated{false}; ///< true for ``dotc`` (sum conj(A)*B), false for the bilinear ``dot``.
};

/**
 * @brief Metadata for the DENSE @ref OpKind::Trace nodes: ``result = sum_i A(i,i)``.
 *
 * Deliberately EMPTY, and present anyway. A trace is fully described by its
 * kind, dtype and operand ids -- there is no prefactor, no index list and no
 * variant of the operation to choose between -- so the type records nothing.
 *
 * What it buys is the per-node verdict. @ref OpKind::Trace also carries the
 * TILED trace, which reduces over a grid rather than over one buffer and has no
 * builder entry; without a descriptor to hold, "dense or tiled" would not be a
 * question @ref reconstruction_blocker could answer from the node alone. The
 * empty type is that answer, and it is the same argument @ref build_executor
 * makes in reverse for @ref OpKind::Transpose, where the alternative that would
 * be recorded already exists and would say something false.
 */
struct TraceDescriptor {};

/**
 * @brief Metadata for @ref OpKind::Gemm nodes: ``C = alpha*op(A)*op(B) + beta*C``.
 *
 * @ref trans_a and @ref trans_b are the BLAS transpose characters -- ``'n'``,
 * ``'t'`` or ``'c'``, lower or upper case -- rather than the pair of bools two
 * of the three capture overloads take, because the third overload accepts
 * ``Transpose::C`` and a bool cannot spell a conjugate transpose. Recording
 * chars loses nothing and lets one descriptor serve all three.
 *
 * The prefactors are SNAPSHOTS, with no live params block beside them, which is
 * the one place this descriptor departs from @ref AxpbyDescriptor and
 * @ref ElementwiseBinaryDescriptor. Nothing rewrites a gemm's prefactors today:
 * ``ScaleAbsorption::fold_site`` folds into Einsum and Axpby only, and
 * @ref Graph::update_prefactors refuses any node that is not an einsum. A
 * shared block exists to be written between replays, and an unwritten one is
 * only a second place for the same number to live. A pass that wants to fold
 * into a gemm adds the block then, and the rule of ``ExecutorBuilder.hpp``
 * applies until it does: rewrite these fields and call @ref build_executor
 * again.
 */
struct GemmDescriptor {
    PrefactorScalar alpha{double{1}}; ///< Prefactor on the matrix product (at-capture snapshot).
    PrefactorScalar beta{double{0}};  ///< Prefactor on the destination (0 = overwrite).
    char            trans_a{'n'};     ///< BLAS transpose character for A: 'n', 't' or 'c'.
    char            trans_b{'n'};     ///< BLAS transpose character for B: 'n', 't' or 'c'.
};

/**
 * @brief Metadata for the DENSE @ref OpKind::Syev nodes: the real symmetric
 *        eigendecomposition @f$A = V \operatorname{diag}(W) V^T@f$, in place.
 *
 * One field, and it is the only part of the capture site that was not already
 * data. ``cg::syev`` takes ``ComputeEigenvectors`` as a TEMPLATE parameter, so
 * the choice between LAPACK's eigenvalues-only job and its full one lived in the
 * type of the baked lambda and nowhere in the node. A file cannot carry a
 * template argument, and a loader guessing it would not merely pick a slower
 * kernel: the eigenvalues-only job leaves ``A`` holding the tridiagonal
 * reduction's scratch rather than the eigenvectors, so every consumer of ``A``
 * downstream would read a different matrix.
 *
 * Nothing else needs recording. The operand roles are fixed by the operation
 * (``A`` is an input AND an output because it is decomposed in place, ``W`` is
 * the second output), the triangle the kernel reads is fixed, and there is no
 * prefactor to fold, so kind, dtype and operand ids carry the rest.
 *
 * The TILED syev shares the kind and records NO descriptor, which is the same
 * split @ref TraceDescriptor describes: it diagonalizes each diagonal block of a
 * block-diagonal operator and has no builder entry, so a missing alternative
 * here is what a per-block decomposition looks like rather than a dropped field.
 * Ask @ref reconstruction_blocker for the per-node answer.
 *
 * @versionadded{2.0.0}
 */
struct SyevDescriptor {
    /// Whether the eigenvectors are written into ``A`` (LAPACK ``jobz='v'``) or
    /// only the eigenvalues are computed (``jobz='n'``).
    bool compute_eigenvectors{true};
};

/**
 * @brief Metadata for @ref OpKind::GroupedDot nodes.
 *
 * A grouped dot is a run of independent `result_i = sum(A_i * B_i)` reductions
 * issued as one node. Unlike @ref GroupedBatchedGemmDescriptor there is nothing
 * to group ON: the entries are not sorted, merged or reshaped, so the only thing
 * a pass or a report needs from the node beyond its operand lists is how many
 * entries it holds.
 *
 * The parent @ref Node carries the inputs interleaved A_0, B_0, A_1, B_1, ...
 * and the outputs in entry order, so entry `i` indexes both.
 */
struct GroupedDotDescriptor {
    int total{0}; ///< How many dot products the node holds.
};

/**
 * @brief Metadata for @ref OpKind::GroupedAxpby nodes.
 *
 * The prefactors are PER ENTRY, which is the whole difference from
 * @ref AxpbyDescriptor: a grouped call exists to merge accumulations that
 * disagree on their coefficients. Snapshots only, and there is deliberately no
 * shared-params handle: ScaleAbsorption and friends fold a scale into a single
 * axpby by rewriting one live scalar, and doing that to one entry of a run would
 * need the pass to know which entry, which no pass does yet.
 */
struct GroupedAxpbyDescriptor {
    int                          total{0}; ///< How many axpby operations the node holds.
    std::vector<PrefactorScalar> alphas;   ///< Per-entry alpha, in entry order.
    std::vector<PrefactorScalar> betas;    ///< Per-entry beta, in entry order.
};

/**
 * @brief Metadata for the grouped element-wise kinds: @ref
 * OpKind::GroupedPermute, @ref OpKind::GroupedDirectProduct and @ref
 * OpKind::GroupedDirectDivision.
 *
 * One type for the three because the cost model prices them the same way -
 * every member touches its operands once, so the node's serial time is the
 * traffic the parent @ref Node's operand lists already describe. What is
 * recorded here is what those lists cannot carry: how many members there are
 * and the per-member prefactors, so a pass can read the node instead of
 * re-deriving it from the closure.
 *
 * Deliberately distinct from @ref GroupedAxpbyDescriptor even though the two
 * hold the same fields. A pass that probes ``get_if<GroupedAxpbyDescriptor>``
 * without checking the node kind would otherwise find a permute and reason
 * about it as an AXPBY over one buffer.
 *
 * The parent @ref Node carries the inputs interleaved per member and the
 * outputs in member order.
 */
struct GroupedElementwiseDescriptor {
    int                          total{0}; ///< How many members the node holds.
    std::vector<PrefactorScalar> alphas;   ///< Per-member prefactor on the sources.
    std::vector<PrefactorScalar> betas;    ///< Per-member prefactor on the destination.
};

/**
 * @brief Metadata for @ref OpKind::GroupedSandwich nodes.
 *
 * A grouped sandwich is a run of independent
 * ``C_i += sum_q B_q S_i B_q^T`` accumulations with the dressed slice
 * ``B_q = A_i[q] - P_i^T M_i[q]`` built in cache per auxiliary slice, one
 * entry per (pair) member. The per-member extents are recorded so the cost
 * model can price the node as the batched arithmetic it is - the same lesson
 * @ref GroupedBatchedGemmDescriptor carries - without re-deriving them from
 * the operand lists.
 *
 * The parent @ref Node carries the inputs interleaved
 * A_0, M_0, P_0, S_0, A_1, ... and the outputs in entry order.
 */
struct GroupedSandwichDescriptor {
    int                       total{0}; ///< How many sandwich accumulations the node holds.
    std::vector<std::int64_t> nq;       ///< Per-entry auxiliary extent.
    std::vector<std::int64_t> nk;       ///< Per-entry dressing (LMO) extent.
    std::vector<std::int64_t> na;       ///< Per-entry PNO extent.
};

/**
 * @brief Metadata for @ref OpKind::GroupedGatherRotate nodes.
 *
 * A grouped gather-rotate is a run of independent
 * ``C_i[q, a, b] = sum_uv src[Q_i[q], U_i[u], U_i[v]] X_i[u, a] X_i[v, b]``
 * blocks, one entry per (triplet, pair) member, with the gathered block
 * existing only one q tile at a time.
 *
 * The per-member extents are recorded so the cost model can price the node
 * without re-deriving them from the index lists, which the node does not
 * carry. Unlike its sibling kinds this one is not arithmetic alone: it also
 * streams ``nq * nu * nu`` elements out of the shared parent, and that traffic
 * is the reason the node exists, so @ref elem_bytes is here to let the model
 * charge for it.
 *
 * The parent @ref Node carries the shared source first and then the per-entry
 * transforms as inputs, and the destinations in entry order as outputs.
 */
struct GroupedGatherRotateDescriptor {
    int                       total{0};      ///< How many gather-rotate blocks the node holds.
    std::vector<std::int64_t> nq;            ///< Per-entry auxiliary extent.
    std::vector<std::int64_t> nu;            ///< Per-entry source (PAO) extent.
    std::vector<std::int64_t> nt;            ///< Per-entry rotated (TNO/PNO) extent.
    std::int64_t              elem_bytes{0}; ///< Size of one element, for the streaming term.
};

/**
 * @brief Metadata for a TILED einsum node.
 *
 * A tiled contraction records as ``OpKind::Custom`` and executes per tile through
 * ``detail::tiled_runtime_einsum``. This carries the live state that executor
 * reads, so a pass can inspect and rewrite the operation instead of treating the
 * node as an opaque closure.
 *
 * Deliberately a DISTINCT type from @ref EinsumDescriptor rather than a reuse of
 * it. A whole-tiled contraction is not a dense one -- its operands have no single
 * contiguous buffer -- and several passes probe ``get_if<EinsumDescriptor>``
 * without first checking the node kind, so handing the tiled node an
 * EinsumDescriptor would expose it to passes that assume one buffer per tensor.
 * With its own type it stays invisible to all of them, and only a pass that
 * explicitly asks for a tiled einsum finds it.
 *
 * The operand and output TensorIds are on the node itself; the element type comes
 * from the tensor handles.
 */
struct TiledEinsumDescriptor {
    /// Live index lists, shared with the executor: a pass that rewrites these
    /// changes what the next ``graph.execute()`` contracts.
    std::shared_ptr<EinsumIndices> indices;
    /// Live prefactors, shared with the executor. Same contract as
    /// @ref EinsumDescriptor::params.
    std::shared_ptr<EinsumParams> params;
};

/**
 * @brief Metadata for a TILED permute node (``OpKind::Custom``).
 *
 * A tiled ``C = beta*C + alpha*P(A)`` records as Custom and executes through
 * ``detail::tiled_permute``; this descriptor is what lets TiledExpansion lower
 * it into per-tile dense Permute nodes instead of treating the node as an
 * opaque closure (which strands every tensor it touches out of expansion).
 *
 * A distinct type from @ref PermuteDescriptor for the same reason the tiled
 * einsum has its own: passes probe ``get_if<PermuteDescriptor>`` without
 * checking the node kind and then reason about a single dense buffer, which a
 * tiled operand does not have. The scalars are snapshots matching the baked
 * executor, exactly as the dense permute capture stores them.
 */
struct TiledPermuteDescriptor {
    std::vector<std::string> c_indices;        ///< Output index names
    std::vector<std::string> a_indices;        ///< Input index names
    PrefactorScalar          alpha{double{1}}; ///< Source prefactor
    PrefactorScalar          beta{double{0}};  ///< Destination prefactor (0 = overwrite)
};

/**
 * @brief Metadata for a TILED dot / dotc node (``OpKind::Dot``).
 *
 * A tiled scalar reduction sums per-tile dense dots over the operands' shared
 * tiles into a dense 1-element result. Without a descriptor the node is an
 * opaque closure whose whole-tensor tiled reads strand every tensor it
 * touches out of TiledExpansion; with it, the pass can lower the node into a
 * single reduction over PER-TILE ids, freeing the whole-tensor ids entirely.
 */
struct TiledDotDescriptor {
    bool conjugated{false}; ///< true for dotc (sum conj(A)*B), false for dot
};

/// Which elementwise operation a @ref TiledElementwiseDescriptor describes.
enum class TiledElementwiseOp : std::uint8_t {
    Scale,  ///< ``A = alpha * A``
    Axpy,   ///< ``Y = Y + alpha * X``
    Divide, ///< ``C = alpha * (A / B) + beta * C``
};

/// @copydoc ElementwiseParams
/// Spelled for a tiled elementwise node. Only @ref TiledElementwiseOp::Divide reads
/// @ref ElementwiseParams::beta; Scale and Axpy leave it at its default.
using TiledElementwiseParams = ElementwiseParams;

/**
 * @brief Metadata for a TILED elementwise node (scale or axpy).
 *
 * Like @ref TiledEinsumDescriptor, these record as ``OpKind::Custom`` and run
 * per tile, so without a descriptor the operation and its scalar are sealed
 * inside a closure. Carrying them lets TiledExpansion lower the node into one
 * dense op per populated tile.
 *
 * A distinct type from @ref ScaleDescriptor and @ref AxpbyDescriptor for the
 * same reason the tiled einsum has its own: passes probe ``get_if<...>`` for
 * those without checking the node kind, and a tiled operand has no single
 * buffer for them to work on.
 *
 * The operand TensorIds are on the node itself. A Scale reads and writes one
 * tensor, listed in both @ref Node::inputs and @ref Node::outputs; an Axpy
 * reads X and Y and writes Y, so Y appears in both lists too. A Divide reads A
 * and B, writes C, and additionally reads C when ``beta != 0``.
 */
struct TiledElementwiseDescriptor {
    TiledElementwiseOp op{TiledElementwiseOp::Scale};
    /// Live scalar, shared with the executor: a pass that rewrites this changes
    /// what the next ``graph.execute()`` computes.
    std::shared_ptr<TiledElementwiseParams> params;
};

/**
 * @brief Metadata for conditional (if-then-else) nodes.
 *
 * Contains a predicate and two subgraphs. The predicate is evaluated at
 * execution time; if true, the then_branch executes, otherwise else_branch.
 *
 * The predicate is a @ref PredExpr rather than a bare ``std::function``, which
 * is what lets a conditional be data: a literal, a comparison over
 * @ref ParamTable entries, or one slot of a @ref GateFlags array all rebuild
 * from the descriptor, and only the callback arm does not. See
 * @ref reconstruction_blocker, which reports that arm per node.
 *
 * @code
 * auto [then_graph, else_graph] = graph.add_conditional("check", [&]() {
 *     return energy_diff < threshold;
 * });
 * @endcode
 */
struct ConditionalDescriptor {
    PredExpr               predicate;   ///< Evaluated at runtime to select branch
    std::shared_ptr<Graph> then_branch; ///< Executed if the predicate is true
    std::shared_ptr<Graph> else_branch; ///< Executed if the predicate is false (may be empty)
};

/**
 * @brief Live state a @ref LoopDescriptor shares with its executor.
 *
 * Separate from the descriptor for the reason ``ExecutorBuilder.hpp`` states
 * once: nodes live in a ``std::vector`` that passes insert into, erase from and
 * reorder, so an executor cannot hold a pointer into one. Everything a replay
 * WRITES therefore lives in a shared block the executor holds by
 * ``shared_ptr``, exactly as @ref EinsumParams and @ref AxpbyParams do for
 * prefactors.
 */
struct LoopState {
    /// How many iterations the most recent execute() ran.
    size_t last_iteration_count{0};
};

/**
 * @brief Metadata for loop nodes.
 *
 * Contains a body subgraph executed repeatedly until the condition is false or
 * max_iterations is reached. The condition is evaluated AFTER each iteration
 * and can inspect tensor values for convergence checking.
 *
 * @code
 * auto &body = graph.add_loop("converge", 100, [&](size_t iter) {
 *     return std::abs(energy - energy_old) > 1e-8;
 * });
 * @endcode
 */
struct LoopDescriptor {
    std::shared_ptr<Graph> body;                 ///< Subgraph to execute each iteration
    size_t                 max_iterations{1000}; ///< Safety limit

    /// After each iteration: true = continue, false = stop. A default-constructed
    /// @ref PredExpr is an unconditional true, which is what an absent condition
    /// has always meant: run to @ref max_iterations.
    PredExpr condition;

    /// Live loop state, shared with the executor. Null on a node some pass
    /// assembled by hand; @ref build_executor then synthesizes a private block,
    /// so the node still runs and simply reports no iteration count.
    std::shared_ptr<LoopState> state;

    /// @brief How many iterations the most recent execute() ran.
    /// @return The count, or 0 when the node has never run or carries no live state.
    ///
    /// This used to be a plain field, and the executor wrote it into its own
    /// COPY of the descriptor, so it was never observable on the node at all.
    [[nodiscard]] size_t last_iteration_count() const noexcept { return state ? state->last_iteration_count : 0; }
};

/**
 * @brief Live state a @ref SetupDescriptor shares with its executor.
 *
 * Held in a shared block for the reason @ref LoopState states: an executor cannot hold a
 * pointer into the node vector that passes insert into and reorder.
 *
 * The two keys are what makes the setup body skippable across a re-bind rather than only
 * across a replay. @ref pending_key is what the CALLER says the current problem is, written
 * by @ref Graph::set_setup_key; @ref computed_key is the problem the body last actually ran
 * for. A bind clears @ref computed, and the executor then reruns unless the two keys agree
 * and are non-empty, which is the caller stating that this is the same problem and the
 * factors on hand are still its factors.
 *
 * An empty @ref pending_key disables the cache, which is the default: refitting is always
 * correct, and a graph whose caller has said nothing about problem identity gets the
 * behavior that cannot be wrong.
 */
struct SetupState {
    /// Whether the body has run since the last @ref Graph::invalidate_setup.
    bool computed{false};

    /// The problem the body last ran for, or empty when it has never run or ran with no key.
    std::string computed_key;

    /// The problem the caller says is bound now, or empty when the caller has not said.
    std::string pending_key;
};

/**
 * @brief Metadata for setup nodes: a body computed once per bound problem.
 *
 * A setup body holds the computation that depends only on what a caller binds and not on
 * how many times the graph is replayed: a fitting, a metric inverse, an integral transform.
 * Structurally it is a sub-graph exactly as a loop body is, and everything that walks,
 * expands, or refuses a sub-graph reaches it through @ref is_control_flow. What differs is
 * when it runs: the executor skips the body entirely once it has computed, so the node
 * costs one boolean per replay, and a bind is what puts it back to work.
 *
 * Its outputs therefore live ACROSS replays and must not be freed between them, which
 * @ref passes::FreeInsertion is told about rather than left to infer.
 *
 * @see Graph::add_setup
 * @see SetupState
 */
struct SetupDescriptor {
    std::shared_ptr<Graph>      body;  ///< The subgraph computed once per bound problem.
    std::shared_ptr<SetupState> state; ///< Live state shared with the executor.

    /// @brief Whether the body has run since the last bind.
    /// @return True when the outputs on hand are current; false on a node carrying no state.
    [[nodiscard]] bool computed() const noexcept { return state != nullptr && state->computed; }
};

/**
 * @brief Per-axis specification for a @c View op.
 *
 * Each axis of the parent tensor maps to one of:
 * - Full: keep the entire axis (the result preserves this dimension).
 * - Range: half-open ``[lo, hi)`` slice; the result keeps this dimension
 *               with extent ``hi - lo``.
 * - Drop: pick a single index; the result loses this dimension
 *               (rank-reducing). Honored by the runtime-rank ``view_runtime``;
 *               the typed ``cg::view`` throws on it.
 *
 * Bounds are @ref BoundExpr values, so they may be compile-time constants
 * (``0``, ``5``), references to a Pipeline parameter (``"n_occ"``), or
 * arbitrary callbacks (interpreted-mode only).
 */
struct ViewAxis {
    enum class Kind : std::uint8_t { Full, Range, Drop };
    Kind      kind{Kind::Full};
    BoundExpr lo; ///< Range/Drop start (Drop reads only this).
    BoundExpr hi; ///< Range exclusive end. Unused for Full / Drop.

    static ViewAxis full() { return ViewAxis{.kind = Kind::Full}; }
    static ViewAxis range(BoundExpr lo, BoundExpr hi) { return ViewAxis{.kind = Kind::Range, .lo = std::move(lo), .hi = std::move(hi)}; }
    static ViewAxis drop(BoundExpr i) { return ViewAxis{.kind = Kind::Drop, .lo = std::move(i)}; }
};

/**
 * @brief Metadata for @c View nodes, non-owning slice/alias of another tensor.
 *
 * The output tensor's ``TensorHandle::aliases`` is set to ``parent_id``.
 * Each iteration the executor resolves the per-axis @ref BoundExpr values
 * against the active Pipeline's @ref ParamTable and rebuilds the underlying
 * Einsums TensorView, rebinding the output handle's data/strides/dims.
 */
struct ViewDescriptor {
    TensorId              parent_id{0};   ///< The tensor being sliced.
    std::vector<ViewAxis> axes;           ///< One entry per parent-tensor axis.
    size_t                result_rank{0}; ///< parent.rank - count(Drop). Cached for passes.
    /// Axis permutation: result axis ``i`` reads parent axis ``permutation[i]``
    /// (and ``axes[i]`` slices that parent axis). Empty == identity (no
    /// transpose). Used to express ``.T`` / transpose-via-view as a
    /// graph-registered, parent-aliasing view.
    std::vector<size_t> permutation;
};

/**
 * @brief The stored C++ type of a @c WriteParam node's source scalar.
 *
 * @c write_param accepts any arithmetic scalar and narrows it to the
 * @c std::int64_t a @ref ParamTable holds, so an executor rebuilt from data has
 * to be told the storage it is reading. One enumerator per arithmetic type
 * rather than a (size, signedness) pair, because a saved graph writes the NAME
 * and a name that spells the type is the one a reader can check against the
 * tensor it binds.
 */
enum class ParamSourceType : std::uint8_t {
    Bool,       ///< @c bool
    Char,       ///< @c char, whose signedness is implementation-defined
    SChar,      ///< @c signed char
    UChar,      ///< @c unsigned char
    Short,      ///< @c short
    UShort,     ///< @c unsigned short
    Int,        ///< @c int
    UInt,       ///< @c unsigned int
    Long,       ///< @c long
    ULong,      ///< @c unsigned long
    LongLong,   ///< @c long long
    ULongLong,  ///< @c unsigned long long
    Float,      ///< @c float
    Double,     ///< @c double
    LongDouble, ///< @c long double
    /// The width-named spellings the graph itself uses. @c std::int64_t is one
    /// of @c long / @c long long depending on the platform, so this is an alias
    /// for whichever of those it is rather than a fifteenth distinct type.
    /// Spelling it as either one outright is wrong on half the world: glibc's
    /// LP64 @c int64_t is @c long, libc++'s is @c long long, and a hardcoded
    /// alias disagrees with @ref param_source_type there.
    Int64 = std::is_same_v<std::int64_t, long> ? Long : LongLong,
};

/**
 * @brief The name of a @ref ParamSourceType, for diagnostics and the saved form.
 * @param[in] type The stored type to name.
 * @return Its C++ spelling ("bool", "char", "signed char", ... "long double").
 *
 * The spelling is the TYPE's own, which is the point of naming these at all: a
 * reader can check the name against the tensor it is about to bind, where a
 * numeric enumerator would only say "the fourteenth one".
 *
 * @ref ParamSourceType::Int64 is an alias for whichever of @c long /
 * @c long long is 64 bits on this platform, so it shares that one's name rather
 * than getting a spelling of its own; naming it separately would make the file
 * say something platform-dependent about a type that is not.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::string_view param_source_type_name(ParamSourceType type) noexcept {
    switch (type) {
    case ParamSourceType::Bool:
        return "bool";
    case ParamSourceType::Char:
        return "char";
    case ParamSourceType::SChar:
        return "signed char";
    case ParamSourceType::UChar:
        return "unsigned char";
    case ParamSourceType::Short:
        return "short";
    case ParamSourceType::UShort:
        return "unsigned short";
    case ParamSourceType::Int:
        return "int";
    case ParamSourceType::UInt:
        return "unsigned int";
    case ParamSourceType::Long:
        return "long";
    case ParamSourceType::ULong:
        return "unsigned long";
    case ParamSourceType::LongLong:
        return "long long";
    case ParamSourceType::ULongLong:
        return "unsigned long long";
    case ParamSourceType::Float:
        return "float";
    case ParamSourceType::Double:
        return "double";
    case ParamSourceType::LongDouble:
        return "long double";
    }
    return "int";
}

/**
 * @brief The @ref ParamSourceType spelled @p name, if there is one.
 * @param[in] name A spelling @ref param_source_type_name produces.
 * @return The enumerator, or an empty optional when nothing is spelled that way.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::optional<ParamSourceType> param_source_type_from_name(std::string_view name) noexcept {
    for (auto const type : {ParamSourceType::Bool, ParamSourceType::Char, ParamSourceType::SChar, ParamSourceType::UChar,
                            ParamSourceType::Short, ParamSourceType::UShort, ParamSourceType::Int, ParamSourceType::UInt,
                            ParamSourceType::Long, ParamSourceType::ULong, ParamSourceType::LongLong, ParamSourceType::ULongLong,
                            ParamSourceType::Float, ParamSourceType::Double, ParamSourceType::LongDouble}) {
        if (param_source_type_name(type) == name) {
            return type;
        }
    }
    return std::nullopt;
}

/**
 * @brief The @ref ParamSourceType naming @p T.
 * @tparam T An arithmetic type.
 * @return Its enumerator.
 */
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr ParamSourceType param_source_type() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, bool>) {
        return ParamSourceType::Bool;
    } else if constexpr (std::is_same_v<U, char>) {
        return ParamSourceType::Char;
    } else if constexpr (std::is_same_v<U, signed char>) {
        return ParamSourceType::SChar;
    } else if constexpr (std::is_same_v<U, unsigned char>) {
        return ParamSourceType::UChar;
    } else if constexpr (std::is_same_v<U, short>) {
        return ParamSourceType::Short;
    } else if constexpr (std::is_same_v<U, unsigned short>) {
        return ParamSourceType::UShort;
    } else if constexpr (std::is_same_v<U, int>) {
        return ParamSourceType::Int;
    } else if constexpr (std::is_same_v<U, unsigned int>) {
        return ParamSourceType::UInt;
    } else if constexpr (std::is_same_v<U, long>) {
        return ParamSourceType::Long;
    } else if constexpr (std::is_same_v<U, unsigned long>) {
        return ParamSourceType::ULong;
    } else if constexpr (std::is_same_v<U, long long>) {
        return ParamSourceType::LongLong;
    } else if constexpr (std::is_same_v<U, unsigned long long>) {
        return ParamSourceType::ULongLong;
    } else if constexpr (std::is_same_v<U, float>) {
        return ParamSourceType::Float;
    } else if constexpr (std::is_same_v<U, double>) {
        return ParamSourceType::Double;
    } else {
        static_assert(std::is_same_v<U, long double>, "param_source_type: unhandled arithmetic type");
        return ParamSourceType::LongDouble;
    }
}

/// The alias and the mapping have to agree, or a descriptor written by
/// @ref param_source_type never compares equal to the alias a caller wrote.
/// The two are derived from the same platform question, so this only fires if
/// one of them stops asking it.
static_assert(param_source_type<std::int64_t>() == ParamSourceType::Int64,
              "ParamSourceType::Int64 must alias whichever of long / long long is std::int64_t here");

/**
 * @brief Metadata for @c WriteParam nodes, explicit dataflow write into a Pipeline parameter.
 *
 * Reads a scalar tensor's value (or evaluates a callback) and stores the
 * result into ``params[name]``. Makes the parameter dependency visible to
 * the scheduler so subsequent @c View nodes that reference the same
 * parameter are correctly ordered.
 */
struct WriteParamDescriptor {
    std::string name; ///< Parameter name to write.

    /// Scalar tensor to read (0 when using @ref source_expr).
    ///
    /// A structural record, not the operand the executor reads: execution
    /// resolves the source through the node's own @ref Node::inputs, which is
    /// what every pass rewrites. The two agree at capture and the dataflow list
    /// is authoritative if they ever disagree.
    TensorId source_id{0};

    /// The C++ type of the scalar at @ref source_id.
    ///
    /// @ref TensorHandle::dtype cannot answer this: it names the four BLAS
    /// element types and reports ``Unknown`` for every integral one, and a loop
    /// bound written through @c write_param is an integer far more often than
    /// it is a double. A rebuilt executor has to know the width and the
    /// signedness of the storage it reads before it can narrow the value to the
    /// @c std::int64_t a @ref ParamTable holds.
    ///
    /// Meaningless on the callback arm, where there is no storage to decode.
    ParamSourceType source_type{ParamSourceType::Int64};

    /// Optional: compute the value from an expression rather than from a tensor.
    ///
    /// Unset means the tensor arm above. Set means the value comes from a
    /// @ref BoundExpr, which is a literal, a named @ref ParamTable entry, or a
    /// callback. Only the callback arm blocks a save, and
    /// @ref reconstruction_blocker inspects the arm to say so per node --
    /// @ref OpKind::WriteParam is a reconstructible KIND either way.
    ///
    /// An ``optional`` rather than a bare @ref BoundExpr because a default
    /// ``BoundExpr`` is the literal 0, which is a legitimate value and
    /// therefore cannot double as "no expression here".
    std::optional<BoundExpr> source_expr;
};

namespace detail {

/// Fill an EinsumDescriptor's snapshot fields from a parsed spec. Does NOT set
/// the live @c params / @c indices handles or the gemm hint; callers that need a
/// self-contained node should use @ref Graph::make_einsum_node instead of
/// assembling those by hand.
inline EinsumDescriptor build_einsum_descriptor(ParsedEinsumSpec const &parsed, PrefactorScalar c_pf, PrefactorScalar ab_pf,
                                                bool conj_a = false, bool conj_b = false) {
    EinsumDescriptor desc;
    desc.c_prefactor         = c_pf;
    desc.ab_prefactor        = ab_pf;
    desc.conj_a              = conj_a;
    desc.conj_b              = conj_b;
    desc.spec.c_indices      = parsed.c_indices;
    desc.spec.a_indices      = parsed.a_indices;
    desc.spec.b_indices      = parsed.b_indices;
    desc.spec.link_indices   = parsed.link_indices();
    desc.spec.target_indices = parsed.target_indices();
    desc.spec.all_indices    = desc.spec.target_indices;
    desc.spec.all_indices.insert(desc.spec.all_indices.end(), desc.spec.link_indices.begin(), desc.spec.link_indices.end());
    return desc;
}

/**
 * @brief Name a space for a diagnostic, without trusting the id.
 * @param[in] registry Registry the id is expected to come from. May be null.
 * @param[in] id The id to name.
 * @return The registered name, or a ``#<value>`` placeholder when the id cannot be resolved.
 *
 * An id that does not resolve is a caller error the conflict message still has to be able to
 * print, so this never throws and never leaves the reader with nothing to go on.
 */
[[nodiscard]] inline std::string space_label(SpaceRegistry const *registry, SpaceId id) {
    if (registry != nullptr && id.valid() && id.value() < registry->size()) {
        return registry->space(id).name;
    }
    return "#" + std::to_string(id.value());
}

/**
 * @brief One operand's contribution to a contraction's letter-to-space map.
 *
 * A plain pair of pointers rather than a copy: @ref build_letter_spaces reads both vectors and
 * keeps neither, and a contraction has exactly three of these.
 */
struct LetterSpaceOperand {
    char const                     *label{"A"};       ///< Operand name used in diagnostics.
    std::vector<std::string> const *indices{nullptr}; ///< This operand's index letters, in slot order.
    std::vector<SpaceId> const     *spaces{nullptr};  ///< This operand's per-slot annotation, possibly empty.
};

/**
 * @brief Resolve the index letters of one contraction against its operands' slot annotations.
 *
 * Walks A, then B, then C, binding each letter to the space annotated on the slot it occupies.
 * The rules are the strict ones of the design's section 1.3, and inheritance (writing a resolved
 * space back onto an unannotated slot) is deliberately NOT one of them: that is a propagation
 * pass's job, and doing it here would spread one wrong annotation silently.
 *
 * - A letter meeting only unannotated slots gets no entry at all.
 * - A letter meeting one annotated and one unannotated slot takes the annotated side, for THIS
 *   node's map only.
 * - A letter meeting two DIFFERENT spaces is a capture-time error, whether the two slots sit on
 *   different operands or are the two slots of a repeated (diagonal) letter within one operand.
 *
 * @param[in] operands The operands to walk, in the order their names should appear in a
 *            diagnostic.
 * @param[in] registry Registry the annotated ids belong to, used only to name spaces in the
 *            error message. May be null.
 * @param[in] context Caller name to prefix a diagnostic with, e.g. ``"cg::einsum"``.
 * @return The letter-to-space bindings, sorted by letter. Empty when no operand is annotated.
 * @throws std::invalid_argument when one letter binds two different spaces.
 *
 * An operand whose annotation is shorter than its index list (which a well-formed graph never
 * produces, since annotation is validated against the tensor's rank) contributes only the slots
 * it covers.
 */
[[nodiscard]] inline std::vector<std::pair<std::string, SpaceId>>
build_letter_spaces(std::span<LetterSpaceOperand const> operands, SpaceRegistry const *registry, std::string_view context) {
    std::vector<std::pair<std::string, SpaceId>> bound;
    std::vector<char const *>                    origin; // parallel to `bound`: operand each entry came from

    for (auto const &operand : operands) {
        if (operand.indices == nullptr || operand.spaces == nullptr || operand.spaces->empty()) {
            continue;
        }
        std::size_t const slots = std::min(operand.indices->size(), operand.spaces->size());
        for (std::size_t slot = 0; slot < slots; ++slot) {
            SpaceId const id = (*operand.spaces)[slot];
            if (!id.valid()) {
                continue; // a partially annotated tensor: this axis simply says nothing
            }
            std::string const &letter = (*operand.indices)[slot];

            auto const existing = std::ranges::find_if(bound, [&letter](auto const &e) { return e.first == letter; });
            if (existing == bound.end()) {
                bound.emplace_back(letter, id);
                origin.push_back(operand.label);
                continue;
            }
            if (existing->second != id) {
                std::size_t const at = static_cast<std::size_t>(existing - bound.begin());
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "{}: index letter '{}' binds space '{}' on operand {} and space '{}' on operand {} within one "
                                        "contraction; a letter ranges over exactly one space per contraction",
                                        context, letter, space_label(registry, existing->second), origin[at], space_label(registry, id),
                                        operand.label);
            }
        }
    }

    std::ranges::sort(bound, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
    return bound;
}

/**
 * @brief Derive an output tensor's per-slot annotation from a contraction's letter map.
 * @param[in] c_indices The output operand's index letters, in slot order.
 * @param[in] letter_spaces The contraction's letter-to-space bindings.
 * @return One space per output slot, or an EMPTY vector when any output letter is unbound.
 *
 * All-or-nothing on purpose: a half-filled annotation would be indistinguishable from a
 * deliberately partial one, and the point of inferring here is that the result is as trustworthy
 * as a declaration.
 */
[[nodiscard]] inline std::vector<SpaceId> spaces_from_letters(std::vector<std::string> const                     &c_indices,
                                                              std::vector<std::pair<std::string, SpaceId>> const &letter_spaces) {
    if (c_indices.empty() || letter_spaces.empty()) {
        return {};
    }
    std::vector<SpaceId> out;
    out.reserve(c_indices.size());
    for (auto const &letter : c_indices) {
        auto const found = std::ranges::find_if(letter_spaces, [&letter](auto const &e) { return e.first == letter; });
        if (found == letter_spaces.end()) {
            return {};
        }
        out.push_back(found->second);
    }
    return out;
}

} // namespace detail

/**
 * @brief Type-erased operation metadata variant.
 *
 * Each Node stores an OpData that may contain operation-specific metadata
 * for use by optimization passes. Nodes with no special metadata use
 * std::monostate.
 */
/// New alternatives are APPENDED, never inserted. The variant index is not a
/// serialized property today - the round-trip IR writes
/// descriptors by NAME, precisely so adding one in the middle cannot silently
/// reinterpret an old file - but keeping the order append-only costs nothing
/// and keeps every debug dump that prints ``op_data.index()`` comparable
/// across builds.
using OpData =
    std::variant<std::monostate, EinsumDescriptor, ScaleDescriptor, PermuteDescriptor, ConditionalDescriptor, LoopDescriptor,
                 AllocDescriptor, TransferDescriptor, DiskIODescriptor, CommDescriptor, InitializeDescriptor, BatchedGemmDescriptor,
                 GroupedBatchedGemmDescriptor, ViewDescriptor, WriteParamDescriptor, AxpbyDescriptor, GroupedDotDescriptor,
                 GroupedAxpbyDescriptor, GroupedElementwiseDescriptor, GroupedSandwichDescriptor, GroupedGatherRotateDescriptor,
                 TiledEinsumDescriptor, TiledElementwiseDescriptor, TiledPermuteDescriptor, TiledDotDescriptor, ElementwiseBinaryDescriptor,
                 DotDescriptor, TraceDescriptor, GemmDescriptor, ElementTransformDescriptor, SetupDescriptor, SyevDescriptor>;

/**
 * @brief A single operation node in the computation graph.
 *
 * Each node represents one captured operation (einsum, scale, gemm, etc.).
 * It contains:
 * - A type-erased executor lambda that performs the actual computation
 * - Input/output tensor IDs expressing data dependencies
 * - Operation metadata for optimization passes
 *
 * Nodes are created automatically by the graph-aware operation wrappers
 * in the einsums::compute_graph namespace during capture.
 *
 * @see Graph::add_node()
 * @see CaptureContext::record()
 */
struct Node {
    NodeId id{0};                ///< Unique identifier assigned by Graph::add_node()
    OpKind kind{OpKind::Custom}; ///< Operation type for pattern matching
    Target target{Target::CPU};  ///< Execution target (set by GPUPlacement pass)

    /// @brief Threads this node's kernel is to execute with; 0 means unplanned
    ///        and is executed exactly as width 1.
    ///
    /// @ref DataflowExecutor guarantees a node with a width above 1 sees exactly
    /// that many threads in @c omp_get_max_threads(), for the duration of the
    /// node and on whichever thread runs it. The vendor BLAS sees the width
    /// only where the vendor supports per-caller widths (MKL); an
    /// OpenMP-threaded OpenBLAS does not, so vendor calls made under the width
    /// are clamped back to one thread (see @ref blas::set_moldable_width_scope)
    /// and the planner never assigns such a node a width to begin with
    /// (@ref blas_route_is_moldable).
    /// @ref SequentialExecutor and @ref OpenMPExecutor ignore the field, so a
    /// graph carrying widths behaves exactly as an unplanned one under them.
    ///
    /// A width is a tuning artifact of one machine and one process: it is chosen
    /// against the thread count the plan was made for and is deliberately never
    /// serialized with the graph. Sits in the padding after @c target rather
    /// than beside the other scheduling metadata so it costs the node nothing.
    std::uint16_t thread_width{0};

    /// @brief Admission urgency: the longest remaining path from this node to a
    ///        sink, in estimated NANOSECONDS under the planned widths.
    ///
    /// @ref DataflowExecutor admits ready nodes in decreasing order of this,
    /// ties broken by node position. 0 means unplanned, and then the executor
    /// derives a structural hop count instead, which is the same ordering
    /// without the times.
    ///
    /// Nanoseconds rather than a double because the budget's ordering key is an
    /// integer and an exact comparison is what makes admission order a function
    /// of the graph rather than of rounding. Written by the ThreadPlanning pass
    /// alongside @ref thread_width, and never serialized for the same reason.
    std::int64_t admission_priority{0};

    std::string label; ///< Human-readable label for profiling and debugging

    /**
     * @brief Type-erased executor that performs the captured operation.
     *
     * This lambda captures the fully-resolved template call at capture time.
     * All template parameters (types, ranks, indices) are baked into the lambda.
     * On execution, it simply calls the captured function, no re-dispatch needed.
     *
     * @warning The lambda captures tensor references. All referenced tensors must
     *          outlive the graph. Use Graph::create_tensor() for intermediates.
     */
    std::function<void()> execute;

    /**
     * @brief CPU fallback executor for GPU nodes.
     *
     * When a GPU node's execute() throws, the runtime can fall back to this
     * lambda which performs the same operation on the CPU. Set automatically
     * by GPUPlacement when it promotes a node: the original execute is saved
     * as cpu_fallback before the executor is replaced with a GPU version.
     *
     * Empty for CPU nodes and transfer nodes.
     */
    std::function<void()> cpu_fallback;

    /**
     * @brief Asynchronous start phase for I/O nodes.
     *
     * When set, the DataflowExecutor calls async_start to begin an
     * asynchronous operation (e.g., initiate a disk read) as soon as
     * predecessors complete. Independent compute nodes can then overlap
     * with the I/O. The async_finish lambda is called before any consumer
     * of this node runs.
     *
     * Empty for non-async nodes. SequentialExecutor and OpenMPExecutor
     * ignore this field and call the synchronous execute lambda instead.
     */
    std::function<void()> async_start;

    /**
     * @brief Asynchronous finish/synchronize phase for I/O nodes.
     *
     * When set, waits for the operation started by async_start to complete.
     * Called by the DataflowExecutor before any consumer of this node runs.
     *
     * Empty for non-async nodes.
     */
    std::function<void()> async_finish;

    std::vector<TensorId> inputs;  ///< TensorIds of tensors read by this operation
    std::vector<TensorId> outputs; ///< TensorIds of tensors written by this operation

    /**
     * @brief Operation-specific metadata for optimization passes.
     *
     * Contains EinsumDescriptor, ScaleDescriptor, PermuteDescriptor, or
     * std::monostate for operations without special metadata.
     */
    OpData op_data;

    size_t estimated_flops{0}; ///< Estimated floating-point operations (for cost modeling)
    size_t estimated_bytes{0}; ///< Estimated memory traffic in bytes (for cost modeling)

    /// Stream assignment for async execution (set by StreamAssignment pass).
    /// 0 = default/compute stream, 1 = transfer stream.
    int stream_id{0};
};

/// @brief Names of @ref ParamTable entries this node WRITES - the parameter
///        analogue of @ref Node::outputs.
///
/// A @c WriteParam node's entire effect is this write. It travels through the
/// parameter table rather than through any tensor, so a TensorId-keyed
/// dependency scan cannot see it, and in the callback form the node carries no
/// tensor operands at all - which reads to an unguarded pass as "no inputs,
/// therefore unconditionally movable". Schedulers pair this with
/// @ref param_reads to order parameter writes against their consumers.
[[nodiscard]] inline std::vector<std::string> param_writes(Node const &node) {
    if (auto const *wd = std::get_if<WriteParamDescriptor>(&node.op_data)) {
        return {wd->name};
    }
    return {};
}

/// @brief Names of @ref ParamTable entries this node READS - the analogue of
///        @ref Node::inputs.
///
/// A @c View whose slice bounds name a parameter re-resolves them every time it
/// executes, so it must stay ordered after whatever writes them. The same is
/// true of a conditional or a loop whose predicate compares parameters, and of
/// a @c WriteParam whose source is itself a named parameter.
///
/// Callback-valued bounds and callback predicates are deliberately NOT reported
/// here: they name nothing, so no edge can be derived. Use
/// @ref has_runtime_view_bounds to ask the weaker question "does this slice move
/// at all", which is what hoisting and folding need. A @ref PredExpr::FlagTest
/// names nothing either, by construction: its array rides outside the dataflow
/// exactly as @ref LuPivots does.
[[nodiscard]] inline std::vector<std::string> param_reads(Node const &node) {
    std::vector<std::string> names;
    auto const               add = [&names](BoundExpr const &bound) {
        if (bound.is_param()) {
            names.push_back(bound.param_name());
        }
    };

    if (auto const *vd = std::get_if<ViewDescriptor>(&node.op_data)) {
        for (auto const &ax : vd->axes) {
            add(ax.lo);
            if (ax.kind == ViewAxis::Kind::Range) {
                add(ax.hi);
            }
        }
    } else if (auto const *cd = std::get_if<ConditionalDescriptor>(&node.op_data)) {
        cd->predicate.collect_param_names(names);
    } else if (auto const *ld = std::get_if<LoopDescriptor>(&node.op_data)) {
        ld->condition.collect_param_names(names);
    } else if (auto const *wd = std::get_if<WriteParamDescriptor>(&node.op_data)) {
        if (wd->source_expr.has_value()) {
            add(*wd->source_expr);
        }
    }
    return names;
}

/// @brief True when a @c View's slice is resolved from runtime state (a named
///        parameter or a callback) rather than from literals.
///
/// Such a view aliases the same parent every iteration but describes a
/// different slice each time. Passes that would evaluate a node once and reuse
/// the result - constant folding, loop-invariant hoisting - must refuse it: the
/// parent input they inspect is genuinely invariant, and the part that moves is
/// not expressed as dataflow at all.
[[nodiscard]] inline bool has_runtime_view_bounds(Node const &node) {
    if (node.kind != OpKind::View) {
        return false;
    }
    auto const *vd = std::get_if<ViewDescriptor>(&node.op_data);
    if (vd == nullptr) {
        return true; // no descriptor to inspect: assume the slice moves
    }
    return std::ranges::any_of(
        vd->axes, [](ViewAxis const &ax) { return !ax.lo.is_const() || (ax.kind == ViewAxis::Kind::Range && !ax.hi.is_const()); });
}

/**
 * @brief Visit the graphs one node CONTAINS: a loop's body, a conditional's two branches,
 *        a setup's body.
 *
 * @tparam NodeT ``Node`` or ``Node const``; a const node hands the visitor ``Graph const &``.
 * @tparam F Invocable taking that graph reference.
 * @param[in] node The node to look inside. A node holding any other descriptor has no child
 *            graph and is visited zero times.
 * @param[in] visit Called once per child graph that exists.
 * @param[in] include_setup False to skip a @c Setup body. For the one caller whose walk order
 *            has to agree with a component that descends into loops and conditionals only;
 *            see @ref Graph::finish_replay_thread_plan.
 *
 * Which descriptors carry a sub-graph, and in what order, was written out at seven sites
 * across three files. The order is part of the contract rather than a detail: the graph-IR
 * writer interns a control-flow node's boundary tensors BY POSITION, so a walk that visited
 * the else-branch first would place them in the dense order at indices nothing mentions them
 * at. Empty bodies are skipped, so a visitor never sees a null graph.
 *
 * Visits one level. A visitor that wants the whole subtree recurses itself, which is what
 * @ref Graph::for_each_subgraph is: this over every node of a graph.
 */
template <typename NodeT, typename F>
    requires std::is_same_v<std::remove_cv_t<NodeT>, Node>
void for_each_child_graph(NodeT &node, F &&visit, bool include_setup = true) {
    using GraphRef  = std::conditional_t<std::is_const_v<NodeT>, Graph const, Graph> &;
    auto const step = [&visit](std::shared_ptr<Graph> const &child) {
        if (child) {
            visit(static_cast<GraphRef>(*child));
        }
    };

    if (auto *loop = std::get_if<LoopDescriptor>(&node.op_data); loop != nullptr) {
        step(loop->body);
    } else if (auto *cond = std::get_if<ConditionalDescriptor>(&node.op_data); cond != nullptr) {
        step(cond->then_branch);
        step(cond->else_branch);
    } else if (include_setup) {
        if (auto *setup = std::get_if<SetupDescriptor>(&node.op_data); setup != nullptr) {
            step(setup->body);
        }
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
