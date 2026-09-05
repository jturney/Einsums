//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file LaplaceTransform.hpp
 * @brief Replace a tagged energy denominator by a quadrature, and decouple what it multiplied.
 *
 * @par What it recognizes
 * Nothing by shape. A caller TAGS the reciprocal denominator, in the pattern
 * @ref MetricFitFactorization set with @c "eri", because a tensor named "denominator" says
 * nothing a pass can act on and a pass guessing from an elementwise kernel's name would be
 * guessing at the sign convention. The tag is @c laplace_denominator and its attributes are
 * the substance: per axis @c k of the tagged tensor, @c axis<k> names the orbital-energy
 * vector supplying that axis and @c sign<k> is @c + or @c - for whether that energy enters the
 * denominator added or subtracted. @ref denominator_tag builds one.
 *
 * The tagged tensor is the RECIPROCAL,
 * @f$D[i_0,\dots] = 1/\sum_k \sigma_k \varepsilon_k[i_k]@f$, because that is the object every
 * program forms and multiplies; a tag on the sum would name a value the graph never keeps.
 *
 * @par What it does
 * @f[ D[x] \approx \sum_j w_j \prod_k E_k[j, x_k], \qquad
 *     E_k[j, i] = e^{-\sigma_k t_j \varepsilon_k[i]} @f]
 * and the value of that is only realized when the per-axis factors are pushed onto the
 * OPERANDS of whatever the denominator multiplies. So the rewrite is a region rewrite over the
 * denominator's consumer cone, which is two statements: the contraction that forms the
 * numerator, and the direct product that applies the denominator to it. Given
 *
 * @code
 * N[x] = fA * A[u] * B[v]
 * P[x] = beta*P[x] + alpha * N[x] * D[x]
 * @endcode
 *
 * with the axes of @c x partitioned between @c u and @c v, the rewrite emits one scaling per
 * axis and one contraction:
 *
 * @code
 * At[j,u] = A[u] * E_k[j,x_k]     (once per axis of x that u carries)
 * Bt[j,v] = B[v] * E_k[j,x_k]     (once per axis of x that v carries)
 * P[x]    = beta*P[x] + alpha*fA * At[j,u] * Bt[j,v]
 * @endcode
 *
 * The quadrature index @c j is an ORDINARY contracted letter of the final contraction rather
 * than a loop over terms, which is what keeps the emitted node count independent of the point
 * count. The coupled four-index product is gone: each operand is scaled only on the axes it
 * already carried, and the sum over quadrature points is the same contraction the numerator
 * always was.
 *
 * @par What it declines, and why each is a real case
 * - A tag that does not reach the operand. Provenance does not cross a view, so a SLICED
 *   denominator is untagged and there is nothing to recognize; the tally says so rather than
 *   the pass guessing that a slice of a reciprocal is a reciprocal.
 * - A tagged tensor carrying FEWER AXES than the energies its denominator sums. The
 *   pair-driven MP2 form folds two occupied energies into a scalar prefactor, so its
 *   denominator is a two-axis object whose other two energies the graph cannot see. Carrying
 *   folded scalars as attributes would bake a bound value into structure, which is the mistake
 *   the density-fitting drop threshold already taught.
 * - A consumer that is not a direct product, or whose numerator is not a contraction inside
 *   the region. There is nothing to push the exponentials onto, and a substitution alone is
 *   more arithmetic than what it replaced rather than less.
 * - A numerator the escape analysis says is observed from outside the region. The rewrite
 *   dissolves it, and a value someone else reads cannot be dissolved.
 * - A complex denominator. A complex energy is not a thing, and the exponential integral that
 *   represents a reciprocal needs a spectral range on the real line.
 * - A tagged tensor some node WRITES. Its quadrature would go stale whenever it changed, and
 *   the setup body runs once per bound problem. This is the same refusal
 *   @ref FactorizationPass makes, for the same reason.
 *
 * @par The knob is a tolerance, not a point count
 * @c einsums:graph:laplace-epsilon states the accuracy, because a tolerance is what composes
 * with the accuracy budget and what an approximation record can state, while a point count is
 * a means. The count is derived from the tolerance and from the spectral range the bound
 * energies have, and reaches the setup node's DESCRIPTOR, so a saved graph means one thing
 * wherever it is loaded. @ref set_points overrides it, for reproducing a published number.
 *
 * @par The record is MEASURED
 * The quadrature's error is a function of one scalar variable over a bounded interval, so the
 * pass samples it and records the largest relative deviation from the exact reciprocal. That
 * is affordable here and was not for a metric fit, whose error is the difference from the very
 * tensor the fit exists to avoid forming. The setup node measures it again on every bind and
 * writes it to a tensor @ref error_tensor_name names, because the bound problem may have a
 * wider range than the one the count was chosen for.
 *
 * @see LaplaceQuadrature.hpp for the rule and its error bound
 * @see RegionRewrite.hpp for the framework
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Passes/RegionRewrite.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Substitute a quadrature for a tagged energy denominator and decouple its consumer.
 *
 * @see LaplaceTransform.hpp for the mathematics, the tag and every refusal
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT LaplaceTransform : public RegionRewrite {
  public:
    /// @brief Default-construct. Explicit so the binding codegen has a constructor to annotate.
    APIARY_EXPOSE LaplaceTransform() = default;

    /// @brief The provenance tag this pass claims.
    /// @return ``"laplace_denominator"``.
    APIARY_EXPOSE [[nodiscard]] static std::string tag_name();

    /**
     * @brief Hand the pass one of the orbital-energy vectors the tags name.
     *
     * @param[in] name The name the denominator's tag uses for this axis.
     * @param[in] vector The energies. Its storage must outlive every replay, since the setup
     *            node reads it on each bind.
     * @throws std::invalid_argument When @p vector is not a non-empty rank-1 tensor, or when
     *         a different vector is already registered under @p name.
     *
     * Supplied to the PASS rather than found in the graph, which is the shape
     * @ref MetricFitFactorization already has and for the same reason: the energies go into
     * the reciprocal before the capture starts, so no node mentions them and the graph has
     * never seen them. Registering them on the graph instead would leave it holding a handle
     * nothing uses beside the one the setup body captures, two tensors under one name, and a
     * manifest binds by name.
     *
     * The vector becomes an interface tensor of the transformed graph, under this name,
     * because the setup body captures it. That is what lets a saved graph refit its quadrature
     * at a new geometry: a bind moves the energies and the next execute rebuilds the rule.
     */
    APIARY_EXPOSE void add_energy(std::string name, RuntimeTensor<double> const &vector);

    /// @brief The same, for a single-precision problem.
    /// @param[in] name The name the denominator's tag uses for this axis.
    /// @param[in] vector The energies.
    APIARY_EXPOSE APIARY_RENAME("add_energy_f32") void add_energy(std::string name, RuntimeTensor<float> const &vector);

    /// @brief The same, taking a compile-time-rank tensor.
    /// @param[in] name The name the denominator's tag uses for this axis.
    /// @param[in] vector The energies.
    void add_energy(std::string name, Tensor<double, 1> const &vector);

    /// @brief The same, single precision.
    /// @param[in] name The name the denominator's tag uses for this axis.
    /// @param[in] vector The energies.
    void add_energy(std::string name, Tensor<float, 1> const &vector);

    /// @brief Forget every energy vector handed over so far.
    APIARY_EXPOSE void clear_energies();

    /**
     * @brief Build the tag a caller puts on a reciprocal denominator.
     *
     * @param[in] energies One orbital-energy tensor NAME per axis of the denominator, in axis
     *            order. Names rather than ids, because they are what a manifest binds by and
     *            what lets a saved graph refit at a new geometry.
     * @param[in] signs One character per axis, @c '+' or @c '-', saying whether that axis's
     *            energy enters the denominator added or subtracted.
     * @return The tag, ready for @ref Graph::annotate_tag.
     * @throws std::invalid_argument When the two lists differ in length, when either is empty,
     *         or when @p signs holds a character that is neither @c '+' nor @c '-'.
     *
     * A helper rather than a documented convention, so the pass reads the attributes out of
     * the same place a caller writes them in.
     */
    APIARY_EXPOSE [[nodiscard]] static ProvenanceTag denominator_tag(std::vector<std::string> const &energies, std::string const &signs);

    /**
     * @brief The tensor a bind's measured quadrature error is written to.
     *
     * @param[in] denominator The tagged tensor's name.
     * @return The name of a one-element tensor the setup body writes on every bind.
     *
     * A function rather than a convention, for the reason
     * @ref MetricFitFactorization::dropped_param_name is one: a caller reads the number from
     * the same place the graph writes it. The approximation record carries the error measured
     * at OPTIMIZE time, over the range the energies then had; this is what the range currently
     * bound actually achieves, and the two differ exactly when a rebind widened the interval.
     */
    APIARY_EXPOSE [[nodiscard]] static std::string error_tensor_name(std::string const &denominator);

    /// @brief The pass name.
    /// @return ``"LaplaceTransform"``.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "LaplaceTransform"; }

    /// @copydoc OptimizerPass::tier
    /// Substitutes a numerical quadrature for the reciprocal it approximates, under a tolerance
    /// it records through OptimizerPass::approximate. Never eligible for a default manager,
    /// which is what this tier means.
    [[nodiscard]] PassTier tier() const override { return PassTier::Lossy; }

    /// @brief Zero the per-apply counters.
    void reset_stats() override;

    /// @brief Apply, then emit the setup bodies the accepted rewrites asked for.
    /// @param[in,out] graph The graph to rewrite.
    /// @return True when anything was rewritten.
    bool run(Graph &graph) override;

    /**
     * @brief The target relative accuracy of the quadrature.
     *
     * @param[in] epsilon The tolerance. Must be finite and strictly between zero and one.
     * @throws std::invalid_argument When it is not.
     *
     * Overrides ``einsums:graph:laplace-epsilon`` for this pass object, which is the per-
     * pipeline half of the knob convention every rewrite knob here follows.
     */
    APIARY_EXPOSE void set_epsilon(double epsilon);

    /// @brief The tolerance this pass will use.
    /// @return The value @ref set_epsilon last took, or the option's.
    APIARY_EXPOSE APIARY_GETTER("epsilon") [[nodiscard]] double epsilon() const;

    /**
     * @brief Fix the quadrature point count rather than deriving it from the tolerance.
     *
     * @param[in] points The count, or zero to go back to deriving it. Must not be one: a
     *            trapezoidal rule needs two points or none.
     * @throws std::invalid_argument When @p points is one or negative.
     *
     * The override exists for reproducing a published number, and it is an override rather
     * than the interface because a count is a means: it says nothing about what the result is
     * worth, and the same count over a wider spectral range is a different approximation.
     */
    APIARY_EXPOSE void set_points(std::int64_t points);

    /// @brief The point-count override, or zero when the count is derived.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("points") [[nodiscard]] std::int64_t points() const { return _points; }

    /// @brief How many denominators were replaced by a quadrature on the last run.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_transformed") [[nodiscard]] std::size_t num_transformed() const { return _num_transformed; }

    /// @brief How many quadrature points the last accepted rewrite used.
    /// @return The count, or zero when nothing was rewritten.
    APIARY_EXPOSE APIARY_GETTER("last_point_count") [[nodiscard]] std::size_t last_point_count() const { return _last_points; }

    /// @brief The relative error the last accepted rewrite measured, at optimize time.
    /// @return The measured deviation, or zero when nothing was rewritten.
    APIARY_EXPOSE APIARY_GETTER("last_measured_error") [[nodiscard]] double last_measured_error() const { return _last_measured; }

    /// One energy vector the caller handed over, held type-erased.
    ///
    /// A @c shared_ptr to a view rather than the view itself, because the ADDRESS is the
    /// identity a capture registers by: the setup body has to capture the same object on
    /// every apply, or a second run of the pass would give the graph a second tensor under
    /// one name.
    ///
    /// Public because the rewrite this pass drives is also driven by `FactorizationPass`, on a
    /// trial expression, and the two share one implementation that has to name this type.
    struct EnergyVector {
        std::string             name;
        packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Float64};
        std::size_t             extent{0};
        std::shared_ptr<void>   view;
    };

    /// @brief The vector registered under @p name, or null.
    /// @param[in] name The tag's axis name.
    /// @return The entry, or null when the caller supplied none.
    [[nodiscard]] EnergyVector const *energy(std::string const &name) const;

    /// @brief The smallest and largest element of a registered vector.
    /// @param[in] held The entry to read.
    /// @return The extremes, or nothing when the vector has no data.
    [[nodiscard]] static std::optional<std::pair<double, double>> energy_extremes(EnergyVector const &held);

  protected:
    /// @copydoc RegionRewrite::rewrite
    bool rewrite(Graph &graph, Region const &region, TensorExpr &expr) override;

    /// @brief Nothing to do unless some tensor carries the tag.
    /// @param[in] graph The graph.
    /// @return True when at least one does.
    [[nodiscard]] bool applicable(Graph const &graph) const override;

    /// @brief The pass's own report lines.
    /// @return One line when anything was transformed.
    [[nodiscard]] std::vector<std::string> describe() const override;

    /// @brief A region rewrite needs the numerator and the direct product both.
    /// @return Two.
    [[nodiscard]] std::size_t min_region_nodes() const override { return 2; }

  private:
    /// A setup body an accepted rewrite still needs emitted. Held until the region loop is
    /// over, because adding a node while a region is open moves the region out from under the
    /// splice that is about to replace it.
    struct PendingSetup {
        std::string                           label;
        std::function<void(Graph &, Graph &)> emit;
    };

    /// @brief Keep one energy vector, refusing a second under the same name.
    /// @param[in] name The tag's axis name.
    /// @param[in] dtype Its element type.
    /// @param[in] extent Its length.
    /// @param[in] view The held view.
    void record_energy(std::string name, packed_gemm::ScalarType dtype, std::size_t extent, std::shared_ptr<void> view);

    std::vector<EnergyVector> _energies;

    std::vector<PendingSetup> _pending;
    std::size_t               _num_transformed{0};
    std::size_t               _last_points{0};
    double                    _last_measured{0};
    double                    _epsilon{0}; ///< Zero means "read the option".
    std::int64_t              _points{0};  ///< Zero means "derive from the tolerance".

    /// Denominators this run has already claimed, so the sweep that reports an unclaimed tag
    /// does not report one the rewrite took.
    std::vector<TensorId> _claimed;

    /// Denominators whose writer chain the rewrite verified and whose nodes it may now erase.
    /// Held until the region loop is over for the reason @ref PendingSetup is.
    std::vector<TensorId> _dissolve;
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
