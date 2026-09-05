//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file ThcFactorization.hpp
 * @brief The second factorization provider: a four-index tensor as a five-factor grid chain.
 *
 * @par What it is, in the vocabulary of the layer it lives in
 * Given a collocation matrix @c X[m,P], the values of the @c m basis functions at the @c P grid
 * points, and a three-index tensor @c B[A,m,n] over the same basis, this offers
 *
 * @f[ T[m,n,p,q] \approx \sum_{PQ} X[m,P]\,X[n,P]\,Z[P,Q]\,X[p,Q]\,X[q,Q] @f]
 *
 * with @c Z fitted by least squares from @c B. Five factors over two new letters, which is what
 * a plan naming a factor LIST exists for: it cannot be written as a split of two.
 *
 * @par Per-axis collocation, and why the matrices are handed to the PROVIDER
 * A tagged tensor's axes need not all run over the whole basis. An @f$(ia|jb)@f$ integral and a
 * doubles amplitude @f$t[i,a,j,b]@f$ are both blocks of it, one axis occupied and the next
 * virtual, and a provider fitting over the whole basis on every axis can propose nothing for
 * either. So a caller may hand over one collocation matrix per axis, and the fit becomes
 *
 * @f[ T[m,n,p,q] \approx \sum_{PQ} X_0[m,P]\,X_1[n,P]\,Z[P,Q]\,X_0[p,Q]\,X_1[q,Q] @f]
 *
 * with @f$S[P,Q] = (\sum_m X_0[m,P] X_0[m,Q])(\sum_n X_1[n,P] X_1[n,Q])@f$ and
 * @f$\tilde{B}[A,P] = \sum_{mn} B[A,m,n] X_0[m,P] X_1[n,P]@f$, from a three-index tensor over
 * the same two blocks. One collocation for every axis is the case where @f$X_0@f$ and @f$X_1@f$
 * are one matrix, and the algebra above is then the Hadamard square this always fitted.
 *
 * The two PAIRS must present the same blocks, axis 0 with axis 2 and axis 1 with axis 3, which
 * is what @f$(ia|jb)@f$ and @f$t[i,a,j,b]@f$ both are. A @f$[i,j,a,b]@f$ layout pairs occupied
 * with occupied and virtual with virtual, so its two halves have different metrics and different
 * three-index tensors; that needs two fits rather than one and is declined with the reason.
 *
 * The matrices are given to the provider rather than named by the tag, and the choice is about
 * what a SAVED graph can do. A tag saying "this axis is the occupied block" would leave the
 * provider to slice the caller's whole-basis collocation, and a slice is a view: a provenance
 * tag does not cross one, and the block boundary would be frozen into the emitted structure at
 * optimize time, so a graph rebound at a geometry with a different occupation could not follow
 * it. Handed over, each matrix becomes an interface tensor of the transformed graph under its
 * own name, which is the same shape @ref MetricFitFactorization has for its three-index tensor
 * and @c passes::LaplaceTransform has for its orbital energies, and it is bound by name at every
 * later bind.
 *
 * @par What chemistry calls it
 * Tensor hypercontraction, and this class registered on the tag @c "eri" is the ``AutoTHC`` the
 * design asks for. The name stays generic for the reason @ref MetricFitFactorization's does:
 * nothing in the class knows what an integral is, and the chemistry lives in the registration.
 * The two are ALTERNATIVES on one tag rather than a composition, and the registry's ranking by
 * profitability is what chooses between them per tagged tensor.
 *
 * @par The fit, and why it is fitted FROM the density fit
 * The least-squares solution of @f$B[A,m,n] \approx \sum_P C[A,P] X[m,P] X[n,P]@f$ is
 * @f$C = \tilde{B} S^{-1}@f$ with
 *
 * @f[ S[P,Q] = \left(\sum_m X[m,P] X[m,Q]\right)^2, \qquad
 *     \tilde{B}[A,P] = \sum_{mn} B[A,m,n] X[m,P] X[n,P] @f]
 *
 * the square being ELEMENTWISE, and the four-index fit follows as
 * @f$Z = S^{-1} \tilde{B}^{T} \tilde{B} S^{-1}@f$. Fitting from the three-index tensor rather
 * than from the exact integrals is what keeps this validatable offline against the same fixture
 * @ref MetricFitFactorization was: the exact four-index quantity comes from an integral engine
 * this library does not have.
 *
 * @c S is the Hadamard square of a Gram matrix and therefore positive semi-definite, so its
 * inverse goes through the same guarded route the metric fit uses, twice: @c inv_sqrt_or_zero
 * over the eigenvalues gives @f$S^{-1/2}@f$ and the square of that is @f$S^{-1}@f$. The drop
 * threshold is the caller's for the reason it is there, and reaches the node rather than the
 * process.
 *
 * @par The grid is a space, and its extent is symbolic
 * The factors carry an index space named by @ref grid_space_name, whose @c dim_symbol is
 * @ref grid_dim_symbol. That is what makes a saved THC graph replayed at a new geometry a
 * REBIND rather than a re-capture, and it is what makes the factorization pass's numeric veto
 * abstain: a grid is chosen per problem, so the number of points a capture happened to have is
 * a placeholder rather than a size anything runs at. Register the space with
 * @ref register_grid_space before applying the pass; a graph whose registry does not hold it
 * gets factors with no space and a weaker cost comparison rather than a wrong one.
 *
 * @par The accuracy statement
 * The bound is asserted, from @c einsums:graph:thc-epsilon or from the constructor, because an
 * @ref ApproximationRecord is written once at optimize time. What is MEASURED, per bind and
 * exactly as @ref MetricFitFactorization measures its dropped directions, is the least-squares
 * residual of the fit against the three-index tensor it was fitted from: two parameters, named
 * by @ref residual_param_name and @ref reference_param_name, whose ratio's square root is the
 * relative residual. The four-index error is not measurable without the tensor the fit exists
 * to avoid forming; this one is, and it costs one contraction more than the fit already does.
 *
 * @par Choosing the grid is the CALLER's, and it cannot be anything else
 * A raw Becke grid is not a grid this fits on. At water/cc-pVDZ psi4's grid is 5241 points
 * against a basis-pair space of rank at most @f$n(n+1)/2@f$ in the basis size, so handing the
 * raw grid over makes @c S singular in five thousand directions and @c Z a matrix hundreds of
 * times larger than its own rank. What works is the pivoted selection interpolative separable
 * density fitting uses: take the point whose basis-pair product has the largest remaining
 * norm, project it out, repeat. It stops on its own, at 280 points for that molecule, which
 * is the grid saying it has no more independent directions to give.
 *
 * So the point count is a property of the basis and the geometry rather than of an accuracy
 * target, which is why the grid extent is a SYMBOL a bind resolves rather than a number. The
 * approximation record does not tell a caller whether their grid was enough; the measured
 * residual does, and it errs on the safe side: at water/cc-pVDZ it reads @c 4e-4 against an
 * energy error of @c 2e-5.
 *
 * @par The drop threshold, and why the inherited default is wrong here
 * @c S runs continuously from about @c 5e-14 to @c 3 with no null space, where a Coulomb metric
 * has a clear one. An absolute cutoff at @ref default_drop_threshold therefore keeps near-null
 * directions whose inverse amplifies rounding into the fit, and two implementations of the same
 * formula then disagree by more than the fit's own error, because which side of the cutoff an
 * eigenvalue lands on decides the answer. At @c 1e-8 they agree to five digits and the fit is
 * at its most accurate on that grid. The number that ought to be compared against is a fraction
 * of the largest eigenvalue, and a relative cutoff needs a reduction feeding a value into a
 * guard whose policy number is bound when its executor is built; that is the mechanism
 * @ref MetricFitFactorization already records as one to add, met here on a real system. Until
 * it exists, a caller states a threshold suited to their grid rather than taking the default.
 *
 * @par Grid letters and a following quadrature do not compete
 * A tensor this fits may also be the numerator a @ref passes::LaplaceTransform rewrites, and
 * the chain contracts to @c sum_Q @c L[Q,i,a] @c R[Q,j,b] with the grid index where an
 * auxiliary index would be, so the transform rides on it unchanged. There is no sharing
 * question between quadrature terms, because there are no terms: the quadrature index is an
 * ordinary contracted letter of one contraction rather than a loop over points, so the emitted
 * form carries the grid letter exactly once however many points the rule takes.
 *
 * @see Factorization.hpp for what a provider is
 * @see MetricFitFactorization.hpp for the provider this is an alternative to
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class SpaceRegistry;

/**
 * @brief Offer a four-index tensor as a fitted chain over two grid indices.
 *
 * @see ThcFactorization.hpp for the mathematics and for what chemistry calls it
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ThcFactorization : public FactorizationProvider {
  public:
    /// @brief The drop threshold a caller who states none gets.
    ///
    /// The same number @ref MetricFitFactorization uses, and INHERITED rather than chosen for a
    /// grid: it is the magnitude a Coulomb metric's null space sits below, and a collocation
    /// metric has no null space to sit below. See the file note on why a caller with a real
    /// grid states a threshold of their own instead.
    static constexpr double default_drop_threshold = 1.0e-10;

    /**
     * @brief Construct the provider.
     *
     * @param[in] tag The provenance tag this claims, e.g. ``"eri"``.
     * @param[in] three_index @c B[A,m,n], the density-fitted three-index tensor the grid fit is
     *            fitted FROM. Must outlive the provider and every graph it factorizes.
     * @param[in] collocation @c X[m,P], the basis functions' values at the grid points. Must
     *            outlive the same, and becomes an interface tensor of the transformed graph.
     * @param[in] bound The relative error the caller asserts this fit has, or zero to take
     *            ``einsums:graph:thc-epsilon``.
     * @param[in] drop_threshold Grid directions whose eigenvalue of @c S does not exceed this
     *            are dropped from the fit; see @ref default_drop_threshold.
     * @param[in] name The provider's name, which the approximation record and the report carry.
     * @throws std::invalid_argument When @p drop_threshold is negative or not finite, or when
     *         @p bound is negative. A negative threshold reads as "keep more than everything";
     *         the kernel floors it at zero anyway, and running at a threshold the caller did
     *         not ask for is the failure the parameter exists to prevent.
     */
    ThcFactorization(std::string tag, RuntimeTensorView<double> three_index, RuntimeTensorView<double> collocation, double bound = 0.0,
                     double drop_threshold = default_drop_threshold, std::string name = "Thc");

    /**
     * @brief The primary constructor: one collocation matrix per axis, or one for every axis.
     *
     * @param[in] tag The provenance tag this claims.
     * @param[in] three_index @c B[A,m,n] over the blocks axes 0 and 1 run over.
     * @param[in] collocations The per-axis matrices. Every other constructor forwards to this one
     *            with a list of one.
     * @param[in] bound The relative error the caller asserts, or zero to take the option.
     * @param[in] drop_threshold See @ref default_drop_threshold.
     * @param[in] name The provider's name.
     * @throws std::invalid_argument When @p collocations is empty, or on the numeric grounds the
     *         other constructors refuse.
     */
    ThcFactorization(std::string tag, RuntimeTensorView<double> three_index, std::vector<RuntimeTensorView<double>> collocations,
                     double bound = 0.0, double drop_threshold = default_drop_threshold, std::string name = "Thc");

    /// @brief The same, taking runtime-rank tensors. This is the overload Python gets.
    ///
    /// @note The drop-threshold default is spelled with its class qualifier because the binding
    ///       generator copies a default argument through verbatim, and an unqualified
    ///       class-scope constant does not resolve where the generated code spells it.
    APIARY_EXPOSE ThcFactorization(std::string tag, RuntimeTensor<double> const &three_index, RuntimeTensor<double> const &collocation,
                                   double bound = 0.0, double drop_threshold = ThcFactorization::default_drop_threshold,
                                   std::string name = "Thc");

    /**
     * @brief The same, with one collocation matrix per axis of the tagged tensor.
     *
     * @param[in] tag The provenance tag this claims.
     * @param[in] three_index @c B[A,m,n] over the two blocks axes 0 and 1 run over.
     * @param[in] collocations One matrix per axis, or one matrix for every axis. Each must
     *            outlive the provider, and each becomes an interface tensor of the transformed
     *            graph under its own name.
     * @param[in] bound As above.
     * @param[in] drop_threshold As above.
     * @param[in] name As above.
     * @throws std::invalid_argument When @p collocations is empty, or on the same numeric
     *         grounds the other constructors refuse.
     *
     * Which axes may differ is checked in @ref propose rather than here, because it is a
     * statement about the tagged tensor and this constructor has not met one yet.
     */
    APIARY_EXPOSE ThcFactorization(std::string tag, RuntimeTensor<double> const &three_index,
                                   std::vector<RuntimeTensor<double> const *> collocations, double bound = 0.0,
                                   double drop_threshold = ThcFactorization::default_drop_threshold, std::string name = "Thc");

    /// @brief The same, taking compile-time-rank tensors.
    ///
    /// Kept as its own overload rather than left to the implicit conversion, because that
    /// conversion does not carry the NAME: a view built from a ``Tensor<double, N>`` is
    /// anonymous, the fitting captures it as an interface tensor, and a manifest binds by name.
    ThcFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &collocation, double bound = 0.0,
                     double drop_threshold = default_drop_threshold, std::string name = "Thc");

    /**
     * @brief Fit the TAGGED tensor itself, for an amplitude a solver rewrites every iteration.
     *
     * @param[in] tag The provenance tag this claims, e.g. ``"amplitude"``.
     * @param[in] amplitude The very tensor the caller tags. Checked against the tagged handle's
     *            buffer in @ref propose, so a caller who hands over a different tensor gets a
     *            decline rather than a fit of something else.
     * @param[in] collocations One matrix per axis, or one for every axis.
     * @param[in] bound The relative error asserted, or zero to take ``einsums:graph:thc-epsilon``.
     * @param[in] drop_threshold See @ref default_drop_threshold.
     * @param[in] name The provider's name.
     * @return The provider.
     *
     * @par Why this is a different provider rather than a different tag
     * A density fit reads a three-index tensor the caller handed over and merely CLAIMS it
     * approximates the tagged four-index one; the factors are the same however often that tensor
     * moves. This one projects the tagged tensor onto the grid basis,
     * @f$Z = S^{-1} (X^T T X) S^{-1}@f$, so the factors are a function of the tensor and are
     * re-fitted wherever it is written. That is what @c FactorizationPlan::fits_from_tagged says,
     * and it is what lets a tensor a loop body updates be factorized at all.
     *
     * @par The residual is measured without forming the fit
     * @f$\lVert T - \tilde{T}\rVert^2 = \lVert T\rVert^2 - 2\langle Z, P\rangle +
     * \langle Z, S Z S\rangle@f$ with @f$P = X^T T X@f$, every term of which is a grid-sized
     * contraction the fit already computes or a dot of the tagged tensor with itself. So the
     * accuracy statement is a real measurement here, where the integral fit's is an assertion,
     * and the record names the tensor it lands in.
     */
    static std::shared_ptr<ThcFactorization> for_amplitude(std::string tag, RuntimeTensorView<double> amplitude,
                                                           std::vector<RuntimeTensorView<double>> collocations, double bound = 0.0,
                                                           double      drop_threshold = default_drop_threshold,
                                                           std::string name           = "ThcAmplitude");

    /// @brief The same, taking runtime-rank tensors. This is the overload Python gets.
    /// @param[in] tag The provenance tag this claims.
    /// @param[in] amplitude The very tensor the caller tags.
    /// @param[in] collocations One matrix per axis, or one for every axis.
    /// @param[in] bound The relative error asserted, or zero to take the option.
    /// @param[in] drop_threshold See @ref default_drop_threshold.
    /// @param[in] name The provider's name.
    /// @return The provider.
    APIARY_EXPOSE static std::shared_ptr<ThcFactorization>
    for_amplitude(std::string tag, RuntimeTensor<double> const &amplitude, std::vector<RuntimeTensor<double> const *> collocations,
                  double bound = 0.0, double drop_threshold = ThcFactorization::default_drop_threshold, std::string name = "ThcAmplitude");

    /**
     * @brief Fit the three-index tensor of a density fit, rather than the four-index tensor.
     *
     * @param[in] tag The provenance tag this claims, e.g. ``"eri"``.
     * @param[in] three_index @c B[A,m,n], which is also the tensor the caller tags. Checked
     *            against the tagged handle's buffer in @ref propose.
     * @param[in] collocations One matrix per BASIS axis, so two for a rank-3 tensor, or one for
     *            both. Axis 0 is the auxiliary axis and has no collocation.
     * @param[in] bound The relative error asserted, or zero to take ``einsums:graph:thc-epsilon``.
     * @param[in] drop_threshold See @ref default_drop_threshold.
     * @param[in] name The provider's name.
     * @return The provider.
     *
     * @par Why the three-index tensor rather than the four-index one
     * A density-fitted program never forms @f$(ab|ef)@f$, so a provider that only claims a
     * rank-4 tensor has nothing to offer it: what the program holds is @c B and what it writes
     * is a chain over the auxiliary index. The fit is the one the four-index fit already
     * computes on its way to @c Z, stopped one step earlier,
     *
     * @f[ B[A,m,n] \approx \sum_P C[A,P]\,X_0[m,P]\,X_1[n,P],
     *     \qquad C = \tilde{B} S^{-1} @f]
     *
     * so it is three factors over one grid letter where the four-index form is five over two.
     * It is also the smaller object: @c C is auxiliary-by-grid where @c Z is grid-by-grid, and
     * the two occurrences of @c B in a ladder term get one fitting between them.
     *
     * @par Its accuracy statement is measured
     * The exact quantity is the tagged tensor and the fit reads it, so the residual is a number
     * rather than a claim, exactly as it is for an amplitude fit. That is also what
     * @c FactorizationPlan::fits_from_tagged says of it.
     */
    static std::shared_ptr<ThcFactorization> for_three_index(std::string tag, RuntimeTensorView<double> three_index,
                                                             std::vector<RuntimeTensorView<double>> collocations, double bound = 0.0,
                                                             double      drop_threshold = default_drop_threshold,
                                                             std::string name           = "ThcThreeIndex");

    /// @brief The same, taking runtime-rank tensors. This is the overload Python gets.
    /// @param[in] tag The provenance tag this claims.
    /// @param[in] three_index The very tensor the caller tags.
    /// @param[in] collocations One matrix per basis axis, or one for both.
    /// @param[in] bound The relative error asserted, or zero to take the option.
    /// @param[in] drop_threshold See @ref default_drop_threshold.
    /// @param[in] name The provider's name.
    /// @return The provider.
    APIARY_EXPOSE static std::shared_ptr<ThcFactorization> for_three_index(std::string tag, RuntimeTensor<double> const &three_index,
                                                                           std::vector<RuntimeTensor<double> const *> collocations,
                                                                           double                                     bound = 0.0,
                                                                           double drop_threshold = ThcFactorization::default_drop_threshold,
                                                                           std::string name      = "ThcThreeIndex");

    /**
     * @brief Where an amplitude fit writes what it was worth on this bind.
     *
     * @param[in] residual A rank-1 tensor the fit writes @f$\lVert T - \tilde{T}\rVert^2@f$ into.
     * @param[in] reference A rank-1 tensor the fit writes @f$\lVert T\rVert^2@f$ into.
     *
     * The caller's own tensors, for the reason the energies of a Laplace transform are: a
     * measurement a caller cannot read is a measurement nobody makes, and a value the graph
     * declares for itself is one the graph keeps to itself. The record points at @p residual by
     * name, so a caller holding a record knows which of their tensors to look in.
     *
     * Without this an amplitude fit emits no measurement and its record asserts its bound, which
     * is what the integral fit does in every case.
     */
    void report_residual_into(RuntimeTensorView<double> residual, RuntimeTensorView<double> reference);

    /// @brief The same, taking runtime-rank tensors. This is the overload Python gets.
    /// @param[in] residual A rank-1 tensor for the squared residual.
    /// @param[in] reference A rank-1 tensor for the squared norm of what was fitted.
    APIARY_EXPOSE void report_residual_into(RuntimeTensor<double> const &residual, RuntimeTensor<double> const &reference);

    /// @brief The same, taking compile-time-rank tensors.
    ///
    /// Its own overload rather than left to the implicit conversion, for the reason the
    /// constructors' is: a view built from a @c Tensor is anonymous, and the record names the
    /// destination by NAME.
    /// @param[in] residual A rank-1 tensor for the squared residual.
    /// @param[in] reference A rank-1 tensor for the squared norm of what was fitted.
    void report_residual_into(Tensor<double, 1> const &residual, Tensor<double, 1> const &reference);

    /// @copydoc FactorizationProvider::name
    [[nodiscard]] std::string name() const override { return _name; }

    /// @copydoc FactorizationProvider::tag
    [[nodiscard]] std::string tag() const override { return _tag; }

    /**
     * @brief Offer the grid chain, or say why this tensor is not one.
     *
     * @param[in] graph The graph holding @p tensor.
     * @param[in] tensor The tagged tensor.
     * @return The plan, or the reason there is not one: a rank other than four, a collocation
     *         list that is neither one matrix nor one per axis, two pairs that do not present
     *         the same blocks, or extents that do not match a collocation matrix's basis axis.
     */
    [[nodiscard]] expected<FactorizationPlan, std::string> propose(Graph const &graph, TensorId tensor) const override;

    /// @brief The tolerance this provider asserts, taking the constructor over the option.
    /// @return The relative bound.
    APIARY_EXPOSE APIARY_GETTER("epsilon") [[nodiscard]] double epsilon() const;

    /// @brief The threshold below which this provider drops a grid direction.
    /// @return The number handed to the constructor, or @ref default_drop_threshold.
    APIARY_EXPOSE APIARY_GETTER("drop_threshold") [[nodiscard]] double drop_threshold() const noexcept { return _drop_threshold; }

    /// @brief The index space the fitted factors' grid axes carry.
    /// @return ``"grid"``.
    APIARY_EXPOSE [[nodiscard]] static std::string grid_space_name();

    /// @brief The symbolic extent a grid axis goes by.
    /// @return ``"ngrid"``.
    APIARY_EXPOSE [[nodiscard]] static std::string grid_dim_symbol();

    /**
     * @brief Register the grid space in @p graph's own registry, if it is not there already.
     *
     * A space is a statement about a graph rather than about a provider, and @ref propose is
     * handed a const graph precisely so a provider cannot make one. So a caller registers it,
     * and this is the one call that does: idempotent for a repeated identical declaration, as
     * @ref SpaceRegistry::register_space is.
     *
     * @param[in,out] graph The graph whose registry gains the space.
     * @return The space's id in that registry.
     */
    APIARY_EXPOSE static SpaceId register_grid_space(Graph &graph);

    /**
     * @brief The parameter the squared residual of the fit is reported under.
     *
     * @param[in] provider The provider's @ref name.
     * @param[in] tensor The tagged tensor's name.
     * @return The key to read out of the graph's parameter table after an execute.
     *
     * Divided by @ref reference_param_name and square-rooted, this is the RELATIVE least-squares
     * residual of the grid fit against the three-index tensor it was fitted from. Per bind and
     * not saved, because it is a property of the grid and the integrals currently bound rather
     * than of the structure.
     */
    APIARY_EXPOSE [[nodiscard]] static std::string residual_param_name(std::string const &provider, std::string const &tensor);

    /// @brief The parameter the squared norm of the three-index tensor is reported under.
    /// @param[in] provider The provider's @ref name.
    /// @param[in] tensor The tagged tensor's name.
    /// @return The key. See @ref residual_param_name for what the two are for.
    APIARY_EXPOSE [[nodiscard]] static std::string reference_param_name(std::string const &provider, std::string const &tensor);

  private:
    std::string               _tag;
    std::string               _name;
    RuntimeTensorView<double> _three_index;
    /// Optional rather than an empty view, because assigning a @ref RuntimeTensorView COPIES
    /// what it names rather than rebinding, and a rank-4 view assigned onto a rank-0 one throws.
    /// A view is constructed in place or not at all.
    std::optional<RuntimeTensorView<double>> _amplitude;
    std::optional<RuntimeTensorView<double>> _residual_report;
    std::optional<RuntimeTensorView<double>> _reference_report;
    std::vector<RuntimeTensorView<double>>   _collocations;
    /// Whether this fits the three-index tensor itself rather than a four-index tensor. The
    /// three-index tensor is in @ref _three_index either way, so the mode cannot be read off
    /// the members and is stated.
    bool   _fits_three_index{false};
    double _bound;
    double _drop_threshold;
};

EINSUMS_NAMESPACE_END(compute_graph)
