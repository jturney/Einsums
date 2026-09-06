//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file NaturalAuxiliaryFactorization.hpp
 * @brief The third factorization provider, and the first that makes an index SMALLER.
 *
 * @par What it is, in the vocabulary of the layer it lives in
 * Given a three-index tensor @c B[Q,m,n], the matrix it makes against the pair index has a
 * singular value decomposition, and the auxiliary directions whose singular values are small
 * carry almost nothing. Keeping the rest offers
 *
 * @f[ B[Q,m,n] \approx \sum_{Q'} U[Q,Q']\,\tilde{B}[Q',m,n] @f]
 *
 * with @f$U@f$ the dominant eigenvectors of @f$\sum_{mn} B[P,m,n] B[Q,m,n]@f$ and @f$Q'@f$
 * smaller than @f$Q@f$. Two factors over one new letter on a rank-3 tagged tensor, which is a
 * shape @ref FactorizationPlan has expressed since the metric fit shipped.
 *
 * @par Where the win comes from, and why no identity is claimed
 * A density-fitted term reads the three-index tensor TWICE, and substituting both occurrences
 * puts two @c U factors in one product. The bracketing search contracts those two over @c Q
 * first, into a @f$Q'@f$ by @f$Q'@f$ matrix, and runs the truncated factors through it, so the
 * expensive contraction is over @f$Q'@f$ where it used to be over @f$Q@f$. That small matrix is
 * the identity, and nothing here says so: the cost model ranks the truncated form cheaper on the
 * extents alone, and a provider that claimed an identity would be teaching the pass an algebraic
 * fact it does not need and could not check.
 *
 * @par What chemistry calls it
 * Natural auxiliary functions (Kallay 2014), and this class registered on the tag @c "eri" is
 * what that name means. The name stays generic for the reason @ref MetricFitFactorization's
 * does: nothing in the class knows what an integral is, and the chemistry lives in the
 * registration. It is an ALTERNATIVE to @ref ThcFactorization on the same tag and the same
 * rank-3 tensor, and the registry's ranking by profitability chooses between them.
 *
 * @par The count is decided at optimize time, and refitted at that count on every bind
 * How many directions survive is a property of the integrals, so it is read off the tensor the
 * provider was handed, once, when the plan is proposed. It cannot be decided per bind: the
 * emitted node set is sized by it, and a count that moved at bind would be a node set that moved
 * at bind. This is the narrowing @c passes::LaplaceTransform already made for its point count,
 * met again for the same reason. What a rebind gets is the same count over whatever integrals it
 * then finds, and the honest report of that is the measured error, which the fitting writes on
 * every bind when a caller asks for it through @ref report_dropped_into.
 *
 * The axis still carries an index space with a dim symbol, so the extent is a symbol a later
 * bind may move and the factorization pass's numeric veto abstains over it, which is the same
 * treatment a grid axis gets and for the same reason.
 *
 * @par The truncation is a contraction, not a slice
 * Taking the leading columns of an eigenvector matrix is the natural spelling and is not one this
 * node set has: @c cg::block_copy records a @c Custom node and @c cg::view_runtime records a
 * @c View, and neither is reconstructible, so a fitting that sliced could not be saved. The
 * truncation is therefore spelled as what it is algebraically, a contraction against the leading
 * columns of an identity matrix, which is an ordinary @c Einsum the builder already rebuilds. The
 * matrix is a constant of the STRUCTURE rather than of the problem, it is the provider's own
 * tensor, and it becomes an interface tensor of the transformed graph under the name
 * @ref keep_matrix_name gives it, exactly as a collocation matrix does.
 *
 * @par What the threshold can and cannot resolve
 * The spectrum is read off the eigenvalues of @f$B B^T@f$, which are the SQUARES of the singular
 * values, so a singular value is resolved to about the square root of machine epsilon. Below a
 * relative threshold of @c 1e-8 a numerically absent direction and a genuinely small one are the
 * same number, and a caller asking for one is asking for noise. Reading the singular values
 * directly would need a decomposition of the flattened tensor rather than of a matrix the size of
 * the auxiliary set, which is a far larger object; the Gram route is the one every implementation
 * of this method takes, and the floor is stated rather than left to be discovered.
 *
 * @par The accuracy statement is measured
 * The exact quantity is the tagged tensor and the fit reads it, so the error is a number rather
 * than a claim: the norm of the dropped singular values against the whole. It is measured at
 * optimize time for the record, since a record is written once, and re-measured on every bind
 * into the tensor @ref report_dropped_into names, which is where a caller reads what the current
 * integrals were worth.
 *
 * @see Factorization.hpp for what a provider is
 * @see ThcFactorization.hpp for the provider this is an alternative to
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

/**
 * @brief Offer a three-index tensor over a truncated auxiliary index.
 *
 * @see NaturalAuxiliaryFactorization.hpp for the mathematics and for what chemistry calls it
 * @versionadded{2.1.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT NaturalAuxiliaryFactorization
    : public FactorizationProvider {
  public:
    /**
     * @brief Construct the provider.
     *
     * @param[in] tag The provenance tag this claims, e.g. ``"eri"``.
     * @param[in] three_index @c B[Q,m,n], which is also the tensor the caller tags. Checked
     *            against the tagged handle's buffer in @ref propose, so a caller who hands over a
     *            different tensor gets a decline rather than a truncation of something else.
     * @param[in] threshold Auxiliary directions whose singular value is at or below this fraction
     *            of the largest are dropped, or zero to take ``einsums:graph:naf-threshold``.
     * @param[in] name The provider's name, which the approximation record and the report carry.
     * @throws std::invalid_argument When @p threshold is not finite, is negative, or is at least
     *         one. One would drop every direction but the first, which is a decomposition of
     *         nothing, and there is nothing a negative threshold could mean.
     *
     * @par Why the threshold is relative where the metric fit's is absolute
     * The two guards are applied in different places. A metric fit's runs inside a captured
     * element operation, whose policy number is bound when its executor is built, so a relative
     * cutoff there would need a reduction feeding a value into a guard; that is the mechanism
     * @ref MetricFitFactorization records as one to add. This one runs in ordinary code, at
     * propose time, over eigenvalues held in a local buffer, so comparing against the largest of
     * them costs nothing and the conventional relative cutoff is simply available.
     */
    APIARY_EXPOSE NaturalAuxiliaryFactorization(std::string tag, RuntimeTensor<double> const &three_index, double threshold = 0.0,
                                                std::string name = "NaturalAuxiliary");

    /// @brief The same, taking a runtime-rank view.
    /// @param[in] tag The provenance tag this claims.
    /// @param[in] three_index The very tensor the caller tags.
    /// @param[in] threshold The relative singular-value cutoff, or zero to take the option.
    /// @param[in] name The provider's name.
    NaturalAuxiliaryFactorization(std::string tag, RuntimeTensorView<double> three_index, double threshold = 0.0,
                                  std::string name = "NaturalAuxiliary");

    /// @brief The same, taking a compile-time-rank tensor.
    ///
    /// Its own overload rather than left to the implicit conversion, for the reason every other
    /// provider's is: a view built from a @c Tensor is anonymous, the fitting captures it as an
    /// interface tensor, and a manifest binds by NAME.
    /// @param[in] tag The provenance tag this claims.
    /// @param[in] three_index The very tensor the caller tags.
    /// @param[in] threshold The relative singular-value cutoff, or zero to take the option.
    /// @param[in] name The provider's name.
    NaturalAuxiliaryFactorization(std::string tag, Tensor<double, 3> const &three_index, double threshold = 0.0,
                                  std::string name = "NaturalAuxiliary");

    /**
     * @brief Where the truncation writes what it threw away on this bind.
     *
     * @param[in] dropped A rank-1 tensor the fitting writes the norm of the dropped singular
     *            values, relative to the norm of all of them, into.
     *
     * The caller's own tensor, for the reason a grid fit's residual destination is one: a
     * measurement a caller cannot read is a measurement nobody makes. The record points at it by
     * name, so a caller holding a record knows which of their tensors to look in.
     *
     * Without it the fitting emits no measurement nodes and the record carries the number
     * measured at optimize time alone, which is what the structure claims rather than what this
     * bind found.
     */
    APIARY_EXPOSE void report_dropped_into(RuntimeTensor<double> const &dropped);

    /// @brief The same, taking a compile-time-rank tensor.
    /// @param[in] dropped A rank-1 tensor for the relative dropped norm.
    void report_dropped_into(Tensor<double, 1> const &dropped);

    /// @copydoc FactorizationProvider::name
    [[nodiscard]] std::string name() const override { return _name; }

    /// @copydoc FactorizationProvider::tag
    [[nodiscard]] std::string tag() const override { return _tag; }

    /**
     * @brief Offer the truncated auxiliary index, or say why this tensor has none to give.
     *
     * @param[in] graph The graph holding @p tensor.
     * @param[in] tensor The tagged tensor.
     * @return The plan, or the reason there is not one: a rank other than three, a tensor other
     *         than the one this provider was handed, an empty auxiliary axis, or a spectrum this
     *         threshold drops nothing from, which is a rename rather than a factorization.
     */
    [[nodiscard]] expected<FactorizationPlan, std::string> propose(Graph const &graph, TensorId tensor) const override;

    /// @brief The relative singular-value cutoff, taking the constructor over the option.
    /// @return The threshold.
    APIARY_EXPOSE APIARY_GETTER("threshold") [[nodiscard]] double threshold() const;

    /// @brief How many auxiliary directions the last @ref propose kept, or zero before one ran.
    /// @return The count.
    ///
    /// A property of the tensor the provider was handed rather than of the graph, which is what
    /// lets a caller read it without going through the pass's report.
    APIARY_EXPOSE APIARY_GETTER("kept") [[nodiscard]] std::size_t kept() const noexcept { return _kept; }

    /// @brief The relative dropped norm the last @ref propose measured.
    /// @return The number, or zero before one ran. This is what the record carries.
    APIARY_EXPOSE APIARY_GETTER("dropped_norm") [[nodiscard]] double dropped_norm() const noexcept { return _dropped; }

    /// @brief The index space the truncated auxiliary axis carries.
    /// @return ``"naf"``.
    APIARY_EXPOSE [[nodiscard]] static std::string space_name();

    /// @brief The symbolic extent a truncated auxiliary axis goes by.
    /// @return ``"nnaf"``.
    APIARY_EXPOSE [[nodiscard]] static std::string dim_symbol();

    /**
     * @brief Register the truncated auxiliary space in @p graph's own registry.
     *
     * A space is a statement about a graph rather than about a provider, and @ref propose is
     * handed a const graph precisely so a provider cannot make one. So a caller registers it, and
     * this is the one call that does: idempotent for a repeated identical declaration, as
     * @ref SpaceRegistry::register_space is.
     *
     * It also DECLARES the two relations that make the truncation reasonable to a pass, when the
     * registry already holds the space they relate to: the truncated space is contained in
     * @p outer and its scale is smaller. Containment is what says a contraction over the
     * truncated axis is a restriction of the one over the full axis rather than an unrelated
     * quantity, and the scale order is what lets the cost comparison rank the two forms without
     * falling through to its arbitrary tie-break. Declared rather than inferred, which is the
     * registry's own rule; when @p outer is not registered, neither relation is stated and the
     * comparison is weaker rather than wrong.
     *
     * The typical extent is DERIVED from @p outer's, at @p fraction of it, rather than left
     * unstated the way a grid's is. A grid has no relation to the basis and a truncated auxiliary
     * index is a fixed fraction of the auxiliary set by construction, so the fraction is the
     * method's own claim about itself. It is also what the cost comparison requires: its second
     * rung ranks a polynomial it cannot substitute below one it can, so a truncated form carrying
     * a space with no typical extent is declined for being unmeasurable rather than for being
     * expensive, which is the shape of a decline this call exists to prevent.
     *
     * Register @p outer BEFORE calling this, since the extent is read from it here and
     * @ref SpaceRegistry::register_space is idempotent only for an identical redeclaration.
     *
     * @param[in,out] graph The graph whose registry gains the space.
     * @param[in] outer The name of the space the auxiliary axis of the untruncated tensor carries,
     *            or empty to declare no relation and no extent.
     * @param[in] fraction What fraction of @p outer's typical extent the truncated space is
     *            expected to be. The published measurements for natural auxiliary functions put
     *            it at about a half, which is the default.
     * @return The space's id in that registry.
     * @throws std::invalid_argument When @p fraction does not lie strictly between zero and one.
     */
    APIARY_EXPOSE static SpaceId register_naf_space(Graph &graph, std::string const &outer = "aux", double fraction = 0.5);

    /**
     * @brief The name the rectangular truncation matrix is bound under.
     *
     * @param[in] provider The provider's @ref name.
     * @param[in] tensor The tagged tensor's name.
     * @return The interface name.
     *
     * A constant of the structure rather than of the problem: it is the leading columns of an
     * identity matrix, and the fitting contracts the eigenvectors against it because the node set
     * has no reconstructible slice. A caller rebinding a saved graph supplies it under this name,
     * the way they supply a collocation matrix.
     */
    APIARY_EXPOSE [[nodiscard]] static std::string keep_matrix_name(std::string const &provider, std::string const &tensor);

  private:
    std::string                              _tag;
    std::string                              _name;
    RuntimeTensorView<double>                _three_index;
    std::optional<RuntimeTensorView<double>> _dropped_report;
    double                                   _threshold;

    /// The rectangular truncation matrix, sized when a plan is proposed. Held by the provider
    /// because the fitting reads it and a captured read is what makes it an interface tensor.
    mutable std::shared_ptr<RuntimeTensor<double>> _keep;
    mutable std::size_t                            _kept{0};
    mutable double                                 _dropped{0.0};
};

EINSUMS_NAMESPACE_END(compute_graph)
