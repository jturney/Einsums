//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file BasisTruncation.hpp
 * @brief Replace an index space by a contained subspace, and project everything that runs over it.
 *
 * @par What it is, in the vocabulary of the layer it lives in
 * Every other lossy pass rewrites an EXPRESSION. This one replaces a SPACE. The virtual index of a
 * correlated method is truncated to the directions a first-order density says carry occupation,
 * every tensor whose slot is annotated with the full space is projected into the subspace, the
 * orbital energies are replaced by the ones the truncated Fock block has, and the method then runs
 * unchanged over shorter axes.
 *
 * Nothing is factored, which is why this is not a @ref FactorizationProvider: there is no product
 * of factors to substitute and no bracketing to search. What there is instead is a registry
 * relation, a setup that builds the transformation, and a manifest that grows one entry per
 * projected tensor.
 *
 * @par The construction
 * From the first-order amplitudes @f$t[i,a,j,b]@f$ the virtual-virtual block of the MP2 density is
 *
 * @f[ D[a,b] = 2\sum_{ijc} t[i,a,j,c]\,\bigl(2\,t[i,b,j,c] - t[i,c,j,b]\bigr) @f]
 *
 * symmetrized. Its eigenvectors are the natural orbitals and its eigenvalues their occupations;
 * the ones above @c einsums:graph:fno-occupation are kept. The kept block is not canonical, so the
 * Fock matrix is transformed into it and diagonalized a second time, which restores a diagonal
 * denominator and hands back the new orbital energies in ascending order. The transformation is
 * the product of the two rotations.
 *
 * @par What chemistry calls it
 * Frozen natural orbitals (Taube and Bartlett 2005; DePrince and Sherrill 2013). Nothing in the
 * class knows what an orbital is: it truncates the space a caller names, from the density a caller
 * hands it, and the chemistry lives in the naming.
 *
 * @par The count is decided at optimize time
 * How many natural orbitals survive is read off the amplitudes once, when the pass runs, and the
 * emitted node set is sized by it. The same narrowing @c LaplaceTransform made for its point count
 * and @ref NaturalAuxiliaryFactorization for its auxiliary count, for the same reason: a count that
 * moved at bind would be a node set that moved at bind. What a rebind gets is the same count over
 * whatever amplitudes it then finds, refitted.
 *
 * The truncated axis carries an index space with a dim symbol regardless, so the extent is one a
 * later bind may move and every extent-sensitive rule treats it as a placeholder.
 *
 * @par The truncation is a contraction, not a slice
 * Taking the leading columns of an eigenvector matrix is not an operation this node set can save,
 * for the reason @ref NaturalAuxiliaryFactorization states at length: @c block_copy records a
 * @c Custom node and a view records a @c View, and neither is reconstructible. The selection is
 * spelled as a contraction against the leading columns of an identity matrix, which is an ordinary
 * @c Einsum, and that matrix is a constant of the structure the pass owns and a bind supplies.
 *
 * @see NaturalAuxiliaryFactorization.hpp for the same truncation on the auxiliary index
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Truncate an index space to the natural orbitals a density says are occupied.
 *
 * @see BasisTruncation.hpp for the construction and for what chemistry calls it
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT BasisTruncation : public OptimizerPass {
  public:
    /// @brief Default-construct. Explicit so the binding codegen has a constructor to annotate.
    APIARY_EXPOSE BasisTruncation() = default;

    /// @copydoc OptimizerPass::name
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "BasisTruncation"; }

    /// @copydoc OptimizerPass::tier
    /// Replaces a space by a contained subspace under a tolerance it records through
    /// OptimizerPass::approximate. Never eligible for a default manager, which is what this tier means.
    [[nodiscard]] PassTier tier() const override { return PassTier::Lossy; }

    /// @brief The index space the truncated axes carry.
    /// @return ``"fno"``.
    APIARY_EXPOSE [[nodiscard]] static std::string space_name();

    /// @brief The symbolic extent a truncated axis goes by.
    /// @return ``"nfno"``.
    APIARY_EXPOSE [[nodiscard]] static std::string dim_symbol();

    /**
     * @brief Register the truncated space in @p graph's own registry, inside @p outer.
     *
     * The same shape @ref NaturalAuxiliaryFactorization::register_naf_space has, and the same three
     * statements: the space itself, its containment in the space it truncates, and the scale order
     * beside it. Containment is what says a contraction over the subspace is a RESTRICTION of the
     * one over the parent rather than an unrelated quantity; the scale order is what lets a cost
     * comparison rank the two.
     *
     * The typical extent is derived from @p outer's at @p fraction of it, for the reason the
     * auxiliary truncation's is: a space declared smaller than another must carry a typical extent
     * when that other one does, or the cost comparison's second rung ranks the restricted form as
     * unmeasurable rather than as cheap.
     *
     * Register @p outer BEFORE calling this.
     *
     * @param[in,out] graph The graph whose registry gains the space.
     * @param[in] outer The name of the space being truncated, or empty for no relation.
     * @param[in] fraction What fraction of @p outer's typical extent the truncated space is
     *            expected to be. A half is the conventional frozen-natural-orbital saving.
     * @return The space's id in that registry.
     * @throws std::invalid_argument When @p fraction does not lie strictly between zero and one.
     */
    APIARY_EXPOSE static SpaceId register_fno_space(Graph &graph, std::string const &outer = "vir", double fraction = 0.5);

    /**
     * @brief Hand the pass the first-order amplitudes the density is built from.
     *
     * @param[in] amplitudes @c t[i,a,j,b], rank 4, with the virtual axes second and fourth. Its
     *            storage must outlive every replay, since the setup body reads it on each bind.
     * @throws std::invalid_argument When @p amplitudes is not a rank-4 tensor whose two virtual
     *         axes agree.
     *
     * Supplied to the PASS rather than found in the graph, which is the shape
     * @c LaplaceTransform has for its energies and @ref MetricFitFactorization for its three-index
     * tensor, and for the same reason: a caller who computed the amplitudes outside the graph has
     * a tensor no node mentions, and the graph has never seen it.
     *
     * It becomes an interface tensor of the transformed graph, under its own name, because the
     * setup body captures it. That is what lets a saved graph re-truncate at a new geometry.
     */
    APIARY_EXPOSE void set_amplitudes(RuntimeTensor<double> const &amplitudes);

    /// @brief The same, taking a compile-time-rank tensor.
    /// @param[in] amplitudes @c t[i,a,j,b].
    void set_amplitudes(Tensor<double, 4> const &amplitudes);

    /**
     * @brief Hand the pass the Fock block over the space being truncated.
     *
     * @param[in] fock @c F[a,b] over the full virtual space, rank 2 and square.
     * @throws std::invalid_argument When @p fock is not a square rank-2 tensor.
     *
     * Needed because the kept natural orbitals are not canonical: their Fock block is not
     * diagonal, so an energy denominator built from the old orbital energies would be wrong.
     * Diagonalizing it inside the subspace is what restores a diagonal denominator and produces
     * the new energies.
     */
    APIARY_EXPOSE void set_fock(RuntimeTensor<double> const &fock);

    /// @brief The same, taking a compile-time-rank tensor.
    /// @param[in] fock @c F[a,b].
    void set_fock(Tensor<double, 2> const &fock);

    /**
     * @brief Hand the pass the orbital energies of the space the amplitudes' other axes run over.
     *
     * @param[in] energies @c eps[i], one per occupied orbital, rank 1.
     * @throws std::invalid_argument When @p energies is not rank 1 or does not have one entry
     *         per occupied axis of the amplitudes. Set the amplitudes first.
     *
     * @par What it is for, since the construction does not need it
     * The record this pass writes carries @f$E_{MP2}(\text{full}) - E_{MP2}(\text{fno})@f$ as
     * its measured effect, and the two energies are what needs these. The integrals are not
     * asked for separately because the amplitudes already carry them: a first-order amplitude
     * is the integral over its denominator, and the denominator is these energies against the
     * diagonal of the Fock block the pass already holds.
     *
     * A projection with no way to measure its own effect is declined rather than recorded with
     * a bound nobody computed, so these are required to PROJECT and not to build the space.
     */
    APIARY_EXPOSE void set_occupied_energies(RuntimeTensor<double> const &energies);

    /// @brief The same, taking a compile-time-rank tensor.
    /// @param[in] energies @c eps[i].
    void set_occupied_energies(Tensor<double, 1> const &energies);

    /**
     * @brief Override ``einsums:graph:fno-occupation`` for this pipeline.
     * @param[in] occupation Natural orbitals whose occupation exceeds this are kept.
     * @throws std::invalid_argument When @p occupation is not finite or is negative.
     */
    APIARY_EXPOSE void set_occupation(double occupation);

    /// @brief The occupation cutoff, taking @ref set_occupation over the option.
    /// @return The cutoff.
    APIARY_EXPOSE APIARY_GETTER("occupation") [[nodiscard]] double occupation() const;

    /// @brief The name of the space this truncates.
    /// @param[in] name An index-space name, e.g. ``"vir"``.
    APIARY_EXPOSE void set_virtual_space(std::string name);

    /// @brief The space this truncates.
    /// @return The name, ``"vir"`` unless @ref set_virtual_space changed it.
    APIARY_EXPOSE APIARY_GETTER("virtual_space") [[nodiscard]] std::string virtual_space() const { return _virtual_space; }

    /// @brief How many natural orbitals the last run kept, or zero before one ran.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("kept") [[nodiscard]] std::size_t kept() const noexcept { return _kept; }

    /// @brief The occupations the last run read off the density, largest first.
    /// @return The eigenvalues of the virtual-virtual density, or empty before a run.
    APIARY_EXPOSE APIARY_GETTER("occupations") [[nodiscard]] std::vector<double> occupations() const { return _occupations; }

    /**
     * @brief The transformation the setup writes, @c U over the full space by the truncated one.
     * @return The tensor. Empty until the pass has run; written on every bind after that.
     *
     * The pass's own tensor rather than the caller's, because its second extent is decided by the
     * pass and a caller cannot size a buffer for a count they do not yet know. It becomes an
     * interface tensor of the transformed graph under @ref transformation_name.
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) [[nodiscard]] RuntimeTensor<double> &transformation();

    /// @brief The orbital energies of the truncated space, ascending.
    /// @return The tensor. Empty until the pass has run; written on every bind after that.
    APIARY_EXPOSE APIARY_RVP(reference_internal) [[nodiscard]] RuntimeTensor<double> &energies();

    /// @brief The interface name @ref transformation is bound under.
    /// @return ``"fno_transformation"``.
    APIARY_EXPOSE [[nodiscard]] static std::string transformation_name();

    /// @brief The interface name @ref energies is bound under.
    /// @return ``"fno_energies"``.
    APIARY_EXPOSE [[nodiscard]] static std::string energies_name();

    /// @brief The interface name the rectangular selection matrix is bound under.
    /// @return ``"fno_keep"``. See the file note on why the selection is a contraction.
    APIARY_EXPOSE [[nodiscard]] static std::string keep_matrix_name();

    /**
     * @brief What the setup writes the projection of @p name into.
     * @param[in] name The name of the tensor being projected.
     * @return ``"<name>@fno"``.
     *
     * @par Not a manifest entry, and the difference is the whole of the contract
     * A manifest entry is something a bind must SUPPLY. These are graph-owned intermediates
     * whose value the setup supplies and which nobody may bind. What a rebind supplies is the
     * UNTRUNCATED tensor, under its own name for whatever else reads it and under
     * @ref projection_input_name for the projection to read.
     */
    APIARY_EXPOSE [[nodiscard]] static std::string projected_name(std::string const &name);

    /**
     * @brief The interface name the projection reads @p name under.
     * @param[in] name The name of the tensor being projected.
     * @return ``"<name>@fno_in"``.
     *
     * The projection is captured, so what it reads becomes an interface tensor of the
     * transformed graph. The caller's own algebra no longer reads that tensor at all, so
     * naming the projection's copy is what keeps the untruncated tensor bindable; a rebind
     * supplies the same tensor under this name. The same shape a fit that reads the tensor it
     * replaces already has.
     */
    APIARY_EXPOSE [[nodiscard]] static std::string projection_input_name(std::string const &name);

    /// @brief The names the last run projected, in the order they were projected.
    /// @return One entry per projected tensor, spelled by @ref projected_name.
    APIARY_EXPOSE APIARY_GETTER("projected") [[nodiscard]] std::vector<std::string> projected() const { return _projected; }

    /**
     * @brief The MP2 energy the truncation removed, @f$E_{MP2}(\text{full}) - E_{MP2}(\text{fno})@f$.
     * @return The correction, or zero when the last run projected nothing.
     *
     * The record's measured quantity rather than an extra output: a program that wants the
     * corrected number adds this to whatever the correlated method produced in the truncated
     * space, which is the same place the density-fitting and quadrature effects already
     * compose.
     */
    APIARY_EXPOSE APIARY_GETTER("correction") [[nodiscard]] double correction() const noexcept { return _correction; }

    /// @copydoc OptimizerPass::phase
    /// Structural-algebraic, for the reason the other two lossy passes are: replacing a space is a
    /// statement about the arithmetic, and one a reload quietly dropped would change what the graph
    /// computes.
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }
    /// @copydoc OptimizerPass::understood_features
    /// Only what commutes with a real rotation of one index space: repointing is by id and does not reach loop bodies, so views, control
    /// flow and redirected slots stay out.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override {
        return NodeFeatures{} | NodeFeature::Conjugation | NodeFeature::ComplexPrefactor | NodeFeature::MixedPrecision |
               NodeFeature::RawScalar;
    }

    /// @copydoc OptimizerPass::run
    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

    /// @copydoc OptimizerPass::reset_stats
    void reset_stats() override;

  private:
    /// @brief One tensor of the caller's whose axes run over the space being replaced.
    struct Target {
        TensorId                  source{0};            ///< The caller's tensor, as this graph knows it.
        TensorId                  projected{0};         ///< The graph-owned tensor the setup writes.
        RuntimeTensorView<double> input;                ///< The projection's own view of the caller's tensor.
        RuntimeTensor<double>    *destination{nullptr}; ///< Where the projection writes.
        std::vector<std::size_t>  axes;                 ///< Which of its axes carry the space being replaced.
        std::string               name;                 ///< The caller's name for it.
    };

    /// @brief Read the occupations off the amplitudes and decide how many survive.
    /// @return The occupations, largest first, or a reason there are none.
    [[nodiscard]] expected<std::vector<double>, std::string> decide_count() const;

    /// @brief Capture the construction of @ref transformation and @ref energies into @p body.
    /// @param[in,out] body The setup body.
    /// @param[in] targets The tensors whose projection is captured under the same guard.
    void emit_setup(Graph &body, std::vector<Target> const &targets) const;

    /// @brief The transformation and the energies, computed here rather than read off the graph.
    /// @return The isometry over the full space by the truncated one, and the semicanonical
    ///         energies, in the order @ref emit_setup will produce them.
    [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> ordinary_truncation() const;

    /// @brief The MP2 energy the truncation removed, from the amplitudes and the two energy sets.
    /// @param[in] rotation The transformation, row-major over the full space by the truncated one.
    /// @param[in] energies The semicanonical orbital energies of the truncated space.
    /// @return The difference, or a reason it could not be measured.
    [[nodiscard]] expected<double, std::string> measure_correction(std::vector<double> const &rotation,
                                                                   std::vector<double> const &energies) const;

    std::optional<RuntimeTensorView<double>> _amplitudes;
    std::optional<RuntimeTensorView<double>> _fock;
    std::optional<RuntimeTensorView<double>> _occupied;
    std::string                              _virtual_space{"vir"};
    double                                   _occupation{0}; ///< Zero means "read the option".

    std::shared_ptr<RuntimeTensor<double>> _transformation;
    std::shared_ptr<RuntimeTensor<double>> _energies;
    /// The rectangular selection matrix, sized when the count is decided. Held by the pass because
    /// the setup reads it and a captured read is what makes it an interface tensor.
    std::shared_ptr<RuntimeTensor<double>> _keep;

    std::size_t              _kept{0};
    std::vector<double>      _occupations;
    std::size_t              _num_truncated{0};
    std::vector<std::string> _projected;
    double                   _correction{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
