//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/NaturalAuxiliaryFactorization.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// A view of @p tensor carrying its name, which the implicit conversion drops.
template <typename TensorType>
RuntimeTensorView<double> named_view(TensorType const &tensor) {
    RuntimeTensorView<double> view{tensor};
    view.set_name(tensor.name());
    return view;
}

/// @p view under a name of its own, for a fit that reads the tensor the caller TAGGED.
///
/// A fitting is captured, so whatever it reads becomes an interface tensor of the transformed
/// graph. When the fit reads a tensor the caller's own algebra also holds, the pass and the
/// fitting each hold their own handle for it; naming the fitting's copy is what says so, and a
/// caller rebinding the graph supplies the same tensor under both names.
RuntimeTensorView<double> fit_input_view(RuntimeTensorView<double> view) {
    std::string const stem = view.name();
    view.set_name(fmt::format("{}@fit", stem.empty() ? std::string{"tagged"} : stem));
    return view;
}

} // namespace

NaturalAuxiliaryFactorization::NaturalAuxiliaryFactorization(std::string tag, RuntimeTensorView<double> three_index, double threshold,
                                                             std::string name)
    : _tag(std::move(tag)), _name(std::move(name)), _three_index(fit_input_view(std::move(three_index))), _threshold(threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "NaturalAuxiliaryFactorization: the relative threshold must be finite and not negative; got {}. Zero means "
                                "take einsums:graph:naf-threshold",
                                threshold);
    }
    if (threshold >= 1.0) {
        EINSUMS_THROW_EXCEPTION(
            std::invalid_argument,
            "NaturalAuxiliaryFactorization: the relative threshold must be below one; got {}. One drops every direction "
            "but the largest, which is a rank-one approximation rather than a truncation",
            threshold);
    }
}

NaturalAuxiliaryFactorization::NaturalAuxiliaryFactorization(std::string tag, RuntimeTensor<double> const &three_index, double threshold,
                                                             std::string name)
    : NaturalAuxiliaryFactorization(std::move(tag), named_view(three_index), threshold, std::move(name)) {
}

NaturalAuxiliaryFactorization::NaturalAuxiliaryFactorization(std::string tag, Tensor<double, 3> const &three_index, double threshold,
                                                             std::string name)
    : NaturalAuxiliaryFactorization(std::move(tag), named_view(three_index), threshold, std::move(name)) {
}

void NaturalAuxiliaryFactorization::report_dropped_into(RuntimeTensor<double> const &dropped) {
    _dropped_report.emplace(named_view(dropped));
}

void NaturalAuxiliaryFactorization::report_dropped_into(Tensor<double, 1> const &dropped) {
    _dropped_report.emplace(named_view(dropped));
}

double NaturalAuxiliaryFactorization::threshold() const {
    return _threshold > 0.0 ? _threshold : config::get(option::GraphNafThreshold);
}

std::string NaturalAuxiliaryFactorization::space_name() {
    return "naf";
}

std::string NaturalAuxiliaryFactorization::dim_symbol() {
    return "nnaf";
}

SpaceId NaturalAuxiliaryFactorization::register_naf_space(Graph &graph, std::string const &outer, double fraction) {
    if (!std::isfinite(fraction) || fraction <= 0.0 || fraction >= 1.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "NaturalAuxiliaryFactorization::register_naf_space: the fraction must lie strictly between zero and one; "
                                "got {}. The truncated space exists because it is smaller than the one it sits inside, and a fraction of "
                                "one or more says it is not",
                                fraction);
    }
    // A typical extent DERIVED from the space this one sits inside, rather than none.
    //
    // A grid space declares none on purpose, because a grid is chosen per problem and has no
    // relation to the basis. A truncated auxiliary index is the opposite: it is a fixed fraction
    // of the auxiliary set by construction, and that fraction is the method's own claim. Leaving
    // it unstated is not neutral either, because the cost comparison's second rung ranks a
    // polynomial it cannot substitute BELOW one it can, so a truncated form with no typical
    // extent is refused for being unmeasurable rather than for being expensive. Stating it is
    // also what the comparison's own discipline requires of a declared scale order: either every
    // space a declared order relates carries a typical extent or none does.
    double                       typical = 0.0;
    std::optional<SpaceId> const full    = outer.empty() ? std::nullopt : graph.space_registry().find(outer);
    if (full.has_value()) {
        typical = fraction * graph.space_registry().space(*full).typical_extent;
    }

    // The extent is a SYMBOL rather than the count a capture happened to decide, which is what
    // makes a saved graph replayed at a new geometry a rebind and what makes the factorization
    // pass's numeric veto abstain over the axis.
    SpaceId const truncated =
        graph.space_registry().register_space(make_index_space(space_name(), "y", typical, GrowthClass::linear(), dim_symbol()));
    if (!full.has_value()) {
        return truncated;
    }
    // Declared, never inferred, which is the registry's own rule. Containment is what says a
    // contraction over the truncated axis is a restriction of the one over the full auxiliary
    // index rather than an unrelated quantity; the scale order is what lets the cost comparison
    // rank the two forms without falling through to its arbitrary tie-break.
    if (graph.space_registry().is_contained(truncated, *full) != Tristate::Yes) {
        graph.space_registry().declare_contained(truncated, *full);
    }
    if (graph.space_registry().is_less(truncated, *full) != Tristate::Yes) {
        graph.space_registry().declare_less(truncated, *full);
    }
    return truncated;
}

std::string NaturalAuxiliaryFactorization::keep_matrix_name(std::string const &provider, std::string const &tensor) {
    return fmt::format("{}.{}.keep", provider, tensor);
}

namespace {

/// The singular-value spectrum of @p three_index against its pair index, largest first.
///
/// The tensor is flattened once into a matrix and the Gram matrix taken with a GEMM, because the
/// element-by-element form of the same product is a hundred times slower on anything larger than
/// a fixture and this runs on a real molecule's integrals.
std::vector<double> auxiliary_spectrum(RuntimeTensorView<double> const &three_index) {
    std::size_t const naux = three_index.dim(0);
    std::size_t const pair = three_index.dim(1) * three_index.dim(2);

    Tensor<double, 2> flat("naf_flat", static_cast<int>(naux), static_cast<int>(pair));
    for (std::size_t aux = 0; aux < naux; ++aux) {
        for (std::size_t row = 0; row < three_index.dim(1); ++row) {
            for (std::size_t col = 0; col < three_index.dim(2); ++col) {
                flat(aux, row * three_index.dim(2) + col) = three_index(aux, row, col);
            }
        }
    }

    Tensor<double, 2> gram("naf_gram", static_cast<int>(naux), static_cast<int>(naux));
    linear_algebra::gemm<false, true>(1.0, flat, flat, 0.0, &gram);

    Tensor<double, 1> values("naf_values", static_cast<int>(naux));
    linear_algebra::syev<false>(&gram, &values);

    std::vector<double> spectrum(naux, 0.0);
    for (std::size_t which = 0; which < naux; ++which) {
        // Ascending out of syev, and the fitting takes the dominant directions first, so the
        // order is reversed here to match the order the fitting will see.
        spectrum[which] = std::max(values(naux - 1 - which), 0.0);
    }
    return spectrum;
}

} // namespace

expected<FactorizationPlan, std::string> NaturalAuxiliaryFactorization::propose(Graph const &graph, TensorId tensor) const {
    TensorHandle const *handle = graph.find_tensor(tensor);
    if (handle == nullptr) {
        return unexpected(std::string{"the tagged tensor is not registered in this graph"});
    }
    if (handle->rank != 3) {
        return unexpected(fmt::format("an auxiliary truncation replaces a rank-3 tensor; this one is rank {}", handle->rank));
    }
    if (handle->data_ptr != static_cast<void const *>(_three_index.data())) {
        return unexpected(std::string{"the tagged tensor is not the one this provider was given to truncate"});
    }
    if (_three_index.rank() != 3) {
        return unexpected(fmt::format("this provider was handed a rank-{} tensor and needs a rank-3 one", _three_index.rank()));
    }
    std::size_t const naux = _three_index.dim(0);
    if (naux == 0 || _three_index.dim(1) == 0 || _three_index.dim(2) == 0) {
        return unexpected(std::string{"the tagged tensor has an empty axis, so there is no spectrum to truncate"});
    }
    if (handle->dims[0] != naux || handle->dims[1] != _three_index.dim(1) || handle->dims[2] != _three_index.dim(2)) {
        return unexpected(fmt::format("the tagged tensor is [{}] and this provider holds [{}, {}, {}]", fmt::join(handle->dims, ", "), naux,
                                      _three_index.dim(1), _three_index.dim(2)));
    }

    // How many directions survive is read off the integrals ONCE, here, and never again. The
    // emitted node set is sized by the count, so a count that moved at bind would be a node set
    // that moved at bind; what a rebind gets is the same count over whatever is then bound, and
    // the measurement the fitting writes is the honest report of that.
    std::vector<double> const spectrum = auxiliary_spectrum(_three_index);
    double const              largest  = std::sqrt(spectrum.front());
    if (!(largest > 0.0)) {
        return unexpected(std::string{"the tagged tensor is zero, so every auxiliary direction carries nothing"});
    }
    double const cutoff = threshold() * largest;
    std::size_t  kept   = 0;
    double       whole  = 0.0;
    for (double const value : spectrum) {
        whole += value;
        if (std::sqrt(value) > cutoff) {
            ++kept;
        }
    }
    if (kept == 0) {
        return unexpected(fmt::format("a threshold of {:g} keeps no auxiliary direction at all", threshold()));
    }
    if (kept == naux) {
        return unexpected(fmt::format("a threshold of {:g} drops no auxiliary direction of {}, so the substitution would be a rename "
                                      "rather than a truncation",
                                      threshold(), naux));
    }
    double dropped = 0.0;
    for (std::size_t which = kept; which < naux; ++which) {
        dropped += spectrum[which];
    }
    _kept    = kept;
    _dropped = whole > 0.0 ? std::sqrt(dropped / whole) : 0.0;

    // The rectangular truncation matrix: the leading columns of an identity. Sized here because
    // this is where the count is known, and held by the provider because the fitting READS it and
    // a captured read is what makes it an interface tensor a later bind supplies.
    std::string const keep_name = keep_matrix_name(_name, handle->name);
    if (!_keep || _keep->dim(0) != naux || _keep->dim(1) != kept) {
        _keep = std::make_shared<RuntimeTensor<double>>(keep_name, std::vector<std::size_t>{naux, kept});
    }
    _keep->set_name(keep_name);
    _keep->zero();
    for (std::size_t which = 0; which < kept; ++which) {
        (*_keep)(which, which) = 1.0;
    }

    FactorizationPlan plan;
    plan.provider         = _name;
    plan.tagged_letters   = {"Q", "m", "n"};
    plan.fits_from_tagged = true;
    plan.factors.push_back(FactorTensor{.name    = "U",
                                        .letters = {"Q", "Y"},
                                        .dims    = {naux, kept},
                                        .spaces  = {std::string{}, space_name()},
                                        .dtype   = packed_gemm::ScalarType::Float64});
    plan.factors.push_back(FactorTensor{.name    = "Bt",
                                        .letters = {"Y", "m", "n"},
                                        .dims    = {kept, _three_index.dim(1), _three_index.dim(2)},
                                        .spaces  = {space_name(), std::string{}, std::string{}},
                                        .dtype   = packed_gemm::ScalarType::Float64});

    // MEASURED, and the number is the norm of the dropped singular values against the whole,
    // which the eigenvalues above already hold. The tolerance beside it is the knob the caller
    // turned, which is a different thing: a threshold says which directions go and the bound says
    // what going cost.
    plan.accuracy =
        make_approximation_record(_name, ApproximationEffect::NormRelative, threshold(), _dropped, {}, {space_name()}, "",
                                  ApproximationOrigin::Measured, _dropped_report.has_value() ? _dropped_report->name() : std::string{});

    RuntimeTensorView<double> const              three_index = _three_index;
    std::shared_ptr<RuntimeTensor<double>> const keep        = _keep;
    std::optional<RuntimeTensorView<double>>     report      = _dropped_report;
    std::size_t const                            rows        = _three_index.dim(1);
    std::size_t const                            cols        = _three_index.dim(2);

    plan.emit_setup = [three_index, keep, report, naux, kept, rows, cols](Graph &parent, Graph &body,
                                                                          std::vector<TensorId> const &factors) {
        auto *U  = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);
        auto *Bt = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[1]).tensor_ptr);

        // Named after the graph they are declared in, because one program may hold several of
        // these truncations and the storage auditor keys its duplicate check on the name.
        auto const named = [&body](std::string_view stem) { return fmt::format("{}_{}", body.name(), stem); };

        auto &gram    = body.declare_runtime_tensor<double>(named("naf_gram"), {naux, naux}, /*intermediate=*/true);
        auto &vectors = body.declare_runtime_tensor<double>(named("naf_vectors"), {naux, naux}, /*intermediate=*/true);
        auto &values  = body.declare_runtime_tensor<double>(named("naf_values"), {naux}, /*intermediate=*/true);

        bool const measure = report.has_value();
        // The caller's tensor reached through a heap copy the BODY owns, because a node holds a
        // pointer to its destination and a view living in this lambda's own storage would move
        // with every copy of the lambda.
        RuntimeTensorView<double> *destination = nullptr;
        if (measure) {
            auto *held  = new RuntimeTensorView<double>(*report);
            destination = held;
            body.adopt([held]() { delete held; });
        }

        {
            CaptureGuard const guard(body);

            // M[P,Q] = sum_mn B[P,m,n] B[Q,m,n], NEGATED before the decomposition so that syev's
            // ascending order puts the dominant directions first. That is what lets the
            // truncation take the leading columns, which is the one column set a rectangular
            // identity can select without an offset.
            einsum("P,m,n ; Q,m,n -> P,Q", &gram, three_index, three_index);
            permute("P,Q <- P,Q", 0.0, &vectors, -1.0, gram);
            syev(&vectors, &values);

            // The truncation, spelled as the contraction it is. Slicing the eigenvectors is the
            // natural spelling and is not one this node set can save: block_copy records a Custom
            // node and a view records a View, and neither is reconstructible.
            einsum("P,R ; R,Y -> P,Y", U, vectors, *keep);
            einsum("Q,Y ; Q,m,n -> Y,m,n", Bt, *U, three_index);

            if (measure) {
                // What the truncation threw away on THIS bind, as the norm of the dropped
                // singular values against the whole. Both norms are dots the fit already has the
                // operands for: the whole is the Frobenius norm of the tagged tensor and the kept
                // part is that of its projection, since the columns of U are orthonormal.
                auto &whole_squared = body.declare_runtime_tensor<double>(named("naf_whole"), {std::size_t{1}}, /*intermediate=*/true);
                auto &kept_squared  = body.declare_runtime_tensor<double>(named("naf_kept"), {std::size_t{1}}, /*intermediate=*/true);
                auto &difference    = body.declare_runtime_tensor<double>(named("naf_difference"), {std::size_t{1}}, /*intermediate=*/true);
                dot_python(&whole_squared, three_index, three_index);
                dot_python(&kept_squared, *Bt, *Bt);
                axpby(1.0, whole_squared, 0.0, &difference);
                axpby(-1.0, kept_squared, 1.0, &difference);
                direct_division(1.0, difference, whole_squared, 0.0, destination);
                element_transform(destination, "sqrt_or_zero");
            }
        }

        (void)rows;
        (void)cols;

        // The workspace is declared and left DEFERRED. Materializing it here would bake a
        // resource decision into structure, which is the mistake the metric fit's own comment
        // records; the resource phase places it, inside this body, where a fitting's scratch
        // belongs.
    };

    return plan;
}

EINSUMS_NAMESPACE_END(compute_graph)
