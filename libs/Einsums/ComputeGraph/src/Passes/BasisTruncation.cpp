//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/BasisTruncation.hpp>
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
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// A view of @p tensor carrying its name, which the implicit conversion drops.
template <typename TensorType>
RuntimeTensorView<double> named_view(TensorType const &tensor) {
    RuntimeTensorView<double> view{tensor};
    view.set_name(tensor.name());
    return view;
}

} // namespace

std::string BasisTruncation::space_name() {
    return "fno";
}

std::string BasisTruncation::dim_symbol() {
    return "nfno";
}

std::string BasisTruncation::transformation_name() {
    return "fno_transformation";
}

std::string BasisTruncation::energies_name() {
    return "fno_energies";
}

std::string BasisTruncation::keep_matrix_name() {
    return "fno_keep";
}

SpaceId BasisTruncation::register_fno_space(Graph &graph, std::string const &outer, double fraction) {
    if (!std::isfinite(fraction) || fraction <= 0.0 || fraction >= 1.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BasisTruncation::register_fno_space: the fraction must lie strictly between zero and one; got {}. The "
                                "truncated space exists because it is smaller than the one it sits inside, and a fraction of one or more "
                                "says it is not",
                                fraction);
    }
    // Derived rather than left unstated, for the reason the auxiliary truncation's is: a space
    // declared smaller than another must carry a typical extent when that one does, or the cost
    // comparison's second rung ranks the restricted form as unmeasurable rather than as cheap.
    double                       typical = 0.0;
    std::optional<SpaceId> const full    = outer.empty() ? std::nullopt : graph.space_registry().find(outer);
    if (full.has_value()) {
        typical = fraction * graph.space_registry().space(*full).typical_extent;
    }

    SpaceId const truncated =
        graph.space_registry().register_space(make_index_space(space_name(), "f", typical, GrowthClass::linear(), dim_symbol()));
    if (!full.has_value()) {
        return truncated;
    }
    if (graph.space_registry().is_contained(truncated, *full) != Tristate::Yes) {
        graph.space_registry().declare_contained(truncated, *full);
    }
    if (graph.space_registry().is_less(truncated, *full) != Tristate::Yes) {
        graph.space_registry().declare_less(truncated, *full);
    }
    return truncated;
}

void BasisTruncation::set_amplitudes(RuntimeTensor<double> const &amplitudes) {
    if (amplitudes.rank() != 4) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BasisTruncation::set_amplitudes: the amplitudes must be rank 4, spelled t[i,a,j,b]; got rank {}",
                                amplitudes.rank());
    }
    if (amplitudes.dim(1) != amplitudes.dim(3) || amplitudes.dim(0) != amplitudes.dim(2)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BasisTruncation::set_amplitudes: the amplitudes are [{}, {}, {}, {}] and t[i,a,j,b] needs its two "
                                "occupied axes and its two virtual axes to agree",
                                amplitudes.dim(0), amplitudes.dim(1), amplitudes.dim(2), amplitudes.dim(3));
    }
    _amplitudes.emplace(named_view(amplitudes));
}

void BasisTruncation::set_amplitudes(Tensor<double, 4> const &amplitudes) {
    RuntimeTensor<double> const runtime{amplitudes};
    set_amplitudes(runtime);
    _amplitudes.emplace(named_view(amplitudes));
}

void BasisTruncation::set_fock(RuntimeTensor<double> const &fock) {
    if (fock.rank() != 2 || fock.dim(0) != fock.dim(1)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BasisTruncation::set_fock: the Fock block must be a square rank-2 tensor over the space being truncated");
    }
    _fock.emplace(named_view(fock));
}

void BasisTruncation::set_fock(Tensor<double, 2> const &fock) {
    RuntimeTensor<double> const runtime{fock};
    set_fock(runtime);
    _fock.emplace(named_view(fock));
}

void BasisTruncation::set_occupation(double occupation) {
    if (!std::isfinite(occupation) || occupation < 0.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BasisTruncation::set_occupation: the cutoff must be finite and not negative; got {}. Zero means take "
                                "einsums:graph:fno-occupation",
                                occupation);
    }
    _occupation = occupation;
}

double BasisTruncation::occupation() const {
    return _occupation > 0.0 ? _occupation : config::get(option::GraphFnoOccupation);
}

void BasisTruncation::set_virtual_space(std::string name) {
    _virtual_space = std::move(name);
}

RuntimeTensor<double> &BasisTruncation::transformation() {
    if (!_transformation) {
        _transformation = std::make_shared<RuntimeTensor<double>>(transformation_name(), std::vector<std::size_t>{0, 0});
    }
    return *_transformation;
}

RuntimeTensor<double> &BasisTruncation::energies() {
    if (!_energies) {
        _energies = std::make_shared<RuntimeTensor<double>>(energies_name(), std::vector<std::size_t>{0});
    }
    return *_energies;
}

void BasisTruncation::reset_stats() {
    _kept          = 0;
    _num_truncated = 0;
    _occupations.clear();
}

std::vector<std::string> BasisTruncation::explain() const {
    if (_num_truncated == 0) {
        return {};
    }
    return {fmt::format("BasisTruncation: kept {} of {} natural orbital(s) of '{}' above an occupation of {:g}", _kept,
                        _amplitudes.has_value() ? _amplitudes->dim(1) : 0, _virtual_space, occupation())};
}

expected<std::vector<double>, std::string> BasisTruncation::decide_count() const {
    if (!_amplitudes.has_value()) {
        return unexpected(std::string{"no first-order amplitudes were handed to the pass, so there is no density to diagonalize"});
    }
    if (!_fock.has_value()) {
        return unexpected(std::string{"no Fock block was handed to the pass, so the kept orbitals could not be semicanonicalized"});
    }
    std::size_t const nocc = _amplitudes->dim(0);
    std::size_t const nvir = _amplitudes->dim(1);
    if (nocc == 0 || nvir == 0) {
        return unexpected(std::string{"the amplitudes have an empty axis, so there is no density to diagonalize"});
    }
    if (_fock->dim(0) != nvir) {
        return unexpected(
            fmt::format("the Fock block is {} by {} and the amplitudes have {} virtual functions", _fock->dim(0), _fock->dim(1), nvir));
    }

    // The density, formed once here to decide the count and again in the setup body on every bind.
    // Twice rather than once because the count sizes the node set and the node set is built before
    // anything runs, which is the same shape the auxiliary truncation has.
    //
    // Flattened into matrices and multiplied with a GEMM: the element-by-element form of the same
    // product is the pass's whole cost on anything larger than a fixture.
    std::size_t const                block = nocc * nocc * nvir;
    Tensor<double, 2>                left("fno_t", static_cast<int>(nvir), static_cast<int>(block));
    Tensor<double, 2>                right("fno_tbar", static_cast<int>(nvir), static_cast<int>(block));
    RuntimeTensorView<double> const &t = *_amplitudes;
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            for (std::size_t j = 0; j < nocc; ++j) {
                for (std::size_t c = 0; c < nvir; ++c) {
                    std::size_t const column = (i * nocc + j) * nvir + c;
                    left(a, column)          = t(i, a, j, c);
                    right(a, column)         = 2.0 * t(i, a, j, c) - t(i, c, j, a);
                }
            }
        }
    }

    Tensor<double, 2> density("fno_density", static_cast<int>(nvir), static_cast<int>(nvir));
    linear_algebra::gemm<false, true>(1.0, left, right, 0.0, &density);
    // D = E + E^T, which is the symmetrized form of twice the conventional density: the factor of
    // two and the halving of the symmetrization cancel, and the occupations are what a caller's
    // threshold is compared against, so the convention is stated rather than folded away.
    Tensor<double, 2> symmetric("fno_symmetric", static_cast<int>(nvir), static_cast<int>(nvir));
    for (std::size_t a = 0; a < nvir; ++a) {
        for (std::size_t b = 0; b < nvir; ++b) {
            symmetric(a, b) = density(a, b) + density(b, a);
        }
    }

    Tensor<double, 1> values("fno_occupations", static_cast<int>(nvir));
    linear_algebra::syev<false>(&symmetric, &values);

    std::vector<double> spectrum(nvir, 0.0);
    for (std::size_t which = 0; which < nvir; ++which) {
        // Ascending out of syev, and the setup takes the dominant orbitals first, so the order is
        // reversed here to match the order the setup will see.
        spectrum[which] = values(nvir - 1 - which);
    }
    return spectrum;
}

void BasisTruncation::emit_setup(Graph &body) const {
    std::size_t const               nocc = _amplitudes->dim(0);
    std::size_t const               nvir = _amplitudes->dim(1);
    std::size_t const               kept = _kept;
    RuntimeTensorView<double> const t    = *_amplitudes;
    RuntimeTensorView<double> const fock = *_fock;

    auto const named = [&body](std::string_view stem) { return fmt::format("{}_{}", body.name(), stem); };

    auto &swapped    = body.declare_runtime_tensor<double>(named("fno_swapped"), {nocc, nvir, nocc, nvir}, /*intermediate=*/true);
    auto &combined   = body.declare_runtime_tensor<double>(named("fno_combined"), {nocc, nvir, nocc, nvir}, /*intermediate=*/true);
    auto &raw        = body.declare_runtime_tensor<double>(named("fno_raw"), {nvir, nvir}, /*intermediate=*/true);
    auto &transposed = body.declare_runtime_tensor<double>(named("fno_transposed"), {nvir, nvir}, /*intermediate=*/true);
    auto &negated    = body.declare_runtime_tensor<double>(named("fno_negated"), {nvir, nvir}, /*intermediate=*/true);
    auto &values     = body.declare_runtime_tensor<double>(named("fno_values"), {nvir}, /*intermediate=*/true);
    auto &natural    = body.declare_runtime_tensor<double>(named("fno_natural"), {nvir, kept}, /*intermediate=*/true);
    auto &half       = body.declare_runtime_tensor<double>(named("fno_half"), {kept, nvir}, /*intermediate=*/true);
    auto &block      = body.declare_runtime_tensor<double>(named("fno_block"), {kept, kept}, /*intermediate=*/true);

    RuntimeTensor<double> *rotation = _transformation.get();
    RuntimeTensor<double> *spectrum = _energies.get();
    RuntimeTensor<double> *keep     = _keep.get();

    CaptureGuard const guard(body);

    // The spin-summed first-order density of the virtual block:
    //   D[a,b] = sum_ijc t[i,a,j,c] (2 t[i,b,j,c] - t[i,c,j,b]) + the same with a and b swapped.
    permute("i,a,j,b <- i,b,j,a", 0.0, &swapped, 1.0, t);
    axpby(2.0, t, 0.0, &combined);
    axpby(-1.0, swapped, 1.0, &combined);
    einsum("i,a,j,c ; i,b,j,c -> a,b", &raw, t, combined);
    permute("a,b <- b,a", 0.0, &transposed, 1.0, raw);

    // NEGATED before the decomposition, so syev's ascending order puts the most occupied natural
    // orbitals first. That is what lets the selection take the leading columns, which is the one
    // column set a rectangular identity picks out without an offset.
    axpby(-1.0, raw, 0.0, &negated);
    axpby(-1.0, transposed, 1.0, &negated);
    syev(&negated, &values);
    einsum("a,r ; r,y -> a,y", &natural, negated, *keep);

    // The kept orbitals are NOT canonical, so their Fock block is not diagonal and a denominator
    // built from the old orbital energies would be wrong. Diagonalizing it inside the subspace
    // restores a diagonal denominator and hands back the new energies, ascending, which is the
    // order a caller's own energies were in.
    einsum("a,y ; a,b -> y,b", &half, natural, fock);
    einsum("y,b ; b,z -> y,z", &block, half, natural);
    syev(&block, spectrum);
    einsum("a,y ; y,z -> a,z", rotation, natural, block);
}

bool BasisTruncation::run(Graph &graph) {
    if (!_amplitudes.has_value() || !_fock.has_value()) {
        note_skip("no amplitudes or no Fock block were handed to the pass, so there is no density to truncate a space from");
        return false;
    }
    auto const space = graph.space_registry().find(_virtual_space);
    if (!space.has_value()) {
        note_skip("the space this pass truncates is not registered in this graph's registry", fmt::format("space '{}'", _virtual_space));
        return false;
    }

    auto spectrum = decide_count();
    if (!spectrum) {
        note_skip("the density could not be formed", spectrum.error());
        return false;
    }
    _occupations             = std::move(*spectrum);
    std::size_t const nvir   = _amplitudes->dim(1);
    double const      cutoff = occupation();
    _kept                    = 0;
    for (double const value : _occupations) {
        if (value > cutoff) {
            ++_kept;
        }
    }
    if (_kept == 0) {
        note_skip("an occupation cutoff keeps no natural orbital at all", fmt::format("cutoff {:g}", cutoff));
        return false;
    }
    if (_kept == nvir) {
        note_skip("an occupation cutoff drops no natural orbital, so the truncation would be a rotation rather than a truncation",
                  fmt::format("cutoff {:g} against {} orbital(s)", cutoff, nvir));
        return false;
    }

    // The pass's own tensors, sized here because this is where the count is known. A caller cannot
    // size a buffer for a count they do not yet have, which is why these are not handed over the
    // way a measurement destination is.
    _transformation = std::make_shared<RuntimeTensor<double>>(transformation_name(), std::vector<std::size_t>{nvir, _kept});
    _energies       = std::make_shared<RuntimeTensor<double>>(energies_name(), std::vector<std::size_t>{_kept});
    _keep           = std::make_shared<RuntimeTensor<double>>(keep_matrix_name(), std::vector<std::size_t>{nvir, _kept});
    _keep->zero();
    for (std::size_t which = 0; which < _kept; ++which) {
        (*_keep)(which, which) = 1.0;
    }

    Graph &body = graph.add_setup(fmt::format("{}({})", name(), _virtual_space));
    emit_setup(body);

    ++_num_truncated;
    report(1, fmt::format("kept {} of {} natural orbital(s) of '{}' above an occupation of {:g}", _kept, nvir, _virtual_space, cutoff));
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
