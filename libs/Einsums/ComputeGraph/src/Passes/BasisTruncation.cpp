//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/BasisTruncation.hpp>
#include <Einsums/ComputeGraph/ThcFactorization.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
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

/// @p view under the name the PROJECTION reads it by.
///
/// The projection is captured, so what it reads becomes an interface tensor of the transformed
/// graph, and the caller's own algebra no longer reads the untruncated tensor at all. Naming
/// the projection's copy is what keeps that tensor bindable; a rebind supplies the same tensor
/// under both names. The same shape a fit that reads the tensor it replaces already has.
RuntimeTensorView<double> projection_input(RuntimeTensorView<double> view, std::string const &stem) {
    view.set_name(BasisTruncation::projection_input_name(stem));
    return view;
}

/// The symmetrized virtual-virtual MP2 density of @p t, NEGATED.
///
///   D[a,b] = sum_ijc t[i,a,j,c] (2 t[i,b,j,c] - t[i,c,j,b]),  returned as -(D + D^T).
///
/// Negated because that is the form the setup diagonalizes: @c syev orders ascending, so the
/// most occupied natural orbitals come first and the leading columns are the right ones.
///
/// Flattened into matrices and multiplied with a GEMM: the element-by-element form of the same
/// product is the pass's whole cost on anything larger than a fixture.
Tensor<double, 2> negated_density(RuntimeTensorView<double> const &t) {
    std::size_t const nocc  = t.dim(0);
    std::size_t const nvir  = t.dim(1);
    std::size_t const block = nocc * nocc * nvir;

    Tensor<double, 2> left("fno_t", static_cast<int>(nvir), static_cast<int>(block));
    Tensor<double, 2> right("fno_tbar", static_cast<int>(nvir), static_cast<int>(block));
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

    // D = E + E^T, which is the symmetrized form of twice the conventional density: the factor
    // of two and the halving of the symmetrization cancel, and the occupations are what a
    // caller's threshold is compared against, so the convention is stated rather than folded
    // away.
    Tensor<double, 2> negated("fno_negated", static_cast<int>(nvir), static_cast<int>(nvir));
    for (std::size_t a = 0; a < nvir; ++a) {
        for (std::size_t b = 0; b < nvir; ++b) {
            negated(a, b) = -(density(a, b) + density(b, a));
        }
    }
    return negated;
}

/// The index letters of a tensor of @p rank whose axes at @p axes carry the space being replaced.
///
/// The truncated axes take @c "a" and @c "b", which is what the transformation's own letters
/// meet; everything else takes a letter that cannot collide with those or with the truncated
/// space's @c "y" and @c "z".
std::vector<std::string> projection_letters(std::size_t rank, std::vector<std::size_t> const &axes) {
    static constexpr std::array<char const *, 6> spectators = {"i", "j", "k", "l", "p", "q"};
    static constexpr std::array<char const *, 2> truncated  = {"a", "b"};

    std::vector<std::string> letters(rank);
    std::size_t              next = 0;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        auto const which = std::ranges::find(axes, axis);
        if (which != axes.end()) {
            letters[axis] = truncated[static_cast<std::size_t>(which - axes.begin())];
            continue;
        }
        letters[axis] = spectators[next++];
    }
    return letters;
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

std::string BasisTruncation::projected_name(std::string const &name) {
    return fmt::format("{}@fno", name);
}

std::string BasisTruncation::projection_input_name(std::string const &name) {
    return fmt::format("{}@fno_in", name);
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

void BasisTruncation::set_occupied_energies(RuntimeTensor<double> const &energies) {
    if (energies.rank() != 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BasisTruncation::set_occupied_energies: the orbital energies must be rank 1; got rank {}",
                                energies.rank());
    }
    if (_amplitudes.has_value() && energies.dim(0) != _amplitudes->dim(0)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BasisTruncation::set_occupied_energies: {} energies against amplitudes with {} occupied function(s)",
                                energies.dim(0), _amplitudes->dim(0));
    }
    _occupied.emplace(named_view(energies));
}

void BasisTruncation::set_occupied_energies(Tensor<double, 1> const &energies) {
    RuntimeTensor<double> const runtime{energies};
    set_occupied_energies(runtime);
    _occupied.emplace(named_view(energies));
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
    _correction    = 0.0;
    _occupations.clear();
    _projected.clear();
}

std::vector<std::string> BasisTruncation::explain() const {
    if (_num_truncated == 0) {
        return {};
    }
    std::vector<std::string> lines{fmt::format("BasisTruncation: kept {} of {} natural orbital(s) of '{}' above an occupation of {:g}",
                                               _kept, _amplitudes.has_value() ? _amplitudes->dim(1) : 0, _virtual_space, occupation())};
    if (!_projected.empty()) {
        lines.push_back(fmt::format("BasisTruncation: projected {} into '{}', removing {:.4e} of correlation energy",
                                    fmt::join(_projected, ", "), space_name(), _correction));
    }
    return lines;
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
    Tensor<double, 2> negated = negated_density(*_amplitudes);
    Tensor<double, 1> values("fno_occupations", static_cast<int>(nvir));
    linear_algebra::syev<false>(&negated, &values);

    std::vector<double> spectrum(nvir, 0.0);
    for (std::size_t which = 0; which < nvir; ++which) {
        // The eigenvalues of the NEGATED density, ascending, are the occupations descending,
        // which is the order the setup takes its columns in.
        spectrum[which] = -values(which);
    }
    return spectrum;
}

std::pair<std::vector<double>, std::vector<double>> BasisTruncation::ordinary_truncation() const {
    std::size_t const nvir = _amplitudes->dim(1);
    std::size_t const kept = _kept;

    Tensor<double, 2> natural = negated_density(*_amplitudes);
    Tensor<double, 1> values("fno_values", static_cast<int>(nvir));
    linear_algebra::syev<true>(&natural, &values);

    // The leading columns of the eigenvectors, which is what the rectangular identity selects
    // in the setup. Read directly here, because ordinary code has no reason to spell a slice as
    // a contraction; what the setup is spelling around is a node set that has to survive a save.
    RuntimeTensorView<double> const &fock = *_fock;
    Tensor<double, 2>                half("fno_half", static_cast<int>(kept), static_cast<int>(nvir));
    for (std::size_t y = 0; y < kept; ++y) {
        for (std::size_t b = 0; b < nvir; ++b) {
            double sum = 0.0;
            for (std::size_t a = 0; a < nvir; ++a) {
                sum += natural(a, y) * fock(a, b);
            }
            half(y, b) = sum;
        }
    }
    Tensor<double, 2> block("fno_block", static_cast<int>(kept), static_cast<int>(kept));
    for (std::size_t y = 0; y < kept; ++y) {
        for (std::size_t z = 0; z < kept; ++z) {
            double sum = 0.0;
            for (std::size_t b = 0; b < nvir; ++b) {
                sum += half(y, b) * natural(b, z);
            }
            block(y, z) = sum;
        }
    }

    Tensor<double, 1> spectrum("fno_spectrum", static_cast<int>(kept));
    linear_algebra::syev<true>(&block, &spectrum);

    std::vector<double> rotation(nvir * kept, 0.0);
    for (std::size_t a = 0; a < nvir; ++a) {
        for (std::size_t z = 0; z < kept; ++z) {
            double sum = 0.0;
            for (std::size_t y = 0; y < kept; ++y) {
                sum += natural(a, y) * block(y, z);
            }
            rotation[a * kept + z] = sum;
        }
    }
    std::vector<double> energies(kept, 0.0);
    for (std::size_t z = 0; z < kept; ++z) {
        energies[z] = spectrum(z);
    }
    return {std::move(rotation), std::move(energies)};
}

expected<double, std::string> BasisTruncation::measure_correction() const {
    if (!_occupied.has_value()) {
        return unexpected(std::string{"no occupied orbital energies were handed to the pass"});
    }
    std::size_t const                nocc = _amplitudes->dim(0);
    std::size_t const                nvir = _amplitudes->dim(1);
    std::size_t const                kept = _kept;
    RuntimeTensorView<double> const &t    = *_amplitudes;
    RuntimeTensorView<double> const &fock = *_fock;

    std::vector<double> eps_occ(nocc, 0.0);
    for (std::size_t i = 0; i < nocc; ++i) {
        eps_occ[i] = (*_occupied)(i);
    }
    std::vector<double> eps_vir(nvir, 0.0);
    for (std::size_t a = 0; a < nvir; ++a) {
        eps_vir[a] = fock(a, a);
    }

    auto const [rotation, energies] = ordinary_truncation();

    // The integrals, recovered from the amplitudes rather than asked for. A first-order
    // amplitude is the integral over its denominator, and both halves of the denominator are
    // already in hand, so a caller who handed over t[i,a,j,b] has handed over the integrals too.
    std::vector<double> integrals(nocc * nvir * nocc * nvir, 0.0);
    double              full = 0.0;
    auto const at = [&](std::size_t i, std::size_t a, std::size_t j, std::size_t b) { return ((i * nvir + a) * nocc + j) * nvir + b; };
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            for (std::size_t j = 0; j < nocc; ++j) {
                for (std::size_t b = 0; b < nvir; ++b) {
                    integrals[at(i, a, j, b)] = t(i, a, j, b) * (eps_occ[i] - eps_vir[a] + eps_occ[j] - eps_vir[b]);
                }
            }
        }
    }
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t a = 0; a < nvir; ++a) {
            for (std::size_t j = 0; j < nocc; ++j) {
                for (std::size_t b = 0; b < nvir; ++b) {
                    full += (2.0 * integrals[at(i, a, j, b)] - integrals[at(i, b, j, a)]) * t(i, a, j, b);
                }
            }
        }
    }

    // The same integrals in the truncated space, through the transformation the setup will
    // build, and the same energy over the semicanonical denominators.
    std::vector<double> half(nocc * kept * nocc * nvir, 0.0);
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t y = 0; y < kept; ++y) {
            for (std::size_t j = 0; j < nocc; ++j) {
                for (std::size_t b = 0; b < nvir; ++b) {
                    double sum = 0.0;
                    for (std::size_t a = 0; a < nvir; ++a) {
                        sum += integrals[at(i, a, j, b)] * rotation[a * kept + y];
                    }
                    half[((i * kept + y) * nocc + j) * nvir + b] = sum;
                }
            }
        }
    }
    std::vector<double> truncated(nocc * kept * nocc * kept, 0.0);
    auto const tat = [&](std::size_t i, std::size_t y, std::size_t j, std::size_t z) { return ((i * kept + y) * nocc + j) * kept + z; };
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t y = 0; y < kept; ++y) {
            for (std::size_t j = 0; j < nocc; ++j) {
                for (std::size_t z = 0; z < kept; ++z) {
                    double sum = 0.0;
                    for (std::size_t b = 0; b < nvir; ++b) {
                        sum += half[((i * kept + y) * nocc + j) * nvir + b] * rotation[b * kept + z];
                    }
                    truncated[tat(i, y, j, z)] = sum;
                }
            }
        }
    }

    double reduced = 0.0;
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t y = 0; y < kept; ++y) {
            for (std::size_t j = 0; j < nocc; ++j) {
                for (std::size_t z = 0; z < kept; ++z) {
                    double const gap = eps_occ[i] - energies[y] + eps_occ[j] - energies[z];
                    if (std::abs(gap) < 1e-12) {
                        return unexpected(std::string{"a truncated denominator vanishes, so the reduced MP2 energy is not defined"});
                    }
                    reduced += (2.0 * truncated[tat(i, y, j, z)] - truncated[tat(i, z, j, y)]) * truncated[tat(i, y, j, z)] / gap;
                }
            }
        }
    }
    return full - reduced;
}

void BasisTruncation::emit_setup(Graph &body, std::vector<Target> const &targets) const {
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

    // One intermediate per target whose axes over the space being replaced number two, which is
    // every four-external integral and every amplitude. Declared HERE, before the guard opens,
    // for the reason every workspace in a setup body is: a declaration is not a capture, and a
    // buffer the resource phase is to place must exist before the nodes that read it do.
    std::vector<RuntimeTensor<double> *> halves(targets.size(), nullptr);
    for (std::size_t which = 0; which < targets.size(); ++which) {
        Target const &target = targets[which];
        if (target.axes.size() < 2) {
            continue;
        }
        std::vector<std::size_t> dims(target.input.rank(), 0);
        for (std::size_t axis = 0; axis < dims.size(); ++axis) {
            dims[axis] = target.input.dim(axis);
        }
        dims[target.axes[0]] = kept;
        halves[which]        = &body.declare_runtime_tensor<double>(named(fmt::format("fno_project_{}", which)), dims,
                                                                    /*intermediate=*/true);
    }

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

    // The projections, under the SAME guard as the construction they read. A projection emitted
    // after the guard closed runs eagerly, against a destination that has no storage yet, which
    // is a segfault rather than a wrong number.
    for (std::size_t which = 0; which < targets.size(); ++which) {
        Target const                  &target  = targets[which];
        std::vector<std::string> const letters = projection_letters(target.input.rank(), target.axes);

        std::vector<std::string> middle = letters;
        middle[target.axes[0]]          = "y";
        // The spec outlives the call rather than being a temporary: it is a runtime string, and
        // the format type holds a view of whatever it was handed.
        std::string const first = fmt::format("{} ; a,y -> {}", fmt::join(letters, ","), fmt::join(middle, ","));
        if (target.axes.size() == 1) {
            einsum(std::string_view{first}, target.destination, target.input, *rotation);
            continue;
        }
        std::vector<std::string> outer = middle;
        outer[target.axes[1]]          = "z";
        std::string const second       = fmt::format("{} ; b,z -> {}", fmt::join(middle, ","), fmt::join(outer, ","));
        einsum(std::string_view{first}, halves[which], target.input, *rotation);
        einsum(std::string_view{second}, target.destination, *halves[which], *rotation);
    }
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
    auto const contained = graph.space_registry().find(space_name());
    if (!contained.has_value()) {
        note_skip("the truncated space is not registered in this graph's registry; call register_fno_space before applying this pass",
                  fmt::format("space '{}'", space_name()));
        return false;
    }
    SpaceId const truncated = *contained;

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

    // ── What runs over the space being replaced ──────────────────────────────
    //
    // Every handle is inspected once, in a deterministic order: a tensor map is unordered, and
    // a pass whose output depends on its iteration order is a pass whose output depends on the
    // platform, which this module has shipped twice.
    struct Candidate {
        TensorId                 id{0};
        std::string              name;
        std::vector<std::size_t> axes;
        std::size_t              rank{0};
        bool                     written{false};
        bool                     read{false};
        bool                     grid{false};
        bool                     usable{false};
        bool                     intermediate{false};
    };
    std::optional<SpaceId> const grid_space = graph.space_registry().find(ThcFactorization::grid_space_name());

    std::vector<Candidate> candidates;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        std::vector<SpaceId> const &spaces = graph.tensor_spaces(tid);
        if (spaces.empty()) {
            continue;
        }
        Candidate found;
        found.id           = tid;
        found.name         = handle.name;
        found.rank         = handle.rank;
        found.intermediate = handle.is_intermediate;
        found.usable       = handle.dtype == packed_gemm::ScalarType::Float64 && handle.impl_fn != nullptr;
        for (std::size_t axis = 0; axis < spaces.size(); ++axis) {
            if (spaces[axis] == *space) {
                found.axes.push_back(axis);
            }
            found.grid = found.grid || (grid_space.has_value() && spaces[axis] == *grid_space);
        }
        if (found.axes.empty()) {
            continue;
        }
        candidates.push_back(std::move(found));
    }
    for (auto const &node : graph.nodes()) {
        if (node.kind == OpKind::Setup) {
            continue; // Its lists are derived from a body this pass has not written yet.
        }
        for (auto &candidate : candidates) {
            for (TensorId const in : node.inputs) {
                candidate.read = candidate.read || graph.resolve_alias(in) == candidate.id;
            }
            for (TensorId const out : node.outputs) {
                candidate.written = candidate.written || graph.resolve_alias(out) == candidate.id;
            }
        }
    }
    std::ranges::sort(candidates, [](Candidate const &left, Candidate const &right) {
        return left.name == right.name ? left.id < right.id : left.name < right.name;
    });

    // Composition with a grid fit on the same space is DECLINED, and it is the tally line that
    // says so rather than a silent projection of half the program. A collocation matrix is over
    // the UNTRUNCATED basis, and whether projecting one is even the right thing is a separate
    // question from this one.
    for (auto const &candidate : candidates) {
        if (candidate.grid) {
            note_skip("a tensor over the space being replaced also runs over a grid, and projecting a collocation matrix is a "
                      "question this pass does not answer",
                      fmt::format("'{}'", candidate.name));
            return false;
        }
    }
    for (auto const &record : graph.approximations()) {
        if (std::ranges::find(record.spaces, ThcFactorization::grid_space_name()) != record.spaces.end() &&
            std::ranges::find(record.spaces, _virtual_space) != record.spaces.end()) {
            note_skip("a grid fit has already been applied over the space being replaced, and composing the two is a question this "
                      "pass does not answer",
                      fmt::format("record from '{}'", record.pass_name));
            return false;
        }
    }

    // An interface tensor the algebra WRITES cannot be projected: its extents are the caller's,
    // and a graph that wrote a truncated answer into a full-sized buffer would be reporting a
    // number for a space nobody asked about.
    for (auto const &candidate : candidates) {
        if (candidate.written && !candidate.intermediate) {
            note_skip("a tensor this graph writes runs over the space being replaced, and its extents belong to the caller",
                      fmt::format("'{}'", candidate.name));
            return false;
        }
    }

    std::vector<Candidate const *> to_project;
    std::vector<Candidate const *> to_replace;
    for (auto const &candidate : candidates) {
        if (candidate.intermediate || !candidate.read) {
            continue;
        }
        if (!candidate.usable) {
            note_skip("a tensor over the space being replaced is not a real double tensor this pass can project",
                      fmt::format("'{}'", candidate.name));
            return false;
        }
        // An energy vector over the space being replaced is REPLACED rather than projected.
        // Projecting one gives the diagonal of the Fock block in the natural-orbital basis,
        // where a denominator wants the semicanonical eigenvalues; they are different numbers
        // and only one of them makes a denominator right.
        if (candidate.rank == 1) {
            to_replace.push_back(&candidate);
            continue;
        }
        if (candidate.axes.size() > 2) {
            note_skip("a tensor carries more than two axes over the space being replaced, and the projection this pass emits is a "
                      "chain of at most two contractions",
                      fmt::format("'{}' with {} such axes", candidate.name, candidate.axes.size()));
            return false;
        }
        to_project.push_back(&candidate);
    }

    // The energy vector the denominator recipe names, recognized by its VALUES rather than by
    // its name: the diagonal of the Fock block the pass holds is what a canonical energy vector
    // over this space is, and a rank-1 tensor over the space that is not that is something else
    // whose meaning this pass has no way to guess.
    for (Candidate const *candidate : to_replace) {
        TensorHandle const *handle = graph.find_tensor(candidate->id);
        if (handle == nullptr || handle->dims.front() != nvir) {
            note_skip("a rank-1 tensor over the space being replaced has the wrong extent to be its orbital energies",
                      fmt::format("'{}'", candidate->name));
            return false;
        }
        auto const *impl = static_cast<::einsums::detail::TensorImpl<double> const *>(handle->impl_fn());
        if (impl == nullptr) {
            note_skip("a rank-1 tensor over the space being replaced has no live storage to check against the Fock block",
                      fmt::format("'{}'", candidate->name));
            return false;
        }
        RuntimeTensorView<double> const values{*impl};
        for (std::size_t a = 0; a < nvir; ++a) {
            double const held = (*_fock)(a, a);
            if (std::abs(values(a) - held) > 1e-8 * std::max(1.0, std::abs(held))) {
                note_skip("a rank-1 tensor over the space being replaced is not the orbital energies the Fock block carries, so what "
                          "it means to this program is not something this pass can guess",
                          fmt::format("'{}' differs from the Fock diagonal at entry {}", candidate->name, a));
                return false;
            }
        }
    }

    // A projection with no way to measure its own effect is declined rather than recorded with a
    // bound nobody computed. Building the space alone asks for neither, which is why a graph
    // with nothing to project still truncates.
    if (!to_project.empty() && !_occupied.has_value()) {
        note_skip("the correlation energy the truncation removes cannot be measured without the occupied orbital energies, and an "
                  "unmeasured bound is not the record this rewrite owes",
                  "call set_occupied_energies");
        return false;
    }

    // ── The projected tensors ────────────────────────────────────────────────
    //
    // Graph-owned intermediates whose value the setup supplies, NOT manifest entries: a manifest
    // entry is something a bind must supply, and nobody may bind these. What a rebind supplies is
    // the untruncated tensor, under its own name for whatever else reads it and under the
    // projection's name for the projection to read.
    std::vector<Target> targets;
    targets.reserve(to_project.size());
    for (Candidate const *candidate : to_project) {
        TensorHandle const      *handle = graph.find_tensor(candidate->id);
        std::vector<std::size_t> dims   = handle->dims;
        for (std::size_t const axis : candidate->axes) {
            dims[axis] = _kept;
        }
        auto &destination = graph.declare_runtime_tensor<double>(projected_name(candidate->name), dims, /*intermediate=*/true);

        std::vector<SpaceId> spaces = graph.tensor_spaces(candidate->id);
        for (std::size_t const axis : candidate->axes) {
            spaces[axis] = truncated;
        }
        TensorId const projected = graph.find_tensor_id_by_ptr(&destination);
        graph.annotate_spaces(projected, std::move(spaces));

        auto const *impl = static_cast<::einsums::detail::TensorImpl<double> const *>(handle->impl_fn());
        targets.push_back(Target{.source      = candidate->id,
                                 .projected   = projected,
                                 .input       = projection_input(RuntimeTensorView<double>{*impl}, candidate->name),
                                 .destination = &destination,
                                 .axes        = candidate->axes,
                                 .name        = candidate->name});
    }

    // At the FRONT, which is the reason add_setup_at exists: the nodes that read the projections
    // are already there, and a setup appended behind them is a writer behind its readers.
    Graph &body = graph.add_setup_at(fmt::format("{}({})", name(), _virtual_space), 0);
    emit_setup(body, targets);
    graph.refresh_setup_io();

    // ── Repointing the algebra ───────────────────────────────────────────────
    //
    // BOTH halves, and the caches besides. The operand LISTS are what every later pass reads and
    // the SLOTS are what an executor baked at capture resolves through, so a rewrite of one
    // without the other is half a redirect: rewriting the ids alone leaves a baked lambda reading
    // the untruncated buffer, and redirecting the slot alone leaves every analysis describing the
    // untruncated program. A declared tensor no capture has touched has no slot at all, and
    // redirect_slot at a tensor with no slot is a SILENT no-op, so the slot is created first.
    auto repoint = [&graph](TensorId from, TensorId to) {
        for (auto &node : graph.nodes()) {
            if (node.kind == OpKind::Setup) {
                continue; // Derived from its body, and refreshed rather than rewritten.
            }
            for (TensorId &id : node.inputs) {
                id = id == from ? to : id;
            }
            for (TensorId &id : node.outputs) {
                id = id == from ? to : id;
            }
        }
        graph.redirect_slot(from, to);
    };
    for (auto const &target : targets) {
        graph.get_or_create_slot(*target.destination, target.projected);
        repoint(target.source, target.projected);
        _projected.push_back(projected_name(target.name));
    }
    std::string replaced_energies;
    if (!to_replace.empty()) {
        TensorId const energies_id = graph.find_tensor_id_by_ptr(_energies.get());
        if (energies_id == 0) {
            note_skip("the setup's energy vector is not registered in this graph, so nothing could be pointed at it", energies_name());
            return false;
        }
        graph.get_or_create_slot(*_energies, energies_id);
        for (Candidate const *candidate : to_replace) {
            repoint(candidate->id, energies_id);
            replaced_energies = candidate->name;
        }
    }

    // The node list did not move and its operands did, which is a state nothing else in this
    // module produces, so every position-keyed analysis and the dependency lists are about a
    // graph that no longer exists.
    graph.mark_sorted();

    // ── The intermediates the algebra writes on the way ──────────────────────
    //
    // DERIVED from what now writes them rather than told: a contraction chain reaching a
    // truncated operand writes something over the truncated space at every step, and the shape
    // of each step is what its index letters say against the operands it now has. Their space
    // annotations are re-stated beside the extents, because an annotation naming the space that
    // was replaced would be a claim about the graph that is no longer true.
    for (auto const &candidate : candidates) {
        if (!candidate.intermediate) {
            continue;
        }
        std::vector<SpaceId> spaces = graph.tensor_spaces(candidate.id);
        for (std::size_t const axis : candidate.axes) {
            spaces[axis] = truncated;
        }
        graph.annotate_spaces(candidate.id, std::move(spaces));
    }
    graph.rederive_intermediate_extents();

    // ── What it cost ─────────────────────────────────────────────────────────
    _correction = 0.0;
    if (!targets.empty()) {
        auto measured = measure_correction();
        if (!measured) {
            note_skip("the correlation energy the truncation removes could not be measured", measured.error());
            return false;
        }
        _correction = *measured;
        approximate(graph, make_approximation_record(name(), ApproximationEffect::EnergyLike, cutoff, std::abs(_correction), {},
                                                     {_virtual_space, space_name()}, body.name(), ApproximationOrigin::Measured));
    }

    ++_num_truncated;
    report(1, fmt::format("kept {} of {} natural orbital(s) of '{}' above an occupation of {:g}", _kept, nvir, _virtual_space, cutoff));
    if (!targets.empty()) {
        report(1, fmt::format("projected {} tensor(s) into '{}' and measured a correlation energy effect of {:.4e}", targets.size(),
                              space_name(), _correction));
    }
    if (!replaced_energies.empty()) {
        report(1, fmt::format("replaced the orbital energies '{}' with the semicanonical ones the setup produces", replaced_energies));
    }
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
