//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/LaplaceQuadrature.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/LaplaceTransform.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ContractionTreeSearch.hpp"
#include "LaplaceRewrite.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

constexpr std::string_view kTag = "laplace_denominator";

/// The letters of an index list, in order.
std::vector<std::string> letters_of(std::vector<ExprIndex> const &indices) {
    std::vector<std::string> out;
    out.reserve(indices.size());
    for (auto const &index : indices) {
        out.push_back(index.letter);
    }
    return out;
}

bool contains(std::vector<std::string> const &haystack, std::string const &needle) {
    return std::ranges::find(haystack, needle) != haystack.end();
}

/// A letter nothing in @p used spells, seeded from @p wanted.
std::string fresh_letter(std::vector<std::string> const &used, std::string const &wanted) {
    if (!contains(used, wanted)) {
        return wanted;
    }
    for (int suffix = 1;; ++suffix) {
        std::string candidate = fmt::format("{}{}", wanted, suffix);
        if (!contains(used, candidate)) {
            return candidate;
        }
    }
}

/// Whether any node of @p graph writes the buffer @p id names.
///
/// The gate on which tensors may be transformed at all, and the same one
/// @ref FactorizationPass applies: a tensor the graph PRODUCES would have to have its
/// quadrature refitted whenever it changed, and the setup body runs once per bound problem.
bool written_anywhere(Graph const &graph, TensorId id) {
    auto const owner = graph.resolve_alias(id);
    for (auto const &node : graph.nodes()) {
        for (TensorId const out : node.outputs) {
            if (graph.resolve_alias(out) == owner) {
                return true;
            }
        }
    }
    return false;
}

/// Whether @p dtype is one this pass can represent an exponential in.
bool is_real_dtype(packed_gemm::ScalarType dtype) {
    return dtype == packed_gemm::ScalarType::Float32 || dtype == packed_gemm::ScalarType::Float64;
}

/// Declare a graph-owned deferred intermediate of the given shape.
///
/// Deferred and runtime-rank for the two reasons every pass-created tensor is both: the memory
/// passes can only manage storage they are allowed to place, and a later bind can only move an
/// extent whose storage has not been committed.
TensorId declare_scratch(Graph &graph, std::string name, packed_gemm::ScalarType dtype, std::vector<std::size_t> const &dims) {
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> TensorId {
        auto &tensor = graph.declare_runtime_tensor<T>(std::move(name), dims, /*intermediate=*/true);
        return graph.live_tensor_id_by_ptr(&tensor, {});
    });
}

/// The product of two real prefactors.
///
/// Real because the pass has already declined a complex denominator, and a complex prefactor
/// on a real contraction is a caller error rather than a case to carry.
PrefactorScalar multiply_real(PrefactorScalar const &lhs, PrefactorScalar const &rhs) {
    return PrefactorScalar{as_real<double>(lhs) * as_real<double>(rhs)};
}

} // namespace

std::optional<std::pair<double, double>> LaplaceTransform::energy_extremes(EnergyVector const &held) {
    return detail::dispatch_scalar_type(held.dtype, [&]<typename T>(T /*tag*/) -> std::optional<std::pair<double, double>> {
        if constexpr (!std::is_floating_point_v<T>) {
            return std::nullopt;
        } else {
            auto const *view = static_cast<RuntimeTensorView<T> const *>(held.view.get());
            auto const &impl = view->impl();
            if (impl.data() == nullptr || impl.rank() != 1 || impl.dim(0) == 0) {
                return std::nullopt;
            }
            T const          *data = impl.data();
            std::size_t const inc  = impl.stride(0);
            T                 low  = data[0];
            T                 high = data[0];
            for (std::size_t i = 1; i < impl.dim(0); ++i) {
                low  = std::min(low, data[i * inc]);
                high = std::max(high, data[i * inc]);
            }
            return std::make_pair(static_cast<double>(low), static_cast<double>(high));
        }
    });
}

std::string LaplaceTransform::tag_name() {
    return std::string(kTag);
}

ProvenanceTag LaplaceTransform::denominator_tag(std::vector<std::string> const &energies, std::string const &signs) {
    if (energies.empty() || energies.size() != signs.size()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "LaplaceTransform::denominator_tag: {} energy name(s) and {} sign(s); one of each per axis of the "
                                "denominator is required",
                                energies.size(), signs.size());
    }
    std::vector<std::pair<std::string, std::string>> attributes;
    attributes.reserve(energies.size() * 2);
    for (std::size_t axis = 0; axis < energies.size(); ++axis) {
        if (signs[axis] != '+' && signs[axis] != '-') {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "LaplaceTransform::denominator_tag: axis {}'s sign is '{}'; it says whether that energy enters the "
                                    "denominator added or subtracted and is '+' or '-'",
                                    axis, signs[axis]);
        }
        attributes.emplace_back(fmt::format("axis{}", axis), energies[axis]);
        attributes.emplace_back(fmt::format("sign{}", axis), std::string(1, signs[axis]));
    }
    return ProvenanceTag::make_with_attributes(std::string(kTag), std::move(attributes));
}

namespace {

/// One energy vector, held as a stable-address view of whatever the caller passed.
template <typename T, typename TensorType>
std::shared_ptr<void> hold_energy(std::string const &name, TensorType const &vector, std::size_t &extent) {
    // A compile-time-rank tensor answers with a constant and a runtime-rank one with a member,
    // and asking each the way it can answer is cheaper than a concept nobody else needs.
    std::size_t rank = 1;
    if constexpr (requires { vector.rank(); }) {
        rank = vector.rank();
    } else {
        rank = TensorType::Rank;
    }
    if (rank != 1 || vector.dim(0) == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "LaplaceTransform::add_energy('{}'): an orbital-energy axis is a non-empty vector; this one is rank {}",
                                name, rank);
    }
    auto view = std::make_shared<RuntimeTensorView<T>>(vector);
    view->set_name(name);
    extent = vector.dim(0);
    return view;
}

} // namespace

void LaplaceTransform::add_energy(std::string name, RuntimeTensor<double> const &vector) {
    std::size_t extent = 0;
    auto        view   = hold_energy<double>(name, vector, extent);
    record_energy(std::move(name), packed_gemm::ScalarType::Float64, extent, std::move(view));
}

void LaplaceTransform::add_energy(std::string name, RuntimeTensor<float> const &vector) {
    std::size_t extent = 0;
    auto        view   = hold_energy<float>(name, vector, extent);
    record_energy(std::move(name), packed_gemm::ScalarType::Float32, extent, std::move(view));
}

void LaplaceTransform::add_energy(std::string name, Tensor<double, 1> const &vector) {
    std::size_t extent = 0;
    auto        view   = hold_energy<double>(name, vector, extent);
    record_energy(std::move(name), packed_gemm::ScalarType::Float64, extent, std::move(view));
}

void LaplaceTransform::add_energy(std::string name, Tensor<float, 1> const &vector) {
    std::size_t extent = 0;
    auto        view   = hold_energy<float>(name, vector, extent);
    record_energy(std::move(name), packed_gemm::ScalarType::Float32, extent, std::move(view));
}

void LaplaceTransform::record_energy(std::string name, packed_gemm::ScalarType dtype, std::size_t extent, std::shared_ptr<void> view) {
    for (auto const &held : _energies) {
        if (held.name == name) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "LaplaceTransform::add_energy('{}'): a different vector is already registered under that name, and a "
                                    "manifest binds by name",
                                    name);
        }
    }
    _energies.push_back(EnergyVector{.name = std::move(name), .dtype = dtype, .extent = extent, .view = std::move(view)});
}

void LaplaceTransform::clear_energies() {
    _energies.clear();
}

LaplaceTransform::EnergyVector const *LaplaceTransform::energy(std::string const &name) const {
    for (auto const &held : _energies) {
        if (held.name == name) {
            return &held;
        }
    }
    return nullptr;
}

std::string LaplaceTransform::error_tensor_name(std::string const &denominator) {
    return fmt::format("LaplaceTransform.{}.measured_error", denominator);
}

void LaplaceTransform::reset_stats() {
    RegionRewrite::reset_stats();
    _num_transformed = 0;
    _last_points     = 0;
    _last_measured   = 0;
    _pending.clear();
    _claimed.clear();
}

void LaplaceTransform::set_epsilon(double epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "LaplaceTransform::set_epsilon: the target accuracy must be finite and strictly between 0 and 1; got {}",
                                epsilon);
    }
    _epsilon = epsilon;
}

double LaplaceTransform::epsilon() const {
    return _epsilon > 0.0 ? _epsilon : config::get(option::GraphLaplaceEpsilon);
}

void LaplaceTransform::set_points(std::int64_t points) {
    if (points < 0 || points == 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "LaplaceTransform::set_points: {} is not a point count; a trapezoidal rule needs at least two, and zero "
                                "means derive the count from the tolerance",
                                points);
    }
    _points = points;
}

std::vector<std::string> LaplaceTransform::describe() const {
    if (_num_transformed == 0) {
        return {};
    }
    return {fmt::format("LaplaceTransform: replaced {} energy denominator(s) with a {}-point quadrature, measured relative error {:.3e}",
                        _num_transformed, _last_points, _last_measured)};
}

bool LaplaceTransform::applicable(Graph const &graph) const {
    return std::ranges::any_of(graph.tensors_map(), [](auto const &entry) { return entry.second.tag.name == kTag; });
}

bool LaplaceTransform::run(Graph &graph) {
    _pending.clear();
    _claimed.clear();
    bool modified = RegionRewrite::run(graph);

    // After the region loop, never inside it: a region is a range of positions in the node
    // vector and those positions stay live for the whole of the loop above.
    //
    // Position 0, which is correct by construction rather than by luck: the pass only accepts
    // a denominator no node writes and energies no node writes, so a quadrature reading them
    // depends on nothing this graph computes and every reader of its exponentials comes later.
    for (auto const &pending : _pending) {
        Graph &body = graph.add_setup_at(pending.label, 0);
        pending.emit(graph, body);
        modified = true;
    }
    if (!_pending.empty()) {
        report(1, fmt::format("emitted {} setup bod(y/ies) holding the quadratures", _pending.size()));
    }
    _pending.clear();

    // A tag nothing claimed is a decline rather than a silence. It is the only report a
    // SLICED denominator can produce: provenance does not cross a view, so the operand the
    // product actually reads carries no tag and the pass never sees a candidate at all.
    std::vector<std::string> unclaimed;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.tag.name == kTag && std::ranges::find(_claimed, id) == _claimed.end()) {
            unclaimed.push_back(handle.name);
        }
    }
    std::ranges::sort(unclaimed);
    for (auto const &name : unclaimed) {
        note_skip("a tagged denominator has no direct-product consumer whose numerator is a contraction in the same region",
                  fmt::format("tensor '{}'", name));
    }
    _claimed.clear();
    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::quadrature)

bool expression_carries_denominator(Graph const &graph, TensorExpr const &expr) {
    for (auto const &term : expr.terms) {
        if (term.kind != TermKind::Leaf) {
            continue;
        }
        TensorHandle const *handle = graph.find_tensor(term.tensor);
        if (handle != nullptr && handle->tag.name == kTag) {
            return true;
        }
    }
    return false;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity): the recognizer is one argument and
// splitting it would put the halves out of sight of each other.
std::vector<RewriteOutcome> rewrite_denominators(Graph &graph, std::vector<TensorId> const &internal, TensorExpr &expr,
                                                 RewriteOptions const &options, std::vector<TensorId> &claimed) {
    std::vector<RewriteOutcome>                            outcomes;
    std::unordered_map<TensorId, std::vector<std::size_t>> shapes;
    std::vector<std::pair<std::string, std::size_t>>       constants;

    // Every tensor this rewrite makes goes through the caller's factory, which is the one
    // difference between running on a live region and costing a trial.
    auto const make = [&](std::string const &name, packed_gemm::ScalarType dtype, std::vector<std::size_t> const &dims) {
        TensorId const id = options.declare(name, dtype, dims);
        shapes.emplace(id, dims);
        return id;
    };

    // What a decline looks like from here: a record rather than a report, because the two
    // callers say it in different places.
    auto const decline = [&](TensorId id, std::string const &name, std::string reason, std::string detail) {
        outcomes.push_back(
            RewriteOutcome{.applied = false, .reason = std::move(reason), .detail = std::move(detail), .denominator = id, .name = name});
        claimed.push_back(id);
    };

    // At most one rewrite per sweep, and a fresh sweep after each: an accepted rewrite erases
    // the numerator's statement and inserts several, so every index into the statement list is
    // stale afterwards. Bounded by the number of tagged tensors, which is small.
    bool progress = true;
    while (progress) {
        progress = false;

        for (std::size_t position = 0; position < expr.statements.size() && !progress; ++position) {
            ExprStatement const apply = expr.statements[position];
            if (apply.value == invalid_term) {
                continue;
            }
            // BY VALUE, every one of these. Adding a term reallocates the arena and inserting a
            // statement reallocates the statement list, so a reference taken before the rewrite
            // starts dangles the moment the rewrite emits anything.
            ExprTerm const product = expr.at(apply.value);
            if (product.kind != TermKind::Elementwise || product.element_kind != OpKind::DirectProduct || product.operands.size() != 2) {
                continue;
            }

            // Which operand, if either, is a tagged denominator.
            std::size_t denominator_slot = 2;
            for (std::size_t slot = 0; slot < 2; ++slot) {
                TensorHandle const *handle = graph.find_tensor(expr.at(product.operands[slot]).tensor);
                if (handle != nullptr && handle->tag.name == kTag) {
                    denominator_slot = slot;
                    break;
                }
            }
            if (denominator_slot == 2) {
                continue;
            }
            std::size_t const   numerator_slot = 1 - denominator_slot;
            TensorId const      denominator_id = expr.at(product.operands[denominator_slot]).tensor;
            TensorId const      numerator_id   = expr.at(product.operands[numerator_slot]).tensor;
            TensorHandle const *denominator    = graph.find_tensor(denominator_id);
            std::string const   name           = denominator->name;
            if (std::ranges::find(claimed, denominator_id) != claimed.end()) {
                continue;
            }

            if (!is_real_dtype(denominator->dtype)) {
                decline(denominator_id, name,
                        "the tagged denominator is complex, and a reciprocal of a complex sum has no exponential-integral form",
                        fmt::format("tensor '{}'", name));
                continue;
            }
            if (written_anywhere(graph, denominator_id)) {
                decline(denominator_id, name, "the tagged denominator is written by this graph, so its quadrature could go stale",
                        fmt::format("tensor '{}'", name));
                continue;
            }

            // The tag's attributes, one axis at a time. More axes than the tensor has is the
            // folded form: a caller who multiplied two of the energies into a scalar prefactor
            // is asking this pass to carry a bound value as structure.
            std::vector<std::string> energy_names;
            std::vector<int>         signs;
            for (std::size_t axis = 0;; ++axis) {
                auto const energy = denominator->tag.attribute(fmt::format("axis{}", axis));
                auto const sign   = denominator->tag.attribute(fmt::format("sign{}", axis));
                if (!energy.has_value() && !sign.has_value()) {
                    break;
                }
                if (!energy.has_value() || !sign.has_value() || (*sign != "+" && *sign != "-")) {
                    energy_names.clear();
                    break;
                }
                energy_names.push_back(*energy);
                signs.push_back(*sign == "+" ? 1 : -1);
            }
            if (energy_names.empty()) {
                decline(denominator_id, name, "the tag does not carry an axis name and a '+' or '-' sign for every axis",
                        fmt::format("tensor '{}'", name));
                continue;
            }
            if (energy_names.size() != denominator->rank) {
                decline(denominator_id, name,
                        "the tag names more energies than the tagged tensor has axes, which is the pair-driven form whose folded "
                        "energies the graph cannot see",
                        fmt::format("tensor '{}' is rank {} and the tag names {} energies", name, denominator->rank, energy_names.size()));
                continue;
            }

            // The numerator's own statement, which the rewrite dissolves.
            std::size_t numerator_position = expr.statements.size();
            for (std::size_t earlier = 0; earlier < position; ++earlier) {
                if (expr.statements[earlier].target == numerator_id) {
                    numerator_position = earlier;
                }
            }
            if (numerator_position == expr.statements.size()) {
                decline(denominator_id, name,
                        "the denominator's consumer multiplies something this region does not form, so there are no factors to push "
                        "the exponentials onto",
                        fmt::format("tensor '{}'", name));
                continue;
            }
            ExprStatement const numerator = expr.statements[numerator_position];
            ExprTerm const      formation = expr.at(numerator.value);
            if (formation.kind != TermKind::Contraction || formation.operands.size() != 2 || formation.operand_indices.size() != 2) {
                decline(denominator_id, name,
                        "the numerator is not formed by a two-operand contraction, so the quadrature has no operands to ride on",
                        fmt::format("tensor '{}'", name));
                continue;
            }
            if (!is_zero(numerator.target_prefactor)) {
                decline(denominator_id, name,
                        "the numerator accumulates rather than being written outright, so dissolving it would drop what else wrote it",
                        fmt::format("tensor '{}'", name));
                continue;
            }
            if (std::ranges::find(internal, numerator_id) == internal.end()) {
                decline(denominator_id, name,
                        "the numerator is observed from outside the region, and a value someone else reads cannot be dissolved",
                        fmt::format("tensor '{}'", name));
                continue;
            }

            // Nothing between the two statements may write an operand the rewrite is about to
            // read later than it was read before, and nothing else may read the numerator.
            bool interference = false;
            for (std::size_t between = 0; between < expr.statements.size(); ++between) {
                if (between == numerator_position || between == position) {
                    continue;
                }
                ExprStatement const &other = expr.statements[between];
                if (other.value != invalid_term) {
                    for (TermId const operand : expr.at(other.value).operands) {
                        if (expr.at(operand).tensor == numerator_id) {
                            interference = true;
                        }
                    }
                }
                if (between > numerator_position && between < position) {
                    for (TermId const operand : formation.operands) {
                        if (other.target == expr.at(operand).tensor) {
                            interference = true;
                        }
                    }
                    if (other.target == denominator_id || other.target == apply.target) {
                        interference = true;
                    }
                }
            }
            if (interference) {
                decline(denominator_id, name,
                        "another statement reads the numerator or rewrites an operand between its formation and its use",
                        fmt::format("tensor '{}'", name));
                continue;
            }

            // The direct product's own scalars. Its destination prefactor lives in the
            // descriptor rather than in the raised statement, because an elementwise node
            // records accumulation by listing its destination among its inputs and the
            // descriptor is where the number is.
            auto const *scalars = std::get_if<ElementwiseBinaryDescriptor>(&product.descriptor);
            if (scalars == nullptr || !is_real_valued(live_alpha(*scalars)) || !is_real_valued(live_beta(*scalars))) {
                decline(denominator_id, name, "the direct product carries a complex prefactor, which a real quadrature has nowhere to put",
                        fmt::format("tensor '{}'", name));
                continue;
            }

            // The energies, by name, from what the caller handed the pass. Names rather than
            // ids because a manifest binds by name and that is what lets a saved graph refit
            // at a new geometry.
            std::vector<std::shared_ptr<void>>     energy_views;
            std::vector<std::pair<double, double>> extremes;
            std::string                            trouble;
            for (std::size_t axis = 0; axis < energy_names.size(); ++axis) {
                LaplaceTransform::EnergyVector const *held = options.energy(energy_names[axis]);
                if (held == nullptr) {
                    trouble = fmt::format("no vector was registered as '{}'; hand it over with add_energy before applying the pass",
                                          energy_names[axis]);
                    break;
                }
                if (held->dtype != denominator->dtype) {
                    trouble = fmt::format("'{}' and the denominator have different element types", energy_names[axis]);
                    break;
                }
                if (held->extent != denominator->dims[axis]) {
                    trouble = fmt::format("'{}' has {} entries and the denominator's axis {} has {}", energy_names[axis], held->extent,
                                          axis, denominator->dims[axis]);
                    break;
                }
                auto const range = LaplaceTransform::energy_extremes(*held);
                if (!range.has_value()) {
                    trouble = fmt::format("'{}' has no data to read a spectral range from at optimize time", energy_names[axis]);
                    break;
                }
                energy_views.push_back(held->view);
                extremes.push_back(*range);
            }
            if (!trouble.empty()) {
                decline(denominator_id, name, "the tag names an energy vector this pass cannot use",
                        fmt::format("tensor '{}': {}", name, trouble));
                continue;
            }

            // The rule, and the point count it needs. Both are decided HERE and both reach the
            // node, because the emitted node set is sized by the count and a saved graph has
            // to mean one thing wherever it is loaded.
            std::vector<double> low;
            std::vector<double> high;
            for (auto const &[axis_low, axis_high] : extremes) {
                low.push_back(axis_low);
                high.push_back(axis_high);
            }
            double      tolerance = options.epsilon;
            std::size_t count     = 0;
            double      measured  = 0;
            try {
                laplace::SpectralRange const range = laplace::spectral_range(low, high, signs);
                count = options.points > 0 ? static_cast<std::size_t>(options.points) : laplace::quadrature_point_count(range, tolerance);
                measured = laplace::measure_quadrature_error(laplace::build_quadrature(range, tolerance, count));
            } catch (std::exception const &error) {
                decline(denominator_id, name, "no quadrature represents this denominator",
                        fmt::format("tensor '{}': {}", name, error.what()));
                continue;
            }

            // The accuracy statement, before anything is rewritten and before any tensor is
            // made. A refusal here leaves the region exactly as it was and leaves the caller to
            // put its own reason in the tally.
            RewriteOutcome outcome;
            outcome.applied     = true;
            outcome.denominator = denominator_id;
            outcome.name        = name;
            outcome.points      = count;
            outcome.measured    = measured;
            outcome.tolerance   = tolerance;
            outcome.setup_label = fmt::format("LaplaceTransform({})", name);
            if (options.accept && !options.accept(outcome)) {
                claimed.push_back(denominator_id);
                continue;
            }

            // ── Everything below rewrites, and nothing below may decline ──────────────

            std::vector<ExprIndex> const target_indices = numerator.target_indices;
            std::vector<ExprIndex> const a_indices      = formation.operand_indices[0];
            std::vector<ExprIndex> const b_indices      = formation.operand_indices[1];

            std::vector<std::string> used = letters_of(target_indices);
            for (auto const &letter : letters_of(a_indices)) {
                if (!contains(used, letter)) {
                    used.push_back(letter);
                }
            }
            for (auto const &letter : letters_of(b_indices)) {
                if (!contains(used, letter)) {
                    used.push_back(letter);
                }
            }
            ExprIndex const quadrature{.letter = fresh_letter(used, "laplace_t"), .space = SpaceId{}};

            // Every letter of everything about to be emitted, with its extent, so the terms
            // this rewrite builds carry a cost. A term with none reads as free, and a region
            // report then offers a rewrite to nothing as evidence that it paid; it is also
            // what lets a caller cost this rewrite jointly with its own.
            search::LetterTable table;
            auto const          observe = [&](std::vector<ExprIndex> const &indices, TensorId id) {
                TensorHandle const *held = graph.find_tensor(id);
                if (held == nullptr) {
                    return;
                }
                for (std::size_t slot = 0; slot < indices.size() && slot < held->dims.size(); ++slot) {
                    table.observe(indices[slot], held->dims[slot]);
                }
            };
            observe(target_indices, apply.target);
            observe(a_indices, expr.at(formation.operands[0]).tensor);
            observe(b_indices, expr.at(formation.operands[1]).tensor);
            // The quadrature letter is a CONSTANT of the rewrite: its length is fixed by the
            // tolerance and does not move when the problem does. A cost model that gave it a
            // scale variable would rank every decoupled form one scale order worse than it is.
            table.observe_constant(quadrature.letter, count);
            constants.emplace_back(quadrature.letter, count);

            // The exponential matrices, the points and the weights: parent tensors the setup
            // body writes and the rewritten region reads.
            std::vector<TensorId> exponentials;
            for (std::size_t axis = 0; axis < energy_names.size(); ++axis) {
                exponentials.push_back(
                    make(fmt::format("LaplaceTransform.{}.exp{}", name, axis), denominator->dtype, {count, denominator->dims[axis]}));
            }
            TensorId const points_id  = make(fmt::format("LaplaceTransform.{}.points", name), denominator->dtype, {count});
            TensorId const weights_id = make(fmt::format("LaplaceTransform.{}.weights", name), denominator->dtype, {count});
            TensorId const error_id   = make(LaplaceTransform::error_tensor_name(name), denominator->dtype, {1});

            auto make_leaf = [&expr, &graph](TensorId id, std::vector<ExprIndex> indices) {
                ExprTerm            leaf;
                TensorHandle const *held = graph.find_tensor(id);
                leaf.kind                = TermKind::Leaf;
                leaf.tensor              = id;
                leaf.name                = held != nullptr ? held->name : std::string{};
                leaf.indices             = std::move(indices);
                return expr.add(std::move(leaf));
            };

            // Which operand of the numerator carries each axis of the denominator. Every axis
            // is carried by one of them, because the contraction's output IS that index list;
            // an axis both carry is given to the first, so its exponential is applied once.
            std::vector<ExprStatement> scalings;
            struct Side {
                TermId                 leaf;
                std::vector<ExprIndex> indices;
                bool                   scaled{false};
            };
            Side side_a{.leaf = formation.operands[0], .indices = a_indices};
            Side side_b{.leaf = formation.operands[1], .indices = b_indices};

            for (std::size_t axis = 0; axis < target_indices.size(); ++axis) {
                std::string const &letter = target_indices[axis].letter;
                Side              *side   = contains(letters_of(a_indices), letter) ? &side_a : &side_b;

                std::vector<ExprIndex> scaled_indices;
                scaled_indices.push_back(quadrature);
                for (auto const &index : side->indices) {
                    if (index.letter != quadrature.letter) {
                        scaled_indices.push_back(index);
                    }
                }

                // The extent of every letter, read off whichever operand spells that axis.
                // From the handles rather than from anything remembered, because only the
                // graph knows what the numerator's operands are shaped like.
                std::vector<std::size_t> dims;
                for (auto const &index : scaled_indices) {
                    if (index.letter == quadrature.letter) {
                        dims.push_back(count);
                        continue;
                    }
                    std::size_t extent = 0;
                    auto const  from   = [&](std::vector<ExprIndex> const &list, TensorId id) {
                        TensorHandle const *handle = graph.find_tensor(id);
                        if (handle == nullptr) {
                            return;
                        }
                        for (std::size_t slot = 0; slot < list.size() && slot < handle->dims.size(); ++slot) {
                            if (list[slot].letter == index.letter && extent == 0) {
                                extent = handle->dims[slot];
                            }
                        }
                    };
                    from(a_indices, expr.at(formation.operands[0]).tensor);
                    from(b_indices, expr.at(formation.operands[1]).tensor);
                    from(target_indices, apply.target);
                    dims.push_back(extent);
                }

                std::string const scaled_name = fmt::format("LaplaceTransform.{}.scaled{}", name, scalings.size());
                TensorId const    scaled      = make(scaled_name, denominator->dtype, dims);

                ExprTerm term;
                term.kind    = TermKind::Contraction;
                term.indices = scaled_indices;
                term.operands.assign({side->leaf, make_leaf(exponentials[axis], {quadrature, target_indices[axis]})});
                term.operand_indices.assign({side->indices, {quadrature, target_indices[axis]}});
                // The numerator's conjugation rides on the FIRST scaling of that operand and
                // nowhere else, so an operand scaled twice is not conjugated twice.
                bool const conjugate = !side->scaled && !formation.conjugate.empty() && formation.conjugate[side == &side_a ? 0 : 1];
                term.conjugate.assign({conjugate, false});
                term.factor = PrefactorScalar{double{1}};
                term.cost =
                    search::contraction_cost(search::letters_of(side->indices), search::letters_of({quadrature, target_indices[axis]}),
                                             search::letters_of(scaled_indices), table);

                ExprStatement statement;
                statement.target           = scaled;
                statement.target_name      = scaled_name;
                statement.target_indices   = scaled_indices;
                statement.target_prefactor = PrefactorScalar{double{0}};
                statement.value            = expr.add(std::move(term));
                statement.origin           = numerator.origin;
                statement.origin_kind      = OpKind::Einsum;
                statement.origin_label     = fmt::format("LaplaceTransform: {}[{}] = {}[{}] ; exp{}", statement.target_name,
                                                         fmt::join(letters_of(scaled_indices), ","), expr.at(side->leaf).name,
                                                         fmt::join(letters_of(side->indices), ","), axis);
                scalings.push_back(std::move(statement));

                side->leaf    = make_leaf(scaled, scaled_indices);
                side->indices = scaled_indices;
                side->scaled  = true;
            }

            // The contraction the numerator always was, now summing over the quadrature index
            // as well as over whatever it summed over before.
            ExprTerm combined;
            combined.kind    = TermKind::Contraction;
            combined.indices = target_indices;
            combined.operands.assign({side_a.leaf, side_b.leaf});
            combined.operand_indices.assign({side_a.indices, side_b.indices});
            combined.conjugate.assign({!side_a.scaled && !formation.conjugate.empty() && formation.conjugate[0],
                                       !side_b.scaled && !formation.conjugate.empty() && formation.conjugate[1]});
            combined.factor = multiply_real(live_alpha(*scalars), formation.factor);
            combined.cost   = search::contraction_cost(search::letters_of(side_a.indices), search::letters_of(side_b.indices),
                                                       search::letters_of(target_indices), table);

            ExprStatement final_statement;
            final_statement.target           = apply.target;
            final_statement.target_name      = apply.target_name;
            final_statement.target_indices   = target_indices;
            final_statement.target_prefactor = live_beta(*scalars);
            final_statement.value            = expr.add(std::move(combined));
            final_statement.origin           = apply.origin;
            final_statement.origin_kind      = OpKind::Einsum;
            final_statement.origin_label     = fmt::format("LaplaceTransform: {}[{}] over {} quadrature point(s)", apply.target_name,
                                                           fmt::join(letters_of(target_indices), ","), count);

            // Splice: the numerator's statement goes, the scalings and the contraction take
            // the direct product's place. Erasing first and inserting after keeps every index
            // this block still uses valid, which is why the two are not interleaved.
            std::size_t insert_at = position;
            expr.statements.erase(expr.statements.begin() + static_cast<std::ptrdiff_t>(numerator_position));
            if (numerator_position < insert_at) {
                --insert_at;
            }
            expr.statements[insert_at] = std::move(final_statement);
            expr.statements.insert(expr.statements.begin() + static_cast<std::ptrdiff_t>(insert_at), scalings.begin(), scalings.end());

            // The setup body, emitted once the region loop is over.
            std::vector<TensorId> const quadrature_outputs = [&]() {
                std::vector<TensorId> out{points_id, weights_id, error_id};
                out.insert(out.end(), exponentials.begin(), exponentials.end());
                return out;
            }();
            LaplaceQuadratureDescriptor descriptor;
            descriptor.epsilon = tolerance;
            descriptor.points  = static_cast<std::int64_t>(count);
            for (int sign : signs) {
                descriptor.signs.push_back(static_cast<std::int8_t>(sign));
            }
            packed_gemm::ScalarType const dtype = denominator->dtype;
            if (options.want_setup) {
                outcome.emit_setup = [energy_views, quadrature_outputs, descriptor, dtype](Graph &parent, Graph &body) {
                    detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
                        if constexpr (std::is_floating_point_v<T>) {
                            // The outputs are tensors this pass declared, so their handles name
                            // a RuntimeTensor and the cast is what declared them. The energies
                            // are the caller's, held as views at addresses this pass owns, so
                            // capturing them registers one tensor under one name however many
                            // times the pass is applied.
                            auto const owned = [&parent](TensorId id) {
                                return static_cast<RuntimeTensor<T> *>(parent.tensor(id).tensor_ptr);
                            };
                            std::vector<RuntimeTensorView<T> *> energies;
                            energies.reserve(energy_views.size());
                            for (auto const &view : energy_views) {
                                energies.push_back(static_cast<RuntimeTensorView<T> *>(view.get()));
                            }
                            std::vector<RuntimeTensor<T> *> exponentials;
                            for (std::size_t slot = 3; slot < quadrature_outputs.size(); ++slot) {
                                exponentials.push_back(owned(quadrature_outputs[slot]));
                            }
                            CaptureGuard const guard(body);
                            laplace::laplace_quadrature(energies, owned(quadrature_outputs[0]), owned(quadrature_outputs[1]),
                                                        owned(quadrature_outputs[2]), exponentials, descriptor);
                        }
                    });
                };
            }
            outcomes.push_back(std::move(outcome));

            claimed.push_back(denominator_id);
            progress = true;
        }
    }

    for (auto &outcome : outcomes) {
        outcome.shapes           = shapes;
        outcome.constant_letters = constants;
    }
    return outcomes;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::quadrature)

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

bool LaplaceTransform::rewrite(Graph &graph, Region const &region, TensorExpr &expr) {
    quadrature::RewriteOptions options;
    options.epsilon = epsilon();
    options.points  = _points;
    options.energy  = [this](std::string const &wanted) { return energy(wanted); };
    options.declare = [&graph](std::string const &tensor_name, packed_gemm::ScalarType dtype, std::vector<std::size_t> const &dims) {
        return declare_scratch(graph, tensor_name, dtype, dims);
    };

    // The accuracy statement, taken before the splice of the denominator it is about. A refusal
    // here leaves that denominator's region exactly as it was and puts the budget's own reason
    // in the skip tally, which is why it is a callback rather than a check afterwards.
    std::vector<ApproximationRecord> accepted;
    options.accept = [&](quadrature::RewriteOutcome const &offer) {
        ApproximationRecord record = make_approximation_record(name(), ApproximationEffect::NormRelative, offer.tolerance, offer.measured,
                                                               {}, {}, offer.setup_label, ApproximationOrigin::Measured);
        if (!approximate(graph, record)) {
            return false;
        }
        accepted.push_back(std::move(record));
        return true;
    };

    auto const outcomes = quadrature::rewrite_denominators(graph, region.internal, expr, options, _claimed);

    bool        changed = false;
    std::size_t applied = 0;
    for (auto const &outcome : outcomes) {
        if (!outcome.applied) {
            if (!outcome.reason.empty()) {
                note_skip(outcome.reason, outcome.detail);
            }
            continue;
        }
        _pending.push_back(PendingSetup{.label = accepted[applied].setup, .emit = outcome.emit_setup});
        ++applied;
        ++_num_transformed;
        _last_points   = outcome.points;
        _last_measured = outcome.measured;
        changed        = true;
        report(2, fmt::format("replaced '{}' with a {}-point quadrature; measured relative error {:.3e}", outcome.name, outcome.points,
                              outcome.measured));
    }

    if (changed) {
        EINSUMS_LOG_INFO("LaplaceTransform: replaced {} energy denominator(s)", _num_transformed);
    }
    return changed;
}
EINSUMS_NAMESPACE_END(compute_graph::passes)
