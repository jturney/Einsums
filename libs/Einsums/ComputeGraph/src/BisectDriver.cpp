//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/BisectDriver.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/InterfaceManifest.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// @brief The machine epsilon of one element type, as a double.
///
/// Promoted rather than kept per type, because the bound it feeds is a norm-relative figure over a
/// whole tensor and the comparison arithmetic runs in double whatever the operands are.
double epsilon_for(packed_gemm::ScalarType dtype) {
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
    case packed_gemm::ScalarType::Complex64:
        return static_cast<double>(std::numeric_limits<float>::epsilon());
    default:
        return std::numeric_limits<double>::epsilon();
    }
}

/// @brief One interface output, read out of a finished graph.
struct OutputSnapshot {
    std::string                       name;
    bool                              written{false};  ///< Some node in this graph writes it.
    bool                              readable{false}; ///< Its values could be reached.
    std::vector<std::complex<double>> values;
};

/// @brief Does any node of @p graph write @p id, following aliases?
///
/// Aliases are resolved on both sides rather than compared raw: a pass that repoints a reader at a
/// view of the same buffer has not stopped writing the output, and a check that compared ids
/// directly would report the rewrite as a lost output.
bool graph_writes(Graph const &graph, TensorId id) {
    TensorId const root = graph.resolve_alias(id);
    for (auto const &node : graph.nodes()) {
        for (auto const output : node.outputs) {
            if (graph.resolve_alias(output) == root) {
                return true;
            }
        }
    }
    return false;
}

/// @brief Read every element of one tensor, promoted to complex double.
///
/// Promotion is EXACT for all four element types, so two snapshots comparing equal here compare
/// equal in the original bits too, which is what lets one code path serve the bitwise verdict and
/// the norms. Strides are honoured rather than assumed away: an interface output may be a view.
bool read_values(Graph const &graph, TensorId id, std::vector<std::complex<double>> &into) {
    TensorHandle const *handle = graph.find_tensor(id);
    if (handle == nullptr || handle->is_tiled || !handle->impl_fn) {
        return false;
    }
    return detail::dispatch_scalar_type(handle->dtype, [&]<typename T>(T /*tag*/) {
        auto const *impl = static_cast<einsums::detail::TensorImpl<T> const *>(handle->impl_fn());
        if (impl == nullptr || impl->data() == nullptr) {
            return false;
        }
        std::size_t const count = impl->size();
        into.resize(count);
        if (impl->is_contiguous()) {
            for (std::size_t i = 0; i < count; i++) {
                into[i] = static_cast<std::complex<double>>(impl->data()[i]);
            }
            return true;
        }
        std::size_t const        rank = impl->rank();
        std::vector<std::size_t> index(rank, 0);
        for (std::size_t i = 0; i < count; i++) {
            std::size_t offset = 0;
            for (std::size_t axis = 0; axis < rank; axis++) {
                offset += index[axis] * impl->stride(axis);
            }
            into[i] = static_cast<std::complex<double>>(impl->data()[offset]);
            for (std::size_t axis = rank; axis-- > 0;) {
                if (++index[axis] < impl->dim(axis)) {
                    break;
                }
                index[axis] = 0;
            }
        }
        return true;
    });
}

/// @brief Every interface output of @p graph, after it has run.
std::vector<OutputSnapshot> snapshot_outputs(Graph &graph) {
    std::vector<OutputSnapshot> out;
    auto const                  manifest = graph.manifest();
    out.reserve(manifest.outputs.size());
    for (auto const &entry : manifest.outputs) {
        OutputSnapshot snapshot;
        snapshot.name    = entry.name;
        snapshot.written = graph_writes(graph, entry.id);
        if (snapshot.written) {
            snapshot.readable = read_values(graph, entry.id, snapshot.values);
        }
        out.push_back(std::move(snapshot));
    }
    // Sorted by name so a trial and the baseline are matched by the binding key rather than by
    // position, which a rewrite that reorders the manifest would otherwise scramble.
    std::ranges::sort(out, [](OutputSnapshot const &lhs, OutputSnapshot const &rhs) { return lhs.name < rhs.name; });
    return out;
}

/// @brief The gap between two snapshots of one program.
struct Gap {
    bool        comparable{false};
    bool        bitwise{true};
    bool        nonfinite_disagreement{false};
    double      norm_relative{0.0};
    double      max_absolute{0.0};
    std::size_t not_written{0};
    std::string note;
};

bool finite(std::complex<double> const &value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

Gap compare_snapshots(std::vector<OutputSnapshot> const &baseline, std::vector<OutputSnapshot> const &trial) {
    Gap                                                     gap;
    std::unordered_map<std::string, OutputSnapshot const *> by_name;
    by_name.reserve(trial.size());
    for (auto const &snapshot : trial) {
        by_name.emplace(snapshot.name, &snapshot);
    }

    double squared_error = 0.0, squared_reference = 0.0;
    for (auto const &want : baseline) {
        auto const hit = by_name.find(want.name);
        if (hit == by_name.end() || !hit->second->written) {
            // The pass stopped writing this output. That is a legitimate rewrite (a dissolved
            // producer) rather than a wrong number, so it is counted and excluded.
            gap.not_written++;
            continue;
        }
        OutputSnapshot const &got = *hit->second;
        if (!want.readable || !got.readable) {
            gap.note = "an output could not be read";
            continue;
        }
        if (want.values.size() != got.values.size()) {
            gap.note    = fmt::format("output '{}' changed size", want.name);
            gap.bitwise = false;
            continue;
        }

        gap.comparable = true;
        for (std::size_t i = 0; i < want.values.size(); i++) {
            bool const want_finite = finite(want.values[i]);
            bool const got_finite  = finite(got.values[i]);
            if (want_finite != got_finite) {
                // No norm describes this, so it is reported on its own rather than folded into
                // one. `DeltaElimination` is the standing example of a pass that legitimately
                // removes a NaN, which is why this is a flag and not automatically a divergence.
                gap.nonfinite_disagreement = true;
                gap.bitwise                = false;
                continue;
            }
            if (!want_finite) {
                continue;
            }
            if (want.values[i] != got.values[i]) {
                gap.bitwise = false;
            }
            double const difference = std::abs(got.values[i] - want.values[i]);
            gap.max_absolute        = std::max(gap.max_absolute, difference);
            squared_error += difference * difference;
            squared_reference += std::norm(want.values[i]);
        }
    }

    if (gap.comparable) {
        double const reference = std::sqrt(squared_reference);
        gap.norm_relative      = reference > 0.0 ? std::sqrt(squared_error) / reference : std::sqrt(squared_error);
    }
    return gap;
}

/// @brief The largest epsilon over a graph's interface outputs.
///
/// The largest rather than the smallest: a bound has to hold for every output being compared, and
/// a mixed-precision program's float outputs are the ones that move most.
double epsilon_of_outputs(Graph &graph) {
    double worst = std::numeric_limits<double>::epsilon();
    for (auto const &entry : graph.manifest().outputs) {
        worst = std::max(worst, epsilon_for(entry.dtype));
    }
    return worst;
}

} // namespace

std::string_view bisect_mode_name(BisectMode mode) {
    switch (mode) {
    case BisectMode::Individual:
        return "individual";
    case BisectMode::Cumulative:
        return "cumulative";
    }
    return "individual";
}

std::size_t BisectReport::first_divergence() const {
    for (std::size_t i = 0; i < trials.size(); i++) {
        if (trials[i].diverged) {
            return i;
        }
    }
    return trials.size();
}

bool BisectReport::clean() const {
    return baseline_ok && first_divergence() == trials.size();
}

std::string BisectReport::to_string() const {
    if (!baseline_ok) {
        return fmt::format("bisect: the unoptimized run produced nothing to compare against{}{}\n", baseline_note.empty() ? "" : ": ",
                           baseline_note);
    }
    if (trials.empty()) {
        return {};
    }

    std::size_t width = 4;
    for (auto const &trial : trials) {
        width = std::max(width, trial.name.size());
    }

    std::string out = fmt::format("{:<{}}  {:<12}  {:>10}  {:>10}  {}\n", "pass", width, "tier", "norm-rel", "bound", "verdict");
    for (auto const &trial : trials) {
        std::string verdict;
        if (trial.diverged) {
            verdict = "DIVERGED";
        } else if (!trial.modified) {
            verdict = "did not fire";
        } else if (!trial.comparable) {
            verdict = "nothing comparable";
        } else if (trial.bitwise) {
            verdict = "identical";
        } else {
            verdict = "within bound";
        }
        if (trial.nonfinite_disagreement) {
            verdict += " (non-finite differs)";
        }
        if (trial.outputs_not_written != 0) {
            verdict += fmt::format(" ({} output(s) no longer written)", trial.outputs_not_written);
        }
        if (!trial.note.empty()) {
            verdict += fmt::format(" [{}]", trial.note);
        }
        out += fmt::format("{:<{}}  {:<12}  {:>10.3e}  {:>10.3e}  {}\n", trial.name, width, trial.tier, trial.norm_relative, trial.bound,
                           verdict);
    }

    std::size_t const first = first_divergence();
    out += first == trials.size() ? "bisect: no pass moved an output past its tier's bound\n"
                                  : fmt::format("bisect: the first pass past its bound is '{}'\n", trials[first].name);
    return out;
}

BisectDriver::BisectDriver(Builder builder) : _builder(std::move(builder)) {
    if (!_builder) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "BisectDriver: the builder is empty, so there is no program to bisect");
    }
}

BisectDriver::BisectDriver(std::function<void(Graph *)> builder) {
    if (!builder) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "BisectDriver: the builder is empty, so there is no program to bisect");
    }
    _builder = [inner = std::move(builder)](Graph &graph) { inner(&graph); };
}

void BisectDriver::set_passes(std::vector<std::shared_ptr<OptimizerPass>> passes) {
    _passes = std::move(passes);
}

void BisectDriver::set_bound_scale(double scale) {
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "BisectDriver: a bound scale must be positive and finite, got {}. A scale of zero would report every "
                                "re-associating pass as wrong",
                                scale);
    }
    _bound_scale = scale;
}

BisectReport BisectDriver::run() {
    BisectReport report;

    // The pass set, and the two groups every trial keeps switched on. Built from one default
    // pipeline so the analysis passes a trial runs are the ones the user's pipeline would.
    auto                                        defaults = PassManager::create_default();
    std::vector<std::shared_ptr<OptimizerPass>> analysis;
    std::vector<std::shared_ptr<OptimizerPass>> structural;
    for (auto const &pass : defaults.passes()) {
        if (pass->phase() == PassPhase::Analysis) {
            analysis.push_back(pass);
        } else if (pass->phase() == PassPhase::StructuralAlgebraic || pass->phase() == PassPhase::StructuralResource) {
            structural.push_back(pass);
        }
    }
    std::vector<std::shared_ptr<OptimizerPass>> const &candidates = _passes.empty() ? structural : _passes;

    /// Build the graph, run the given passes plus the always-on ones, execute, snapshot.
    auto trial_run = [&](std::string const &label, std::vector<std::shared_ptr<OptimizerPass>> const &under_test,
                         bool &modified) -> std::vector<OutputSnapshot> {
        Graph graph(label);
        _builder(graph);

        // Three managers rather than one, and the reason is the `modified` flag: `PassManager::run`
        // reports whether ANY of its passes changed the graph, and the always-on ones do. Asking
        // the pass under test in a manager of its own is the difference between a report that can
        // say "did not fire" and one that says "identical" about a pass that never ran.
        {
            PassManager support;
            for (auto const &pass : analysis) {
                support.add(pass);
            }
            support.run(graph);
        }
        {
            PassManager manager;
            for (auto const &pass : under_test) {
                manager.add(pass);
            }
            modified = manager.run(graph);
        }
        {
            // Last, and on every side of the comparison: a graph holding a deferred tensor cannot
            // execute without it, and a pass present in both runs contributes nothing to the gap.
            PassManager materialize;
            materialize.add(std::make_shared<passes::Materialization>());
            materialize.run(graph);
        }

        graph.execute();
        return snapshot_outputs(graph);
    };

    bool                        baseline_modified = false;
    std::vector<OutputSnapshot> baseline;
    try {
        baseline = trial_run("bisect-baseline", {}, baseline_modified);
    } catch (std::exception const &error) {
        report.baseline_note = error.what();
        return report;
    }

    double epsilon = std::numeric_limits<double>::epsilon();
    {
        Graph probe("bisect-epsilon");
        _builder(probe);
        epsilon = epsilon_of_outputs(probe);
    }

    report.baseline_ok = std::ranges::any_of(baseline, [](OutputSnapshot const &s) { return s.written && s.readable; });
    if (!report.baseline_ok) {
        report.baseline_note = "the program has no interface output this driver can read";
        return report;
    }

    std::vector<std::shared_ptr<OptimizerPass>> accumulated;
    for (std::size_t k = 0; k < candidates.size(); k++) {
        auto const &pass = candidates[k];

        BisectTrial trial;
        trial.name  = pass->name();
        trial.phase = std::string(pass_phase_name(pass->phase()));
        trial.tier  = std::string(pass_tier_name(pass->tier()));

        if (_mode == BisectMode::Cumulative) {
            accumulated.push_back(pass);
        }
        std::vector<std::shared_ptr<OptimizerPass>> const &under_test =
            _mode == BisectMode::Cumulative ? accumulated : std::vector<std::shared_ptr<OptimizerPass>>{pass};

        std::vector<OutputSnapshot> got;
        try {
            got = trial_run(fmt::format("bisect-{}", trial.name), under_test, trial.modified);
        } catch (std::exception const &error) {
            // A pass that cannot run is itself a finding, and the passes after it are still worth
            // trying, so this is recorded rather than thrown.
            trial.note     = error.what();
            trial.diverged = true;
            report.trials.push_back(std::move(trial));
            continue;
        }

        Gap const gap                = compare_snapshots(baseline, got);
        trial.comparable             = gap.comparable;
        trial.bitwise                = gap.comparable && gap.bitwise;
        trial.nonfinite_disagreement = gap.nonfinite_disagreement;
        trial.norm_relative          = gap.norm_relative;
        trial.max_absolute           = gap.max_absolute;
        trial.outputs_not_written    = gap.not_written;
        if (!gap.note.empty()) {
            trial.note = gap.note;
        }

        double const lossy_tolerance = [&] {
            if (pass->tier() != PassTier::Lossy) {
                return 0.0;
            }
            // A lossy pass is not held to a constant. It declares its own tolerance and the graph
            // composes them per output, which is the only bound that means anything here.
            Graph probe(fmt::format("bisect-tolerance-{}", trial.name));
            _builder(probe);
            PassManager manager;
            for (auto const &support : analysis) {
                manager.add(support);
            }
            for (auto const &member : under_test) {
                manager.add(member);
            }
            manager.run(probe);
            // The RELATIVE half, because the gap this driver measures is norm-relative. An
            // absolute record bounds a different quantity and does not convert into this one,
            // which is the whole point of ApproximationEffect being three kinds and not a number.
            return probe.approximation_tolerance().relative;
        }();

        trial.bound = pass->tier() == PassTier::Lossy ? lossy_tolerance : _bound_scale * tier_bound(pass->tier(), epsilon);
        // A pass that did not fire cannot have moved anything, and reporting it as the culprit
        // because a later pass in cumulative mode did would point at the wrong line.
        trial.diverged = trial.comparable && trial.modified && trial.norm_relative > trial.bound;

        if (trial.diverged) {
            EINSUMS_LOG_WARN("BisectDriver: pass '{}' moved an output by {:.3e}, past its {} bound of {:.3e}", trial.name,
                             trial.norm_relative, trial.tier, trial.bound);
        }
        report.trials.push_back(std::move(trial));
    }

    return report;
}

EINSUMS_NAMESPACE_END(compute_graph)
