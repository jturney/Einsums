//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file BisectDriver.hpp
 * @brief Which pass made the number wrong, in one run.
 *
 * @par The question this answers
 * "CCSD is wrong with optimization on" is a report nobody can act on. It names a pipeline of
 * twenty passes and says one of them, or some interaction between them, changed an answer. Turning
 * that into "`MultiTermFactorization` is wrong on this graph" is a mechanical job: run the program
 * once with no optimization, then once per structural pass with only that pass, and compare. Doing
 * it by hand means twenty edit-build-run cycles and a spreadsheet.
 *
 * @par Why now and not earlier
 * Every structural pass in this library used to be a recognizer: deterministic, individually
 * tested, and wrong only in ways its own test would catch. A driver against that pass set is a
 * slower way to run tests that already exist, which is why this waited. `MultiTermFactorization`
 * is a search whose plan depends on a cost comparison over a whole region, and two of the three
 * passes in its phase now RE-ASSOCIATE, which changes what the question even is: not "which pass
 * changed the number", since a re-associating pass is supposed to, but "which pass changed it past
 * the bound its tier promises". Nothing but a driver asks that.
 *
 * @par What a trial holds fixed
 * Exactly one thing varies between the baseline and a trial: the structural pass under test. Every
 * trial, the baseline included, runs
 *
 *  - the ANALYSIS passes, because they write annotations and never touch the node set, so they
 *    cannot be the cause of a divergence, and because several structural passes read what they
 *    write (`DeltaElimination` recognizes nothing until `ProvenancePropagation` has run); and
 *  - `Materialization`, which is correctness-enabling rather than an optimization, since a graph
 *    holding a deferred tensor cannot execute at all without it.
 *
 * A pass that is absent from both sides cannot contribute to the gap, and a pass present on both
 * sides cannot either. That is the whole of the method.
 *
 * @par Why a builder and not a graph
 * Comparing two optimizations means running the same computation twice from the same starting
 * state, and @ref Graph is non-copyable. Only the caller knows how to produce that state: an
 * accumulating program reads its output tensor's initial contents, so re-running it is not
 * replaying a graph but rebuilding one. Hence the ``Builder`` this takes, a callable that captures
 * the program afresh. That is the same shape the differential fuzz corpus already uses, and it is
 * more general than the alternative: cloning through @ref save_graph_string and rebinding would
 * work for a saveable graph and refuse the ones holding a node the IR cannot reconstruct, which is
 * the wrong direction for a debugging tool.
 *
 * @par What counts as divergence
 * The gap is norm-relative over every interface OUTPUT the two runs both write, and it is measured
 * against @ref tier_bound for the pass's own tier. So a bitwise-exact pass is held to bit equality,
 * a re-associating one to a generous multiple of epsilon, and a lossy one is not held to a constant
 * at all: its bound is what it recorded through @ref Graph::approximation_tolerance.
 *
 * An output the trial stops writing is NOT a divergence. A structural pass may legitimately
 * dissolve a producer, and comparing a buffer nothing filled against one something did would
 * report the pass's whole purpose as its failure; such an output is excluded with a note that says
 * so.
 *
 * @see Optimizer.hpp for @ref tier_bound, the number this drove into existence
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/// @brief How many passes a trial turns on at once.
enum class APIARY_EXPOSE APIARY_MODULE("graph") BisectMode : std::uint8_t{
    /// One pass per trial. Answers "which pass is wrong on its own", which is the question the
    /// design states and the one a deterministic recognizer can fail.
    Individual,
    /// Passes one through k. Answers a different question, "which pass is wrong AFTER the ones
    /// before it", and it is a real one: a pass can be correct alone and wrong on a node set an
    /// earlier pass rewrote, which is the failure mode a phase with three re-associating members
    /// invites and which @ref BisectMode::Individual cannot see.
    Cumulative,
};

/// @brief Name of a @ref BisectMode value.
/// @param[in] mode The mode to name.
/// @return ``"individual"`` or ``"cumulative"``.
[[nodiscard]] EINSUMS_EXPORT std::string_view bisect_mode_name(BisectMode mode);

/// @brief What one pass did to the numbers.
struct APIARY_EXPOSE APIARY_MODULE("graph") BisectTrial {
    APIARY_EXPOSE std::string name;  ///< The pass under test.
    APIARY_EXPOSE std::string phase; ///< Its phase, as @ref pass_phase_name spells it.
    APIARY_EXPOSE std::string tier;  ///< Its tier, as @ref pass_tier_name spells it.

    /// Whether the pass reported changing anything. A pass that did not fire cannot be the cause
    /// of a divergence, and saying so is the difference between "cleared" and "never ran".
    APIARY_EXPOSE bool modified{false};

    /// Whether any interface output could be compared at all.
    APIARY_EXPOSE bool comparable{false};

    /// Largest norm-relative gap over the outputs both runs wrote.
    APIARY_EXPOSE double norm_relative{0.0};

    /// Largest absolute gap over the same outputs.
    APIARY_EXPOSE double max_absolute{0.0};

    /// Whether every compared element was identical.
    APIARY_EXPOSE bool bitwise{false};

    /// The two runs disagree about WHERE the non-finite values are, which no norm describes.
    APIARY_EXPOSE bool nonfinite_disagreement{false};

    /// What @ref norm_relative was held to. See @ref tier_bound.
    APIARY_EXPOSE double bound{0.0};

    /// Whether the gap exceeded @ref bound.
    APIARY_EXPOSE bool diverged{false};

    /// How many interface outputs this trial stopped writing, and so did not compare.
    APIARY_EXPOSE std::size_t outputs_not_written{0};

    /// Anything that needs saying: a trial that threw, an output that could not be read.
    APIARY_EXPOSE std::string note;
};

/// @brief The whole run.
struct APIARY_EXPOSE APIARY_MODULE("graph") EINSUMS_EXPORT BisectReport {
    /// Whether the unoptimized run produced something to compare against. When false nothing else
    /// here means anything, which is why it is asked first.
    APIARY_EXPOSE bool baseline_ok{false};

    /// Why the baseline failed, when it did.
    APIARY_EXPOSE std::string baseline_note;

    /// One per pass tried, in pipeline order.
    APIARY_EXPOSE std::vector<BisectTrial> trials;

    /// @brief Index into @ref trials of the first pass that diverged.
    /// @return The index, or the size of @ref trials when none did.
    ///
    /// FIRST rather than worst: in cumulative mode every later trial inherits the earlier one's
    /// damage, so the largest gap is almost never the culprit.
    APIARY_EXPOSE APIARY_GETTER("first_divergence") [[nodiscard]] std::size_t first_divergence() const;

    /// @brief Whether the baseline ran and nothing diverged.
    /// @return True when there is nothing to report.
    APIARY_EXPOSE APIARY_GETTER("clean") [[nodiscard]] bool clean() const;

    /// @brief The report as a table, one line per pass.
    /// @return The rendering, newline-terminated, empty when there were no trials.
    APIARY_EXPOSE APIARY_GETTER("text") [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Run a program once per structural pass and report which one moved the answer.
 *
 * @par Example (C++)
 * @code
 * cg::BisectDriver driver([&](cg::Graph &graph) {
 *     // Capture the program. Fresh tensors every call: the driver runs this once per trial and
 *     // the trials must not see each other's writes.
 *     auto &T = graph.declare_runtime_tensor<double>("T", {n, n}, true);
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("i,j <- i,k ; k,j", 0.0, &T, 1.0, A, B);
 *     cg::einsum("i,l <- i,j ; j,l", 0.0, &C, 1.0, T, D);
 * });
 * auto const report = driver.run();
 * if (!report.clean()) {
 *     std::cout << report.to_string();
 * }
 * @endcode
 *
 * @par Limitations
 * - The builder must produce an equivalent program on every call, with tensors it does not share
 *   between calls. A builder that closes over one output buffer hands every trial the previous
 *   trial's answer as its starting state, which reports the FIRST pass as the culprit whatever
 *   the truth is.
 * - Only interface outputs are compared. A pass that changes a graph-internal intermediate and
 *   leaves every output alone is reported as clean, which is correct: nothing observes it.
 * - A trial that throws is recorded and does not stop the run, because a pass that cannot run is
 *   itself a finding and the passes after it are still worth trying.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT BisectDriver {
  public:
    /// @brief Captures the program under test into the graph it is handed.
    using Builder = std::function<void(Graph &)>;

    /// @brief Construct over the program @p builder captures.
    /// @param[in] builder Called once per trial. See the class note on what it must guarantee.
    ///
    /// Spelled out rather than as @ref Builder: the documentation extractor renders a parameter's
    /// type as written, and Sphinx's C++ domain cannot resolve an alias used in a signature. The
    /// alias stays for callers who want to name the type.
    explicit BisectDriver(std::function<void(Graph &)> builder);

    /**
     * @brief The spelling Python gets: the same builder, taking a POINTER.
     *
     * @param[in] builder Called once per trial with the graph to capture into. Never null.
     *
     * The difference is not cosmetic and is not a preference. Handing a C++ object to a Python
     * callable goes through pybind's default conversion, which for an lvalue REFERENCE resolves to
     * copy, and @ref Graph is deliberately non-copyable; the call fails at run time with a message
     * about a return value policy, on a constructor that bound perfectly. For a POINTER the same
     * default resolves to reference, which is what is wanted: Python gets a non-owning view of the
     * graph the driver owns.
     *
     * @warning The view is valid only for the duration of the call. A Python builder that stashes
     *          the graph it was handed is holding a dangling pointer once the trial ends.
     */
    APIARY_EXPOSE explicit BisectDriver(std::function<void(Graph *)> builder);

    /// @brief Try these passes instead of the default pipeline's structural ones.
    ///
    /// The primary use, rather than an escape hatch: bisecting the pipeline a caller is ACTUALLY
    /// running is what answers their question, and ``driver.set_passes(pm.passes())`` is how that
    /// is spelled. It is also the only way to reach a pass that is off by default, since a driver
    /// that switched one on would be testing a configuration nobody runs.
    ///
    /// @param[in] passes The passes to try, in the order they would run.
    APIARY_EXPOSE void set_passes(std::vector<std::shared_ptr<OptimizerPass>> passes);

    /// @brief One pass per trial, or passes one through k. See @ref BisectMode.
    /// @param[in] mode The mode.
    APIARY_EXPOSE void set_mode(BisectMode mode) { _mode = mode; }

    /// @brief Multiply every tier bound by @p scale.
    ///
    /// For the caller whose accumulation is long enough that the default is tight, and for the
    /// one who wants to see how much headroom a clean run actually had. Refuses a non-positive
    /// scale: a bound of zero would report every re-associating pass as wrong.
    ///
    /// @param[in] scale The multiplier.
    /// @throws std::invalid_argument If @p scale is not positive and finite.
    APIARY_EXPOSE void set_bound_scale(double scale);

    /// @brief The bound multiplier. See @ref set_bound_scale.
    /// @return The scale, one by default.
    APIARY_EXPOSE APIARY_GETTER("bound_scale") [[nodiscard]] double bound_scale() const { return _bound_scale; }

    /// @brief Run the baseline and every trial.
    /// @return The report.
    APIARY_EXPOSE [[nodiscard]] BisectReport run();

  private:
    Builder                                     _builder;
    std::vector<std::shared_ptr<OptimizerPass>> _passes;
    BisectMode                                  _mode{BisectMode::Individual};
    double                                      _bound_scale{1.0};
};

EINSUMS_NAMESPACE_END(compute_graph)
