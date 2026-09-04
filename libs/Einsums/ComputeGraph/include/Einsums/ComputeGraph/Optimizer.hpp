//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Approximation.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/**
 * @brief Which phase of the pipeline a pass belongs to, and therefore whether a
 *        saved graph may keep its output.
 *
 * The split exists because "does this pass change the node set" and "is this
 * pass's answer valid on another machine" are independent questions, and a
 * two-way structural/tuning label answers only the first. `TiledExpansion`,
 * `GPUPlacement`, `DistributionPlanning`, `InputSlicing` and `SUMMAExpansion`
 * all rewrite the node set for *machine* reasons: called structural, their
 * output gets written into a file that is then wrong on the next machine;
 * called tuning, nothing stops them running before the algebra is final. So
 * they get a phase of their own.
 *
 * What a save persists is exactly @ref PassPhase::StructuralAlgebraic output.
 * A load re-runs the resource and tuning phases over it, through the same code
 * path a fresh capture takes rather than a second one that can drift. The
 * batched-GEMM cost model (whose optimal bucket moves with thread count) and
 * per-node thread widths (whose safety depends on the BLAS vendor) are the two
 * findings that make this mandatory rather than tidy.
 *
 * @see OptimizerPass::phase
 * @see PassManager::structural_pass_manager
 */
enum class PassPhase : std::uint8_t {
    /// Writes annotations onto tensors or nodes and never rewrites the node
    /// set; ``run()`` returns false. Re-runnable at any point, and the manager
    /// re-runs these at the end of a pipeline whose structure changed.
    Analysis,
    /// Machine-independent rewrites of the mathematics. This is the only phase
    /// whose output a saved graph persists, so a pass here must produce a
    /// result that stays *correct* under any cost model or hardware profile.
    StructuralAlgebraic,
    /// Changes the node set for machine-dependent reasons (tiling, placement,
    /// distribution). Never saved; re-derived on load from the algebraic form.
    StructuralResource,
    /// Schedule, memory, batching and thread decisions over a node set that is
    /// already final. Never saved; re-derived on load.
    Tuning,
    /// Read-only reporting. Changes nothing, including annotations.
    Diagnostic,
};

/**
 * @brief Lower-case hyphenated name of a phase, for reports and test tables.
 *
 * @param[in] phase The phase to name.
 * @return One of ``analysis``, ``structural-algebraic``, ``structural-resource``,
 *         ``tuning``, ``diagnostic``.
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view pass_phase_name(PassPhase phase);

/**
 * @brief How close to the unoptimized answer a pass's output has to be.
 *
 * Orthogonal to @ref PassPhase, and the two are easy to conflate. Phase answers
 * "when does this run, and may a save keep its output"; tier answers "what may
 * it do to the numbers". `RegionIdentity` is structural-algebraic AND bitwise;
 * `FactorizationPass` is structural-algebraic AND lossy.
 *
 * The membership here is MEASURED rather than declared, over the six CI legs
 * that carry the tier report (see ``test_tier_opportunities_python.py``). That
 * matters because the obvious classification is wrong: `PermuteFusion` is
 * bit-identical on five of the six and differs on Accelerate, so a table
 * written from any one machine, or from the agreeing majority, would have put
 * it in the wrong tier.
 *
 * @see OptimizerPass::tier
 */
enum class PassTier : std::uint8_t {
    /// Result-identical to the last bit, on every toolchain. Nothing about
    /// which arithmetic happens, or the order it happens in, changes.
    ///
    /// One qualification, which is about NON-FINITE values rather than about
    /// kernels: removing a multiplication also removes its ability to produce a
    /// NaN, so a pass here may still change an answer where an input is
    /// infinite. `DeltaElimination` stays in this tier because that difference
    /// is one-directional, and the tier requires that direction rather than
    /// literal bit equality on every input.
    BitwiseExact,
    /// Mathematically equivalent, not bit-identical, because floating-point
    /// addition and multiplication are not associative. Declares a bound and is
    /// validated against it, never against bit equality.
    ///
    /// A pass that is exact in its ALGEBRA but hands the vendor a different
    /// KERNEL belongs here rather than in the tier above with a caveat
    /// attached. `PermuteFusion` folds a transpose into a `transa` flag, which
    /// is exact reasoning and a different BLAS path, and a tier whose members
    /// each carry their own exception has stopped drawing the line it exists to
    /// draw.
    ReAssociating,
    /// Same operations in the same order, a different schedule. Result-identical
    /// by construction, for a FIXED configuration: a different thread count
    /// changes a reduction tree, so this is a claim about the pass and not about
    /// the machine.
    Tuning,
    /// Trades accuracy for something, under an explicit tolerance, and records
    /// what it did. Never eligible for a default manager.
    Lossy,
};

/**
 * @brief Lower-case hyphenated name of a tier, for reports and test tables.
 *
 * @param[in] tier The tier to name.
 * @return One of ``bitwise-exact``, ``re-associating``, ``tuning``, ``lossy``.
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view pass_tier_name(PassTier tier);

/**
 * @brief The norm-relative gap a pass of @p tier may open, at a dtype whose machine epsilon is
 *        @p epsilon.
 *
 * @param[in] tier    The tier being held to its promise.
 * @param[in] epsilon The element type's machine epsilon.
 * @return The largest norm-relative difference from the unoptimized answer that is still the tier
 *         behaving as described. Zero for the tiers that promise the same arithmetic.
 *
 * @par Why a tier needs a number at all
 * ``PassTier`` was a label until something had to ACT on it. A bisect driver reporting "the first
 * pass whose output diverges past its tier's declared bound" cannot read a label, and the numbers
 * that existed lived in comments in a test table. This is that table's claim, stated where a
 * program can consult it.
 *
 * @par Where the number comes from, and why a constant is honest here
 * @ref PassTier::BitwiseExact and @ref PassTier::Tuning promise the same arithmetic in the same
 * order, so their bound is zero, with the one qualification @ref PassTier::BitwiseExact already
 * carries about non-finite inputs.
 *
 * @ref PassTier::ReAssociating sums the same products in a different order, and the error that
 * opens is ``O(n)`` in the length of the re-associated sum, which is a property of the problem
 * rather than of the pass. So no constant is exactly right. What makes a constant sufficient is
 * the SEPARATION it has to detect: every measured member of this tier sits between 4.6e-17 and
 * 6.5e-16 norm-relative (that is, at most about three epsilon), while a pass that computes the
 * wrong thing produces an error of order one. Twelve orders of magnitude lie between those, so
 * the bound's job is to sit somewhere in the gap rather than to be tight, and a multiple of
 * epsilon generous enough for a long accumulation is still nowhere near a wrong answer.
 *
 * @ref PassTier::Lossy has no constant, and returning one would be a lie: a lossy pass declares
 * its own tolerance and records it, so its bound is @ref Graph::approximation_tolerance and this
 * function reports infinity to say the question belongs elsewhere.
 *
 * @see BisectDriver.hpp, the consumer that made this a number
 */
[[nodiscard]] EINSUMS_EXPORT double tier_bound(PassTier tier, double epsilon);

/**
 * @brief A wall-clock allowance for one pass's run.
 *
 * Every pass in this library up to now has been cheap and deterministic, so nothing needed one.
 * A SEARCH pass does: its runtime is a function of how many candidates a graph offers rather than
 * of how many nodes it has, and a graph that offers too many is not an error condition anyone can
 * predict from the outside. The contract is therefore that a search pass checks this, stops when
 * it is spent, keeps the best candidate it had found, and SAYS it was cut off, so a report never
 * reads the same for "this graph was already optimal" and "I ran out of time on the first
 * candidate".
 *
 * Default-constructed means unlimited, which is what every non-searching pass gets and what a
 * budget of zero requests explicitly.
 *
 * Deliberately not a thread count, a node count, or a candidate count. The thing that has to be
 * bounded is the wait, and the other three are proxies whose relation to it changes with the
 * machine.
 */
class SearchBudget {
  public:
    /// @brief An unlimited budget.
    SearchBudget() = default;

    /// @brief A budget of @p allowance from now.
    /// @param[in] allowance How long the pass may take. Zero or negative means unlimited.
    explicit SearchBudget(std::chrono::milliseconds allowance)
        : _unlimited{allowance <= std::chrono::milliseconds::zero()}, _deadline{std::chrono::steady_clock::now() + allowance} {}

    /// @brief Whether this budget bounds anything.
    /// @return True when the pass may take as long as it likes.
    [[nodiscard]] bool unlimited() const noexcept { return _unlimited; }

    /// @brief Whether the allowance is gone.
    /// @return True when the pass must stop. Always false for an unlimited budget.
    [[nodiscard]] bool expired() const noexcept { return !_unlimited && std::chrono::steady_clock::now() >= _deadline; }

    /// @brief What is left of the allowance.
    /// @return The remaining time, zero once expired, and zero for an unlimited budget, which
    ///         callers must distinguish with @ref unlimited rather than by testing this.
    [[nodiscard]] std::chrono::milliseconds remaining() const noexcept {
        if (_unlimited) {
            return std::chrono::milliseconds::zero();
        }
        auto const left = std::chrono::duration_cast<std::chrono::milliseconds>(_deadline - std::chrono::steady_clock::now());
        return left > std::chrono::milliseconds::zero() ? left : std::chrono::milliseconds::zero();
    }

  private:
    bool                                  _unlimited{true};
    std::chrono::steady_clock::time_point _deadline{};
};

/**
 * @brief One of a pass's statistics, snapshotted at the top of its @ref OptimizerPass::run.
 *
 * @par The problem this exists for
 * ``PassManager::run`` zeroes a pass's counters ONCE per ``apply()``, then invokes ``run()``
 * once per SUB-GRAPH (see @ref OptimizerPass::reset_stats for why the reset cannot move into
 * ``run``). So a counter is a running total across the whole sub-graph tree, and a ``run()``
 * that answers ``return _num_fused > 0;`` reports THIS graph as modified whenever any EARLIER
 * sub-graph changed something. That is silent for a single loop, whose body is visited last,
 * and wrong the moment a graph holds a conditional or a second loop.
 *
 * Every pass that reports a statistic had independently discovered this and written the same
 * three lines - a local ``_at_entry`` snapshot, a comparison, and a four-line comment
 * explaining it - twenty times over. This is those three lines, once, where the next pass
 * author will find them.
 *
 * @code
 * bool ElementWiseFusion::run(Graph &graph) {
 *     PassCounter const fused{_num_fused};
 *     // ... fuse, incrementing _num_fused ...
 *     return fused.moved();
 * }
 * @endcode
 *
 * Read-only: the pass keeps incrementing its own member as before, and this only observes.
 * That keeps a conversion to it a two-line change per pass rather than a rewrite of every
 * increment site.
 *
 * @note Holds a POINTER to the counter, so it must not outlive it. Declare it as a local in
 *       ``run()``, which is the only place it makes sense: it is a snapshot of one invocation.
 */
class PassCounter {
  public:
    /// @brief Snapshot @p counter as it stands now.
    /// @param[in] counter The pass member to watch. Must outlive this object.
    explicit PassCounter(std::size_t const &counter) noexcept : _counter(&counter), _entry(counter) {}

    /// @brief How much this ``run()`` has added to the counter.
    /// @return The delta since construction.
    ///
    /// This is the number a per-graph report wants, where the counter itself holds the total
    /// over every sub-graph visited so far.
    [[nodiscard]] std::size_t delta() const noexcept { return *_counter - _entry; }

    /// @brief Whether this ``run()`` moved the counter at all.
    /// @return True when it did, which is what ``run()`` returns.
    [[nodiscard]] bool moved() const noexcept { return delta() != 0; }

  private:
    std::size_t const *_counter;
    std::size_t        _entry;
};

/**
 * @brief Abstract base class for optimization passes over a computation graph.
 *
 * An optimization pass inspects a Graph's node list and may modify it to improve
 * performance. Passes can fuse operations, eliminate redundancies, reorder nodes
 * for better memory behavior, or analyze the graph and report recommendations.
 *
 * @par Implementing a custom pass
 * @code
 * class MyPass : public OptimizerPass {
 * public:
 *     std::string name() const override { return "MyPass"; }
 *     bool run(Graph &graph) override {
 *         // Inspect and modify graph.nodes()
 *         return false;
 *     }
 * };
 * @endcode
 *
 * @note Passes that modify the graph should call graph.mark_sorted() if they
 *       produce a valid topological ordering, to prevent unnecessary re-sorting.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) OptimizerPass {
  public:
    virtual ~OptimizerPass() = default;

    /// @name Special members
    /// Spelled out because the virtual destructor suppresses the implicit move constructor: without
    /// these, moving a derived pass (``Graph::apply`` returns one by value) COPIES this base, and
    /// copying @ref _skips allocates. Defaulted, so the behaviour is the compiler's either way.
    /// @{
    OptimizerPass()                                     = default;
    OptimizerPass(OptimizerPass const &)                = default;
    OptimizerPass(OptimizerPass &&) noexcept            = default;
    OptimizerPass &operator=(OptimizerPass const &)     = default;
    OptimizerPass &operator=(OptimizerPass &&) noexcept = default;
    /// @}

    /// @brief Human-readable pass name, exposed so Python tests can
    ///        assert which pass they're invoking.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Run the optimization pass on the given graph.
     * @param[in,out] graph The graph to optimize.
     * @return True if the graph was modified, false if no changes were made.
     */
    virtual bool run(Graph &graph) = 0;

    /**
     * @brief Which pipeline phase this pass belongs to.
     *
     * The default is @ref PassPhase::Tuning, which is the phase whose output is
     * never saved and always re-derived. That is deliberate: a new pass that
     * nobody classified must not have its output silently written into a graph
     * file, so the safe answer is the one that costs a re-run rather than a
     * wrong number on the next machine. Every pass in this library overrides
     * it, and ``PassPhases.cpp`` holds a hard-coded table that fails when one
     * does not, so the default is a backstop and not a convention.
     *
     * Two obligations come with the label:
     *
     * - @ref PassPhase::Analysis and @ref PassPhase::Diagnostic passes must not
     *   change the node set. ``PassManager::run`` watches
     *   ``Graph::structure_version`` around every pass and throws when one of
     *   these moves it, so the rule is mechanical rather than a comment.
     * - @ref PassPhase::StructuralAlgebraic output is what a save persists, so
     *   a pass here may consult a cost model only as a *hint*: its output must
     *   remain valid, if not optimal, under a different one.
     */
    [[nodiscard]] virtual PassPhase phase() const { return PassPhase::Tuning; }

    /**
     * @brief What this pass may do to the numbers. See @ref PassTier.
     *
     * Defaults to @ref PassTier::Tuning, which is the honest default for the
     * same reason the phase default is: a pass nobody has classified is a pass
     * nobody has measured, and Tuning is the tier that claims least about the
     * algebra while still promising the arithmetic is untouched.
     *
     * A pass that CHANGES the arithmetic must say so. Leaving this default on
     * one that re-associates is the failure this is worth guarding against, and
     * the tier table in ``PassPhases.cpp`` is what turns that into a test
     * rather than a comment.
     */
    [[nodiscard]] virtual PassTier tier() const { return PassTier::Tuning; }

    /**
     * @brief Should ``PassManager`` re-invoke this pass on every sub-graph?
     *
     * Controls whether the PassManager automatically descends into
     * loop bodies and conditional branches (via
     * ``Graph::for_each_subgraph``) and calls ``run()`` again at each
     * level. Default ``false``, preserves the historical flat-graph
     * behavior.
     *
     * Passes whose semantics are *correct on a flat sub-graph* (CSE,
     * ScaleAbsorption, PermuteFusion, …) should return ``true`` once
     * verified: that lets the same per-pass tests cover bodies.
     *
     * Passes whose effect must *cross the loop boundary* (Materialization
     * hoisting allocs to the parent, FreeInsertion placing frees after
     * the loop, TransferInsertion hoisting H2D for loop-invariant inputs)
     * should keep this ``false`` and instead walk children themselves
     * inside ``run()`` via ``Graph::for_each_subgraph``. Their output
     * lands in the *parent* graph, not the child.
     *
     * Passes that need parent context to decide correctness must also keep
     * this ``false`` until they grow the required cross-graph reasoning.
     * ``DeadNodeElimination`` and ``CSE`` are the current instances, and both
     * opt out here while walking the tree THEMSELVES inside ``run()``, which
     * is not the same as not running on bodies.
     *
     * Two passes this paragraph used to name have since earned their way in,
     * and what they had to grow is the useful part:
     * ``ConstantFolding`` folds only nodes whose tensors are materialized at
     * pass time, so a body's deferred workspace is skipped rather than
     * executed against unallocated storage;
     * ``Reorder`` encodes WAR edges (reader → writer) and not just
     * ``last_writer``, which is exactly what keeps a loop-carried
     * read-before-write ahead of the write that would otherwise overtake it.
     *
     * Known gap, measured rather than assumed: the program-order guard in
     * ``PassManager::run`` checks the TOP-LEVEL graph only, so a pass that
     * returns ``true`` here rewrites bodies with that guard switched off.
     * Extending the guard over the sub-graph tree was tried on 2026-08-08 and
     * withdrawn. It reports false positives in bodies, because its
     * "was there an earlier writer of this buffer in scan order" test is a
     * proxy for "this read has a producer", and Reorder - which builds real
     * RAW edges and never inverts a genuine producer/consumer pair - trips the
     * proxy without changing semantics. Sixteen control-flow fuzz programs
     * flagged, all four dtypes, and all 798 numerically identical with the
     * check disabled. Closing the gap needs the guard to reason from real
     * dependence edges rather than scan position, which is a bigger change
     * than the guard is.
     *
     * One thing recursion does NOT buy you: a rewrite that bakes a lambda over
     * ``TensorHandle::tensor_ptr`` is wrong at every level, loop or not. Use
     * ``Graph::live_tensor_ptr``.
     */
    [[nodiscard]] virtual bool recurse_into_subgraphs() const { return false; }

    /**
     * @brief Zero this pass's per-apply statistics.
     *
     * ``PassManager::run`` calls this ONCE per ``apply()``, before descending,
     * so the counter getters report totals over the whole subgraph tree rather
     * than whatever the last-visited subgraph happened to contribute.
     *
     * Any pass that maintains counters must override this and must NOT zero
     * them inside ``run()``: with ``recurse_into_subgraphs() == true`` the
     * driver calls ``run()`` once per subgraph, so a reset there discards every
     * earlier level. That is silent for a single loop (the body is visited
     * last, so its count survives) and wrong the moment a graph has a
     * conditional -- the empty else-branch resets the counters to zero -- or
     * more than one loop. ``graph.explain()`` reads these getters.
     *
     * The corollary is that a counter is a RUNNING TOTAL inside one ``apply()``, so ``run()``
     * cannot test it against zero to decide what it returns. @ref PassCounter is that test.
     */
    virtual void reset_stats() {}

    /**
     * @brief Zero everything the manager owns, then the pass's own counters.
     *
     * ``PassManager::run`` calls THIS, not @ref reset_stats, so the skip-reason
     * tally is cleared once per ``apply()`` without every pass having to
     * remember to chain to a base implementation.
     */
    void reset_all_stats() {
        _skips.clear();
        reset_stats();
    }

    /**
     * @brief Why this pass declined the candidates it looked at.
     *
     * One entry per distinct reason, with the number of candidates that hit it,
     * ordered most-frequent first. Populated by @ref note_skip during
     * ``run()``; empty for a pass that never declines anything or has not been
     * taught to explain itself yet.
     *
     * This is the negative half of @ref explain: `explain()` says what a pass
     * DID, `skip_reasons()` says what it declined and why. A pipeline that
     * looks inert usually has a full skip tally - that is the useful signal,
     * and reading it should not require rebuilding with a debugger attached.
     *
     * @return One (reason, count) pair per distinct reason.
     *
     * Exposed because "reading it should not require a debugger" is a claim
     * about every language the library is used from, and the passes whose whole
     * behaviour is a list of refusals are the ones a caller most needs it for.
     * The report @ref PassManager::explain builds carries the REGION-level
     * declines only, so a per-candidate reason is otherwise reachable from C++
     * and nowhere else.
     */
    APIARY_EXPOSE APIARY_GETTER("skip_reasons") [[nodiscard]] EINSUMS_EXPORT std::vector<std::pair<std::string, std::size_t>>
                                                                             skip_reasons() const;

    /**
     * @brief What this pass did on its last run, for @ref PassManager::explain.
     *
     * One entry per line of the report; empty means there is nothing worth
     * saying, which is the default and what a pass with no counters wants. A
     * pass that did nothing should return empty rather than a line saying so,
     * so a quiet report means a quiet pipeline.
     *
     * Lives here, next to the counters it reads, rather than in a type switch
     * inside the manager: a pass that grows a statistic updates one file, and a
     * pass defined outside this library gets summarized like any other.
     */
    [[nodiscard]] virtual std::vector<std::string> explain() const { return {}; }

    /**
     * @brief Reads this pass intentionally redirected to a tensor's INITIAL
     *        contents by compensating the reader, exempted from the
     *        program-order validator.
     *
     * The validator in ``PassManager::run`` throws when a read that observed an
     * in-graph writer flips to observing the tensor's initial contents - the
     * writer-removed-under-a-reader bug class. A pass that removes a
     * writer and instead COMPENSATES the reader (e.g. ScaleAbsorption deleting a
     * ``scale`` and folding its factor into a downstream einsum's
     * ``ab_prefactor``, so the read is exact despite losing the writer) declares
     * the affected ``(reader NodeId, TensorId)`` pairs here; the validator skips
     * exactly those. The pass owns the compensation's correctness — its own
     * numeric tests must cover it, since the structural guard is waived. Return
     * the pairs recorded during ``run()``; empty by default (no exemptions).
     */
    [[nodiscard]] virtual std::vector<std::pair<NodeId, TensorId>> compensated_reads() const { return {}; }

    /**
     * @brief Set the pass's introspection verbosity.
     *
     * Levels: 0 = silent (default), 1 = summary (aggregate effect),
     * 2 = detail (each modification applied), 3 = trace (each candidate
     * examined, including why it was rejected). Output goes to stderr,
     * prefixed with the pass name. Usually set in bulk via
     * ``PassManager::set_verbosity`` rather than per pass.
     *
     * From level 2 up, ``PassManager::explain()`` also grows a "not applied"
     * section built from @ref skip_reasons - the aggregated form of what
     * level 3 narrates per candidate. Reach for that first when a pipeline
     * looks inert: the tally names the gate, and is a returned string rather
     * than stderr noise you have to read past.
     */
    APIARY_EXPOSE void set_verbosity(int level) { _verbosity = level; }

    /// @brief Current verbosity level (see set_verbosity).
    APIARY_EXPOSE APIARY_GETTER("verbosity") [[nodiscard]] int verbosity() const { return _verbosity; }

    /**
     * @brief Give this pass a wall-clock allowance for its next @ref run.
     *
     * ``PassManager::run`` sets one before every pass, freshly, so an allowance is per-run and
     * not per-pipeline; a pass that kept a deadline across runs would give the second graph
     * whatever the first one left. Passes that do not search ignore it.
     *
     * @param[in] budget The allowance. A default-constructed one is unlimited.
     */
    void set_budget(SearchBudget budget) { _budget = budget; }

    /// @brief The allowance for this run.
    /// @return The budget the manager set, or an unlimited one when nobody set anything.
    [[nodiscard]] SearchBudget const &budget() const noexcept { return _budget; }

  protected:
    /**
     * @brief Emit ``[PassName] message`` to stderr when ``_verbosity >= level``.
     *
     * Passes call this to narrate what they see and the rewrites they make:
     * @code
     * report(3, fmt::format("examining node {} ({})", i, node.label)); // trace
     * report(2, fmt::format("folded {} contractions into {}", n, name)); // detail
     * report(1, fmt::format("folded {} groups", _num_groups));          // summary
     * @endcode
     */
    EINSUMS_EXPORT void report(int level, std::string_view message) const;

    /**
     * @brief Record that a candidate was examined and declined, and why.
     *
     * ``reason`` must be a short, *shape-independent* phrase so repeated hits
     * aggregate into one counted line - "operands are not runtime tensors",
     * not "tensor 'Wmbej' (id=17) is not a runtime tensor". Put the specifics
     * in ``detail``, which is emitted to stderr at verbosity 3 but never
     * aggregated:
     *
     * @code
     * note_skip("operands are not runtime tensors",
     *           fmt::format("output '{}' has is_runtime=false", name));
     * @endcode
     *
     * Cheap enough to call on every rejected candidate: the aggregate is a
     * small map keyed by reason, and the stderr line is formatted only when
     * verbosity warrants it.
     */
    EINSUMS_EXPORT void note_skip(std::string_view reason, std::string_view detail = {}) const;

    /**
     * @brief Ask whether this pass may apply a lossy rewrite, and record it when it may.
     *
     * @param[in,out] graph The graph the rewrite would be applied to.
     * @param[in] record What the rewrite is and what it would cost. See @ref ApproximationRecord.
     * @return True when the record was accepted and the pass may proceed. False when it was
     *         refused, in which case the reason has already gone into the skip tally and the
     *         pass must leave the graph alone.
     *
     * The whole of Part 5.2's "a pass that cannot bound its effect under already-applied
     * approximations declines with a reported reason", in one place, so that every lossy
     * pass declines the same way rather than each inventing its own.
     *
     * Called BEFORE the rewrite, not after. @ref Graph::note_approximation throws on a
     * refusal precisely so that a pass which rewrote first and asked afterwards cannot
     * quietly carry on: at that point the graph is already wrong and there is nothing a
     * return value could do about it.
     *
     * @code
     * if (!approximate(graph, {.pass = name(), .tolerance = _epsilon,
     *                          .effect = ApproximationEffect::NormRelative, .bound = estimated})) {
     *     return false;
     * }
     * @endcode
     * @versionadded{2.0.0}
     */
    EINSUMS_EXPORT bool approximate(Graph &graph, ApproximationRecord record) const;

    int          _verbosity{0};
    SearchBudget _budget;

  private:
    /// Reason -> number of candidates declined for it. Mutable so a pass can
    /// record from a const analysis helper; cleared per apply() by
    /// reset_all_stats().
    mutable std::vector<std::pair<std::string, std::size_t>> _skips;
};

/**
 * @brief Optimization level for Graph::optimize(), compiler-style.
 *
 * - O0: no passes; the graph runs exactly as captured.
 * - O1: node-count cleanup only (constant folding, dead-scale removal,
 *       permute fusion, CSE, dead-node elimination, elementwise fusion) -
 *       cheap, no restructuring, no memory planning.
 * - O2: the full default pipeline (cleanup + loop hoisting + cost-model
 *       chain restructuring + batching + reorder + distribution/GPU when
 *       available + in-place merging + free insertion + the memory arena).
 */
enum class APIARY_EXPOSE OptLevel : std::uint8_t {
    O0 = 0,
    O1 = 1,
    O2 = 2,
};

/**
 * @brief Manages an ordered sequence of optimization passes.
 *
 * The PassManager collects passes and runs them in order on a graph.
 * Use add() to append individual passes, or use create_default() to
 * get a PassManager pre-loaded with all built-in passes in the
 * recommended order.
 *
 * @par Example: custom pass pipeline
 * @code
 * cg::PassManager pm;
 * pm.add<cg::passes::ConstantFolding>();
 * pm.add<cg::passes::ScaleAbsorption>();
 * pm.add<cg::passes::CSE>();
 * pm.add<cg::passes::Reorder>();
 *
 * graph.apply(pm);
 * @endcode
 *
 * @par Example: default pass pipeline
 * @code
 * auto pm = cg::PassManager::create_default();
 * graph.apply(pm);
 * @endcode
 *
 * @see Graph::apply(PassManager&)
 * @see Pipeline::apply(PassManager&)
 * @see create_for(OptLevel)
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE EINSUMS_EXPORT PassManager {
  public:
    /// Default-construct an empty PassManager. Explicit (rather than
    /// implicit) so the binding codegen has a constructor declaration to
    /// annotate with APIARY_EXPOSE.
    APIARY_EXPOSE PassManager() = default;

    // The vector-of-shared_ptr storage makes PassManager copyable; we
    // don't promise that to C++ callers (NOCOPY/NOMOVE controls the
    // Python binding), but no explicit delete is needed for the binding
    // to compile.

    /**
     * @brief Add a pass to the end of the pipeline.
     *
     * The pass is constructed in-place and owned by the PassManager.
     *
     * @tparam PassType The pass class (must derive from OptimizerPass).
     * @tparam Args Constructor argument types.
     * @param[in] args Arguments forwarded to the pass constructor.
     * @return Reference to this PassManager (for chaining).
     *
     * @code
     * pm.add<cg::passes::CSE>()
     *   .add<cg::passes::Reorder>()
     *   .add<cg::passes::MemoryPlanning>();
     * @endcode
     */
    template <typename PassType, typename... Args>
    PassManager &add(Args &&...args) {
        auto pass = std::make_shared<PassType>(std::forward<Args>(args)...);
        if (_verbosity != 0) {
            pass->set_verbosity(_verbosity);
        }
        _passes.push_back(std::move(pass));
        return *this;
    }

    /**
     * @brief Non-templated overload taking shared ownership of a pass.
     *
     * Exists so the Python binding can write ``pm.add(cg.CSE())``: the
     * templated form can't be bound directly because pybind11 has no way
     * to deduce ``PassType`` from a Python call site. C++ callers should
     * prefer ``add<PassType>(...)`` since it constructs in place. Stored
     * as ``shared_ptr`` to match the pybind11 holder type, pybind can't
     * transfer ownership of a Python-held ``unique_ptr`` across the FFI
     * boundary without ``py::smart_holder``.
     */
    // The parameter name ``optimizer_pass`` rather than ``pass`` so the
    // pyi codegen (which copies parameter names verbatim) doesn't emit
    // a method signature whose argument name collides with Python's
    // ``pass`` keyword, pyright can't parse it.
    APIARY_EXPOSE APIARY_RVP(reference_internal) PassManager &add(std::shared_ptr<OptimizerPass> optimizer_pass) {
        if (_verbosity != 0) {
            optimizer_pass->set_verbosity(_verbosity);
        }
        _passes.push_back(std::move(optimizer_pass));
        return *this;
    }

    /**
     * @brief Switch one pass off by name, without removing it from the pipeline.
     *
     * The programmatic half of ``einsums:pass:disable``. Both exist because they
     * answer to different people: the option is what a user reaches for from a
     * command line with no rebuild, and this is what a driver reaches for when it
     * has to run the same pipeline N times with a different pass switched off each
     * time. A driver that had only the option would have to mutate process-global
     * configuration to do its job.
     *
     * The name is the pass's @ref OptimizerPass::name, and it need not be in the
     * pipeline yet: switches are consulted at @ref run, so disabling before
     * @ref populate_default works. A name that still matches no pass when
     * ``run()`` finishes is REPORTED through @ref explain rather than ignored -
     * a mistyped pass name that silently does nothing is the failure mode this
     * whole surface exists to avoid.
     *
     * @param[in] pass_name The pass to switch off.
     * @return Reference to this PassManager, for chaining.
     *
     * @code
     * pm.disable("CSE").disable("Reorder");
     * @endcode
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) PassManager &disable(std::string pass_name);

    /**
     * @brief Switch one pass back on, overriding ``einsums:pass:disable``.
     *
     * Undoes a @ref disable, and also overrides the option: an explicit call here
     * beats an environment variable, because the more specific statement about
     * this pipeline should win, and a program that asked for a pass by name and
     * silently did not get it is the surprise this surface is meant to remove.
     *
     * @param[in] pass_name The pass to switch on.
     * @return Reference to this PassManager, for chaining.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) PassManager &enable(std::string pass_name);

    /**
     * @brief The passes the last @ref run actually skipped, in pipeline order.
     *
     * Empty before the first run, and empty on a run that skipped nothing.
     * Reports what HAPPENED rather than what was requested, so a name in
     * ``einsums:pass:disable`` that matched no pass does not appear here; see
     * @ref unmatched_switches for those.
     *
     * @return The skipped passes' names.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> disabled_passes() const { return _last_skipped; }

    /**
     * @brief Switch names, from either source, that matched no pass in the pipeline.
     *
     * A mistyped name is indistinguishable from a correctly typed one for a pass
     * this build does not have, and both produce a pipeline that quietly did not
     * do what was asked. Sorted, so the report is diffable.
     *
     * @return The unmatched names.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> unmatched_switches() const { return _last_unmatched; }

    /**
     * @brief Set introspection verbosity for every pass in the pipeline.
     *
     * Propagates @p level to all currently-registered passes and to any added
     * afterward, and makes ``run()`` print a per-pass summary line
     * (``name: MODIFIED N -> M nodes (T ms)``) to stderr. See
     * ``OptimizerPass::set_verbosity`` for the level meanings.
     *
     * @code
     * pm = cg.default_pass_manager()
     * pm.set_verbosity(2)   # narrate each modification
     * g.apply(pm)
     * @endcode
     */
    APIARY_EXPOSE void set_verbosity(int level) {
        _verbosity = level;
        for (auto &pass : _passes) {
            pass->set_verbosity(level);
        }
    }

    /**
     * @brief Give every pass in this pipeline a wall-clock allowance for its run.
     *
     * The programmatic half of ``einsums:graph:optimizer-budget``, and it exists for the same
     * reason ``disable`` does: a driver that wanted to bound one pipeline would otherwise have to
     * mutate process-global configuration to do it, which also makes it unrunnable concurrently.
     * An explicit setting wins over the option, because the more specific statement about this
     * pipeline is the one that should.
     *
     * @param[in] allowance How long each pass may take. Zero or negative means unlimited.
     */
    APIARY_EXPOSE void set_optimizer_budget(std::int64_t allowance_ms) {
        _budget_ms       = allowance_ms;
        _budget_explicit = true;
    }

    /**
     * @brief Run all passes in order on the given graph.
     *
     * @param[in,out] graph The graph to optimize.
     * @return True if any pass modified the graph.
     */
    APIARY_EXPOSE bool run(Graph &graph);

    /**
     * @brief Get the list of passes for inspection.
     * @return Const reference to the pass list.
     */
    [[nodiscard]] std::vector<std::shared_ptr<OptimizerPass>> const &passes() const { return _passes; }

    /**
     * @brief Number of passes in the pipeline.
     */
    APIARY_EXPOSE APIARY_GETTER("size") [[nodiscard]] size_t size() const { return _passes.size(); }

    /**
     * @brief Create a PassManager with all built-in passes in recommended order.
     *
     * Ordering rationale: lowering first (TiledExpansion, so every pass below
     * sees dense nodes), then graph-transforming cleanups (fold/absorb/fuse/
     * eliminate), then planning and fusion, then scheduling (Reorder/
     * IOPrefetch), then deferred materialization, then backend placement (GPU,
     * distributed), then memory management. The GPU and distributed blocks are
     * compile-time-gated by backend availability; the numbering below assumes
     * both are present. populate_default() in Optimizer.cpp carries the
     * per-pass ordering rationale.
     *
     * Each entry names its @ref PassPhase. The sequence is *not* sorted by
     * phase, and one entry is a recorded deviation from the phase rule: see
     * ``passes::TiledExpansion::phase`` for why lowering runs ahead of the
     * algebraic cleanups it feeds. The phase labels are what
     * @ref structural_pass_manager and friends filter on, and what a saved
     * graph consults to decide which output it may keep.
     *
     *  1. TiledExpansion (structural-resource): lower tiled ops into per-tile dense nodes
     *  2. ConstantFolding (structural-algebraic): evaluate constant-input nodes at compile time
     *  3. ScaleAbsorption (structural-algebraic): drop a Scale(α) made dead by the next op overwriting it
     *  4. PermuteFusion (structural-algebraic): absorb leading permutes into the GEMM trans flags
     *  5. CSE (structural-algebraic): common subexpression elimination
     *  6. DeadNodeElimination (structural-algebraic): drop nodes whose outputs are unused
     *  7. SymmetrizedAccumulation (structural-algebraic): fold the r2 += s*(tmp + P(tmp)) idiom
     *  8. ElementWiseFusion (structural-algebraic): merge adjacent element-wise ops
     *  9. LinearCombinationContractionFolding (structural-algebraic): fold transpose-paired contractions
     * 10. DistributiveFactoring (structural-algebraic): factor a shared operand out of sibling contractions
     * 11. LoopInvariantHoisting (structural-algebraic): move invariant ops out of Loop bodies
     * 12. ScratchPrivatization (structural-resource): clone reused scratch to break false WAR/WAW chains
     * 13. ContractionPlanning (structural-algebraic): cost-model chain reassociation
     * 14. GEMMBatching (tuning): collapse compatible GEMMs into one BatchedGemm
     * 15. Reorder (tuning): memory-aware topological sort
     * 16. IOPrefetch (tuning): overlap DiskRead with compute
     * 17. DistributionPlanning (structural-resource): classify indices for distributed dispatch
     * 18. Materialization (tuning): resize deferred tensors to local partitions
     * 19. SymmetryPropagation (analysis): infer symmetry on graph intermediates and
     *                                 push to backing tensors for rank-2 BLAS dispatch
     * 20. SpacePropagation (analysis): infer per-slot index spaces on graph intermediates
     * 21. CrossSpaceValidation (diagnostic): flag letters binding slots of different index spaces
     * 22. ScalingAnalysis (diagnostic): report every contraction's cost polynomial and what limits it
     * 23. StreamContractionFusion (tuning): loop-fuse sibling contractions over one big tensor
     *
     * GPU block (when a GPU backend or mock is available):
     * 24. GPUPlacement (structural-resource): cost-model based node-to-GPU assignment
     * 25. TransferInsertion (structural-resource): insert HostToDevice / DeviceToHost nodes
     * 26. TransferElimination (structural-resource): drop redundant transfers
     * 27. GPUDiagnostics (diagnostic): log placement decisions
     * 28. StreamAssignment (tuning): assign CUDA/HIP streams for overlap
     *
     * Distributed block (when MPI or its mock is available):
     * 29. InputSlicing (structural-resource): create per-rank views of distributed inputs
     * 30. SUMMAExpansion (structural-resource): expand einsums to SUMMA loops on square grids
     * 31. CommunicationInsertion (structural-resource): insert allreduces for replicated outputs
     * 32. CommunicationElimination (structural-resource): drop redundant communications
     * 33. CommunicationScheduling (structural-resource): split allreduce into async iallreduce + wait
     *
     * Tail (always registered):
     * 34. InplaceOptimization (tuning): merge elementwise outputs into dying inputs
     * 35. FreeInsertion (tuning): free intermediates after last consumer
     * 36. MemoryPlanning (tuning): tensor liveness, peak memory, and arena planning
     *
     * @return A fully-populated PassManager.
     */
    static PassManager create_default();

    /**
     * @brief The read-only phases of the default pipeline: analysis and diagnostic.
     *
     * Holds every @ref PassPhase::Analysis and @ref PassPhase::Diagnostic pass
     * of @ref create_default, in the same relative order. The two share one
     * manager because they share the property that makes this manager useful:
     * running it changes nothing an executor can observe, so it is safe on a
     * graph in any state, including one just loaded from a file and not yet
     * re-planned.
     *
     * @return A PassManager holding only the read-only passes.
     */
    static PassManager analysis_pass_manager();

    /**
     * @brief The machine-independent rewrites: @ref PassPhase::StructuralAlgebraic.
     *
     * This is the phase whose output a saved graph persists, so this manager is
     * what an offline optimizer runs before writing a file, and what a load
     * does *not* re-run.
     *
     * @return A PassManager holding only the structural-algebraic passes.
     */
    static PassManager structural_pass_manager();

    /**
     * @brief The machine-dependent node-set changes: @ref PassPhase::StructuralResource.
     *
     * Tiling, GPU placement, distribution planning, slicing and SUMMA
     * expansion. Never saved, always re-derived, so a load runs this manager
     * and then @ref tuning_pass_manager over the file's algebraic form.
     *
     * @return A PassManager holding only the structural-resource passes.
     */
    static PassManager resource_pass_manager();

    /**
     * @brief The schedule, memory and batching decisions: @ref PassPhase::Tuning.
     *
     * @return A PassManager holding only the tuning passes.
     */
    static PassManager tuning_pass_manager();

    /**
     * @brief Every pass in this pipeline, in order, with the phase it declares.
     *
     * Introspection for tests and for anything that has to decide what a saved
     * graph may keep. Reads @ref OptimizerPass::phase, so a pass defined
     * outside this library is described like any other.
     *
     * @return ``(name, phase)`` pairs in pipeline order.
     */
    [[nodiscard]] std::vector<std::pair<std::string, PassPhase>> phase_of_each() const;

    /// Factory for a given optimization level (create_default() == O2).
    static PassManager create_for(OptLevel level);

    /**
     * @brief One human-readable report of what the last run() did.
     *
     * Harvests the per-pass statistics (nodes eliminated, chains
     * restructured with estimated speedups, buffers merged in place, Frees
     * inserted, arena size vs the buffers it hosts, batches formed and
     * profitability-gate skips) into a few lines of text. Empty until run()
     * has been called. See Graph::explain() for the graph-level entry point.
     *
     * At verbosity >= 2 the report gains a "not applied" section listing, per
     * pass, how many candidates it declined and why (see
     * ``OptimizerPass::skip_reasons``). "(no optimizations applied)" on its own
     * cannot tell you whether the graph was already optimal or whether every
     * candidate hit one satisfiable gate - and those call for opposite
     * responses - so raise the verbosity before concluding a pipeline is inert.
     *
     * Each line is prefixed with the reporting pass's @ref PassPhase, because
     * "which phase produced this" is the first question asked of a rewrite that
     * turns out to be wrong on one machine and right on another.
     */
    APIARY_EXPOSE [[nodiscard]] std::string explain() const;

    /**
     * @brief Populate this PassManager with the default pass list (in place).
     *
     * Equivalent to ``*this = create_default()`` but doesn't require the
     * class to be move-assignable. Used by the Python binding, which
     * can't bind ``add<PassType>()`` (templated) and so needs an
     * instance method to put the canonical pass list together.
     */
    APIARY_EXPOSE void populate_default();

    /// @name In-place phase population, for the Python binding
    ///
    /// The four static factories above return a @ref PassManager by value, which a
    /// language without move semantics cannot receive: the class is deliberately
    /// NOCOPY/NOMOVE on the binding side. These do the same job in place, exactly
    /// as @ref populate_default does for @ref create_default, and
    /// ``einsums.graph`` wraps each one as ``<phase>_pass_manager()``.
    ///
    /// This matters more than the symmetry suggests. A saved graph's documented
    /// load path is "run the resource phase, then the tuning phase", and until
    /// these existed a Python caller could only run the whole default pipeline -
    /// which re-runs the structural-algebraic phase a load is specified NOT to
    /// re-run. Each appends, so a manager may be built from more than one phase.
    /// @{

    /// @brief Append the read-only passes. @see analysis_pass_manager
    /// @versionadded{2.0.0}
    APIARY_EXPOSE void populate_analysis();

    /// @brief Append the machine-independent rewrites. @see structural_pass_manager
    /// @versionadded{2.0.0}
    APIARY_EXPOSE void populate_structural();

    /// @brief Append the machine-dependent node-set changes. @see resource_pass_manager
    /// @versionadded{2.0.0}
    APIARY_EXPOSE void populate_resource();

    /// @brief Append the schedule, memory and batching passes. @see tuning_pass_manager
    /// @versionadded{2.0.0}
    APIARY_EXPOSE void populate_tuning();
    /// @}

  private:
    /// Build the default pass list and append the phases in @p keep, preserving
    /// their relative order. The in-place counterpart of @ref filtered_default.
    void populate_filtered(std::initializer_list<PassPhase> keep);

    /**
     * @brief Construct the canonical pass list, once, for every consumer of it.
     *
     * @ref populate_default and the four phase-filtered factories all go
     * through this, so the ordering rationale lives in exactly one place and a
     * pass added to the default pipeline cannot be missing from the phase
     * views (or ordered differently in them).
     */
    static std::vector<std::shared_ptr<OptimizerPass>> build_default_passes();

    /// Build the default pass list and keep only the phases in @p keep,
    /// preserving their relative order in the default sequence.
    static PassManager filtered_default(std::initializer_list<PassPhase> keep);

    /// The allowance to hand the next pass: the explicit setting when there is one, the option
    /// otherwise.
    ///
    /// No EINSUMS_EXPORT: the class already carries it, and clang-cl rejects the attribute on a
    /// member of a class that has it ("attribute 'dllexport' cannot be applied to member of
    /// 'dllexport' class"). Harmless duplication under the Itanium ABI, which is why it built
    /// everywhere but Windows.
    [[nodiscard]] SearchBudget pass_budget() const;

    std::vector<std::shared_ptr<OptimizerPass>> _passes;
    int                                         _verbosity{0};

    /// Per-pass switches set through @ref disable / @ref enable: ``false`` is off,
    /// ``true`` is on and overriding ``einsums:pass:disable``. Ordered rather than
    /// hashed so the report is the same on every run.
    std::map<std::string, bool> _switches;

    /// What the last @ref run skipped, and which switch names matched nothing.
    /// Written by ``run()``, read by @ref explain, which is const.
    std::vector<std::string> _last_skipped;
    std::vector<std::string> _last_unmatched;

    /// Wall-clock allowance handed to each pass, in milliseconds. Consulted only when
    /// @ref _budget_explicit; otherwise ``run`` reads the option. Zero means unlimited.
    std::int64_t _budget_ms{0};
    bool         _budget_explicit{false};
};

EINSUMS_NAMESPACE_END(compute_graph)
