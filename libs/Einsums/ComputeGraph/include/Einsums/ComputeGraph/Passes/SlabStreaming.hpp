//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Read a backed tensor a slab at a time instead of loading it whole.
 *
 * A tensor annotated with a backing (@ref Graph::annotate_backing) is loaded whole before its
 * first reader, which needs memory for all of it. This pass rewrites each contraction that reads
 * one into a loop over blocks of the backed tensor's LAST axis: every iteration reads one slab
 * from the storage into a buffer the size of the block, then runs the same contraction on it.
 * The last axis because the storage keeps elements column-major, so a block of the last axis is
 * one contiguous extent, read in one piece. A backed tensor no node reads whole any more is then
 * never allocated.
 *
 * What a block does to the destination follows the letter the last axis carries. When the
 * destination carries it too, each block writes its own slice and the result is the captured
 * one exactly. When the letter is summed, the blocks add into the destination: the first block
 * applies the destination's prefactor and the rest add, which is the captured contraction's own
 * order of operations with its sum cut into parts, so the result moves within the
 * re-association tolerance.
 *
 * A tensor with a WRITE backing is streamed the other way. The contraction that produces it
 * becomes a loop over blocks of its last axis: every iteration computes one block into a buffer
 * the size of the block and writes that slab to the storage, so the tensor is never held whole.
 * The contraction must overwrite the tensor (a destination prefactor of zero), be its only
 * writer in memory, and read no backed tensor; otherwise the tensor is produced in memory and
 * Materialization stores it whole after its last writer. Its readers are streamed as any backed
 * tensor's are, but only once nothing writes it in memory, and they run after the writes: every
 * read and write of a storage carries the storage's key.
 *
 * It acts only under a memory budget, the same one AxisTiling plans against: a backed tensor is
 * streamed when it is larger than its share of @c --einsums:max-memory
 * (@c --einsums:graph:tiling-budget-share). One that fits is loaded once per replay, which beats
 * one read per reading contraction. It runs after AxisTiling and inside every loop body, so a
 * contraction AxisTiling placed in a block loop is streamed there.
 *
 * Declined, with the reason in the skip tally, and left to the whole load: a backed tensor whose
 * last-axis letter is repeated in any operand or the destination; one read through a view other
 * than a full one; a contraction both of whose operands are backed; one with permutation
 * operators; any node but a contraction; operands that are not rank-erased tensors.
 *
 * The slab reads run in order with the contraction: a block's read finishes before its
 * contraction starts. A graph this pass has rewritten cannot be saved, like one AxisTiling has
 * rewritten; save the graph before the resource passes.
 *
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT SlabStreaming : public OptimizerPass {
  public:
    /// @brief Default-construct, reading the budget from the options.
    APIARY_EXPOSE SlabStreaming() = default;

    /// @copydoc OptimizerPass::name
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "SlabStreaming"; }

    /// @copydoc OptimizerPass::phase
    ///
    /// Taken for a machine reason, how much memory there is, with the algebra untouched.
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralResource; }

    /// @copydoc OptimizerPass::tier
    ///
    /// A summed letter's blocks add their partial sums in a different order.
    [[nodiscard]] PassTier tier() const override { return PassTier::ReAssociating; }

    /// @copydoc OptimizerPass::recurse_into_subgraphs
    ///
    /// The contraction worth streaming is often inside a loop: an iterative method's body, or a
    /// block loop AxisTiling emitted. The pass walks those bodies itself rather than being handed
    /// them one at a time, because how a backed tensor is best stored follows from every box of it
    /// the whole tree moves, which is known only once the walk is over.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return false; }

    /// @copydoc OptimizerPass::understood_features
    ///
    /// Views, because the operands of a contraction in a block loop are views, and the rewrite
    /// views the tensor underneath each one directly and removes the original when nothing else
    /// uses it; a view that slices the streamed letter is declined. Conjugation, because the
    /// rewrite re-emits the contraction's own conjugation flags.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override {
        return NodeFeatures{} | NodeFeature::Views | NodeFeature::WritesView | NodeFeature::Conjugation | NodeFeature::ComplexPrefactor |
               NodeFeature::MixedPrecision | NodeFeature::Grouped | NodeFeature::ControlFlow | NodeFeature::Tiled | NodeFeature::RawScalar;
    }

    /// @copydoc OptimizerPass::run
    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::reset_stats
    void reset_stats() override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

    /// @copydoc OptimizerPass::compensated_reads
    ///
    /// A contraction that produces a write-backed tensor into its storage leaves the tensor's
    /// remaining readers without an in-graph writer. They read the stored tensor instead: its fill
    /// from the storage carries the storage's key, which orders it after every slab write.
    [[nodiscard]] std::vector<std::pair<NodeId, TensorId>> compensated_reads() const override { return _compensated; }

    /**
     * @brief Override the slab budget for this pipeline, in bytes.
     * @param[in] bytes A backed tensor larger than this is streamed, in slabs no larger than it.
     *            Zero turns the pass off; a negative value derives it from @c --einsums:max-memory
     *            and @c --einsums:graph:tiling-budget-share, which is the default.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void set_slab_budget(std::int64_t bytes) { _budget_override = bytes; }

    /// @brief How many contractions the last apply streamed.
    /// @return The count, zero when the pass declined everything.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("num_streamed") [[nodiscard]] std::size_t num_streamed() const noexcept { return _num_streamed; }

    /// @brief The backed tensors the last apply streamed, by name, one entry per streamed contraction.
    /// @return The names, in the order the contractions were rewritten.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("streamed") [[nodiscard]] std::vector<std::string> streamed() const { return _streamed; }

    /// @brief How many of the streamed contractions produce their tensor into its storage.
    /// @return The count of contractions turned into loops ending in a slab write.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("num_written") [[nodiscard]] std::size_t num_written() const noexcept { return _num_written; }

    /// @brief The block each streamed contraction reads, along the backed tensor's last axis.
    /// @return One entry per streamed contraction, in the order of @ref streamed.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("blocks") [[nodiscard]] std::vector<std::int64_t> blocks() const { return _blocks; }

    /// @brief How many views of a backed tensor the last apply turned into a read of the box each one names.
    /// @return The count, one per view, whichever loop chose the box.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("num_boxes") [[nodiscard]] std::size_t num_boxes() const noexcept { return _num_boxes; }

    /// @brief The order the last apply left a backed tensor stored in.
    /// @param[in] name The backed tensor's name.
    /// @return The declared axis at each stored position, as @ref TensorBacking::stored_order gives
    ///         it; empty for a tensor the apply moved no box of, or whose storage has no order.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE [[nodiscard]] std::vector<std::size_t> stored_order(std::string const &name) const;

  private:
    /// One streamed contraction's report line.
    struct Record {
        std::string tensor;
        std::string label;
        std::size_t extent{0};
        std::size_t block{0};
        std::size_t bytes{0};
        double      flops{0.0};
        bool        summed{false};
        bool        writes{false}; ///< The contraction produces the tensor into its storage.
        bool        box{false};    ///< A view's box is moved in place of the view.
        /// For a box: its extent on each declared axis, and whether that is the whole axis, which
        /// with the order the tensor ends up stored in gives its runs.
        std::vector<std::size_t> extents;
        std::vector<bool>        whole_axis;
    };

    /// One shape of box a backed tensor is moved in, and how many such boxes a replay moves: what
    /// the choice of its stored order weighs.
    struct BoxUse {
        std::shared_ptr<TensorBacking const> backing;
        std::string                          name;
        std::vector<std::size_t>             extents;
        std::vector<bool>                    whole_axis;
        double                               count{1.0}; ///< Transfers of it per replay of the root.
    };

    /// What a graph knows about one backed tensor object.
    struct BackedEntry {
        std::string                          name;
        std::shared_ptr<TensorBacking const> backing;
        /// For a write-backed tensor: an operation outside this graph and the graphs below it
        /// writes it in memory, so neither its producer nor its readers here may be streamed.
        bool written_outside{false};
    };

    /// The backed tensor objects visible from a graph, which the enclosing graph hands each body it
    /// holds: a body's own handle for one carries none.
    using Backed = std::unordered_map<void const *, BackedEntry>;

    /// The budget in force for this apply, in bytes; zero when off.
    [[nodiscard]] std::size_t budget() const;

    /// Stream @p graph's contractions, then each body below it, handing each what it inherits.
    /// @p replays is how many times @p graph runs per replay of the root.
    bool walk(Graph &graph, Backed backed, double replays);
    /// Note a box shape @p backing is moved in, @p count times per replay of the root.
    void note_box(std::shared_ptr<TensorBacking const> const &backing, std::string const &name, std::vector<std::size_t> extents,
                  std::vector<bool> whole_axis, double count);
    /// Store each written tensor in the order that makes the boxes moved of it the fewest runs, where
    /// it may still be chosen, and advise where it may not; after the walk.
    void choose_orders();

    bool stream_graph(Graph &graph, Backed const &backed);
    bool stream_write(Graph &graph, Backed const &backed, std::size_t index, std::size_t cap);
    /// Replace the View node @p view_node, a slice of the backed tensor @p name, with a buffer of the
    /// slice's shape and a transfer of the box the view names: a read where the view stood, or,
    /// when @p writer is the node that produces the slice, a write after it.
    bool stream_box(Graph &graph, std::size_t view_node, std::string const &name, std::shared_ptr<TensorBacking const> const &backing,
                    std::string const &label, std::optional<std::size_t> writer);

    std::int64_t                                    _budget_override{-1};
    Graph const                                    *_root{nullptr};
    std::vector<BoxUse>                             _box_uses;
    double                                          _replays{1.0}; ///< How often the graph being streamed runs per replay of the root.
    std::map<std::string, std::vector<std::size_t>> _orders;
    std::vector<std::string>                        _order_lines;
    std::vector<std::pair<NodeId, TensorId>>        _compensated;
    std::vector<Record>                             _records;
    std::size_t                                     _num_streamed{0};
    std::size_t                                     _num_written{0};
    std::size_t                                     _num_boxes{0};
    std::vector<std::string>                        _streamed;
    std::vector<std::int64_t>                       _blocks;
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
