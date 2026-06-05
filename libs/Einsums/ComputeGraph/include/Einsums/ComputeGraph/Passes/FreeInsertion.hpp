//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Insert Free nodes to release intermediate tensors after their last consumer.
 *
 * For each intermediate tensor (is_intermediate == true), finds the last node
 * that reads or writes it and inserts a Free node immediately after. The Free
 * node calls release_fn() on the TensorHandle, which frees the backing storage
 * and returns the tensor to a deferred-like state.
 *
 * On re-execution, the Materialize node at the tensor's first use re-allocates
 * the storage via materialize_fn() (which is idempotent).
 *
 * Only frees intermediate (graph-owned) tensors. User-provided tensors
 * (inputs/outputs of the whole computation) are never freed.
 *
 * Runs AFTER Reorder (which optimizes node order for memory) and communication
 * passes (which may add nodes that reference tensors).
 */
class EINSUMS_EXPORT FreeInsertion : public OptimizerPass {
  public:
    /// @param min_bytes Only free intermediates larger than this (default 1MB).
    ///                  Small tensors are kept alive to avoid alloc/free overhead
    ///                  in re-executed graphs (loops, Pipeline stages).
    explicit FreeInsertion(size_t min_bytes = static_cast<size_t>(1024 * 1024)) : _min_bytes(min_bytes) {}

    [[nodiscard]] std::string name() const override { return "FreeInsertion"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_freed() const { return _num_freed; }

  private:
    size_t _min_bytes;
    size_t _num_freed{0};
};

} // namespace einsums::compute_graph::passes
