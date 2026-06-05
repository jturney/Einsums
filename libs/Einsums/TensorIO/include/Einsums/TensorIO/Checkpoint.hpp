//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

/// @file Checkpoint.hpp
/// @brief Save/restore tensor state to .etn files.
///
/// Works with Graph, Workspace, and Pipeline:
///
/// @code
/// namespace ckpt = einsums::tensor_io::checkpoint;
///
/// ckpt::save("scf_iter_5.etn", graph);
/// ckpt::save("workspace.etn", workspace);
/// ckpt::restore("scf_iter_5.etn", graph);
///
/// // Distributed
/// ckpt::save_distributed("checkpoint.etn", graph);
/// ckpt::restore_distributed("checkpoint.etn", graph);
/// @endcode

#include <Einsums/TensorIO/DistributedTensorFile.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <string>
#include <vector>

// Forward declarations to avoid pulling in full headers
namespace einsums::compute_graph {
class Graph;
class Workspace;
} // namespace einsums::compute_graph

namespace einsums::tensor_io::checkpoint {

// ═══════════════════════════════════════════════════════════════════════════════
// Serial (single-file POSIX I/O)
// ═══════════════════════════════════════════════════════════════════════════════

/// Save all materialized tensors from a Graph.
EINSUMS_EXPORT void save(std::string const &path, compute_graph::Graph const &graph, std::vector<std::string> const &tensor_names = {});

/// Save all tensors from a Workspace.
EINSUMS_EXPORT void save(std::string const &path, compute_graph::Workspace const &workspace,
                         std::vector<std::string> const &tensor_names = {});

/// Restore tensor data into a Graph.
EINSUMS_EXPORT void restore(std::string const &path, compute_graph::Graph &graph);

/// Restore tensor data into a Workspace.
EINSUMS_EXPORT void restore(std::string const &path, compute_graph::Workspace &workspace);

// ═══════════════════════════════════════════════════════════════════════════════
// Distributed (MPI-coordinated single-file I/O)
// ═══════════════════════════════════════════════════════════════════════════════

/// Save distributed Graph tensors — replicated written once, distributed per-rank.
EINSUMS_EXPORT void save_distributed(std::string const &path, compute_graph::Graph const &graph,
                                     std::vector<std::string> const &tensor_names = {});

/// Restore distributed Graph tensors.
EINSUMS_EXPORT void restore_distributed(std::string const &path, compute_graph::Graph &graph);

} // namespace einsums::tensor_io::checkpoint
