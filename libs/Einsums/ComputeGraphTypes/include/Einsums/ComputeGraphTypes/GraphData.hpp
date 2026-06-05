//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace einsums::compute_graph {

/**
 * @brief Serialized representation of a tensor in a compute graph snapshot.
 *
 * Used for profiling, visualization, and inter-process communication.
 * Mirrors the GraphTensor struct in the imgui_viewer ProfileData.
 */
struct GraphTensorData {
    uint64_t                 id{0};
    std::string              name;
    size_t                   rank{0};
    std::vector<size_t>      dims;
    std::vector<std::string> dim_names; ///< Symbolic dim names for code gen (e.g. "N", "nao"). Empty = use numeric dims.
    size_t                   element_size{0};
    std::string              dtype;
    bool                     is_intermediate{false};
    bool                     infer_dims{false};                 ///< If true, dimensions are auto-inferred from contraction patterns
    TensorOwnership          ownership{TensorOwnership::Graph}; ///< Ownership level for code gen and visual placement
};

/**
 * @brief Serialized representation of a computation node in a graph snapshot.
 *
 * Used for profiling, visualization, and inter-process communication.
 * Mirrors the GraphNode struct in the imgui_viewer ProfileData.
 */
struct GraphNodeData {
    uint64_t              id{0};
    std::string           kind;
    std::string           label;
    std::string           target{"CPU"};
    int                   stream_id{0};
    std::vector<uint64_t> inputs;
    std::vector<uint64_t> outputs;
    double                timing_ms{-1.0};

    // Operation-specific parameters
    double      c_prefactor{0.0};
    double      ab_prefactor{1.0};
    double      scale_factor{1.0};
    double      alpha{1.0};
    double      beta{0.0};
    std::string c_indices;
    std::string a_indices;
    std::string b_indices;
    bool        conj_a{false};
    bool        conj_b{false};
};

/**
 * @brief Serialized representation of a data-flow edge in a graph snapshot.
 *
 * Mirrors the GraphEdge struct in the imgui_viewer ProfileData.
 */
struct GraphEdgeData {
    uint64_t from{0};
    uint64_t to{0};
    uint64_t tensor_id{0};
    bool     loop_back{false}; ///< True if this edge goes from a later node to an earlier one (cycle)
};

/**
 * @brief Complete serialized compute graph, as used for profiling and visualization.
 *
 * Captures a named graph with its stage context (pipeline, workspace, stage)
 * and all nodes, tensors, and edges. Mirrors the ComputeGraphData struct
 * in the imgui_viewer ProfileData.
 */
struct ComputeGraphData {
    std::string                  name;
    std::string                  pipeline_name;        ///< Parent pipeline (empty if standalone)
    std::string                  workspace_name;       ///< Parent workspace (empty if none)
    std::string                  stage_name;           ///< Stage within pipeline
    std::string                  stage_type;           ///< "graph" or "loop"
    int                          stage_index{-1};      ///< Order within pipeline
    int                          max_iterations{50};   ///< For loop stages: max iteration count
    std::string                  convergence_variable; ///< For loop stages: variable name for convergence threshold
    std::vector<GraphTensorData> tensors;
    std::vector<GraphNodeData>   nodes;
    std::vector<GraphEdgeData>   edges;
};

} // namespace einsums::compute_graph
