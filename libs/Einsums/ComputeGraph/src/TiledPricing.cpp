//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Detail/TiledPricing.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

double tile_pair_cost_us(CostModel const &model, TiledContractionShape const &shape, std::vector<int> const &ucoord) {
    auto const &part = *shape.part;
    std::size_t tM = 1, tN = 1, tK = 1;
    for (std::size_t u = 0; u < part.size(); ++u) {
        auto const ext = static_cast<std::size_t>(part[u][static_cast<std::size_t>(ucoord[u])]);
        if (shape.in_c[u]) {
            (shape.in_a[u] ? tM : tN) *= ext;
        } else {
            tK *= ext;
        }
    }
    return model.estimate_total_gemm_time_us(tM, tN, tK, shape.element_size, Target::CPU);
}

double densified_cost_us(CostModel const &model, TiledContractionShape const &shape) {
    auto const &part = *shape.part;
    std::size_t dM = 1, dN = 1, dK = 1;
    for (std::size_t u = 0; u < part.size(); ++u) {
        std::size_t extent = 0;
        for (int const sz : part[u]) {
            extent += static_cast<std::size_t>(sz);
        }
        if (shape.in_c[u]) {
            (shape.in_a[u] ? dM : dN) *= extent;
        } else {
            dK *= extent;
        }
    }
    std::size_t const bytes = shape.element_size * (dM * dK + dK * dN + 2 * dM * dN);
    return model.estimate_total_gemm_time_us(dM, dN, dK, shape.element_size, Target::CPU) +
           model.estimate_memory_time_us(bytes, Target::CPU);
}

char const *&last_tiled_lowering() {
    thread_local char const *lowering = "";
    return lowering;
}

CostModel const &eager_tiled_cost_model() {
    static CostModel const model = CostModel::detect_default();
    return model;
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
