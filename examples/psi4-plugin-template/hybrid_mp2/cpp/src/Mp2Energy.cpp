//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Scaffolded by 'python -m einsums.stages promote' from hybrid_mp2.stages.
// Written once and never overwritten: this file is yours.

// The port of hybrid_mp2.mp2.mp2_energy: same operations, same order, same
// executor, so the two backends agree bit for bit and the differential test
// under tests/ is a real check rather than a tolerance negotiation.
//
// The idioms worth copying into your own port (all learned in the DLPNO
// ports, examples/dlpno/cpp/src/):
//
// * Every operand a captured op touches is created BEFORE the CaptureGuard,
//   in vectors sized up front - the graph records addresses, and a vector
//   that reallocates mid-capture leaves nodes pointing at freed storage.
// * The graph executes AFTER the guard is destroyed; executing under capture
//   would record the execution into the graph being built.
// * Scalar results are graph-written length-1 tensors (`dot_python`), because
//   this function returns before anyone calls it under a session capture.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstddef>
#include <cstdint>
#include <hybrid_mp2/Mp2Energy.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace cg = einsums::compute_graph;

namespace hybrid_mp2 {

namespace {

einsums::SliceSpec spec_full() {
    return {};
}

einsums::SliceSpec spec_index(std::int64_t i) {
    einsums::SliceSpec s;
    s.kind  = einsums::SliceSpec::Kind::Index;
    s.index = i;
    return s;
}

} // namespace

Mp2Energy mp2_energy(einsums::RuntimeTensor<double> const &iajb, einsums::RuntimeTensor<double> const &eps_occ,
                     einsums::RuntimeTensor<double> const &eps_vir) {
    auto const    nocc = eps_occ.dim(0);
    auto const    nvir = eps_vir.dim(0);
    double const *eo   = eps_occ.data();

    Mp2Energy out{
        .e_corr = einsums::create_zero_tensor<double>("E corr", std::size_t{1}),
        .e_os   = einsums::create_zero_tensor<double>("E os", std::size_t{1}),
    };

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    pairs.reserve(nocc * (nocc + 1) / 2);
    for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t j = i; j < nocc; ++j) {
            pairs.emplace_back(i, j);
        }
    }
    if (pairs.empty()) {
        return out;
    }

    // -e_a - e_b is pair-independent: built once, eagerly, before the capture.
    // (Declared as the runtime-rank type: create_zero_tensor returns a
    // static-rank Tensor, and the cg ops take RuntimeTensor.)
    einsums::RuntimeTensor<double> base = einsums::create_zero_tensor<double>("-ea-eb", nvir, nvir);
    cg::outer_sum(&base, std::vector<einsums::RuntimeTensor<double> const *>{&eps_vir, &eps_vir}, {-1.0, -1.0});

    // Per-pair scratch and input views, all allocated before the capture.
    std::vector<einsums::RuntimeTensor<double>>     I, K, D, T, e, eos;
    std::vector<einsums::RuntimeTensorView<double>> blocks;
    I.reserve(pairs.size());
    K.reserve(pairs.size());
    D.reserve(pairs.size());
    T.reserve(pairs.size());
    e.reserve(pairs.size());
    eos.reserve(pairs.size());
    blocks.reserve(pairs.size());
    for (auto const &[i, j] : pairs) {
        I.emplace_back(einsums::create_zero_tensor<double>("I", nvir, nvir));
        K.emplace_back(einsums::create_zero_tensor<double>("K", nvir, nvir));
        D.emplace_back(einsums::create_zero_tensor<double>("D", nvir, nvir));
        T.emplace_back(einsums::create_zero_tensor<double>("T", nvir, nvir));
        e.emplace_back(einsums::create_zero_tensor<double>("e pair", std::size_t{1}));
        eos.emplace_back(einsums::create_zero_tensor<double>("e pair os", std::size_t{1}));
        blocks.emplace_back(
            iajb.at_view({spec_index(static_cast<std::int64_t>(i)), spec_full(), spec_index(static_cast<std::int64_t>(j)), spec_full()}));
    }

    cg::Graph g("mp2 pairs");
    {
        cg::CaptureGuard const guard(g);
        for (std::size_t p = 0; p < pairs.size(); ++p) {
            auto const [i, j] = pairs[p];
            cg::axpby(1.0, blocks[p], 0.0, &I[p]); // densify the pair block
            cg::permute("ab <- ba", &K[p], I[p]);  // K = I^T
            cg::axpby(2.0, I[p], -1.0, &K[p]);     // K = 2 I - I^T
            cg::axpby(1.0, base, 0.0, &D[p]);
            cg::shift(eo[i] + eo[j], &D[p]); // D = e_i + e_j - e_a - e_b
            cg::direct_division(1.0, I[p], D[p], 0.0, &T[p]);
            cg::dot_python(&e[p], K[p], T[p]);
            cg::dot_python(&eos[p], I[p], T[p]);
            double const f = (i == j) ? 1.0 : 2.0;
            cg::axpby(f, e[p], 1.0, &out.e_corr);
            cg::axpby(f, eos[p], 1.0, &out.e_os);
        }
    }
    g.set_executor(std::make_shared<cg::OpenMPExecutor>());
    g.execute();

    return out;
}

} // namespace hybrid_mp2
