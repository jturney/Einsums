//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AliasOrderSharedScratch.cpp
/// @brief Independent chains sharing one scratch buffer must not land in the
/// same execution level.
///
/// Found in DLPNO-(T0). A one-shot phase gave every triplet its own chain -
/// gather into scratch, transform out of it - and shared the two largest
/// scratch buffers across all of them, sized to the largest triplet, which is
/// the same economy `examples/dlpno/dlpno/lccsd.py::_shared` makes for its
/// dressed three-index factors. Replayed under `OpenMPExecutor` it returned
/// wrong answers that changed from run to run: the water dimer's (T0) energy
/// came out anywhere between -0.038 and -0.082 against a true -0.006037088212,
/// while the CCSD energy computed underneath was bit-identical every time.
///
/// The chains are independent in every respect EXCEPT the buffer, so the only
/// thing that can order them is the hazard scan noticing they touch the same
/// storage. When it does, they occupy consecutive levels and the executor runs
/// them one at a time; when it does not, they share level 0 and clobber each
/// other.
///
/// This file tests the SCHEDULE rather than the numbers, deliberately. A race
/// reproduces probabilistically - the first four synthetic reductions of this
/// pattern all came back green because their tensors were small enough that
/// the window never opened - so a value check here would be a flaky test that
/// passes for the wrong reason. `dependencies().levels` is the property that
/// actually has to hold, and it is deterministic.
///
/// The disjoint case is the control. Scratch carved into non-overlapping
/// pieces, one per chain, is the fix DLPNO took and it MUST stay parallel: if
/// it ever starts serializing, the ordering has been restored by over-linking
/// and the phase has lost the reason it was captured at all.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Per chain: write a view of the scratch, then read it into a private output.
///
/// @param overlapping true  - every chain uses a PREFIX of the scratch, so the
///                            views overlap and the chains must be ordered.
///                    false - every chain gets its own disjoint slice, so they
///                            must stay independent.
/// @param chains      how many chains share the buffer.
/// @param ragged      give the chains different extents, which is the DLPNO
///                    case (one triplet per PNO/TNO count) and the one where a
///                    box-based disjointness test has the most room to get the
///                    answer wrong.
/// @return the widest level in the schedule: 1 when the chains are serialized,
///         `chains` when they run together.
size_t widest_level(bool overlapping, size_t chains, bool ragged, size_t full = 16) {
    RuntimeTensor<double> scratch("scratch", std::vector<size_t>{full, full * chains});

    std::vector<RuntimeTensor<double>> sources, outputs;
    sources.reserve(chains);
    outputs.reserve(chains);
    std::vector<size_t> extent(chains, full);
    for (size_t i = 0; i < chains; ++i) {
        if (ragged) {
            extent[i] = full - (i % 3);
        }
        RuntimeTensor<double> s("source", std::vector<size_t>{extent[i], full});
        s.zero();
        sources.push_back(std::move(s));
        RuntimeTensor<double> o("out", std::vector<size_t>{extent[i], full});
        o.zero();
        outputs.push_back(std::move(o));
    }

    // Views made once and kept alive for the graph's lifetime, as a captured
    // view operand must be.
    std::vector<RuntimeTensorView<double>> views;
    views.reserve(chains);
    for (size_t i = 0; i < chains; ++i) {
        size_t const lo = overlapping ? 0 : i * full;
        views.push_back(scratch(Range{0, extent[i]}, Range{lo, lo + full}));
    }

    cg::Graph graph("shared scratch");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t i = 0; i < chains; ++i) {
            cg::axpby(1.0, sources[i], 0.0, &views[i]); // write the scratch
            cg::axpby(1.0, views[i], 0.0, &outputs[i]); // read it straight back
        }
    }

    // The schedule is only built on demand; asking for it is what the
    // executors do.
    graph.topological_sort();
    size_t widest = 0;
    for (auto const &level : graph.dependencies().levels) {
        widest = std::max(widest, level.size());
    }
    return widest;
}

} // namespace

TEST_CASE("chains sharing one scratch buffer are serialized", "[compute-graph][alias]") {
    // Two chains is the smallest statement of the defect: chain 1's write must
    // not be free to run alongside chain 0's read.
    REQUIRE(widest_level(true, 2, false) == 1);
    REQUIRE(widest_level(true, 8, false) == 1);

    // Ragged extents are the DLPNO shape, and the case a box-based
    // disjointness test is most likely to mis-answer: the boxes differ, they
    // all start at the origin, and they all overlap.
    REQUIRE(widest_level(true, 8, true) == 1);

    // At the scale the defect was actually seen. Size should not matter to a
    // dependency question and it does not, but the value-level reproduction
    // needed 40 chains of a few hundred elements a side before the race window
    // opened at all, so the shape that failed is the shape that gets asserted.
    REQUIRE(widest_level(true, 40, true, 300) == 1);
}

TEST_CASE("chains on disjoint slices of one buffer stay parallel", "[compute-graph][alias]") {
    // The control. Over-linking would make the fix above look correct while
    // removing every reason to capture a wide, shallow phase.
    REQUIRE(widest_level(false, 8, false) == 8);
    REQUIRE(widest_level(false, 8, true) == 8);
}
