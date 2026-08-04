//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// cg::gather - index-list extraction along any subset of axes.
//
// The oracle everywhere is the same explicit loop over the index lists that
// the numpy expression A[np.ix_(rows, cols)] denotes, so these tests pin the
// outer-product (not zipped) selection semantics as well as the mechanics.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <numeric>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEST_CASE("gather rows and columns", "[compute-graph][gather]") {
    auto A = create_random_tensor<double>("A", 7, 5);

    std::vector<size_t> rows{5, 1, 1, 3}; // out of order and repeated on purpose
    std::vector<size_t> cols{4, 0};

    auto out = create_zero_tensor<double>("out", rows.size(), cols.size());
    compute_graph::gather(&out, A, {rows, cols});

    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < cols.size(); ++j) {
            REQUIRE(out(i, j) == A(rows[i], cols[j]));
        }
    }
}

static std::vector<size_t> whole(size_t n) {
    std::vector<size_t> v(n);
    std::iota(v.begin(), v.end(), size_t{0});
    return v;
}

TEST_CASE("gather: an empty index list selects nothing, it is not a wildcard", "[compute-graph][gather]") {
    // The semantics that matter for domain-restricted callers: an empty domain
    // must stay empty. Expanding it to the whole axis would silently turn a
    // screened-out domain into a full-rank one.
    auto A   = create_random_tensor<double>("A", 4, 6);
    auto out = create_zero_tensor<double>("out", size_t{0}, size_t{6});
    REQUIRE_NOTHROW(compute_graph::gather(&out, A, {std::vector<size_t>{}, whole(6)}));

    // Asking for the whole axis is an explicit range, and is then a plain copy.
    auto all = create_zero_tensor<double>("all", 4, 6);
    compute_graph::gather(&all, A, {whole(4), whole(6)});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            REQUIRE(all(i, j) == A(i, j));
        }
    }
}

TEST_CASE("gather on a rank-3 tensor along two axes", "[compute-graph][gather]") {
    // The shape the DLPNO three-index integrals are sliced in: restrict the
    // auxiliary and virtual axes to a domain, keep the occupied axis whole.
    auto A = create_random_tensor<double>("A", 6, 3, 8);

    std::vector<size_t> aux{4, 0, 5};
    std::vector<size_t> vir{7, 2};

    auto out = create_zero_tensor<double>("out", aux.size(), 3, vir.size());
    compute_graph::gather(&out, A, {aux, whole(3), vir});

    for (size_t q = 0; q < aux.size(); ++q) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t a = 0; a < vir.size(); ++a) {
                REQUIRE(out(q, i, a) == A(aux[q], i, vir[a]));
            }
        }
    }
}

TEST_CASE("gather captures and replays", "[compute-graph][gather]") {
    auto A = create_random_tensor<double>("A", 6, 6);

    std::vector<size_t> dom{4, 1, 0};
    auto                out = create_zero_tensor<double>("out", dom.size(), dom.size());

    compute_graph::Graph g("gather");
    {
        compute_graph::CaptureGuard guard(g);
        compute_graph::gather(&out, A, {dom, dom});
    }
    REQUIRE(g.num_nodes() == 1);
    g.execute();

    for (size_t i = 0; i < dom.size(); ++i) {
        for (size_t j = 0; j < dom.size(); ++j) {
            REQUIRE(out(i, j) == A(dom[i], dom[j]));
        }
    }

    // Replay must see the source's current values, not those at capture time:
    // the whole point is capture-once, replay-many.
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            A(i, j) = A(i, j) + 1.0;
        }
    }
    g.execute();
    for (size_t i = 0; i < dom.size(); ++i) {
        for (size_t j = 0; j < dom.size(); ++j) {
            REQUIRE(out(i, j) == A(dom[i], dom[j]));
        }
    }
}

TEST_CASE("gather rejects mismatched shapes and bad indices", "[compute-graph][gather]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    SECTION("dst extent must match the index count") {
        auto out = create_zero_tensor<double>("out", 3, 4);
        REQUIRE_THROWS_AS(compute_graph::gather(&out, A, {std::vector<size_t>{0, 1}, whole(4)}), std::invalid_argument);
    }

    SECTION("index out of range") {
        auto out = create_zero_tensor<double>("out", 2, 4);
        REQUIRE_THROWS_AS(compute_graph::gather(&out, A, {std::vector<size_t>{0, 9}, whole(4)}), std::out_of_range);
    }

    SECTION("rank mismatch") {
        auto out = create_zero_tensor<double>("out", size_t{2});
        REQUIRE_THROWS_AS(compute_graph::gather(&out, A, {std::vector<size_t>{0, 1}, whole(4)}), rank_error);
    }

    SECTION("empty indices") {
        auto out = create_zero_tensor<double>("out", 2, 2);
        REQUIRE_THROWS_AS(compute_graph::gather(&out, A, {}), std::invalid_argument);
    }
}
