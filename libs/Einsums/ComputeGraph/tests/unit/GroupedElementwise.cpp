//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The grouped element-wise forms against the emission they replace: the SAME
// single-tensor call, once per member, in the same order. Their contract is
// bit-identity rather than agreement to roundoff, because an element-wise
// kernel's result cannot depend on how the run was divided, so every
// comparison here is on the exact bytes.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

RuntimeTensor<double> randt(std::string const &name, std::vector<std::size_t> const &dims, std::mt19937 &gen) {
    RuntimeTensor<double>                  out(name, dims);
    std::uniform_real_distribution<double> dist(0.5, 1.5);
    double                                *p = out.data();
    for (std::size_t i = 0; i < out.size(); i++) {
        p[i] = dist(gen);
    }
    return out;
}

std::vector<unsigned char> bytes_of(RuntimeTensor<double> const &t) {
    std::vector<unsigned char> out(t.size() * sizeof(double));
    std::memcpy(out.data(), t.data(), out.size());
    return out;
}

/// Member extents chosen to span the cases that break a grouped form: several
/// sizes so no member's shape is the run's, and a zero extent so an empty
/// member is carried rather than special-cased.
std::vector<std::vector<std::size_t>> const kShapes = {{4, 3, 5}, {2, 2, 2}, {0, 3, 4}, {7, 1, 2}, {3, 6, 1}, {5, 5, 5}};

} // namespace

TEST_CASE("GroupedPermute - matches the per-member permute bit for bit", "[ComputeGraph][GroupedElementwise]") {
    std::mt19937 gen(7);

    std::vector<RuntimeTensor<double>> a, want, got;
    std::vector<double>                c_pfs, a_pfs;
    for (std::size_t i = 0; i < kShapes.size(); i++) {
        auto const &d = kShapes[i];
        a.push_back(randt("a", d, gen));
        // "abc <- acb" keeps the leading extent and swaps the other two.
        want.push_back(randt("want", {d[0], d[2], d[1]}, gen));
        got.push_back(want.back());
        c_pfs.push_back(i % 2 == 0 ? 0.0 : 1.0);
        a_pfs.push_back(0.25 * static_cast<double>(i) - 0.5);
    }

    for (std::size_t i = 0; i < a.size(); i++) {
        cg::string_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", &want[i], a[i], c_pfs[i], a_pfs[i]);
    }

    std::vector<RuntimeTensor<double> *>       c_list;
    std::vector<RuntimeTensor<double> const *> a_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        c_list.push_back(&got[i]);
        a_list.push_back(&a[i]);
    }
    cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", c_list, a_list, c_pfs, a_pfs);

    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEST_CASE("GroupedPermute - capture replays match eager and each other", "[ComputeGraph][GroupedElementwise]") {
    std::mt19937 gen(11);

    std::vector<RuntimeTensor<double>> a, want, got;
    std::vector<double>                c_pfs, a_pfs;
    for (auto const &d : kShapes) {
        a.push_back(randt("a", d, gen));
        want.push_back(randt("want", {d[0], d[2], d[1]}, gen));
        got.push_back(want.back());
        c_pfs.push_back(1.0);
        a_pfs.push_back(-2.0);
    }

    std::vector<RuntimeTensor<double> *>       w_list, g_list;
    std::vector<RuntimeTensor<double> const *> a_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        w_list.push_back(&want[i]);
        g_list.push_back(&got[i]);
        a_list.push_back(&a[i]);
    }
    cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", w_list, a_list, c_pfs, a_pfs);

    cg::Graph g("grouped permute");
    {
        cg::CaptureGuard guard(g);
        cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", g_list, a_list, c_pfs, a_pfs);
    }
    REQUIRE(g.num_nodes() == 1);
    g.execute();
    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }

    // A second replay accumulates again; the eager loop is stepped alongside it
    // so the comparison stays against the emission rather than against a
    // remembered value.
    cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", w_list, a_list, c_pfs, a_pfs);
    g.execute();
    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEST_CASE("GroupedDirectProduct - matches the per-member product bit for bit", "[ComputeGraph][GroupedElementwise]") {
    std::mt19937 gen(13);

    std::vector<RuntimeTensor<double>> a, b, want, got;
    std::vector<double>                alphas, betas;
    for (std::size_t i = 0; i < kShapes.size(); i++) {
        auto const &d = kShapes[i];
        a.push_back(randt("a", d, gen));
        b.push_back(randt("b", d, gen));
        want.push_back(randt("want", d, gen));
        got.push_back(want.back());
        alphas.push_back(0.5 + 0.125 * static_cast<double>(i));
        betas.push_back(i % 3 == 0 ? 0.0 : 1.0);
    }

    for (std::size_t i = 0; i < a.size(); i++) {
        cg::direct_product<double, RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(alphas[i], a[i], b[i], betas[i],
                                                                                                        &want[i]);
    }

    std::vector<RuntimeTensor<double> const *> a_list, b_list;
    std::vector<RuntimeTensor<double> *>       c_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        a_list.push_back(&a[i]);
        b_list.push_back(&b[i]);
        c_list.push_back(&got[i]);
    }
    cg::grouped_direct_product<double, RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(alphas, a_list, b_list, betas,
                                                                                                            c_list);

    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEST_CASE("GroupedDirectDivision - matches the per-member division, eager and replayed", "[ComputeGraph][GroupedElementwise]") {
    std::mt19937 gen(17);

    std::vector<RuntimeTensor<double>> a, b, want, got;
    std::vector<double>                alphas, betas;
    for (std::size_t i = 0; i < kShapes.size(); i++) {
        auto const &d = kShapes[i];
        a.push_back(randt("a", d, gen));
        b.push_back(randt("b", d, gen));
        want.push_back(randt("want", d, gen));
        got.push_back(want.back());
        alphas.push_back(-1.0);
        betas.push_back(i % 2 == 0 ? 0.0 : 1.0);
    }

    for (std::size_t i = 0; i < a.size(); i++) {
        cg::direct_division<double, RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(alphas[i], a[i], b[i], betas[i],
                                                                                                         &want[i]);
    }

    std::vector<RuntimeTensor<double> const *> a_list, b_list;
    std::vector<RuntimeTensor<double> *>       c_list;
    for (std::size_t i = 0; i < a.size(); i++) {
        a_list.push_back(&a[i]);
        b_list.push_back(&b[i]);
        c_list.push_back(&got[i]);
    }

    cg::Graph g("grouped division");
    {
        cg::CaptureGuard guard(g);
        cg::grouped_direct_division<double, RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(alphas, a_list, b_list,
                                                                                                                 betas, c_list);
    }
    REQUIRE(g.num_nodes() == 1);
    g.execute();

    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(bytes_of(got[i]) == bytes_of(want[i]));
    }
}

TEST_CASE("GroupedElementwise - rejects the runs it cannot compute", "[ComputeGraph][GroupedElementwise]") {
    std::mt19937 gen(19);

    auto x = randt("x", {3, 4, 2}, gen);
    auto y = randt("y", {3, 4, 2}, gen);
    auto z = randt("z", {3, 2, 4}, gen);

    std::vector<RuntimeTensor<double> const *> a_list{&x, &y};
    std::vector<RuntimeTensor<double> *>       c_list{&z, &z};

    // Two members writing one tensor: the run threads, so the node has no
    // ordering to give them.
    REQUIRE_THROWS_AS(
        (cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", c_list, a_list, {0.0, 0.0}, {1.0, 1.0})),
        std::invalid_argument);

    // Lists of different lengths.
    std::vector<RuntimeTensor<double> *> one{&z};
    REQUIRE_THROWS_AS((cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", one, a_list, {0.0}, {1.0})),
                      std::invalid_argument);

    // An empty run is a caller mistake, not a no-op: the emission it replaces
    // had nothing to merge.
    std::vector<RuntimeTensor<double> *>       none_c;
    std::vector<RuntimeTensor<double> const *> none_a;
    REQUIRE_THROWS_AS((cg::grouped_permute<RuntimeTensor<double>, RuntimeTensor<double>>("abc <- acb", none_c, none_a, {}, {})),
                      std::invalid_argument);

    // Members whose operands disagree on shape.
    std::vector<RuntimeTensor<double> const *> mixed_a{&x};
    std::vector<RuntimeTensor<double> const *> mixed_b{&z};
    std::vector<RuntimeTensor<double> *>       mixed_c{&y};
    REQUIRE_THROWS_AS((cg::grouped_direct_product<double, RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(
                          {1.0}, mixed_a, mixed_b, {0.0}, mixed_c)),
                      dimension_error);
}
