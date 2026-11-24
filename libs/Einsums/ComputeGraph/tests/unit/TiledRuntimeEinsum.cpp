//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Coverage: cg::einsum over TiledRuntimeTensor operands. The whole
// tiled tensor becomes a graph operand; the contraction walks the tile grid and
// composes dense per-tile einsums. Tests cover eager + captured execution, the
// non-symmetric (off-diagonal/rectangular) sparsity with infer-and-create
// output tiles, and the tile-partition alignment guard.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/DeadNodeElimination.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

using Grid = std::vector<std::vector<int>>;

// Fill every grid tile of a 2-D tiled tensor from a global (row, col) function.
template <typename V, typename F>
void fill_tiled(TiledRuntimeTensor<V> &T, F &&f) {
    auto const &off = T.tile_offsets();
    auto const &sz  = T.tile_sizes();
    for (int ti = 0; ti < static_cast<int>(sz[0].size()); ++ti) {
        for (int tj = 0; tj < static_cast<int>(sz[1].size()); ++tj) {
            auto &tile = T.tile({ti, tj});
            tile.materialize();
            for (int lr = 0; lr < sz[0][ti]; ++lr) {
                for (int lc = 0; lc < sz[1][tj]; ++lc) {
                    tile(std::vector<size_t>{static_cast<size_t>(lr), static_cast<size_t>(lc)}) = f(off[0][ti] + lr, off[1][tj] + lc);
                }
            }
        }
    }
}

// Reconstruct a dense R×C matrix from a 2-D tiled tensor (absent tiles read 0).
template <typename V>
std::vector<std::vector<V>> gather(TiledRuntimeTensor<V> const &T, int R, int C) {
    std::vector<std::vector<V>> M(R, std::vector<V>(C, V{}));
    auto const                 &off = T.tile_offsets();
    auto const                 &sz  = T.tile_sizes();
    for (auto const &[coord, tile] : T.tiles()) {
        int const ti = coord[0];
        int const tj = coord[1];
        for (int lr = 0; lr < sz[0][ti]; ++lr) {
            for (int lc = 0; lc < sz[1][tj]; ++lc) {
                M[off[0][ti] + lr][off[1][tj] + lc] = tile(std::vector<size_t>{static_cast<size_t>(lr), static_cast<size_t>(lc)});
            }
        }
    }
    return M;
}

/// Within 100 epsilon of @p want, relative to it and absolute near zero, for any element type.
template <typename V>
bool close_to(V got, V want) {
    return std::abs(got - want) <= 100.0 * std::numeric_limits<RemoveComplexT<V>>::epsilon() * (1.0 + std::abs(want));
}

/// The global coordinates of every element of a tile, visited in order: @p f(local, global).
template <typename F>
void for_each_tile_element(std::vector<int> const &coord, std::vector<std::vector<int>> const &sizes,
                           std::vector<std::vector<int>> const &offsets, F &&f) {
    size_t const        rank = coord.size();
    std::vector<size_t> local(rank, 0);
    std::vector<size_t> global(rank);
    for (size_t ax = 0; ax < rank; ++ax) {
        if (sizes[ax][coord[ax]] == 0) {
            return;
        }
    }
    while (true) {
        for (size_t ax = 0; ax < rank; ++ax) {
            global[ax] = static_cast<size_t>(offsets[ax][coord[ax]]) + local[ax];
        }
        f(local, global);
        size_t ax = rank;
        while (ax > 0) {
            --ax;
            if (++local[ax] < static_cast<size_t>(sizes[ax][coord[ax]])) {
                break;
            }
            local[ax] = 0;
            if (ax == 0) {
                return;
            }
        }
        if (rank == 0) {
            return;
        }
    }
}

/// A tiled tensor of any rank as a dense one; absent tiles read zero.
template <typename V>
RuntimeTensor<V> densify(TiledRuntimeTensor<V> const &T) {
    std::vector<size_t> dims;
    for (auto const &part : T.tile_sizes()) {
        size_t n = 0;
        for (int const p : part) {
            n += static_cast<size_t>(p);
        }
        dims.push_back(n);
    }
    RuntimeTensor<V> D("D", dims);
    D.zero();
    for (auto const &[coord, tile] : T.tiles()) {
        for_each_tile_element(coord, T.tile_sizes(), T.tile_offsets(),
                              [&](std::vector<size_t> const &local, std::vector<size_t> const &global) { D(global) = tile(local); });
    }
    return D;
}

/// Store every tile of @p T whose irreps (one per axis, irrep = tile index) multiply to the totally
/// symmetric one (XOR, as in an abelian point group), filled from a deterministic global function.
template <typename V>
void fill_symmetric(TiledRuntimeTensor<V> &T, int nirrep, double seed) {
    size_t const     rank = T.tile_sizes().size();
    std::vector<int> coord(rank, 0);
    while (true) {
        int x = 0;
        for (int const h : coord) {
            x ^= h;
        }
        if (x == 0) {
            auto &tile = T.tile(coord);
            tile.materialize();
            for_each_tile_element(coord, T.tile_sizes(), T.tile_offsets(),
                                  [&](std::vector<size_t> const &local, std::vector<size_t> const &g) {
                                      double v = seed;
                                      for (size_t ax = 0; ax < g.size(); ++ax) {
                                          v += (0.37 + 0.11 * static_cast<double>(ax)) * static_cast<double>(g[ax]);
                                      }
                                      tile(local) = testing::prefactor<V>(std::sin(v), std::cos(1.7 * v));
                                  });
        }
        size_t ax = rank;
        while (ax > 0) {
            --ax;
            if (++coord[ax] < nirrep) {
                break;
            }
            coord[ax] = 0;
            if (ax == 0) {
                return;
            }
        }
    }
}

/// The set of stored tile coordinates.
template <typename V>
std::set<std::vector<int>> tile_set(TiledRuntimeTensor<V> const &T) {
    std::set<std::vector<int>> out;
    for (auto const &[coord, tile] : T.tiles()) {
        out.insert(coord);
    }
    return out;
}

/// Every element of @p got within tolerance of @p want.
template <typename V>
bool all_close(RuntimeTensor<V> const &got, RuntimeTensor<V> const &want) {
    V const *g = got.data();
    V const *w = want.data();
    for (size_t e = 0; e < want.size(); ++e) {
        if (!close_to<V>(g[e], w[e])) {
            return false;
        }
    }
    return got.size() == want.size();
}

} // namespace

TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - tiled einsum GEMM (eager) matches dense", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    // i:{1,2}=3, k:{2,1}=3, j:{1,1}=2 ; contracted k partition matches in A and B.
    TiledRuntimeTensor<T> A("A", Grid{{1, 2}, {2, 1}});
    TiledRuntimeTensor<T> B("B", Grid{{2, 1}, {1, 1}});
    TiledRuntimeTensor<T> C("C", Grid{{1, 2}, {1, 1}});

    auto af = [](int r, int c) { return testing::prefactor<T>(1.0 + 3 * r + c, 0.5 * r - c); };
    auto bf = [](int r, int c) { return testing::prefactor<T>(0.5 + r - 2 * c, 1.0 - 0.25 * r); };
    fill_tiled(A, af);
    fill_tiled(B, bf);

    cg::einsum("ik;kj->ij", &C, A, B); // eager (c_pf=0, ab_pf=1)

    auto Cg = gather(C, 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            T expected{};
            for (int k = 0; k < 3; ++k) {
                expected += af(i, k) * bf(k, j);
            }
            REQUIRE(close_to<T>(Cg[i][j], expected));
        }
    }
    // Infer-and-create: every output grid cell received a contribution.
    REQUIRE(C.num_filled_tiles() == 4);
}

TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - tiled einsum GEMM inside a captured graph", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    TiledRuntimeTensor<T> A("A", Grid{{1, 2}, {2, 1}});
    TiledRuntimeTensor<T> B("B", Grid{{2, 1}, {1, 1}});
    TiledRuntimeTensor<T> C("C", Grid{{1, 2}, {1, 1}});

    auto af = [](int r, int c) { return testing::prefactor<T>(2.0 - r + 0.25 * c, 0.75 * c - 0.5); };
    auto bf = [](int r, int c) { return testing::prefactor<T>(1.0 + 0.5 * r * c, r - 0.5 * c); };
    fill_tiled(A, af); // inputs must be materialized before execute
    fill_tiled(B, bf);

    cg::Graph g("tiled_gemm");
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    g.execute();

    auto Cg = gather(C, 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            T expected{};
            for (int k = 0; k < 3; ++k) {
                expected += af(i, k) * bf(k, j);
            }
            REQUIRE(close_to<T>(Cg[i][j], expected));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - sparse inputs: missing tiles contribute zero", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    TiledRuntimeTensor<T> A("A", Grid{{1, 2}, {2, 1}});
    TiledRuntimeTensor<T> B("B", Grid{{2, 1}, {1, 1}});
    TiledRuntimeTensor<T> C("C", Grid{{1, 2}, {1, 1}});

    auto af = [](int r, int c) { return testing::prefactor<T>(1.0 + 3 * r + c, 0.5 * r - c); };
    auto bf = [](int r, int c) { return testing::prefactor<T>(0.5 + r - 2 * c, 1.0 - 0.25 * r); };

    // Populate only a subset of A's tiles (off-diagonal/rectangular pattern):
    // present: (0,0)=1x2, (1,1)=2x1.  Absent: (0,1), (1,0).
    auto const &asz  = A.tile_sizes();
    auto const &aoff = A.tile_offsets();
    for (std::vector<int> coord : {std::vector<int>{0, 0}, std::vector<int>{1, 1}}) {
        auto &tile = A.tile(coord);
        tile.materialize();
        for (int lr = 0; lr < asz[0][coord[0]]; ++lr) {
            for (int lc = 0; lc < asz[1][coord[1]]; ++lc) {
                tile(std::vector<size_t>{static_cast<size_t>(lr), static_cast<size_t>(lc)}) =
                    af(aoff[0][coord[0]] + lr, aoff[1][coord[1]] + lc);
            }
        }
    }
    fill_tiled(B, bf);

    cg::einsum("ik;kj->ij", &C, A, B);

    // Reference: A is zero wherever its tile is absent.
    auto a_present = [](int i, int k) { return (i == 0 && k < 2) || (i >= 1 && k >= 2); };
    auto Cg        = gather(C, 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            T expected{};
            for (int k = 0; k < 3; ++k) {
                if (a_present(i, k)) {
                    expected += af(i, k) * bf(k, j);
                }
            }
            REQUIRE(close_to<T>(Cg[i][j], expected));
        }
    }
}

TEST_CASE("TiledRuntimeTensor - misaligned contracted partition throws", "[ComputeGraph][TiledRuntime]") {
    TiledRuntimeTensor<double> A("A", Grid{{1, 2}, {2, 1}}); // k partition {2,1}
    TiledRuntimeTensor<double> B("B", Grid{{1, 2}, {1, 1}}); // k partition {1,2}, mismatch
    TiledRuntimeTensor<double> C("C", Grid{{1, 2}, {1, 1}});

    fill_tiled(A, [](int, int) { return 1.0; });
    fill_tiled(B, [](int, int) { return 1.0; });

    REQUIRE_THROWS(cg::einsum("ik;kj->ij", &C, A, B));
}

TEST_CASE("TiledRuntimeTensor - accumulating tiled einsum in a loop is not hoisted", "[ComputeGraph][TiledRuntime]") {
    // C += A*B with A and B loop-invariant: the CCSD residual shape. A tiled
    // einsum is an OpKind::Custom node carrying a TiledEinsumDescriptor, so
    // LoopInvariantHoisting cannot read its destination prefactor the way it
    // reads a dense EinsumDescriptor's, and its inputs are {A,B} by the same
    // descriptor-driven convention the dense einsum uses. Both self-modifying
    // checks therefore miss it, and the accumulation gets hoisted out of the
    // loop to run once.
    //
    // Compared differentially against the identical dense loop so the test does
    // not encode add_loop's iteration semantics.
    auto af = [](int r, int c) { return 1.0 + 0.5 * r - 0.25 * c; };
    auto bf = [](int r, int c) { return 0.75 - 0.5 * r + c; };

    auto build_loop = [](cg::Graph &g, auto &A, auto &B, auto &C) {
        auto                  &body = g.add_loop("loop", 4, [](size_t iter) { return iter < 3; });
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B); // C += A*B
    };

    // Dense oracle: OpKind::Einsum with a nonzero c_prefactor, which
    // reads_destination() already recognizes.
    Tensor<double, 2> Ad("Ad", 3, 3);
    Tensor<double, 2> Bd("Bd", 3, 2);
    Tensor<double, 2> Cd("Cd", 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            Ad(i, k) = af(i, k);
        }
    }
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 2; ++j) {
            Bd(k, j) = bf(k, j);
        }
    }
    Cd.zero();
    cg::Graph gd("dense_accum_loop");
    build_loop(gd, Ad, Bd, Cd);
    auto [dense_modified, dense_pass] = gd.apply<cg::passes::LoopInvariantHoisting>();
    CHECK(dense_pass.num_hoisted() == 0);
    gd.execute();

    // One pass of A*B at (0,0), to confirm the loop ran more than once.
    double single = 0.0;
    for (int k = 0; k < 3; ++k) {
        single += af(0, k) * bf(k, 0);
    }
    REQUIRE(std::abs(Cd(0, 0) - single) > 1e-12);

    TiledRuntimeTensor<double> At("At", Grid{{1, 2}, {2, 1}});
    TiledRuntimeTensor<double> Bt("Bt", Grid{{2, 1}, {1, 1}});
    TiledRuntimeTensor<double> Ct("Ct", Grid{{1, 2}, {1, 1}});
    fill_tiled(At, af);
    fill_tiled(Bt, bf);
    fill_tiled(Ct, [](int, int) { return 0.0; });
    cg::Graph gt("tiled_accum_loop");
    build_loop(gt, At, Bt, Ct);
    auto [tiled_modified, tiled_pass] = gt.apply<cg::passes::LoopInvariantHoisting>();
    REQUIRE(tiled_pass.num_hoisted() == 0);
    gt.execute();

    auto Ctg = gather(Ct, 3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            REQUIRE(std::abs(Ctg[i][j] - Cd(i, j)) < 1e-12);
        }
    }
}

// An irrep with no orbitals of a kind (water has no occupied A2) leaves tiles with zero extent. A pair
// of stored tiles that touches one contributes nothing, and the contraction skips its dense call; but
// it still decides the output's tile structure, which TiledExpansion reproduces, so the C tile it
// reaches must exist all the same.
TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - zero-extent tiles: output tiles still created, values match", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    TiledRuntimeTensor<T> A("A", Grid{{2, 0}, {1, 0, 2}}); // i: second block empty; k: middle block empty
    TiledRuntimeTensor<T> B("B", Grid{{1, 0, 2}, {0, 2}}); // j: first block empty
    TiledRuntimeTensor<T> C("C", Grid{{2, 0}, {0, 2}});
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 3; ++k) {
            A.tile({i, k}).materialize();
        }
    }
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 2; ++j) {
            B.tile({k, j}).materialize();
        }
    }
    for (auto &[coord, tile] : A.tiles()) {
        for (size_t e = 0; e < tile.size(); ++e) {
            tile.data()[e] = testing::prefactor<T>(0.5 + static_cast<double>(e), -0.25 * static_cast<double>(e));
        }
    }
    for (auto &[coord, tile] : B.tiles()) {
        for (size_t e = 0; e < tile.size(); ++e) {
            tile.data()[e] = testing::prefactor<T>(1.0 - 0.3 * static_cast<double>(e), 0.1 * static_cast<double>(e));
        }
    }

    cg::einsum("ik;kj->ij", &C, A, B);

    // Every (i, j) block is reached by some stored pair, the empty ones included.
    REQUIRE(C.num_filled_tiles() == 4);
    RuntimeTensor<T> want("want", std::vector<size_t>{2, 2});
    testing::reference_einsum("ij <- ik ; kj", &want, densify(A), densify(B));
    REQUIRE(all_close(densify(C), want));
}

// The CCSD ring term over symmetry-blocked operands, with an irrep that holds no occupied orbitals:
// pairs of stored tiles meet on the letters A and B share (m and e), and each C tile accumulates its
// contributions in the order the full grid walk gave them.
TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - symmetry-blocked rank-4 contraction matches a dense reference",
                        "[ComputeGraph][TiledRuntime]", testing::AllScalarTypes) {
    using T = TestType;
    std::vector<int> const occ{2, 0, 1, 1};
    std::vector<int> const vir{3, 2, 1, 2};
    TiledRuntimeTensor<T>  t2("t2", Grid{occ, occ, vir, vir}); // i m a e
    TiledRuntimeTensor<T>  W("W", Grid{occ, vir, vir, occ});   // m b e j
    TiledRuntimeTensor<T>  R("R", Grid{occ, occ, vir, vir});   // i j a b
    fill_symmetric(t2, 4, 0.1);
    fill_symmetric(W, 4, -0.4);
    fill_symmetric(R, 4, 0.7);
    RuntimeTensor<T> want = densify(R);

    cg::einsum("ijab <- imae ; mbej", T{0.5}, &R, T{2.0}, t2, W);

    testing::reference_einsum("ijab <- imae ; mbej", T{0.5}, &want, T{2.0}, densify(t2), densify(W));
    REQUIRE(all_close(densify(R), want));
    // R's stored tiles were all symmetry-allowed, and every contribution lands on one.
    for (auto const &coord : tile_set(R)) {
        REQUIRE((coord[0] ^ coord[1] ^ coord[2] ^ coord[3]) == 0);
    }
}

// An outer product shares no letter between A and B, so every stored A tile meets every stored B tile.
TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - tiled outer product pairs every stored tile", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    TiledRuntimeTensor<T> a("a", Grid{{2, 1, 3}});
    TiledRuntimeTensor<T> b("b", Grid{{1, 2}});
    TiledRuntimeTensor<T> C("C", Grid{{2, 1, 3}, {1, 2}});
    for (int t : {0, 2}) { // block 1 of a absent
        a.tile({t}).materialize();
    }
    b.tile({0}).materialize();
    b.tile({1}).materialize();
    for (auto *x : {&a, &b}) {
        for (auto &[coord, tile] : x->tiles()) {
            for (size_t e = 0; e < tile.size(); ++e) {
                tile.data()[e] = testing::prefactor<T>(1.0 + static_cast<double>(e) + coord[0], 0.5 * coord[0]);
            }
        }
    }

    cg::einsum("ij <- i ; j", &C, a, b);

    REQUIRE(tile_set(C) == std::set<std::vector<int>>{{0, 0}, {0, 1}, {2, 0}, {2, 1}});
    RuntimeTensor<T> want("want", std::vector<size_t>{6, 3});
    testing::reference_einsum("ij <- i ; j", &want, densify(a), densify(b));
    REQUIRE(all_close(densify(C), want));
}

namespace {

/// Occupied and virtual spaces over the irreps of an MO axis with @p nmo orbitals and @p nocc
/// occupied per irrep.
std::pair<IndexSpace, IndexSpace> occ_vir(std::vector<int> const &nmo, std::vector<int> const &nocc) {
    std::vector<std::pair<int, int>> o, v;
    for (size_t h = 0; h < nmo.size(); ++h) {
        o.emplace_back(0, nocc[h]);
        v.emplace_back(nocc[h], nmo[h]);
    }
    return {IndexSpace(o), IndexSpace(v)};
}

/// The view's blocks, copied into a tiled tensor of their own on the same grid.
template <typename V>
TiledRuntimeTensor<V> copy_of(TiledRuntimeTensorView<V> &view) {
    TiledRuntimeTensor<V>         out("copy", view.tile_sizes());
    std::vector<std::vector<int>> offsets(view.tile_sizes().size());
    for (size_t ax = 0; ax < offsets.size(); ++ax) {
        int at = 0;
        for (int const n : view.tile_sizes()[ax]) {
            offsets[ax].push_back(at);
            at += n;
        }
    }
    for (auto const &[coord, parent_tile] : view.parent().tiles()) {
        RuntimeTensorView<V> src  = view.tile_view(coord);
        auto                &tile = out.tile(coord);
        tile.materialize();
        for_each_tile_element(coord, view.tile_sizes(), offsets,
                              [&](std::vector<size_t> const &local, std::vector<size_t> const &) { tile(local) = src(local); });
    }
    return out;
}

} // namespace

// A pycc port slices irrep-blocked integrals by orbital space, as pycc slices ERI[v, v, v, v]. A view
// operand reads the parent's tiles cut to its spaces, on the parent's grid, so the contraction pairs
// the same tiles a copy of those blocks would. Not bit for bit: a view's tile is a strided slice of
// the parent's, and the dense kernel it takes may sum in another order than a contiguous copy's.
TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - a view operand contracts as a copy of its blocks", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::vector<int> const nmo{5, 2, 3, 4};
    std::vector<int> const nocc{2, 0, 1, 1};
    std::vector<int>       vir;
    for (size_t h = 0; h < nmo.size(); ++h) {
        vir.push_back(nmo[h] - nocc[h]);
    }
    auto const [o, v] = occ_vir(nmo, nocc);

    TiledRuntimeTensor<T> ERI("ERI", Grid{nmo, nmo, nmo, nmo});
    fill_symmetric(ERI, 4, 0.3);
    TiledRuntimeTensor<T> t2("t2", Grid{nocc, nocc, vir, vir});
    fill_symmetric(t2, 4, -0.2);

    auto                  vvvv     = ERI.view({v, v, v, v});
    TiledRuntimeTensor<T> eri_vvvv = copy_of(vvvv);

    TiledRuntimeTensor<T> by_view("by_view", Grid{nocc, nocc, vir, vir});
    TiledRuntimeTensor<T> by_copy("by_copy", Grid{nocc, nocc, vir, vir});
    cg::einsum("ijab <- ijef ; abef", T{0}, &by_view, T{0.5}, t2, vvvv);
    cg::einsum("ijab <- ijef ; abef", T{0}, &by_copy, T{0.5}, t2, eri_vvvv);

    REQUIRE(tile_set(by_view) == tile_set(by_copy));
    REQUIRE(all_close(densify(by_view), densify(by_copy)));
    RuntimeTensor<T> want = densify(by_copy);
    want.zero();
    testing::reference_einsum("ijab <- ijef ; abef", T{0}, &want, T{0.5}, densify(t2), densify(eri_vvvv));
    REQUIRE(all_close(densify(by_view), want));
}

// Through a view, C is scaled and written only within the view's sub-blocks: the parent's other
// elements keep their values, and an absent parent tile the contraction reaches is created whole,
// zero outside the view's slice.
TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - a view output touches only its sub-blocks", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::vector<int> const nmo{3, 2};
    std::vector<int> const nocc{1, 1};
    auto const [o, v] = occ_vir(nmo, nocc);

    TiledRuntimeTensor<T> A("A", Grid{{1, 1}, {2, 1}}); // i (occ), e (vir)
    TiledRuntimeTensor<T> B("B", Grid{{2, 1}, {2, 1}}); // e (vir), a (vir)
    fill_symmetric(A, 2, 0.4);
    fill_symmetric(B, 2, -0.6);

    // The MO-span output stores only its (0, 0) tile, filled everywhere with 3.
    TiledRuntimeTensor<T> F("F", Grid{nmo, nmo});
    auto                 &f00 = F.tile({0, 0});
    f00.materialize();
    f00.set_all(T{3});
    auto ov = F.view({o, v});

    cg::einsum("ia <- ie ; ea", T{0.5}, &ov, T{1}, A, B);

    // Both diagonal blocks are reached; (1, 1) did not exist and is created.
    REQUIRE(tile_set(F) == std::set<std::vector<int>>{{0, 0}, {1, 1}});
    RuntimeTensor<T> want("want", std::vector<size_t>{2, 3});
    want.set_all(T{0});
    {
        RuntimeTensor<T> prior("prior", std::vector<size_t>{2, 3});
        prior.zero();
        prior(std::vector<size_t>{0, 0}) = T{3}; // F(0, 1) and F(0, 2): irrep 0's o x v block, filled with 3
        prior(std::vector<size_t>{0, 1}) = T{3};
        testing::reference_einsum("ia <- ie ; ea", T{0}, &want, T{1}, densify(A), densify(B));
        for (size_t e = 0; e < want.size(); ++e) {
            want.data()[e] += T{0.5} * prior.data()[e];
        }
    }
    // The view's sub-blocks hold the result.
    RuntimeTensor<T> got("got", std::vector<size_t>{2, 3});
    got.zero();
    for (int h = 0; h < 2; ++h) {
        RuntimeTensorView<T> slice = ov.tile_view({h, h});
        for (size_t r = 0; r < slice.dim(0); ++r) {
            for (size_t c = 0; c < slice.dim(1); ++c) {
                got(std::vector<size_t>{static_cast<size_t>(h) + r, (h == 0 ? 0 : 2) + c}) = slice(std::vector<size_t>{r, c});
            }
        }
    }
    REQUIRE(all_close(got, want));
    // Outside them, F(0, 0) of irrep 0 and the o x o, v x o, v x v elements keep their 3.
    REQUIRE(f00(std::vector<size_t>{0, 0}) == T{3});
    REQUIRE(f00(std::vector<size_t>{1, 1}) == T{3});
    REQUIRE(f00(std::vector<size_t>{2, 2}) == T{3});
    REQUIRE(f00(std::vector<size_t>{1, 0}) == T{3});
    // The created tile is zero outside its o x v slice.
    auto const &f11 = F.tile({1, 1});
    REQUIRE(f11(std::vector<size_t>{0, 0}) == T{0});
    REQUIRE(f11(std::vector<size_t>{1, 0}) == T{0});
    REQUIRE(f11(std::vector<size_t>{1, 1}) == T{0});
}

// Captured, a view is a copy: a view operand's slices are copied into a graph-owned tiled tensor,
// and a view output is contracted into one and copied back into its parent's slices, so the graph
// holds only whole tiled tensors. Replayed, raw or optimized, it must give what the eager calls give,
// accumulation included: c_pf scales the view's sub-blocks on every replay and nothing else.
TEST_CASE("TiledRuntimeTensor - view operands inside a captured graph match the eager calls", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const nmo{5, 2, 3, 4};
    std::vector<int> const nocc{2, 0, 1, 1};
    std::vector<int>       vir;
    for (size_t h = 0; h < nmo.size(); ++h) {
        vir.push_back(nmo[h] - nocc[h]);
    }
    auto const [o, v] = occ_vir(nmo, nocc);

    auto const build = [&](char const *tag) {
        struct Operands {
            TiledRuntimeTensor<double> ERI, t2, R2, F, A, B;
        };
        Operands ops{TiledRuntimeTensor<double>(std::string("ERI") + tag, Grid{nmo, nmo, nmo, nmo}),
                     TiledRuntimeTensor<double>(std::string("t2") + tag, Grid{nocc, nocc, vir, vir}),
                     TiledRuntimeTensor<double>(std::string("R2") + tag, Grid{nocc, nocc, vir, vir}),
                     TiledRuntimeTensor<double>(std::string("F") + tag, Grid{nmo, nmo}),
                     TiledRuntimeTensor<double>(std::string("A") + tag, Grid{nocc, vir}),
                     TiledRuntimeTensor<double>(std::string("B") + tag, Grid{vir, vir})};
        fill_symmetric(ops.ERI, 4, 0.3);
        fill_symmetric(ops.t2, 4, -0.2);
        fill_symmetric(ops.R2, 4, 0.6);
        fill_symmetric(ops.F, 4, 0.9);
        fill_symmetric(ops.A, 4, 0.4);
        fill_symmetric(ops.B, 4, -0.6);
        return ops;
    };

    // The eager reference: both contractions, twice.
    auto ref = build("_ref");
    for (int replay = 0; replay < 2; ++replay) {
        auto vvvv = ref.ERI.view({v, v, v, v});
        auto ov   = ref.F.view({o, v});
        cg::einsum("ijab <- ijef ; abef", 0.5, &ref.R2, 1.0, ref.t2, vvvv);
        cg::einsum("ia <- ie ; ea", 0.5, &ov, 1.0, ref.A, ref.B);
    }

    for (bool const optimize : {false, true}) {
        auto      ops = build(optimize ? "_opt" : "_raw");
        cg::Graph graph(optimize ? "views_opt" : "views_raw");
        {
            cg::CaptureGuard const guard(graph);
            auto                   vvvv = ops.ERI.view({v, v, v, v});
            auto                   ov   = ops.F.view({o, v});
            cg::einsum("ijab <- ijef ; abef", 0.5, &ops.R2, 1.0, ops.t2, vvvv);
            cg::einsum("ia <- ie ; ea", 0.5, &ov, 1.0, ops.A, ops.B);
        }
        if (optimize) {
            graph.optimize();
        }
        graph.execute();
        graph.execute();
        INFO("optimized: " << optimize);
        REQUIRE(all_close(densify(ops.R2), densify(ref.R2)));
        REQUIRE(all_close(densify(ops.F), densify(ref.F))); // the view's slices and everything outside them
        REQUIRE(tile_set(ops.F) == tile_set(ref.F));
    }
}

// A captured view copy is a tiled node, and a tiled node records no descriptor a saved graph can
// rebuild, so saving refuses it as it refuses every tiled node.
TEST_CASE("TiledRuntimeTensor - a graph holding a tiled view copy refuses to save", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const nmo{3, 2};
    std::vector<int> const nocc{1, 1};
    auto const [o, v] = occ_vir(nmo, nocc);
    TiledRuntimeTensor<double> M("M", Grid{nmo, nmo});
    fill_symmetric(M, 2, 0.1);
    TiledRuntimeTensor<double> C("C", Grid{nocc, nocc});
    cg::Graph                  graph("view_save");
    {
        cg::CaptureGuard const guard(graph);
        auto                   ov = M.view({o, v});
        cg::einsum("ij <- ia ; ja", &C, ov, ov);
    }
    REQUIRE_FALSE(graph.serializability_report().empty());
}

// An eager tiled einsum chooses between running each pair of stored tiles and densifying (gather A
// and B, contract once, add the result into C's tiles), priced by the shared cost model as
// TiledExpansion prices a captured one. Both lowerings must create the same output tiles and agree
// to rounding, whatever the operands' sparsity and C's prior tiles.
TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - per-tile and densified lowerings agree", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::vector<int> const occ{2, 0, 1, 1};
    std::vector<int> const vir{3, 2, 1, 2};
    TiledRuntimeTensor<T>  t2("t2", Grid{occ, occ, vir, vir});
    TiledRuntimeTensor<T>  W("W", Grid{occ, vir, vir, occ});
    fill_symmetric(t2, 4, 0.1);
    fill_symmetric(W, 4, -0.4);
    auto const parsed = cg::detail::parse_einsum_cached("ijab <- imae ; mbej");

    std::set<std::vector<int>> tiles_by[2];
    RuntimeTensor<T>           result_by[2];
    for (int mode = 0; mode < 2; ++mode) {
        TiledRuntimeTensor<T> R("R", Grid{occ, occ, vir, vir});
        // A prior tile the contraction reaches, and one it does not (absent occupied block on i).
        R.tile({0, 0, 0, 0}).materialize();
        R.tile({0, 0, 0, 0}).set_all(T{2});
        R.tile({1, 1, 0, 0}).materialize();
        R.tile({1, 1, 0, 0}).set_all(T{5});
        RuntimeTensor<T> want     = densify(R);
        auto const       lowering = mode == 0 ? cg::detail::TiledLowering::PerTile : cg::detail::TiledLowering::Densify;
        cg::detail::tiled_runtime_einsum<T>(*parsed, T{0.5}, &R, T{2.0}, t2, W, lowering);
        REQUIRE(std::string(cg::detail::last_tiled_lowering()) == (mode == 0 ? "per_tile" : "densified"));

        testing::reference_einsum("ijab <- imae ; mbej", T{0.5}, &want, T{2.0}, densify(t2), densify(W));
        REQUIRE(all_close(densify(R), want));
        tiles_by[mode]  = tile_set(R);
        result_by[mode] = densify(R);
    }
    REQUIRE(tiles_by[0] == tiles_by[1]);
    REQUIRE(all_close(result_by[1], result_by[0]));
}

// A view operand and a view output take the densified lowering too: A reads through the view's
// slices, and the result is added only into the view's sub-blocks of C's parent.
TEMPLATE_LIST_TEST_CASE("TiledRuntimeTensor - densified lowering reads and writes through views", "[ComputeGraph][TiledRuntime]",
                        testing::AllScalarTypes) {
    using T = TestType;
    std::vector<int> const nmo{5, 2, 3, 4};
    std::vector<int> const nocc{2, 0, 1, 1};
    std::vector<int>       vir;
    for (size_t h = 0; h < nmo.size(); ++h) {
        vir.push_back(nmo[h] - nocc[h]);
    }
    auto const [o, v] = occ_vir(nmo, nocc);
    TiledRuntimeTensor<T> ERI("ERI", Grid{nmo, nmo, nmo, nmo});
    fill_symmetric(ERI, 4, 0.3);
    TiledRuntimeTensor<T> t2("t2", Grid{nocc, nocc, vir, vir});
    fill_symmetric(t2, 4, -0.2);
    auto       vvvv   = ERI.view({v, v, v, v});
    auto const parsed = cg::detail::parse_einsum_cached("ijab <- ijef ; abef");

    TiledRuntimeTensor<T> per_tile("per_tile", Grid{nocc, nocc, vir, vir});
    TiledRuntimeTensor<T> dense("dense", Grid{nocc, nocc, vir, vir});
    cg::detail::tiled_runtime_einsum<T>(*parsed, T{0}, &per_tile, T{0.5}, t2, vvvv, cg::detail::TiledLowering::PerTile);
    cg::detail::tiled_runtime_einsum<T>(*parsed, T{0}, &dense, T{0.5}, t2, vvvv, cg::detail::TiledLowering::Densify);
    REQUIRE(tile_set(per_tile) == tile_set(dense));
    REQUIRE(all_close(densify(dense), densify(per_tile)));

    // Writing through F[o, v] densified leaves F's other elements untouched.
    TiledRuntimeTensor<T> F("F", Grid{nmo, nmo});
    fill_symmetric(F, 4, 0.9);
    RuntimeTensor<T> const before = densify(F);
    TiledRuntimeTensor<T>  A("A", Grid{nocc, vir});
    TiledRuntimeTensor<T>  B("B", Grid{vir, vir});
    fill_symmetric(A, 4, 0.4);
    fill_symmetric(B, 4, -0.6);
    auto       ov = F.view({o, v});
    auto const p2 = cg::detail::parse_einsum_cached("ia <- ie ; ea");
    cg::detail::tiled_runtime_einsum<T>(*p2, T{1}, &ov, T{1}, A, B, cg::detail::TiledLowering::Densify);
    RuntimeTensor<T> const after = densify(F);
    // Rows are the MO index; an occupied row of irrep h is offset(h) + [0, nocc[h]), a virtual column
    // offset(h) + [nocc[h], nmo[h]).
    std::vector<int> off{0};
    for (int const n : nmo) {
        off.push_back(off.back() + n);
    }
    auto const in_ov = [&](size_t r, size_t c) {
        for (size_t h = 0; h < nmo.size(); ++h) {
            if (static_cast<int>(r) >= off[h] && static_cast<int>(r) < off[h] + nocc[h]) {
                for (size_t g = 0; g < nmo.size(); ++g) {
                    if (static_cast<int>(c) >= off[g] + nocc[g] && static_cast<int>(c) < off[g + 1]) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    size_t const n = static_cast<size_t>(off.back());
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            if (!in_ov(r, c)) {
                REQUIRE(after(std::vector<size_t>{r, c}) == before(std::vector<size_t>{r, c}));
            }
        }
    }
}

// Auto prices both lowerings with the machine's cost model, so only choices with a margin no machine
// closes are pinned: a CCD-style contraction over many tiny irrep tiles makes hundreds of calls that
// one small dense contraction replaces, and four large block-diagonal tiles would densify into 64
// times their arithmetic.
TEST_CASE("TiledRuntimeTensor - automatic lowering densifies tiny tiles and keeps large sparse ones", "[ComputeGraph][TiledRuntime]") {
    {
        std::vector<int> const     occ{3, 0, 1, 1};
        std::vector<int> const     vir{8, 2, 3, 6};
        TiledRuntimeTensor<double> t2("t2", Grid{occ, occ, vir, vir});
        TiledRuntimeTensor<double> W("W", Grid{occ, vir, vir, occ});
        TiledRuntimeTensor<double> R("R", Grid{occ, occ, vir, vir});
        fill_symmetric(t2, 4, 0.1);
        fill_symmetric(W, 4, -0.4);
        cg::einsum("ijab <- imae ; mbej", &R, t2, W);
        REQUIRE(std::string(cg::detail::last_tiled_lowering()) == "densified");
    }
    {
        std::vector<int> const     blocks{128, 128, 128, 128};
        TiledRuntimeTensor<double> A("A", Grid{blocks, blocks});
        TiledRuntimeTensor<double> B("B", Grid{blocks, blocks});
        TiledRuntimeTensor<double> C("C", Grid{blocks, blocks});
        for (int h = 0; h < 4; ++h) {
            A.tile({h, h}).materialize();
            A.tile({h, h}).set_all(1.0);
            B.tile({h, h}).materialize();
            B.tile({h, h}).set_all(0.5);
        }
        cg::einsum("ij <- ik ; kj", &C, A, B);
        REQUIRE(std::string(cg::detail::last_tiled_lowering()) == "per_tile");
        REQUIRE(C.num_filled_tiles() == 4);
        REQUIRE(C.tile({2, 2})(std::vector<size_t>{3, 5}) == Catch::Approx(64.0));
    }
}

// A repeated eager contraction replays a remembered plan: the pairs, the output tiles they reach and
// the dense buffers, keyed on the spec, the partitions and which tiles A and B store. Nothing the plan
// keeps may go stale: new values on the same tiles, a tile added between calls, and a NaN in one call
// followed by finite data must each give exactly what a first call would.
TEST_CASE("TiledRuntimeTensor - a remembered densified plan never serves stale data", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const     occ{3, 0, 1, 1};
    std::vector<int> const     vir{8, 2, 3, 6};
    TiledRuntimeTensor<double> t2("t2", Grid{occ, occ, vir, vir});
    TiledRuntimeTensor<double> W("W", Grid{occ, vir, vir, occ});
    fill_symmetric(t2, 4, 0.1);
    fill_symmetric(W, 4, -0.4);

    auto const run = [&](double c_pf) {
        TiledRuntimeTensor<double> R("R", Grid{occ, occ, vir, vir});
        fill_symmetric(R, 4, 0.7);
        RuntimeTensor<double> want = densify(R);
        cg::einsum("ijab <- imae ; mbej", c_pf, &R, 1.0, t2, W);
        REQUIRE(std::string(cg::detail::last_tiled_lowering()) == "densified");
        testing::reference_einsum("ijab <- imae ; mbej", c_pf, &want, 1.0, densify(t2), densify(W));
        return all_close(densify(R), want);
    };

    REQUIRE(run(0.5)); // plans
    REQUIRE(run(0.5)); // replays

    SECTION("new values on the same tiles") {
        fill_symmetric(t2, 4, 0.9);
        fill_symmetric(W, 4, 0.3);
        REQUIRE(run(1.0));
    }
    SECTION("a tile added to A") {
        // A symmetry-forbidden block: it changes which pairs contribute, so the plan must too.
        auto &extra = t2.tile({0, 0, 0, 1});
        extra.materialize();
        extra.set_all(0.25);
        REQUIRE(run(0.5));
    }
    SECTION("a NaN in one call does not reach the next") {
        auto        &tile                     = t2.tile({0, 0, 0, 0});
        double const kept                     = tile(std::vector<size_t>{0, 0, 0, 0});
        tile(std::vector<size_t>{0, 0, 0, 0}) = std::numeric_limits<double>::quiet_NaN();
        TiledRuntimeTensor<double> R("R", Grid{occ, occ, vir, vir});
        cg::einsum("ijab <- imae ; mbej", &R, t2, W);
        tile(std::vector<size_t>{0, 0, 0, 0}) = kept;
        REQUIRE(run(0.0));
    }
}

// Two views of one parent with the same extents select different elements; they share a plan's key,
// so the plan must read through whichever view the call passes.
TEST_CASE("TiledRuntimeTensor - views sharing a plan read their own slices", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const nmo{6, 2, 4, 4};
    std::vector<int> const nocc{3, 1, 2, 2}; // occupied and virtual extents equal in every irrep
    auto const [o, v] = occ_vir(nmo, nocc);
    TiledRuntimeTensor<double> M("M", Grid{nmo, nmo});
    fill_symmetric(M, 4, 0.2);
    TiledRuntimeTensor<double> X("X", Grid{nocc, nocc});
    fill_symmetric(X, 4, -0.3);

    for (auto const &[first, second] : {std::pair{o, v}, std::pair{v, o}, std::pair{o, v}}) {
        auto                       slice = M.view({first, second});
        TiledRuntimeTensor<double> C("C", Grid{nocc, nocc});
        cg::einsum("ij <- ik ; kj", &C, X, slice);
        RuntimeTensor<double> want("want", std::vector<size_t>{8, 8});
        want.zero();
        // The slice as a dense matrix, read element by element through the view.
        RuntimeTensor<double> dense_slice("slice", std::vector<size_t>{8, 8});
        dense_slice.zero();
        std::vector<int> off{0};
        for (int const n : nocc) {
            off.push_back(off.back() + n);
        }
        for (int h = 0; h < 4; ++h) {
            if (!slice.has_tile({h, h})) {
                continue;
            }
            RuntimeTensorView<double> b = slice.tile_view({h, h});
            for (size_t r = 0; r < b.dim(0); ++r) {
                for (size_t c = 0; c < b.dim(1); ++c) {
                    dense_slice(std::vector<size_t>{off[h] + r, off[h] + c}) = b(std::vector<size_t>{r, c});
                }
            }
        }
        testing::reference_einsum("ij <- ik ; kj", &want, densify(X), dense_slice);
        REQUIRE(all_close(densify(C), want));
    }
}

// TiledExpansion lowers a view copy to one node per tile, reading the whole parent tile and copying
// its slice, so a parent touched only through views expands like any tiled tensor instead of
// stranding every contraction on it. Both lowerings of the contraction must then reproduce the eager
// calls, replay after replay.
TEST_CASE("TiledRuntimeTensor - TiledExpansion lowers view copies per tile", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const nmo{5, 2, 3, 4};
    std::vector<int> const nocc{2, 0, 1, 1};
    std::vector<int>       vir;
    for (size_t h = 0; h < nmo.size(); ++h) {
        vir.push_back(nmo[h] - nocc[h]);
    }
    auto const [o, v] = occ_vir(nmo, nocc);

    for (auto const mode : {cg::passes::Densify::Never, cg::passes::Densify::Always}) {
        TiledRuntimeTensor<double> ERI("ERI", Grid{nmo, nmo, nmo, nmo}), t2("t2", Grid{nocc, nocc, vir, vir}),
            R2("R2", Grid{nocc, nocc, vir, vir}), F("F", Grid{nmo, nmo}), A("A", Grid{nocc, vir}), B("B", Grid{vir, vir});
        TiledRuntimeTensor<double> R2_ref("R2r", Grid{nocc, nocc, vir, vir}), F_ref("Fr", Grid{nmo, nmo});
        fill_symmetric(ERI, 4, 0.3);
        fill_symmetric(t2, 4, -0.2);
        fill_symmetric(A, 4, 0.4);
        fill_symmetric(B, 4, -0.6);
        fill_symmetric(R2, 4, 0.6);
        fill_symmetric(R2_ref, 4, 0.6);
        fill_symmetric(F, 4, 0.9);
        fill_symmetric(F_ref, 4, 0.9);

        for (int replay = 0; replay < 2; ++replay) {
            auto vvvv = ERI.view({v, v, v, v});
            auto ov   = F_ref.view({o, v});
            cg::einsum("ijab <- ijef ; abef", 0.5, &R2_ref, 1.0, t2, vvvv);
            cg::einsum("ia <- ie ; ea", 0.5, &ov, 1.0, A, B);
        }

        cg::Graph graph("views_expand");
        {
            cg::CaptureGuard const guard(graph);
            auto                   vvvv = ERI.view({v, v, v, v});
            auto                   ov   = F.view({o, v});
            cg::einsum("ijab <- ijef ; abef", 0.5, &R2, 1.0, t2, vvvv);
            cg::einsum("ia <- ie ; ea", 0.5, &ov, 1.0, A, B);
        }
        cg::PassManager pm;
        pm.add(std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, mode, cg::passes::FuseTiles::Never));
        REQUIRE(graph.apply(pm));

        size_t copies_in = 0, copies_out = 0;
        for (auto const &nd : graph.nodes()) {
            INFO("node: " << nd.label);
            REQUIRE_FALSE(nd.label.starts_with("tiled view copy"));
            REQUIRE_FALSE(nd.label.starts_with("tiled einsum"));
            copies_in += nd.label.starts_with("tile_view_copy_in") ? 1 : 0;
            copies_out += nd.label.starts_with("tile_view_copy_out") ? 1 : 0;
        }
        CHECK(copies_in > 0);
        CHECK(copies_out > 0);

        graph.execute();
        graph.execute();
        INFO("densify: " << (mode == cg::passes::Densify::Always));
        REQUIRE(all_close(densify(R2), densify(R2_ref)));
        REQUIRE(all_close(densify(F), densify(F_ref)));
        REQUIRE(tile_set(F) == tile_set(F_ref));
    }
}

// Two contractions read the same slices of P, and P is scaled between them. The second must see the
// scaled values: capture may share one copy of a slice only while nothing writes the parent, and a
// densified gather may read the parent in place of a copy only when nothing writes it between the
// copy and the gather. Checked raw and fully optimized, replayed twice, against the eager calls.
TEST_CASE("TiledRuntimeTensor - a write to the parent between two view reads is seen by the second", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const nmo{5, 2, 3, 4};
    std::vector<int> const nocc{2, 0, 1, 1};
    std::vector<int>       vir;
    for (size_t h = 0; h < nmo.size(); ++h) {
        vir.push_back(nmo[h] - nocc[h]);
    }
    auto const [o, v] = occ_vir(nmo, nocc);

    auto const run = [&](cg::Graph *graph, bool optimize) {
        TiledRuntimeTensor<double> P("P", Grid{nmo, nmo}), X("X", Grid{nocc, nocc}), C1("C1", Grid{nocc, vir}), C2("C2", Grid{nocc, vir});
        fill_symmetric(P, 4, 0.5);
        fill_symmetric(X, 4, -0.3);
        auto const body = [&] {
            auto ov = P.view({o, v});
            cg::einsum("ia <- ij ; ja", &C1, X, ov);
            cg::scale(2.0, &P);
            auto ov2 = P.view({o, v});
            cg::einsum("ia <- ij ; ja", &C2, X, ov2);
        };
        if (graph == nullptr) {
            body();
            body();
        } else {
            {
                cg::CaptureGuard const guard(*graph);
                body();
            }
            if (optimize) {
                graph->optimize();
            }
            graph->execute();
            graph->execute();
        }
        return std::pair{densify(C1), densify(C2)};
    };

    auto const [want1, want2] = run(nullptr, false);
    for (bool const optimize : {false, true}) {
        cg::Graph graph(optimize ? "parent_write_opt" : "parent_write_raw");
        auto const [got1, got2] = run(&graph, optimize);
        INFO("optimized: " << optimize);
        REQUIRE(all_close(got1, want1));
        REQUIRE(all_close(got2, want2));
    }
}

// A densified contraction reading a view gathers the parent's slices directly, so the view's copy
// has no reader left and dead-node elimination removes it; ERI[o, o, v, v] read under two names is
// one copy, not two, wherever a copy survives.
TEST_CASE("TiledRuntimeTensor - densified view reads leave no copy behind", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const nmo{5, 2, 3, 4};
    std::vector<int> const nocc{2, 0, 1, 1};
    std::vector<int>       vir;
    for (size_t h = 0; h < nmo.size(); ++h) {
        vir.push_back(nmo[h] - nocc[h]);
    }
    auto const [o, v] = occ_vir(nmo, nocc);
    TiledRuntimeTensor<double> ERI("ERI", Grid{nmo, nmo, nmo, nmo}), t2("t2", Grid{nocc, nocc, vir, vir});
    TiledRuntimeTensor<double> F1("F1", Grid{vir, vir}), F2("F2", Grid{vir, vir});
    TiledRuntimeTensor<double> F1_ref("F1r", Grid{vir, vir}), F2_ref("F2r", Grid{vir, vir});
    fill_symmetric(ERI, 4, 0.3);
    fill_symmetric(t2, 4, -0.2);
    {
        auto oovv = ERI.view({o, o, v, v});
        cg::einsum("ae <- mnaf ; mnef", &F1_ref, t2, oovv);
        cg::einsum("ae <- mnaf ; mnef", 2.0, &F2_ref, 1.0, t2, oovv);
    }

    cg::Graph graph("dead_copies");
    {
        cg::CaptureGuard const guard(graph);
        auto                   oovv_a = ERI.view({o, o, v, v});
        auto                   oovv_b = ERI.view({o, o, v, v}); // the same block under a second name
        cg::einsum("ae <- mnaf ; mnef", &F1, t2, oovv_a);
        cg::einsum("ae <- mnaf ; mnef", 2.0, &F2, 1.0, t2, oovv_b);
    }
    size_t captured_copies = 0;
    for (auto const &nd : graph.nodes()) {
        captured_copies += nd.label.starts_with("tiled view copy") ? 1 : 0;
    }
    CHECK(captured_copies == 1);

    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::TiledExpansion>(4096, -1.0, cg::passes::Densify::Always, cg::passes::FuseTiles::Never));
    pm.add(std::make_shared<cg::passes::DeadNodeElimination>());
    REQUIRE(graph.apply(pm));
    for (auto const &nd : graph.nodes()) {
        INFO("node: " << nd.label);
        CHECK_FALSE(nd.label.starts_with("tile_view_copy"));
    }
    graph.execute();
    REQUIRE(all_close(densify(F1), densify(F1_ref)));
    REQUIRE(all_close(densify(F2), densify(F2_ref)));
}

// In a captured CC loop the integrals do not change, so the copies of their view slices are loop
// invariant: LoopInvariantHoisting lifts them out of the loop body and they run once, not once per
// iteration. The accumulation itself stays in the loop.
TEST_CASE("TiledRuntimeTensor - copies of a constant view are hoisted out of a captured loop", "[ComputeGraph][TiledRuntime]") {
    std::vector<int> const nmo{5, 2, 3, 4};
    std::vector<int> const nocc{2, 0, 1, 1};
    std::vector<int>       vir;
    for (size_t h = 0; h < nmo.size(); ++h) {
        vir.push_back(nmo[h] - nocc[h]);
    }
    auto const [o, v] = occ_vir(nmo, nocc);
    TiledRuntimeTensor<double> ERI("ERI", Grid{nmo, nmo, nmo, nmo}), t2("t2", Grid{nocc, nocc, vir, vir});
    TiledRuntimeTensor<double> R2("R2", Grid{nocc, nocc, vir, vir}), R2_ref("R2r", Grid{nocc, nocc, vir, vir});
    fill_symmetric(ERI, 4, 0.3);
    fill_symmetric(t2, 4, -0.2);
    fill_symmetric(R2, 4, 0.6);
    fill_symmetric(R2_ref, 4, 0.6);
    for (int iter = 0; iter < 3; ++iter) {
        auto vvvv = ERI.view({v, v, v, v});
        cg::einsum("ijab <- ijef ; abef", 1.0, &R2_ref, 0.5, t2, vvvv);
    }

    cg::Graph graph("hoisted_views");
    {
        auto                  &body = graph.add_loop("cc", 3, [](size_t iter) { return iter < 3; });
        cg::CaptureGuard const guard(body);
        auto                   vvvv = ERI.view({v, v, v, v});
        cg::einsum("ijab <- ijef ; abef", 1.0, &R2, 0.5, t2, vvvv);
    }
    graph.optimize();
    size_t hoisted = 0;
    for (auto const &nd : graph.nodes()) {
        hoisted += nd.label.starts_with("tile_view_copy") || nd.label.starts_with("tile_gather") ? 1 : 0;
    }
    CHECK(hoisted > 0);
    graph.execute();
    REQUIRE(all_close(densify(R2), densify(R2_ref)));
}
