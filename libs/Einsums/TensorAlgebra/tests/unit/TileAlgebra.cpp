//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The compile-time tiled einsum (Backends/TileAlgebra.hpp). Threads divide the output tiles, each
// owned by one thread that sums its contributions in grid order, and a contraction too small to be
// worth a thread team runs serially. Results must not depend on which: the same contraction threaded
// and serial agrees with a dense reference, and two threaded runs agree bit for bit.

#include <Einsums/Hardware/Options.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;

namespace {

// Four irreps, one with no orbitals.
constexpr std::array<int, 4> kPart{3, 0, 2, 4};
constexpr int                kN = 9;

int offset_of(int h) {
    int at = 0;
    for (int k = 0; k < h; ++k) {
        at += kPart[k];
    }
    return at;
}

/// Store every tile (h, g) with h ^ g == 0 on an abelian group: the diagonal blocks here.
void fill_symmetric(TiledTensor<double, 2> &T, double seed) {
    for (int h = 0; h < 4; ++h) {
        auto &tile = T.tile(std::array<int, 2>{h, h});
        for (int r = 0; r < kPart[h]; ++r) {
            for (int c = 0; c < kPart[h]; ++c) {
                int const gr = offset_of(h) + r, gc = offset_of(h) + c;
                tile(r, c) = std::sin(seed + 0.37 * gr + 0.11 * gc);
            }
        }
    }
}

std::vector<double> dense(TiledTensor<double, 2> const &T) {
    std::vector<double> out(kN * kN, 0.0);
    for (auto const &[coord, tile] : T.tiles()) {
        for (int r = 0; r < kPart[coord[0]]; ++r) {
            for (int c = 0; c < kPart[coord[1]]; ++c) {
                out[(offset_of(coord[0]) + r) * kN + offset_of(coord[1]) + c] = tile(r, c);
            }
        }
    }
    return out;
}

bool close(std::vector<double> const &got, std::vector<double> const &want) {
    for (size_t e = 0; e < want.size(); ++e) {
        if (std::abs(got[e] - want[e]) > 1e-12 * (1.0 + std::abs(want[e]))) {
            return false;
        }
    }
    return true;
}

/// Pin the contraction thread gate for one scope: 0 always threads, a huge value never does.
struct ThreadGate {
    std::int64_t previous;
    explicit ThreadGate(std::int64_t flops) : previous(config::get(option::HardwareOmpMinParallelFlops)) {
        config::set(option::HardwareOmpMinParallelFlops, flops);
    }
    ~ThreadGate() { config::set(option::HardwareOmpMinParallelFlops, previous); }
    ThreadGate(ThreadGate const &)            = delete;
    ThreadGate &operator=(ThreadGate const &) = delete;
};

} // namespace

TEST_CASE("TileAlgebra - tiled GEMM agrees with a dense reference, threaded or serial", "[TensorAlgebra][TileAlgebra]") {
    TiledTensor<double, 2> A("A", kPart), B("B", kPart);
    fill_symmetric(A, 0.3);
    fill_symmetric(B, -0.7);
    std::vector<double> const a = dense(A), b = dense(B);

    for (std::int64_t const gate : {std::int64_t{0}, std::int64_t{1} << 62}) {
        ThreadGate const       pin(gate);
        TiledTensor<double, 2> C("C", kPart);
        fill_symmetric(C, 1.1); // prior content, scaled by c_pf
        std::vector<double> want = dense(C);
        for (auto &x : want) {
            x *= 0.5;
        }
        for (int i = 0; i < kN; ++i) {
            for (int j = 0; j < kN; ++j) {
                for (int k = 0; k < kN; ++k) {
                    want[i * kN + j] += 2.0 * a[i * kN + k] * b[k * kN + j];
                }
            }
        }
        einsum(0.5, Indices{i, j}, &C, 2.0, Indices{i, k}, A, Indices{k, j}, B);
        INFO("gate " << gate);
        REQUIRE(close(dense(C), want));
    }
}

TEST_CASE("TileAlgebra - a threaded tiled contraction gives the same bits every run", "[TensorAlgebra][TileAlgebra]") {
    ThreadGate const       pin(0); // threaded whenever there are threads
    TiledTensor<double, 2> A("A", kPart), B("B", kPart);
    fill_symmetric(A, 0.3);
    fill_symmetric(B, -0.7);
    std::vector<double> runs[2];
    for (auto &run : runs) {
        TiledTensor<double, 2> C("C", kPart);
        einsum(Indices{j, i}, &C, Indices{i, k}, A, Indices{k, j}, B); // C's letters transposed against A's
        run = dense(C);
    }
    REQUIRE(runs[0] == runs[1]);

    std::vector<double> const a = dense(A), b = dense(B);
    std::vector<double>       want(kN * kN, 0.0);
    for (int ii = 0; ii < kN; ++ii) {
        for (int jj = 0; jj < kN; ++jj) {
            for (int kk = 0; kk < kN; ++kk) {
                want[jj * kN + ii] += a[ii * kN + kk] * b[kk * kN + jj];
            }
        }
    }
    REQUIRE(close(runs[0], want));
}

TEST_CASE("TileAlgebra - a tiled full contraction agrees with a dense reference", "[TensorAlgebra][TileAlgebra]") {
    TiledTensor<double, 2> A("A", kPart), B("B", kPart);
    fill_symmetric(A, 0.3);
    fill_symmetric(B, -0.7);
    std::vector<double> const a = dense(A), b = dense(B);
    double                    want = 0.25 * 1.5;
    for (int e = 0; e < kN * kN; ++e) {
        want += a[e] * b[e];
    }
    for (std::int64_t const gate : {std::int64_t{0}, std::int64_t{1} << 62}) {
        ThreadGate const  pin(gate);
        Tensor<double, 0> C;
        C = 1.5;
        einsum(0.25, Indices{}, &C, 1.0, Indices{i, j}, A, Indices{i, j}, B);
        INFO("gate " << gate);
        REQUIRE(std::abs(static_cast<double>(C) - want) <= 1e-12 * (1.0 + std::abs(want)));
    }
}
