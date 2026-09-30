//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The packed engine's options reach the engine a CALLER compiles.
//
// blis_contraction is a header template, so the copy that runs here is instantiated in this test,
// not in the library, and it reads this binary's own copy of every descriptor, which the library's
// registration never fills in. config::get used to trust that copy's empty entry and answer with
// the declared default however the option was set: --einsums:packed-gemm:dump-plan printed nothing
// and c-temp-budget changed nothing, and every other test still passed. get now looks an empty
// entry up by key, one path for every option type; this case fails if it stops.

#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/PackedGemmIndices.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <complex>
#include <cstdint>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

namespace {

/// Set the C-temp budget for one scope and put back what it held before it.
struct CTempBudget {
    explicit CTempBudget(std::int64_t kib) { config::set(option::PackedGemmCTempBudget, kib); }
    CTempBudget(CTempBudget const &)            = delete;
    CTempBudget &operator=(CTempBudget const &) = delete;
    ~CTempBudget() { config::set(option::PackedGemmCTempBudget, _kib); }

  private:
    std::int64_t _kib{config::get(option::PackedGemmCTempBudget)};
};

/// The CCSD ring, C(a,b,i,j) = A(a,e,i,m) * B(e,b,m,j), in complex: C's M and N groups interleave, so
/// the result is scattered, which complex does by the block-GEMM strategy the C-temp budget sizes.
/// Returns the blocks the engine ran with.
packed_gemm::PackedBlocking ring_blocking() {
    using T = std::complex<double>;
    auto A  = create_random_tensor<T>("A", 24, 12, 13, 9);
    auto B  = create_random_tensor<T>("B", 12, 22, 9, 11);
    auto C  = create_random_tensor<T>("C", 24, 22, 13, 11);

    packed_gemm::last_packed_blocking() = {};
    bool const handled = tensor_algebra::detail::try_packed_gemm_indices<false, false>(T{0.0}, Indices{a, b, i, j}, &C, T{1.0},
                                                                                       Indices{a, e, i, m}, A, Indices{e, b, m, j}, B);
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    return packed_gemm::last_packed_blocking();
}

} // namespace

TEST_CASE("engine options: c-temp-budget resizes the caller's block strategy", "[PackedGemm][Options]") {
    packed_gemm::PackedBlocking derived;
    {
        CTempBudget const budget{0};
        derived = ring_blocking();
    }
    // The budget sizes only the block strategy, which complex takes on rungs without a 1m route: x86
    // below the flag, and aarch64's NEON rung. The per-rung registrations reach it on any machine.
    std::string const engine = packed_gemm::last_packed_engine();
    if (engine != "block_gemm" && engine != "3m") {
        SKIP("complex runs the " << engine << " engine on this rung, which has no C temporary to budget");
    }

    packed_gemm::PackedBlocking tiny;
    {
        CTempBudget const budget{1};
        tiny = ring_blocking();
    }
    REQUIRE(derived.mc > 0);
    REQUIRE(tiny.mc > 0);
    // One KiB is far below the derived budget of at least 512 KiB, so the blocks must shrink.
    CHECK(tiny.mc * tiny.nc < derived.mc * derived.nc);
}
