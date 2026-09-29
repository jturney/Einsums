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
// entry up by key; these cases fail if it stops.

#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/PackedGemmIndices.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <io.h>
#    define EINSUMS_TEST_DUP    _dup
#    define EINSUMS_TEST_DUP2   _dup2
#    define EINSUMS_TEST_CLOSE  _close
#    define EINSUMS_TEST_FILENO _fileno
#else
#    include <unistd.h>
#    define EINSUMS_TEST_DUP    dup
#    define EINSUMS_TEST_DUP2   dup2
#    define EINSUMS_TEST_CLOSE  close
#    define EINSUMS_TEST_FILENO fileno
#endif

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

namespace {

/// Everything written to file descriptor 2 while it lives, which is where the plan dump goes.
class StderrCapture {
  public:
    StderrCapture() : _path(std::filesystem::temp_directory_path() / "einsums_engine_options_stderr.txt") {
        std::fflush(stderr);
        _file  = std::fopen(_path.string().c_str(), "w+");
        _saved = EINSUMS_TEST_DUP(2);
        EINSUMS_TEST_DUP2(EINSUMS_TEST_FILENO(_file), 2);
    }

    StderrCapture(StderrCapture const &)            = delete;
    StderrCapture &operator=(StderrCapture const &) = delete;

    ~StderrCapture() { restore(); }

    /// Stop capturing and return what was written.
    std::string text() {
        restore();
        std::ifstream     in(_path);
        std::string const out{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        std::filesystem::remove(_path);
        return out;
    }

  private:
    void restore() {
        if (_saved < 0) {
            return;
        }
        std::fflush(stderr);
        EINSUMS_TEST_DUP2(_saved, 2);
        EINSUMS_TEST_CLOSE(_saved);
        _saved = -1;
        std::fclose(_file);
    }

    std::filesystem::path _path;
    std::FILE            *_file{nullptr};
    int                   _saved{-1};
};

/// Set the two options for one scope and put back what they held before it.
struct EngineOptions {
    EngineOptions(bool dump_plan, std::int64_t c_temp_kib) {
        config::set(option::PackedGemmDumpPlan, dump_plan);
        config::set(option::PackedGemmCTempBudget, c_temp_kib);
    }
    EngineOptions(EngineOptions const &)            = delete;
    EngineOptions &operator=(EngineOptions const &) = delete;
    ~EngineOptions() {
        config::set(option::PackedGemmDumpPlan, _dump_plan);
        config::set(option::PackedGemmCTempBudget, _c_temp_kib);
    }

  private:
    bool         _dump_plan{config::get(option::PackedGemmDumpPlan)};
    std::int64_t _c_temp_kib{config::get(option::PackedGemmCTempBudget)};
};

/// The CCSD ring, C(a,b,i,j) = A(a,e,i,m) * B(e,b,m,j), in complex: C's M and N groups interleave, so
/// the result is scattered, which complex does by the block-GEMM strategy the C-temp budget sizes.
/// Returns the plan lines the engine printed.
std::vector<std::string> ring_plan_lines() {
    using T = std::complex<double>;
    auto A  = create_random_tensor<T>("A", 24, 12, 13, 9);
    auto B  = create_random_tensor<T>("B", 12, 22, 9, 11);
    auto C  = create_random_tensor<T>("C", 24, 22, 13, 11);

    StderrCapture capture;
    bool const    handled = tensor_algebra::detail::try_packed_gemm_indices<false, false>(T{0.0}, Indices{a, b, i, j}, &C, T{1.0},
                                                                                          Indices{a, e, i, m}, A, Indices{e, b, m, j}, B);
    std::string const printed = capture.text();
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");

    std::vector<std::string> lines;
    for (std::size_t pos = printed.find("[packed plan]"); pos != std::string::npos; pos = printed.find("[packed plan]", pos + 1)) {
        lines.push_back(printed.substr(pos, printed.find('\n', pos) - pos));
    }
    return lines;
}

} // namespace

TEST_CASE("engine options: dump-plan prints from the caller's engine", "[PackedGemm][Options]") {
    SECTION("off prints nothing") {
        EngineOptions const options{false, 0};
        CHECK(ring_plan_lines().empty());
    }
    SECTION("on prints the plan") {
        EngineOptions const options{true, 0};
        CHECK_FALSE(ring_plan_lines().empty());
    }
}

TEST_CASE("engine options: c-temp-budget resizes the caller's block strategy", "[PackedGemm][Options]") {
    std::vector<std::string> derived;
    {
        EngineOptions const options{true, 0};
        derived = ring_plan_lines();
    }
    // The budget sizes only the block strategy, which complex takes on rungs without a 1m route: x86
    // below the flag, and aarch64's NEON rung. The per-rung registrations reach it on any machine.
    std::string const engine = packed_gemm::last_packed_engine();
    if (engine != "block_gemm" && engine != "3m") {
        SKIP("complex runs the " << engine << " engine on this rung, which has no C temporary to budget");
    }

    std::vector<std::string> tiny;
    {
        EngineOptions const options{true, 1};
        tiny = ring_plan_lines();
    }
    REQUIRE_FALSE(derived.empty());
    REQUIRE_FALSE(tiny.empty());
    // One KiB is far below the derived budget of at least 512 KiB, so the blocks must shrink.
    CHECK(derived != tiny);
}
