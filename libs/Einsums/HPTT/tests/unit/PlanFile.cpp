//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Plan files: a plan round-trips through a file, and a file built for another
// vector width, element size or format version is refused instead of run.
//
// A plan stores loop increments equal to the macro-kernel block, four vectors
// of the writing rung. Files once recorded nothing about that, so a plan
// written on one rung and read on another stepped by the wrong block and left
// elements of B unwritten. A float plan read as double is the same defect on a
// single machine (the float block is twice the double block), and it produced
// a max error of 1000 here before the file recorded its geometry.

#include <Einsums/HPTT/Files.hpp>
#include <Einsums/HPTT/HPTT.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

constexpr size_t N0 = 67, N1 = 45, N2 = 3;

std::vector<size_t> const size_a{N0, N1, N2};
std::vector<int> const    perm{1, 0, 2};
std::vector<size_t> const size_b{N1, N0, N2};

template <typename T>
std::vector<T> source() {
    std::vector<T> a(N0 * N1 * N2);
    for (size_t k = 0; k < a.size(); ++k) {
        a[k] = static_cast<T>(k % 1000) + T(0.25);
    }
    return a;
}

// B(j, i, l) = A(i, j, l), column-major.
template <typename T>
double max_error(std::vector<T> const &a, std::vector<T> const &b) {
    double err = 0;
    for (size_t l = 0; l < N2; ++l)
        for (size_t j = 0; j < N1; ++j)
            for (size_t i = 0; i < N0; ++i) {
                T const want = a[i + N0 * (j + N1 * l)];
                T const got  = b[j + N1 * (i + N0 * l)];
                err          = std::max(err, std::abs(static_cast<double>(want) - static_cast<double>(got)));
            }
    return err;
}

struct FileCloser {
    void operator()(std::FILE *fp) const { std::fclose(fp); }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

template <typename T>
File write_plan() {
    auto           a = source<T>();
    std::vector<T> b(a.size());
    auto           plan = hptt::create_plan(perm, 3, T(1), a.data(), size_a, size_a, T(0), b.data(), size_b, hptt::ESTIMATE, 1);
    File           fp(std::tmpfile());
    REQUIRE(fp != nullptr);
    hptt::setup_file(fp.get());
    plan->write_to_file(fp.get());
    return fp;
}

// Rewrite the checksum after a deliberate edit, so the file is a well-formed
// one from another machine rather than a corrupt one.
void reseal(std::FILE *fp) {
    uint32_t const check = hptt::compute_checksum(fp);
    REQUIRE(std::fseek(fp, offsetof(hptt::FileHeader, checksum), SEEK_SET) == 0);
    REQUIRE(std::fwrite(&check, sizeof(check), 1, fp) == 1);
    std::fflush(fp);
}

hptt::PlanTarget read_target(std::FILE *fp) {
    hptt::PlanTarget target{};
    REQUIRE(std::fseek(fp, sizeof(hptt::FileHeader), SEEK_SET) == 0);
    REQUIRE(std::fread(&target, sizeof(target), 1, fp) == 1);
    return target;
}

void write_target(std::FILE *fp, hptt::PlanTarget const &target) {
    REQUIRE(std::fseek(fp, sizeof(hptt::FileHeader), SEEK_SET) == 0);
    REQUIRE(std::fwrite(&target, sizeof(target), 1, fp) == 1);
    reseal(fp);
}

} // namespace

TEMPLATE_TEST_CASE("PlanFile - a plan round-trips through a file", "[hptt][plan-file]", float, double) {
    File fp = write_plan<TestType>();
    REQUIRE(hptt::verify_file(fp.get()) == 0);

    auto const target = read_target(fp.get());
    CHECK(target.element_size == sizeof(TestType));
    CHECK(target.vector_bits >= 128);

    auto                  a = source<TestType>();
    std::vector<TestType> b(a.size(), TestType(-1));
    auto                  plan = hptt::Transpose<TestType>::read_from_file(fp.get(), TestType(1), a.data(), TestType(0), b.data());
    plan->execute();
    CHECK(max_error(a, b) == 0.0);
}

TEST_CASE("PlanFile - a plan for another element size is refused", "[hptt][plan-file]") {
    File fp = write_plan<float>();

    auto                a = source<double>();
    std::vector<double> b(a.size(), -1.0);
    CHECK_THROWS_WITH(hptt::Transpose<double>::read_from_file(fp.get(), 1.0, a.data(), 0.0, b.data()),
                      Catch::Matchers::ContainsSubstring("4-byte elements") && Catch::Matchers::ContainsSubstring("8-byte elements"));
}

TEST_CASE("PlanFile - a plan from a rung with another vector width is refused", "[hptt][plan-file]") {
    // What a plan written on a wider rung looks like: the same file with twice
    // the recorded width. x86-64-v3 writes 256 where baseline reads 128.
    File fp     = write_plan<double>();
    auto target = read_target(fp.get());
    target.vector_bits *= 2;
    write_target(fp.get(), target);
    REQUIRE(hptt::verify_file(fp.get()) == 0);

    auto                a = source<double>();
    std::vector<double> b(a.size(), -1.0);
    CHECK_THROWS_WITH(hptt::Transpose<double>::read_from_file(fp.get(), 1.0, a.data(), 0.0, b.data()),
                      Catch::Matchers::ContainsSubstring("recreate the plan"));
}

TEST_CASE("PlanFile - a rung with the same vector width reads the plan", "[hptt][plan-file]") {
    // Baseline and x86-64-v2 build identical plans; only the width and element
    // size decide whether a plan applies, and the rung is informational.
    File fp     = write_plan<double>();
    auto target = read_target(fp.get());
    target.rung = target.rung == 0 ? 1 : 0;
    write_target(fp.get(), target);

    auto                a = source<double>();
    std::vector<double> b(a.size(), -1.0);
    auto                plan = hptt::Transpose<double>::read_from_file(fp.get(), 1.0, a.data(), 0.0, b.data());
    plan->execute();
    CHECK(max_error(a, b) == 0.0);
}

TEST_CASE("PlanFile - a file from before the format recorded its geometry is refused", "[hptt][plan-file]") {
    File fp = write_plan<double>();

    // Format version 0: the old header, byte 6 zero.
    char const old_version = 0;
    REQUIRE(std::fseek(fp.get(), offsetof(hptt::FileHeader, version) + 2, SEEK_SET) == 0);
    REQUIRE(std::fwrite(&old_version, 1, 1, fp.get()) == 1);
    std::fflush(fp.get());
    CHECK(hptt::verify_file(fp.get()) == 6);

    auto                a = source<double>();
    std::vector<double> b(a.size(), -1.0);
    CHECK_THROWS_WITH(hptt::Transpose<double>::read_from_file(fp.get(), 1.0, a.data(), 0.0, b.data()),
                      Catch::Matchers::ContainsSubstring("format version 0"));
}
