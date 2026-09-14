//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/Options.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/StorageBlock.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

namespace {

/// The named counter (in kB) from a /proc file of "Name:   value kB" lines, or -1 when absent.
std::int64_t proc_field_kb(char const *path, std::string const &field) {
    std::ifstream in(path);
    std::string   line;
    while (std::getline(in, line)) {
        if (line.rfind(field, 0) == 0) {
            return std::stoll(line.substr(field.size()));
        }
    }
    return -1;
}

std::int64_t anon_huge_kb() {
    return proc_field_kb("/proc/self/smaps_rollup", "AnonHugePages:");
}

/// /proc/vmstat counts THP faults the kernel could not satisfy; a rise during
/// the test means the machine had no huge pages to give, not that the advice
/// was missing.
std::int64_t thp_fault_fallback() {
    std::ifstream in("/proc/vmstat");
    std::string   key;
    std::int64_t  value = 0;
    while (in >> key >> value) {
        if (key == "thp_fault_fallback") {
            return value;
        }
    }
    return -1;
}

/// The bracketed choice in /sys/kernel/mm/transparent_hugepage/enabled, or "" off Linux.
std::string thp_policy() {
    std::ifstream in("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string   text;
    std::getline(in, text);
    auto const open = text.find('[');
    auto const shut = text.find(']');
    if (open == std::string::npos || shut == std::string::npos || shut <= open) {
        return "";
    }
    return text.substr(open + 1, shut - open - 1);
}

struct RestoreOption {
    bool previous = einsums::config::get(einsums::option::TensorHugePages);
    ~RestoreOption() { einsums::config::set(einsums::option::TensorHugePages, previous); }
};

constexpr std::int64_t kTensorBytes = std::int64_t{64} << 20;
constexpr std::int64_t kHugePageKb  = 2048;

} // namespace

TEST_CASE("advise_huge_pages is harmless on what it must not touch", "[tensor][hugepages]") {
    using namespace einsums;

    REQUIRE(detail::huge_page_advice_threshold() >= (size_t{2} << 20));

    // Nothing to advise: must not fault, must not care.
    detail::advise_huge_pages(nullptr, size_t{1} << 30);
    std::vector<char> small(1024, 'x');
    detail::advise_huge_pages(small.data(), small.size());
    REQUIRE(small[0] == 'x');

    // A buffer big enough to be advised keeps its contents and its size.
    std::vector<double> big;
    big.reserve((size_t{8} << 20) / sizeof(double));
    detail::advise_huge_pages(big.data(), big.capacity() * sizeof(double));
    big.assign(big.capacity(), 1.5);
    REQUIRE(big.back() == 1.5);
}

TEST_CASE("large tensor storage faults in huge pages", "[tensor][hugepages]") {
    using namespace einsums;

    auto const policy = thp_policy();
    if (policy.empty() || policy == "never") {
        SKIP("transparent huge pages unavailable or disabled on this host (policy '" << policy << "')");
    }
    if (anon_huge_kb() < 0) {
        SKIP("no /proc/self/smaps_rollup to read");
    }

    RestoreOption restore;
    config::set(option::TensorHugePages, true);

    auto const fallback_before = thp_fault_fallback();
    auto const huge_before     = anon_huge_kb();

    // 64 MiB: dozens of whole 2 MB pages however the buffer is aligned.
    RuntimeTensor<double> t("huge", std::vector<size_t>{static_cast<size_t>(kTensorBytes / sizeof(double) / 1024), 1024});
    t.data()[0] = 1.0;

    auto const huge_after = anon_huge_kb();
    if (thp_fault_fallback() > fallback_before) {
        SKIP("the kernel fell back to small pages during the test (no free huge pages)");
    }
    // Head and tail of the buffer may be small pages; the middle must not be.
    REQUIRE(huge_after - huge_before >= (kTensorBytes / 1024) / 2);
    REQUIRE(huge_after - huge_before >= kHugePageKb);
}

TEST_CASE("without --einsums:tensor:huge-pages tensor storage stays on small pages", "[tensor][hugepages]") {
    using namespace einsums;

    // Only the madvise policy makes the flag observable: under 'always' every
    // large mapping is huge whether or not we asked.
    if (thp_policy() != "madvise") {
        SKIP("THP policy is not 'madvise'; the option's effect is not observable");
    }
    if (anon_huge_kb() < 0) {
        SKIP("no /proc/self/smaps_rollup to read");
    }

    RestoreOption restore;
    config::set(option::TensorHugePages, false);

    auto const            huge_before = anon_huge_kb();
    RuntimeTensor<double> t("small", std::vector<size_t>{static_cast<size_t>(kTensorBytes / sizeof(double) / 1024), 1024});
    t.data()[0] = 1.0;
    // Nothing else in this process allocates at this scale here, so the count
    // must not move by even one huge page's worth.
    REQUIRE(anon_huge_kb() - huge_before < kHugePageKb);
}
