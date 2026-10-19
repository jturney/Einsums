//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TensorFileCommit.cpp
/// @brief What a crash leaves of an .etn file: every operation stopped at each call it makes, and
///        every subset of its unsynced writes lost to a power cut, must leave the file as it was
///        before the operation or as it is after it.

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorIO/Backing.hpp>
#include <Einsums/TensorIO/Entries.hpp>
#include <Einsums/TensorIO/Options.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "FileCalls.hpp"

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_io;
namespace cg        = einsums::compute_graph;
namespace io_detail = einsums::tensor_io::detail;

namespace {

struct TempFile {
    std::string path;
    std::string compacted;
    explicit TempFile(std::string const &name)
        : path((std::filesystem::temp_directory_path() / ("einsums_commit_" + name)).string()), compacted(path + ".compact") {
        clean();
    }
    ~TempFile() { clean(); }
    TempFile(TempFile const &)            = delete;
    TempFile &operator=(TempFile const &) = delete;
    TempFile(TempFile &&)                 = delete;
    TempFile &operator=(TempFile &&)      = delete;

  private:
    void clean() const noexcept {
        std::remove(path.c_str());
        std::remove(compacted.c_str());
    }
};

template <typename T>
RuntimeTensor<T> filled(std::string name, std::vector<size_t> const &dims, double seed) {
    RuntimeTensor<T> out(std::move(name), dims);
    for (size_t n = 0; n < out.size(); ++n) {
        double const x = seed + static_cast<double>(n);
        if constexpr (IsComplexV<T>) {
            out.data()[n] = T{static_cast<RemoveComplexT<T>>(x), static_cast<RemoveComplexT<T>>(-x)};
        } else {
            out.data()[n] = static_cast<T>(x);
        }
    }
    return out;
}

/// A file with entries of several types and orders, and a gap where a removed entry was.
void make_file(std::string const &path) {
    {
        TensorFile out(path, TensorFile::Mode::Write);
        out.write("a", filled<double>("a", {3, 4}, 1.0));
        out.write("gone", filled<double>("gone", {5, 5}, 50.0));
        out.write("c", filled<std::complex<float>>("c", {5}, 7.0));
        auto const b = filled<double>("b", {2, 3, 4}, 100.0);
        out.reserve<double>("b", {2, 3, 4}, {2, 0, 1});
        out.write_slice("b", b, {{0, 2}, {0, 3}, {0, 4}});
        out.reserve<double>("g", {2, 3, 4}, {1, 2, 0});
        out.write_slice("g", filled<double>("g", {2, 3, 4}, 900.0), {{0, 2}, {0, 3}, {0, 4}});
        out.write("cur", filled<double>("cur", {4, 3}, 3.0));
    }
    TensorFile rw(path, TensorFile::Mode::ReadWrite);
    rw.remove("gone");
}

struct Operation {
    std::string                              name;
    std::function<void(std::string const &)> run;
    bool                                     power_loss{true}; ///< Whether the power-cut model covers it (not a rename of the file).
    /// The entries a crash must leave as they were or as they become, when the operation commits
    /// more than once and so passes through states of the file between the two. Empty: all of them.
    std::vector<std::string> watched{};
};

std::vector<Operation> operations() {
    return {
        {.name = "write a new entry",
         .run =
             [](std::string const &path) {
                 TensorFile rw(path, TensorFile::Mode::ReadWrite);
                 rw.write("d", filled<double>("d", {30, 20}, 2.0));
             }},
        {.name = "rewrite an entry of the same shape",
         .run =
             [](std::string const &path) {
                 TensorFile rw(path, TensorFile::Mode::ReadWrite);
                 rw.write("a", filled<double>("a", {3, 4}, -40.0));
             }},
        {.name = "rewrite an entry with another shape",
         .run =
             [](std::string const &path) {
                 TensorFile rw(path, TensorFile::Mode::ReadWrite);
                 rw.write("a", filled<double>("a", {6, 2}, -60.0));
             }},
        {.name = "reserve an entry and fill it by slabs",
         .run =
             [](std::string const &path) {
                 TensorFile                rw(path, TensorFile::Mode::ReadWrite);
                 std::vector<double> const half(10, 4.0);
                 rw.reserve<double>("e", {4, 5});
                 rw.write_slab("e", half.data(), {{0, 2}, {0, 5}});
                 rw.write_slab("e", half.data(), {{2, 4}, {0, 5}});
             }},
        {.name = "copy into a new entry, a few bytes at a time",
         .run =
             [](std::string const &path) {
                 TensorFile rw(path, TensorFile::Mode::ReadWrite);
                 rw.copy(rw, "b", "f", 64);
             }},
        {.name = "copy into an entry stored in another order",
         .run =
             [](std::string const &path) {
                 TensorFile rw(path, TensorFile::Mode::ReadWrite);
                 rw.copy(rw, "b", "g", 64);
             }},
        {.name = "rename", .run = [](std::string const &path) { rename_entry(path, "a", "z"); }},
        {.name = "swap", .run = [](std::string const &path) { swap_entries(path, "a", "c"); }},
        {.name = "remove", .run = [](std::string const &path) { remove_entry(path, "c"); }},
        {.name = "replace",
         .run =
             [](std::string const &path) {
                 TensorFile file(path, TensorFile::Mode::ReadWrite);
                 file.replace("a", "c");
             }},
        {.name = "restage", .run = [](std::string const &path) { restage_entry(path, "b", {0, 1, 2}); }},
        {.name = "compact", .run = [](std::string const &path) { compact_file(path); }, .power_loss = false},
        {.name = "replay a backed graph that ends in a swap",
         .run =
             [](std::string const &path) {
                 cg::Graph   graph("step");
                 auto const &cur  = graph.declare_runtime_tensor<double>("cur", {4, 3});
                 auto       &next = graph.declare_runtime_tensor<double>("next", {4, 3});
                 graph.annotate_backing(cur, etn_backing(path, "cur"));
                 graph.annotate_backing(next, etn_backing(path, "next", true));
                 {
                     cg::CaptureGuard const guard(graph);
                     cg::permute("i,j <- i,j", 0.0, &next, 2.0, cur);
                     swap_entries(path, "cur", "next");
                 }
                 graph.optimize();
                 graph.execute();
             },
         .watched = {"cur"}},
        // The scratch entry comes and goes, so only the entry the replay updates is watched: it holds
        // the value the replay started from until the publish, and the next one after it.
        {.name = "replay a graph that updates an entry in place",
         .run =
             [](std::string const &path) {
                 cg::Graph graph("update");
                 auto     &cur = graph.declare_runtime_tensor<double>("cur", etn_backing(path, "cur", false, {}, true));
                 {
                     cg::CaptureGuard const guard(graph);
                     cg::permute("i,j <- i,j", 0.0, &cur, 2.0, cur);
                 }
                 graph.optimize();
                 graph.execute();
             },
         .watched = {"cur"}},
    };
}

/// One call an operation made, kept to replay onto an image of the file.
struct Recorded {
    io_detail::FileCall::Kind kind;
    uint64_t                  offset;
    std::vector<char>         bytes;
};

std::vector<Recorded> record(std::function<void()> const &operation) {
    std::vector<Recorded>          calls;
    file_calls::FileCallHook const hook([&](io_detail::FileCall const &call) {
        auto const *data = static_cast<char const *>(call.data);
        calls.push_back({.kind   = call.kind,
                         .offset = call.offset,
                         .bytes  = data == nullptr ? std::vector<char>{} : std::vector<char>(data, data + call.bytes)});
        return true;
    });
    operation();
    return calls;
}

void apply(std::vector<char> &image, Recorded const &call) {
    if (call.kind == io_detail::FileCall::Kind::Write) {
        if (image.size() < call.offset + call.bytes.size()) {
            image.resize(call.offset + call.bytes.size(), 0);
        }
        std::ranges::copy(call.bytes, image.begin() + static_cast<std::ptrdiff_t>(call.offset));
    } else if (call.kind == io_detail::FileCall::Kind::Truncate) {
        image.resize(call.offset, 0);
    }
}

/// @p state reduced to the entries in @p watched, or whole when that is empty.
file_calls::FileState only(file_calls::FileState state, std::vector<std::string> const &watched) {
    if (watched.empty()) {
        return state;
    }
    std::erase_if(state, [&](auto const &entry) { return std::ranges::find(watched, entry.first) == watched.end(); });
    return state;
}

/// Whether the file at @p path opens and reads as @p before or as @p after, in the entries @p watched.
bool before_or_after(std::string const &path, file_calls::FileState const &before, file_calls::FileState const &after,
                     std::vector<std::string> const &watched = {}) {
    try {
        auto const state = only(file_calls::file_state(path), watched);
        return state == only(before, watched) || state == only(after, watched);
    } catch (std::exception const &) {
        return false;
    }
}

/// Every file a power cut during @p calls can leave, from @p start: for each call it can stop
/// after, everything up to the last sync landed, and any subset of the writes since it. Calls
/// itself with each image, and stops at the first for which @p check returns false.
bool every_power_cut(std::vector<char> const &start, std::vector<Recorded> const &calls,
                     std::function<bool(std::vector<char> const &)> const &check) {
    for (size_t stop = 0; stop <= calls.size(); ++stop) {
        std::vector<char>   durable = start;
        std::vector<size_t> pending;
        for (size_t i = 0; i < stop; ++i) {
            if (calls[i].kind == io_detail::FileCall::Kind::Sync) {
                for (size_t const p : pending) {
                    apply(durable, calls[p]);
                }
                pending.clear();
            } else {
                pending.push_back(i);
            }
        }
        // Every subset when there are few; otherwise those that keep or lose at most two.
        std::vector<std::vector<bool>> subsets;
        size_t const                   n = pending.size();
        if (n <= 12) {
            for (size_t mask = 0; mask < (size_t{1} << n); ++mask) {
                std::vector<bool> keep(n);
                for (size_t i = 0; i < n; ++i) {
                    keep[i] = ((mask >> i) & 1U) != 0;
                }
                subsets.push_back(std::move(keep));
            }
        } else {
            for (bool const lose : {false, true}) {
                for (size_t i = 0; i <= n; ++i) {
                    for (size_t j = i; j <= n; ++j) {
                        std::vector<bool> keep(n, lose);
                        if (i < n) {
                            keep[i] = !lose;
                        }
                        if (j < n) {
                            keep[j] = !lose;
                        }
                        subsets.push_back(std::move(keep));
                    }
                }
            }
        }
        for (auto const &keep : subsets) {
            std::vector<char> image = durable;
            for (size_t i = 0; i < n; ++i) {
                if (keep[i]) {
                    apply(image, calls[pending[i]]);
                }
            }
            if (!check(image)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

TEST_CASE("An operation stopped at any call leaves the file as it was or as it is after it", "[TensorIO][Crash]") {
    TempFile const tmp("killed.etn");
    for (auto const &operation : operations()) {
        DYNAMIC_SECTION(operation.name) {
            make_file(tmp.path);
            auto const   before       = file_calls::file_bytes(tmp.path);
            auto const   before_state = file_calls::file_state(tmp.path);
            auto const   run          = [&] { operation.run(tmp.path); };
            size_t const calls        = file_calls::count_file_calls(run);
            auto const   after_state  = file_calls::file_state(tmp.path);
            REQUIRE(calls > 0);

            for (size_t n = 0; n <= calls; ++n) {
                CAPTURE(n, calls);
                file_calls::put_file_bytes(tmp.path, before);
                std::remove((tmp.path + ".compact").c_str());
                // A killed process goes no further; here the program goes on, its calls dropped,
                // and may find the file as the last commit left it rather than as it expected.
                try {
                    file_calls::run_killed_at(n, run);
                } catch (std::exception const &stopped) {
                    UNSCOPED_INFO("the program went on and stopped with: " << stopped.what());
                }
                CHECK(before_or_after(tmp.path, before_state, after_state, operation.watched));
            }
        }
    }
}

TEST_CASE("An operation cut short by a power loss leaves the file as it was or as it is after it", "[TensorIO][Crash]") {
    TempFile const tmp("power.etn");
    TempFile const image_file("power_image.etn");
    for (auto const &operation : operations()) {
        if (!operation.power_loss) {
            continue;
        }
        DYNAMIC_SECTION(operation.name) {
            make_file(tmp.path);
            auto const before       = file_calls::file_bytes(tmp.path);
            auto const before_state = file_calls::file_state(tmp.path);
            auto const calls        = record([&] { operation.run(tmp.path); });
            auto const after_state  = file_calls::file_state(tmp.path);

            CHECK(every_power_cut(before, calls, [&](std::vector<char> const &image) {
                file_calls::put_file_bytes(image_file.path, image);
                return before_or_after(image_file.path, before_state, after_state, operation.watched);
            }));
        }
    }
}

// The model above would pass a writer that never syncs only if it could not tell; without the syncs
// it must find a power cut that leaves neither state.
TEST_CASE("Without syncs a power loss can leave neither state", "[TensorIO][Crash]") {
    TempFile const tmp("unsynced.etn");
    TempFile const image_file("unsynced_image.etn");
    auto const     saved = config::get(option::IoSync);
    config::set(option::IoSync, false);
    make_file(tmp.path);
    auto const before       = file_calls::file_bytes(tmp.path);
    auto const before_state = file_calls::file_state(tmp.path);
    auto const calls        = record([&] { operations().front().run(tmp.path); });
    auto const after_state  = file_calls::file_state(tmp.path);
    config::set(option::IoSync, saved);

    CHECK_FALSE(every_power_cut(before, calls, [&](std::vector<char> const &image) {
        file_calls::put_file_bytes(image_file.path, image);
        return before_or_after(image_file.path, before_state, after_state);
    }));
}

TEST_CASE("Removing, restaging and compacting keep every other entry", "[TensorIO][Entries]") {
    TempFile const tmp("maintenance.etn");
    make_file(tmp.path);
    auto const before = file_calls::file_state(tmp.path);

    SECTION("remove") {
        remove_entry(tmp.path, "c");
        auto expected = before;
        expected.erase("c");
        CHECK(file_calls::file_state(tmp.path) == expected);
        CHECK_THROWS_WITH(remove_entry(tmp.path, "c"), Catch::Matchers::ContainsSubstring("not found"));
    }

    SECTION("restage") {
        restage_entry(tmp.path, "b", {1, 0, 2});
        CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).stored_order("b") == std::vector<size_t>{1, 0, 2});
        CHECK(file_calls::file_state(tmp.path) == before);
        CHECK_THROWS_AS(restage_entry(tmp.path, "b", {0, 0, 1}), std::invalid_argument);
        CHECK(file_calls::file_state(tmp.path) == before);
    }

    SECTION("compact") {
        auto const unused = TensorFile(tmp.path, TensorFile::Mode::Read).unused_bytes();
        REQUIRE(unused >= sizeof(double) * 25);
        auto const size = std::filesystem::file_size(tmp.path);
        compact_file(tmp.path);
        CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).unused_bytes() < ETN_DATA_ALIGNMENT * 8);
        CHECK(std::filesystem::file_size(tmp.path) <= size - sizeof(double) * 25);
        CHECK(file_calls::file_state(tmp.path) == before);
        CHECK_FALSE(std::filesystem::exists(tmp.path + ".compact"));
    }

    // An empty table used to be committed at the end of the data, which the truncation after the
    // commit then cut off, so the file no longer opened.
    SECTION("removing every entry leaves a file that opens") {
        for (auto const &name : TensorFile(tmp.path, TensorFile::Mode::Read).tensor_names()) {
            remove_entry(tmp.path, name);
        }
        CHECK(file_calls::file_state(tmp.path).empty());
        compact_file(tmp.path);
        CHECK(file_calls::file_state(tmp.path).empty());
        CHECK(std::filesystem::file_size(tmp.path) == sizeof(FileHeader));
    }

    SECTION("a handle on a compacted file refuses to change it") {
        TensorFile stale(tmp.path, TensorFile::Mode::ReadWrite);
        compact_file(tmp.path);
        CHECK_THROWS_WITH(stale.write("x", filled<double>("x", {2}, 1.0)), Catch::Matchers::ContainsSubstring("replaced"));
        CHECK(file_calls::file_state(tmp.path) == before);
    }
}

// A defaulted move left the moved-from handle holding the same descriptor. Once the moved-to handle
// closed it, the next file opened took its number, and the moved-from handle's commit at its own
// close wrote its table and header into that file.
TEST_CASE("A moved handle owns the file and the moved-from one holds nothing", "[TensorIO][Entries]") {
    TempFile const            tmp("moved.etn");
    TempFile const            bystander("bystander.etn");
    std::optional<TensorFile> first;
    first.emplace(tmp.path, TensorFile::Mode::Write);
    first->write("a", filled<double>("a", {2}, 1.0));
    {
        TensorFile second(std::move(*first));
        second.write("b", filled<double>("b", {2}, 2.0));
    }
    {
        TensorFile other(bystander.path, TensorFile::Mode::Write);
        other.write("o", filled<double>("o", {3}, 5.0));
        first.reset();
    }
    CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).tensor_names() == std::vector<std::string>{"a", "b"});
    CHECK(TensorFile(bystander.path, TensorFile::Mode::Read).tensor_names() == std::vector<std::string>{"o"});
}
