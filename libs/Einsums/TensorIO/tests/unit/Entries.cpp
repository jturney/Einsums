//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Entries.cpp
/// @brief Copying, renaming and swapping the entries of ``.etn`` files, now and inside a graph.
///
/// A rename or swap rewrites only the entry table, and a copy moves the elements through a
/// bounded buffer whatever the two entries' stored orders. Captured, each is a node that runs at
/// every replay in its place among the reads and writes of the entries it names, including those
/// of tensors backed by them, which is what lets one graph read an entry, write the next value
/// beside it and swap the two.

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorIO/Backing.hpp>
#include <Einsums/TensorIO/Entries.hpp>
#include <Einsums/TensorIO/GraphIO.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
using tensor_io::TensorFile;

namespace {

struct TempFile {
    std::string path;
    explicit TempFile(std::string const &name) : path((std::filesystem::temp_directory_path() / ("einsums_entries_" + name)).string()) {
        std::remove(path.c_str());
    }
    ~TempFile() { std::remove(path.c_str()); }
    TempFile(TempFile const &)            = delete;
    TempFile &operator=(TempFile const &) = delete;
};

/// A reproducible filling, distinct for each @p seed; complex elements get an imaginary part.
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

template <typename T>
RuntimeTensor<T> read_entry(std::string const &path, std::string const &name) {
    TensorFile       file(path, TensorFile::Mode::Read);
    RuntimeTensor<T> out(name, file.dims(name));
    file.read(name, out);
    return out;
}

template <typename T>
bool same_elements(RuntimeTensor<T> const &a, RuntimeTensor<T> const &b) {
    return a.dims() == b.dims() && std::equal(a.data(), a.data() + a.size(), b.data());
}

/// Reserve @p name stored in @p order and fill it whole with @p values.
template <typename T>
void write_ordered(TensorFile &file, std::string const &name, RuntimeTensor<T> const &values, std::vector<size_t> const &order) {
    std::vector<std::pair<size_t, size_t>> whole;
    whole.reserve(values.rank());
    for (size_t d = 0; d < values.rank(); ++d) {
        whole.emplace_back(0, values.dim(d));
    }
    file.reserve<T>(name, values.dims(), order);
    file.write_slice(name, values, whole);
}

} // namespace

// ── Rename and swap ──────────────────────────────────────────────────────────

TEST_CASE("Renaming an entry keeps its elements and frees its old name", "[TensorIO][Entries]") {
    TempFile const tmp("rename.etn");
    auto const     a = filled<double>("a", {3, 4}, 1.0);
    auto const     b = filled<double>("b", {2}, 100.0);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("a", a);
        out.write("b", b);
    }
    auto const size_before = std::filesystem::file_size(tmp.path);

    // Only the table is written, beside the committed one, so the file grows by at most a table.
    tensor_io::rename_entry(tmp.path, "a", "c");
    CHECK(std::filesystem::file_size(tmp.path) <=
          size_before + tensor_io::align_up(2 * sizeof(tensor_io::TensorEntry), tensor_io::ETN_DATA_ALIGNMENT));
    CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).tensor_names() == std::vector<std::string>{"c", "b"});
    CHECK(same_elements(read_entry<double>(tmp.path, "c"), a));

    // Renaming an entry to its own name does nothing.
    tensor_io::rename_entry(tmp.path, "c", "c");
    CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).tensor_names() == std::vector<std::string>{"c", "b"});

    SECTION("refusals leave the file as it was") {
        CHECK_THROWS_AS(tensor_io::rename_entry(tmp.path, "a", "d"), std::runtime_error);
        CHECK_THROWS_AS(tensor_io::rename_entry(tmp.path, "c", "b"), std::invalid_argument);
        CHECK_THROWS_AS(tensor_io::rename_entry(tmp.path, "c", std::string(tensor_io::ETN_MAX_NAME, 'n')), std::invalid_argument);
        CHECK_THROWS_AS(tensor_io::rename_entry(tmp.path, "c", ""), std::invalid_argument);
        {
            TensorFile reading(tmp.path, TensorFile::Mode::Read);
            CHECK_THROWS_AS(reading.rename("c", "d"), std::runtime_error);
        }
        CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).tensor_names() == std::vector<std::string>{"c", "b"});
        CHECK(same_elements(read_entry<double>(tmp.path, "b"), b));
    }
}

TEST_CASE("Swapping two entries exchanges what their names reach", "[TensorIO][Entries]") {
    TempFile const tmp("swap.etn");
    auto const     cur  = filled<double>("cur", {3, 4}, 1.0);
    auto const     next = filled<double>("next", {3, 4}, 50.0);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("cur", cur);
        out.write("next", next);
    }

    tensor_io::swap_entries(tmp.path, "cur", "next");
    CHECK(same_elements(read_entry<double>(tmp.path, "cur"), next));
    CHECK(same_elements(read_entry<double>(tmp.path, "next"), cur));

    tensor_io::swap_entries(tmp.path, "next", "cur");
    CHECK(same_elements(read_entry<double>(tmp.path, "cur"), cur));

    // An entry swapped with itself stays, and a missing one is refused before anything changes.
    tensor_io::swap_entries(tmp.path, "cur", "cur");
    CHECK_THROWS_AS(tensor_io::swap_entries(tmp.path, "cur", "gone"), std::runtime_error);
    CHECK(same_elements(read_entry<double>(tmp.path, "cur"), cur));
    CHECK(same_elements(read_entry<double>(tmp.path, "next"), next));
}

TEST_CASE("Replacing an entry gives its name another entry's elements and frees its own", "[TensorIO][Entries]") {
    TempFile const tmp("replace.etn");
    auto const     cur  = filled<double>("cur", {3, 4}, 1.0);
    auto const     next = filled<float>("next", {5}, 50.0);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("cur", cur);
        out.write("next", next);
    }

    {
        TensorFile file(tmp.path, TensorFile::Mode::ReadWrite);
        file.replace("next", "cur");
        // The old elements of "cur" are no longer reached: their space is free.
        CHECK(file.unused_bytes() >= cur.size() * sizeof(double));
    }
    CHECK(same_elements(read_entry<float>(tmp.path, "cur"), next));
    CHECK_FALSE(TensorFile(tmp.path, TensorFile::Mode::Read).contains("next"));

    TensorFile file(tmp.path, TensorFile::Mode::ReadWrite);
    // Onto a name nothing holds, it is a rename; onto itself, nothing.
    file.replace("cur", "other");
    file.replace("other", "other");
    CHECK(file.tensor_names() == std::vector<std::string>{"other"});
    // A missing source, or a name no entry can hold, is refused before anything changes.
    CHECK_THROWS_AS(file.replace("gone", "other"), std::runtime_error);
    CHECK_THROWS_AS(file.replace("other", std::string(tensor_io::ETN_MAX_NAME, 'x')), std::invalid_argument);
    CHECK(file.tensor_names() == std::vector<std::string>{"other"});
    CHECK_THROWS_AS(TensorFile(tmp.path, TensorFile::Mode::Read).replace("other", "x"), std::runtime_error);
}

// ── Copy ─────────────────────────────────────────────────────────────────────

TEMPLATE_LIST_TEST_CASE("A copy moves every element through a buffer smaller than the entry", "[TensorIO][Entries]",
                        testing::AllScalarTypes) {
    using T = TestType;
    TempFile const tmp("copy.etn");
    TempFile const other("copy_other.etn");
    auto const     values = filled<T>("values", {5, 4, 3}, 2.0);

    SECTION("entries stored in the same order") {
        {
            TensorFile file(tmp.path, TensorFile::Mode::Write);
            file.write("src", values);
            // Three elements at a time: the copy runs in many pieces, the last a short one.
            file.copy(file, "src", "dst", 3 * sizeof(T));
        }
        CHECK(same_elements(read_entry<T>(tmp.path, "dst"), values));
        CHECK(same_elements(read_entry<T>(tmp.path, "src"), values));
    }
    SECTION("a new entry takes the source's stored order") {
        std::vector<size_t> const order{2, 0, 1};
        {
            TensorFile file(tmp.path, TensorFile::Mode::Write);
            write_ordered(file, "src", values, order);
            file.copy(file, "src", "dst", 7 * sizeof(T));
        }
        TensorFile const file(tmp.path, TensorFile::Mode::Read);
        CHECK(file.stored_order("dst") == order);
        CHECK(same_elements(read_entry<T>(tmp.path, "dst"), values));
    }
    SECTION("an existing entry keeps its own stored order") {
        // Each budget cuts the boxes differently: one element, part of the fastest stored axis,
        // whole stored axes and a block of the next, and the whole entry.
        for (size_t const elements : {size_t{1}, size_t{2}, size_t{7}, size_t{15}, size_t{1000}}) {
            INFO("buffer of " << elements << " elements");
            {
                TensorFile file(tmp.path, TensorFile::Mode::Write);
                write_ordered(file, "src", values, {2, 0, 1});
                file.reserve<T>("dst", values.dims(), {1, 2, 0});
                file.copy(file, "src", "dst", elements * sizeof(T));
            }
            TensorFile const file(tmp.path, TensorFile::Mode::Read);
            CHECK(file.stored_order("dst") == std::vector<size_t>{1, 2, 0});
            CHECK(same_elements(read_entry<T>(tmp.path, "dst"), values));
        }
    }
    SECTION("between two files, creating the second") {
        {
            TensorFile file(tmp.path, TensorFile::Mode::Write);
            file.write("src", values);
        }
        tensor_io::copy_entry(tmp.path, "src", other.path, "dst");
        CHECK(TensorFile(other.path, TensorFile::Mode::Read).tensor_names() == std::vector<std::string>{"dst"});
        CHECK(same_elements(read_entry<T>(other.path, "dst"), values));
    }
}

TEST_CASE("A copy never changes an entry's shape or type", "[TensorIO][Entries]") {
    TempFile const tmp("copy_refused.etn");
    auto const     values = filled<double>("values", {4, 3}, 2.0);
    auto const     kept   = filled<double>("kept", {3, 4}, 9.0);
    auto const     single = filled<float>("single", {4, 3}, 9.0);
    {
        TensorFile file(tmp.path, TensorFile::Mode::Write);
        file.write("src", values);
        file.write("shape", kept);
        file.write("type", single);
    }
    CHECK_THROWS_AS(tensor_io::copy_entry(tmp.path, "src", tmp.path, "shape"), std::invalid_argument);
    CHECK_THROWS_AS(tensor_io::copy_entry(tmp.path, "src", tmp.path, "type"), std::invalid_argument);
    CHECK_THROWS_AS(tensor_io::copy_entry(tmp.path, "gone", tmp.path, "dst"), std::runtime_error);
    CHECK(same_elements(read_entry<double>(tmp.path, "shape"), kept));
    CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).tensor_names() == std::vector<std::string>{"src", "shape", "type"});

    // An entry copied onto itself stays as it is.
    tensor_io::copy_entry(tmp.path, "src", tmp.path, "src");
    CHECK(same_elements(read_entry<double>(tmp.path, "src"), values));
}

TEST_CASE("A copy of an entry with a zero extent creates an empty entry", "[TensorIO][Entries]") {
    TempFile const tmp("copy_empty.etn");
    {
        TensorFile file(tmp.path, TensorFile::Mode::Write);
        file.write("src", RuntimeTensor<double>("src", std::vector<size_t>{3, 0}));
    }
    tensor_io::copy_entry(tmp.path, "src", tmp.path, "dst");
    CHECK(TensorFile(tmp.path, TensorFile::Mode::Read).dims("dst") == std::vector<size_t>{3, 0});
}

// ── Captured ─────────────────────────────────────────────────────────────────

// One graph per step: it reads the backed "cur", writes the doubled value into the write-backed
// "next", and swaps the two. The swap must follow the store of "next", and the next replay must
// read the entry the swap made "cur".
TEST_CASE("A captured swap makes the next value current at every replay", "[TensorIO][Entries][Graph]") {
    TempFile const tmp("step.etn");
    auto const     start = filled<double>("start", {4, 3}, 1.0);
    {
        TensorFile file(tmp.path, TensorFile::Mode::Write);
        file.write("cur", start);
    }

    cg::Graph   graph("step");
    auto const &cur  = graph.declare_runtime_tensor<double>("cur", {4, 3});
    auto       &next = graph.declare_runtime_tensor<double>("next", {4, 3});
    graph.annotate_backing(cur, tensor_io::etn_backing(tmp.path, "cur"));
    graph.annotate_backing(next, tensor_io::etn_backing(tmp.path, "next", true));
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("i,j <- i,j", 0.0, &next, 2.0, cur);
        tensor_io::swap_entries(tmp.path, "cur", "next");
    }
    graph.optimize();

    for (int step = 0; step < 3; ++step) {
        graph.execute();
    }
    auto       now      = read_entry<double>(tmp.path, "cur");
    auto       previous = read_entry<double>(tmp.path, "next");
    auto const n        = start.size();
    for (size_t i = 0; i < n; ++i) {
        CHECK(now.data()[i] == 8.0 * start.data()[i]);
        CHECK(previous.data()[i] == 4.0 * start.data()[i]);
    }
}

TEST_CASE("A captured copy runs between a write of its source and a read of its destination", "[TensorIO][Entries][Graph]") {
    TempFile const tmp("graph_copy.etn");
    auto           x = filled<double>("x", {6, 5}, 3.0);
    auto           y = RuntimeTensor<double>("y", std::vector<size_t>{6, 5});
    y.zero();

    cg::Graph graph("copy");
    {
        cg::CaptureGuard const guard(graph);
        tensor_io::write_etn(tmp.path, "x", &x);
        tensor_io::copy_entry(tmp.path, "x", tmp.path, "y");
        tensor_io::read_etn(tmp.path, "y", &y);
    }
    // Nothing ran at capture: the file does not exist until the graph writes it.
    CHECK_FALSE(std::filesystem::exists(tmp.path));
    graph.optimize();
    graph.execute();
    CHECK(same_elements(y, x));

    // A second replay copies the new value.
    for (size_t i = 0; i < x.size(); ++i) {
        x.data()[i] = -x.data()[i];
    }
    graph.execute();
    CHECK(same_elements(y, x));
}

// A swap has no inputs and no outputs, so a pass that took it for pure would hoist it out of its
// loop, and run it once rather than at every iteration.
TEST_CASE("A captured swap runs at every iteration of its loop", "[TensorIO][Entries][Graph]") {
    TempFile const tmp("loop.etn");
    auto const     a = filled<double>("a", {3}, 1.0);
    auto const     b = filled<double>("b", {3}, 10.0);
    {
        TensorFile file(tmp.path, TensorFile::Mode::Write);
        file.write("a", a);
        file.write("b", b);
    }

    cg::Graph graph("loop");
    auto     &body = graph.add_loop("iterate", 2, cg::PredExpr::always(true));
    {
        cg::CaptureGuard const guard(body);
        tensor_io::swap_entries(tmp.path, "a", "b");
    }
    graph.optimize();
    graph.execute();

    // Swapped twice: back where they started.
    CHECK(same_elements(read_entry<double>(tmp.path, "a"), a));
    CHECK(same_elements(read_entry<double>(tmp.path, "b"), b));
}

TEST_CASE("A saved graph keeps its entry operations", "[TensorIO][Entries][SaveLoad]") {
    TempFile const tmp("saved.etn");
    auto const     a = filled<double>("a", {3}, 1.0);
    auto const     b = filled<double>("b", {3}, 10.0);
    auto const     m = filled<double>("m", {3, 2}, 20.0);
    {
        TensorFile file(tmp.path, TensorFile::Mode::Write);
        file.write("a", a);
        file.write("b", b);
        file.write("m", m);
    }

    cg::Graph graph("entries");
    {
        cg::CaptureGuard const guard(graph);
        tensor_io::copy_entry(tmp.path, "a", tmp.path, "c");
        tensor_io::rename_entry(tmp.path, "b", "d");
        tensor_io::swap_entries(tmp.path, "c", "d");
        tensor_io::remove_entry(tmp.path, "a");
        tensor_io::restage_entry(tmp.path, "m", {1, 0});
        tensor_io::compact_file(tmp.path);
    }
    auto const saved = cg::save_graph_string(graph);
    REQUIRE(saved.has_value());
    for (char const *kind : {"tensor_io.CopyEntry", "tensor_io.RenameEntry", "tensor_io.SwapEntries", "tensor_io.RemoveEntry",
                             "tensor_io.RestageEntry", "tensor_io.CompactFile"}) {
        CHECK_THAT(*saved, Catch::Matchers::ContainsSubstring(kind));
    }

    auto loaded = cg::load_graph_string(*saved);
    INFO((loaded ? std::string{} : loaded.error().message));
    REQUIRE(loaded.has_value());
    loaded->execute();

    TensorFile const in(tmp.path, TensorFile::Mode::Read);
    CHECK(in.tensor_names() == std::vector<std::string>{"c", "m", "d"});
    CHECK(in.stored_order("m") == std::vector<size_t>{1, 0});
    CHECK(in.unused_bytes() < tensor_io::ETN_DATA_ALIGNMENT * 4);
    CHECK(same_elements(read_entry<double>(tmp.path, "c"), b));
    CHECK(same_elements(read_entry<double>(tmp.path, "d"), a));
    CHECK(same_elements(read_entry<double>(tmp.path, "m"), m));
}
