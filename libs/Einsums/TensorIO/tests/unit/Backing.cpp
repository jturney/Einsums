//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Backing.cpp
/// @brief A ``.etn`` entry as the storage behind a tensor a graph reads but never holds.
///
/// The backed tensor is declared on the graph and never allocated by the caller; every replay
/// fills it from the file. These cases pin what that promises: the replay equals one over the
/// same values held in memory, a file rewritten between replays is read afresh, the annotation
/// refuses every tensor it could not fill correctly, a write to a backed tensor is a verify
/// failure at any depth, and a saved graph keeps the backing and checks the storage at load.
/// A WRITE backing is the other direction: the graph produces the tensor and the entry receives
/// it, whole after its last writer or a slab per block when SlabStreaming produces it there.

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/ComputeGraph/Passes/SlabStreaming.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorBase/Options.hpp>
#include <Einsums/TensorIO/Backing.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
using tensor_io::TensorFile;

namespace {

struct TempFile {
    std::string path;
    explicit TempFile(std::string name)
        : path((std::filesystem::temp_directory_path() / ("einsums_backing_" + std::move(name))).string()) {}
    ~TempFile() { std::remove(path.c_str()); }
    TempFile(TempFile const &)            = delete;
    TempFile &operator=(TempFile const &) = delete;
};

/// A reproducible filling; complex elements get an imaginary part, so a read that dropped or
/// swapped halves of an element would show.
template <typename T>
RuntimeTensor<T> filled(std::string name, std::vector<size_t> dims, double scale) {
    RuntimeTensor<T> out(std::move(name), std::move(dims));
    for (size_t n = 0; n < out.size(); ++n) {
        double const x = scale * (static_cast<double>(n % 17) - 8.0);
        if constexpr (IsComplexV<T>) {
            out.data()[n] = T{static_cast<RemoveComplexT<T>>(x), static_cast<RemoveComplexT<T>>(-0.25 * x)};
        } else {
            out.data()[n] = static_cast<T>(x);
        }
    }
    return out;
}

template <typename T>
void write_entry(std::string const &path, std::string const &name, RuntimeTensor<T> const &tensor) {
    TensorFile file(path, TensorFile::Mode::Write);
    file.write(name, tensor);
}

/// The tensor's bytes, as unsigned char: Catch2's prebuilt library exports no printer for std::byte
/// on Windows, so a failed comparison of std::byte vectors does not link there.
template <typename T>
std::vector<unsigned char> bytes_of(RuntimeTensor<T> const &tensor) {
    std::vector<unsigned char> out(tensor.size() * sizeof(T));
    if (!out.empty()) {
        std::memcpy(out.data(), tensor.data(), out.size());
    }
    return out;
}

/// Materialization alone: enough to make a graph with a deferred tensor executable, and nothing
/// that would rewrite what the structural cases look at.
std::shared_ptr<cg::passes::Materialization> materialize(cg::Graph &graph) {
    auto            pass = std::make_shared<cg::passes::Materialization>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);
    return pass;
}

/// C[i,j] = sum_k A[i,k] B[k,j], with B either the given tensor or, when @p backing is set, a
/// deferred one declared on the graph and backed by it.
template <typename T>
struct Product {
    RuntimeTensor<T>  a;
    RuntimeTensor<T>  c;
    RuntimeTensor<T> *b{nullptr};
    cg::Graph         graph;

    Product(RuntimeTensor<T> const &a_in, std::vector<size_t> const &b_dims, std::optional<cg::TensorBacking> backing,
            RuntimeTensor<T> *resident)
        : a(a_in), c("C", std::vector<size_t>{a.dim(0), b_dims[1]}), graph("product") {
        c.zero();
        if (backing.has_value()) {
            b = &graph.declare_runtime_tensor<T>("B", b_dims);
            graph.annotate_backing(*b, std::move(*backing));
        } else {
            b = resident;
        }
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &c, a, *b);
    }
};

} // namespace

TEMPLATE_LIST_TEST_CASE("A backed tensor replays equal to the same values held in memory", "[TensorIO][Backing]", testing::AllScalarTypes) {
    using T = TestType;
    TempFile const tmp("replay.etn");
    auto           b = filled<T>("B", {4, 3}, 0.5);
    write_entry(tmp.path, "B", b);

    Product<T> resident(filled<T>("A", {5, 4}, 0.25), {4, 3}, std::nullopt, &b);
    resident.graph.execute();

    Product<T> backed(filled<T>("A", {5, 4}, 0.25), {4, 3}, tensor_io::etn_backing(tmp.path, "B"), nullptr);
    auto const pass = materialize(backed.graph);
    CHECK(pass->num_loaded() == 1);
    // Never allocated by the caller, and not by the declaration either: the Initialize fills it.
    CHECK_FALSE(backed.b->is_materialized());
    backed.graph.execute();

    CHECK(bytes_of(backed.c) == bytes_of(resident.c));
    // The values that came off the disk are the ones written, bit for bit.
    CHECK(bytes_of(*backed.b) == bytes_of(b));

    // A second replay reads again and gets the same answer.
    backed.c.zero();
    backed.graph.execute();
    CHECK(bytes_of(backed.c) == bytes_of(resident.c));
}

TEST_CASE("A file rewritten between replays is read afresh", "[TensorIO][Backing]") {
    TempFile const tmp("rewritten.etn");
    write_entry(tmp.path, "B", filled<double>("B", {4, 3}, 0.5));

    Product<double> backed(filled<double>("A", {5, 4}, 0.25), {4, 3}, tensor_io::etn_backing(tmp.path, "B"), nullptr);
    materialize(backed.graph);
    backed.graph.execute();

    auto replacement = filled<double>("B", {4, 3}, -2.0);
    std::remove(tmp.path.c_str());
    write_entry(tmp.path, "B", replacement);
    backed.c.zero();
    backed.graph.execute();

    Product<double> expected(filled<double>("A", {5, 4}, 0.25), {4, 3}, std::nullopt, &replacement);
    expected.graph.execute();
    CHECK(bytes_of(backed.c) == bytes_of(expected.c));

    SECTION("and one rewritten with another shape is refused rather than read") {
        std::remove(tmp.path.c_str());
        write_entry(tmp.path, "B", filled<double>("B", {3, 4}, 1.0));
        CHECK_THROWS_WITH(backed.graph.execute(), Catch::Matchers::ContainsSubstring("is now [3,4], not the [4,3]"));
    }
}

// A backing writes its storage's column-major order, and Python declares tensors row-major. The
// whole load refused any tensor not laid out column-major, so every backed tensor a Python program
// declared failed at its first replay; it is now filled through a scatter by the tensor's strides.
TEST_CASE("A row-major backed tensor is filled in its own layout", "[TensorIO][Backing]") {
    TempFile const tmp("row_major.etn");
    auto const     stored = filled<double>("B", {4, 3}, 0.5);
    write_entry(tmp.path, "B", stored);

    bool const previous = config::get(option::RowMajor);
    config::set(option::RowMajor, true);
    struct Restore {
        bool previous;
        ~Restore() { config::set(option::RowMajor, previous); }
    } const restore{previous};

    cg::Graph graph("row_major");
    auto     &b = graph.declare_runtime_tensor<double>("B", {4, 3});
    REQUIRE(b.impl().is_row_major());
    graph.annotate_backing(b, tensor_io::etn_backing(tmp.path, "B"));
    auto c = RuntimeTensor<double>("C", std::vector<size_t>{3});
    auto v = filled<double>("v", {4}, 1.0);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("j <- k ; k,j", &c, v, b);
    }
    materialize(graph);
    graph.execute();
    for (size_t k = 0; k < 4; ++k) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK(b(k, j) == stored(k, j));
        }
    }
}

TEST_CASE("A backed tensor with a zero extent reads nothing and still runs", "[TensorIO][Backing]") {
    TempFile const tmp("empty.etn");
    write_entry(tmp.path, "B", RuntimeTensor<double>("B", std::vector<size_t>{0, 3}));

    Product<double> backed(filled<double>("A", {5, 0}, 1.0), {0, 3}, tensor_io::etn_backing(tmp.path, "B"), nullptr);
    materialize(backed.graph);
    backed.c.data()[0] = 7.0;
    backed.graph.execute();
    // An empty contraction still applies C's prefactor, which is zero here.
    for (size_t n = 0; n < backed.c.size(); ++n) {
        CHECK(backed.c.data()[n] == 0.0);
    }
}

TEST_CASE("annotate_backing refuses a tensor it could not fill correctly", "[TensorIO][Backing]") {
    TempFile const tmp("refusals.etn");
    write_entry(tmp.path, "B", filled<double>("B", {4, 3}, 0.5));
    auto const backing = tensor_io::etn_backing(tmp.path, "B");
    CHECK(backing.kind == "tensor_io.etn");
    CHECK(backing.describe() == "tensor_io.etn (entry=B, path=" + tmp.path + ")");

    cg::Graph  graph("refusals");
    auto const refused = [&](auto &tensor, std::string const &why) {
        CHECK_THROWS_WITH(graph.annotate_backing(tensor, backing), Catch::Matchers::ContainsSubstring(why));
    };

    SECTION("other extents") {
        refused(graph.declare_runtime_tensor<double>("B", {3, 4}), "the tensor is [3,4] and the storage holds [4,3]");
    }
    SECTION("another element type") {
        refused(graph.declare_runtime_tensor<float>("B", {4, 3}), "the tensor holds float32 and the storage holds float64");
    }
    SECTION("a tensor the caller already holds") {
        RuntimeTensor<double> held("B", std::vector<size_t>{4, 3});
        refused(held, "already allocated");
    }
    SECTION("an intermediate") {
        refused(graph.declare_runtime_tensor<double>("B", {4, 3}, /*intermediate=*/true), "declared as an intermediate");
    }
    SECTION("a tensor filled some other way") {
        refused(graph.declare_zero_runtime_tensor<double>("B", {4, 3}), "already initialized by 'zero'");
    }
    SECTION("a second backing") {
        auto &b = graph.declare_runtime_tensor<double>("B", {4, 3});
        graph.annotate_backing(b, backing);
        refused(b, "already backed by tensor_io.etn");
    }
    SECTION("a refused backing at declaration leaves nothing declared") {
        CHECK_THROWS_WITH(graph.declare_runtime_tensor<double>("B", {3, 4}, false, backing),
                          Catch::Matchers::ContainsSubstring("the tensor is [3,4] and the storage holds [4,3]"));
        CHECK_THROWS_WITH(graph.declare_runtime_tensor<double>("B", {4, 3}, /*intermediate=*/true, backing),
                          Catch::Matchers::ContainsSubstring("declared as an intermediate"));
        CHECK(graph.num_tensors() == 0);
    }
    SECTION("a missing entry or file is refused when the backing is built") {
        CHECK_THROWS_WITH(tensor_io::etn_backing(tmp.path, "nope"), Catch::Matchers::ContainsSubstring("'nope' not found"));
        CHECK_THROWS(tensor_io::etn_backing(tmp.path + ".missing", "B"));
    }
}

TEST_CASE("A backing given at declaration is the same as one annotated after", "[TensorIO][Backing]") {
    TempFile const tmp("declared.etn");
    auto const     stored = filled<double>("B", {4, 3}, 0.5);
    write_entry(tmp.path, "B", stored);

    cg::Graph   graph("declared");
    auto const &b = graph.declare_runtime_tensor<double>("B", {4, 3}, false, tensor_io::etn_backing(tmp.path, "B"));
    REQUIRE(graph.tensor_backing(b).has_value());
    CHECK(graph.tensor_backing(b)->describe() == "tensor_io.etn (entry=B, path=" + tmp.path + ")");

    auto c = RuntimeTensor<double>("C", std::vector<size_t>{3});
    auto v = filled<double>("v", {4}, 1.0);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("j <- k ; k,j", &c, v, b);
    }
    materialize(graph);
    graph.execute();
    for (size_t j = 0; j < 3; ++j) {
        double expected = 0.0;
        for (size_t k = 0; k < 4; ++k) {
            expected += v(k) * stored(k, j);
        }
        CHECK(c(j) == expected);
    }
}

TEST_CASE("A backed tensor keeps its provenance tag beside its backing", "[TensorIO][Backing]") {
    TempFile const tmp("tagged.etn");
    write_entry(tmp.path, "B", filled<double>("B", {4, 3}, 0.5));

    cg::Graph   graph("tagged");
    auto const &b = graph.declare_runtime_tensor<double>("B", {4, 3});
    graph.annotate_tag(b, cg::ProvenanceTag{.name = "eri"});
    graph.annotate_backing(b, tensor_io::etn_backing(tmp.path, "B"));
    CHECK(graph.tensor_tag(b).name == "eri");
    REQUIRE(graph.tensor_backing(b).has_value());
    CHECK(graph.tensor_backing(b)->kind == "tensor_io.etn");
}

TEST_CASE("A write to a backed tensor fails verification, at any depth", "[TensorIO][Backing]") {
    TempFile const tmp("written.etn");
    write_entry(tmp.path, "B", filled<double>("B", {4, 3}, 0.5));
    auto const other = filled<double>("X", {4, 3}, 1.0);

    auto const writes_backed = [](std::vector<std::string> const &problems) {
        return std::ranges::any_of(problems,
                                   [](std::string const &p) { return p.find("a backed tensor is read-only") != std::string::npos; });
    };

    SECTION("in the graph that declared it") {
        cg::Graph graph("writer");
        auto     &b = graph.declare_runtime_tensor<double>("B", {4, 3});
        graph.annotate_backing(b, tensor_io::etn_backing(tmp.path, "B"));
        {
            cg::CaptureGuard const guard(graph);
            cg::axpby(1.0, other, 1.0, &b);
        }
        CHECK(writes_backed(graph.verify()));
    }
    SECTION("inside a loop body, whose handle for it carries no backing") {
        cg::Graph graph("writer");
        auto     &b = graph.declare_runtime_tensor<double>("B", {4, 3});
        graph.annotate_backing(b, tensor_io::etn_backing(tmp.path, "B"));
        auto &body = graph.add_loop("iterate", 2, cg::PredExpr::always(true));
        {
            cg::CaptureGuard const guard(body);
            cg::axpby(1.0, other, 1.0, &b);
        }
        CHECK(writes_backed(graph.verify()));
    }
    SECTION("through a view of it") {
        cg::Graph graph("writer");
        auto     &b = graph.declare_runtime_tensor<double>("B", {4, 3});
        graph.annotate_backing(b, tensor_io::etn_backing(tmp.path, "B"));
        {
            cg::CaptureGuard const guard(graph);
            auto                  &view = cg::view_runtime(b, std::vector<cg::ViewAxis>{cg::ViewAxis::full(), cg::ViewAxis::full()});
            cg::axpby(1.0, other, 1.0, &view);
        }
        CHECK(writes_backed(graph.verify()));
    }
    // A View aliases the tensor and writes nothing. The write rule once counted the View node's
    // output as a write, so verification rejected any graph that merely READ a backed tensor
    // through a view, which is how AxisTiling's block loops read the operands they do not slice.
    SECTION("but not a view of it that is only read") {
        auto      c = RuntimeTensor<double>("C", std::vector<std::size_t>{4, 3});
        cg::Graph graph("reader");
        auto     &b = graph.declare_runtime_tensor<double>("B", {4, 3});
        graph.annotate_backing(b, tensor_io::etn_backing(tmp.path, "B"));
        {
            cg::CaptureGuard const guard(graph);
            auto                  &view = cg::view_runtime(b, std::vector<cg::ViewAxis>{cg::ViewAxis::full(), cg::ViewAxis::full()});
            cg::axpby(1.0, view, 0.0, &c);
        }
        CHECK_FALSE(writes_backed(graph.verify()));
    }
    SECTION("but not the lifecycle that fills it, nor a reader") {
        Product<double> backed(filled<double>("A", {5, 4}, 0.25), {4, 3}, tensor_io::etn_backing(tmp.path, "B"), nullptr);
        materialize(backed.graph);
        CHECK_FALSE(writes_backed(backed.graph.verify()));
    }
}

TEST_CASE("A saved graph keeps the backing and checks the storage at load", "[TensorIO][Backing][SaveLoad]") {
    TempFile const tmp("saved.etn");
    auto           b = filled<double>("B", {4, 3}, 0.5);
    write_entry(tmp.path, "B", b);

    Product<double> const backed(filled<double>("A", {5, 4}, 0.25), {4, 3}, tensor_io::etn_backing(tmp.path, "B"), nullptr);
    auto const            saved = cg::save_graph_string(backed.graph);
    REQUIRE(saved.has_value());
    CHECK_THAT(*saved, Catch::Matchers::ContainsSubstring(R"("backing":)"));
    CHECK_THAT(*saved, Catch::Matchers::ContainsSubstring(R"("kind":"tensor_io.etn")"));

    SECTION("it loads deferred and backed, outside the interface, and replays the same") {
        auto loaded = cg::load_graph_string(*saved);
        INFO((loaded ? std::string{} : loaded.error().message));
        REQUIRE(loaded.has_value());
        auto const manifest = loaded->manifest();
        CHECK(manifest.find("B") == nullptr);
        REQUIRE(manifest.find("A") != nullptr);

        auto a = filled<double>("A", {5, 4}, 0.25);
        auto c = RuntimeTensor<double>("C", std::vector<size_t>{5, 3});
        c.zero();
        loaded->bind("A", a, "C", c);
        materialize(*loaded);
        loaded->execute();

        Product<double> expected(filled<double>("A", {5, 4}, 0.25), {4, 3}, std::nullopt, &b);
        expected.graph.execute();
        CHECK(bytes_of(c) == bytes_of(expected.c));
    }
    SECTION("binding it by name is refused, since it is not part of the interface") {
        auto loaded = cg::load_graph_string(*saved);
        REQUIRE(loaded.has_value());
        auto replacement = filled<double>("B", {4, 3}, 1.0);
        CHECK_THROWS(loaded->bind("B", replacement));
    }
    SECTION("a missing file fails the load, naming the tensor") {
        std::remove(tmp.path.c_str());
        auto const loaded = cg::load_graph_string(*saved);
        REQUIRE_FALSE(loaded.has_value());
        CHECK_THAT(loaded.error().message, Catch::Matchers::ContainsSubstring("tensor 'B'"));
    }
    SECTION("storage of another shape fails the load") {
        std::remove(tmp.path.c_str());
        write_entry(tmp.path, "B", filled<double>("B", {3, 4}, 1.0));
        auto const loaded = cg::load_graph_string(*saved);
        REQUIRE_FALSE(loaded.has_value());
        CHECK_THAT(loaded.error().message, Catch::Matchers::ContainsSubstring("the tensor is [4,3] and the storage holds [3,4]"));
    }
    SECTION("a kind nobody registered fails the load by name") {
        std::string text = *saved;
        auto const  at   = text.find("tensor_io.etn");
        REQUIRE(at != std::string::npos);
        text.replace(at, std::string_view{"tensor_io.etn"}.size(), "nobody.kind");
        auto const loaded = cg::load_graph_string(text);
        REQUIRE_FALSE(loaded.has_value());
        CHECK_THAT(loaded.error().message, Catch::Matchers::ContainsSubstring("no backing kind is registered under 'nobody.kind'"));
    }
}

namespace {

/// T[i,j] = sum_k A[i,k] B[k,j] into a deferred T backed by @p backing, captured on @p graph.
template <typename T>
RuntimeTensor<T> &produce(cg::Graph &graph, cg::TensorBacking backing, RuntimeTensor<T> const &a, RuntimeTensor<T> const &b) {
    auto &t = graph.declare_runtime_tensor<T>("T", {a.dim(0), b.dim(1)});
    graph.annotate_backing(t, std::move(backing));
    cg::CaptureGuard const guard(graph);
    cg::einsum("i,j <- i,k ; k,j", &t, a, b);
    return t;
}

/// The same product, computed eagerly.
template <typename T>
RuntimeTensor<T> product_of(RuntimeTensor<T> const &a, RuntimeTensor<T> const &b) {
    RuntimeTensor<T> out("T", std::vector<size_t>{a.dim(0), b.dim(1)});
    out.zero();
    cg::einsum("i,j <- i,k ; k,j", &out, a, b);
    return out;
}

template <typename T>
RuntimeTensor<T> read_entry(std::string const &path, std::string const &name, std::vector<size_t> dims) {
    RuntimeTensor<T> out(name, std::move(dims));
    TensorFile       file(path, TensorFile::Mode::Read);
    file.read(name, out);
    return out;
}

} // namespace

TEMPLATE_LIST_TEST_CASE("A write-backed tensor is stored to its entry after its last writer", "[TensorIO][Backing]",
                        testing::AllScalarTypes) {
    using T = TestType;
    TempFile const tmp("write.etn");
    auto const     a = filled<T>("A", {5, 4}, 0.25);
    auto const     b = filled<T>("B", {4, 3}, 0.5);

    // The file does not exist yet: annotating the tensor creates it and reserves the entry.
    REQUIRE_FALSE(std::filesystem::exists(tmp.path));
    cg::Graph graph("write");
    auto     &t = produce(graph, tensor_io::etn_backing(tmp.path, "T", true), a, b);
    CHECK(std::filesystem::exists(tmp.path));
    CHECK(graph.verify().empty());

    auto const pass = materialize(graph);
    CHECK(pass->num_stored() == 1);
    CHECK(pass->num_loaded() == 0);
    graph.execute();

    auto const expected = product_of(a, b);
    CHECK(bytes_of(read_entry<T>(tmp.path, "T", {5, 3})) == bytes_of(expected));
    // The tensor stays readable in memory too: the store copies it out, it does not move it.
    CHECK(bytes_of(t) == bytes_of(expected));
}

TEST_CASE("A write backing reuses an entry of the tensor's shape and refuses another", "[TensorIO][Backing]") {
    TempFile const tmp("reuse.etn");
    auto const     a     = filled<double>("A", {5, 4}, 0.25);
    auto const     b     = filled<double>("B", {4, 3}, 0.5);
    auto const     other = filled<double>("X", {2, 2}, 3.0);

    SECTION("an entry of the same shape and type is overwritten, and the file's other entries kept") {
        {
            TensorFile file(tmp.path, TensorFile::Mode::Write);
            file.write("T", filled<double>("T", {5, 3}, -9.0));
            file.write("X", other);
        }
        cg::Graph graph("reuse");
        produce(graph, tensor_io::etn_backing(tmp.path, "T", true), a, b);
        materialize(graph);
        graph.execute();
        CHECK(bytes_of(read_entry<double>(tmp.path, "T", {5, 3})) == bytes_of(product_of(a, b)));
        CHECK(bytes_of(read_entry<double>(tmp.path, "X", {2, 2})) == bytes_of(other));
    }
    SECTION("an entry of another shape is refused when the tensor is annotated") {
        write_entry(tmp.path, "T", filled<double>("T", {3, 5}, 1.0));
        cg::Graph graph("reshaped");
        CHECK_THROWS_WITH(produce(graph, tensor_io::etn_backing(tmp.path, "T", true), a, b),
                          Catch::Matchers::ContainsSubstring("already holds [3,5] of another shape or type than the tensor's [5,3]"));
    }
    SECTION("an entry of another type is refused when the tensor is annotated") {
        write_entry(tmp.path, "T", filled<float>("T", {5, 3}, 1.0));
        cg::Graph graph("retyped");
        CHECK_THROWS_WITH(produce(graph, tensor_io::etn_backing(tmp.path, "T", true), a, b),
                          Catch::Matchers::ContainsSubstring("of another shape or type"));
    }
}

TEST_CASE("SlabStreaming produces a write-backed tensor into its entry a slab at a time", "[TensorIO][Backing]") {
    TempFile const tmp("streamed_write.etn");
    auto const     a = filled<double>("A", {5, 4}, 0.25);
    auto const     b = filled<double>("B", {4, 7}, 0.5);

    cg::Graph graph("streamed_write");
    auto     &t = produce(graph, tensor_io::etn_backing(tmp.path, "T", true), a, b);
    auto      d = RuntimeTensor<double>("D", std::vector<size_t>{5, 2});
    d.zero();
    auto const e = filled<double>("E", {7, 2}, 0.75);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,l <- i,j ; j,l", &d, t, e);
    }

    auto stream = std::make_shared<cg::passes::SlabStreaming>();
    stream->set_slab_budget(static_cast<std::int64_t>(2 * 5 * sizeof(double)));
    cg::PassManager manager;
    manager.add(stream);
    manager.add(std::make_shared<cg::passes::Materialization>());
    graph.apply(manager);
    REQUIRE(stream->num_written() == 1);
    CHECK(stream->num_streamed() == 2);
    graph.execute();
    CHECK_FALSE(t.is_materialized());

    // Each block's columns are computed alone, which is the same arithmetic per element.
    auto const expected = product_of(a, b);
    CHECK(bytes_of(read_entry<double>(tmp.path, "T", {5, 7})) == bytes_of(expected));
    auto want = RuntimeTensor<double>("D", std::vector<size_t>{5, 2});
    want.zero();
    cg::einsum("i,l <- i,j ; j,l", &want, expected, e);
    for (size_t n = 0; n < want.size(); ++n) {
        CHECK_THAT(d.data()[n], Catch::Matchers::WithinRel(want.data()[n], 1e-12) || Catch::Matchers::WithinAbs(want.data()[n], 1e-12));
    }
}

namespace {

/// W[Q,a,b] = H[Q,a,n] C[n,b] with H[Q,a,n] = P[Q,m,n] C[m,a] an intermediate over the cap: AxisTiling
/// loops over an axis of H, and W is written a box at a time through the loop's view. Applies the
/// passes and returns SlabStreaming, to read what it chose.
struct BoxWrite {
    static constexpr std::size_t naux = 6;
    static constexpr std::size_t nbf  = 4;
    static constexpr std::size_t v    = 5;

    RuntimeTensor<double>                      p{filled<double>("P", {naux, nbf, nbf}, 0.25)};
    RuntimeTensor<double>                      c{filled<double>("C", {nbf, v}, 0.5)};
    cg::Graph                                  graph{"box_write"};
    std::shared_ptr<cg::passes::SlabStreaming> stream{std::make_shared<cg::passes::SlabStreaming>()};

    explicit BoxWrite(cg::TensorBacking backing) {
        // Runs cost nothing while the passes choose, so the loop is the one memory traffic picks and
        // it writes W a box at a time; tensors this small would otherwise be written whole.
        std::int64_t const previous = config::get(option::GraphStorageRunBytes);
        config::set(option::GraphStorageRunBytes, std::int64_t{0});
        struct Restore {
            std::int64_t previous;
            ~Restore() { config::set(option::GraphStorageRunBytes, previous); }
        } const restore{previous};
        auto &w = graph.declare_runtime_tensor<double>("W", {naux, v, v});
        graph.annotate_backing(w, std::move(backing));
        auto &half = graph.declare_runtime_tensor<double>("H", std::vector<std::size_t>{naux, v, nbf}, /*intermediate=*/true);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("Q,a,n <- Q,m,n ; m,a", &half, p, c);
            cg::einsum("Q,a,b <- Q,a,n ; n,b", &w, half, c);
        }
        auto tiling = std::make_shared<cg::passes::AxisTiling>();
        tiling->set_memory_cap(static_cast<std::int64_t>(2 * naux * nbf * sizeof(double)));
        stream->set_slab_budget(static_cast<std::int64_t>(naux * v * 2 * sizeof(double)));
        cg::PassManager manager;
        manager.add(tiling);
        manager.add(stream);
        manager.add(std::make_shared<cg::passes::Materialization>());
        graph.apply(manager);
    }

    [[nodiscard]] RuntimeTensor<double> expected() const {
        RuntimeTensor<double> half("Href", std::vector<std::size_t>{naux, v, nbf});
        half.zero();
        cg::einsum("Q,a,n <- Q,m,n ; m,a", &half, p, c);
        RuntimeTensor<double> want("W", std::vector<std::size_t>{naux, v, v});
        want.zero();
        cg::einsum("Q,a,b <- Q,a,n ; n,b", &want, half, c);
        return want;
    }

    [[nodiscard]] bool explains(std::string const &fragment) const {
        auto const lines = stream->explain();
        return std::ranges::any_of(lines, [&](std::string const &line) { return line.find(fragment) != std::string::npos; });
    }
};

void check_entry(std::string const &path, RuntimeTensor<double> const &want) {
    auto const got = read_entry<double>(path, "W", {BoxWrite::naux, BoxWrite::v, BoxWrite::v});
    for (std::size_t n = 0; n < want.size(); ++n) {
        CHECK_THAT(got.data()[n], Catch::Matchers::WithinRel(want.data()[n], 1e-12) || Catch::Matchers::WithinAbs(want.data()[n], 1e-12));
    }
}

} // namespace

TEST_CASE("A new entry is stored in the order that makes the boxes written of it contiguous", "[TensorIO][Backing][Order]") {
    // The loop writes W a block of one axis at a time. Stored as declared, a block of any axis but
    // the last is many runs of the file; the pass stores the entry with the looped axis last, the
    // file records it, and the entry still reads back in its declared order.
    TempFile const tmp("box_order.etn");
    BoxWrite       program(tensor_io::etn_backing(tmp.path, "W", true));
    REQUIRE(program.stream->num_boxes() >= 1);
    auto const order = program.stream->stored_order("W");
    CHECK(order != std::vector<std::size_t>{0, 1, 2});
    CHECK(tensor_io::TensorFile(tmp.path, tensor_io::TensorFile::Mode::Read).stored_order("W") == order);
    CHECK(program.explains("stores 'W' with its axes in the order"));
    CHECK(program.explains("in 1 run(s)"));
    CHECK(program.graph.verify().empty());

    program.graph.execute();
    check_entry(tmp.path, program.expected());
}

TEST_CASE("An entry another run created keeps its order, and the report says what another would save", "[TensorIO][Backing][Order]") {
    TempFile const tmp("box_order_reused.etn");
    {
        tensor_io::TensorFile file(tmp.path, tensor_io::TensorFile::Mode::Write);
        file.reserve<double>("W", {BoxWrite::naux, BoxWrite::v, BoxWrite::v});
    }
    BoxWrite program(tensor_io::etn_backing(tmp.path, "W", true));
    CHECK(program.stream->stored_order("W") == std::vector<std::size_t>{0, 1, 2});
    CHECK(tensor_io::TensorFile(tmp.path, tensor_io::TensorFile::Mode::Read).stored_order("W") == std::vector<std::size_t>{0, 1, 2});
    CHECK(program.explains("is stored with its axes in the order [0,1,2]"));

    program.graph.execute();
    check_entry(tmp.path, program.expected());
}

TEST_CASE("An order the caller gives is the one the entry is stored in", "[TensorIO][Backing][Order]") {
    TempFile const            tmp("box_order_given.etn");
    std::vector<size_t> const given{2, 0, 1};
    BoxWrite                  program(tensor_io::etn_backing(tmp.path, "W", true, given));
    CHECK(tensor_io::TensorFile(tmp.path, tensor_io::TensorFile::Mode::Read).stored_order("W") == given);
    CHECK(program.stream->stored_order("W") == given);
    CHECK_FALSE(program.explains("stores 'W'"));

    program.graph.execute();
    check_entry(tmp.path, program.expected());

    // Reusing the entry asks for the order it holds.
    cg::Graph again("again");
    auto     &w = again.declare_runtime_tensor<double>("W", {BoxWrite::naux, BoxWrite::v, BoxWrite::v});
    CHECK_THROWS_AS(again.annotate_backing(w, tensor_io::etn_backing(tmp.path, "W", true, {0, 1, 2})), std::invalid_argument);
}

TEST_CASE("A saved graph keeps a write backing's access", "[TensorIO][Backing][SaveLoad]") {
    TempFile const tmp("saved_write.etn");
    auto const     a = filled<double>("A", {5, 4}, 0.25);
    auto const     b = filled<double>("B", {4, 3}, 0.5);
    cg::Graph      graph("saved_write");
    produce(graph, tensor_io::etn_backing(tmp.path, "T", true), a, b);
    auto const saved = cg::save_graph_string(graph);
    REQUIRE(saved.has_value());
    CHECK_THAT(*saved, Catch::Matchers::ContainsSubstring(R"("access":"write")"));

    // The entry is reserved at annotation; the loaded graph finds and reuses it, and replays into it.
    auto loaded = cg::load_graph_string(*saved);
    INFO((loaded ? std::string{} : loaded.error().message));
    REQUIRE(loaded.has_value());
    auto a_in = a;
    auto b_in = b;
    loaded->bind("A", a_in, "B", b_in);
    CHECK(loaded->verify().empty());
    materialize(*loaded);
    loaded->execute();
    CHECK(bytes_of(read_entry<double>(tmp.path, "T", {5, 3})) == bytes_of(product_of(a, b)));

    SECTION("and a read backing saves none") {
        TempFile const input("saved_read.etn");
        write_entry(input.path, "B", b);
        Product<double> const backed(a, {4, 3}, tensor_io::etn_backing(input.path, "B"), nullptr);
        auto const            read_saved = cg::save_graph_string(backed.graph);
        REQUIRE(read_saved.has_value());
        CHECK_THAT(*read_saved, !Catch::Matchers::ContainsSubstring(R"("access")"));
    }
}

TEST_CASE("A tensor declared from a read backing takes the storage's extents", "[TensorIO][Backing]") {
    TempFile const tmp("from_backing.etn");
    auto const     stored = filled<double>("B", {4, 3}, 0.5);
    write_entry(tmp.path, "B", stored);

    cg::Graph   graph("from_backing");
    auto const &b = graph.declare_runtime_tensor<double>("B", tensor_io::etn_backing(tmp.path, "B"));
    CHECK(b.rank() == 2);
    CHECK(b.dim(0) == 4);
    CHECK(b.dim(1) == 3);
    REQUIRE(graph.tensor_backing(b).has_value());
    CHECK(graph.tensor_backing(b)->dtype_name() == std::optional<std::string>{"float64"});

    CHECK_THROWS_WITH(graph.declare_runtime_tensor<float>("B32", tensor_io::etn_backing(tmp.path, "B")),
                      Catch::Matchers::ContainsSubstring("the tensor holds float32 and the storage holds float64"));
    CHECK_THROWS_WITH(graph.declare_runtime_tensor<double>("T", tensor_io::etn_backing(tmp.path, "T", true)),
                      Catch::Matchers::ContainsSubstring("a write backing takes its shape from the tensor"));
    CHECK(graph.num_tensors() == 1);
    // A write backing has no element type until its tensor completes it.
    CHECK_FALSE(tensor_io::etn_backing(tmp.path, "T", true).dtype_name().has_value());
}

TEST_CASE("EtnFile builds the backings of its entries", "[TensorIO][Backing]") {
    TempFile const tmp("named_once.etn");
    write_entry(tmp.path, "B", filled<double>("B", {4, 3}, 0.5));

    tensor_io::EtnFile const file(tmp.path);
    CHECK(file.path() == tmp.path);
    auto const read = file.entry("B");
    CHECK(read.describe() == tensor_io::etn_backing(tmp.path, "B").describe());
    CHECK(read.dims == std::vector<size_t>{4, 3});
    CHECK_FALSE(read.write);
    CHECK(file.entry("T", true).write);
    CHECK_THROWS_WITH(file.entry("nope"), Catch::Matchers::ContainsSubstring("'nope' not found"));
}

// Every term of a residual adds into it, so its storage has to start each replay at zero. A write
// backing used to refuse a tensor declared at zero, and a saved graph cleared the zero start of
// every backed tensor at load.
TEST_CASE("A write-backed tensor declared at zero sums its writers at every replay", "[TensorIO][Backing]") {
    TempFile const tmp("summed.etn");
    // Not const: a loaded graph binds them.
    auto a  = filled<double>("A", {5, 4}, 0.25);
    auto b  = filled<double>("B", {4, 3}, 0.5);
    auto b2 = filled<double>("B2", {4, 3}, 0.75);

    RuntimeTensor<double> expected("T", std::vector<size_t>{5, 3});
    expected.zero();
    cg::einsum("i,j <- i,k ; k,j", 1.0, &expected, 1.0, a, b);
    cg::einsum("i,j <- i,k ; k,j", 1.0, &expected, 1.0, a, b2);

    cg::Graph graph("summed");
    auto     &t = graph.declare_zero_runtime_tensor<double>("T", {5, 3}, false, tensor_io::etn_backing(tmp.path, "T", true));
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &t, 1.0, a, b);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &t, 1.0, a, b2);
    }
    CHECK(graph.verify().empty());

    SECTION("replayed twice, the storage holds one sum") {
        materialize(graph);
        graph.execute();
        graph.execute();
        CHECK(bytes_of(read_entry<double>(tmp.path, "T", {5, 3})) == bytes_of(expected));
    }
    SECTION("a saved graph keeps the zero start") {
        auto const saved = cg::save_graph_string(graph);
        REQUIRE(saved.has_value());
        auto loaded = cg::load_graph_string(*saved);
        INFO((loaded ? std::string{} : loaded.error().message));
        REQUIRE(loaded.has_value());
        bool found = false;
        for (auto const &[id, handle] : loaded->tensors_map()) {
            if (handle.name == "T") {
                found = true;
                CHECK(handle.init_kind == cg::InitKind::Zero);
                REQUIRE(handle.backing != nullptr);
                CHECK(handle.backing->write);
            }
        }
        CHECK(found);
        loaded->bind("A", a, "B", b, "B2", b2);
        materialize(*loaded);
        loaded->execute();
        loaded->execute();
        CHECK(bytes_of(read_entry<double>(tmp.path, "T", {5, 3})) == bytes_of(expected));
    }
    SECTION("a read backing is still refused at zero") {
        CHECK_THROWS_WITH(graph.declare_zero_runtime_tensor<double>("R", {5, 3}, false, tensor_io::etn_backing(tmp.path, "T")),
                          Catch::Matchers::ContainsSubstring("already initialized by 'zero'"));
    }
}

namespace {

/// The entries of the file at @p path, sorted.
std::vector<std::string> entries_of(std::string const &path) {
    TensorFile const file(path, TensorFile::Mode::Read);
    auto             names = file.tensor_names();
    std::ranges::sort(names);
    return names;
}

/// t <- 0.5 * (A t), the step each replay of the update graphs below takes, computed eagerly.
RuntimeTensor<double> step_of(RuntimeTensor<double> const &a, RuntimeTensor<double> const &t) {
    RuntimeTensor<double> product("p", std::vector<size_t>{t.dim(0), t.dim(1)});
    product.zero();
    cg::einsum("i,j <- i,k ; k,j", 1.0, &product, 1.0, a, t);
    RuntimeTensor<double> out("t", std::vector<size_t>{t.dim(0), t.dim(1)});
    cg::permute("i,j <- i,j", 0.0, &out, 0.5, product);
    return out;
}

} // namespace

TEST_CASE("An update-backed tensor starts each replay from its entry and leaves its next value there", "[TensorIO][Backing]") {
    TempFile const tmp("update.etn");
    auto const     t0 = filled<double>("t", {4, 3}, 0.5);
    auto const     a  = filled<double>("A", {4, 4}, 0.125);
    write_entry(tmp.path, "t", t0);

    cg::Graph graph("update");
    auto     &t = graph.declare_runtime_tensor<double>("t", tensor_io::EtnFile(tmp.path).entry("t", false, {}, true));
    auto     &r = graph.declare_zero_runtime_tensor<double>("r", {4, 3});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &r, 1.0, a, t); // reads the value the replay started from
        cg::permute("i,j <- i,j", 0.0, &t, 0.5, r);         // writes the next one
    }
    CHECK(graph.verify().empty());
    auto const pass = materialize(graph);
    CHECK(pass->num_loaded() == 1);
    CHECK(pass->num_stored() == 1);
    CHECK(pass->num_published() == 1);

    RuntimeTensor<double> expected = t0;
    for (int replay = 0; replay < 3; ++replay) {
        graph.execute();
        expected = step_of(a, expected);
        CHECK(bytes_of(read_entry<double>(tmp.path, "t", {4, 3})) == bytes_of(expected));
        // The scratch entry became the entry: nothing is left beside it.
        CHECK(entries_of(tmp.path) == std::vector<std::string>{"t"});
    }
}

TEST_CASE("An update-backed tensor whose first use overwrites it is not loaded", "[TensorIO][Backing]") {
    TempFile const tmp("update_overwrite.etn");
    write_entry(tmp.path, "t", filled<double>("t", {4, 3}, 0.5));
    auto const x = filled<double>("x", {4, 3}, 2.0);

    cg::Graph graph("overwrite");
    auto     &t = graph.declare_runtime_tensor<double>("t", tensor_io::etn_backing(tmp.path, "t", false, {}, true));
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("i,j <- i,j", 0.0, &t, 1.0, x);
    }
    auto const pass = materialize(graph);
    CHECK(pass->num_loaded() == 0);
    graph.execute();
    CHECK(bytes_of(read_entry<double>(tmp.path, "t", {4, 3})) == bytes_of(x));
}

TEST_CASE("SlabStreaming reads an update-backed tensor from its entry only before its first write", "[TensorIO][Backing]") {
    TempFile const tmp("update_streamed.etn");
    auto const     t0 = filled<double>("t", {4, 16}, 0.5);
    auto const     a  = filled<double>("A", {4, 4}, 0.125);
    write_entry(tmp.path, "t", t0);

    cg::Graph graph("update_streamed");
    auto     &t     = graph.declare_runtime_tensor<double>("t", tensor_io::etn_backing(tmp.path, "t", false, {}, true));
    auto     &r     = graph.declare_zero_runtime_tensor<double>("r", {4, 16});
    auto      after = RuntimeTensor<double>("after", std::vector<size_t>{4, 16});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &r, 1.0, a, t);     // before the write: may stream from the entry
        cg::permute("i,j <- i,j", 0.0, &t, 0.5, r);             // the write
        cg::einsum("i,j <- i,k ; k,j", 0.0, &after, 1.0, a, t); // after it: reads the new value, from memory
    }

    auto stream = std::make_shared<cg::passes::SlabStreaming>();
    stream->set_slab_budget(static_cast<std::int64_t>(4 * 4 * sizeof(double)));
    cg::PassManager manager;
    manager.add(stream);
    manager.add(std::make_shared<cg::passes::Materialization>());
    graph.apply(manager);
    CHECK(stream->num_streamed() == 1);
    auto const decided = [&](cg::Reason const &reason) {
        std::vector<cg::Decision> out;
        std::ranges::copy_if(stream->decisions(), std::back_inserter(out), [&](cg::Decision const &d) { return d.code == reason.code; });
        return out;
    };
    CHECK(decided(cg::passes::SlabStreaming::ReadAfterUpdate).size() == 1);
    // The one read before the write is streamed, from the entry, in blocks of four columns.
    auto const streamed = decided(cg::passes::SlabStreaming::StreamedRead);
    REQUIRE(streamed.size() == 1);
    CHECK(streamed.at(0).tensors == std::vector<std::string>{"t"});
    CHECK(streamed.at(0).fact("block") == 4);
    CHECK(streamed.at(0).fact("extent") == 16);

    RuntimeTensor<double> expected = t0;
    for (int replay = 0; replay < 2; ++replay) {
        graph.execute();
        expected = step_of(a, expected);
        CHECK(bytes_of(read_entry<double>(tmp.path, "t", {4, 16})) == bytes_of(expected));
        auto want = RuntimeTensor<double>("want", std::vector<size_t>{4, 16});
        want.zero();
        cg::einsum("i,j <- i,k ; k,j", 1.0, &want, 1.0, a, expected);
        CHECK(bytes_of(after) == bytes_of(want));
    }
}

TEST_CASE("A saved graph keeps an update backing", "[TensorIO][Backing][SaveLoad]") {
    TempFile const tmp("update_saved.etn");
    auto const     t0 = filled<double>("t", {4, 3}, 0.5);
    auto           a  = filled<double>("A", {4, 4}, 0.125);
    write_entry(tmp.path, "t", t0);

    cg::Graph graph("update_saved");
    auto     &t = graph.declare_runtime_tensor<double>("t", tensor_io::etn_backing(tmp.path, "t", false, {}, true));
    auto     &r = graph.declare_zero_runtime_tensor<double>("r", {4, 3});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &r, 1.0, a, t);
        cg::permute("i,j <- i,j", 0.0, &t, 0.5, r);
    }
    auto const saved = cg::save_graph_string(graph);
    REQUIRE(saved.has_value());
    CHECK_THAT(*saved, Catch::Matchers::ContainsSubstring(R"("next":"t.next")"));

    auto loaded = cg::load_graph_string(*saved);
    INFO((loaded ? std::string{} : loaded.error().message));
    REQUIRE(loaded.has_value());
    loaded->bind("A", a);
    auto const pass = materialize(*loaded);
    CHECK(pass->num_loaded() == 1);
    CHECK(pass->num_stored() == 1);
    CHECK(pass->num_published() == 1);
    RuntimeTensor<double> expected = t0;
    for (int replay = 0; replay < 2; ++replay) {
        INFO("replay " << replay);
        loaded->execute();
        expected = step_of(a, expected);
        CHECK(bytes_of(read_entry<double>(tmp.path, "t", {4, 3})) == bytes_of(expected));
    }
}

TEST_CASE("An update backing refuses what it cannot honour", "[TensorIO][Backing]") {
    TempFile const tmp("update_refusals.etn");
    write_entry(tmp.path, "t", filled<double>("t", {4, 3}, 0.5));

    CHECK_THROWS_WITH(tensor_io::etn_backing(tmp.path, "t", true, {}, true),
                      Catch::Matchers::ContainsSubstring("takes neither write nor order"));
    CHECK_THROWS_WITH(tensor_io::etn_backing(tmp.path, "t", false, {1, 0}, true),
                      Catch::Matchers::ContainsSubstring("takes neither write nor order"));
    CHECK_THROWS_WITH(tensor_io::etn_backing(tmp.path, "missing", false, {}, true),
                      Catch::Matchers::ContainsSubstring("'missing' not found"));
    // The scratch entry's name has to fit an entry: ".next" on a name already at the limit does not.
    std::string const longest(tensor_io::ETN_MAX_NAME - 1, 'x');
    {
        TensorFile file(tmp.path, TensorFile::Mode::ReadWrite);
        file.write(longest, filled<double>("x", {2}, 1.0));
    }
    CHECK_THROWS(tensor_io::etn_backing(tmp.path, longest, false, {}, true));

    auto const update = tensor_io::etn_backing(tmp.path, "t", false, {}, true);
    CHECK(update.write);
    CHECK(update.update);
    CHECK(update.dims == std::vector<size_t>{4, 3});
    cg::Graph graph("refusals");
    CHECK_THROWS_WITH(graph.declare_zero_runtime_tensor<double>("z", {4, 3}, false, update),
                      Catch::Matchers::ContainsSubstring("already initialized by 'zero'"));
    CHECK_THROWS_WITH(graph.declare_runtime_tensor<double>("w", {3, 4}, false, update),
                      Catch::Matchers::ContainsSubstring("the tensor is [3,4] and the storage holds [4,3]"));
    CHECK(graph.num_tensors() == 0);
    // Nothing was reserved for the refused declarations.
    CHECK(entries_of(tmp.path) == std::vector<std::string>{"t", longest});

    // A graph that never writes it would publish nothing it computed, as for any write backing.
    auto const &u = graph.declare_runtime_tensor<double>("u", update);
    (void)u;
    auto const problems = graph.verify();
    REQUIRE(problems.size() == 1);
    CHECK_THAT(problems[0], Catch::Matchers::ContainsSubstring("no node writes it"));
}

TEST_CASE("The update backings of one file are published by one node", "[TensorIO][Backing]") {
    TempFile const tmp("update_two.etn");
    auto const     t0 = filled<double>("t", {4, 3}, 0.5);
    auto const     u0 = filled<double>("u", {2, 2}, 0.25);
    auto const     a  = filled<double>("A", {4, 4}, 0.125);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("t", t0);
        out.write("u", u0);
    }

    tensor_io::EtnFile const file(tmp.path);
    cg::Graph                graph("update_two");
    auto                    &t = graph.declare_runtime_tensor<double>("t", file.entry("t", false, {}, true));
    auto                    &u = graph.declare_runtime_tensor<double>("u", file.entry("u", false, {}, true));
    auto                    &r = graph.declare_zero_runtime_tensor<double>("r", {4, 3});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &r, 1.0, a, t);
        cg::permute("i,j <- i,j", 0.0, &t, 0.5, r);
        cg::permute("i,j <- i,j", 0.0, &u, 2.0, u);
    }
    CHECK(graph.tensor_backing(t)->publish_scope == graph.tensor_backing(u)->publish_scope);
    auto const pass = materialize(graph);
    CHECK(pass->num_published() == 2);
    auto const publishes =
        std::ranges::count_if(graph.nodes(), [](cg::Node const &node) { return node.label.starts_with("publish_backing("); });
    CHECK(publishes == 1);
    CHECK(graph.nodes().back().label == "publish_backing(t, u)");

    RuntimeTensor<double> expected_t = t0;
    RuntimeTensor<double> expected_u = u0;
    for (int replay = 0; replay < 2; ++replay) {
        graph.execute();
        expected_t = step_of(a, expected_t);
        RuntimeTensor<double> doubled("u", std::vector<size_t>{2, 2});
        cg::permute("i,j <- i,j", 0.0, &doubled, 2.0, expected_u);
        expected_u = doubled;
        CHECK(bytes_of(read_entry<double>(tmp.path, "t", {4, 3})) == bytes_of(expected_t));
        CHECK(bytes_of(read_entry<double>(tmp.path, "u", {2, 2})) == bytes_of(expected_u));
        CHECK(entries_of(tmp.path) == std::vector<std::string>{"t", "u"});
    }
}

// Two backed operands used to make SlabStreaming load both whole, even when one fits its share and is
// loaded whole anyway: a small backed tensor beside a large one cost the large one's streaming.
TEST_CASE("A contraction streams its large backed operand beside a small backed one held in memory", "[TensorIO][Backing]") {
    TempFile const tmp("two_backed.etn");
    auto const     small = filled<double>("S", {4, 4}, 0.25);
    auto const     large = filled<double>("L", {4, 16}, 0.5);
    {
        TensorFile out(tmp.path, TensorFile::Mode::Write);
        out.write("S", small);
        out.write("L", large);
    }

    tensor_io::EtnFile const file(tmp.path);
    cg::Graph                graph("two_backed");
    auto const              &s = graph.declare_runtime_tensor<double>("S", file.entry("S"));
    auto const              &l = graph.declare_runtime_tensor<double>("L", file.entry("L"));
    auto                     c = RuntimeTensor<double>("C", std::vector<size_t>{4, 16});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 0.0, &c, 1.0, s, l);
    }

    auto stream = std::make_shared<cg::passes::SlabStreaming>();
    stream->set_slab_budget(static_cast<std::int64_t>(4 * 4 * sizeof(double)));
    cg::PassManager manager;
    manager.add(stream);
    manager.add(std::make_shared<cg::passes::Materialization>());
    graph.apply(manager);
    CHECK(stream->num_streamed() == 1);
    CHECK(stream->streamed() == std::vector<std::string>{"L"});
    graph.execute();

    auto expected = RuntimeTensor<double>("E", std::vector<size_t>{4, 16});
    expected.zero();
    cg::einsum("i,j <- i,k ; k,j", 0.0, &expected, 1.0, small, large);
    CHECK(bytes_of(c) == bytes_of(expected));
}

namespace {

/// The run record of @p graph's backed tensor @p name.
cg::BackingRun const &backing_run(cg::Graph const &graph, std::string const &name) {
    auto const &backings = graph.last_run().backings;
    auto const  hit      = std::ranges::find_if(backings, [&](cg::BackingRun const &run) { return run.tensor == name; });
    REQUIRE(hit != backings.end());
    return *hit;
}

} // namespace

TEST_CASE("last_run counts every read and write of a backing, however a pass moved it", "[TensorIO][Backing][RunMetrics]") {
    SECTION("a whole load") {
        TempFile const tmp("count_load.etn");
        write_entry(tmp.path, "B", filled<double>("B", {4, 3}, 0.5));
        Product<double> backed(filled<double>("A", {5, 4}, 1.0), {4, 3}, tensor_io::etn_backing(tmp.path, "B"), nullptr);
        materialize(backed.graph);
        backed.graph.execute();
        auto const &run = backing_run(backed.graph, "B");
        CHECK(run.reads == 1);
        CHECK(run.bytes_read == 4 * 3 * sizeof(double));
        CHECK(run.writes == 0);
        CHECK(run.storage == "tensor_io.etn (entry=B, path=" + tmp.path + ")");
    }
    SECTION("slabs a contraction streams, beside a whole load") {
        TempFile const tmp("count_stream.etn");
        {
            TensorFile out(tmp.path, TensorFile::Mode::Write);
            out.write("S", filled<double>("S", {4, 4}, 0.25));
            out.write("L", filled<double>("L", {4, 16}, 0.5));
        }
        tensor_io::EtnFile const file(tmp.path);
        cg::Graph                graph("count_stream");
        auto const              &s = graph.declare_runtime_tensor<double>("S", file.entry("S"));
        auto const              &l = graph.declare_runtime_tensor<double>("L", file.entry("L"));
        auto                     c = RuntimeTensor<double>("C", std::vector<size_t>{4, 16});
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,k ; k,j", 0.0, &c, 1.0, s, l);
        }
        auto stream = std::make_shared<cg::passes::SlabStreaming>();
        stream->set_slab_budget(static_cast<std::int64_t>(4 * 4 * sizeof(double)));
        cg::PassManager manager;
        manager.add(stream);
        manager.add(std::make_shared<cg::passes::Materialization>());
        graph.apply(manager);
        REQUIRE(stream->blocks() == std::vector<std::int64_t>{4});
        graph.execute();
        // L in four slabs of four columns; S, which fits, once whole.
        CHECK(backing_run(graph, "L").reads == 4);
        CHECK(backing_run(graph, "L").bytes_read == 4 * 16 * sizeof(double));
        CHECK(backing_run(graph, "S").reads == 1);
        CHECK(backing_run(graph, "S").bytes_read == 4 * 4 * sizeof(double));
    }
    SECTION("a whole store after the last writer") {
        TempFile const tmp("count_store.etn");
        cg::Graph      graph("count_store");
        produce(graph, tensor_io::etn_backing(tmp.path, "T", true), filled<double>("A", {5, 4}, 0.25), filled<double>("B", {4, 3}, 0.5));
        materialize(graph);
        graph.execute();
        auto const &run = backing_run(graph, "T");
        CHECK(run.writes == 1);
        CHECK(run.bytes_written == 5 * 3 * sizeof(double));
        CHECK(run.reads == 0);
    }
    SECTION("slabs a streamed producer writes, and the slabs its reader reads back") {
        TempFile const tmp("count_slabs.etn");
        auto const     a = filled<double>("A", {5, 4}, 0.25);
        auto const     b = filled<double>("B", {4, 7}, 0.5);
        cg::Graph      graph("count_slabs");
        auto          &t = produce(graph, tensor_io::etn_backing(tmp.path, "T", true), a, b);
        auto           d = RuntimeTensor<double>("D", std::vector<size_t>{5, 2});
        d.zero();
        auto const e = filled<double>("E", {7, 2}, 0.75);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,l <- i,j ; j,l", &d, t, e);
        }
        auto stream = std::make_shared<cg::passes::SlabStreaming>();
        stream->set_slab_budget(static_cast<std::int64_t>(2 * 5 * sizeof(double)));
        cg::PassManager manager;
        manager.add(stream);
        manager.add(std::make_shared<cg::passes::Materialization>());
        graph.apply(manager);
        REQUIRE(stream->num_written() == 1);
        graph.execute();
        // Seven columns in blocks of two: 2, 2, 2 and 1, written, then read back the same way.
        auto const &run = backing_run(graph, "T");
        CHECK(run.writes == 4);
        CHECK(run.bytes_written == 5 * 7 * sizeof(double));
        CHECK(run.reads == 4);
        CHECK(run.bytes_read == 5 * 7 * sizeof(double));
    }
    SECTION("boxes a loop's view names, each element written once") {
        TempFile const tmp("count_boxes.etn");
        BoxWrite       program(tensor_io::etn_backing(tmp.path, "W", true));
        REQUIRE(program.stream->num_boxes() >= 1);
        program.graph.execute();
        auto const &run = backing_run(program.graph, "W");
        CHECK(run.writes > 1);
        CHECK(run.bytes_written == BoxWrite::naux * BoxWrite::v * BoxWrite::v * sizeof(double));
    }
    SECTION("an update: one load, one store and one publish per replay") {
        TempFile const tmp("count_update.etn");
        write_entry(tmp.path, "t", filled<double>("t", {4, 3}, 0.5));
        auto const a = filled<double>("A", {4, 4}, 0.125);
        cg::Graph  graph("count_update");
        auto      &t = graph.declare_runtime_tensor<double>("t", tensor_io::etn_backing(tmp.path, "t", false, {}, true));
        auto      &r = graph.declare_zero_runtime_tensor<double>("r", {4, 3});
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,k ; k,j", 1.0, &r, 1.0, a, t);
            cg::permute("i,j <- i,j", 0.0, &t, 0.5, r);
        }
        materialize(graph);
        for (int replay = 0; replay < 2; ++replay) {
            graph.execute();
            auto const &run = backing_run(graph, "t");
            CHECK(run.reads == 1);
            CHECK(run.writes == 1);
            CHECK(run.publishes == 1);
            CHECK(run.bytes_read == 4 * 3 * sizeof(double));
        }
    }
}

// A contraction that writes an update-backed tensor is never streamed: the storage holds the value the
// replay started from until the publish, and later readers want the new one in memory.
TEST_CASE("SlabStreaming declines to stream the contraction that writes an update-backed tensor", "[TensorIO][Backing]") {
    TempFile const tmp("update_writer.etn");
    write_entry(tmp.path, "t", filled<double>("t", {4, 16}, 0.5));
    auto const a = filled<double>("A", {4, 4}, 0.125);
    auto const r = filled<double>("R", {4, 16}, 0.25);

    cg::Graph graph("update_writer");
    auto     &t = graph.declare_runtime_tensor<double>("t", tensor_io::etn_backing(tmp.path, "t", false, {}, true));
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", 0.0, &t, 1.0, a, r);
    }
    auto stream = std::make_shared<cg::passes::SlabStreaming>();
    stream->set_slab_budget(static_cast<std::int64_t>(4 * 4 * sizeof(double)));
    cg::PassManager manager;
    manager.add(stream);
    graph.apply(manager);
    CHECK(stream->num_written() == 0);
    auto const decisions = stream->decisions();
    CHECK(std::ranges::count_if(decisions, [](cg::Decision const &d) {
              return d.code == cg::passes::SlabStreaming::WriteUpdateBacked.code && d.kind == cg::DecisionKind::Declined;
          }) == 1);
}
