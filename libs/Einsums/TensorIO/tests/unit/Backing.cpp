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

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorIO/Backing.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
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

template <typename T>
std::vector<std::byte> bytes_of(RuntimeTensor<T> const &tensor) {
    std::vector<std::byte> out(tensor.size() * sizeof(T));
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
    SECTION("a missing entry or file is refused when the backing is built") {
        CHECK_THROWS_WITH(tensor_io::etn_backing(tmp.path, "nope"), Catch::Matchers::ContainsSubstring("'nope' not found"));
        CHECK_THROWS(tensor_io::etn_backing(tmp.path + ".missing", "B"));
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
