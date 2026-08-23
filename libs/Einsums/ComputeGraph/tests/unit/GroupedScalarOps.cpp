//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GroupedScalarOps.cpp
/// @brief Tests for `cg::grouped_dot` and `cg::grouped_axpby`, the nodes that
///        run a whole run of scalar-sized operations one after another.
///
/// The load-bearing property of both is not that they agree with the loop of
/// single calls to within a tolerance, it is that they agree with it BIT FOR
/// BIT, entry by entry. They exist for workloads whose gate is bit-identity, so
/// every check below compares raw element equality against a loop of the single
/// operation rather than a tolerance. The rest is what a merged node has to
/// survive anyway: capture and replay, a rebind, and the dependency edges that
/// keep it after whoever produced its operands.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// One entry of a grouped dot: two operands of one shape and a scalar to
/// receive the reduction.
template <typename T>
struct DotEntry {
    Tensor<T, 2> a, b;
    Tensor<T, 1> r;
};

template <typename T>
DotEntry<T> make_dot_entry(size_t m, size_t n, int seed) {
    return DotEntry<T>{.a = create_random_tensor<T>(fmt::format("A{}", seed), m, n),
                       .b = create_random_tensor<T>(fmt::format("B{}", seed), m, n),
                       .r = create_zero_tensor<T>(fmt::format("r{}", seed), 1)};
}

/// A run whose entries deliberately do not agree on shape, one of them a
/// singleton and one of them empty. The whole point of a grouped form is that
/// nothing has to agree across entries.
template <typename T>
std::vector<DotEntry<T>> mixed_dot_run() {
    std::vector<DotEntry<T>> run;
    run.push_back(make_dot_entry<T>(4, 5, 0));
    run.push_back(make_dot_entry<T>(7, 1, 1));
    run.push_back(make_dot_entry<T>(4, 5, 2));
    run.push_back(make_dot_entry<T>(1, 1, 3));
    run.push_back(make_dot_entry<T>(0, 3, 4));
    run.push_back(make_dot_entry<T>(11, 9, 5));
    return run;
}

template <typename T>
struct DotLists {
    std::vector<Tensor<T, 1> *>       r;
    std::vector<Tensor<T, 2> const *> a, b;
};

template <typename T>
DotLists<T> dot_lists(std::vector<DotEntry<T>> &run) {
    DotLists<T> l;
    for (auto &e : run) {
        l.r.push_back(&e.r);
        l.a.push_back(&e.a);
        l.b.push_back(&e.b);
    }
    return l;
}

/// The oracle: every entry through the single `cg::dot` writer, one at a time.
template <typename T>
std::vector<T> single_dots(std::vector<DotEntry<T>> const &run) {
    std::vector<T> out;
    out.reserve(run.size());
    for (auto const &e : run) {
        Tensor<T, 1> r = create_zero_tensor<T>("ref", 1);
        // `dot_python` is the tensor-destination single form the grouped op
        // mirrors; the same entry point Python's `linalg.dot` binds to.
        cg::dot_python(&r, e.a, e.b);
        out.push_back(r.data()[0]);
    }
    return out;
}

/// One entry of a grouped axpby.
template <typename T>
struct AxpbyEntry {
    Tensor<T, 2> x, y;
};

template <typename T>
AxpbyEntry<T> make_axpby_entry(size_t m, size_t n, int seed) {
    return AxpbyEntry<T>{.x = create_random_tensor<T>(fmt::format("X{}", seed), m, n),
                         .y = create_random_tensor<T>(fmt::format("Y{}", seed), m, n)};
}

} // namespace

TEMPLATE_TEST_CASE("grouped_dot: eager is bit-for-bit the loop of single dots", "[ComputeGraph][GroupedScalarOps]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    auto run = mixed_dot_run<T>();
    auto ref = single_dots<T>(run);

    auto l = dot_lists(run);
    cg::grouped_dot(l.r, l.a, l.b);

    for (size_t i = 0; i < run.size(); i++) {
        INFO("entry " << i);
        REQUIRE(run[i].r.data()[0] == ref[i]);
    }
}

TEST_CASE("grouped_dot: capture and replay are bit-for-bit the loop of single dots", "[ComputeGraph][GroupedScalarOps]") {
    auto run = mixed_dot_run<double>();
    auto ref = single_dots<double>(run);

    auto      l = dot_lists(run);
    cg::Graph graph("grouped_dot");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_dot(l.r, l.a, l.b);
    }
    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::GroupedDot);
    graph.execute();

    for (size_t i = 0; i < run.size(); i++) {
        INFO("entry " << i);
        REQUIRE(run[i].r.data()[0] == ref[i]);
    }
}

TEST_CASE("grouped_dot: the node declares every read and every write", "[ComputeGraph][GroupedScalarOps]") {
    // The hazard scan and verify_level_independence both key off this, and a
    // merged node that under-declared would schedule ahead of its producer.
    auto run = mixed_dot_run<double>();
    auto l   = dot_lists(run);

    cg::Graph graph("grouped_dot_io");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_dot(l.r, l.a, l.b);
    }

    REQUIRE(graph.nodes()[0].inputs.size() == 2 * run.size());
    REQUIRE(graph.nodes()[0].outputs.size() == run.size());

    auto const *d = std::get_if<cg::GroupedDotDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(d != nullptr);
    REQUIRE(d->total == static_cast<int>(run.size()));
}

#ifdef _OPENMP
TEST_CASE("grouped_dot: the same operands give the same bits at every width and under every executor", "[ComputeGraph][GroupedScalarOps]") {
    // The property the operation promises and the reason it fences the vendor:
    // a threaded dot sums per-thread partials, so an unfenced reduction's last
    // bits are a function of the ambient thread count as well as of the
    // operands. The entries here are past the width at which the vendor starts
    // threading a dot, which is the only size where the difference is visible
    // at all - a run of small entries would agree whatever the node did.
    //
    // Widths AND executors, because they are two different ways to reach the
    // node at a width it did not choose: an ambient count set by the caller,
    // and a worker thread the executor pinned.
    int const available = omp_get_max_threads();
    if (available < 2) {
        SUCCEED("one thread available, nothing to compare against");
        return;
    }

    constexpr size_t m = 200, n = 120; // 24,000 elements an entry

    std::vector<DotEntry<double>> run;
    for (int seed = 0; seed < 4; seed++) {
        run.push_back(make_dot_entry<double>(m, n, 100 + seed));
    }
    auto l = dot_lists(run);

    auto results = [&]() {
        std::vector<double> out;
        out.reserve(run.size());
        for (auto const &e : run) {
            out.push_back(e.r.data()[0]);
        }
        return out;
    };

    // `executor == nullptr` means the eager call rather than a replay, which is
    // the third way in and the one the solver's convergence norm takes.
    auto at = [&](int width, std::shared_ptr<cg::Executor> executor) {
        int const prior = omp_get_max_threads();
        omp_set_num_threads(width);
        for (auto &e : run) {
            e.r.data()[0] = std::numeric_limits<double>::quiet_NaN();
        }
        if (executor == nullptr) {
            cg::grouped_dot(l.r, l.a, l.b);
        } else {
            cg::Graph graph("grouped_dot_width");
            {
                cg::CaptureGuard const guard(graph);
                cg::grouped_dot(l.r, l.a, l.b);
            }
            graph.set_executor(std::move(executor));
            graph.execute();
        }
        omp_set_num_threads(prior);
        return results();
    };

    auto const reference = at(1, nullptr);

    struct Case {
        int         width;
        char const *what;
    };
    // Ten as well as the machine's own count: the width a caller presents is
    // not bounded by the cores it has, and the vendor's partition is a function
    // of the number it is given.
    for (Case const &c :
         {Case{1, "eager, width 1"}, Case{2, "eager, width 2"}, Case{10, "eager, width 10"}, Case{available, "eager, every thread"}}) {
        INFO(c.what);
        REQUIRE(at(c.width, nullptr) == reference);
    }

    for (Case const &c : {Case{1, "width 1"}, Case{2, "width 2"}, Case{10, "width 10"}, Case{available, "every thread"}}) {
        INFO(std::string("SequentialExecutor, ") + c.what);
        REQUIRE(at(c.width, std::make_shared<cg::SequentialExecutor>()) == reference);
        INFO(std::string("OpenMPExecutor, ") + c.what);
        REQUIRE(at(c.width, std::make_shared<cg::OpenMPExecutor>()) == reference);
        INFO(std::string("DataflowExecutor, ") + c.what);
        REQUIRE(at(c.width, std::make_shared<cg::DataflowExecutor>()) == reference);
    }
}
#endif

TEST_CASE("grouped_dot: survives a rebind", "[ComputeGraph][GroupedScalarOps][Rebind]") {
    // The one test a node that baked its pointers in would fail.
    auto run = mixed_dot_run<double>();
    auto ref = single_dots<double>(run);
    auto l   = dot_lists(run);

    cg::Graph graph("grouped_dot_rebind");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_dot(l.r, l.a, l.b);
    }

    Tensor<double, 1> fresh = create_zero_tensor<double>("fresh", 1);
    graph.rebind(run[2].r, fresh);
    graph.execute();

    REQUIRE(fresh.data()[0] == ref[2]);
}

TEST_CASE("grouped_dot: entries are ordered, so a repeated destination is the loop", "[ComputeGraph][GroupedScalarOps]") {
    // Sequential execution is the operation's contract, not an implementation
    // detail, and this is what it buys: writing one destination twice means what
    // the loop means rather than racing.
    auto run = mixed_dot_run<double>();
    auto l   = dot_lists(run);
    l.r[3]   = l.r[0]; // entries 0 and 3 share a scalar; 3 is emitted later

    auto ref = single_dots<double>(run);
    cg::grouped_dot(l.r, l.a, l.b);
    REQUIRE(run[0].r.data()[0] == ref[3]);
}

TEST_CASE("grouped_dot: rejects malformed runs", "[ComputeGraph][GroupedScalarOps]") {
    auto run = mixed_dot_run<double>();
    auto l   = dot_lists(run);

    SECTION("empty") {
        std::vector<Tensor<double, 1> *> const       none_r;
        std::vector<Tensor<double, 2> const *> const none_a, none_b;
        REQUIRE_THROWS_AS(cg::grouped_dot(none_r, none_a, none_b), std::invalid_argument);
    }

    SECTION("mismatched list lengths") {
        auto shorter = l.b;
        shorter.pop_back();
        REQUIRE_THROWS_AS(cg::grouped_dot(l.r, l.a, shorter), std::invalid_argument);
    }

    SECTION("null operand") {
        auto holed = l.a;
        holed[2]   = nullptr;
        REQUIRE_THROWS_AS(cg::grouped_dot(l.r, holed, l.b), std::invalid_argument);
    }

    SECTION("an entry whose operands do not agree on shape") {
        // The single form throws from inside `dot`; the grouped form checks at
        // capture so the message can name the entry.
        auto wrong = create_random_tensor<double>("wrong", 3, 5);
        auto bad   = l.b;
        bad[1]     = &wrong;
        REQUIRE_THROWS_AS(cg::grouped_dot(l.r, l.a, bad), DimensionError);
    }
}

TEMPLATE_TEST_CASE("grouped_axpby: eager is bit-for-bit the loop of single axpbys", "[ComputeGraph][GroupedScalarOps]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;

    std::vector<AxpbyEntry<T>> run;
    run.push_back(make_axpby_entry<T>(4, 5, 0));
    run.push_back(make_axpby_entry<T>(1, 1, 1));
    run.push_back(make_axpby_entry<T>(0, 3, 2));
    run.push_back(make_axpby_entry<T>(9, 2, 3));

    // Per-entry prefactors, none of them 1 or 0 twice over, so an entry reading
    // the wrong slot of either list cannot pass.
    std::vector<double> const alphas{2.0, -3.0, 0.5, 1.25};
    std::vector<double> const betas{1.0, -0.75, 2.5, 0.0};

    auto reference = run;
    for (size_t i = 0; i < reference.size(); i++) {
        linear_algebra::axpby(static_cast<T>(alphas[i]), reference[i].x, static_cast<T>(betas[i]), &reference[i].y);
    }

    std::vector<Tensor<T, 2> const *> x_list;
    std::vector<Tensor<T, 2> *>       y_list;
    for (auto &e : run) {
        x_list.push_back(&e.x);
        y_list.push_back(&e.y);
    }
    cg::grouped_axpby(alphas, x_list, betas, y_list);

    for (size_t i = 0; i < run.size(); i++) {
        INFO("entry " << i);
        REQUIRE(run[i].y.size() == reference[i].y.size());
        for (size_t e = 0; e < run[i].y.size(); e++) {
            REQUIRE(run[i].y.data()[e] == reference[i].y.data()[e]);
        }
    }
}

TEST_CASE("grouped_axpby: a chain into one destination keeps its term order", "[ComputeGraph][GroupedScalarOps]") {
    // The DLPNO idiom the operation exists for: several scalars accumulated into
    // one element of a shared matrix. Sequential entries make the sum arrive in
    // emission order, which is what makes merging the family move no bit.
    constexpr size_t               n    = 6;
    auto                           dest = create_zero_tensor<double>("dest", 1, 1);
    std::vector<Tensor<double, 2>> sources;
    sources.reserve(n);
    for (size_t i = 0; i < n; i++) {
        sources.push_back(create_random_tensor<double>(fmt::format("s{}", i), 1, 1));
    }

    std::vector<double>                    alphas, betas;
    std::vector<Tensor<double, 2> const *> x_list;
    std::vector<Tensor<double, 2> *>       y_list;
    for (size_t i = 0; i < n; i++) {
        alphas.push_back(1.0 + 0.25 * static_cast<double>(i));
        betas.push_back(1.0);
        x_list.push_back(&sources[i]);
        y_list.push_back(&dest);
    }

    auto looped = create_zero_tensor<double>("looped", 1, 1);
    for (size_t i = 0; i < n; i++) {
        linear_algebra::axpby(alphas[i], sources[i], 1.0, &looped);
    }

    cg::Graph graph("grouped_axpby_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_axpby(alphas, x_list, betas, y_list);
    }
    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::GroupedAxpby);
    graph.execute();

    REQUIRE(dest.data()[0] == looped.data()[0]);
}

TEST_CASE("grouped_axpby: only the accumulating entries record a read of Y", "[ComputeGraph][GroupedScalarOps]") {
    // beta == 0 overwrites, so listing Y as an input would invent an edge; beta
    // != 0 reads it, so not listing it would lose one (bug-1009).
    auto x0 = create_random_tensor<double>("x0", 3, 3);
    auto x1 = create_random_tensor<double>("x1", 3, 3);
    auto y0 = create_zero_tensor<double>("y0", 3, 3);
    auto y1 = create_zero_tensor<double>("y1", 3, 3);

    std::vector<Tensor<double, 2> const *> const x_list{&x0, &x1};
    std::vector<Tensor<double, 2> *> const       y_list{&y0, &y1};

    cg::Graph graph("grouped_axpby_io");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_axpby({1.0, 1.0}, x_list, {0.0, 1.0}, y_list);
    }

    // Two X reads plus the one accumulating Y.
    REQUIRE(graph.nodes()[0].inputs.size() == 3);
    REQUIRE(graph.nodes()[0].outputs.size() == 2);

    auto const *d = std::get_if<cg::GroupedAxpbyDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(d != nullptr);
    REQUIRE(d->total == 2);
    REQUIRE(d->alphas.size() == 2);
    REQUIRE(std::get<double>(d->betas[0]) == 0.0);
    REQUIRE(std::get<double>(d->betas[1]) == 1.0);
}

TEST_CASE("grouped_axpby: survives a rebind", "[ComputeGraph][GroupedScalarOps][Rebind]") {
    auto x0 = create_random_tensor<double>("x0", 3, 3);
    auto y0 = create_zero_tensor<double>("y0", 3, 3);

    std::vector<Tensor<double, 2> const *> const x_list{&x0};
    std::vector<Tensor<double, 2> *> const       y_list{&y0};

    cg::Graph graph("grouped_axpby_rebind");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_axpby({2.0}, x_list, {0.0}, y_list);
    }

    auto fresh = create_zero_tensor<double>("fresh", 3, 3);
    graph.rebind(y0, fresh);
    graph.execute();

    for (size_t e = 0; e < fresh.size(); e++) {
        REQUIRE(fresh.data()[e] == 2.0 * x0.data()[e]);
    }
}

TEST_CASE("grouped_axpby: rejects malformed runs", "[ComputeGraph][GroupedScalarOps]") {
    auto x0 = create_random_tensor<double>("x0", 3, 3);
    auto y0 = create_zero_tensor<double>("y0", 3, 3);

    std::vector<Tensor<double, 2> const *> const x_list{&x0};
    std::vector<Tensor<double, 2> *> const       y_list{&y0};

    SECTION("empty") {
        std::vector<Tensor<double, 2> const *> const none_x;
        std::vector<Tensor<double, 2> *> const       none_y;
        REQUIRE_THROWS_AS(cg::grouped_axpby({}, none_x, {}, none_y), std::invalid_argument);
    }

    SECTION("mismatched list lengths") {
        REQUIRE_THROWS_AS(cg::grouped_axpby({1.0, 1.0}, x_list, {0.0, 0.0}, y_list), std::invalid_argument);
    }

    SECTION("null operand") {
        std::vector<Tensor<double, 2> const *> const holed{nullptr};
        REQUIRE_THROWS_AS(cg::grouped_axpby({1.0}, holed, {0.0}, y_list), std::invalid_argument);
    }

    SECTION("an entry whose operands do not agree on shape") {
        auto                                         wrong = create_random_tensor<double>("wrong", 2, 4);
        std::vector<Tensor<double, 2> const *> const bad{&wrong};
        REQUIRE_THROWS_AS(cg::grouped_axpby({1.0}, bad, {0.0}, y_list), DimensionError);
    }
}

TEST_CASE("grouped_dot then grouped_axpby: the merged pair keeps the edge between them", "[ComputeGraph][GroupedScalarOps]") {
    // The adopted shape in full: one node reduces a family into scalars, the
    // next accumulates them into one element. The second must not be scheduled
    // beside the first, and its answer must be the interleaved loop's.
    constexpr size_t               n = 5;
    std::vector<Tensor<double, 2>> a, b;
    std::vector<Tensor<double, 2>> s;
    a.reserve(n);
    b.reserve(n);
    s.reserve(n);
    for (size_t i = 0; i < n; i++) {
        a.push_back(create_random_tensor<double>(fmt::format("a{}", i), 4, 4));
        b.push_back(create_random_tensor<double>(fmt::format("b{}", i), 4, 4));
        s.push_back(create_zero_tensor<double>(fmt::format("s{}", i), 1, 1));
    }
    auto dest = create_zero_tensor<double>("dest", 1, 1);

    // The loop this replaces, interleaved dot-then-accumulate.
    auto looped = create_zero_tensor<double>("looped", 1, 1);
    {
        auto scratch = create_zero_tensor<double>("scratch", 1, 1);
        for (size_t i = 0; i < n; i++) {
            scratch.data()[0] = linear_algebra::dot(a[i], b[i]);
            linear_algebra::axpby(-2.0, scratch, 1.0, &looped);
        }
    }

    std::vector<Tensor<double, 2> *>       s_ptr;
    std::vector<Tensor<double, 2> const *> a_ptr, b_ptr, s_in;
    std::vector<Tensor<double, 2> *>       dest_ptr;
    std::vector<double>                    alphas, betas;
    for (size_t i = 0; i < n; i++) {
        s_ptr.push_back(&s[i]);
        a_ptr.push_back(&a[i]);
        b_ptr.push_back(&b[i]);
        s_in.push_back(&s[i]);
        dest_ptr.push_back(&dest);
        alphas.push_back(-2.0);
        betas.push_back(1.0);
    }

    cg::Graph graph("grouped_pair");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_dot(s_ptr, a_ptr, b_ptr);
        cg::grouped_axpby(alphas, s_in, betas, dest_ptr);
    }
    REQUIRE(graph.num_nodes() == 2);

    // The accumulate reads what the reduction writes, so they are two levels.
    auto const deps = graph.dependencies();
    REQUIRE(deps.levels.size() == 2);

    graph.execute();
    REQUIRE(dest.data()[0] == looped.data()[0]);
}
