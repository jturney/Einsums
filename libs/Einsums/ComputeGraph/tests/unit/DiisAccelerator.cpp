//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The fixed point ``t = a * t + b``, whose plain iteration converges at the rate ``a``.
struct FixedPoint {
    double              a{0.93};
    std::vector<double> b;

    [[nodiscard]] std::vector<double> exact() const {
        std::vector<double> x(b.size());
        for (size_t i = 0; i < b.size(); i++) {
            x[i] = b[i] / (1.0 - a);
        }
        return x;
    }
};

FixedPoint make_problem(size_t n) {
    FixedPoint p;
    p.b.resize(n);
    for (size_t i = 0; i < n; i++) {
        p.b[i] = 0.5 + 0.25 * static_cast<double>(i % 7);
    }
    return p;
}

double max_error(RuntimeTensor<double> const &t, std::vector<double> const &exact) {
    double worst = 0.0;
    for (size_t i = 0; i < exact.size(); i++) {
        worst = std::max(worst, std::abs(t.data()[i] - exact[i]));
    }
    return worst;
}

} // namespace

TEST_CASE("DiisAccelerator - eager stepping beats the plain fixed point", "[ComputeGraph][DIIS]") {
    constexpr size_t n     = 24;
    auto const       prob  = make_problem(n);
    auto const       exact = prob.exact();

    auto run = [&](bool accelerate, size_t iterations) {
        RuntimeTensor<double> t("t", std::vector<size_t>{n});
        RuntimeTensor<double> s("s", std::vector<size_t>{n});
        t.zero();
        s.zero();

        auto acc = std::make_shared<cg::DiisAccelerator<double>>(6);
        cg::diis_add_pair(acc.get(), &t, &s);

        for (size_t it = 0; it < iterations; it++) {
            for (size_t i = 0; i < n; i++) {
                s.data()[i] = prob.a * t.data()[i] + prob.b[i] - t.data()[i];
            }
            linear_algebra::axpby(1.0, s, 1.0, &t);
            if (accelerate) {
                acc->step();
            }
        }
        return max_error(t, exact);
    };

    // 12 plain passes of a rate-0.93 iteration are nowhere near converged; DIIS
    // over the same 12 steps is, which is the whole claim.
    CHECK(run(false, 12) > 1e-2);
    CHECK(run(true, 12) < 1e-8);
}

TEST_CASE("DiisAccelerator - the history ramps to k and then evicts", "[ComputeGraph][DIIS]") {
    constexpr size_t      n = 5;
    RuntimeTensor<double> t("t", std::vector<size_t>{n});
    RuntimeTensor<double> s("s", std::vector<size_t>{n});
    t.zero();
    s.zero();

    auto acc = std::make_shared<cg::DiisAccelerator<double>>(3);
    cg::diis_add_pair(acc.get(), &t, &s);
    CHECK(acc->num_pairs() == 1);
    CHECK(acc->max_history() == 3);

    // Distinct steps, so the subspace stays non-singular and nothing is dropped
    // for that reason: what the history does here is ramp and then cap.
    std::vector<size_t> sizes;
    for (size_t it = 0; it < 6; it++) {
        for (size_t i = 0; i < n; i++) {
            s.data()[i] = 1.0 / static_cast<double>(it + 1) + 0.125 * static_cast<double>(i);
        }
        linear_algebra::axpby(1.0, s, 1.0, &t);
        acc->step();
        sizes.push_back(acc->history_size());
    }
    CHECK(sizes == std::vector<size_t>{1, 2, 3, 3, 3, 3});

    acc->reset();
    CHECK(acc->history_size() == 0);
}

TEST_CASE("DiisAccelerator - argument validation", "[ComputeGraph][DIIS]") {
    CHECK_THROWS_AS(cg::DiisAccelerator<double>(1), std::invalid_argument);

    RuntimeTensor<double> t("t", std::vector<size_t>{3});
    RuntimeTensor<double> s("s", std::vector<size_t>{3});
    t.zero();
    s.zero();

    auto acc = std::make_shared<cg::DiisAccelerator<double>>(4);
    CHECK_THROWS_AS(acc->step(), std::invalid_argument); // no pairs yet
    cg::diis_add_pair(acc.get(), &t, &s);
    acc->step();
    // The snapshots are shaped for the pair list that produced them.
    CHECK_THROWS_AS(cg::diis_add_pair(acc.get(), &t, &s), std::logic_error);
}

TEST_CASE("DiisAccelerator - a singular subspace drops the oldest pair", "[ComputeGraph][DIIS]") {
    constexpr size_t      n = 6;
    RuntimeTensor<double> t("t", std::vector<size_t>{n});
    RuntimeTensor<double> s("s", std::vector<size_t>{n});
    t.zero();
    s.zero();
    for (size_t i = 0; i < n; i++) {
        s.data()[i] = 1e-3; // identical steps every pass -> rank-1 B
    }

    auto acc = std::make_shared<cg::DiisAccelerator<double>>(4);
    cg::diis_add_pair(acc.get(), &t, &s);

    for (size_t it = 0; it < 5; it++) {
        acc->step();
        for (size_t i = 0; i < n; i++) {
            CHECK(std::isfinite(t.data()[i]));
        }
        CHECK(acc->history_size() >= 1);
        CHECK(acc->history_size() <= 4);
    }
}

TEST_CASE("DiisAccelerator - capture records one node ordered on the amplitudes", "[ComputeGraph][DIIS]") {
    constexpr size_t n     = 16;
    auto const       prob  = make_problem(n);
    auto const       exact = prob.exact();

    RuntimeTensor<double> b("b", std::vector<size_t>{n});
    for (size_t i = 0; i < n; i++) {
        b.data()[i] = prob.b[i];
    }

    RuntimeTensor<double> t("t", std::vector<size_t>{n});
    RuntimeTensor<double> s("s", std::vector<size_t>{n});
    RuntimeTensor<double> out("out", std::vector<size_t>{n});
    t.zero();
    s.zero();
    out.zero();

    auto acc = std::make_shared<cg::DiisAccelerator<double>>(6);
    cg::diis_add_pair(acc.get(), &t, &s);

    cg::Graph graph("diis body");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(prob.a, t, 0.0, &s); // s = a t
        cg::axpby(1.0, b, 1.0, &s);    // s = a t + b
        cg::axpby(-1.0, t, 1.0, &s);   // s = (a t + b) - t, the update step
        cg::axpby(1.0, s, 1.0, &t);    // t += s
        cg::diis_step(acc);            // extrapolate in place
        cg::axpby(1.0, t, 0.0, &out);  // a consumer of the extrapolated amplitudes
    }

    REQUIRE(graph.num_nodes() == 6);
    REQUIRE(graph.nodes()[4].kind == cg::OpKind::DiisStep);
    // Every node reads what the one before it wrote, and the DIIS node is no
    // exception: it takes the amplitudes and the step as inputs and writes the
    // amplitudes, so the consumer cannot be scheduled beside it.
    CHECK(graph.schedule_level_sizes().size() == 6);

    for (size_t it = 0; it < 12; it++) {
        graph.execute();
    }
    CHECK(max_error(t, exact) < 1e-8);
    // The consumer ran after the extrapolation, so it holds the same values.
    for (size_t i = 0; i < n; i++) {
        CHECK(out.data()[i] == t.data()[i]);
    }
}

TEST_CASE("DiisAccelerator - a replayed node matches the same steps taken eagerly", "[ComputeGraph][DIIS]") {
    constexpr size_t n    = 10;
    auto const       prob = make_problem(n);

    RuntimeTensor<double> b("b", std::vector<size_t>{n});
    for (size_t i = 0; i < n; i++) {
        b.data()[i] = prob.b[i];
    }

    // Graph side.
    RuntimeTensor<double> tg("tg", std::vector<size_t>{n});
    RuntimeTensor<double> sg("sg", std::vector<size_t>{n});
    tg.zero();
    sg.zero();
    auto acc_g = std::make_shared<cg::DiisAccelerator<double>>(5);
    cg::diis_add_pair(acc_g.get(), &tg, &sg);

    cg::Graph graph("diis replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(prob.a, tg, 0.0, &sg);
        cg::axpby(1.0, b, 1.0, &sg);
        cg::axpby(-1.0, tg, 1.0, &sg);
        cg::axpby(1.0, sg, 1.0, &tg);
        cg::diis_step(acc_g);
    }

    // Eager side: the identical op sequence, outside any capture.
    RuntimeTensor<double> te("te", std::vector<size_t>{n});
    RuntimeTensor<double> se("se", std::vector<size_t>{n});
    te.zero();
    se.zero();
    auto acc_e = std::make_shared<cg::DiisAccelerator<double>>(5);
    cg::diis_add_pair(acc_e.get(), &te, &se);

    for (size_t it = 0; it < 15; it++) {
        graph.execute();

        cg::axpby(prob.a, te, 0.0, &se);
        cg::axpby(1.0, b, 1.0, &se);
        cg::axpby(-1.0, te, 1.0, &se);
        cg::axpby(1.0, se, 1.0, &te);
        cg::diis_step(acc_e);

        // Same operations in the same order: the replay owes bit equality.
        for (size_t i = 0; i < n; i++) {
            REQUIRE(tg.data()[i] == te.data()[i]);
        }
        REQUIRE(acc_g->history_size() == acc_e->history_size());
    }
}

TEMPLATE_TEST_CASE("DiisAccelerator - complex amplitudes extrapolate with real coefficients", "[ComputeGraph][DIIS]", std::complex<float>,
                   std::complex<double>) {
    using T = TestType;
    using R = RemoveComplexT<T>;

    constexpr size_t n = 8;
    RuntimeTensor<T> t("t", std::vector<size_t>{n});
    RuntimeTensor<T> s("s", std::vector<size_t>{n});
    t.zero();
    s.zero();

    std::vector<T> rhs(n);
    for (size_t i = 0; i < n; i++) {
        rhs[i] = T{static_cast<R>(0.5 + 0.1 * static_cast<double>(i)), static_cast<R>(-0.25 + 0.05 * static_cast<double>(i))};
    }
    R const a = static_cast<R>(0.9);

    auto acc = std::make_shared<cg::DiisAccelerator<T>>(5);
    cg::diis_add_pair(acc.get(), &t, &s);

    for (size_t it = 0; it < 12; it++) {
        for (size_t i = 0; i < n; i++) {
            s.data()[i] = a * t.data()[i] + rhs[i] - t.data()[i];
        }
        linear_algebra::axpby(T{1.0}, s, T{1.0}, &t);
        acc->step();
    }

    auto const tol = static_cast<double>(std::is_same_v<R, float> ? 1e-3 : 1e-9);
    for (size_t i = 0; i < n; i++) {
        T const exact = rhs[i] / (T{1.0} - T{a});
        CHECK_THAT(static_cast<double>(t.data()[i].real()), Catch::Matchers::WithinAbs(static_cast<double>(exact.real()), tol));
        CHECK_THAT(static_cast<double>(t.data()[i].imag()), Catch::Matchers::WithinAbs(static_cast<double>(exact.imag()), tol));
    }
}

TEST_CASE("DiisAccelerator - the optimizer keeps the node and its place", "[ComputeGraph][DIIS]") {
    // A new OpKind is opaque to every pass, which is what it has to be: the node
    // owns state no pass can see, so eliminating it, folding it or moving it
    // across the writes it reads would all be wrong. This pins that the default
    // pipeline leaves it where capture put it, and that the optimized graph
    // computes what the unoptimized one does.
    constexpr size_t n     = 12;
    auto const       prob  = make_problem(n);
    auto const       exact = prob.exact();

    RuntimeTensor<double> b("b", std::vector<size_t>{n});
    for (size_t i = 0; i < n; i++) {
        b.data()[i] = prob.b[i];
    }

    auto build = [&](RuntimeTensor<double> &t, RuntimeTensor<double> &s, std::shared_ptr<cg::DiisAccelerator<double>> const &acc,
                     cg::Graph &graph) {
        cg::CaptureGuard const guard(graph);
        cg::axpby(prob.a, t, 0.0, &s);
        cg::axpby(1.0, b, 1.0, &s);
        cg::axpby(-1.0, t, 1.0, &s);
        cg::axpby(1.0, s, 1.0, &t);
        cg::diis_step(acc);
    };

    RuntimeTensor<double> t_opt("t", std::vector<size_t>{n});
    RuntimeTensor<double> s_opt("s", std::vector<size_t>{n});
    t_opt.zero();
    s_opt.zero();
    auto      acc_opt = std::make_shared<cg::DiisAccelerator<double>>(6);
    cg::Graph optimized("diis optimized");
    cg::diis_add_pair(acc_opt.get(), &t_opt, &s_opt);
    build(t_opt, s_opt, acc_opt, optimized);
    optimized.optimize();

    size_t diis_nodes = 0, diis_index = 0;
    for (size_t i = 0; i < optimized.nodes().size(); i++) {
        if (optimized.nodes()[i].kind == cg::OpKind::DiisStep) {
            diis_nodes++;
            diis_index = i;
        }
    }
    REQUIRE(diis_nodes == 1);
    // Still last: it reads the step the body just wrote and writes the amplitudes.
    CHECK(diis_index == optimized.nodes().size() - 1);

    RuntimeTensor<double> t_plain("t", std::vector<size_t>{n});
    RuntimeTensor<double> s_plain("s", std::vector<size_t>{n});
    t_plain.zero();
    s_plain.zero();
    auto      acc_plain = std::make_shared<cg::DiisAccelerator<double>>(6);
    cg::Graph plain("diis unoptimized");
    cg::diis_add_pair(acc_plain.get(), &t_plain, &s_plain);
    build(t_plain, s_plain, acc_plain, plain);

    for (size_t it = 0; it < 15; it++) {
        optimized.execute();
        plain.execute();
    }
    CHECK(max_error(t_opt, exact) < 1e-8);
    for (size_t i = 0; i < n; i++) {
        CHECK(t_opt.data()[i] == t_plain.data()[i]);
    }
}

#ifdef _OPENMP
TEST_CASE("DiisAccelerator - the step does not depend on the thread count", "[ComputeGraph][DIIS]") {
    // A threaded vendor `dot` sums per-thread partials, so its last bits move
    // with the thread count; the accelerator pins the vendor to one thread for
    // the duration of a step so its coefficients are a function of the inputs
    // alone. The components here are past the width at which the vendor starts
    // threading a dot, which is the only size where this can be observed.
    int const available = omp_get_max_threads();
    if (available < 2) {
        SUCCEED("one thread available, nothing to compare against");
        return;
    }

    constexpr size_t n = 40000, pairs = 3, steps = 6;

    auto run = [&](int width) {
        int const prior = omp_get_max_threads();
        omp_set_num_threads(width);

        std::vector<RuntimeTensor<double>> ts, ss;
        for (size_t p = 0; p < pairs; p++) {
            ts.emplace_back("t", std::vector<size_t>{n});
            ss.emplace_back("s", std::vector<size_t>{n});
            for (size_t i = 0; i < n; i++) {
                ts.back().data()[i] = 0.001 * static_cast<double>((i + 7 * p) % 101) - 0.05;
                ss.back().data()[i] = 1e-4 * static_cast<double>((i * 13 + p) % 97) - 5e-3;
            }
        }

        auto acc = std::make_shared<cg::DiisAccelerator<double>>(4);
        for (size_t p = 0; p < pairs; p++) {
            cg::diis_add_pair(acc.get(), &ts[p], &ss[p]);
        }

        for (size_t it = 0; it < steps; it++) {
            for (size_t p = 0; p < pairs; p++) {
                for (size_t i = 0; i < n; i++) {
                    ss[p].data()[i] *= 0.5 + 0.4 * static_cast<double>((i + p) % 17) / 17.0;
                    ts[p].data()[i] += ss[p].data()[i];
                }
            }
            acc->step();
        }

        std::vector<double> out;
        out.reserve(pairs * n);
        for (size_t p = 0; p < pairs; p++) {
            out.insert(out.end(), ts[p].data(), ts[p].data() + n);
        }
        omp_set_num_threads(prior);
        return out;
    };

    auto const serial = run(1);
    auto const wide   = run(available);
    REQUIRE(serial.size() == wide.size());
    for (size_t i = 0; i < serial.size(); i++) {
        REQUIRE(serial[i] == wide[i]);
    }
}
#endif
