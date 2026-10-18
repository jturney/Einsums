//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// An einsum over three or more operands: the order plan_nary_einsum chooses, and the results of the
// eager and captured einsums against a brute-force loop over every index assignment, which shares
// no code with the planner or with the two-operand engine the einsum runs on.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Real or complex values that differ per element and per tensor.
template <typename T>
RuntimeTensor<T> filled(std::string const &name, std::vector<size_t> const &dims, double seed) {
    RuntimeTensor<T> out(name, dims);
    T               *data = out.data();
    for (size_t e = 0; e < out.size(); ++e) {
        double const re = std::sin(seed + 0.37 * static_cast<double>(e));
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            data[e] = T(static_cast<typename T::value_type>(re),
                        static_cast<typename T::value_type>(std::cos(seed * 1.3 + 0.11 * static_cast<double>(e))));
        } else {
            data[e] = static_cast<T>(re);
        }
    }
    return out;
}

template <typename T>
T conj_if(T value, bool conj) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return conj ? std::conj(value) : value;
    } else {
        return value;
    }
}

/// The element of @p tensor at @p index, through its strides.
template <typename T>
T element(RuntimeTensor<T> const &tensor, std::vector<size_t> const &index) {
    size_t offset = 0;
    for (size_t axis = 0; axis < index.size(); ++axis) {
        offset += index[axis] * tensor.stride(axis);
    }
    return tensor.data()[offset];
}

/// C = c_pf * C + ab_pf * prod(operands), every letter not in @p out summed, by visiting every assignment.
template <typename T>
void brute_force(std::vector<std::string> const &out, T c_pf, RuntimeTensor<T> &C, T ab_pf,
                 std::vector<std::vector<std::string>> const &letters, std::vector<RuntimeTensor<T> const *> const &operands,
                 std::vector<bool> const &conj) {
    std::map<std::string, size_t> extent;
    for (size_t op = 0; op < letters.size(); ++op) {
        for (size_t axis = 0; axis < letters[op].size(); ++axis) {
            extent[letters[op][axis]] = operands[op]->dim(axis);
        }
    }
    std::vector<std::string> names;
    for (auto const &[name, size] : extent) {
        names.push_back(name);
    }
    for (size_t e = 0; e < C.size(); ++e) {
        C.data()[e] *= c_pf;
    }
    std::map<std::string, size_t> at;
    for (auto const &name : names) {
        at[name] = 0;
    }
    while (true) {
        T product = ab_pf;
        for (size_t op = 0; op < letters.size(); ++op) {
            std::vector<size_t> index;
            for (auto const &letter : letters[op]) {
                index.push_back(at[letter]);
            }
            product *= conj_if(element(*operands[op], index), conj[op]);
        }
        size_t offset = 0;
        for (size_t axis = 0; axis < out.size(); ++axis) {
            offset += at[out[axis]] * C.stride(axis);
        }
        C.data()[offset] += product;

        size_t k = 0;
        for (; k < names.size(); ++k) {
            if (++at[names[k]] < extent[names[k]]) {
                break;
            }
            at[names[k]] = 0;
        }
        if (k == names.size()) {
            break;
        }
    }
}

template <typename T>
void require_close(RuntimeTensor<T> const &got, RuntimeTensor<T> const &want) {
    double const tol = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>> ? 2e-4 : 1e-11;
    REQUIRE(got.size() == want.size());
    for (size_t e = 0; e < got.size(); ++e) {
        CAPTURE(e);
        REQUIRE(std::abs(got.data()[e] - want.data()[e]) <= tol * (1.0 + std::abs(want.data()[e])));
    }
}

/// The ladder's extents: Bvv[Q,a,b], tau[i,j,a,b].
cg::NaryEinsumStep first_step(size_t naux, size_t nocc, size_t nvir, size_t cap = 0) {
    auto const steps =
        cg::plan_nary_einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", {{naux, nvir, nvir}, {naux, nvir, nvir}, {nocc, nocc, nvir, nvir}}, cap);
    REQUIRE(steps.size() == 2);
    return steps.front();
}

} // namespace

TEST_CASE("nary einsum: the plan follows the arithmetic, not the order written", "[ComputeGraph][NaryEinsum]") {
    // The DF particle ladder r2 = Bvv Bvv tau has two orders: (ae|bf) = Bvv Bvv first, Q v^4 + o^2 v^4
    // multiply-adds, or Bvv tau first, 2 Q o^2 v^3. Benzene in cc-pVDZ (Q 420, o 21, v 93) is 6.4e10
    // against 3.0e11, so the integrals are formed; a large virtual space reverses it.
    cg::NaryEinsumStep const benzene = first_step(420, 21, 93);
    CHECK(benzene.left == 0);
    CHECK(benzene.right == 1);

    cg::NaryEinsumStep const large_v = first_step(50, 2, 500);
    CHECK(large_v.right == 2);
}

TEST_CASE("nary einsum: a cap on the partial products changes the order", "[ComputeGraph][NaryEinsum]") {
    // Contracting the first two operands first sums e and holds a 400-element (b,d) partial product;
    // the second and third first form a 60-element outer product (e,a), at 2.6 times the arithmetic.
    std::vector<std::vector<size_t>> const dims{{20, 3, 20}, {3}, {20}};
    auto const                             free = cg::plan_nary_einsum("a,b,d <- b,e,d ; e ; a", dims);
    CHECK(free.front().left == 0);
    CHECK(free.front().right == 1);

    auto const capped = cg::plan_nary_einsum("a,b,d <- b,e,d ; e ; a", dims, 100);
    CHECK(capped.front().left == 1);
    CHECK(capped.front().right == 2);

    // Nothing fits under one element, so the cap is dropped rather than the einsum refused.
    auto const impossible = cg::plan_nary_einsum("a,b,d <- b,e,d ; e ; a", dims, 1);
    CHECK(impossible.front().left == 0);
    CHECK(impossible.front().right == 1);
}

namespace {

/// D(i,j) = A(i,a) B(a,k) C(k,j), with i and j occupied, a virtual, and k over a third space s
/// declared smaller than the virtuals. Contracting A with B first costs o v s + o^2 s and B with C
/// first o v s + o^2 v, so the first is cheaper term for term. Three occupied, two virtual and six
/// over s make the second cheaper at these extents, 54 against 90, which is the order an unannotated
/// plan takes. (With k occupied the two would cost o^2 v + o^3 and 2 o^2 v, which the scale-order
/// rung cannot rank, since it matches terms one to one and does not split a coefficient.)
struct ChainOverSpaces {
    cg::Graph                                   graph{"spaces"};
    cg::SpaceId                                 occ{graph.space_registry().register_space(cg::make_index_space("occ", "o"))};
    cg::SpaceId                                 virt{graph.space_registry().register_space(cg::make_index_space("virt", "v"))};
    cg::SpaceId                                 small{graph.space_registry().register_space(cg::make_index_space("small", "s"))};
    std::string                                 spec = "i,j <- i,a ; a,k ; k,j";
    std::vector<std::vector<std::size_t>> const dims{{3, 2}, {2, 6}, {6, 3}};

    ChainOverSpaces() { graph.space_registry().declare_less(small, virt); }
};

/// The operands of a plan's first step.
std::pair<std::size_t, std::size_t> first_pair(std::vector<cg::NaryEinsumStep> const &steps) {
    REQUIRE_FALSE(steps.empty());
    return {steps.front().left, steps.front().right};
}

} // namespace

TEST_CASE("nary einsum: declared index spaces decide the order where the extents would not", "[ComputeGraph][NaryEinsum]") {
    ChainOverSpaces chain;
    using Pair = std::pair<std::size_t, std::size_t>;

    // By extent alone, B with C first.
    CHECK(first_pair(cg::plan_nary_einsum(chain.spec, chain.dims)) == Pair{1, 2});
    // By the declared scaling, A with B first, as the graph's own factorization ranks it.
    auto const o  = chain.occ;
    auto const v  = chain.virt;
    auto const sm = chain.small;
    CHECK(first_pair(cg::plan_nary_einsum_with_spaces(chain.graph, chain.spec, chain.dims, {{o, v}, {v, sm}, {sm, o}, {o, o}})) ==
          Pair{0, 1});
    // An operand left unannotated does not stop the others binding the letters it shares.
    CHECK(first_pair(cg::plan_nary_einsum_with_spaces(chain.graph, chain.spec, chain.dims, {{o, v}, {}, {sm, o}, {}})) == Pair{0, 1});
    // With no annotation at all the extents decide, as without the graph.
    CHECK(first_pair(cg::plan_nary_einsum_with_spaces(chain.graph, chain.spec, chain.dims, {{}, {}, {}, {}})) == Pair{1, 2});
    CHECK(first_pair(cg::plan_nary_einsum_with_spaces(chain.graph, chain.spec, chain.dims, {})) == Pair{1, 2});

    // One letter annotated with two spaces, and a list that is not one entry per operand and C.
    CHECK_THROWS_AS(cg::plan_nary_einsum_with_spaces(chain.graph, chain.spec, chain.dims, {{o, o}, {v, sm}, {sm, o}, {o, o}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(cg::plan_nary_einsum_with_spaces(chain.graph, chain.spec, chain.dims, {{o, v}, {v, sm}, {sm, o}}),
                    std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE("nary einsum: a captured einsum is planned by its operands' annotations", "[ComputeGraph][NaryEinsum]",
                        testing::AllScalarTypes) {
    using T = TestType;
    ChainOverSpaces chain;
    auto const      A        = filled<T>("A", {3, 2}, 0.4);
    auto const      B        = filled<T>("B", {2, 6}, 1.3);
    auto const      C        = filled<T>("C", {6, 3}, 2.2);
    auto            want     = filled<T>("D", {3, 3}, 3.5);
    auto            captured = want;
    T const         c_pf     = testing::prefactor<T>(0.3, -0.7);
    T const         ab_pf    = testing::prefactor<T>(1.1, 0.4);
    brute_force<T>({"i", "j"}, c_pf, want, ab_pf, {{"i", "a"}, {"a", "k"}, {"k", "j"}}, {&A, &B, &C}, {false, false, false});

    // Annotated before capture, as capture reads the annotations.
    chain.graph.annotate_spaces(A, {chain.occ, chain.virt});
    chain.graph.annotate_spaces(B, {chain.virt, chain.small});
    chain.graph.annotate_spaces(C, {chain.small, chain.occ});
    {
        cg::CaptureGuard const guard(chain.graph);
        cg::einsum("i,j <- i,a ; a,k ; k,j", c_pf, &captured, ab_pf, A, B, C);
    }
    std::vector<std::string> names;
    for (auto const &[id, handle] : chain.graph.tensors_map()) {
        names.push_back(handle.name);
    }
    CHECK(std::ranges::find(names, "A*B") != names.end());
    CHECK(std::ranges::find(names, "B*C") == names.end());

    chain.graph.execute();
    require_close(captured, want);
}

TEST_CASE("nary einsum: malformed specs and operands are refused", "[ComputeGraph][NaryEinsum]") {
    std::vector<std::vector<size_t>> const three{{2, 3}, {3, 4}, {4, 5}};
    CHECK_THROWS_AS(cg::plan_nary_einsum("i,l <- i,j ; j,k", three), std::invalid_argument);            // operand count
    CHECK_THROWS_AS(cg::plan_nary_einsum("i,l <- i,j ; j,k ; k", three), std::invalid_argument);        // rank
    CHECK_THROWS_AS(cg::plan_nary_einsum("i,l <- i,j ; j,k ; j,l", three), std::invalid_argument);      // extent of j
    CHECK_THROWS_AS(cg::plan_nary_einsum("i,m <- i,j ; j,k ; k,l", three), std::invalid_argument);      // m in no operand
    CHECK_THROWS_AS(cg::plan_nary_einsum("i,l <- i,j ; j,k ; P(k) k,l", three), std::invalid_argument); // operator inside
    CHECK_NOTHROW(cg::plan_nary_einsum("i,l <- i,j ; j,k ; k,l", three));
}

TEMPLATE_LIST_TEST_CASE("nary einsum: eager and captured agree with every assignment summed", "[ComputeGraph][NaryEinsum]",
                        testing::AllScalarTypes) {
    using T       = TestType;
    T const c_pf  = testing::prefactor<T>(0.3, -0.7);
    T const ab_pf = testing::prefactor<T>(1.1, 0.4);

    SECTION("the ladder, one operand read twice") {
        auto const B        = filled<T>("B", {4, 3, 3}, 0.2);
        auto const tau      = filled<T>("tau", {2, 2, 3, 3}, 1.7);
        auto       want     = filled<T>("C", {2, 2, 3, 3}, 3.1);
        auto       eager    = want;
        auto       captured = want;
        brute_force<T>({"i", "j", "a", "b"}, c_pf, want, ab_pf, {{"Q", "a", "e"}, {"Q", "b", "f"}, {"i", "j", "e", "f"}}, {&B, &B, &tau},
                       {false, false, false});

        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", c_pf, &eager, ab_pf, B, B, tau);
        require_close(eager, want);

        cg::Graph graph("ladder");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", c_pf, &captured, ab_pf, B, B, tau);
        }
        graph.execute();
        require_close(captured, want);
    }

    SECTION("four operands, a letter shared by three, one summed alone, and a conjugate") {
        // Q is in three operands and not the output; m is in one operand alone, so it is summed
        // where that operand is first contracted.
        auto const A        = filled<T>("A", {3, 4}, 0.5);    // Q,a
        auto const B        = filled<T>("B", {3, 2, 5}, 1.1); // Q,b,m
        auto const W        = filled<T>("W", {3}, 2.3);       // Q
        auto const D        = filled<T>("D", {4, 2}, 0.9);    // a,b
        auto       want     = filled<T>("C", {2, 4}, 4.2);    // b,a
        auto       eager    = want;
        auto       captured = want;
        brute_force<T>({"b", "a"}, c_pf, want, ab_pf, {{"Q", "a"}, {"Q", "b", "m"}, {"Q"}, {"a", "b"}}, {&A, &B, &W, &D},
                       {false, true, false, false});

        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("b,a <- Q,a ; conj(Q,b,m) ; Q ; a,b", c_pf, &eager, ab_pf, A, B, W, D);
        require_close(eager, want);

        cg::Graph graph("four");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("b,a <- Q,a ; conj(Q,b,m) ; Q ; a,b", c_pf, &captured, ab_pf, A, B, W, D);
        }
        graph.execute();
        require_close(captured, want);
    }
}

TEMPLATE_LIST_TEST_CASE("nary einsum: operands that sum to a scalar", "[ComputeGraph][NaryEinsum]", testing::AllScalarTypes) {
    // Every letter is summed and no two operands share one, so any order contracts two operands to a
    // scalar, which a two-operand spec cannot read. The partial product keeps its shortest letter,
    // and the next contraction sums it.
    using T             = TestType;
    auto const A        = filled<T>("A", {2}, 0.4);
    auto const B        = filled<T>("B", {3}, 1.2);
    auto const D        = filled<T>("D", {4}, 2.5);
    auto       want     = filled<T>("C", {}, 0.7);
    auto       eager    = want;
    auto       captured = want;
    T const    c_pf     = testing::prefactor<T>(0.3, -0.7);
    brute_force<T>({}, c_pf, want, T{1}, {{"i"}, {"j"}, {"k"}}, {&A, &B, &D}, {false, false, false});

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum(" <- i ; j ; k", c_pf, &eager, T{1}, A, B, D);
    require_close(eager, want);

    cg::Graph graph("scalar");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum(" <- i ; j ; k", c_pf, &captured, T{1}, A, B, D);
    }
    graph.execute();
    require_close(captured, want);
}

TEMPLATE_LIST_TEST_CASE("nary einsum: a captured einsum survives the default passes and replays", "[ComputeGraph][NaryEinsum]",
                        testing::RealScalarTypes) {
    using T         = TestType;
    auto const B    = filled<T>("B", {4, 3, 3}, 0.2);
    auto const tau  = filled<T>("tau", {2, 2, 3, 3}, 1.7);
    auto       r2   = filled<T>("r2", {2, 2, 3, 3}, 0.0);
    auto       want = r2;
    brute_force<T>({"i", "j", "a", "b"}, T{0}, want, T{1}, {{"Q", "a", "e"}, {"Q", "b", "f"}, {"i", "j", "e", "f"}}, {&B, &B, &tau},
                   {false, false, false});

    cg::Graph graph("ladder");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", &r2, B, B, tau);
    }
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    for (int replay = 0; replay < 2; ++replay) {
        graph.execute();
        require_close(r2, want);
    }
}

TEST_CASE("internal tensors: storage at the first execute when a node uses them, and only then", "[ComputeGraph][NaryEinsum]") {
    // A partial product of a captured einsum is declared on the caller's behalf, so a graph run
    // without the passes must not fail on it the way it fails on a tensor the caller declared and
    // left without storage. One no node uses, as AxisTiling leaves a sliced one, gets none.
    auto const A = filled<double>("A", {3, 3}, 0.1);
    auto       C = filled<double>("C", {3, 3}, 0.0);

    cg::Graph graph("internal");
    auto     &used   = graph.declare_internal_tensor<double>("used", {3, 3});
    auto     &unused = graph.declare_internal_tensor<double>("unused", {64, 64});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &used, A, A);
        cg::einsum("ij <- ik ; kj", &C, used, A);
    }
    REQUIRE_NOTHROW(graph.execute());
    CHECK(used.is_materialized());
    CHECK_FALSE(unused.is_materialized());

    cg::Graph declared("declared");
    auto     &scratch = declared.declare_runtime_tensor<double>("scratch", {3, 3}, /*intermediate=*/true);
    {
        cg::CaptureGuard const guard(declared);
        cg::einsum("ij <- ik ; kj", &scratch, A, A);
        cg::einsum("ij <- ik ; kj", &C, scratch, A);
    }
    CHECK_THROWS_WITH(declared.execute(), Catch::Matchers::ContainsSubstring("still deferred"));
}
