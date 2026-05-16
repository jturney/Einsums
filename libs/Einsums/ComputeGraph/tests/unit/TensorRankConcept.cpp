//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <catch2/catch_all.hpp>

using namespace einsums;
namespace compute_graph = ::einsums::compute_graph;
using compute_graph::HasCompileTimeRank;
using compute_graph::RuntimeRankTensorConcept;

TEST_CASE("HasCompileTimeRank — typed tensors expose ::Rank", "[ComputeGraph][TensorRank]") {
    STATIC_REQUIRE(HasCompileTimeRank<Tensor<double, 1>>);
    STATIC_REQUIRE(HasCompileTimeRank<Tensor<double, 2>>);
    STATIC_REQUIRE(HasCompileTimeRank<Tensor<double, 4>>);
    STATIC_REQUIRE(HasCompileTimeRank<TensorView<double, 2>>);
    STATIC_REQUIRE(HasCompileTimeRank<Tensor<float, 3> const &>);

    STATIC_REQUIRE_FALSE(HasCompileTimeRank<RuntimeTensor<double>>);
    STATIC_REQUIRE_FALSE(HasCompileTimeRank<RuntimeTensor<float>>);
    STATIC_REQUIRE_FALSE(HasCompileTimeRank<RuntimeTensor<std::complex<double>>>);
}

TEST_CASE("RuntimeRankTensorConcept — only RuntimeTensor satisfies", "[ComputeGraph][TensorRank]") {
    STATIC_REQUIRE(RuntimeRankTensorConcept<RuntimeTensor<double>>);
    STATIC_REQUIRE(RuntimeRankTensorConcept<RuntimeTensor<float>>);
    STATIC_REQUIRE(RuntimeRankTensorConcept<RuntimeTensor<std::complex<float>>>);

    STATIC_REQUIRE_FALSE(RuntimeRankTensorConcept<Tensor<double, 2>>);
    STATIC_REQUIRE_FALSE(RuntimeRankTensorConcept<TensorView<double, 2>>);
    STATIC_REQUIRE_FALSE(RuntimeRankTensorConcept<int>);
}

TEST_CASE("BasicTensorConcept — RuntimeTensor satisfies the duck-type API", "[ComputeGraph][TensorRank]") {
    STATIC_REQUIRE(BasicTensorConcept<RuntimeTensor<double>>);
    STATIC_REQUIRE(BasicTensorConcept<Tensor<double, 2>>);
}

TEST_CASE("compute_graph::detail::tensor_rank — compile-time path for typed, runtime path for RuntimeTensor",
          "[ComputeGraph][TensorRank]") {
    Tensor<double, 2> const     typed_2d("typed", 3, 4);
    Tensor<double, 4> const     typed_4d("typed", 2, 2, 2, 2);
    RuntimeTensor<double> const runtime_2d("rt", {3, 4});
    RuntimeTensor<double> const runtime_4d("rt", {2, 2, 2, 2});

    REQUIRE(compute_graph::detail::tensor_rank(typed_2d) == 2);
    REQUIRE(compute_graph::detail::tensor_rank(typed_4d) == 4);
    REQUIRE(compute_graph::detail::tensor_rank(runtime_2d) == 2);
    REQUIRE(compute_graph::detail::tensor_rank(runtime_4d) == 4);
}

TEST_CASE("IsSameRankV — typed tensors compare exact ranks", "[ComputeGraph][TensorRank]") {
    STATIC_REQUIRE(IsSameRankV<Tensor<double, 2>, Tensor<double, 2>>);
    STATIC_REQUIRE(IsSameRankV<Tensor<double, 3>, Tensor<float, 3>, Tensor<int, 3>>);
    STATIC_REQUIRE_FALSE(IsSameRankV<Tensor<double, 2>, Tensor<double, 3>>);
}

TEST_CASE("IsSameRankV — dynamic-rank operands act as compile-time wildcards", "[ComputeGraph][TensorRank][Regression]") {
    // Before introducing einsums::dynamic_rank, evaluating these would have
    // been a hard error: GeneralRuntimeTensor lacked a static ::Rank member.
    // Now both sides must be queryable; mismatches against the sentinel
    // are deferred to runtime.
    STATIC_REQUIRE(IsSameRankV<RuntimeTensor<double>, RuntimeTensor<double>>);
    STATIC_REQUIRE(IsSameRankV<RuntimeTensor<double>, Tensor<double, 2>>);
    STATIC_REQUIRE(IsSameRankV<Tensor<double, 5>, RuntimeTensor<double>>);
    STATIC_REQUIRE(IsSameRankV<RuntimeTensor<double>, RuntimeTensorView<double>>);
}

TEST_CASE("IsSameUnderlyingAndRankV — dtype mismatch still rejected", "[ComputeGraph][TensorRank][Regression]") {
    STATIC_REQUIRE(IsSameUnderlyingAndRankV<RuntimeTensor<double>, RuntimeTensor<double>>);
    STATIC_REQUIRE_FALSE(IsSameUnderlyingAndRankV<RuntimeTensor<double>, RuntimeTensor<float>>);
    STATIC_REQUIRE_FALSE(IsSameUnderlyingAndRankV<Tensor<double, 2>, Tensor<float, 2>>);
}

TEST_CASE("RankTensorConcept — RuntimeTensor still excluded by sentinel", "[ComputeGraph][TensorRank][Regression]") {
    // RuntimeTensor now has a ``Rank`` member but its value is the sentinel
    // dynamic_rank, so the static-rank concept must still reject it.
    STATIC_REQUIRE(RankTensorConcept<Tensor<double, 2>>);
    STATIC_REQUIRE(RankTensorConcept<Tensor<double, 2>, 2>);
    STATIC_REQUIRE_FALSE(RankTensorConcept<Tensor<double, 2>, 3>);
    STATIC_REQUIRE_FALSE(RankTensorConcept<RuntimeTensor<double>>);
    STATIC_REQUIRE_FALSE(RankTensorConcept<RuntimeTensorView<double>>);
}
