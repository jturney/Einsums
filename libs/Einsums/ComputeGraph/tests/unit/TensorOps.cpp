//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The per-type operations table a handle points at. Each table entry stands for one closure
// make_handle installs, so for every capturable type the two have to exist together and give the
// same answer; these cases pin that, and that after a rebind the table reaches the new tensor.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <cstddef>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Every entry make_handle has a closure for exists exactly when that closure does, and the ones
/// that answer a question answer it the same way.
template <typename TensorType>
void check_agrees(TensorType &tensor) {
    cg::TensorHandle const h   = cg::make_handle(tensor, 0);
    cg::TensorOps const   &ops = *h.ops;
    void *const            obj = h.live_ptr();

    CHECK(static_cast<bool>(h.impl_fn) == (ops.impl != nullptr));
    CHECK(static_cast<bool>(h.is_materialized_fn) == (ops.is_materialized != nullptr));
    CHECK(static_cast<bool>(h.materialize_fn) == (ops.materialize != nullptr));
    CHECK(static_cast<bool>(h.materialize_into_fn) == (ops.materialize_into != nullptr));
    CHECK(static_cast<bool>(h.release_fn) == (ops.release != nullptr));
    CHECK(static_cast<bool>(h.storage_of) == (ops.storage != nullptr));
    CHECK(static_cast<bool>(h.allreduce_sum_fn) == (ops.allreduce_sum != nullptr));
    CHECK(static_cast<bool>(h.iallreduce_sum_fn) == (ops.iallreduce_sum != nullptr));
    CHECK(static_cast<bool>(h.declared_symmetry_fn) == (ops.declared_symmetry != nullptr));
    CHECK(static_cast<bool>(h.set_symmetry_fn) == (ops.set_symmetry != nullptr));
    CHECK(static_cast<bool>(h.begin_local_view_fn) == (ops.begin_local_view != nullptr));
    CHECK(static_cast<bool>(h.end_local_view_fn) == (ops.end_local_view != nullptr));

    if (h.impl_fn) {
        CHECK(h.impl_fn() == ops.impl(obj));
    }
    if (h.is_materialized_fn) {
        CHECK(h.is_materialized_fn() == ops.is_materialized(obj));
    }
    if (h.storage_of) {
        CHECK(h.storage_of() == ops.storage(obj));
    }
    if (h.declared_symmetry_fn) {
        auto const closure = h.declared_symmetry_fn();
        auto const table   = ops.declared_symmetry(obj);
        REQUIRE((closure == nullptr) == (table == nullptr));
        if (closure != nullptr) {
            CHECK(*closure == *table);
        }
    }
    REQUIRE(ops.name != nullptr);
    CHECK(ops.name(obj) == h.name);
}

} // namespace

TEST_CASE("TensorOps - the table agrees with make_handle for every capturable type", "[ComputeGraph][TensorOps]") {
    SECTION("Tensor") {
        auto A = create_random_tensor<double>("A", 3, 4);
        A.set_symmetry(SymmetryDescriptor::antisymmetric_pair(0, 1));
        check_agrees(A);
    }
    SECTION("deferred Tensor") {
        Tensor<double, 2> shell(Tensor<double, 2>::DeferredAlloc{}, "shell", 3, 4);
        check_agrees(shell);
    }
    SECTION("TensorView") {
        auto P    = create_random_tensor<double>("P", 2, 3, 4);
        auto view = P(0, All, All);
        check_agrees(view);
    }
    SECTION("RuntimeTensor") {
        RuntimeTensor<float> R("R", {3, 4});
        check_agrees(R);
    }
    SECTION("RuntimeTensorView") {
        RuntimeTensor<double>     R("R", {4, 4});
        RuntimeTensorView<double> view(R);
        check_agrees(view);
    }
    SECTION("TiledRuntimeTensor") {
        TiledRuntimeTensor<double> T{"T", std::vector<std::vector<int>>{{2, 3}, {3, 2}}};
        check_agrees(T);
    }
}

TEST_CASE("TensorOps - the deferred declaration hooks have table entries", "[ComputeGraph][TensorOps]") {
    // make_deferred_handle installs the two initializers unconditionally, and declare_tensor the
    // rank-specific resize and distribution hooks; the table carries all four for the type.
    cg::Graph graph("declared");
    auto     &D = graph.declare_tensor<double, 2>(std::string("D"), 4, 5);
    auto     &E = graph.declare_tensor<double, 2>(std::string("E"), 4, 5);

    cg::TensorHandle const *const hd = graph.find_tensor(graph.find_tensor_id_by_ptr(&D));
    cg::TensorHandle const *const he = graph.find_tensor(graph.find_tensor_id_by_ptr(&E));
    REQUIRE(hd != nullptr);
    REQUIRE(he != nullptr);
    REQUIRE(hd->zero_fn);
    REQUIRE(hd->random_fn);
    REQUIRE(hd->resize_deferred_fn);
    REQUIRE(hd->set_distribution_fn);
    REQUIRE(hd->ops->fill_zero != nullptr);
    REQUIRE(hd->ops->fill_random != nullptr);
    REQUIRE(hd->ops->resize_deferred != nullptr);
    REQUIRE(hd->ops->set_distribution != nullptr);

    // The same resize through the closure and through the table.
    std::vector<std::size_t> const local{2, 3};
    hd->resize_deferred_fn(local);
    he->ops->resize_deferred(he->live_ptr(), local);
    CHECK(D.dim(0) == E.dim(0));
    CHECK(D.dim(1) == E.dim(1));
    CHECK(E.dim(0) == 2);
    CHECK(E.dim(1) == 3);

    // The table's initializer materializes first, as the closure's does.
    REQUIRE_FALSE(E.is_materialized());
    he->ops->fill_zero(he->live_ptr());
    REQUIRE(E.is_materialized());
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(E(i, j) == 0.0);
        }
    }
}

TEST_CASE("TensorOps - a raw scalar has only the allreduce", "[ComputeGraph][TensorOps]") {
    double                 x = 2.0;
    cg::TensorHandle const h = cg::make_scalar_handle(&x, 0);
    CHECK(h.ops->allreduce_sum != nullptr);
    CHECK(h.ops->impl == nullptr);
    CHECK(h.ops->materialize == nullptr);
    CHECK(h.ops->name == nullptr);
}

TEST_CASE("TensorOps - after a rebind the table reaches the new tensor", "[ComputeGraph][TensorOps][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_random_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_ops");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }
    cg::TensorId const id = graph.find_tensor_id_by_ptr(&A1);
    REQUIRE(id != 0);

    // A rebind takes a tensor of the same type, so the table is the same one; what moves is the
    // object it is called on.
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    graph.rebind(A1, A2);

    cg::TensorHandle const *const h = graph.find_tensor(id);
    REQUIRE(h != nullptr);
    REQUIRE(h->ops->impl != nullptr);
    CHECK(h->ops->impl(h->live_ptr()) == static_cast<void *>(&A2.impl()));
    CHECK(h->ops->impl(h->live_ptr()) == h->impl_fn());
    CHECK(h->ops->name(h->live_ptr()) == "A2");
}
