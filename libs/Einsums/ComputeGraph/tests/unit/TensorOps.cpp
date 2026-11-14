//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The per-type operations table a handle points at. For every capturable type each entry exists
// exactly when the type has the method behind it and gives the method's answer; these cases pin
// that, and that after a rebind the table reaches the new tensor.

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

/// Every entry exists exactly when the type has the method behind it, and the ones that answer a
/// question answer it as the method does.
template <typename TensorType>
void check_agrees(TensorType &tensor) {
    cg::TensorHandle const h   = cg::make_handle(tensor, 0);
    cg::TensorOps const   &ops = *h.ops;
    void *const            obj = h.live_ptr();
    REQUIRE(obj == static_cast<void *>(&tensor));

    if constexpr (requires { tensor.impl(); }) {
        REQUIRE(ops.impl != nullptr);
        CHECK(ops.impl(obj) == static_cast<void *>(&tensor.impl()));
    } else {
        CHECK(ops.impl == nullptr);
    }
    if constexpr (requires { tensor.is_materialized(); }) {
        REQUIRE(ops.is_materialized != nullptr);
        CHECK(ops.is_materialized(obj) == tensor.is_materialized());
    } else {
        CHECK(ops.is_materialized == nullptr);
    }
    CHECK((ops.materialize != nullptr) == requires { tensor.materialize(); });
    CHECK((ops.release != nullptr) == requires { tensor.release(); });
    if constexpr (requires {
                      tensor.release();
                      tensor.storage().get();
                  }) {
        REQUIRE(ops.storage != nullptr);
        CHECK(ops.storage(obj) == tensor.storage().get());
    } else {
        CHECK(ops.storage == nullptr);
    }
    CHECK(ops.allreduce_sum != nullptr);
    CHECK(ops.iallreduce_sum != nullptr);
    if constexpr (requires { tensor.symmetry(); }) {
        REQUIRE(ops.declared_symmetry != nullptr);
        auto const *own   = tensor.symmetry();
        auto const  table = ops.declared_symmetry(obj);
        REQUIRE((own == nullptr) == (table == nullptr));
        if (own != nullptr) {
            CHECK(*own == *table);
        }
    } else {
        CHECK(ops.declared_symmetry == nullptr);
    }
    CHECK((ops.begin_local_view != nullptr) == (ops.end_local_view != nullptr));
    REQUIRE(ops.name != nullptr);
    CHECK(ops.name(obj) == tensor.name());
    CHECK(ops.name(obj) == h.name);
}

} // namespace

TEST_CASE("TensorOps - the table answers as each capturable type's own methods do", "[ComputeGraph][TensorOps]") {
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

TEST_CASE("TensorOps - a declared tensor's table resizes, distributes and fills it", "[ComputeGraph][TensorOps]") {
    cg::Graph graph("declared");
    auto     &E = graph.declare_tensor<double, 2>(std::string("E"), 4, 5);

    cg::TensorHandle const *const he = graph.find_tensor(graph.find_tensor_id_by_ptr(&E));
    REQUIRE(he != nullptr);
    CHECK(he->resizable);
    REQUIRE(he->ops->fill_zero != nullptr);
    REQUIRE(he->ops->fill_random != nullptr);
    REQUIRE(he->ops->resize_deferred != nullptr);
    REQUIRE(he->ops->set_distribution != nullptr);

    std::vector<std::size_t> const local{2, 3};
    he->ops->resize_deferred(he->live_ptr(), local);
    CHECK(E.dim(0) == 2);
    CHECK(E.dim(1) == 3);

    // The initializer materializes first: a deferred tensor has no storage until it runs.
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
    CHECK(h->ops->name(h->live_ptr()) == "A2");
}
