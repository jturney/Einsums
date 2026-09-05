//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/ThcFactorization.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

ThcFactorization::ThcFactorization(std::string tag, RuntimeTensorView<double> three_index,
                                   std::vector<RuntimeTensorView<double>> collocations, double bound, double drop_threshold,
                                   std::string name)
    : _tag(std::move(tag)), _name(std::move(name)), _three_index(std::move(three_index)), _collocations(std::move(collocations)),
      _bound(bound), _drop_threshold(drop_threshold) {
    if (_collocations.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "ThcFactorization: at least one collocation matrix is required; a fit over no grid is nothing to offer");
    }
    if (!std::isfinite(drop_threshold) || drop_threshold < 0.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "ThcFactorization: the drop threshold must be finite and not negative; got {}. Zero is the bare "
                                "positivity guard, and there is nothing a negative one could mean",
                                drop_threshold);
    }
    if (!std::isfinite(bound) || bound < 0.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "ThcFactorization: the asserted relative error must be finite and not negative; got {}. Zero means take "
                                "einsums:graph:thc-epsilon",
                                bound);
    }
}

ThcFactorization::ThcFactorization(std::string tag, RuntimeTensorView<double> three_index, RuntimeTensorView<double> collocation,
                                   double bound, double drop_threshold, std::string name)
    : ThcFactorization(std::move(tag), std::move(three_index), std::vector<RuntimeTensorView<double>>{std::move(collocation)}, bound,
                       drop_threshold, std::move(name)) {
}

namespace {
/// A view of @p tensor carrying its name, which the implicit conversion drops.
template <typename TensorType>
RuntimeTensorView<double> named_view(TensorType const &tensor) {
    RuntimeTensorView<double> view{tensor};
    view.set_name(tensor.name());
    return view;
}
} // namespace

ThcFactorization::ThcFactorization(std::string tag, RuntimeTensor<double> const &three_index, RuntimeTensor<double> const &collocation,
                                   double bound, double drop_threshold, std::string name)
    : ThcFactorization(std::move(tag), RuntimeTensorView<double>{three_index}, RuntimeTensorView<double>{collocation}, bound,
                       drop_threshold, std::move(name)) {
}

ThcFactorization::ThcFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &collocation,
                                   double bound, double drop_threshold, std::string name)
    : ThcFactorization(std::move(tag), named_view(three_index), named_view(collocation), bound, drop_threshold, std::move(name)) {
}

namespace {
/// The caller's per-axis matrices as views, refusing a null before anything is moved.
std::vector<RuntimeTensorView<double>> views_of(std::vector<RuntimeTensor<double> const *> const &collocations) {
    std::vector<RuntimeTensorView<double>> out;
    out.reserve(collocations.size());
    for (auto const *matrix : collocations) {
        if (matrix == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "ThcFactorization: a collocation matrix is null");
        }
        out.emplace_back(*matrix);
    }
    return out;
}
} // namespace

ThcFactorization::ThcFactorization(std::string tag, RuntimeTensor<double> const &three_index,
                                   std::vector<RuntimeTensor<double> const *> collocations, double bound, double drop_threshold,
                                   std::string name)
    : ThcFactorization(std::move(tag), RuntimeTensorView<double>{three_index}, views_of(collocations), bound, drop_threshold,
                       std::move(name)) {
}

double ThcFactorization::epsilon() const {
    return _bound > 0.0 ? _bound : config::get(option::GraphThcEpsilon);
}

std::string ThcFactorization::grid_space_name() {
    return "grid";
}

std::string ThcFactorization::grid_dim_symbol() {
    return "ngrid";
}

SpaceId ThcFactorization::register_grid_space(Graph &graph) {
    // A grid is chosen per problem, so its extent is a SYMBOL rather than the number of points
    // a capture happened to have. That is what makes a saved graph replayed at a new geometry a
    // rebind, and it is what makes the factorization pass's numeric veto abstain over it.
    return graph.space_registry().register_space(make_index_space(grid_space_name(), "g", 0.0, GrowthClass::linear(), grid_dim_symbol()));
}

std::string ThcFactorization::residual_param_name(std::string const &provider, std::string const &tensor) {
    return fmt::format("{}.{}.residual_squared", provider, tensor);
}

std::string ThcFactorization::reference_param_name(std::string const &provider, std::string const &tensor) {
    return fmt::format("{}.{}.reference_squared", provider, tensor);
}

expected<FactorizationPlan, std::string> ThcFactorization::propose(Graph const &graph, TensorId tensor) const {
    TensorHandle const *handle = graph.find_tensor(tensor);
    if (handle == nullptr) {
        return unexpected(std::string{"the tagged tensor is not registered in this graph"});
    }
    if (handle->rank != 4) {
        return unexpected(fmt::format("a grid fit replaces a rank-4 tensor; this one is rank {}", handle->rank));
    }
    if (_collocations.size() != 1 && _collocations.size() != handle->rank) {
        return unexpected(fmt::format("there are {} collocation matrix/matrices for a rank-{} tensor; give one per axis or one for all",
                                      _collocations.size(), handle->rank));
    }

    // Which matrix each axis runs over. One matrix means every axis, which is what this fitted
    // before per-axis collocation existed and is still the commonest case.
    auto const per_axis = [&](std::size_t axis) -> RuntimeTensorView<double> const & {
        return _collocations.size() == 1 ? _collocations.front() : _collocations[axis];
    };

    // The two PAIRS must present the same blocks. The chain writes axes 0 and 1 against grid
    // letter P and axes 2 and 3 against Q, so a fit with one metric and one three-index tensor
    // needs axis 2 to run over the same block as axis 0 and axis 3 over the same as axis 1. An
    // [i,j,a,b] layout pairs occupied with occupied and virtual with virtual, which is a
    // different fit with two metrics rather than a spelling of this one.
    auto const same_matrix = [](RuntimeTensorView<double> const &left, RuntimeTensorView<double> const &right) {
        return left.data() == right.data() && left.dim(0) == right.dim(0) && left.dim(1) == right.dim(1);
    };
    if (!same_matrix(per_axis(0), per_axis(2)) || !same_matrix(per_axis(1), per_axis(3))) {
        return unexpected(std::string{"the two index pairs run over different blocks; axis 0 must pair with axis 2 and axis 1 with axis "
                                      "3, and a layout that does not needs two fits rather than one"});
    }

    RuntimeTensorView<double> const &row_matrix = per_axis(0);
    RuntimeTensorView<double> const &col_matrix = per_axis(1);

    std::size_t const naux      = _three_index.dim(0);
    std::size_t const row_basis = row_matrix.dim(0);
    std::size_t const col_basis = col_matrix.dim(0);
    std::size_t const grid      = row_matrix.dim(1);
    if (col_matrix.dim(1) != grid) {
        return unexpected(fmt::format("the collocation matrices carry {} and {} grid points; the chain contracts them against one grid",
                                      grid, col_matrix.dim(1)));
    }
    if (grid == 0) {
        return unexpected(std::string{"the collocation matrix has no grid points"});
    }
    if (_three_index.dim(1) != row_basis || _three_index.dim(2) != col_basis) {
        return unexpected(fmt::format("the three-index tensor is [{}, {}, {}] and the collocation matrices carry {} and {} basis "
                                      "functions",
                                      naux, _three_index.dim(1), _three_index.dim(2), row_basis, col_basis));
    }
    for (std::size_t axis = 0; axis < 4; ++axis) {
        if (handle->dims[axis] != per_axis(axis).dim(0)) {
            return unexpected(fmt::format("the tagged tensor is [{}] and axis {} is fitted over {} basis functions",
                                          fmt::join(handle->dims, ", "), axis, per_axis(axis).dim(0)));
        }
    }

    FactorizationPlan plan;
    plan.provider       = _name;
    plan.tagged_letters = {"m", "n", "p", "q"};

    // Five factors over as few tensors as the caller's blocks allow. The pass reads factors under
    // one name as one buffer, which is what keeps a collocation matrix from being stored twice
    // per pair; and the chain is what a plan naming a factor LIST exists for, since there is no
    // way to write it as a split of two.
    bool const        one_matrix  = _collocations.size() == 1 || same_matrix(row_matrix, col_matrix);
    std::string const row_name    = one_matrix ? "X" : "X_row";
    std::string const col_name    = one_matrix ? "X" : "X_col";
    std::string const grid_space  = grid_space_name();
    auto const collocation_factor = [&](std::string const &name, std::size_t basis, std::string basis_letter, std::string grid_letter) {
        return FactorTensor{.name    = name,
                            .letters = {std::move(basis_letter), std::move(grid_letter)},
                            .dims    = {basis, grid},
                            .spaces  = {std::string{}, grid_space},
                            .dtype   = packed_gemm::ScalarType::Float64};
    };
    plan.factors.push_back(collocation_factor(row_name, row_basis, "m", "P"));
    plan.factors.push_back(collocation_factor(col_name, col_basis, "n", "P"));
    plan.factors.push_back(FactorTensor{.name    = "Z",
                                        .letters = {"P", "Q"},
                                        .dims    = {grid, grid},
                                        .spaces  = {grid_space, grid_space},
                                        .dtype   = packed_gemm::ScalarType::Float64});
    plan.factors.push_back(collocation_factor(row_name, row_basis, "p", "Q"));
    plan.factors.push_back(collocation_factor(col_name, col_basis, "q", "Q"));

    plan.accuracy = make_approximation_record(_name, ApproximationEffect::NormRelative, epsilon(), epsilon());

    // The fitting, as nodes. Every step is a captured operation, so a bind that moves the
    // problem refits rather than replaying a grid fit computed once at optimize time.
    RuntimeTensorView<double> const three_index   = _three_index;
    RuntimeTensorView<double> const row_source    = row_matrix;
    RuntimeTensorView<double> const col_source    = col_matrix;
    double const                    threshold     = _drop_threshold;
    std::string const               residual_key  = residual_param_name(_name, handle->name);
    std::string const               reference_key = reference_param_name(_name, handle->name);

    plan.emit_setup = [three_index, row_source, col_source, one_matrix, naux, row_basis, col_basis, grid, threshold, residual_key,
                       reference_key](Graph &parent, Graph &body, std::vector<TensorId> const &factors) {
        auto *X_row = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);
        auto *X_col = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[1]).tensor_ptr);
        auto *Z     = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[2]).tensor_ptr);

        // The fitting's own workspace, declared on the BODY: it is live only while the fit runs
        // and nothing outside has any use for the eigenvectors of a collocation overlap.
        auto &row_gram  = body.declare_runtime_tensor<double>("thc_gram", {grid, grid}, /*intermediate=*/true);
        auto &col_gram  = one_matrix ? row_gram : body.declare_runtime_tensor<double>("thc_gram_col", {grid, grid}, /*intermediate=*/true);
        auto &metric    = body.declare_runtime_tensor<double>("thc_metric", {grid, grid}, /*intermediate=*/true);
        auto &vectors   = body.declare_runtime_tensor<double>("thc_vectors", {grid, grid}, /*intermediate=*/true);
        auto &values    = body.declare_runtime_tensor<double>("thc_values", {grid}, /*intermediate=*/true);
        auto &scaled    = body.declare_runtime_tensor<double>("thc_scaled", {grid, grid}, /*intermediate=*/true);
        auto &half      = body.declare_runtime_tensor<double>("thc_inv_sqrt", {grid, grid}, /*intermediate=*/true);
        auto &inverse   = body.declare_runtime_tensor<double>("thc_inverse", {grid, grid}, /*intermediate=*/true);
        auto &partial   = body.declare_runtime_tensor<double>("thc_partial", {naux, row_basis, grid}, /*intermediate=*/true);
        auto &projected = body.declare_runtime_tensor<double>("thc_projected", {naux, grid}, /*intermediate=*/true);
        auto &coupling  = body.declare_runtime_tensor<double>("thc_coupling", {grid, grid}, /*intermediate=*/true);
        auto &right     = body.declare_runtime_tensor<double>("thc_right", {grid, grid}, /*intermediate=*/true);
        auto &weights   = body.declare_runtime_tensor<double>("thc_weights", {naux, grid}, /*intermediate=*/true);
        auto &rebuilt   = body.declare_runtime_tensor<double>("thc_rebuilt", {naux, row_basis, grid}, /*intermediate=*/true);
        auto &residual  = body.declare_runtime_tensor<double>("thc_residual", {naux, row_basis, col_basis}, /*intermediate=*/true);

        // The residual's destinations. Plain scalars because that is what the pointer-writing
        // dot and write_param both take, and owned by the BODY so they outlive this call and
        // every replay after it.
        auto *residual_squared  = new double{0.0};
        auto *reference_squared = new double{0.0};
        body.adopt([residual_squared]() { delete residual_squared; });
        body.adopt([reference_squared]() { delete reference_squared; });

        {
            CaptureGuard const guard(body);

            // The two reduction destinations, registered under names of their own BEFORE the
            // dots that write them. A capture names an anonymous scalar destination after the
            // operation, so two dots in one body would present two interface tensors under one
            // name and the manifest would refuse the graph.
            CaptureContext::current().get_or_register_scalar(residual_squared, residual_key);
            CaptureContext::current().get_or_register_scalar(reference_squared, reference_key);

            // The collocation matrices are the caller's, copied into the factors the pass
            // declared, which is what makes them interface tensors of the transformed graph: a
            // bind at a new geometry hands over new matrices and everything below refits from
            // them. One copy when both pairs run over one block, two when they do not.
            permute("m,P <- m,P", 0.0, X_row, 1.0, row_source);
            if (!one_matrix) {
                permute("n,P <- n,P", 0.0, X_col, 1.0, col_source);
            }

            // S[P,Q] = (X_row^T X_row)[P,Q] * (X_col^T X_col)[P,Q], elementwise. A Hadamard
            // product of Gram matrices, and therefore positive semi-definite by the Schur product
            // theorem, which is what lets the guarded inverse below be the same one a metric fit
            // uses. With one block on every axis this is the square this always formed.
            einsum("m,P ; m,Q -> P,Q", &row_gram, *X_row, *X_row);
            if (!one_matrix) {
                einsum("n,P ; n,Q -> P,Q", &col_gram, *X_col, *X_col);
            }
            direct_product(1.0, row_gram, col_gram, 0.0, &metric);

            // S^{-1} through S^{-1/2} squared, so the guard is applied once, to the eigenvalues,
            // exactly as the metric fit applies it. A copy first, because syev destroys what it
            // decomposes.
            permute("P,Q <- P,Q", 0.0, &vectors, 1.0, metric);
            syev(&vectors, &values);
            element_transform(&values, "inv_sqrt_or_zero", threshold);
            einsum("P,R ; R -> P,R", &scaled, vectors, values);
            einsum("P,R ; Q,R -> P,Q", &half, scaled, vectors);
            einsum("P,R ; R,Q -> P,Q", &inverse, half, half);

            // B~[A,P] = sum_mn B[A,m,n] X_row[m,P] X_col[n,P]. The second contraction carries P
            // through both operands and the output, which is the grid index riding along rather
            // than being summed: that is what makes the fit a fit of one function per grid point.
            einsum("A,m,n ; n,P -> A,m,P", &partial, three_index, *X_col);
            einsum("A,m,P ; m,P -> A,P", &projected, partial, *X_row);

            // Z = S^{-1} (B~^T B~) S^{-1}.
            einsum("A,P ; A,Q -> P,Q", &coupling, projected, projected);
            einsum("P,R ; R,Q -> P,Q", &right, coupling, inverse);
            einsum("P,R ; R,Q -> P,Q", Z, inverse, right);

            // What the fit is worth, measured on every bind rather than asserted once. The
            // least-squares weights are C = B~ S^{-1}, the fitted three-index tensor is
            // C[A,P] X_row[m,P] X_col[n,P], and the residual against B is a quantity this library
            // can form where the four-index error is one it cannot.
            einsum("A,R ; R,P -> A,P", &weights, projected, inverse);
            einsum("A,P ; m,P -> A,m,P", &rebuilt, weights, *X_row);
            permute("A,m,n <- A,m,n", 0.0, &residual, 1.0, three_index);
            einsum("A,m,P ; n,P -> A,m,n", 1.0, &residual, -1.0, rebuilt, *X_col);
            dot(residual_squared, residual, residual);
            dot(reference_squared, three_index, three_index);
            write_param(residual_key, *residual_squared);
            write_param(reference_key, *reference_squared);
        }

        // The workspace is declared and left DEFERRED. Materializing it here would bake a
        // resource decision into structure, which is the mistake the metric fit's own comment
        // records; the resource phase places it, inside this body, where a fitting's scratch
        // belongs.
    };

    return plan;
}

EINSUMS_NAMESPACE_END(compute_graph)
