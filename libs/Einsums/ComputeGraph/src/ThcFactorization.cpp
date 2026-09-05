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
#include <optional>
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
/// @p view under a name of its own, for a fit that reads the tensor the caller TAGGED.
///
/// A fitting is captured, so whatever it reads becomes an interface tensor of the transformed
/// graph, bound by name at every later bind. When the fit reads a tensor the caller's own algebra
/// also holds, that is TWO interface tensors over one buffer: the pass and the fitting each hold
/// their own handle for it, and a manifest that binds by name refuses two entries sharing one.
/// Naming the fitting's copy is what makes the pair expressible, and it says what it is: a caller
/// rebinding the graph supplies the same tensor under both names, and both handles are repointed.
/// Without it a grid-fitted graph whose fit reads its own tagged tensor could not be saved at all.
RuntimeTensorView<double> fit_input_view(RuntimeTensorView<double> view) {
    std::string const stem = view.name();
    view.set_name(fmt::format("{}@fit", stem.empty() ? std::string{"tagged"} : stem));
    return view;
}

} // namespace

ThcFactorization::ThcFactorization(std::string tag, RuntimeTensor<double> const &three_index,
                                   std::vector<RuntimeTensor<double> const *> collocations, double bound, double drop_threshold,
                                   std::string name)
    : ThcFactorization(std::move(tag), RuntimeTensorView<double>{three_index}, views_of(collocations), bound, drop_threshold,
                       std::move(name)) {
}

std::shared_ptr<ThcFactorization> ThcFactorization::for_amplitude(std::string tag, RuntimeTensorView<double> amplitude,
                                                                  std::vector<RuntimeTensorView<double>> collocations, double bound,
                                                                  double drop_threshold, std::string name) {
    // The primary constructor takes a three-index tensor to fit FROM; an amplitude fit has none,
    // because what it fits from is the tagged tensor itself. An empty view is what says so, and
    // `propose` reads it as the mode rather than as a missing argument.
    auto provider = std::make_shared<ThcFactorization>(std::move(tag), RuntimeTensorView<double>{}, std::move(collocations), bound,
                                                       drop_threshold, std::move(name));
    provider->_amplitude.emplace(fit_input_view(std::move(amplitude)));
    return provider;
}

std::shared_ptr<ThcFactorization> ThcFactorization::for_amplitude(std::string tag, RuntimeTensor<double> const &amplitude,
                                                                  std::vector<RuntimeTensor<double> const *> collocations, double bound,
                                                                  double drop_threshold, std::string name) {
    return for_amplitude(std::move(tag), named_view(amplitude), views_of(collocations), bound, drop_threshold, std::move(name));
}

std::shared_ptr<ThcFactorization> ThcFactorization::for_three_index(std::string tag, RuntimeTensorView<double> three_index,
                                                                    std::vector<RuntimeTensorView<double>> collocations, double bound,
                                                                    double drop_threshold, std::string name) {
    auto provider = std::make_shared<ThcFactorization>(std::move(tag), fit_input_view(std::move(three_index)), std::move(collocations),
                                                       bound, drop_threshold, std::move(name));
    provider->_fits_three_index = true;
    return provider;
}

std::shared_ptr<ThcFactorization> ThcFactorization::for_three_index(std::string tag, RuntimeTensor<double> const &three_index,
                                                                    std::vector<RuntimeTensor<double> const *> collocations, double bound,
                                                                    double drop_threshold, std::string name) {
    return for_three_index(std::move(tag), named_view(three_index), views_of(collocations), bound, drop_threshold, std::move(name));
}

void ThcFactorization::report_residual_into(RuntimeTensorView<double> residual, RuntimeTensorView<double> reference) {
    _residual_report.emplace(std::move(residual));
    _reference_report.emplace(std::move(reference));
}

void ThcFactorization::report_residual_into(RuntimeTensor<double> const &residual, RuntimeTensor<double> const &reference) {
    report_residual_into(named_view(residual), named_view(reference));
}

void ThcFactorization::report_residual_into(Tensor<double, 1> const &residual, Tensor<double, 1> const &reference) {
    report_residual_into(named_view(residual), named_view(reference));
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

namespace {

/// Finish a plan whose coupling is fitted from the TAGGED tensor.
///
/// The metric is the same Hadamard product of collocation Gram matrices the integral fit uses,
/// and so is the guarded inverse; what differs is where the projected quantity comes from. The
/// projection contracts the tagged tensor against a collocation matrix one axis at a time, with
/// the grid letter riding along on the second of each pair, which is what makes the result one
/// function per grid point rather than a sum over them.
FactorizationPlan amplitude_plan(FactorizationPlan plan, TensorHandle const &handle, RuntimeTensorView<double> row_source,
                                 RuntimeTensorView<double> col_source, RuntimeTensorView<double> amplitude,
                                 std::optional<RuntimeTensorView<double>> residual_report,
                                 std::optional<RuntimeTensorView<double>> reference_report, bool one_matrix, std::size_t grid,
                                 double threshold) {
    std::vector<std::size_t> const dims{handle.dims};

    plan.emit_setup = [row_source, col_source, amplitude, residual_report, reference_report, one_matrix, grid, threshold,
                       dims](Graph &parent, Graph &body, std::vector<TensorId> const &factors) {
        auto *X_row = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);
        auto *X_col = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[1]).tensor_ptr);
        auto *Z     = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[2]).tensor_ptr);

        // Named after the graph they are declared in, because an amplitude fit is emitted TWICE,
        // once into a setup body and once into the loop body that refits it, and the storage
        // auditor keys its duplicate check on the name rather than on the buffer.
        auto const named = [&body](std::string_view stem) { return fmt::format("{}_{}", body.name(), stem); };

        auto &row_gram = body.declare_runtime_tensor<double>(named("thc_gram"), {grid, grid}, /*intermediate=*/true);
        auto &col_gram =
            one_matrix ? row_gram : body.declare_runtime_tensor<double>(named("thc_gram_col"), {grid, grid}, /*intermediate=*/true);
        auto &metric    = body.declare_runtime_tensor<double>(named("thc_metric"), {grid, grid}, /*intermediate=*/true);
        auto &vectors   = body.declare_runtime_tensor<double>(named("thc_vectors"), {grid, grid}, /*intermediate=*/true);
        auto &values    = body.declare_runtime_tensor<double>(named("thc_values"), {grid}, /*intermediate=*/true);
        auto &scaled    = body.declare_runtime_tensor<double>(named("thc_scaled"), {grid, grid}, /*intermediate=*/true);
        auto &half      = body.declare_runtime_tensor<double>(named("thc_inv_sqrt"), {grid, grid}, /*intermediate=*/true);
        auto &inverse   = body.declare_runtime_tensor<double>(named("thc_inverse"), {grid, grid}, /*intermediate=*/true);
        auto &first     = body.declare_runtime_tensor<double>(named("thc_proj1"), {grid, dims[1], dims[2], dims[3]}, true);
        auto &second    = body.declare_runtime_tensor<double>(named("thc_proj2"), {grid, dims[2], dims[3]}, true);
        auto &third     = body.declare_runtime_tensor<double>(named("thc_proj3"), {grid, grid, dims[3]}, true);
        auto &projected = body.declare_runtime_tensor<double>(named("thc_projected"), {grid, grid}, /*intermediate=*/true);
        auto &right     = body.declare_runtime_tensor<double>(named("thc_right"), {grid, grid}, /*intermediate=*/true);

        bool const measure = residual_report.has_value() && reference_report.has_value();

        {
            CaptureGuard const guard(body);

            permute("m,P <- m,P", 0.0, X_row, 1.0, row_source);
            if (!one_matrix) {
                permute("n,P <- n,P", 0.0, X_col, 1.0, col_source);
            }

            einsum("m,P ; m,Q -> P,Q", &row_gram, *X_row, *X_row);
            if (!one_matrix) {
                einsum("n,P ; n,Q -> P,Q", &col_gram, *X_col, *X_col);
            }
            direct_product(1.0, row_gram, col_gram, 0.0, &metric);

            permute("P,Q <- P,Q", 0.0, &vectors, 1.0, metric);
            syev(&vectors, &values);
            element_transform(&values, "inv_sqrt_or_zero", threshold);
            einsum("P,R ; R -> P,R", &scaled, vectors, values);
            einsum("P,R ; Q,R -> P,Q", &half, scaled, vectors);
            einsum("P,R ; R,Q -> P,Q", &inverse, half, half);

            // P[P,Q] = sum_mnpq X_row[m,P] X_col[n,P] T[m,n,p,q] X_row[p,Q] X_col[q,Q], one axis
            // at a time, with the grid letter riding along on the second of each pair.
            einsum("m,P ; m,n,p,q -> P,n,p,q", &first, *X_row, amplitude);
            einsum("n,P ; P,n,p,q -> P,p,q", &second, *X_col, first);
            einsum("P,p,q ; p,Q -> P,Q,q", &third, second, *X_row);
            einsum("P,Q,q ; q,Q -> P,Q", &projected, third, *X_col);

            einsum("P,R ; R,Q -> P,Q", &right, projected, inverse);
            einsum("P,R ; R,Q -> P,Q", Z, inverse, right);

            // What the fit is worth on this bind, measured rather than asserted, and the one
            // place this provider differs from its integral half. There the exact quantity is
            // the four-index tensor the fit exists to avoid forming, so the bound is a claim;
            // here the exact quantity is the tensor being fitted and it is already in hand, so
            // the error is a number. Forming the fitted tensor to subtract is affordable for the
            // same reason: it is the size of a tensor that is already allocated. Emitted only
            // where a caller asked for the measurement, since it is arithmetic the rewrite
            // otherwise exists to avoid.
            if (measure) {
                auto &pair = body.declare_runtime_tensor<double>(named("thc_pair"), {grid, dims[0], dims[1]}, /*intermediate=*/true);
                auto &carried_pair = body.declare_runtime_tensor<double>(named("thc_carried_pair"), {grid, dims[0], dims[1]}, true);
                auto &difference   = body.declare_runtime_tensor<double>(named("thc_difference"), dims, /*intermediate=*/true);
                einsum("m,P ; n,P -> P,m,n", &pair, *X_row, *X_col);
                einsum("P,Q ; P,m,n -> Q,m,n", &carried_pair, *Z, pair);
                permute("m,n,p,q <- m,n,p,q", 0.0, &difference, 1.0, amplitude);
                einsum("Q,m,n ; Q,p,q -> m,n,p,q", 1.0, &difference, -1.0, carried_pair, pair);
                // The caller's own rank-1 tensors, written through their buffers: the pointer
                // form is the only dot this layer has, and the destinations are materialized
                // because a caller supplied them.
                dot(const_cast<double *>(residual_report->data()), difference, difference);
                dot(const_cast<double *>(reference_report->data()), amplitude, amplitude);
            }
        }
    };
    return plan;
}

/// The fitting a three-index plan carries, which is the four-index fitting stopped one step
/// earlier: the weights @c C are what the four-index form squares to make @c Z.
FactorizationPlan three_index_plan(FactorizationPlan plan, RuntimeTensorView<double> three_index, RuntimeTensorView<double> row_source,
                                   RuntimeTensorView<double> col_source, bool one_matrix, std::size_t naux, std::size_t row_basis,
                                   std::size_t col_basis, std::size_t grid, double threshold, std::string residual_key,
                                   std::string reference_key) {
    plan.emit_setup = [three_index, row_source, col_source, one_matrix, naux, row_basis, col_basis, grid, threshold, residual_key,
                       reference_key](Graph &parent, Graph &body, std::vector<TensorId> const &factors) {
        auto *X_row = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);
        auto *X_col = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[1]).tensor_ptr);
        auto *C     = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[2]).tensor_ptr);

        // Named after the graph they are declared in, because one program may hold several of
        // these fittings and the storage auditor keys its duplicate check on the name.
        auto const named = [&body](std::string_view stem) { return fmt::format("{}_{}", body.name(), stem); };

        auto &row_gram = body.declare_runtime_tensor<double>(named("thc3_gram"), {grid, grid}, /*intermediate=*/true);
        auto &col_gram =
            one_matrix ? row_gram : body.declare_runtime_tensor<double>(named("thc3_gram_col"), {grid, grid}, /*intermediate=*/true);
        auto &metric   = body.declare_runtime_tensor<double>(named("thc3_metric"), {grid, grid}, /*intermediate=*/true);
        auto &vectors  = body.declare_runtime_tensor<double>(named("thc3_vectors"), {grid, grid}, /*intermediate=*/true);
        auto &values   = body.declare_runtime_tensor<double>(named("thc3_values"), {grid}, /*intermediate=*/true);
        auto &scaled   = body.declare_runtime_tensor<double>(named("thc3_scaled"), {grid, grid}, /*intermediate=*/true);
        auto &half     = body.declare_runtime_tensor<double>(named("thc3_inv_sqrt"), {grid, grid}, /*intermediate=*/true);
        auto &inverse  = body.declare_runtime_tensor<double>(named("thc3_inverse"), {grid, grid}, /*intermediate=*/true);
        auto &partial  = body.declare_runtime_tensor<double>(named("thc3_partial"), {naux, row_basis, grid}, /*intermediate=*/true);
        auto &weighted = body.declare_runtime_tensor<double>(named("thc3_weighted"), {naux, grid}, /*intermediate=*/true);
        auto &rebuilt  = body.declare_runtime_tensor<double>(named("thc3_rebuilt"), {naux, row_basis, grid}, /*intermediate=*/true);
        auto &residual = body.declare_runtime_tensor<double>(named("thc3_residual"), {naux, row_basis, col_basis}, /*intermediate=*/true);

        auto *residual_squared  = new double{0.0};
        auto *reference_squared = new double{0.0};
        body.adopt([residual_squared]() { delete residual_squared; });
        body.adopt([reference_squared]() { delete reference_squared; });

        {
            CaptureGuard const guard(body);

            CaptureContext::current().get_or_register_scalar(residual_squared, residual_key);
            CaptureContext::current().get_or_register_scalar(reference_squared, reference_key);

            permute("m,P <- m,P", 0.0, X_row, 1.0, row_source);
            if (!one_matrix) {
                permute("n,P <- n,P", 0.0, X_col, 1.0, col_source);
            }

            einsum("m,P ; m,Q -> P,Q", &row_gram, *X_row, *X_row);
            if (!one_matrix) {
                einsum("n,P ; n,Q -> P,Q", &col_gram, *X_col, *X_col);
            }
            direct_product(1.0, row_gram, col_gram, 0.0, &metric);

            permute("P,Q <- P,Q", 0.0, &vectors, 1.0, metric);
            syev(&vectors, &values);
            element_transform(&values, "inv_sqrt_or_zero", threshold);
            einsum("P,R ; R -> P,R", &scaled, vectors, values);
            einsum("P,R ; Q,R -> P,Q", &half, scaled, vectors);
            einsum("P,R ; R,Q -> P,Q", &inverse, half, half);

            // B~[A,P] = sum_mn B[A,m,n] X_row[m,P] X_col[n,P], then C = B~ S^{-1}.
            einsum("A,m,n ; n,P -> A,m,P", &partial, three_index, *X_col);
            einsum("A,m,P ; m,P -> A,P", &weighted, partial, *X_row);
            einsum("A,R ; R,P -> A,P", C, weighted, inverse);

            // What the fit is worth, measured rather than asserted, because the exact quantity
            // here IS the tagged tensor and it is in hand.
            einsum("A,P ; m,P -> A,m,P", &rebuilt, *C, *X_row);
            permute("A,m,n <- A,m,n", 0.0, &residual, 1.0, three_index);
            einsum("A,m,P ; n,P -> A,m,n", 1.0, &residual, -1.0, rebuilt, *X_col);
            dot(residual_squared, residual, residual);
            dot(reference_squared, three_index, three_index);
            write_param(residual_key, *residual_squared);
            write_param(reference_key, *reference_squared);
        }
    };
    return plan;
}

} // namespace

expected<FactorizationPlan, std::string> ThcFactorization::propose(Graph const &graph, TensorId tensor) const {
    TensorHandle const *handle = graph.find_tensor(tensor);
    if (handle == nullptr) {
        return unexpected(std::string{"the tagged tensor is not registered in this graph"});
    }

    if (_fits_three_index) {
        if (handle->rank != 3) {
            return unexpected(fmt::format("this fit replaces a rank-3 tensor; this one is rank {}", handle->rank));
        }
        if (_collocations.size() != 1 && _collocations.size() != 2) {
            return unexpected(fmt::format("there are {} collocation matrix/matrices for the two basis axes of a rank-3 tensor; give one "
                                          "per basis axis or one for both",
                                          _collocations.size()));
        }
        if (handle->data_ptr != static_cast<void const *>(_three_index.data())) {
            return unexpected(std::string{"the tagged tensor is not the one this provider was given to fit"});
        }
        RuntimeTensorView<double> const &row_matrix = _collocations.front();
        RuntimeTensorView<double> const &col_matrix = _collocations.size() == 1 ? _collocations.front() : _collocations[1];
        std::size_t const                grid       = row_matrix.dim(1);
        if (col_matrix.dim(1) != grid) {
            return unexpected(fmt::format("the collocation matrices carry {} and {} grid points; the chain contracts them against one grid",
                                          grid, col_matrix.dim(1)));
        }
        if (grid == 0) {
            return unexpected(std::string{"the collocation matrix has no grid points"});
        }
        if (handle->dims[1] != row_matrix.dim(0) || handle->dims[2] != col_matrix.dim(0)) {
            return unexpected(fmt::format("the tagged tensor is [{}] and its basis axes are fitted over {} and {} basis functions",
                                          fmt::join(handle->dims, ", "), row_matrix.dim(0), col_matrix.dim(0)));
        }

        bool const one_matrix =
            _collocations.size() == 1 ||
            (row_matrix.data() == col_matrix.data() && row_matrix.dim(0) == col_matrix.dim(0) && row_matrix.dim(1) == col_matrix.dim(1));
        std::string const grid_space = grid_space_name();

        FactorizationPlan plan;
        plan.provider         = _name;
        plan.tagged_letters   = {"A", "m", "n"};
        plan.fits_from_tagged = true;
        plan.factors.push_back(FactorTensor{.name    = one_matrix ? "X" : "X_row",
                                            .letters = {"m", "P"},
                                            .dims    = {row_matrix.dim(0), grid},
                                            .spaces  = {std::string{}, grid_space},
                                            .dtype   = packed_gemm::ScalarType::Float64});
        plan.factors.push_back(FactorTensor{.name    = one_matrix ? "X" : "X_col",
                                            .letters = {"n", "P"},
                                            .dims    = {col_matrix.dim(0), grid},
                                            .spaces  = {std::string{}, grid_space},
                                            .dtype   = packed_gemm::ScalarType::Float64});
        plan.factors.push_back(FactorTensor{.name    = "C",
                                            .letters = {"A", "P"},
                                            .dims    = {handle->dims[0], grid},
                                            .spaces  = {std::string{}, grid_space},
                                            .dtype   = packed_gemm::ScalarType::Float64});
        plan.accuracy = make_approximation_record(_name, ApproximationEffect::NormRelative, epsilon(), epsilon(), {}, {}, "",
                                                  ApproximationOrigin::Measured, residual_param_name(_name, handle->name));
        return three_index_plan(std::move(plan), _three_index, row_matrix, col_matrix, one_matrix, handle->dims[0], row_matrix.dim(0),
                                col_matrix.dim(0), grid, _drop_threshold, residual_param_name(_name, handle->name),
                                reference_param_name(_name, handle->name));
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

    bool const        amplitude_mode = _three_index.rank() == 0;
    std::size_t const naux           = amplitude_mode ? 0 : _three_index.dim(0);
    std::size_t const row_basis      = row_matrix.dim(0);
    std::size_t const col_basis      = col_matrix.dim(0);
    std::size_t const grid           = row_matrix.dim(1);
    if (col_matrix.dim(1) != grid) {
        return unexpected(fmt::format("the collocation matrices carry {} and {} grid points; the chain contracts them against one grid",
                                      grid, col_matrix.dim(1)));
    }
    if (grid == 0) {
        return unexpected(std::string{"the collocation matrix has no grid points"});
    }
    if (amplitude_mode) {
        // The caller hands over the very tensor they tag, and this is where that claim is
        // checked rather than believed: a fit OF a tensor is only sound if it reads the tensor
        // the pass is about to substitute away.
        if (!_amplitude.has_value() || _amplitude->rank() != 4) {
            return unexpected(std::string{"an amplitude fit needs the rank-4 tensor it fits, and none was handed over"});
        }
        if (handle->data_ptr != static_cast<void const *>(_amplitude->data())) {
            return unexpected(std::string{"the tagged tensor is not the one this provider was given to fit"});
        }
    } else if (_three_index.dim(1) != row_basis || _three_index.dim(2) != col_basis) {
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

    // The bound is ASSERTED, because a record is written once and a record cannot hold a number a
    // particular bind found. The record names the parameter the fitting writes instead, so a
    // caller holding a record can read what this bind's fit was actually worth; inside a loop
    // body that parameter is rewritten at every refit, and what stands in it when the solver
    // stops is the last iteration's fit.
    // The bound is ASSERTED, because a record is written once and cannot hold a number a
    // particular bind found. The record names the tensor the fitting writes its squared residual
    // into instead, so a caller holding a record can read what this bind's fit was worth; inside
    // a loop body that tensor is rewritten at every refit, and what stands in it when the solver
    // stops is the last iteration's fit.
    plan.accuracy = make_approximation_record(_name, ApproximationEffect::NormRelative, epsilon(), epsilon(), {}, {}, "",
                                              ApproximationOrigin::Asserted, residual_param_name(_name, handle->name));

    if (amplitude_mode) {
        plan.fits_from_tagged = true;
        // MEASURED here, where the integral fit's is asserted, and the difference is what the fit
        // reads: the exact quantity is the tagged tensor and it is in hand, so the residual is a
        // real number rather than a claim. It still never forms the fitted tensor; see the header.
        plan.accuracy = make_approximation_record(_name, ApproximationEffect::NormRelative, epsilon(), epsilon(), {}, {}, "",
                                                  ApproximationOrigin::Measured,
                                                  _residual_report.has_value() ? _residual_report->name() : std::string{});
        return amplitude_plan(std::move(plan), *handle, row_matrix, col_matrix, *_amplitude, _residual_report, _reference_report,
                              one_matrix, grid, _drop_threshold);
    }

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
        //
        // Named after the graph they are declared in, because one program may hold SEVERAL of
        // these fittings and the storage auditor keys its duplicate check on the name: two fits
        // under one set of names present it with one tensor materialized twice.
        auto const named = [&body](std::string_view stem) { return fmt::format("{}_{}", body.name(), stem); };

        auto &row_gram = body.declare_runtime_tensor<double>(named("thc_gram"), {grid, grid}, /*intermediate=*/true);
        auto &col_gram =
            one_matrix ? row_gram : body.declare_runtime_tensor<double>(named("thc_gram_col"), {grid, grid}, /*intermediate=*/true);
        auto &metric    = body.declare_runtime_tensor<double>(named("thc_metric"), {grid, grid}, /*intermediate=*/true);
        auto &vectors   = body.declare_runtime_tensor<double>(named("thc_vectors"), {grid, grid}, /*intermediate=*/true);
        auto &values    = body.declare_runtime_tensor<double>(named("thc_values"), {grid}, /*intermediate=*/true);
        auto &scaled    = body.declare_runtime_tensor<double>(named("thc_scaled"), {grid, grid}, /*intermediate=*/true);
        auto &half      = body.declare_runtime_tensor<double>(named("thc_inv_sqrt"), {grid, grid}, /*intermediate=*/true);
        auto &inverse   = body.declare_runtime_tensor<double>(named("thc_inverse"), {grid, grid}, /*intermediate=*/true);
        auto &partial   = body.declare_runtime_tensor<double>(named("thc_partial"), {naux, row_basis, grid}, /*intermediate=*/true);
        auto &projected = body.declare_runtime_tensor<double>(named("thc_projected"), {naux, grid}, /*intermediate=*/true);
        auto &coupling  = body.declare_runtime_tensor<double>(named("thc_coupling"), {grid, grid}, /*intermediate=*/true);
        auto &right     = body.declare_runtime_tensor<double>(named("thc_right"), {grid, grid}, /*intermediate=*/true);
        auto &weights   = body.declare_runtime_tensor<double>(named("thc_weights"), {naux, grid}, /*intermediate=*/true);
        auto &rebuilt   = body.declare_runtime_tensor<double>(named("thc_rebuilt"), {naux, row_basis, grid}, /*intermediate=*/true);
        auto &residual  = body.declare_runtime_tensor<double>(named("thc_residual"), {naux, row_basis, col_basis}, true);

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
