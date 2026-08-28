//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/MetricFitFactorization.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <fmt/format.h>

#include <cmath>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// @f$B = J^{-1/2} R@f$, computed eagerly, for the error measurement only.
///
/// The graph recomputes this on every bind through the nodes @ref MetricFitFactorization
/// captures; this copy exists so the provider can state what its approximation actually costs
/// rather than what a caller guessed it would.
Tensor<double, 3> fit_eagerly(Tensor<double, 3> const &three_index, Tensor<double, 2> const &metric) {
    std::size_t const naux = metric.dim(0);
    std::size_t const rows = three_index.dim(1);
    std::size_t const cols = three_index.dim(2);

    auto scratch           = Tensor<double, 2>("metric copy", naux, naux);
    scratch                = metric;
    auto [vectors, values] = linear_algebra::syev(scratch);

    for (std::size_t i = 0; i < naux; ++i) {
        values(i) = values(i) > 0.0 ? 1.0 / std::sqrt(values(i)) : 0.0;
    }

    // half[P,Q] = sum_R U[P,R] s[R] U[Q,R]
    auto half = Tensor<double, 2>("metric inverse square root", naux, naux);
    half.zero();
    for (std::size_t p = 0; p < naux; ++p) {
        for (std::size_t q = 0; q < naux; ++q) {
            double sum = 0.0;
            for (std::size_t r = 0; r < naux; ++r) {
                sum += vectors(p, r) * values(r) * vectors(q, r);
            }
            half(p, q) = sum;
        }
    }

    auto fitted = Tensor<double, 3>("fitted", naux, rows, cols);
    fitted.zero();
    for (std::size_t q = 0; q < naux; ++q) {
        for (std::size_t m = 0; m < rows; ++m) {
            for (std::size_t n = 0; n < cols; ++n) {
                double sum = 0.0;
                for (std::size_t p = 0; p < naux; ++p) {
                    sum += half(q, p) * three_index(p, m, n);
                }
                fitted(q, m, n) = sum;
            }
        }
    }
    return fitted;
}

} // namespace

MetricFitFactorization::MetricFitFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &metric,
                                               std::optional<double> declared_bound, std::string name)
    : _tag(std::move(tag)), _name(std::move(name)), _three_index(&three_index), _metric(&metric), _declared(declared_bound) {
}

expected<FactorizationPlan, std::string> MetricFitFactorization::propose(Graph const &graph, TensorId tensor) const {
    TensorHandle const *handle = graph.find_tensor(tensor);
    if (handle == nullptr) {
        return unexpected(std::string{"the tagged tensor is not registered in this graph"});
    }
    if (handle->rank != 4) {
        return unexpected(fmt::format("a metric fit replaces a rank-4 tensor; this one is rank {}", handle->rank));
    }

    std::size_t const naux = _metric->dim(0);
    std::size_t const rows = _three_index->dim(1);
    std::size_t const cols = _three_index->dim(2);
    if (_metric->dim(0) != _metric->dim(1) || _three_index->dim(0) != naux) {
        return unexpected(std::string{"the metric is not square over the three-index tensor's first axis"});
    }
    if (handle->dims[0] != rows || handle->dims[1] != cols || handle->dims[2] != rows || handle->dims[3] != cols) {
        return unexpected(fmt::format("the tagged tensor is [{}] and the fit produces [{}, {}, {}, {}]", fmt::join(handle->dims, ", "),
                                      rows, cols, rows, cols));
    }

    FactorizationPlan plan;
    plan.provider       = _name;
    plan.tagged_letters = {"m", "n", "p", "q"};

    // ONE tensor, offered twice with different letters. The pass reads two factors under one
    // name as one buffer, which is what keeps the fitting from being done and stored twice.
    FactorTensor factor;
    factor.name    = "B";
    factor.letters = {"Q", "m", "n"};
    factor.dims    = {naux, rows, cols};
    factor.dtype   = packed_gemm::ScalarType::Float64;
    plan.factors.push_back(factor);
    factor.letters = {"Q", "p", "q"};
    plan.factors.push_back(factor);

    double bound = 0.0;
    if (_declared.has_value()) {
        bound     = *_declared;
        _measured = -1.0;
    } else {
        // The measurement. Forms the fitted product once, which is the same order of work as
        // one contraction against the tensor this replaces, and buys a record that says what
        // the approximation cost rather than what someone expected it to.
        auto const  fitted = fit_eagerly(*_three_index, *_metric);
        auto const &source = *static_cast<Tensor<double, 4> const *>(handle->tensor_ptr);

        double error_sq = 0.0;
        double norm_sq  = 0.0;
        for (std::size_t m = 0; m < rows; ++m) {
            for (std::size_t n = 0; n < cols; ++n) {
                for (std::size_t p = 0; p < rows; ++p) {
                    for (std::size_t q = 0; q < cols; ++q) {
                        double sum = 0.0;
                        for (std::size_t aux = 0; aux < naux; ++aux) {
                            sum += fitted(aux, m, n) * fitted(aux, p, q);
                        }
                        double const exact_value = source(m, n, p, q);
                        double const delta       = sum - exact_value;
                        error_sq += delta * delta;
                        norm_sq += exact_value * exact_value;
                    }
                }
            }
        }
        _measured = norm_sq > 0.0 ? std::sqrt(error_sq / norm_sq) : std::sqrt(error_sq);
        bound     = _measured;
    }

    plan.accuracy = make_approximation_record(_name, ApproximationEffect::NormRelative, _declared.value_or(bound), bound);

    // The fitting, as nodes. Every step is a captured operation, so a bind that moves the
    // problem refits rather than replaying a metric inverse computed once at optimize time.
    Tensor<double, 3> const *three_index = _three_index;
    Tensor<double, 2> const *metric      = _metric;
    plan.emit_setup = [three_index, metric, naux, rows, cols](Graph &parent, Graph &body, std::vector<TensorId> const &factors) {
        auto *b = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);

        // The fitting's own workspace, declared on the BODY: it is live only while the
        // fitting runs, and nothing outside has any use for the eigenvectors of a metric.
        auto &vectors = body.declare_runtime_tensor<double>("metric_vectors", {naux, naux}, /*intermediate=*/true);
        auto &values  = body.declare_runtime_tensor<double>("metric_values", {naux}, /*intermediate=*/true);
        auto &scaled  = body.declare_runtime_tensor<double>("metric_scaled", {naux, naux}, /*intermediate=*/true);
        auto &half    = body.declare_runtime_tensor<double>("metric_inv_sqrt", {naux, naux}, /*intermediate=*/true);

        {
            CaptureGuard const guard(body);
            // A copy, because syev destroys what it decomposes and the metric is the caller's.
            permute("P,Q <- P,Q", 0.0, &vectors, 1.0, *metric);
            syev(&vectors, &values);
            element_transform(&values, "inv_sqrt_or_zero");
            // Column scaling, spelled as the contraction that sums over nothing.
            einsum("P,R ; R -> P,R", &scaled, vectors, values);
            einsum("P,R ; Q,R -> P,Q", &half, scaled, vectors);
            einsum("Q,P ; P,m,n -> Q,m,n", b, half, *three_index);
        }

        // The workspace is declared and left DEFERRED. Materializing it here would look like
        // the provider taking responsibility for the storage it declared, and it was: a
        // Materialize node carries an allocating closure, a closure is the one thing a file
        // cannot hold, and allocation is a resource decision the design re-derives on load
        // rather than saving. Doing it at capture baked a resource decision into structure and
        // left the fitting unsaveable, which is the one thing a factorization exists to avoid.
        //
        // The resource phase places it now, inside this body, where a fitting's scratch
        // belongs. See passes::Materialization.
    };

    return plan;
}

EINSUMS_NAMESPACE_END(compute_graph)
