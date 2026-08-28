//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/MetricFitFactorization.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <fmt/format.h>

#include <cmath>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

MetricFitFactorization::MetricFitFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &metric,
                                               double bound, std::string name)
    : _tag(std::move(tag)), _name(std::move(name)), _three_index(&three_index), _metric(&metric), _bound(bound) {
}

std::string MetricFitFactorization::dropped_param_name(std::string const &provider, std::string const &tensor) {
    return fmt::format("{}.{}.dropped_directions", provider, tensor);
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

    plan.accuracy = make_approximation_record(_name, ApproximationEffect::NormRelative, _bound, _bound);

    // The fitting, as nodes. Every step is a captured operation, so a bind that moves the
    // problem refits rather than replaying a metric inverse computed once at optimize time.
    Tensor<double, 3> const *three_index = _three_index;
    Tensor<double, 2> const *metric      = _metric;
    std::string const        dropped_key = dropped_param_name(_name, handle->name);
    plan.emit_setup                      = [three_index, metric, naux, rows, cols, dropped_key](Graph &parent, Graph &body,
                                                                           std::vector<TensorId> const &factors) {
        auto *b = static_cast<RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);

        // The fitting's own workspace, declared on the BODY: it is live only while the
        // fitting runs, and nothing outside has any use for the eigenvectors of a metric.
        auto &vectors = body.declare_runtime_tensor<double>("metric_vectors", {naux, naux}, /*intermediate=*/true);
        auto &values  = body.declare_runtime_tensor<double>("metric_values", {naux}, /*intermediate=*/true);
        auto &scaled  = body.declare_runtime_tensor<double>("metric_scaled", {naux, naux}, /*intermediate=*/true);
        auto &half    = body.declare_runtime_tensor<double>("metric_inv_sqrt", {naux, naux}, /*intermediate=*/true);
        auto &dropped = body.declare_runtime_tensor<double>("metric_dropped_flags", {naux}, /*intermediate=*/true);

        // The counter's destination. A plain scalar rather than a tensor because that is what
        // the pointer-writing dot and write_param both take, and owned by the BODY because it
        // has to outlive this call and every replay after it.
        auto *count = new double{0.0};
        body.adopt([count]() { delete count; });

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

            // How many auxiliary directions this fit threw away, counted on every refit
            // rather than once, because it is a property of the metric that is bound now.
            // The entries are exactly zero only where inv_sqrt_or_zero assigned zero, so the
            // indicator is reading that guard's decision rather than testing a float for
            // equality in the usual mistaken way.
            //
            // The dot of the indicator with ITSELF is the count, since every entry is 0 or 1.
            // A dot against a vector of ones would say the same thing and need a fifth tensor
            // declared, materialized and filled to say it.
            permute("R <- R", 0.0, &dropped, 1.0, values);
            element_transform(&dropped, "is_zero");
            dot(count, dropped, dropped);
            write_param(dropped_key, *count);
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
