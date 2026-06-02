//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace einsums::compute_graph {

/**
 * @brief Owns tensors that persist across multiple Graphs and Pipelines.
 *
 * Workspace is the outermost scoping layer for tensor lifetime:
 * - **Workspace** — cross-computation (ERI, MO coefficients, basis data)
 * - **Pipeline** — cross-stage (Fock matrix, density, amplitudes)
 * - **Graph** — single-computation intermediates (temporaries)
 *
 * Tensors declared via Workspace use deferred allocation: they have
 * valid metadata (name, dims, dtype) but no backing data until the
 * MaterializationPass runs. This allows the DistributionPlanningPass
 * to decide how to partition large tensors across MPI ranks before
 * any memory is allocated.
 *
 * @par Example
 * @code
 * namespace cg = einsums::compute_graph;
 *
 * cg::Workspace ws("h2o");
 *
 * // Declare tensors — no data allocated yet
 * auto &eri = ws.declare_tensor<double, 4>("ERI", nao, nao, nao, nao);
 * auto &C   = ws.declare_zero_tensor<double, 2>("C", nao, nmo);
 *
 * // Use in a pipeline
 * cg::Pipeline scf("scf");
 * scf.set_workspace(ws);
 * {
 *     auto &stage = scf.add_stage("compute");
 *     cg::CaptureGuard guard(stage);
 *     cg::einsum("ijkl;kl->ij", 0.0, &F, 1.0, eri, D);
 * }
 *
 * scf.apply(pm);   // Passes decide distribution, then materialize
 * scf.execute();    // Computation runs on materialized tensors
 *
 * // eri and C survive — reuse in MP2, CCSD, etc.
 * @endcode
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE Workspace {
  public:
    APIARY_EXPOSE explicit Workspace(std::string name) : _name(std::move(name)) {}

    ~Workspace() = default;

    // Non-copyable, movable
    Workspace(Workspace const &)            = delete;
    Workspace &operator=(Workspace const &) = delete;
    Workspace(Workspace &&)                 = default;
    Workspace &operator=(Workspace &&)      = default;

    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string const &name() const { return _name; }

    /**
     * @brief Declare a tensor with deferred allocation. No initialization.
     *
     * @tparam T     Element type.
     * @tparam Rank  Number of dimensions.
     * @param  name  Human-readable tensor name.
     * @param  dims  Dimensions of each rank.
     * @return Reference to the shell tensor (valid object, no data yet).
     */
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_tensor(std::string tensor_name, Dims... dims) {
        using TensorType = Tensor<T, Rank>;
        auto *ptr        = new TensorType(typename TensorType::DeferredAlloc{}, std::move(tensor_name), dims...);
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<TensorType *>(p); });

        auto handle            = make_handle(*ptr, 0);
        handle.alloc_state     = AllocState::Deferred;
        handle.is_intermediate = false;
        handle.materialize_fn  = [ptr]() { ptr->materialize(); };
        handle.zero_fn         = [ptr]() {
            ptr->materialize();
            ptr->zero();
        };
        handle.random_fn = [ptr]() {
            ptr->materialize();
            // Fill with random data
            auto *data = ptr->data();
            for (size_t idx = 0; idx < ptr->size(); idx++) {
                // NOLINTNEXTLINE(misc-predictable-rand)
                data[idx] = static_cast<T>(static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0);
            }
        };
        _handles.push_back(std::move(handle));

        return *ptr;
    }

    /**
     * @brief Declare a tensor initialized to zero after materialization.
     */
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_zero_tensor(std::string tensor_name, Dims... dims) {
        auto &t                   = declare_tensor<T, Rank>(tensor_name, dims...);
        _handles.back().init_kind = InitKind::Zero;
        // Tag the tensor itself so the same init kind reaches any Graph
        // that captures this tensor later — make_handle reads
        // tensor.pending_init() and populates the new handle's
        // init_kind / zero_fn. Without this the workspace's canonical
        // handle would have init metadata, but each per-graph handle
        // built from this tensor via capture would not, so the
        // Materialization pass couldn't emit an Initialize node for
        // body-resident workspace tensors.
        t.set_pending_init(PendingInit::Zero);
        return t;
    }

    /**
     * @brief Declare a tensor initialized with random values after materialization.
     */
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_random_tensor(std::string tensor_name, Dims... dims) {
        auto &t                   = declare_tensor<T, Rank>(tensor_name, dims...);
        _handles.back().init_kind = InitKind::Random;
        t.set_pending_init(PendingInit::Random);
        return t;
    }

    /**
     * @brief Runtime-rank analog of declare_tensor().
     *
     * Same deferred-allocation lifecycle as the typed version (no data until
     * materialize_fn fires) but the rank is carried at runtime. Suitable for
     * Python-bound workflows or any workflow that wants a single dispatch
     * surface across ranks.
     */
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &declare_runtime_tensor(std::string tensor_name, std::vector<size_t> dims) {
        using TensorType = GeneralRuntimeTensor<T, Alloc>;
        auto *ptr        = new TensorType(typename TensorType::DeferredAlloc{}, std::move(tensor_name), std::move(dims));
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<TensorType *>(p); });

        auto handle            = make_handle(*ptr, 0);
        handle.alloc_state     = AllocState::Deferred;
        handle.is_intermediate = false;
        handle.materialize_fn  = [ptr]() { ptr->materialize(); };
        handle.zero_fn         = [ptr]() {
            ptr->materialize();
            ptr->zero();
        };
        handle.random_fn = [ptr]() {
            ptr->materialize();
            auto *data = ptr->data();
            for (size_t idx = 0; idx < ptr->size(); idx++) {
                // NOLINTNEXTLINE(misc-predictable-rand)
                data[idx] = static_cast<T>(static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0);
            }
        };
        _handles.push_back(std::move(handle));

        return *ptr;
    }

    /// Runtime-rank analog of declare_zero_tensor().
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &declare_zero_runtime_tensor(std::string tensor_name, std::vector<size_t> dims) {
        auto &t                   = declare_runtime_tensor<T, Alloc>(std::move(tensor_name), std::move(dims));
        _handles.back().init_kind = InitKind::Zero;
        t.set_pending_init(PendingInit::Zero);
        return t;
    }

    /// Runtime-rank analog of declare_random_tensor().
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("declare_random_tensor", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("declare_random_tensor", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("declare_random_tensor", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("declare_random_tensor", T = std::complex<double>,
                                             Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &declare_random_runtime_tensor(std::string tensor_name, std::vector<size_t> dims) {
        auto &t                   = declare_runtime_tensor<T, Alloc>(std::move(tensor_name), std::move(dims));
        _handles.back().init_kind = InitKind::Random;
        t.set_pending_init(PendingInit::Random);
        return t;
    }

    /// Access all declared tensor handles (for passes to inspect).
    [[nodiscard]] std::vector<TensorHandle> const &tensor_handles() const { return _handles; }

    /// Mutable access for passes that modify handles (e.g., DistributionPlanning).
    [[nodiscard]] std::vector<TensorHandle> &tensor_handles() { return _handles; }

    /// Number of declared tensors.
    APIARY_EXPOSE APIARY_GETTER("size") [[nodiscard]] size_t size() const { return _handles.size(); }

    /// Materialize and initialize all deferred tensors.
    APIARY_EXPOSE void materialize_all() {
        for (auto &h : _handles) {
            if (h.alloc_state == AllocState::Deferred) {
                switch (h.init_kind) {
                case InitKind::Zero:
                    if (h.zero_fn)
                        h.zero_fn();
                    break;
                case InitKind::Random:
                    if (h.random_fn)
                        h.random_fn();
                    break;
                default:
                    if (h.materialize_fn)
                        h.materialize_fn();
                    break;
                }
                h.alloc_state = AllocState::Materialized;
            }
        }
    }

  private:
    std::string                                          _name;
    std::vector<std::unique_ptr<void, void (*)(void *)>> _owned_tensors;
    std::vector<TensorHandle>                            _handles;
};

} // namespace einsums::compute_graph
