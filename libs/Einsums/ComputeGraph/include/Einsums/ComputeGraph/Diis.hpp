//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Pulay DIIS extrapolation over a set of (amplitude, step) tensor pairs.
 *
 * Accelerates the ``t <- t + step`` update a fixed-point iteration performs by keeping a short history of (amplitude, step) snapshots
 * and replacing the amplitudes with the least-squares extrapolant. The error vector is the update step itself, the standard
 * coupled-cluster choice, so an iteration whose step the body computes anyway gets DIIS for the cost of a K-sized host-side solve.
 *
 * One @ref step call does the whole update: the snapshot copies, the one new row of the B matrix, the normalized bordered solve, and the
 * extrapolation. Everything tensor-sized is an einsums operation - the snapshots and the extrapolation are ``axpby`` (the amplitudes
 * update IN PLACE, so pointers a graph captured stay valid), the B entries are ``true_dot`` inner products (conjugated, so complex
 * dtypes are exact and the coefficients are real), and the (K+1)-sized system solves with ``gesv``.
 *
 * The B entries are CACHED across steps, keyed by snapshot identity: a step adds one snapshot, so only one row is new. The per-solve
 * copy is normalized by its largest element before the solve, and a singular B drops the oldest history pair and retries.
 *
 * @par Operand binding
 * @ref add_pair binds to the storage the operand holds AT THAT MOMENT, the way a view does. Rebinding or resizing an amplitude or a step
 * tensor afterwards leaves the accelerator reading the old buffer; the caller must add pairs whose storage outlives the accelerator.
 * Adding a pair once a history exists is rejected, because the existing snapshots are shaped for the old pair list.
 *
 * @tparam T Element type of the amplitudes. The coefficients are real regardless.
 */
template <typename T>
// clang-format off
class APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_NOCOPY
APIARY_NOMOVE
APIARY_HOLDER(std::shared_ptr)
APIARY_INSTANTIATE_AS("DiisAcceleratorF", DiisAccelerator<float>)
APIARY_INSTANTIATE_AS("DiisAcceleratorD", DiisAccelerator<double>)
APIARY_INSTANTIATE_AS("DiisAcceleratorC", DiisAccelerator<std::complex<float>>)
APIARY_INSTANTIATE_AS("DiisAcceleratorZ", DiisAccelerator<std::complex<double>>)
    DiisAccelerator {
    // clang-format on
  public:
    /// A history of at least 2 is required: one snapshot has nothing to extrapolate between.
    APIARY_EXPOSE explicit DiisAccelerator(int k = 8) : _k{static_cast<size_t>(k)} {
        if (k < 2) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "DIIS needs a history of at least 2, got k={}", k);
        }
    }

    DiisAccelerator(DiisAccelerator const &)            = delete;
    DiisAccelerator &operator=(DiisAccelerator const &) = delete;

    /**
     * @brief Register one ``(amplitude, step)`` pair.
     *
     * @param amplitude The tensor the iteration updates, written in place by @ref step.
     * @param step      The update the body computed, which is also the error vector.
     * @param amp_id    Capture-time registrar returning the amplitude's TensorId, for @ref diis_step to order its node on.
     * @param step_id   Capture-time registrar for the step tensor's TensorId.
     */
    void add_pair(RuntimeTensorView<T> amplitude, RuntimeTensorView<T> step, std::function<TensorId()> amp_id,
                  std::function<TensorId()> step_id) {
        if (!_history.empty()) {
            EINSUMS_THROW_EXCEPTION(std::logic_error,
                                    "DiisAccelerator::add_pair: the history already holds {} snapshots shaped for the "
                                    "current pair list; add every pair before the first step",
                                    _history.size());
        }
        _amplitudes.push_back(std::move(amplitude));
        _steps.push_back(std::move(step));
        _amplitude_ids.push_back(std::move(amp_id));
        _step_ids.push_back(std::move(step_id));
    }

    /// Push the current (amplitudes, steps) onto the history and extrapolate the amplitudes in place.
    APIARY_EXPOSE void step() {
        LabeledSection("DiisAccelerator::step");
        if (_amplitudes.empty()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "DiisAccelerator::step: no (amplitude, step) pairs were registered");
        }

        snapshot();
        extrapolate();
    }

    /// Snapshots currently held, which ramps to @ref max_history and can drop on a singular subspace.
    APIARY_EXPOSE APIARY_GETTER("history_size") [[nodiscard]] size_t history_size() const { return _history.size(); }

    /// The history depth the accelerator was constructed with.
    APIARY_EXPOSE APIARY_GETTER("max_history") [[nodiscard]] size_t max_history() const { return _k; }

    /// The number of registered ``(amplitude, step)`` pairs.
    APIARY_EXPOSE APIARY_GETTER("num_pairs") [[nodiscard]] size_t num_pairs() const { return _amplitudes.size(); }

    /// Forget every snapshot, keeping the registered pairs. The next step starts the history over.
    APIARY_EXPOSE void reset() {
        _history.clear();
        _dot_cache.clear();
    }

    /// Capture-time registrars for the amplitudes, in registration order.
    [[nodiscard]] std::vector<std::function<TensorId()>> const &amplitude_ids() const { return _amplitude_ids; }

    /// Capture-time registrars for the step tensors, in registration order.
    [[nodiscard]] std::vector<std::function<TensorId()>> const &step_ids() const { return _step_ids; }

  private:
    /// One history entry: a copy of every pair's amplitude and step, plus the identity the B cache keys on.
    struct Snapshot {
        std::vector<std::unique_ptr<RuntimeTensor<T>>> t;
        std::vector<std::unique_ptr<RuntimeTensor<T>>> e;
        uint64_t                                       id{0};
    };

    static double real_part(T const &v) {
        if constexpr (IsComplexV<T>) {
            return static_cast<double>(v.real());
        } else {
            return static_cast<double>(v);
        }
    }

    static std::vector<size_t> dims_of(RuntimeTensorView<T> const &t) {
        std::vector<size_t> dims(t.rank());
        for (size_t d = 0; d < dims.size(); d++) {
            dims[d] = t.dim(static_cast<int>(d));
        }
        return dims;
    }

    /// Copy the live (amplitudes, steps) into a new history entry, evicting and RECYCLING the oldest when the history is full.
    void snapshot() {
        size_t const n = _amplitudes.size();

        // A full history hands its evicted tensors to the new snapshot instead of allocating: the axpby below overwrites every element,
        // so a fresh allocation would pay for a zero fill nothing reads.
        std::vector<std::unique_ptr<RuntimeTensor<T>>> recycled_t, recycled_e;
        if (_history.size() >= _k) {
            recycled_t = std::move(_history.front().t);
            recycled_e = std::move(_history.front().e);
            _history.erase(_history.begin());
        }

        Snapshot snap;
        snap.t.reserve(n);
        snap.e.reserve(n);
        for (size_t c = 0; c < n; c++) {
            auto st = recycled_t.empty() ? std::make_unique<RuntimeTensor<T>>("diis amplitude", dims_of(_amplitudes[c]))
                                         : std::move(recycled_t[c]);
            linear_algebra::axpby(static_cast<T>(1.0), _amplitudes[c], static_cast<T>(0.0), st.get());
            auto se = recycled_e.empty() ? std::make_unique<RuntimeTensor<T>>("diis step", dims_of(_steps[c])) : std::move(recycled_e[c]);
            linear_algebra::axpby(static_cast<T>(1.0), _steps[c], static_cast<T>(0.0), se.get());
            snap.t.push_back(std::move(st));
            snap.e.push_back(std::move(se));
        }
        snap.id = _next_id++;
        _history.push_back(std::move(snap));

        for (auto it = _dot_cache.begin(); it != _dot_cache.end();) {
            if (!is_live(it->first.first) || !is_live(it->first.second)) {
                it = _dot_cache.erase(it);
            } else {
                ++it;
            }
        }
    }

    [[nodiscard]] bool is_live(uint64_t id) const {
        for (auto const &snap : _history) {
            if (snap.id == id) {
                return true;
            }
        }
        return false;
    }

    /// ``B[p,q] = sum over components of Re<e_p, e_q>``, from the cache where it can be.
    double inner_product(size_t p, size_t q) {
        auto key = std::make_pair(_history[p].id, _history[q].id);
        if (auto it = _dot_cache.find(key); it != _dot_cache.end()) {
            return it->second;
        }
        double v = 0.0;
        for (size_t c = 0; c < _history[p].e.size(); c++) {
            v += real_part(linear_algebra::true_dot(*_history[p].e[c], *_history[q].e[c]));
        }
        _dot_cache.emplace(key, v);
        return v;
    }

    /// Solve the bordered least-squares system and write the extrapolant over the amplitudes.
    void extrapolate() {
        while (_history.size() >= 2) {
            size_t const m = _history.size();

            // The augmented matrix is symmetric all the way through the border, so the fill is layout-agnostic: element (p,q) and (q,p)
            // get the same value whichever of the two the storage order names first.
            RuntimeTensor<double> B("diis B", std::vector<size_t>{m + 1, m + 1});
            double               *b_data = B.data();
            auto                  at     = [&](size_t p, size_t q) -> double                      &{ return b_data[p * (m + 1) + q]; };

            double scale = 0.0;
            for (size_t p = 0; p < m; p++) {
                for (size_t q = p; q < m; q++) {
                    double const v = inner_product(p, q);
                    at(p, q) = at(q, p) = v;
                    scale               = std::max(scale, std::abs(v));
                }
            }
            if (scale > 0.0) {
                for (size_t p = 0; p < m; p++) {
                    for (size_t q = 0; q < m; q++) {
                        at(p, q) /= scale;
                    }
                }
            }
            for (size_t p = 0; p < m; p++) {
                at(p, m) = at(m, p) = -1.0;
            }
            at(m, m) = 0.0;

            RuntimeTensor<double> rhs("diis rhs", std::vector<size_t>{m + 1});
            rhs.zero();
            rhs.data()[m] = -1.0;

            try {
                std::ignore = linear_algebra::gesv(&B, &rhs);
            } catch (std::runtime_error const &) {
                // Singular subspace: forget the oldest pair and retry on what is left.
                _history.erase(_history.begin());
                continue;
            }

            double const *coeff = rhs.data();
            for (size_t c = 0; c < _amplitudes.size(); c++) {
                linear_algebra::axpby(static_cast<T>(coeff[0]), *_history[0].t[c], static_cast<T>(0.0), &_amplitudes[c]);
                for (size_t s = 1; s < m; s++) {
                    linear_algebra::axpby(static_cast<T>(coeff[s]), *_history[s].t[c], static_cast<T>(1.0), &_amplitudes[c]);
                }
            }
            return;
        }
    }

    size_t _k;

    std::vector<RuntimeTensorView<T>>      _amplitudes;
    std::vector<RuntimeTensorView<T>>      _steps;
    std::vector<std::function<TensorId()>> _amplitude_ids;
    std::vector<std::function<TensorId()>> _step_ids;

    std::vector<Snapshot> _history;
    uint64_t              _next_id{0};

    /// Inner products of live snapshots, keyed by the pair of snapshot identities. Rebuilding these from scratch every step was measured
    /// as most of the accelerator's cost; a step adds one snapshot, so only its row is new.
    std::map<std::pair<uint64_t, uint64_t>, double> _dot_cache;
};

EINSUMS_NAMESPACE_END(compute_graph)
