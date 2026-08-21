//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

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
 * extrapolation. The amplitudes update IN PLACE, so pointers a graph captured stay valid; the B entries are conjugated inner products,
 * so complex dtypes are exact and the coefficients are real; and the (K+1)-sized system solves with ``gesv``.
 *
 * @par Why the interior is BLAS and memcpy rather than tensor operations
 * Every tensor here is dense, contiguous, of a known dtype, and mostly owned by this object, so the eager tensor layer has nothing left
 * to decide - and its per-call cost is what dominates at the sizes a local-correlation solver produces. A DLPNO-(T) iteration hands the
 * accelerator several hundred pairs, which puts a step near ten thousand operations on operands of a few thousand elements each; through
 * the eager layer each one pays a profiling section, rank and shape checks, a vectorability query and a dispatch, all of which cost more
 * than the arithmetic they guard. So the hot path calls the BLAS wrappers and @c std::copy_n on raw pointers directly. The operations
 * are the SAME ones the eager layer would have reached - it lowers a contiguous ``axpby`` to @c scal plus @c axpy and a contiguous
 * ``true_dot`` to @c dot / @c dotc - which is what keeps the results bit for bit what they were. An operand that is NOT one flat vector
 * (a strided view) keeps the eager path, because that is the code that knows how to walk it.
 *
 * The B entries are CACHED across steps, keyed by snapshot identity: a step adds one snapshot, so only one row is new. Each entry sums
 * its components in registration order, one @c dot per component, which is what the cross-implementation bit-identity gate rests on: one
 * dot over the concatenated components would reduce in a different order. The per-solve copy of B is normalized by its largest element
 * before the solve, and a singular B drops the oldest history pair and retries.
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
        // The views are the general-path objects and the operands the flat-path handles; both are bound now so the hot path never asks
        // the tensor layer for either again.
        _amplitude_views.push_back(std::move(amplitude));
        _step_views.push_back(std::move(step));
        _amplitudes.push_back(bind_operand(_amplitude_views.back()));
        _steps.push_back(bind_operand(_step_views.back()));
        _amplitude_ids.push_back(std::move(amp_id));
        _step_ids.push_back(std::move(step_id));
    }

    /// Push the current (amplitudes, steps) onto the history and extrapolate the amplitudes in place.
    APIARY_EXPOSE void step() {
        LabeledSection("DiisAccelerator::step");
        if (_amplitudes.empty()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "DiisAccelerator::step: no (amplitude, step) pairs were registered");
        }

        SerialVendorScope const serial;
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
    /// Present ONE thread to the vendor for as long as it exists.
    ///
    /// Every kernel a step runs is BLAS level 1 over at most a few tens of thousands of elements, which a vendor that opens a thread team
    /// per call loses on outright: an OpenMP-threaded OpenBLAS forks and joins around each of the several thousand calls a (T)-scale step
    /// makes, and once the operands cross its threading threshold that measured eight times the cost of the same calls made serially.
    /// Width one is also the one value a caller may always present whatever other threads are doing - the vendor's global thread-count
    /// sync early-returns for it, which is the invariant @ref blas::set_moldable_width_scope documents - so this narrows nothing another
    /// thread depends on and needs no coordination.
    ///
    /// It also FIXES the reduction order of the B entries. A threaded @c dot sums per-thread partials, so its last bits depend on how
    /// many threads the vendor happened to use; measured on this machine a 13,824-element @c ddot differs between one thread and ten.
    /// The elementwise kernels here (@c copy, @c axpy) have no such dependence - each output element is one expression over its own
    /// inputs - so pinning the width changes nothing for them. For the dots it makes the accelerator's coefficients a function of the
    /// inputs alone, where before they were a function of the inputs and the machine's thread count.
    class SerialVendorScope {
      public:
        SerialVendorScope() {
            // Read both counts before writing either, for the reason the executor's width guard gives: a vendor count that tracks the
            // OpenMP ICV reads back the value just written otherwise, and the restore would then pin the thread to it.
            _prior_vendor = blas::get_num_threads_this_thread();
#ifdef _OPENMP
            _prior_omp = omp_get_max_threads();
            if (_prior_omp > 1) {
                omp_set_num_threads(1);
            }
#endif
            if (_prior_vendor > 1) {
                blas::set_num_threads_this_thread(1);
            }
        }

        ~SerialVendorScope() {
#ifdef _OPENMP
            if (_prior_omp > 1) {
                omp_set_num_threads(_prior_omp);
            }
#endif
            if (_prior_vendor > 1) {
                blas::set_num_threads_this_thread(_prior_vendor);
            }
        }

        SerialVendorScope(SerialVendorScope const &)            = delete;
        SerialVendorScope &operator=(SerialVendorScope const &) = delete;

      private:
        int _prior_vendor{0};
#ifdef _OPENMP
        int _prior_omp{1};
#endif
    };

    /// What the hot path needs to reach a registered operand without asking the tensor layer again.
    ///
    /// @c flat means the whole tensor is one unit-stride run, which is what lets a BLAS call or a @c memcpy stand in for a tensor
    /// operation; a strided view is not flat and keeps the eager path, for which the parallel view vectors hold the tensor object. This
    /// is deliberately trivial: it is copied whenever its vector grows, and a tensor view inside it would make that copy a throwing one.
    /// @c data points into the operand's STORAGE, so it survives the view vectors reallocating.
    struct Operand {
        T     *data{nullptr};
        size_t size{0};
        bool   flat{false};
    };

    /// One history entry: a copy of every pair's amplitude and step, plus the identity the B cache keys on.
    struct Snapshot {
        std::vector<std::unique_ptr<RuntimeTensor<T>>> t;
        std::vector<std::unique_ptr<RuntimeTensor<T>>> e;
        uint64_t                                       id{0};
    };

    /// Whether the raw-pointer paths may be used at all for this element type.
    static constexpr bool kRawPathAvailable = blas::IsBlasableV<T>;

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

    static Operand bind_operand(RuntimeTensorView<T> &view) {
        Operand operand;
        operand.size = view.size();
        size_t inc   = 1;
        // A unit increment is what both the copy and the BLAS calls below assume; a vectorable-but-strided operand is rare enough that
        // sending it down the eager path costs nothing worth the second code path.
        operand.flat = kRawPathAvailable && view.impl().is_totally_vectorable(&inc) && inc == 1;
        if (operand.flat) {
            operand.data = view.data();
        }
        return operand;
    }

    /// ``dst := src``. The eager form was ``axpby(1, src, 0, dst)``, which lowers to a zero fill and an ``axpy`` by one.
    static void copy_operand(Operand const &src, RuntimeTensorView<T> const &src_view, RuntimeTensor<T> &dst) {
        if (src.flat) {
            std::copy_n(src.data, src.size, dst.data());
        } else {
            linear_algebra::axpby(static_cast<T>(1.0), src_view, static_cast<T>(0.0), &dst);
        }
    }

    /// ``conj(a) . b`` over two snapshot buffers, which are always contiguous because this object allocated them.
    static T snapshot_dot(RuntimeTensor<T> const &a, RuntimeTensor<T> const &b) {
        if constexpr (kRawPathAvailable) {
            auto const n = static_cast<blas::int_t>(a.size());
            if constexpr (IsComplexV<T>) {
                return blas::dotc(n, a.data(), 1, b.data(), 1);
            } else {
                return blas::dot(n, a.data(), 1, b.data(), 1);
            }
        } else {
            return linear_algebra::true_dot(a, b);
        }
    }

    /// ``y := alpha * x``, one rounding per element, which is what ``axpby(alpha, x, 0, y)`` produced.
    static void assign_scaled(T alpha, T const *x, T *y, size_t n) {
        for (size_t i = 0; i < n; i++) {
            y[i] = alpha * x[i];
        }
    }

    /// Copy the live (amplitudes, steps) into a new history entry, evicting and RECYCLING the oldest when the history is full.
    void snapshot() {
        size_t const n = _amplitudes.size();

        // A full history hands its evicted tensors to the new snapshot instead of allocating: the copy below overwrites every element,
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
            auto st = recycled_t.empty() ? std::make_unique<RuntimeTensor<T>>("diis amplitude", dims_of(_amplitude_views[c]))
                                         : std::move(recycled_t[c]);
            copy_operand(_amplitudes[c], _amplitude_views[c], *st);
            auto se =
                recycled_e.empty() ? std::make_unique<RuntimeTensor<T>>("diis step", dims_of(_step_views[c])) : std::move(recycled_e[c]);
            copy_operand(_steps[c], _step_views[c], *se);
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
            v += real_part(snapshot_dot(*_history[p].e[c], *_history[q].e[c]));
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
                Operand const &amplitude = _amplitudes[c];
                if (amplitude.flat) {
                    auto const n = static_cast<blas::int_t>(amplitude.size);
                    assign_scaled(static_cast<T>(coeff[0]), _history[0].t[c]->data(), amplitude.data, amplitude.size);
                    for (size_t s = 1; s < m; s++) {
                        blas::axpy(n, static_cast<T>(coeff[s]), _history[s].t[c]->data(), 1, amplitude.data, 1);
                    }
                } else {
                    linear_algebra::axpby(static_cast<T>(coeff[0]), *_history[0].t[c], static_cast<T>(0.0), &_amplitude_views[c]);
                    for (size_t s = 1; s < m; s++) {
                        linear_algebra::axpby(static_cast<T>(coeff[s]), *_history[s].t[c], static_cast<T>(1.0), &_amplitude_views[c]);
                    }
                }
            }
            return;
        }
    }

    size_t _k;

    std::vector<RuntimeTensorView<T>>      _amplitude_views;
    std::vector<RuntimeTensorView<T>>      _step_views;
    std::vector<Operand>                   _amplitudes;
    std::vector<Operand>                   _steps;
    std::vector<std::function<TensorId()>> _amplitude_ids;
    std::vector<std::function<TensorId()>> _step_ids;

    std::vector<Snapshot> _history;
    uint64_t              _next_id{0};

    /// Inner products of live snapshots, keyed by the pair of snapshot identities. Rebuilding these from scratch every step was measured
    /// as most of the accelerator's cost; a step adds one snapshot, so only its row is new.
    std::map<std::pair<uint64_t, uint64_t>, double> _dot_cache;
};

EINSUMS_NAMESPACE_END(compute_graph)
