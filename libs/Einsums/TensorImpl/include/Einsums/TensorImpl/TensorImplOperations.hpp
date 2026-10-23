//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/StridedForEach.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(detail)

template <typename From, typename To>
constexpr To convert(From from) noexcept {
    if constexpr (IsComplexV<To> && IsComplexV<From>) {
        return To{static_cast<RemoveComplexT<To>>(std::real(from)), static_cast<RemoveComplexT<To>>(std::imag(from))};
    } else if constexpr (IsComplexV<To> && !IsComplexV<From>) {
        return To{static_cast<RemoveComplexT<To>>(from)};
    } else {
        return from;
    }
}

// ---------------------------------------------------------------------------
// Every elementwise operation walks its operands with strided_for_each_run, which hands over each
// maximal stretch the operands step through with fixed increments: the whole tensor when the operands
// are one vector in the same order, a stretch per column of a padded view, and a stretch along the
// written operand's fastest axis when the layouts differ. A long enough stretch goes to the BLAS
// routine where the types allow one; any other runs the element loop.
// ---------------------------------------------------------------------------

/// Stretches shorter than this take the element loop even where a vector routine applies: a call per
/// few elements costs more than the loop.
inline constexpr size_t vector_run_min = 16;

/// Whether a stretch of a walk over @p total elements goes to a vector routine: every operand stepping
/// alike, and the stretch long enough or the whole walk. Operands that step differently are laid out
/// differently, such as a transposed input beside its output, and there a call per stretch costs more
/// than the loop does. A stretch that is the whole walk is one call either way, and taking the vector
/// routine for it keeps a small tensor's result identical to the vector routine's on the same data.
template <typename R0, typename... Rs>
bool use_vector_routine(size_t n, size_t total, R0 const &first, Rs const &...rest) {
    return (n >= vector_run_min || n == total) && ((rest.inc == first.inc) && ...);
}

/// The library's element threshold for parallel loops, read on first use: a walk whose stretches all
/// go to vector routines, as a small dense tensor's single stretch does, never pays for the read.
class ParallelThreshold {
  public:
    size_t operator()() {
        if (!_known) {
            _value = ::einsums::hardware::omp_min_parallel_elements();
            _known = true;
        }
        return _value;
    }

  private:
    size_t _value = 0;
    bool   _known = false;
};

/// The parallel element loop over one stretch, apart from run_elements so that the parallel region's
/// outlined code does not keep the serial loop from inlining into every stretch.
template <typename F, typename... Rs>
void run_elements_parallel(size_t n, F const &f, Rs const &...r) {
    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < n; i++) {
        f(r.data[i * r.inc]...);
    }
}

/// The element loop over one stretch: @p f on element i of every operand, in parallel once the stretch
/// reaches the library's element threshold.
///
/// A shorter stretch runs a plain loop with no parallel region at all: entering one, even with a false
/// if clause, costs tens of nanoseconds, which dominates a walk of many short stretches. For the same
/// reason this is inlined into each stretch's handler: a call per stretch shows on short stretches.
template <typename F, typename... Rs>
EINSUMS_ALWAYS_INLINE void run_elements(size_t n, ParallelThreshold &parallel_min, F const &f, Rs const &...r) {
    if (n >= parallel_min()) [[unlikely]] {
        run_elements_parallel(n, f, r...);
    } else if (((r.inc == 1) && ...)) {
        EINSUMS_OMP_SIMD
        for (size_t i = 0; i < n; i++) {
            f(r.data[i]...);
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            f(r.data[i * r.inc]...);
        }
    }
}

/// Throws unless @p a and @p b have the same rank and extents; @p verb names the operation.
template <typename T, typename U>
void require_same_shape(TensorImpl<T> const &a, TensorImpl<U> const &b, char const *verb) {
    if (a.rank() != b.rank()) {
        EINSUMS_THROW_EXCEPTION(RankError, "Can not {} two tensors of different ranks!", verb);
    }
    if (a.dims() != b.dims()) {
        EINSUMS_THROW_EXCEPTION(DimensionError, "Can not {} two tensors with different sizes!", verb);
    }
}

[[noreturn]] inline void throw_complex_to_real() {
    EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                            "Can not convert complex to real! Please extract the components you want to use before operating.");
}

/// out = the real (@p Part 0) or imaginary (@p Part 1) part of in.
///
/// A complex value is its real part followed by its imaginary part, so the parts of a stretch are every
/// other real value: a strided copy.
template <int Part, typename T, typename TOther>
void impl_complex_part(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    require_same_shape(in, out, "copy");
    ParallelThreshold par;
    size_t const      total = out.size();
    strided_for_each_run(
        out.dims(),
        [&par, total](size_t n, RunOperand<T> o, RunOperand<TOther const> i) {
            if constexpr (blas::IsBlasableV<T> && std::is_same_v<std::remove_cv_t<TOther>, std::complex<T>>) {
                if (use_vector_routine(n, total, o, i)) {
                    blas::copy(n, reinterpret_cast<T const *>(i.data) + Part, 2 * i.inc, o.data, o.inc);
                    return;
                }
            }
            run_elements(n, par, [](T &out_v, TOther const &in_v) { out_v = Part == 0 ? std::real(in_v) : std::imag(in_v); }, o, i);
        },
        strided(out), strided(in));
}

template <typename T, typename TOther>
void impl_real(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();
    impl_complex_part<0>(in, out);
}

template <typename T, typename TOther>
void impl_imag(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();
    impl_complex_part<1>(in, out);
}

template <typename T, typename TOther>
void impl_abs(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();
    require_same_shape(in, out, "copy");
    ParallelThreshold par;
    strided_for_each_run(
        out.dims(),
        [&par](size_t n, RunOperand<T> o, RunOperand<TOther const> i) {
            run_elements(n, par, [](T &out_v, TOther const &in_v) { out_v = std::abs(in_v); }, o, i);
        },
        strided(out), strided(in));
}

template <typename T>
void impl_conj(TensorImpl<T> &x) {
    WAGGLE_ZONE_FUNC();

    if constexpr (IsComplexV<T>) {
        ParallelThreshold par;
        size_t const      total = x.size();
        strided_for_each_run(
            x.dims(),
            [&par, total](size_t n, RunOperand<T> r) {
                if constexpr (blas::IsBlasableV<T>) {
                    if (n >= vector_run_min || n == total) {
                        blas::lacgv(n, r.data, r.inc);
                        return;
                    }
                }
                run_elements(n, par, [](T &v) { v = std::conj(v); }, r);
            },
            strided(x));
    }
}

template <typename T, typename U>
void impl_scal(U alpha, TensorImpl<T> &out);

template <typename T, typename TOther>
void impl_copy(TensorImpl<TOther> const &in, TensorImpl<T> &out);

/// Whether @p a and @p b are the same operand: one start, one layout.
template <typename T, typename U>
bool impl_same_operand(TensorImpl<T> const &a, TensorImpl<U> const &b) {
    return static_cast<void const *>(a.data()) == static_cast<void const *>(b.data()) && a.strides() == b.strides();
}

/// Whether the storage @p a and @p b span, from first element to one past the last, intersects.
///
/// Conservative byte-interval intersection: interleaved disjoint slices only cost a copy.
template <typename T, typename U>
bool impl_storage_overlaps(TensorImpl<T> const &a, TensorImpl<U> const &b) {
    auto const bytes_of = [](auto const &t) {
        auto const *lo   = reinterpret_cast<unsigned char const *>(t.data());
        size_t      last = 0;
        for (size_t d = 0; d < t.rank(); d++) {
            last += (t.dim(d) - 1) * t.stride(d);
        }
        return std::pair{lo, lo + (last + 1) * sizeof(*t.data())};
    };
    if (a.size() == 0 || b.size() == 0) {
        return false;
    }
    auto const [a_lo, a_hi] = bytes_of(a);
    auto const [b_lo, b_hi] = bytes_of(b);
    return a_lo < b_hi && b_lo < a_hi;
}

/// Calls @p f with a packed copy of @p in, in @p in's own storage order.
///
/// For an input overlapping its output, which BLAS forbids and which would read values already written.
template <typename T, typename F>
void impl_with_snapshot(TensorImpl<T> const &in, F &&f) {
    using V = std::remove_cv_t<T>;
    std::vector<V> buffer(in.size());
    TensorImpl<V>  copy(buffer.data(), in.dims(), in.is_row_major());
    impl_copy(in, copy);
    std::forward<F>(f)(std::as_const(copy));
}

template <typename T, typename TOther, typename U>
void impl_axpy(U alpha, TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();
    require_same_shape(in, out, "add");

    // An operand that is its destination: out = (1 + alpha) * out, since ?axpy forbids the overlap.
    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>) {
        if (impl_same_operand(in, out)) {
            impl_scal(T{1} + static_cast<T>(alpha), out);
            return;
        }
    }

    // Any other overlap: read the input through a copy.
    if (impl_storage_overlaps(in, out)) {
        impl_with_snapshot(in, [&](auto const &snapshot) { impl_axpy(alpha, snapshot, out); });
        return;
    }

    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        throw_complex_to_real();
    } else {
        T const           a = static_cast<T>(alpha);
        ParallelThreshold par;
        size_t const      total = out.size();
        strided_for_each_run(
            out.dims(),
            [a, &par, total](size_t n, RunOperand<T> o, RunOperand<TOther const> i) {
                if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
                    if (use_vector_routine(n, total, o, i)) {
                        blas::axpy(n, a, i.data, i.inc, o.data, o.inc);
                        return;
                    }
                }
                run_elements(n, par, [a](T &out_v, TOther const &in_v) { out_v += a * convert<TOther, T>(in_v); }, o, i);
            },
            strided(out), strided(in));
    }
}

template <typename T, typename U>
void impl_scal(U alpha, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();

    // Scaling by zero assigns zero, as gemm's beta = 0 does, so NaN in an uninitialized destination
    // does not survive.
    if (alpha == U{}) {
        impl_scalar_copy(alpha, out);
        return;
    }

    T const           a = static_cast<T>(alpha);
    ParallelThreshold par;
    size_t const      total = out.size();
    strided_for_each_run(
        out.dims(),
        [a, &par, total](size_t n, RunOperand<T> o) {
            if constexpr (blas::IsBlasableV<T>) {
                if (n >= vector_run_min || n == total) {
                    blas::scal(n, a, o.data, o.inc);
                    return;
                }
            }
            run_elements(n, par, [a](T &v) { v *= a; }, o);
        },
        strided(out));
}

template <typename T, typename U>
void impl_div_scalar(U alpha, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();

    T const           a = static_cast<T>(alpha);
    ParallelThreshold par;
    size_t const      total = out.size();
    strided_for_each_run(
        out.dims(),
        [a, &par, total](size_t n, RunOperand<T> o) {
            if constexpr (blas::IsBlasableV<T>) {
                if (n >= vector_run_min || n == total) {
                    blas::rscl(n, a, o.data, o.inc);
                    return;
                }
            }
            run_elements(n, par, [a](T &v) { v /= a; }, o);
        },
        strided(out));
}

template <typename T, typename TOther>
void impl_mult(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();
    require_same_shape(in, out, "multiply");

    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        throw_complex_to_real();
    } else {
        ParallelThreshold par;
        strided_for_each_run(
            out.dims(),
            [&par](size_t n, RunOperand<T> o, RunOperand<TOther const> i) {
                run_elements(n, par, [](T &out_v, TOther const &in_v) { out_v *= convert<TOther, T>(in_v); }, o, i);
            },
            strided(out), strided(in));
    }
}

template <typename T, typename TOther>
void impl_div(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();
    require_same_shape(in, out, "divide");

    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        throw_complex_to_real();
    } else {
        ParallelThreshold par;
        strided_for_each_run(
            out.dims(),
            [&par](size_t n, RunOperand<T> o, RunOperand<TOther const> i) {
                run_elements(n, par, [](T &out_v, TOther const &in_v) { out_v /= convert<TOther, T>(in_v); }, o, i);
            },
            strided(out), strided(in));
    }
}

template <typename T, typename TOther>
void impl_copy(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    WAGGLE_ZONE_FUNC();
    require_same_shape(in, out, "copy");

    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>) {
        if (impl_same_operand(in, out)) {
            return;
        }
    }
    if (impl_storage_overlaps(in, out)) {
        impl_with_snapshot(in, [&](auto const &snapshot) { impl_copy(snapshot, out); });
        return;
    }

    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        throw_complex_to_real();
    } else {
        ParallelThreshold par;
        size_t const      total = out.size();
        strided_for_each_run(
            out.dims(),
            [&par, total](size_t n, RunOperand<T> o, RunOperand<TOther const> i) {
                if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
                    if (use_vector_routine(n, total, o, i)) {
                        blas::copy(n, i.data, i.inc, o.data, o.inc);
                        return;
                    }
                }
                run_elements(n, par, [](T &out_v, TOther const &in_v) { out_v = convert<TOther, T>(in_v); }, o, i);
            },
            strided(out), strided(in));
    }
}

template <typename T, typename U>
void impl_scalar_add(U alpha, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        throw_complex_to_real();
    } else {
        WAGGLE_ZONE_FUNC();

        T const           a = convert<U, T>(alpha);
        ParallelThreshold par;
        size_t const      total = out.size();
        strided_for_each_run(
            out.dims(),
            [a, &par, total](size_t n, RunOperand<T> o) {
                if constexpr (blas::IsBlasableV<T>) {
                    if (n >= vector_run_min || n == total) {
                        // An increment of zero adds the one value to every output.
                        blas::axpy(n, T{1}, &a, 0, o.data, o.inc);
                        return;
                    }
                }
                run_elements(n, par, [a](T &v) { v += a; }, o);
            },
            strided(out));
    }
}

template <typename T, typename U>
void impl_scalar_copy(U alpha, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        throw_complex_to_real();
    } else {
        WAGGLE_ZONE_FUNC();

        T const           a = convert<U, T>(alpha);
        ParallelThreshold par;
        size_t const      total = out.size();
        strided_for_each_run(
            out.dims(),
            [a, &par, total](size_t n, RunOperand<T> o) {
                if constexpr (blas::IsBlasableV<T>) {
                    if (n >= vector_run_min || n == total) {
                        // An increment of zero copies the one value to every output.
                        blas::copy(n, &a, 0, o.data, o.inc);
                        return;
                    }
                }
                run_elements(n, par, [a](T &v) { v = a; }, o);
            },
            strided(out));
    }
}

/**
 * @brief Add the data from the input to the output.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void add_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() += *in.data();
            } else {
                impl_scalar_add(*in.data(), out);
            }
        } else {
            impl_axpy(convert<double, T>(1.0), in, out);
        }
    }
}

/**
 * @brief Subtract the input data from the output data.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void sub_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() -= *in.data();
            } else {
                impl_scalar_add(-*in.data(), out);
            }
        } else {
            impl_axpy(convert<double, T>(-1.0), in, out);
        }
    }
}

/**
 * @brief Multiply the output data by the input data.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void mult_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() *= *in.data();
            } else {
                impl_scal(*in.data(), out);
            }
        } else {
            impl_mult(in, out);
        }
    }
}

/**
 * @brief Divide the output data by the input data.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void div_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() /= *in.data();
            } else {
                impl_div_scalar(*in.data(), out);
            }
        } else {
            impl_div(in, out);
        }
    }
}

/**
 * @brief Copy the data from the input to the output.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void copy_to(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() = *in.data();
            } else {
                impl_scalar_copy(*in.data(), out);
            }
        } else {
            impl_copy(in, out);
        }
    }
}

/**
 * @brief Add a scalar to the output.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void add_assign(U in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() += in;
        } else {
            impl_scalar_add(in, out);
        }
    }
}

/**
 * @brief Subtract a scalar from the output.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void sub_assign(U in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() -= in;
        } else {
            impl_scalar_add(-in, out);
        }
    }
}

/**
 * @brief Multiply the output by a scalar.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void mult_assign(U in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() *= in;
        } else {
            impl_scal(in, out);
        }
    }
}

/**
 * @brief Divide the output by a scalar.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void div_assign(U in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() /= in;
        } else {
            impl_div_scalar(in, out);
        }
    }
}

/**
 * @brief Fill the output with a value.
 *
 * @param in The value to set.
 * @param out The output data.
 */
template <typename T, typename U>
void copy_to(U in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(ComplexConversionError,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() = in;
        } else {
            impl_scalar_copy(in, out);
        }
    }
}

template <typename T>
void copy_real(TensorImpl<std::complex<T>> const &in, TensorImpl<T> &out) {
    if (out.rank() == 0 && in.rank() == 0) {
        *out.data() = std::real(*in.data());
    } else if (in.rank() == 0) {
        copy_to(std::real(*in.data()), out);
    } else {
        impl_real(in, out);
    }
}

template <typename T>
void copy_imag(TensorImpl<std::complex<T>> const &in, TensorImpl<T> &out) {
    if (out.rank() == 0 && in.rank() == 0) {
        *out.data() = std::imag(*in.data());
    } else if (in.rank() == 0) {
        copy_to(std::imag(*in.data()), out);
    } else {
        impl_imag(in, out);
    }
}

template <typename T, typename TOther>
void copy_abs(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if (out.rank() == 0 && in.rank() == 0) {
        *out.data() = std::abs(*in.data());
    } else if (in.rank() == 0) {
        copy_to(std::abs(*in.data()), out);
    } else {
        impl_bas(in, out);
    }
}

EINSUMS_NAMESPACE_END(detail)
