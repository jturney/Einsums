//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra/Bases/high_precision.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename T, typename TOther>
BiggestTypeT<T, TOther> impl_dot_contiguous(einsums::detail::TensorImpl<T> const &in, einsums::detail::TensorImpl<TOther> const &out) {
    if constexpr (std::is_same_v<T, TOther> && blas::IsBlasableV<T>) {
        return blas::dot(in.size(), in.data(), in.get_incx(), out.data(), out.get_incx());
    } else {
        T const      *in_data  = in.data();
        TOther const *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();

        BiggestTypeT<T, TOther> sum{0.0};

        EINSUMS_OMP_PRAGMA(parallel for reduction(+: sum))
        for (size_t i = 0; i < size; i++) {
            sum += in_data[i * incx] * out_data[i * incy];
        }

        return sum;
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
BiggestTypeT<T, TOther> impl_dot_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, T const *in,
                                                          InStrides const &in_strides, size_t inc_in, TOther const *out,
                                                          OutStrides const &out_strides, size_t inc_out) {

    using U = BiggestTypeT<T, TOther>;

    if (depth == hard_rank) {
        if constexpr (std::is_same_v<T, TOther> && blas::IsBlasableV<T>) {
            return blas::dot(easy_size, in, inc_in, out, inc_out);
        } else {
            BiggestTypeT<T, TOther> sum{0.0};

            EINSUMS_OMP_PRAGMA(parallel for reduction(+: sum))
            for (size_t i = 0; i < easy_size; i++) {
                sum += in[i * inc_in] * out[i * inc_out];
            }
            return sum;
        }
    } else {
        if constexpr (IsComplexV<U>) {
            BiggestTypeT<T, TOther> big_sum{0.0}, medium_sum{0.0}, small_sum{0.0};

            bool not_big_re = true, not_big_im = true;

            for (int i = 0; i < dims[depth]; i++) {
                auto ret = impl_dot_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides,
                                                             inc_in, out + i * out_strides[depth], out_strides, inc_out);

                add_scale(ret, big_sum, medium_sum, small_sum, not_big_re, not_big_im);
            }
            return combine_accum(big_sum, medium_sum, small_sum);
        } else {
            BiggestTypeT<T, TOther> big_sum{0.0}, medium_sum{0.0}, small_sum{0.0};

            bool not_big = true;

            for (int i = 0; i < dims[depth]; i++) {
                auto ret = impl_dot_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides,
                                                             inc_in, out + i * out_strides[depth], out_strides, inc_out);

                add_scale(ret, big_sum, medium_sum, small_sum, not_big);
            }
            return combine_accum(big_sum, medium_sum, small_sum);
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
BiggestTypeT<T, TOther> impl_dot_noncontiguous(int depth, int rank, Dims const &dims, T const *in, InStrides const &in_strides,
                                               TOther const *out, OutStrides const &out_strides) {

    using U = BiggestTypeT<T, TOther>;

    if (depth == rank) {
        return *in * *out;
    } else {
        if constexpr (IsComplexV<U>) {
            BiggestTypeT<T, TOther> big_sum{0.0}, medium_sum{0.0}, small_sum{0.0};

            bool not_big_re = true, not_big_im = true;

            for (int i = 0; i < dims[depth]; i++) {
                auto ret = impl_dot_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides,
                                                  out + i * out_strides[depth], out_strides);

                add_scale(ret, big_sum, medium_sum, small_sum, not_big_re, not_big_im);
            }
            return combine_accum(big_sum, medium_sum, small_sum);
        } else {
            BiggestTypeT<T, TOther> big_sum{0.0}, medium_sum{0.0}, small_sum{0.0};

            bool not_big = true;

            for (int i = 0; i < dims[depth]; i++) {
                auto ret = impl_dot_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides,
                                                  out + i * out_strides[depth], out_strides);

                add_scale(ret, big_sum, medium_sum, small_sum, not_big);
            }
            return combine_accum(big_sum, medium_sum, small_sum);
        }
    }
}

template <typename T, typename TOther>
BiggestTypeT<T, TOther> impl_dot(einsums::detail::TensorImpl<T> const &in, einsums::detail::TensorImpl<TOther> const &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not dot two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not add two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        return impl_dot_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using dot.");

        return impl_dot_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over dot.");

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
        }

        hard_dims.resize(in.rank() - easy_rank);

        if (in.stride(0) < in.stride(-1)) {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i + easy_rank);
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = in.dim(i + easy_rank);
            }
        } else {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i);
                out_strides[i] = out.stride(i);
                hard_dims[i]   = in.dim(i);
            }
        }

        return impl_dot_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx, out.data(),
                                                 out_strides, out_incx);
    }
}

template <Complex T, typename TOther>
BiggestTypeT<T, TOther> impl_true_dot_contiguous(einsums::detail::TensorImpl<T> const &in, einsums::detail::TensorImpl<TOther> const &out) {
    if constexpr (std::is_same_v<T, TOther> && blas::IsBlasableV<T>) {
        return blas::dotc(in.size(), in.data(), in.get_incx(), out.data(), out.get_incx());
    } else {
        T const      *in_data  = in.data();
        TOther const *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();

        BiggestTypeT<T, TOther> sum{0.0};

        EINSUMS_OMP_PRAGMA(parallel for reduction(+: sum))
        for (size_t i = 0; i < size; i++) {
            sum += std::conj(in_data[i * incx]) * out_data[i * incy];
        }

        return sum;
    }
}

template <Complex T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
BiggestTypeT<T, TOther> impl_true_dot_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims,
                                                               T const *in, InStrides const &in_strides, size_t inc_in, TOther const *out,
                                                               OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            return blas::dotc(easy_size, in, inc_in, out, inc_out);
        } else {
            BiggestTypeT<T, TOther> sum{0.0};

            EINSUMS_OMP_PRAGMA(parallel for reduction(+: sum))
            for (size_t i = 0; i < easy_size; i++) {
                sum += std::conj(in[i * inc_in]) * out[i * inc_out];
            }
            return sum;
        }
    } else {
        BiggestTypeT<T, TOther> sum{0.0};

        for (int i = 0; i < dims[depth]; i++) {
            sum += impl_true_dot_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides,
                                                          inc_in, out + i * out_strides[depth], out_strides, inc_out);
        }
        return sum;
    }
}

template <Complex T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
BiggestTypeT<T, TOther> impl_true_dot_noncontiguous(int depth, int rank, Dims const &dims, T const *in, InStrides const &in_strides,
                                                    TOther const *out, OutStrides const &out_strides) {
    if (depth == rank) {
        return *in * *out;
    } else {
        BiggestTypeT<T, TOther> sum{0.0};
        for (int i = 0; i < dims[depth]; i++) {
            sum += impl_dot_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                          out_strides);
        }
        return sum;
    }
}

template <typename T, typename TOther>
BiggestTypeT<T, TOther> impl_true_dot(einsums::detail::TensorImpl<T> const &in, einsums::detail::TensorImpl<TOther> const &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not dot two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not add two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        if constexpr (IsComplexV<T>) {
            return impl_true_dot_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
        } else {
            return impl_dot_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
        }
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using dot.");

        if constexpr (IsComplexV<T>) {
            return impl_true_dot_contiguous(in, out);
        } else {
            return impl_dot_contiguous(in, out);
        }
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over dot.");

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
        }

        hard_dims.resize(in.rank() - easy_rank);

        if (in.stride(0) < in.stride(-1)) {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i + easy_rank);
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = in.dim(i + easy_rank);
            }
        } else {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i);
                out_strides[i] = out.stride(i);
                hard_dims[i]   = in.dim(i);
            }
        }

        if constexpr (IsComplexV<T>) {
            return impl_true_dot_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx,
                                                          out.data(), out_strides, out_incx);
        } else {
            return impl_dot_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx,
                                                     out.data(), out_strides, out_incx);
        }
    }
}

template <typename A, typename B, typename C>
BiggestTypeT<A, B, C> impl_dot_contiguous(einsums::detail::TensorImpl<A> const &a, einsums::detail::TensorImpl<B> const &b,
                                          einsums::detail::TensorImpl<C> const &c) {
    A const     *a_data = a.data();
    B const     *b_data = b.data();
    C const     *c_data = c.data();
    size_t const inca = a.get_incx(), incb = b.get_incx(), incc = c.get_incx(), size = a.size();

    BiggestTypeT<A, B, C> sum{0.0};

    EINSUMS_OMP_PRAGMA(parallel for reduction(+: sum))
    for (size_t i = 0; i < size; i++) {
        sum += a_data[i * inca] * b_data[i * incb] * c_data[i * incc];
    }

    return sum;
}

template <typename A, typename B, typename C, Container HardDims, Container AStrides, Container BStrides, Container CStrides>
BiggestTypeT<A, B, C> impl_dot_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, A const *a,
                                                        AStrides const &a_strides, size_t inca, B const *b, BStrides const &b_strides,
                                                        size_t incb, C const *c, CStrides const &c_strides, size_t incc) {
    if (depth == hard_rank) {
        BiggestTypeT<A, B, C> sum{0.0};

        EINSUMS_OMP_PRAGMA(parallel for reduction(+: sum))
        for (size_t i = 0; i < easy_size; i++) {
            sum += a[i * inca] * b[i * incb] * c[i * incc];
        }

        return sum;
    } else {
        BiggestTypeT<A, B, C> sum{0.0};

        for (int i = 0; i < dims[depth]; i++) {
            sum += impl_dot_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, a + i * a_strides[depth], a_strides, inca,
                                                     b + i * b_strides[depth], b_strides, incb, c + i * c_strides[depth], c_strides, incc);
        }
        return sum;
    }
}

template <typename A, typename B, typename C, Container Dims, Container AStrides, Container BStrides, Container CStrides>
BiggestTypeT<A, B, C> impl_dot_noncontiguous(int depth, int rank, Dims const &dims, A const *a, AStrides const &a_strides, B const *b,
                                             BStrides const &b_strides, C const *c, CStrides const &c_strides) {
    if (depth == rank) {
        return *a * *b * *c;
    } else {
        BiggestTypeT<A, B, C> sum{0.0};
        for (int i = 0; i < dims[depth]; i++) {
            sum += impl_dot_noncontiguous(depth + 1, rank, dims, a + i * a_strides[depth], a_strides, b + i * b_strides[depth], b_strides,
                                          c + i * c_strides[depth], c_strides);
        }
        return sum;
    }
}

template <typename A, typename B, typename C>
BiggestTypeT<A, B, C> impl_dot(einsums::detail::TensorImpl<A> const &a, einsums::detail::TensorImpl<B> const &b,
                               einsums::detail::TensorImpl<C> const &c) {
    LabeledSection0();

    if (a.rank() != b.rank() || a.rank() != c.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not dot three tensors of different ranks!");
    }

    if (a.dims() != b.dims() || a.dims() != c.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not add three tensors with different sizes!");
    }

    if (a.is_column_major() != b.is_column_major() || a.is_column_major() != c.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        return impl_dot_noncontiguous(0, a.rank(), a.dims(), a.data(), a.strides(), b.data(), b.strides(), c.data(), c.strides());
    } else if (a.is_totally_vectorable() && b.is_totally_vectorable() && c.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using dot.");

        return impl_dot_contiguous(a, b, c);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over dot.");

        size_t easy_size, a_easy_size, b_easy_size, c_easy_size, a_hard_size, b_hard_size, c_hard_size, easy_rank, a_easy_rank, b_easy_rank,
            c_easy_rank, a_incx, b_incx, c_incx;
        BufferVector<size_t> hard_dims, a_strides, b_strides, c_strides;

        a.query_vectorable_params(&a_easy_size, &a_hard_size, &a_easy_rank, &a_incx);
        b.query_vectorable_params(&b_easy_size, &b_hard_size, &b_easy_rank, &b_incx);
        c.query_vectorable_params(&c_easy_size, &c_hard_size, &c_easy_rank, &c_incx);

        if (a_easy_rank < b_easy_rank) {
            if (c_easy_rank < a_easy_rank) {
                easy_rank = c_easy_rank;
                easy_size = c_easy_size;
            } else {
                easy_rank = a_easy_rank;
                easy_size = a_easy_size;
            }
        } else {
            if (c_easy_rank < b_easy_rank) {
                easy_rank = c_easy_rank;
                easy_size = c_easy_size;
            } else {
                easy_rank = b_easy_rank;
                easy_size = b_easy_size;
            }
        }

        hard_dims.resize(a.rank() - easy_rank);

        if (a.stride(0) < a.stride(-1)) {
            a_strides.resize(a.rank() - easy_rank);
            b_strides.resize(b.rank() - easy_rank);
            c_strides.resize(c.rank() - easy_rank);

            for (int i = 0; i < a.rank() - easy_rank; i++) {
                a_strides[i] = a.stride(i + easy_rank);
                b_strides[i] = b.stride(i + easy_rank);
                c_strides[i] = c.stride(i + easy_rank);
                hard_dims[i] = a.dim(i + easy_rank);
            }
        } else {
            a_strides.resize(a.rank() - easy_rank);
            b_strides.resize(b.rank() - easy_rank);
            c_strides.resize(c.rank() - easy_rank);

            for (int i = 0; i < a.rank() - easy_rank; i++) {
                a_strides[i] = a.stride(i);
                b_strides[i] = b.stride(i);
                c_strides[i] = c.stride(i);
                hard_dims[i] = a.dim(i);
            }
        }

        return impl_dot_noncontiguous_vectorable(0, a.rank() - easy_rank, easy_size, hard_dims, a.data(), a_strides, a_incx, b.data(),
                                                 b_strides, b_incx, c.data(), c_strides, c_incx);
    }
}

#ifndef DOXYGEN
extern template EINSUMS_EXPORT float  impl_dot<float, float>(einsums::detail::TensorImpl<float> const &a,
                                                            einsums::detail::TensorImpl<float> const &b);
extern template EINSUMS_EXPORT double impl_dot<double, double>(einsums::detail::TensorImpl<double> const &a,
                                                               einsums::detail::TensorImpl<double> const &b);
extern template EINSUMS_EXPORT        std::complex<float>
                               impl_dot<std::complex<float>, std::complex<float>>(einsums::detail::TensorImpl<std::complex<float>> const &a,
                                                   einsums::detail::TensorImpl<std::complex<float>> const &b);
extern template EINSUMS_EXPORT std::complex<double>
impl_dot<std::complex<double>, std::complex<double>>(einsums::detail::TensorImpl<std::complex<double>> const &a,
                                                     einsums::detail::TensorImpl<std::complex<double>> const &b);

extern template EINSUMS_EXPORT float  impl_true_dot<float, float>(einsums::detail::TensorImpl<float> const &a,
                                                                 einsums::detail::TensorImpl<float> const &b);
extern template EINSUMS_EXPORT double impl_true_dot<double, double>(einsums::detail::TensorImpl<double> const &a,
                                                                    einsums::detail::TensorImpl<double> const &b);
extern template EINSUMS_EXPORT        std::complex<float>
impl_true_dot<std::complex<float>, std::complex<float>>(einsums::detail::TensorImpl<std::complex<float>> const &a,
                                                        einsums::detail::TensorImpl<std::complex<float>> const &b);
extern template EINSUMS_EXPORT std::complex<double>
impl_true_dot<std::complex<double>, std::complex<double>>(einsums::detail::TensorImpl<std::complex<double>> const &a,
                                                          einsums::detail::TensorImpl<std::complex<double>> const &b);
#endif
} // namespace detail
} // namespace linear_algebra
} // namespace einsums