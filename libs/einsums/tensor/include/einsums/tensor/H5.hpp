//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <h5cpp/core>
#include <type_traits>

namespace h5::impl {

// 1. Object -> H5T_xxx
//    Conversion from Tensor to the underlying double.
template <typename T, size_t Rank>
struct decay<::einsums::Tensor<T, Rank>> {
    using type = T;
};

template <typename T, size_t Rank>
struct decay<::einsums::TensorView<T, Rank>> {
    using type = T;
};

template <typename T, size_t Rank>
auto data(const ::einsums::Tensor<T, Rank> &ref) -> const T * {
    return ref.data();
}

template <typename T, size_t Rank>
auto data(const ::einsums::TensorView<T, Rank> &ref) -> const T * {
    return ref.data();
}

template <typename T, size_t Rank>
auto data(::einsums::Tensor<T, Rank> &ref) -> T * {
    return ref.data();
}

// Determine rank and dimensions
template <typename T, size_t Rank>
struct rank<::einsums::Tensor<T, Rank>> : public std::integral_constant<size_t, Rank> {};

template <typename T, size_t Rank>
struct rank<::einsums::TensorView<T, Rank>> : public std::integral_constant<size_t, Rank> {};

template <typename T, size_t Rank>
inline auto size(const ::einsums::Tensor<T, Rank> &ref) -> std::array<std::int64_t, Rank> {
    return ref.dims();
}

template <typename T, size_t Rank>
inline auto size(const ::einsums::TensorView<T, Rank> &ref) -> std::array<std::int64_t, Rank> {
    return ref.dims();
}

// Constructors
//  Not sure if this can be generalized.
//  Only allow Tensor to be read in and not TensorView
template <typename T>
struct get<::einsums::Tensor<T, 1>> {
    static inline auto ctor(std::array<size_t, 1> dims) -> ::einsums::Tensor<T, 1> {
        return ::einsums::Tensor<T, 1>("hdf5 auto created", dims[0]);
    }
};
template <typename T>
struct get<::einsums::Tensor<T, 2>> {
    static inline auto ctor(std::array<size_t, 2> dims) -> ::einsums::Tensor<T, 2> {
        return ::einsums::Tensor<T, 2>("hdf5 auto created", dims[0], dims[1]);
    }
};
template <typename T>
struct get<::einsums::Tensor<T, 3>> {
    static inline auto ctor(std::array<size_t, 3> dims) -> ::einsums::Tensor<T, 3> {
        return ::einsums::Tensor<T, 3>("hdf5 auto created", dims[0], dims[1], dims[2]);
    }
};
template <typename T>
struct get<::einsums::Tensor<T, 4>> {
    static inline auto ctor(std::array<size_t, 4> dims) -> ::einsums::Tensor<T, 4> {
        return ::einsums::Tensor<T, 4>("hdf5 auto created", dims[0], dims[1], dims[2], dims[3]);
    }
};

} // namespace h5::impl

#include <h5cpp/io>

namespace h5 {

inline bool exists(hid_t hid, const std::string &name) {
    return H5Lexists(hid, name.c_str(), H5P_DEFAULT) > 0 ? true : false;
}

} // namespace h5