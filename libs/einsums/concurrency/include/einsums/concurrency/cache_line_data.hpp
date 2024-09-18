//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace einsums::concurrency::detail {

/**
 * Abstract away cache-line size.
 */
constexpr std::size_t get_cache_line_size() noexcept {
#if defined(powerpc) || defined(__powerpc__) || defined(__ppc__)
    return 128;
#else
    return 64;
#endif
}

/// Computes the padding required to fill up a full cache line after data_size bytes
constexpr std::size_t get_cache_line_padding_size(std::size_t data_size) noexcept {
    return (get_cache_line_size() - (data_size % get_cache_line_size())) % get_cache_line_size();
}

template <typename Data>
struct needs_padding : std::integral_constant<bool, detail::get_cache_line_padding_size(sizeof(Data)) != 0> {};

// special struct to ensure cache line alignment of a data type
template <typename Data, typename NeedsPadding = typename detail::needs_padding<Data>::type>
struct cache_aligned_data {
    // We have an explicit (non-default) constructor here to avoid for
    // the entire cache-line to be initialized by the compiler.
    cache_aligned_data() : data_() {}

    cache_aligned_data(Data &&data) noexcept : data_{std::move(data)} {}

    cache_aligned_data(Data const &data) : data_{data} {}

    // pad to cache line size bytes
    Data data_;

    //  cppcheck-suppress unusedVariable
    char cacheline_pad[detail::get_cache_line_padding_size(
        // NOLINTNEXTLINE(bugprone-sizeof-expression)
        sizeof(Data))];
};

template <typename Data>
struct cache_aligned_data<Data, std::false_type> {
    cache_aligned_data() = default;

    cache_aligned_data(Data &&data) noexcept : data_{std::move(data)} {}

    cache_aligned_data(Data const &data) : data_{data} {}

    // no need to pad to cache line size
    Data data_;
};

///////////////////////////////////////////////////////////////////////////
// special struct to ensure cache line alignment of a data type
template <typename Data, typename NeedsPadding = typename detail::needs_padding<Data>::type>
struct cache_aligned_data_derived : Data {
    // We have an explicit (non-default) constructor here to avoid for
    // the entire cache-line to be initialized by the compiler.
    cache_aligned_data_derived() : Data() {}

    cache_aligned_data_derived(Data &&data) noexcept : Data{std::move(data)} {}

    cache_aligned_data_derived(Data const &data) : Data{data} {}

    //  cppcheck-suppress unusedVariable
    char cacheline_pad[detail::get_cache_line_padding_size(
        // NOLINTNEXTLINE(bugprone-sizeof-expression)
        sizeof(Data))];
};

template <typename Data>
struct cache_aligned_data_derived<Data, std::false_type> : Data {
    cache_aligned_data_derived() = default;

    cache_aligned_data_derived(Data &&data) noexcept : Data{std::move(data)} {}

    cache_aligned_data_derived(Data const &data) : Data{data} {}

    // no need to pad to cache line size
};

///////////////////////////////////////////////////////////////////////////
// special struct to data type is cache line aligned and fully occupies a
// cache line
template <typename Data>
using cache_line_data = cache_aligned_data<Data>;
} // namespace einsums::concurrency::detail