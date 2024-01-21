//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstdlib>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace einsums::debug::detail {

template <typename T>
struct demangle_helper {
    [[nodiscard]] auto type_id() const -> char const * { return typeid(T).name(); }
};
} // namespace einsums::debug::detail

#if defined(__GNUG__)

#    include <cxxabi.h>
#    include <memory>

namespace einsums::debug::detail {

template <typename T>
struct cxxabi_demangle_helper {
  private:
    std::unique_ptr<char, void (*)(void *)> _demangled;

  public:
    cxxabi_demangle_helper()
        : _demangled{abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr), std::free} {}

    [[nodiscard]] auto type_id() const -> char const * { return _demangled ? _demangled.get() : typeid(T).name(); }
};

} // namespace einsums::debug::detail

#else

namespace pika::debug::detail {
template <typename T>
using cxxabi_demangle_helper = demangle_helper<T>;
} // namespace pika::debug::detail

#endif

namespace einsums::debug::detail {
template <typename T>
struct type_id {
    static demangle_helper<T> typeid_;
};

template <typename T>
demangle_helper<T> type_id<T>::typeid_ = demangle_helper<T>();

#if defined(__GNUG__)
template <typename T>
struct cxx_type_id {
    static cxxabi_demangle_helper<T> typeid_;
};

template <typename T>
cxxabi_demangle_helper<T> cxx_type_id<T>::typeid_ = cxxabi_demangle_helper<T>();
#else
template <typename T>
using cxx_type_id = type_id<T>;
#endif

// --------------------------------------------------------------------
// print type information
// usage : std::cout << print_type<args...>("separator")
// separator is appended if the number of types > 1
// --------------------------------------------------------------------
template <typename T = void>
inline auto print_type(char const * = "") -> std::string {
    return std::string(cxx_type_id<T>::typeid_.type_id());
}

template <>
inline auto print_type<>(char const *) -> std::string {
    return "void";
}

template <typename T, typename... Args>
inline auto print_type(char const *delim = "") -> std::string
    requires(sizeof...(Args) != 0)
{
    std::string temp(cxx_type_id<T>::typeid_.type_id());
    return temp + delim + print_type<Args...>(delim);
}
} // namespace einsums::debug::detail
