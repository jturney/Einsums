// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#if __has_include(<cxxabi.h>)
#    include <cxxabi.h>

namespace einsums::debugging::detail {
using support_cxxabi    = std::true_type;
constexpr auto demangle = abi::__cxa_demangle;
} // namespace einsums::debugging::detail

#else

namespace einsums::debugging::detail {
using support_cxxabi = std::false_type;
template <typename... Ts>
constexpr char *demangle(Ts... ts) {
    return nullptr;
}
} // namespace einsums::debugging::detail

#endif

namespace einsums::debugging::detail {

// default, use built-in typeid to get the best info we can
template <typename T, typename Enabled = std::false_type>
struct demangle_helper {
    [[nodiscard]] char const *type_id() const { return typeid(T).name(); }
};

// if available, demangle an arbitrary c++ type using gnu utility
template <typename T>
struct demangle_helper<T, std::true_type> {
    demangle_helper() : _demangled{demangle(typeid(T).name(), nullptr, nullptr, nullptr), std::free} {}

    [[nodiscard]] char const *type_id() const { return _demangled ? _demangled.get() : typeid(T).name(); }

  private:
    std::unique_ptr<char, void (*)(void *)> _demangled;
};

template <typename T>
using cxx_type_id = demangle_helper<T, support_cxxabi>;

} // namespace einsums::debugging::detail

namespace einsums::debugging {

template <typename T = void>
inline std::string print_type(const char * = "") {
    return std::string(detail::cxx_type_id<T>().type_id());
}

template <>
inline std::string print_type<>(char const *) {
    return "<>";
}

template <typename T, typename... Args>
inline std::string print_type(const char *delim = "")
    requires(sizeof...(Args) != 0)
{
    std::string temp(print_type<T>());
    return temp + delim + print_type<Args...>(delim);
}

} // namespace einsums::debugging