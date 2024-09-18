//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
#    include <einsums/functional/invoke.hpp>
#    include <einsums/functional/traits/get_function_address.hpp>
#    include <einsums/functional/traits/get_function_annotation.hpp>
#    include <einsums/threading_base/scoped_annotation.hpp>
#    include <einsums/threading_base/thread_description.hpp>
#    include <einsums/threading_base/thread_helpers.hpp>
#    include <einsums/type_support/decay.hpp>

#    if EINSUMS_HAVE_ITTNOTIFY != 0
#        include <einsums/modules/itt_notify.hpp>
#    elif defined(EINSUMS_HAVE_APEX)
#        include <einsums/threading_base/external_timer.hpp>
#    endif
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace einsums {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////
namespace detail {
template <typename F>
struct annotated_function {
    using fun_type = detail::decay_unwrap_t<F>;

    annotated_function() noexcept : _name(nullptr) {}

    annotated_function(F const &f, char const *name) : _f(f), _name(name) {}

    annotated_function(F &&f, char const *name) : _f(EINSUMS_MOVE(f)), _name(name) {}

    template <typename... Ts>
    std::invoke_result_t<fun_type, Ts...> operator()(Ts &&...ts) {
        scoped_annotation annotate(get_function_annotation());
        return EINSUMS_INVOKE(_f, std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////
    /// \brief Returns the function address
    ///
    /// This function returns the passed function address.
    /// \param none
    constexpr std::size_t get_function_address() const { return einsums::detail::get_function_address<fun_type>::call(_f); }

    ///////////////////////////////////////////////////////////////////
    /// \brief Returns the function annotation
    ///
    /// This function returns the function annotation, if it has a name
    /// name is returned, name is returned; if name is empty the typeid
    /// is returned
    ///
    /// \param none
    constexpr char const *get_function_annotation() const noexcept { return _name ? _name : typeid(_f).name(); }

    constexpr fun_type const &get_bound_function() const noexcept { return _f; }

  private:
    fun_type    _f;
    char const *_name;
};
} // namespace detail

template <typename F>
detail::annotated_function<std::decay_t<F>> annotated_function(F &&f, char const *name = nullptr) {
    using result_type = detail::annotated_function<std::decay_t<F>>;

    return result_type(std::forward<F>(f), name);
}

template <typename F>
detail::annotated_function<std::decay_t<F>> annotated_function(F &&f, std::string name) {
    using result_type = detail::annotated_function<std::decay_t<F>>;

    // Store string in a set to ensure it lives for the entire duration of
    // the task.
    char const *name_c_str = einsums::detail::store_function_annotation(std::move(name));
    return result_type(std::forward<F>(f), name_c_str);
}

#else
///////////////////////////////////////////////////////////////////////////
/// \brief Returns a function annotated with the given annotation.
///
/// Annotating includes setting the thread description per thread id.
///
/// \param function
template <typename F>
constexpr F &&annotated_function(F &&f, char const * = nullptr) noexcept {
    return std::forward<F>(f);
}

template <typename F>
constexpr F &&annotated_function(F &&f, std::string const &) noexcept {
    return std::forward<F>(f);
}
#endif
} // namespace einsums

namespace einsums::detail {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
template <typename F>
struct get_function_address<einsums::detail::annotated_function<F>> {
    static constexpr std::size_t call(einsums::detail::annotated_function<F> const &f) noexcept { return f.get_function_address(); }
};

template <typename F>
struct get_function_annotation<einsums::detail::annotated_function<F>> {
    static constexpr char const *call(einsums::detail::annotated_function<F> const &f) noexcept { return f.get_function_annotation(); }
};
#endif
} // namespace einsums::detail
