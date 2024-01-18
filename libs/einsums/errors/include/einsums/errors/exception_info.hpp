//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error_code.hpp>

#include <exception>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#if defined(EINSUMS_WINDOWS)
#    include <excpt.h>
#    undef exception_info
#endif

namespace einsums {

///////////////////////////////////////////////////////////////////////////
template <typename Tag, typename Type>
struct error_info {
    using tag  = Tag;
    using type = Type;

    explicit error_info(Type const &value) : _value(value) {}

    explicit error_info(Type &&value) : _value(EINSUMS_MOVE(value)) {}

    Type _value;
};

#define EINSUMS_DEFINE_ERROR_INFO(NAME, TYPE)                                                                          \
    struct NAME : ::einsums::error_info<NAME, TYPE> {                                                                  \
        explicit NAME(TYPE const &value) : error_info(value) {                                                         \
        }                                                                                                              \
                                                                                                                       \
        explicit NAME(TYPE &&value) : error_info(EINSUMS_FORWARD(TYPE, value)) {                                       \
        }                                                                                                              \
    } /**/

///////////////////////////////////////////////////////////////////////////
namespace detail {

struct exception_info_node_base {
    std::shared_ptr<exception_info_node_base> next;

    virtual ~exception_info_node_base()                                                         = default;
    [[nodiscard]] virtual auto lookup(std::type_info const &tag) const noexcept -> void const * = 0;
};

template <typename... Ts>
struct exception_info_node : public exception_info_node_base, Ts... {
    using exception_info_node_base::next;

    template <typename... ErrorInfo>
    explicit exception_info_node(ErrorInfo &&...tagged_values) : Ts(tagged_values)... {}

    [[nodiscard]] auto lookup(std::type_info const &tag) const noexcept -> void const * override {
        using entry_type           = std::pair<std::type_info const &, void const *>;
        entry_type const entries[] = {
            {typeid(typename Ts::tag), std::addressof(static_cast<Ts const *>(this)->_value)}...};

        for (auto const &entry : entries) {
            if (entry.first == tag)
                return entry.second;
        }

        return next ? next->lookup(tag) : nullptr;
    }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////

struct exception_info {
  private:
    using node_ptr = std::shared_ptr<detail::exception_info_node_base>;

    node_ptr _data;

  public:
    exception_info() noexcept : _data(nullptr) {}

    exception_info(exception_info const &other) noexcept = default;
    exception_info(exception_info &&other) noexcept      = default;

    auto operator=(exception_info const &other) noexcept -> exception_info & = default;
    auto operator=(exception_info &&other) noexcept -> exception_info      & = default;

    virtual ~exception_info() = default;

    template <typename... ErrorInfo>
    exception_info &set(ErrorInfo &&...tagged_values) {
        using node_type = detail::exception_info_node<ErrorInfo...>;

        node_ptr node = std::make_shared<node_type>(EINSUMS_FORWARD(ErrorInfo, tagged_values)...);
        node->next    = EINSUMS_MOVE(_data);
        _data         = EINSUMS_MOVE(node);
        return *this;
    }

    template <typename Tag>
    typename Tag::type const *get() const noexcept {
        auto const *data = _data.get();
        return static_cast<typename Tag::type const *>(data ? data->lookup(typeid(typename Tag::tag)) : nullptr);
    }
};

///////////////////////////////////////////////////////////////////////////
namespace detail {
struct exception_with_info_base : public exception_info {
    exception_with_info_base(std::type_info const &type, exception_info xi)
        : exception_info(EINSUMS_MOVE(xi)), type(type) {}

    std::type_info const &type;
};

template <typename E>
struct exception_with_info : public E, public exception_with_info_base {
    explicit exception_with_info(E const &e, exception_info xi)
        : E(e), exception_with_info_base(typeid(E), EINSUMS_MOVE(xi)) {}

    explicit exception_with_info(E &&e, exception_info xi)
        : E(EINSUMS_MOVE(e)), exception_with_info_base(typeid(E), EINSUMS_MOVE(xi)) {}
};
} // namespace detail

template <typename E>
[[noreturn]] void throw_with_info(E &&e, exception_info &&xi = exception_info()) {
    using ED = std::decay_t<E>;
    static_assert(std::is_class_v<ED> && !std::is_final_v<ED>, "E shall be a valid base class");
    static_assert(!std::is_base_of_v<exception_info, ED>, "E shall not derive from exception_info");

    throw detail::exception_with_info<ED>(EINSUMS_FORWARD(E, e), EINSUMS_MOVE(xi));
}

template <typename E>
[[noreturn]] void throw_with_info(E &&e, exception_info const &xi) {
    throw_with_info(EINSUMS_FORWARD(E, e), exception_info(xi));
}

///////////////////////////////////////////////////////////////////////////
template <typename E>
auto get_exception_info(E &e) -> exception_info * {
    return dynamic_cast<exception_info *>(std::addressof(e));
}

template <typename E>
auto get_exception_info(E const &e) -> exception_info const * {
    return dynamic_cast<exception_info const *>(std::addressof(e));
}

///////////////////////////////////////////////////////////////////////////
template <typename E, typename F>
auto invoke_with_exception_info(E const &e, F &&f)
    -> decltype(EINSUMS_FORWARD(F, f)(std::declval<exception_info const *>())) {
    return EINSUMS_FORWARD(F, f)(dynamic_cast<exception_info const *>(std::addressof(e)));
}

template <typename F>
auto invoke_with_exception_info(std::exception_ptr const &p, F &&f)
    -> decltype(EINSUMS_FORWARD(F, f)(std::declval<exception_info const *>())) {
    try {
        if (p)
            std::rethrow_exception(p);
    } catch (exception_info const &xi) {
        return EINSUMS_FORWARD(F, f)(&xi);
    } catch (...) {
    }
    return EINSUMS_FORWARD(F, f)(nullptr);
}

template <typename F>
auto invoke_with_exception_info(einsums::error_code const &ec, F &&f)
    -> decltype(EINSUMS_FORWARD(F, f)(std::declval<exception_info const *>())) {
    return invoke_with_exception_info(detail::access_exception(ec), EINSUMS_FORWARD(F, f));
}

} // namespace einsums