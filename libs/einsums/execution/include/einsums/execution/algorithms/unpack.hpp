//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/try_catch_exception_ptr.hpp>
#include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#include <einsums/execution_base/completion_scheduler.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/execution_base/sender.hpp>
#include <einsums/functional/bind_front.hpp>
#include <einsums/functional/detail/tag_fallback_invoke.hpp>
#include <einsums/functional/invoke.hpp>
#include <einsums/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace einsums::unpack_detail {
template <typename Receiver>
struct unpack_receiver_impl {
    struct unpack_receiver_type;
};

template <typename Receiver>
using unpack_receiver = typename unpack_receiver_impl<Receiver>::unpack_receiver_type;

template <typename Receiver>
struct unpack_receiver_impl<Receiver>::unpack_receiver_type {
    EINSUMS_STDEXEC_RECEIVER_CONCEPT

    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

    template <typename Error>
    friend void tag_invoke(einsums::execution::experimental::set_error_t, unpack_receiver_type &&r, Error &&error) noexcept {
        einsums::execution::experimental::set_error(std::move(r.receiver), std::forward<Error>(error));
    }

    friend void tag_invoke(einsums::execution::experimental::set_stopped_t, unpack_receiver_type &&r) noexcept {
        einsums::execution::experimental::set_stopped(std::move(r.receiver));
    }

    template <typename Ts>
    friend void tag_invoke(einsums::execution::experimental::set_value_t, unpack_receiver_type &&r, Ts &&ts) noexcept {
        std::apply(einsums::util::detail::bind_front(einsums::execution::experimental::set_value, std::move(r.receiver)),
                   std::forward<Ts>(ts));
    }

    friend constexpr einsums::execution::experimental::empty_env tag_invoke(einsums::execution::experimental::get_env_t,
                                                                            unpack_receiver_type const &) noexcept {
        return {};
    }
};

#if defined(EINSUMS_HAVE_STDEXEC)
template <typename IndexPack, typename T>
struct make_value_type;

template <typename T, std::size_t... Is>
struct make_value_type<einsums::util::detail::index_pack<Is...>, T> {
    using type = einsums::execution::experimental::set_value_t(decltype(std::get<Is>(std::declval<T>()))...);
};

template <typename... Ts>
struct invoke_result_helper {
    static_assert(sizeof...(Ts) == 0, "unpack expects the predecessor sender to send exactly one tuple-like type in each "
                                      "variant");
};

template <typename T>
struct invoke_result_helper<T> {
    using type = typename make_value_type<einsums::util::detail::make_index_pack_t<std::tuple_size_v<std::decay_t<T>>>, T>::type;
};

template <typename... Ts>
using invoke_result_helper_t = einsums::execution::experimental::completion_signatures<typename invoke_result_helper<Ts...>::type>;
#else
template <typename Tuple>
struct invoke_result_helper;

template <template <typename...> class Tuple, typename... Ts>
struct invoke_result_helper<Tuple<Ts...>> {
    static_assert(sizeof(Tuple<Ts...>) == 0, "unpack expects the predecessor sender to send exactly one tuple-like type in each "
                                             "variant");
};

template <typename IndexPack, template <typename...> class Tuple, typename T>
struct make_value_type;

template <template <typename...> class Tuple, typename T, std::size_t... Is>
struct make_value_type<einsums::util::detail::index_pack<Is...>, Tuple, T> {
    using type = Tuple<decltype(std::get<Is>(std::declval<T>()))...>;
};

template <template <typename...> class Tuple, typename T>
struct invoke_result_helper<Tuple<T>> {
    using type = typename make_value_type<einsums::util::detail::make_index_pack_t<std::tuple_size_v<std::decay_t<T>>>, Tuple, T>::type;
};

template <template <typename...> class Tuple>
struct invoke_result_helper<Tuple<>> {
    using type = Tuple<>;
};
#endif

template <typename Sender>
struct unpack_sender_impl {
    struct unpack_sender_type;
};

template <typename Sender>
using unpack_sender = typename unpack_sender_impl<Sender>::unpack_sender_type;

template <typename Sender>
struct unpack_sender_impl<Sender>::unpack_sender_type {
    EINSUMS_STDEXEC_SENDER_CONCEPT

    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;

#if defined(EINSUMS_HAVE_STDEXEC)
    using completion_signatures = einsums::execution::experimental::transform_completion_signatures_of<
        std::decay_t<Sender>, einsums::execution::experimental::empty_env,
        einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_error_t(std::exception_ptr)>,
        invoke_result_helper_t>;
#else
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = einsums::util::detail::unique_t<einsums::util::detail::transform_t<
        typename einsums::execution::experimental::sender_traits<Sender>::template value_types<Tuple, Variant>, invoke_result_helper>>;

    template <template <typename...> class Variant>
    using error_types = typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>;

    static constexpr bool sends_done = false;
#endif

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, unpack_sender_type &&s, Receiver &&receiver) {
        return einsums::execution::experimental::connect(std::move(s.sender), unpack_receiver<Receiver>{std::forward<Receiver>(receiver)});
    }

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, unpack_sender_type const &r, Receiver &&receiver) {
        return einsums::execution::experimental::connect(r.sender, unpack_receiver<Receiver>{std::forward<Receiver>(receiver)});
    }

    friend decltype(auto) tag_invoke(einsums::execution::experimental::get_env_t, unpack_sender_type const &s) noexcept {
        return einsums::execution::experimental::get_env(s.sender);
    }
};
} // namespace einsums::unpack_detail

namespace einsums::execution::experimental {
inline constexpr struct unpack_t final : einsums::functional::detail::tag_fallback<unpack_t> {
  private:
    template <typename Sender>
        requires(is_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(unpack_t, Sender &&sender) {
        return unpack_detail::unpack_sender<Sender>{std::forward<Sender>(sender)};
    }

    friend constexpr EINSUMS_FORCEINLINE auto tag_invoke(unpack_t) { return detail::partial_algorithm<unpack_t>{}; }
} unpack{};
} // namespace einsums::execution::experimental
