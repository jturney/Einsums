//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

// #include <einsums/concepts/concepts.hpp>
#include <einsums/errors/try_catch_exception_ptr.hpp>
#include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#include <einsums/execution_base/completion_scheduler.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/execution_base/sender.hpp>
#include <einsums/functional/detail/tag_fallback_invoke.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace einsums::drop_value_detail {
template <typename Receiver>
struct drop_value_receiver_impl {
    struct drop_value_receiver_type;
};

template <typename Receiver>
using drop_value_receiver = typename drop_value_receiver_impl<Receiver>::drop_value_receiver_type;

template <typename Receiver>
struct drop_value_receiver_impl<Receiver>::drop_value_receiver_type {
    EINSUMS_STDEXEC_RECEIVER_CONCEPT

    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

    template <typename Error>
    friend void tag_invoke(einsums::execution::experimental::set_error_t, drop_value_receiver_type &&r, Error &&error) noexcept {
        einsums::execution::experimental::set_error(std::move(r.receiver), std::forward<Error>(error));
    }

    friend void tag_invoke(einsums::execution::experimental::set_stopped_t, drop_value_receiver_type &&r) noexcept {
        einsums::execution::experimental::set_stopped(std::move(r.receiver));
    }

    template <typename... Ts>
    friend void tag_invoke(einsums::execution::experimental::set_value_t, drop_value_receiver_type &&r, Ts &&...) noexcept {
        einsums::execution::experimental::set_value(std::move(r.receiver));
    }

    friend constexpr einsums::execution::experimental::empty_env tag_invoke(einsums::execution::experimental::get_env_t,
                                                                            drop_value_receiver_type const &) noexcept {
        return {};
    }
};

template <typename Sender>
struct drop_value_sender_impl {
    struct drop_value_sender_type;
};

template <typename Sender>
using drop_value_sender = typename drop_value_sender_impl<Sender>::drop_value_sender_type;

template <typename Sender>
struct drop_value_sender_impl<Sender>::drop_value_sender_type {
    EINSUMS_STDEXEC_SENDER_CONCEPT

    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;

#if defined(EINSUMS_HAVE_STDEXEC)
    template <class...>
    using empty_set_value = einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_value_t()>;

    using completion_signatures = einsums::execution::experimental::transform_completion_signatures_of<
        Sender, einsums::execution::experimental::empty_env, einsums::execution::experimental::completion_signatures<>, empty_set_value>;
#else
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <typename...> class Variant>
    using error_types = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
        typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>, std::exception_ptr>>;

    static constexpr bool sends_done = false;
#endif

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, drop_value_sender_type &&s, Receiver &&receiver) {
        return einsums::execution::experimental::connect(std::move(s.sender),
                                                         drop_value_receiver<Receiver>{std::forward<Receiver>(receiver)});
    }

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, drop_value_sender_type const &r, Receiver &&receiver) {
        return einsums::execution::experimental::connect(r.sender, drop_value_receiver<Receiver>{std::forward<Receiver>(receiver)});
    }

    friend decltype(auto) tag_invoke(einsums::execution::experimental::get_env_t, drop_value_sender_type const &s) noexcept {
        return einsums::execution::experimental::get_env(s.sender);
    }
};
} // namespace einsums::drop_value_detail

namespace einsums::execution::experimental {
inline constexpr struct drop_value_t final : einsums::functional::detail::tag_fallback<drop_value_t> {
    template <typename Sender>
        requires(is_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(drop_value_t, Sender &&sender) {
        return drop_value_detail::drop_value_sender<Sender>{std::forward<Sender>(sender)};
    }

    // TODO: Use static operator() here, will go to customization afterwards anyway.
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(drop_value_t) { return detail::partial_algorithm<drop_value_t>{}; }
} drop_value{};
} // namespace einsums::execution::experimental
