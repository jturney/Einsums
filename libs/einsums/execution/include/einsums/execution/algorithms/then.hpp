//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
#    include <einsums/errors/try_catch_exception_ptr.hpp>
#    include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#    include <einsums/execution_base/completion_scheduler.hpp>
#    include <einsums/execution_base/receiver.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/functional/detail/tag_fallback_invoke.hpp>
#    include <einsums/functional/invoke.hpp>
#    include <einsums/type_support/pack.hpp>

#    include <exception>
#    include <type_traits>
#    include <utility>

namespace einsums::then_detail {
template <typename Receiver, typename F>
struct then_receiver_impl {
    struct then_receiver_type;
};

template <typename Receiver, typename F>
using then_receiver = typename then_receiver_impl<Receiver, F>::then_receiver_type;

template <typename Receiver, typename F>
struct then_receiver_impl<Receiver, F>::then_receiver_type {
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<F> f;

    template <typename Error>
    friend void tag_invoke(einsums::execution::experimental::set_error_t, then_receiver_type &&r, Error &&error) noexcept {
        einsums::execution::experimental::set_error(std::move(r.receiver), std::forward<Error>(error));
    }

    friend void tag_invoke(einsums::execution::experimental::set_stopped_t, then_receiver_type &&r) noexcept {
        einsums::execution::experimental::set_stopped(std::move(r.receiver));
    }

  private:
    template <typename... Ts>
    void set_value_helper(Ts &&...ts) noexcept {
        einsums::detail::try_catch_exception_ptr(
            [&]() {
                if constexpr (std::is_void_v<std::invoke_result_t<F, Ts...>>) {
                // Certain versions of GCC with optimizations fail on
                // the move with an internal compiler error.
#    if defined(EINSUMS_GCC_VERSION) && (EINSUMS_GCC_VERSION < 100000)
                    EINSUMS_INVOKE(std::move(f), std::forward<Ts>(ts)...);
#    else
                    EINSUMS_INVOKE(std::move(f), std::forward<Ts>(ts)...);
#    endif
                    einsums::execution::experimental::set_value(std::move(receiver));
                } else {
                // Certain versions of GCC with optimizations fail on
                // the move with an internal compiler error.
#    if defined(EINSUMS_GCC_VERSION) && (EINSUMS_GCC_VERSION < 100000)
                    einsums::execution::experimental::set_value(std::move(receiver), EINSUMS_INVOKE(std::move(f), std::forward<Ts>(ts)...));
#    else
                    einsums::execution::experimental::set_value(std::move(receiver), EINSUMS_INVOKE(std::move(f), std::forward<Ts>(ts)...));
#    endif
                }
            },
            [&](std::exception_ptr ep) { einsums::execution::experimental::set_error(std::move(receiver), std::move(ep)); });
    }

    template <typename... Ts>
    friend void tag_invoke(einsums::execution::experimental::set_value_t, then_receiver_type &&r, Ts &&...ts) noexcept {
        // GCC 7 fails with an internal compiler error unless the actual
        // body is in a helper function.
        r.set_value_helper(std::forward<Ts>(ts)...);
    }
};

template <typename Sender, typename F>
struct then_sender_impl {
    struct then_sender_type;
};

template <typename Sender, typename F>
using then_sender = typename then_sender_impl<Sender, F>::then_sender_type;

template <typename Sender, typename F>
struct then_sender_impl<Sender, F>::then_sender_type {
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<F> f;

    template <typename Tuple>
    struct invoke_result_helper;

    template <template <typename...> class Tuple, typename... Ts>
    struct invoke_result_helper<Tuple<Ts...>> {
        using result_type = std::invoke_result_t<F, Ts...>;
        using type        = std::conditional_t<std::is_void<result_type>::value, Tuple<>, Tuple<result_type>>;
    };

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = einsums::util::detail::unique_t<einsums::util::detail::transform_t<
        typename einsums::execution::experimental::sender_traits<Sender>::template value_types<Tuple, Variant>, invoke_result_helper>>;

    template <template <typename...> class Variant>
    using error_types = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
        typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>, std::exception_ptr>>;

    static constexpr bool sends_done = false;

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, then_sender_type &&s, Receiver &&receiver) {
        return einsums::execution::experimental::connect(std::move(s.sender),
                                                         then_receiver<Receiver, F>{std::forward<Receiver>(receiver), std::move(s.f)});
    }

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, then_sender_type const &r, Receiver &&receiver) {
        return einsums::execution::experimental::connect(r.sender, then_receiver<Receiver, F>{std::forward<Receiver>(receiver), r.f});
    }

    friend decltype(auto) tag_invoke(einsums::execution::experimental::get_env_t, then_sender_type const &s) noexcept {
        return einsums::execution::experimental::get_env(s.sender);
    }
};
} // namespace einsums::then_detail

namespace einsums::execution::experimental {
inline constexpr struct then_t final : einsums::functional::detail::tag_fallback<then_t> {
  private:
    template <typename Sender, typename F>
        requires(is_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(then_t, Sender &&sender, F &&f) {
        return then_detail::then_sender<Sender, F>{std::forward<Sender>(sender), std::forward<F>(f)};
    }

    template <typename F>
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(then_t, F &&f) {
        return detail::partial_algorithm<then_t, F>{std::forward<F>(f)};
    }
} then{};
} // namespace einsums::execution::experimental
#endif
