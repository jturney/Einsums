//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
// #include <einsums/concepts/concepts.hpp>
#include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#include <einsums/execution_base/operation_state.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/execution_base/sender.hpp>
#include <einsums/functional/bind_front.hpp>
#include <einsums/functional/detail/tag_fallback_invoke.hpp>
#include <einsums/type_support/detail/with_result_of.hpp>
#include <einsums/type_support/pack.hpp>

#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace einsums::drop_op_state_detail {
template <typename OpState>
struct drop_op_state_receiver_impl {
    struct drop_op_state_receiver_type;
};

template <typename OpState>
using drop_op_state_receiver = typename drop_op_state_receiver_impl<OpState>::drop_op_state_receiver_type;

template <typename OpState>
struct drop_op_state_receiver_impl<OpState>::drop_op_state_receiver_type {
    EINSUMS_STDEXEC_RECEIVER_CONCEPT

    OpState *op_state = nullptr;

    template <typename Error>
    friend void tag_invoke(einsums::execution::experimental::set_error_t, drop_op_state_receiver_type r, Error &&error) noexcept {
        EINSUMS_ASSERT(r.op_state != nullptr);
        EINSUMS_ASSERT(r.op_state->op_state.has_value());

        try {
            auto error_local = std::forward<Error>(error);
            r.op_state->op_state.reset();

            einsums::execution::experimental::set_error(std::move(r.op_state->receiver), std::move(error_local));
        } catch (...) {
            r.op_state->op_state.reset();

            einsums::execution::experimental::set_error(std::move(r.op_state->receiver), std::current_exception());
        }
    }

    friend void tag_invoke(einsums::execution::experimental::set_stopped_t, drop_op_state_receiver_type r) noexcept {
        EINSUMS_ASSERT(r.op_state != nullptr);
        EINSUMS_ASSERT(r.op_state->op_state.has_value());

        r.op_state->op_state.reset();

        einsums::execution::experimental::set_stopped(std::move(r.op_state->receiver));
    };

    template <typename... Ts>
    friend void tag_invoke(einsums::execution::experimental::set_value_t, drop_op_state_receiver_type r, Ts &&...ts) noexcept {
        EINSUMS_ASSERT(r.op_state != nullptr);
        EINSUMS_ASSERT(r.op_state->op_state.has_value());

        try {
            std::tuple<std::decay_t<Ts>...> ts_local(std::forward<Ts>(ts)...);
            r.op_state->op_state.reset();

            std::apply(einsums::util::detail::bind_front(einsums::execution::experimental::set_value, std::move(r.op_state->receiver)),
                       std::move(ts_local));
        } catch (...) {
            r.op_state->op_state.reset();

            einsums::execution::experimental::set_error(std::move(r.op_state->receiver), std::current_exception());
        }
    }

    friend constexpr einsums::execution::experimental::empty_env tag_invoke(einsums::execution::experimental::get_env_t,
                                                                            drop_op_state_receiver_type const &) noexcept {
        return {};
    }
};

template <typename Sender, typename Receiver>
struct drop_op_state_op_state_impl {
    struct drop_op_state_op_state_type;
};

template <typename Sender, typename Receiver>
using drop_op_state_op_state = typename drop_op_state_op_state_impl<Sender, Receiver>::drop_op_state_op_state_type;

template <typename Sender, typename Receiver>
struct drop_op_state_op_state_impl<Sender, Receiver>::drop_op_state_op_state_type {
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
    using operation_state_type =
        einsums::execution::experimental::connect_result_t<Sender, drop_op_state_receiver<drop_op_state_op_state_type>>;
    std::optional<operation_state_type> op_state;

    template <typename Receiver_>
    drop_op_state_op_state_type(std::decay_t<Sender> sender, Receiver_ &&receiver)
        : receiver(std::forward<Receiver_>(receiver)), op_state(einsums::detail::with_result_of([&]() mutable {
              return einsums::execution::experimental::connect(std::move(sender),
                                                               drop_op_state_receiver<drop_op_state_op_state_type>{this});
          })) {}
                                 drop_op_state_op_state_type(drop_op_state_op_state_type &)       = delete;
    drop_op_state_op_state_type &operator=(drop_op_state_op_state_type &)                         = delete;
                                 drop_op_state_op_state_type(drop_op_state_op_state_type const &) = delete;
    drop_op_state_op_state_type &operator=(drop_op_state_op_state_type const &)                   = delete;

    friend void tag_invoke(einsums::execution::experimental::start_t, drop_op_state_op_state_type &os) noexcept {
        EINSUMS_ASSERT(os.op_state.has_value());
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        einsums::execution::experimental::start(*(os.op_state));
    }
};

template <typename Sender>
struct drop_op_state_sender_impl {
    struct drop_op_state_sender_type;
};

template <typename Sender>
using drop_op_state_sender = typename drop_op_state_sender_impl<Sender>::drop_op_state_sender_type;

template <typename Sender>
struct drop_op_state_sender_impl<Sender>::drop_op_state_sender_type {
    EINSUMS_STDEXEC_SENDER_CONCEPT

    std::decay_t<Sender> sender;

#if defined(EINSUMS_HAVE_STDEXEC)
    template <typename... Ts>
    using value_types_helper =
        einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_value_t(std::decay_t<Ts> &&...)>;

    using completion_signatures = einsums::execution::experimental::transform_completion_signatures_of<
        std::decay_t<Sender>, einsums::execution::experimental::empty_env,
        einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_error_t(std::exception_ptr)>,
        value_types_helper>;
#else
    template <typename Tuple>
    struct value_types_helper {
        using type = einsums::util::detail::transform_t<einsums::util::detail::transform_t<Tuple, std::decay>, std::add_rvalue_reference>;
    };

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = einsums::util::detail::transform_t<
        typename einsums::execution::experimental::sender_traits<Sender>::template value_types<Tuple, Variant>, value_types_helper>;

    template <template <typename...> class Variant>
    using error_types = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
        einsums::util::detail::transform_t<typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>,
                                           std::decay>,
        std::exception_ptr>>;

    static constexpr bool sends_done = false;
#endif

    template <typename Sender_, typename Enable = std::enable_if_t<!std::is_same_v<std::decay_t<Sender_>, drop_op_state_sender_type>>>
    explicit drop_op_state_sender_type(Sender_ &&sender) : sender(std::forward<Sender_>(sender)) {}

                               drop_op_state_sender_type(drop_op_state_sender_type const &) = default;
    drop_op_state_sender_type &operator=(drop_op_state_sender_type const &)                 = default;
                               drop_op_state_sender_type(drop_op_state_sender_type &&)      = default;
    drop_op_state_sender_type &operator=(drop_op_state_sender_type &&)                      = default;

    template <typename Receiver>
    friend drop_op_state_op_state<Sender, Receiver> tag_invoke(einsums::execution::experimental::connect_t, drop_op_state_sender_type &&s,
                                                               Receiver &&receiver) {
        return {std::move(s.sender), std::forward<Receiver>(receiver)};
    }

    template <typename Receiver>
    friend drop_op_state_op_state<Sender, Receiver> tag_invoke(einsums::execution::experimental::connect_t,
                                                               drop_op_state_sender_type const &s, Receiver &&receiver) {
        return {s.sender, std::forward<Receiver>(receiver)};
    }
};
} // namespace einsums::drop_op_state_detail

namespace einsums::execution::experimental {
inline constexpr struct drop_operation_state_t final {
    template <typename Sender>
        requires(is_sender_v<Sender>)
    constexpr EINSUMS_FORCEINLINE auto EINSUMS_STATIC_CALL_OPERATOR(Sender &&sender) {
        return drop_op_state_detail::drop_op_state_sender<Sender>{std::forward<Sender>(sender)};
    }

    constexpr EINSUMS_FORCEINLINE auto EINSUMS_STATIC_CALL_OPERATOR() { return detail::partial_algorithm<drop_operation_state_t>{}; }
} drop_operation_state{};
} // namespace einsums::execution::experimental
