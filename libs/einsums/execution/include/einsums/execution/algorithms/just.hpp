//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
#    include <einsums/datastructures/member_pack.hpp>
#    include <einsums/errors/try_catch_exception_ptr.hpp>
#    include <einsums/execution_base/receiver.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/type_support/pack.hpp>

#    include <cstddef>
#    include <exception>
#    include <stdexcept>
#    include <utility>

namespace einsums::just_detail {
template <typename Is, typename... Ts>
struct just_sender_impl;

template <typename std::size_t... Is, typename... Ts>
struct just_sender_impl<einsums::util::detail::index_pack<Is...>, Ts...> {
    struct just_sender_type {
        einsums::util::detail::member_pack_for<std::decay_t<Ts>...> ts;

        constexpr just_sender_type() = default;

        template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, just_sender_type>>>
        explicit constexpr just_sender_type(T &&t) : ts(std::piecewise_construct, std::forward<T>(t)) {}

        template <typename T0, typename T1, typename... Ts_>
        explicit constexpr just_sender_type(T0 &&t0, T1 &&t1, Ts_ &&...ts)
            : ts(std::piecewise_construct, std::forward<T0>(t0), std::forward<T1>(t1), std::forward<Ts_>(ts)...) {}

                          just_sender_type(just_sender_type &&)      = default;
                          just_sender_type(just_sender_type const &) = default;
        just_sender_type &operator=(just_sender_type &&)             = default;
        just_sender_type &operator=(just_sender_type const &)        = default;

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<std::decay_t<Ts>...>>;

        template <template <typename...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename Receiver>
        struct operation_state {
            EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver>            receiver;
            einsums::util::detail::member_pack_for<std::decay_t<Ts>...> ts;

            template <typename Receiver_>
            operation_state(Receiver_ &&receiver, einsums::util::detail::member_pack_for<std::decay_t<Ts>...> ts)
                : receiver(std::forward<Receiver_>(receiver)), ts(std::move(ts)) {}

                             operation_state(operation_state &&)      = delete;
            operation_state &operator=(operation_state &&)            = delete;
                             operation_state(operation_state const &) = delete;
            operation_state &operator=(operation_state const &)       = delete;

            friend void tag_invoke(einsums::execution::experimental::start_t, operation_state &os) noexcept {
                einsums::detail::try_catch_exception_ptr(
                    [&]() { einsums::execution::experimental::set_value(std::move(os.receiver), std::move(os.ts).template get<Is>()...); },
                    [&](std::exception_ptr ep) { einsums::execution::experimental::set_error(std::move(os.receiver), std::move(ep)); });
            }
        };

        template <typename Receiver>
        friend auto tag_invoke(einsums::execution::experimental::connect_t, just_sender_type &&s, Receiver &&receiver) {
            return operation_state<Receiver>{std::forward<Receiver>(receiver), std::move(s.ts)};
        }

        template <typename Receiver>
        friend auto tag_invoke(einsums::execution::experimental::connect_t, just_sender_type const &s, Receiver &&receiver) {
            return operation_state<Receiver>{std::forward<Receiver>(receiver), s.ts};
        }
    };
};

template <typename Is, typename... Ts>
using just_sender = typename just_sender_impl<Is, Ts...>::just_sender_type;
} // namespace einsums::just_detail

namespace einsums::execution::experimental {
inline constexpr struct just_t final {
    template <typename... Ts>
    constexpr EINSUMS_FORCEINLINE auto EINSUMS_STATIC_CALL_OPERATOR(Ts &&...ts) {
        return just_detail::just_sender<typename einsums::util::detail::make_index_pack<sizeof...(Ts)>::type, Ts...>{
            std::forward<Ts>(ts)...};
    }
} just{};
} // namespace einsums::execution::experimental
#endif
