//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
#    include <einsums/datastructures/variant.hpp>
#    include <einsums/errors/try_catch_exception_ptr.hpp>
#    include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#    include <einsums/execution/algorithms/then.hpp>
#    include <einsums/execution_base/completion_scheduler.hpp>
#    include <einsums/execution_base/receiver.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/functional/detail/tag_priority_invoke.hpp>
#    include <einsums/iterator/counting_shape.hpp>
#    include <einsums/type_support/pack.hpp>

#    include <exception>
#    include <iterator>
#    include <tuple>
#    include <type_traits>
#    include <utility>

namespace einsums::bulk_detail {
template <typename Sender, typename Shape, typename F>
struct bulk_sender_impl {
    struct bulk_sender_type;
};

template <typename Sender, typename Shape, typename F>
using bulk_sender = typename bulk_sender_impl<Sender, Shape, F>::bulk_sender_type;

template <typename Sender, typename Shape, typename F>
struct bulk_sender_impl<Sender, Shape, F>::bulk_sender_type {
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<F> f;

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = typename einsums::execution::experimental::sender_traits<Sender>::template value_types<Tuple, Variant>;

    template <template <typename...> class Variant>
    using error_types = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
        typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>, std::exception_ptr>>;

    static constexpr bool sends_done = false;

    friend constexpr decltype(auto) tag_invoke(einsums::execution::experimental::get_env_t, bulk_sender_type const &s) noexcept {
        return einsums::execution::experimental::get_env(s.sender);
    }

    template <typename Receiver>
    struct bulk_receiver {
        EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
        EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
        EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<F> f;

        template <typename Receiver_, typename Shape_, typename F_>
        bulk_receiver(Receiver_ &&receiver, Shape_ &&shape, F_ &&f)
            : receiver(std::forward<Receiver_>(receiver)), shape(std::forward<Shape_>(shape)), f(std::forward<F_>(f)) {}

        template <typename Error>
        friend void tag_invoke(einsums::execution::experimental::set_error_t, bulk_receiver &&r, Error &&error) noexcept {
            einsums::execution::experimental::set_error(std::move(r.receiver), std::forward<Error>(error));
        }

        friend void tag_invoke(einsums::execution::experimental::set_stopped_t, bulk_receiver &&r) noexcept {
            einsums::execution::experimental::set_stopped(std::move(r.receiver));
        }

        template <typename... Ts>
        void set_value(Ts &&...ts) {
            einsums::detail::try_catch_exception_ptr(
                [&]() {
                    for (auto const &s : shape) {
                        EINSUMS_INVOKE(f, s, ts...);
                    }
                    einsums::execution::experimental::set_value(std::move(receiver), std::forward<Ts>(ts)...);
                },
                [&](std::exception_ptr ep) { einsums::execution::experimental::set_error(std::move(receiver), std::move(ep)); });
        }

        template <typename... Ts>
        friend auto tag_invoke(einsums::execution::experimental::set_value_t, bulk_receiver &&r, Ts &&...ts) noexcept
            -> decltype(einsums::execution::experimental::set_value(std::declval<std::decay_t<Receiver> &&>(), std::forward<Ts>(ts)...),
                        void()) {
            // set_value is in a member function only because of a
            // compiler bug in GCC 7. When the body of set_value is
            // inlined here compilation fails with an internal compiler
            // error.
            r.set_value(std::forward<Ts>(ts)...);
        }
    };

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, bulk_sender_type &&s, Receiver &&receiver) {
        return einsums::execution::experimental::connect(
            std::move(s.sender), bulk_receiver<Receiver>(std::forward<Receiver>(receiver), std::move(s.shape), std::move(s.f)));
    }

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, bulk_sender_type const &s, Receiver &&receiver) {
        return einsums::execution::experimental::connect(s.sender, bulk_receiver<Receiver>(std::forward<Receiver>(receiver), s.shape, s.f));
    }
};
} // namespace einsums::bulk_detail

namespace einsums::execution::experimental {
inline constexpr struct bulk_t final : einsums::functional::detail::tag_priority<bulk_t> {
  private:
    template <typename Sender, typename Shape, typename F>
        requires(is_sender_v<Sender> && einsums::execution::experimental::detail::is_completion_scheduler_tag_invocable_v<
                                            einsums::execution::experimental::set_value_t, Sender, bulk_t, Shape, F>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_override_invoke(bulk_t, Sender &&sender, Shape const &shape, F &&f) {
        auto scheduler = einsums::execution::experimental::get_completion_scheduler<einsums::execution::experimental::set_value_t>(
            einsums::execution::experimental::get_env(sender));
        return einsums::functional::detail::tag_invoke(bulk_t{}, std::move(scheduler), std::forward<Sender>(sender), shape,
                                                       std::forward<F>(f));
    }

    template <typename Sender, typename Shape, typename F>
        requires(is_sender_v<Sender> && std::is_integral<Shape>::value)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(bulk_t, Sender &&sender, Shape const &shape, F &&f) {
        return bulk_detail::bulk_sender<Sender, einsums::util::detail::counting_shape_type<Shape>, F>{
            std::forward<Sender>(sender), einsums::util::detail::make_counting_shape(shape), std::forward<F>(f)};
    }

    template <typename Sender, typename Shape, typename F>
        requires(is_sender_v<Sender> && !std::is_integral<std::decay_t<Shape>>::value)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(bulk_t, Sender &&sender, Shape &&shape, F &&f) {
        return bulk_detail::bulk_sender<Sender, Shape, F>{std::forward<Sender>(sender), std::forward<Shape>(shape), std::forward<F>(f)};
    }

    template <typename Shape, typename F>
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(bulk_t, Shape &&shape, F &&f) {
        return detail::partial_algorithm<bulk_t, Shape, F>{std::forward<Shape>(shape), std::forward<F>(f)};
    }
} bulk{};
} // namespace einsums::execution::experimental
#endif
