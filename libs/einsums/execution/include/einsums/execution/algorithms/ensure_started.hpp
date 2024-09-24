//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#endif

#include <einsums/allocator_support/allocator_deleter.hpp>
#include <einsums/allocator_support/internal_allocator.hpp>
#include <einsums/allocator_support/traits/is_allocator.hpp>
#include <einsums/assert.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/datastructures/variant.hpp>
#include <einsums/execution/algorithms/detail/helpers.hpp>
#include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#include <einsums/execution_base/operation_state.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/execution_base/sender.hpp>
#include <einsums/functional/bind_front.hpp>
#include <einsums/functional/detail/tag_fallback_invoke.hpp>
#include <einsums/functional/unique_function.hpp>
#include <einsums/memory/intrusive_ptr.hpp>
#include <einsums/thread_support/atomic_count.hpp>
#include <einsums/type_support/detail/with_result_of.hpp>
#include <einsums/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace einsums::ensure_started_detail {
template <typename Receiver>
struct error_visitor {
    std::decay_t<Receiver> &receiver;

    template <typename Error>
    void operator()(Error &&error) {
        einsums::execution::experimental::set_error(std::move(receiver), std::forward<Error>(error));
    }
};

template <typename Receiver>
struct value_visitor {
    std::decay_t<Receiver> &receiver;

    void operator()(std::monostate) { EINSUMS_UNREACHABLE; }

    template <typename Ts>
    void operator()(Ts &&ts) {
        std::apply(einsums::util::detail::bind_front(einsums::execution::experimental::set_value, std::move(receiver)),
                   std::forward<Ts>(ts));
    }
};

template <typename Sender, typename Allocator>
struct ensure_started_sender_impl {
    struct ensure_started_sender_type;
};

template <typename Sender, typename Allocator>
using ensure_started_sender = typename ensure_started_sender_impl<Sender, Allocator>::ensure_started_sender_type;

template <typename Sender, typename Allocator>
struct ensure_started_sender_impl<Sender, Allocator>::ensure_started_sender_type {
    EINSUMS_STDEXEC_SENDER_CONCEPT

    struct ensure_started_sender_tag {};

    using allocator_type = Allocator;

#if defined(EINSUMS_HAVE_STDEXEC)
    template <typename... Ts>
    using value_types_helper =
        einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_value_t(std::decay_t<Ts>...)>;

    template <typename T>
    using error_types_helper =
        einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_error_t(std::decay_t<T>)>;

    using completion_signatures = einsums::execution::experimental::transform_completion_signatures_of<
        Sender, einsums::execution::experimental::empty_env,
        einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_stopped_t(),
                                                                einsums::execution::experimental::set_error_t(std::exception_ptr)>,
        value_types_helper, error_types_helper>;
#else
    template <typename Tuple>
    struct value_types_helper {
        using type = einsums::util::detail::transform_t<Tuple, std::decay>;
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

    struct shared_state {
        struct ensure_started_receiver;

        using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<shared_state>;
        EINSUMS_NO_UNIQUE_ADDRESS allocator_type alloc;
        using mutex_type = einsums::concurrency::detail::spinlock;
        mutex_type                    mtx;
        einsums::detail::atomic_count reference_count{0};
        std::atomic<bool>             start_called{false};
        std::atomic<bool>             predecessor_done{false};

        using operation_state_type = std::decay_t<einsums::execution::experimental::connect_result_t<Sender, ensure_started_receiver>>;
        // We store the operation state in an optional so that we can
        // reset it as soon as the ensure_started_receiver has been
        // signaled.  This is useful to ensure that resources held by
        // the predecessor work is released as soon as possible.
        std::optional<operation_state_type> os;

        template <typename Tuple>
        struct value_types_helper {
            using type = einsums::util::detail::transform_t<Tuple, std::decay>;
        };
#if defined(EINSUMS_HAVE_STDEXEC)
        using value_type =
            einsums::util::detail::prepend_t<einsums::util::detail::transform_t<typename einsums::execution::experimental::value_types_of_t<
                                                                                    Sender, einsums::execution::experimental::empty_env,
                                                                                    std::tuple, einsums::detail::variant>,
                                                                                value_types_helper>,
                                             einsums::detail::monostate>;
        using error_type = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
            einsums::util::detail::transform_t<einsums::execution::experimental::error_types_of_t<
                                                   Sender, einsums::execution::experimental::empty_env, einsums::detail::variant>,
                                               std::decay>,
            std::exception_ptr>>;
#else
        using value_type = einsums::util::detail::prepend_t<
            einsums::util::detail::transform_t<typename einsums::execution::experimental::sender_traits<Sender>::template value_types<
                                                   std::tuple, einsums::detail::variant>,
                                               value_types_helper>,
            einsums::detail::monostate>;
        using error_type =
            einsums::util::detail::unique_t<einsums::util::detail::prepend_t<error_types<einsums::detail::variant>, std::exception_ptr>>;
#endif
        einsums::detail::variant<einsums::detail::monostate, einsums::execution::detail::stopped_type, error_type, value_type> v;

        using continuation_type = einsums::util::detail::unique_function<void()>;
        std::optional<continuation_type> continuation;

        struct ensure_started_receiver {
            EINSUMS_STDEXEC_RECEIVER_CONCEPT

            einsums::intrusive_ptr<shared_state> state;

            template <typename Error>
            friend void tag_invoke(einsums::execution::experimental::set_error_t, ensure_started_receiver r, Error &&error) noexcept {
                r.state->v.template emplace<error_type>(error_type(std::forward<Error>(error)));
                r.state->set_predecessor_done();
            }

            friend void tag_invoke(einsums::execution::experimental::set_stopped_t, ensure_started_receiver r) noexcept {
                r.state->v.template emplace<einsums::execution::detail::stopped_type>();
                r.state->set_predecessor_done();
            };

            // These typedefs are duplicated from the parent struct. The
            // parent typedefs are not instantiated early enough for use
            // here.
            template <typename Tuple>
            struct value_types_helper {
                using type = einsums::util::detail::transform_t<Tuple, std::decay>;
            };
#if defined(EINSUMS_HAVE_STDEXEC)
            using value_type = einsums::util::detail::prepend_t<
                einsums::util::detail::transform_t<
                    typename einsums::execution::experimental::value_types_of_t<Sender, einsums::execution::experimental::empty_env,
                                                                                std::tuple, einsums::detail::variant>,
                    value_types_helper>,
                einsums::detail::monostate>;
#else
            using value_type = einsums::util::detail::prepend_t<
                einsums::util::detail::transform_t<typename einsums::execution::experimental::sender_traits<Sender>::template value_types<
                                                       std::tuple, einsums::detail::variant>,
                                                   value_types_helper>,
                einsums::detail::monostate>;
#endif

            template <typename... Ts>
            friend auto tag_invoke(einsums::execution::experimental::set_value_t, ensure_started_receiver r, Ts &&...ts) noexcept
                -> decltype(std::declval<einsums::detail::variant<einsums::detail::monostate, value_type>>().template emplace<value_type>(
                                std::make_tuple<>(std::forward<Ts>(ts)...)),
                            void()) {
                r.state->v.template emplace<value_type>(std::make_tuple<>(std::forward<Ts>(ts)...));
                r.state->set_predecessor_done();
            }
        };

        template <typename Sender_, typename = std::enable_if_t<!std::is_same<std::decay_t<Sender_>, shared_state>::value>>
        shared_state(Sender_ &&sender, allocator_type const &alloc) : alloc(alloc) {
            os.emplace(einsums::detail::with_result_of(
                [&]() { return einsums::execution::experimental::connect(std::forward<Sender_>(sender), ensure_started_receiver{this}); }));
        }

        template <typename Receiver>
        struct stopped_error_value_visitor {
            std::decay_t<Receiver> &receiver;

            template <typename T, typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, einsums::detail::monostate> &&
                                                              !std::is_same_v<std::decay_t<T>, value_type>>>
            [[noreturn]] void operator()(T &&) const {
                EINSUMS_UNREACHABLE;
            }

            void operator()(einsums::execution::detail::stopped_type) {
                einsums::execution::experimental::set_stopped(std::move(receiver));
            }

            void operator()(error_type &&error) { einsums::detail::visit(error_visitor<Receiver>{receiver}, std::move(error)); }

            template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, einsums::detail::monostate> &&
                                                              std::is_same_v<std::decay_t<T>, value_type>>>
            void operator()(T &&t) {
                einsums::detail::visit(value_visitor<Receiver>{receiver}, std::forward<T>(t));
            }
        };

        void set_predecessor_done() {
            // We reset the operation state as soon as the predecessor
            // is done to release any resources held by it. Any values
            // sent by the predecessor have already been stored in the
            // shared state by now.
            os.reset();

            predecessor_done = true;

            {
                // We require taking the lock here to synchronize with
                // threads attempting to add continuations to the vector
                // of continuations. However, it is enough to take it
                // once and release it immediately.
                //
                // Without the lock we may not see writes to the vector.
                // With the lock threads attempting to add continuations
                // will either:
                // - See predecessor_done = true in which case they will
                //   call the continuation directly without adding it to
                //   the vector of continuations. Accessing the vector
                //   below without the lock is safe in this case because
                //   the vector is not modified.
                // - See predecessor_done = false and proceed to take
                //   the lock. If they see predecessor_done after taking
                //   the lock they can again release the lock and call
                //   the continuation directly. Accessing the vector
                //   without the lock is again safe because the vector
                //   is not modified.
                // - See predecessor_done = false and proceed to take
                //   the lock. If they see predecessor_done is still
                //   false after taking the lock, they will proceed to
                //   add a continuation to the vector. Since they keep
                //   the lock they can safely write to the vector. This
                //   thread will not proceed past the lock until they
                //   have finished writing to the vector.
                //
                // Importantly, once this thread has taken and released
                // this lock, threads attempting to add continuations to
                // the vector must see predecessor_done = true after
                // taking the lock in their threads and will not add
                // continuations to the vector.
                std::lock_guard<mutex_type> l{mtx};
            }

            if (continuation) {
                (*continuation)();
                continuation.reset();
            }
        }

        template <typename Receiver>
        void add_continuation(Receiver &receiver) {
            EINSUMS_ASSERT(!continuation.has_value());

            if (predecessor_done) {
                // If we read predecessor_done here it means that one of
                // set_error/set_stopped/set_value has been called and
                // values/errors have been stored into the shared state.
                // We can trigger the continuation directly.
                // TODO: Should this preserve the scheduler? It does not
                // if we call set_* inline.
                einsums::detail::visit(stopped_error_value_visitor<Receiver>{receiver}, std::move(v));
            } else {
                // If predecessor_done is false, we have to take the
                // lock to potentially store the continuation.
                std::unique_lock<mutex_type> l{mtx};

                if (predecessor_done) {
                    // By the time the lock has been taken,
                    // predecessor_done might already be true and we can
                    // release the lock early and call the continuation
                    // directly again.
                    l.unlock();
                    einsums::detail::visit(stopped_error_value_visitor<Receiver>{receiver}, std::move(v));
                } else {
                    // If predecessor_done is still false, we store the
                    // continuation. This has to be done while holding
                    // the lock since predecessor signalling completion
                    // may otherwise not see the continuation.
                    continuation.emplace([this, &receiver]() mutable {
                        einsums::detail::visit(stopped_error_value_visitor<Receiver>{receiver}, std::move(v));
                    });
                }
            }
        }

        void start() & noexcept {
            if (!start_called.exchange(true)) {
                EINSUMS_ASSERT(os.has_value());
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                einsums::execution::experimental::start(*os);
            }
        }

        friend void intrusive_ptr_add_ref(shared_state *p) { ++p->reference_count; }

        friend void intrusive_ptr_release(shared_state *p) {
            if (--p->reference_count == 0) {
                allocator_type other_alloc(p->alloc);
                std::allocator_traits<allocator_type>::destroy(other_alloc, p);
                std::allocator_traits<allocator_type>::deallocate(other_alloc, p, 1);
            }
        }
    };

    einsums::intrusive_ptr<shared_state> state;

    template <typename Sender_>
    ensure_started_sender_type(Sender_ &&sender, Allocator const &allocator) {
        using allocator_type   = Allocator;
        using other_allocator  = typename std::allocator_traits<allocator_type>::template rebind_alloc<shared_state>;
        using allocator_traits = std::allocator_traits<other_allocator>;
        using unique_ptr       = std::unique_ptr<shared_state, einsums::detail::allocator_deleter<other_allocator>>;

        other_allocator alloc(allocator);
        unique_ptr      p(allocator_traits::allocate(alloc, 1), einsums::detail::allocator_deleter<other_allocator>{alloc});

        new (p.get()) shared_state{std::forward<Sender_>(sender), allocator};
        state = p.release();

        state->start();
    }

                                ensure_started_sender_type(ensure_started_sender_type &&)      = default;
    ensure_started_sender_type &operator=(ensure_started_sender_type &&)                       = default;
                                ensure_started_sender_type(ensure_started_sender_type const &) = delete;
    ensure_started_sender_type &operator=(ensure_started_sender_type const &)                  = delete;

    template <typename Receiver>
    struct operation_state {
        EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
        einsums::intrusive_ptr<shared_state>             state;

        template <typename Receiver_>
        operation_state(Receiver_ &&receiver, einsums::intrusive_ptr<shared_state> state)
            : receiver(std::forward<Receiver_>(receiver)), state(std::move(state)) {}

                         operation_state(operation_state &&)      = delete;
        operation_state &operator=(operation_state &&)            = delete;
                         operation_state(operation_state const &) = delete;
        operation_state &operator=(operation_state const &)       = delete;

        friend void tag_invoke(einsums::execution::experimental::start_t, operation_state &os) noexcept {
            os.state->add_continuation(os.receiver);
        }
    };

    template <typename Receiver>
    friend operation_state<Receiver> tag_invoke(einsums::execution::experimental::connect_t, ensure_started_sender_type &&s,
                                                Receiver &&receiver) {
        return {std::forward<Receiver>(receiver), std::move(s.state)};
    }

    template <typename Receiver>
    friend operation_state<Receiver> tag_invoke(einsums::execution::experimental::connect_t, ensure_started_sender_type const &,
                                                Receiver &&) {
        static_assert(sizeof(Receiver) == 0, "Are you missing a std::move? The ensure_started sender is not copyable and thus "
                                             "not l-value connectable. Make sure you are passing a non-const r-value reference "
                                             "of the sender.");
    }
};

template <typename Sender, typename Enable = void>
struct is_ensure_started_sender_impl : std::false_type {};

template <typename Sender>
struct is_ensure_started_sender_impl<Sender, std::void_t<typename Sender::ensure_started_sender_tag>> : std::true_type {};

template <typename Sender>
inline constexpr bool is_ensure_started_sender_v = is_ensure_started_sender_impl<std::decay_t<Sender>>::value;
} // namespace einsums::ensure_started_detail

namespace einsums::execution::experimental {
inline constexpr struct ensure_started_t final : einsums::functional::detail::tag_fallback<ensure_started_t> {
  private:
    template <typename Sender>
        requires(is_sender_v<Sender> && !ensure_started_detail::is_ensure_started_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(ensure_started_t, Sender &&sender) {
        return ensure_started_detail::ensure_started_sender<Sender, einsums::detail::internal_allocator<>>{std::forward<Sender>(sender),
                                                                                                           {}};
    }

    template <typename Sender, typename Allocator>
        requires(is_sender_v<Sender> && !ensure_started_detail::is_ensure_started_sender_v<Sender> &&
                 einsums::detail::is_allocator_v<Allocator>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(ensure_started_t, Sender &&sender, Allocator const &allocator) {
        return ensure_started_detail::ensure_started_sender<Sender, Allocator>{std::forward<Sender>(sender), allocator};
    }

    template <typename Sender, typename Allocator>
        requires(ensure_started_detail::is_ensure_started_sender_v<Sender> &&
                 std::is_same_v<typename Sender::allocator_type, std::decay_t<Allocator>>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(ensure_started_t, Sender &&sender, Allocator const &) {
        return std::forward<Sender>(sender);
    }

    template <typename Sender, typename Allocator>
        requires(ensure_started_detail::is_ensure_started_sender_v<Sender> &&
                 !std::is_same_v<typename Sender::allocator_type, std::decay_t<Allocator>>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(ensure_started_t, Sender &&sender, Allocator const &allocator) {
        return ensure_started_detail::ensure_started_sender<Sender, Allocator>{std::forward<Sender>(sender), allocator};
    }

    template <typename Sender>
        requires(ensure_started_detail::is_ensure_started_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(ensure_started_t, Sender &&sender) {
        return std::forward<Sender>(sender);
    }

    template <typename Allocator = einsums::detail::internal_allocator<>>
        requires(einsums::detail::is_allocator_v<Allocator>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(ensure_started_t, Allocator const &allocator = {}) {
        return detail::partial_algorithm<ensure_started_t, Allocator>{allocator};
    }
} ensure_started{};
} // namespace einsums::execution::experimental
