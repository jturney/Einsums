//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/errors/throw_exception.hpp>
#include <einsums/execution_base/agent_base.hpp>
#include <einsums/execution_base/context_base.hpp>
#include <einsums/execution_base/detail/spinlock_deadlock_detection.hpp>
#include <einsums/execution_base/this_thread.hpp>
#include <einsums/logging.hpp>
#include <einsums/timing/steady_clock.hpp>
#include <einsums/type_support/unused.hpp>

#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
#    include <einsums/errors/throw_exception.hpp>
#endif

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/std.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#if defined(EINSUMS_WINDOWS)
#    include <windows.h>
#else
#    ifndef _AIX
#        include <sched.h>
#    else
// AIX's sched.h defines ::var which sometimes conflicts with Lambda's var
extern "C" int sched_yield(void);
#    endif
#    include <time.h>
#endif

namespace einsums::execution {

namespace detail {
namespace {

struct default_context : context_base {
    resource_base const &resource() const override { return resource_; }
    resource_base        resource_;
};

struct default_agent : detail::agent_base {
    default_agent();

    std::string description() const override { return fmt::format("{}", id_); }

    default_context const &context() const override { return context_; }

    void yield(char const *desc) override;
    void yield_k(std::size_t k, char const *desc) override;
    void spin_k(std::size_t k, char const *desc) override;
    void suspend(char const *desc) override;
    void resume(char const *desc) override;
    void abort(char const *desc) override;
    void sleep_for(einsums::chrono::steady_duration const &sleep_duration, char const *desc) override;
    void sleep_until(einsums::chrono::steady_time_point const &sleep_time, char const *desc) override;

  private:
    bool                    running_;
    bool                    aborted_;
    std::thread::id         id_;
    std::mutex              mtx_;
    std::condition_variable suspend_cv_;
    std::condition_variable resume_cv_;

    default_context context_;
};

default_agent::default_agent() : running_(true), aborted_(false), id_(std::this_thread::get_id()) {
}

void default_agent::yield(char const * /* desc */) {
#if defined(EINSUMS_WINDOWS)
    Sleep(0);
#else
    sched_yield();
#endif
}

void default_agent::yield_k(std::size_t k, char const * /* desc */) {
    if (k < 16) {
        EINSUMS_SMT_PAUSE;
    } else if (k < 32 || k & 1) {
#if defined(EINSUMS_WINDOWS)
        Sleep(0);
#else
        sched_yield();
#endif
    } else {
#if defined(EINSUMS_WINDOWS)
        Sleep(1);
#else
        // g++ -Wextra warns on {} or {0}
        struct timespec rqtp = {0, 0};

        // POSIX says that timespec has tv_sec and tv_nsec
        // But it doesn't guarantee order or placement

        rqtp.tv_sec  = 0;
        rqtp.tv_nsec = 1000;

        nanosleep(&rqtp, nullptr);
#endif
    }
}

void default_agent::spin_k(std::size_t k, char const * /* desc */) {
    for (std::size_t i = 0; i < k; ++i) {
        EINSUMS_SMT_PAUSE;
    }
}

void default_agent::suspend(char const * /* desc */) {
    std::unique_lock<std::mutex> l(mtx_);
    EINSUMS_ASSERT(running_);

    running_ = false;
    resume_cv_.notify_all();

    suspend_cv_.wait(l, [&] { return running_; });

    if (aborted_) {
        EINSUMS_THROW_EXCEPTION(einsums::error::yield_aborted, "std::thread({}) aborted (yield returned wait_abort)", id_);
    }
}

void default_agent::resume(char const * /* desc */) {
    std::unique_lock<std::mutex> l(mtx_);
    resume_cv_.wait(l, [&] { return !running_; });
    running_ = true;
    suspend_cv_.notify_one();
}

void default_agent::abort(char const * /* desc */) {
    std::unique_lock<std::mutex> l(mtx_);
    resume_cv_.wait(l, [&] { return !running_; });
    running_ = true;
    aborted_ = true;
    suspend_cv_.notify_one();
}

void default_agent::sleep_for(einsums::chrono::steady_duration const &sleep_duration, char const * /* desc */) {
    std::this_thread::sleep_for(sleep_duration.value());
}

void default_agent::sleep_until(einsums::chrono::steady_time_point const &sleep_time, char const * /* desc */) {
    std::this_thread::sleep_until(sleep_time.value());
}
} // namespace
} // namespace detail

namespace detail {
agent_base &get_default_agent() {
    static thread_local default_agent agent;
    return agent;
}
} // namespace detail

namespace this_thread::detail {

struct agent_storage {
    agent_storage() : impl_(&einsums::execution::detail::get_default_agent()) {}

    execution::detail::agent_base *set(execution::detail::agent_base *context) noexcept {
        std::swap(context, impl_);
        return context;
    }

    execution::detail::agent_base *impl_;
};

agent_storage *get_agent_storage() {
    static thread_local agent_storage storage;
    return &storage;
}

reset_agent::reset_agent(detail::agent_storage *storage, execution::detail::agent_base &impl)
    : storage_(storage), old_(storage_->set(&impl)) {
}

reset_agent::reset_agent(execution::detail::agent_base &impl) : reset_agent(detail::get_agent_storage(), impl) {
}

reset_agent::~reset_agent() {
    storage_->set(old_);
}

einsums::execution::detail::agent_ref agent() {
    return einsums::execution::detail::agent_ref(detail::get_agent_storage()->impl_);
}

void yield(char const *desc) {
    agent().yield(desc);
}

void check_spinlock_deadlock(std::size_t k, char const *name, char const *desc) {
#ifdef EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION
    if (einsums::util::detail::get_spinlock_break_on_deadlock_enabled()) {
        if (desc == nullptr) {
            desc = "";
        }

        auto const deadlock_detection_limit = einsums::util::detail::get_spinlock_deadlock_detection_limit();
        if (k >= deadlock_detection_limit) {
            EINSUMS_THROW_EXCEPTION(einsums::error::deadlock,
                                    "{} spun {} times. This may indicate a deadlock in your "
                                    "application or a bug in einsums. Stopping because "
                                    "einsums.spinlock_deadlock_detection_limit={}.",
                                    name, k, deadlock_detection_limit);
        }

        auto const deadlock_warning_limit = einsums::util::detail::get_spinlock_deadlock_warning_limit();
        if (k >= deadlock_warning_limit && k % deadlock_warning_limit == 0) {
            EINSUMS_LOG(warn,
                        "desc: {}. {} already spun {} times "
                        "(einsums.spinlock_deadlock_warning_limit={}). This may indicate a deadlock "
                        "in your application or a bug in einsums. Stopping after "
                        "einsums.spinlock_deadlock_detection_limit={} iterations.",
                        desc, name, k, deadlock_warning_limit, deadlock_detection_limit);
        }
    }
#else
    EINSUMS_UNUSED(k);
    EINSUMS_UNUSED(name);
    EINSUMS_UNUSED(desc);
#endif
}

void yield_k(std::size_t k, char const *desc) {
    check_spinlock_deadlock(k, "yield_k", desc);
    agent().yield_k(k, desc);
}

void spin_k(std::size_t k, char const *desc) {
    check_spinlock_deadlock(k, "spin_k", desc);
    agent().spin_k(k, desc);
}

void suspend(char const *desc) {
    agent().suspend(desc);
}
} // namespace this_thread::detail
} // namespace einsums::execution
