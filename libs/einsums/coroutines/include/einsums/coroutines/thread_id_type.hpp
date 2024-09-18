//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/modules/memory.hpp>
#include <einsums/thread_support/atomic_count.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string_view>
#include <utility>

namespace einsums::threads::detail {

struct thread_id {
  private:
    using thread_id_repr = void *;

  public:
    thread_id() noexcept = default;

    thread_id(thread_id const &)            = default;
    thread_id &operator=(thread_id const &) = default;

    constexpr thread_id(thread_id &&rhs) noexcept : _thrd(rhs._thrd) { rhs._thrd = nullptr; }
    constexpr thread_id &operator=(thread_id &&rhs) noexcept {
        _thrd     = rhs._thrd;
        rhs._thrd = nullptr;
        return *this;
    }

    explicit constexpr thread_id(thread_id_repr const &thrd) noexcept : _thrd(thrd) {}
    constexpr thread_id &operator=(thread_id_repr const &rhs) noexcept {
        _thrd = rhs;
        return *this;
    }

    explicit constexpr operator bool() const noexcept { return nullptr != _thrd; }

    constexpr thread_id_repr get() const noexcept { return _thrd; }

    constexpr void reset() noexcept { _thrd = nullptr; }

    friend constexpr bool operator==(std::nullptr_t, thread_id const &rhs) noexcept { return nullptr == rhs._thrd; }

    friend constexpr bool operator!=(std::nullptr_t, thread_id const &rhs) noexcept { return nullptr != rhs._thrd; }

    friend constexpr bool operator==(thread_id const &lhs, std::nullptr_t) noexcept { return nullptr == lhs._thrd; }

    friend constexpr bool operator!=(thread_id const &lhs, std::nullptr_t) noexcept { return nullptr != lhs._thrd; }

    friend constexpr bool operator==(thread_id const &lhs, thread_id const &rhs) noexcept { return lhs._thrd == rhs._thrd; }

    friend constexpr bool operator!=(thread_id const &lhs, thread_id const &rhs) noexcept { return lhs._thrd != rhs._thrd; }

    friend constexpr bool operator<(thread_id const &lhs, thread_id const &rhs) noexcept {
        return std::less<thread_id_repr>{}(lhs._thrd, rhs._thrd);
    }

    friend constexpr bool operator>(thread_id const &lhs, thread_id const &rhs) noexcept {
        return std::less<thread_id_repr>{}(rhs._thrd, lhs._thrd);
    }

    friend constexpr bool operator<=(thread_id const &lhs, thread_id const &rhs) noexcept { return !(rhs > lhs); }

    friend constexpr bool operator>=(thread_id const &lhs, thread_id const &rhs) noexcept { return !(rhs < lhs); }

    template <typename Char, typename Traits>
    friend std::basic_ostream<Char, Traits> &operator<<(std::basic_ostream<Char, Traits> &os, thread_id const &id) {
        os << id.get();
        return os;
    }

  private:
    thread_id_repr _thrd = nullptr;
};

enum class thread_id_addref { yes, no };

struct thread_data_reference_counting;

void intrusive_ptr_add_ref(thread_data_reference_counting *p);
void intrusive_ptr_release(thread_data_reference_counting *p);

struct thread_data_reference_counting {
    explicit thread_data_reference_counting(thread_id_addref addref = thread_id_addref::yes) : _count(addref == thread_id_addref::yes) {}

    virtual ~thread_data_reference_counting() = default;
    virtual void destroy_thread()             = 0;

    // reference counting
    friend void intrusive_ptr_add_ref(thread_data_reference_counting *p) { ++p->_count; }

    friend void intrusive_ptr_release(thread_data_reference_counting *p) {
        EINSUMS_ASSERT(p->_count != 0);
        if (--p->_count == 0) {
            // give this object back to the system
            p->destroy_thread();
        }
    }

  private:
    ::einsums::detail::atomic_count _count;
};

struct thread_id_ref {
  private:
    using thread_id_repr = einsums::intrusive_ptr<detail::thread_data_reference_counting>;

  public:
    thread_id_ref() noexcept = default;

    thread_id_ref(thread_id_ref const &)            = default;
    thread_id_ref &operator=(thread_id_ref const &) = default;

    thread_id_ref(thread_id_ref &&rhs) noexcept            = default;
    thread_id_ref &operator=(thread_id_ref &&rhs) noexcept = default;

    explicit thread_id_ref(thread_id_repr const &thrd) noexcept : _thrd(thrd) {}
    explicit thread_id_ref(thread_id_repr &&thrd) noexcept : _thrd(std::move(thrd)) {}

    thread_id_ref &operator=(thread_id_repr const &rhs) noexcept {
        _thrd = rhs;
        return *this;
    }
    thread_id_ref &operator=(thread_id_repr &&rhs) noexcept {
        _thrd = std::move(rhs);
        return *this;
    }

    using thread_repr = detail::thread_data_reference_counting;

    explicit thread_id_ref(thread_repr *thrd, thread_id_addref addref = thread_id_addref::yes) noexcept
        : _thrd(thrd, addref == thread_id_addref::yes) {}

    thread_id_ref &operator=(thread_repr *rhs) noexcept {
        _thrd.reset(rhs);
        return *this;
    }

    thread_id_ref(thread_id const &noref) : _thrd(static_cast<thread_repr *>(noref.get())) {}

    thread_id_ref(thread_id &&noref) noexcept : _thrd(static_cast<thread_repr *>(noref.get())) { noref.reset(); }

    thread_id_ref &operator=(thread_id const &noref) {
        _thrd.reset(static_cast<thread_repr *>(noref.get()));
        return *this;
    }

    thread_id_ref &operator=(thread_id &&noref) noexcept {
        _thrd.reset(static_cast<thread_repr *>(noref.get()));
        noref.reset();
        return *this;
    }

    explicit operator bool() const noexcept { return nullptr != _thrd; }

    thread_id noref() const noexcept { return thread_id(_thrd.get()); }

    thread_id_repr  &get()  &noexcept { return _thrd; }
    thread_id_repr &&get() && noexcept { return std::move(_thrd); }

    thread_id_repr const &get() const & noexcept { return _thrd; }

    void reset() noexcept { _thrd.reset(); }

    void reset(thread_repr *thrd, bool add_ref = true) noexcept { _thrd.reset(thrd, add_ref); }

    constexpr thread_repr *detach() noexcept { return _thrd.detach(); }

    friend bool operator==(std::nullptr_t, thread_id_ref const &rhs) noexcept { return nullptr == rhs._thrd; }

    friend bool operator!=(std::nullptr_t, thread_id_ref const &rhs) noexcept { return nullptr != rhs._thrd; }

    friend bool operator==(thread_id_ref const &lhs, std::nullptr_t) noexcept { return nullptr == lhs._thrd; }

    friend bool operator!=(thread_id_ref const &lhs, std::nullptr_t) noexcept { return nullptr != lhs._thrd; }

    friend bool operator==(thread_id_ref const &lhs, thread_id_ref const &rhs) noexcept { return lhs._thrd == rhs._thrd; }

    friend bool operator!=(thread_id_ref const &lhs, thread_id_ref const &rhs) noexcept { return lhs._thrd != rhs._thrd; }

    friend bool operator<(thread_id_ref const &lhs, thread_id_ref const &rhs) noexcept {
        return std::less<thread_repr const *>{}(lhs._thrd.get(), rhs._thrd.get());
    }

    friend bool operator>(thread_id_ref const &lhs, thread_id_ref const &rhs) noexcept {
        return std::less<thread_repr const *>{}(rhs._thrd.get(), lhs._thrd.get());
    }

    friend bool operator<=(thread_id_ref const &lhs, thread_id_ref const &rhs) noexcept { return !(rhs > lhs); }

    friend bool operator>=(thread_id_ref const &lhs, thread_id_ref const &rhs) noexcept { return !(rhs < lhs); }

    template <typename Char, typename Traits>
    friend std::basic_ostream<Char, Traits> &operator<<(std::basic_ostream<Char, Traits> &os, thread_id_ref const &id) {
        os << id.get();
        return os;
    }

  private:
    thread_id_repr _thrd;
};

inline constexpr thread_id const invalid_thread_id;

} // namespace einsums::threads::detail

namespace std {
template <>
struct hash<::einsums::threads::detail::thread_id> {
    std::size_t EINSUMS_STATIC_CALL_OPERATOR(::einsums::threads::detail::thread_id const &v) noexcept {
        std::hash<std::size_t> hasher_;
        return hasher_(reinterpret_cast<std::size_t>(v.get()));
    }
};

template <>
struct hash<::einsums::threads::detail::thread_id_ref> {
    std::size_t EINSUMS_STATIC_CALL_OPERATOR(::einsums::threads::detail::thread_id_ref const &v) noexcept {
        std::hash<std::size_t> hasher_;
        return hasher_(reinterpret_cast<std::size_t>(v.get().get()));
    }
};
} // namespace std

template <>
struct fmt::formatter<einsums::threads::detail::thread_id> : fmt::formatter<void *> {
    template <typename FormatContext>
    auto format(einsums::threads::detail::thread_id id, FormatContext &ctx) const {
        return fmt::formatter<void *>::format(static_cast<void *>(id.get()), ctx);
    }
};

template <>
struct fmt::formatter<einsums::threads::detail::thread_id_ref> : fmt::formatter<void *> {
    template <typename FormatContext>
    auto format(einsums::threads::detail::thread_id_ref id, FormatContext &ctx) const {
        return fmt::formatter<void *>::format(static_cast<void *>(id.noref().get()), ctx);
    }
};
