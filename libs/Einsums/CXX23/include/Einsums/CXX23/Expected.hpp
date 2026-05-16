//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Expected.hpp
/// @brief C++23 std::expected<T, E> backport for C++20.
///
/// A vocabulary type for functions that can fail: holds either a value (T)
/// or an error (E). Unlike exceptions, the caller MUST handle the error —
/// no silent propagation. Unlike error codes, the value type is preserved.
///
/// When C++23 is available, define EINSUMS_USE_STD_EXPECTED to use the
/// standard library version.

#include <Einsums/Config.hpp>

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202211L
#    include <expected>
#endif

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>
#include <variant>

namespace einsums {

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202211L

// Use the standard library version when available
template <typename T, typename E>
using expected = std::expected<T, E>;

template <typename E>
using unexpected = std::unexpected<E>;

#else

/**
 * @brief Tag type wrapping an error value for expected.
 *
 * @tparam E Error type.
 *
 * @code
 * return einsums::unexpected(std::string("file not found"));
 * @endcode
 */
template <typename E>
class unexpected {
  public:
    constexpr unexpected(unexpected const &)            = default;
    constexpr unexpected(unexpected &&)                 = default;
    constexpr unexpected &operator=(unexpected const &) = default;
    constexpr unexpected &operator=(unexpected &&)      = default;

    template <typename Err = E>
        requires(!std::same_as<std::remove_cvref_t<Err>, unexpected> && std::constructible_from<E, Err>)
    constexpr explicit unexpected(Err &&err) : error_(std::forward<Err>(err)) {}

    [[nodiscard]] constexpr E const  &error() const  &{ return error_; }
    [[nodiscard]] constexpr E        &error()        &{ return error_; }
    [[nodiscard]] constexpr E const &&error() const && { return std::move(error_); }
    [[nodiscard]] constexpr E       &&error()       &&{ return std::move(error_); }

  private:
    E error_;
};

// Deduction guide
template <typename E>
unexpected(E) -> unexpected<E>;

/**
 * @brief A value-or-error type (C++23 std::expected backport).
 *
 * Holds either a value of type T or an error of type E. The caller must
 * check before accessing the value.
 *
 * @tparam T Value type.
 * @tparam E Error type.
 *
 * @par Example
 * @code
 * einsums::expected<HardwareProfile, std::string> load_profile(path) {
 *     if (!file_exists(path))
 *         return einsums::unexpected("file not found: " + path);
 *     return HardwareProfile{...};
 * }
 *
 * auto result = load_profile("hw.json");
 * if (result) {
 *     use(result.value());
 * } else {
 *     log_error(result.error());
 * }
 * @endcode
 */
template <typename T, typename E>
class expected {
  public:
    using value_type = T;
    using error_type = E;

    // ── Constructors ───────────────────────────────────────────────────────

    /// Default: constructs with a value-initialized T.
    constexpr expected()
        requires std::default_initializable<T>
        : storage_(T{}) {}

    /// Construct from a value.
    constexpr expected(T val) : storage_(std::move(val)) {}

    /// Construct from an unexpected error.
    constexpr expected(unexpected<E> err) : storage_(std::move(err)) {}

    constexpr expected(expected const &)            = default;
    constexpr expected(expected &&)                 = default;
    constexpr expected &operator=(expected const &) = default;
    constexpr expected &operator=(expected &&)      = default;

    // ── Observers ──────────────────────────────────────────────────────────

    /// True if holds a value (not an error).
    [[nodiscard]] constexpr bool has_value() const noexcept { return std::holds_alternative<T>(storage_); }

    /// Boolean conversion: true if has value.
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_value(); }

    /// Access the value. UB if holds an error.
    [[nodiscard]] constexpr T &value() & {
        assert(has_value());
        return std::get<T>(storage_);
    }

    [[nodiscard]] constexpr T const &value() const & {
        assert(has_value());
        return std::get<T>(storage_);
    }

    [[nodiscard]] constexpr T &&value() && {
        assert(has_value());
        return std::get<T>(std::move(storage_));
    }

    /// Access the error. UB if holds a value.
    [[nodiscard]] constexpr E &error() & {
        assert(!has_value());
        return std::get<unexpected<E>>(storage_).error();
    }

    [[nodiscard]] constexpr E const &error() const & {
        assert(!has_value());
        return std::get<unexpected<E>>(storage_).error();
    }

    /// Dereference: access value.
    [[nodiscard]] constexpr T       &operator*()       &{ return value(); }
    [[nodiscard]] constexpr T const &operator*() const & { return value(); }
    [[nodiscard]] constexpr T       *operator->() { return &value(); }
    [[nodiscard]] constexpr T const *operator->() const { return &value(); }

    /// Value or default.
    template <std::convertible_to<T> U>
    [[nodiscard]] constexpr T value_or(U &&default_val) const & {
        return has_value() ? value() : static_cast<T>(std::forward<U>(default_val));
    }

    template <std::convertible_to<T> U>
    [[nodiscard]] constexpr T value_or(U &&default_val) && {
        return has_value() ? std::move(value()) : static_cast<T>(std::forward<U>(default_val));
    }

    // ── Monadic operations ─────────────────────────────────────────────────

    /// Transform the value with f. If error, propagates the error.
    template <typename F>
    [[nodiscard]] constexpr auto and_then(F &&f) & {
        using U = std::invoke_result_t<F, T &>;
        if (has_value())
            return std::invoke(std::forward<F>(f), value());
        return U(unexpected(error()));
    }

    template <typename F>
    [[nodiscard]] constexpr auto and_then(F &&f) const & {
        using U = std::invoke_result_t<F, T const &>;
        if (has_value())
            return std::invoke(std::forward<F>(f), value());
        return U(unexpected(error()));
    }

    /// Transform the value, wrapping in a new expected.
    template <typename F>
    [[nodiscard]] constexpr auto transform(F &&f) & {
        using U = std::invoke_result_t<F, T &>;
        if (has_value())
            return expected<U, E>(std::invoke(std::forward<F>(f), value()));
        return expected<U, E>(unexpected(error()));
    }

    template <typename F>
    [[nodiscard]] constexpr auto transform(F &&f) const & {
        using U = std::invoke_result_t<F, T const &>;
        if (has_value())
            return expected<U, E>(std::invoke(std::forward<F>(f), value()));
        return expected<U, E>(unexpected(error()));
    }

  private:
    std::variant<T, unexpected<E>> storage_;
};

/**
 * @brief Specialization for void value type.
 *
 * Represents an operation that either succeeds (no value) or fails with an error.
 *
 * @code
 * einsums::expected<void, std::string> save_file(path) {
 *     if (failed) return einsums::unexpected("write error");
 *     return {};
 * }
 * @endcode
 */
template <typename E>
class expected<void, E> {
  public:
    using value_type = void;
    using error_type = E;

    constexpr expected() noexcept : has_val_(true) {}
    constexpr expected(unexpected<E> err) : err_(std::move(err)), has_val_(false) {}

    constexpr expected(expected const &)            = default;
    constexpr expected(expected &&)                 = default;
    constexpr expected &operator=(expected const &) = default;
    constexpr expected &operator=(expected &&)      = default;

    [[nodiscard]] constexpr bool has_value() const noexcept { return has_val_; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_val_; }

    [[nodiscard]] constexpr E &error() & {
        assert(!has_val_);
        return err_.error();
    }
    [[nodiscard]] constexpr E const &error() const & {
        assert(!has_val_);
        return err_.error();
    }

  private:
    unexpected<E> err_{E{}};
    bool          has_val_{true};
};

#endif // __cpp_lib_expected

} // namespace einsums
