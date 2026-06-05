//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file FlatSet.hpp
/// @brief C++23 std::flat_set backport for C++20.
///
/// A sorted associative container backed by a contiguous std::vector.
/// Provides O(log n) lookup via binary search with cache-friendly iteration
/// (all elements in one contiguous allocation, unlike std::set's tree nodes).
///
/// Ideal for small-to-medium sets that are built once and queried many times
/// (e.g., tensor ID sets in optimization passes).

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace einsums {

/**
 * @brief A sorted set backed by a contiguous vector.
 *
 * @tparam Key      Element type.
 * @tparam Compare  Comparison function (default: std::less<Key>).
 *
 * @par Example
 * @code
 * einsums::flat_set<int> ids;
 * ids.insert(3);
 * ids.insert(1);
 * ids.insert(2);
 * // Iteration: 1, 2, 3 (sorted)
 * // Lookup: O(log n) via binary search
 *
 * if (ids.contains(2)) { ... }
 * @endcode
 */
template <typename Key, typename Compare = std::less<Key>>
class flat_set {
  public:
    using key_type        = Key;
    using value_type      = Key;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using key_compare     = Compare;
    using iterator        = typename std::vector<Key>::iterator;
    using const_iterator  = typename std::vector<Key>::const_iterator;

    // ── Construction ───────────────────────────────────────────────────────

    flat_set() = default;

    flat_set(std::initializer_list<Key> init) : data_(init) {
        std::sort(data_.begin(), data_.end(), comp_);
        data_.erase(std::unique(data_.begin(), data_.end()), data_.end());
    }

    template <typename InputIt>
    flat_set(InputIt first, InputIt last) : data_(first, last) {
        std::sort(data_.begin(), data_.end(), comp_);
        data_.erase(std::unique(data_.begin(), data_.end()), data_.end());
    }

    // ── Iterators ──────────────────────────────────────────────────────────

    [[nodiscard]] iterator       begin() noexcept { return data_.begin(); }
    [[nodiscard]] const_iterator begin() const noexcept { return data_.begin(); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return data_.cbegin(); }
    [[nodiscard]] iterator       end() noexcept { return data_.end(); }
    [[nodiscard]] const_iterator end() const noexcept { return data_.end(); }
    [[nodiscard]] const_iterator cend() const noexcept { return data_.cend(); }

    // ── Capacity ───────────────────────────────────────────────────────────

    [[nodiscard]] bool      empty() const noexcept { return data_.empty(); }
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    void                    reserve(size_type cap) { data_.reserve(cap); }

    // ── Modifiers ──────────────────────────────────────────────────────────

    /// Insert an element. Returns pair(iterator, inserted).
    std::pair<iterator, bool> insert(Key const &value) {
        auto it = std::lower_bound(data_.begin(), data_.end(), value, comp_);
        if (it != data_.end() && !comp_(value, *it))
            return {it, false}; // Already present
        return {data_.insert(it, value), true};
    }

    std::pair<iterator, bool> insert(Key &&value) {
        auto it = std::lower_bound(data_.begin(), data_.end(), value, comp_);
        if (it != data_.end() && !comp_(value, *it))
            return {it, false};
        return {data_.insert(it, std::move(value)), true};
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args &&...args) {
        return insert(Key(std::forward<Args>(args)...));
    }

    /// Erase by iterator.
    iterator erase(const_iterator pos) { return data_.erase(pos); }

    /// Erase by value. Returns number of elements removed (0 or 1).
    size_type erase(Key const &value) {
        auto it = find(value);
        if (it == end())
            return 0;
        data_.erase(it);
        return 1;
    }

    void clear() noexcept { data_.clear(); }

    // ── Lookup ─────────────────────────────────────────────────────────────

    [[nodiscard]] iterator find(Key const &value) {
        auto it = std::lower_bound(data_.begin(), data_.end(), value, comp_);
        return (it != data_.end() && !comp_(value, *it)) ? it : data_.end();
    }

    [[nodiscard]] const_iterator find(Key const &value) const {
        auto it = std::lower_bound(data_.begin(), data_.end(), value, comp_);
        return (it != data_.end() && !comp_(value, *it)) ? it : data_.end();
    }

    [[nodiscard]] bool contains(Key const &value) const { return find(value) != end(); }

    [[nodiscard]] size_type count(Key const &value) const { return contains(value) ? 1 : 0; }

    // ── Comparison ─────────────────────────────────────────────────────────

    [[nodiscard]] bool operator==(flat_set const &other) const { return data_ == other.data_; }

    // ── Access to underlying data ──────────────────────────────────────────

    [[nodiscard]] Key const *data() const noexcept { return data_.data(); }

  private:
    std::vector<Key> data_;
    Compare          comp_;
};

} // namespace einsums
