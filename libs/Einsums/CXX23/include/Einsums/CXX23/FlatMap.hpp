//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file FlatMap.hpp
/// @brief C++23 std::flat_map backport for C++20.
///
/// A sorted associative container backed by two contiguous vectors (keys + values).
/// Provides O(log n) lookup via binary search on keys, with cache-friendly iteration.
///
/// Unlike std::map (red-black tree with scattered nodes), all keys and values
/// live in contiguous memory, making iteration and small-map operations faster.

#include <algorithm>
#include <cassert>
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

namespace einsums {

/**
 * @brief A sorted map backed by contiguous vectors.
 *
 * @tparam Key     Key type.
 * @tparam Value   Mapped value type.
 * @tparam Compare Key comparison function (default: std::less<Key>).
 *
 * @par Example
 * @code
 * einsums::flat_map<std::string, int> counts;
 * counts["einsum"] = 5;
 * counts["scale"]  = 3;
 * // Iteration: sorted by key
 * // Lookup: O(log n)
 * @endcode
 */
template <typename Key, typename Value, typename Compare = std::less<Key>>
class flat_map {
  public:
    using key_type       = Key;
    using mapped_type    = Value;
    using value_type     = std::pair<Key, Value>;
    using size_type      = std::size_t;
    using iterator       = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    // ── Construction ───────────────────────────────────────────────────────

    flat_map() = default;

    flat_map(std::initializer_list<value_type> init) : data_(init) { sort_and_unique(); }

    template <typename InputIt>
    flat_map(InputIt first, InputIt last) : data_(first, last) {
        sort_and_unique();
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

    // ── Element access ─────────────────────────────────────────────────────

    /// Access or insert element by key.
    Value &operator[](Key const &key) {
        auto it = lower_bound(key);
        if (it != data_.end() && !comp_(key, it->first))
            return it->second;
        it = data_.insert(it, {key, Value{}});
        return it->second;
    }

    Value &operator[](Key &&key) {
        auto it = lower_bound(key);
        if (it != data_.end() && !comp_(key, it->first))
            return it->second;
        it = data_.insert(it, {std::move(key), Value{}});
        return it->second;
    }

    /// Access element by key. Asserts key exists.
    [[nodiscard]] Value &at(Key const &key) {
        auto it = find(key);
        assert(it != end());
        return it->second;
    }

    [[nodiscard]] Value const &at(Key const &key) const {
        auto it = find(key);
        assert(it != end());
        return it->second;
    }

    // ── Modifiers ──────────────────────────────────────────────────────────

    std::pair<iterator, bool> insert(value_type const &kv) {
        auto it = lower_bound(kv.first);
        if (it != data_.end() && !comp_(kv.first, it->first))
            return {it, false};
        return {data_.insert(it, kv), true};
    }

    std::pair<iterator, bool> insert(value_type &&kv) {
        auto it = lower_bound(kv.first);
        if (it != data_.end() && !comp_(kv.first, it->first))
            return {it, false};
        return {data_.insert(it, std::move(kv)), true};
    }

    /// Insert or assign: updates value if key exists, inserts otherwise.
    std::pair<iterator, bool> insert_or_assign(Key const &key, Value value) {
        auto it = lower_bound(key);
        if (it != data_.end() && !comp_(key, it->first)) {
            it->second = std::move(value);
            return {it, false}; // Updated
        }
        return {data_.insert(it, {key, std::move(value)}), true}; // Inserted
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args &&...args) {
        return insert(value_type(std::forward<Args>(args)...));
    }

    iterator erase(const_iterator pos) { return data_.erase(pos); }

    size_type erase(Key const &key) {
        auto it = find(key);
        if (it == end())
            return 0;
        data_.erase(it);
        return 1;
    }

    void clear() noexcept { data_.clear(); }

    // ── Lookup ─────────────────────────────────────────────────────────────

    [[nodiscard]] iterator find(Key const &key) {
        auto it = lower_bound(key);
        return (it != data_.end() && !comp_(key, it->first)) ? it : data_.end();
    }

    [[nodiscard]] const_iterator find(Key const &key) const {
        auto it = lower_bound(key);
        return (it != data_.end() && !comp_(key, it->first)) ? it : data_.end();
    }

    [[nodiscard]] bool contains(Key const &key) const { return find(key) != end(); }

    [[nodiscard]] size_type count(Key const &key) const { return contains(key) ? 1 : 0; }

    // ── Comparison ─────────────────────────────────────────────────────────

    [[nodiscard]] bool operator==(flat_map const &other) const { return data_ == other.data_; }

  private:
    std::vector<value_type> data_;
    Compare                 comp_;

    iterator lower_bound(Key const &key) {
        return std::lower_bound(data_.begin(), data_.end(), key,
                                [this](value_type const &elem, Key const &k) { return comp_(elem.first, k); });
    }

    const_iterator lower_bound(Key const &key) const {
        return std::lower_bound(data_.begin(), data_.end(), key,
                                [this](value_type const &elem, Key const &k) { return comp_(elem.first, k); });
    }

    void sort_and_unique() {
        std::sort(data_.begin(), data_.end(), [this](value_type const &a, value_type const &b) { return comp_(a.first, b.first); });
        auto last = std::unique(data_.begin(), data_.end(), [](value_type const &a, value_type const &b) { return a.first == b.first; });
        data_.erase(last, data_.end());
    }
};

} // namespace einsums
