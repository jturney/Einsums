//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

namespace einsums {

template <class K, class V, class Hash = std::hash<K>, class Eq = std::equal_to<K>>
class InsertionOrderedMap {
  public:
    // NOLINTNEXTLINE(readability-identifier-naming)
    using map_type = std::unordered_map<K, V, Hash, Eq>;
    // NOLINTNEXTLINE(readability-identifier-naming)
    using value_type = typename map_type::value_type; // pair<const K, V>
    // NOLINTNEXTLINE(readability-identifier-naming)
    using size_type = typename map_type::size_type;
    // NOLINTNEXTLINE(readability-identifier-naming)
    using map_iterator = typename map_type::iterator;
    // NOLINTNEXTLINE(readability-identifier-naming)
    using map_citerator = typename map_type::const_iterator;

  private:
    // We iterate over order_ (vector of iterators into map_) to preserve insertion order.
    std::vector<map_iterator>               _order;
    map_type                                _map;
    std::unordered_map<K, size_t, Hash, Eq> _pos; // key -> index in order_ for O(1) find

  public:
    InsertionOrderedMap() = default;

    // ----- Iterator types that yield references into map_ -----
    class Iterator {
      public:
        // NOLINTNEXTLINE(readability-identifier-naming)
        using vec_iter = typename std::vector<map_iterator>::const_iterator;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using difference_type = std::ptrdiff_t;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using iterator_category = std::bidirectional_iterator_tag;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using reference = value_type &;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using pointer = value_type *;

        Iterator() = default;
        Iterator(vec_iter it) : _it(it) {}

        reference operator*() const { return **_it; }
        pointer   operator->() const { return std::addressof(**_it); }

        Iterator &operator++() {
            ++_it;
            return *this;
        }
        Iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }
        Iterator &operator--() {
            --_it;
            return *this;
        }
        Iterator operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        friend bool operator==(Iterator const &a, Iterator const &b) { return a._it == b._it; }
        friend bool operator!=(Iterator const &a, Iterator const &b) { return !(a == b); }
        // expose order iterator for internal construction
        vec_iter raw() const { return _it; }

      private:
        vec_iter _it{};
    };

    class ConstIterator {
      public:
        // NOLINTNEXTLINE(readability-identifier-naming)
        using vec_iter = typename std::vector<map_iterator>::const_iterator;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using difference_type = std::ptrdiff_t;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using iterator_category = std::bidirectional_iterator_tag;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using reference = value_type const &;
        // NOLINTNEXTLINE(readability-identifier-naming)
        using pointer = value_type const *;

        ConstIterator() = default;
        ConstIterator(vec_iter it) : _it(it) {}
        ConstIterator(Iterator const &it) : _it(it.raw()) {}

        reference operator*() const { return **_it; }
        pointer   operator->() const { return std::addressof(**_it); }

        ConstIterator &operator++() {
            ++_it;
            return *this;
        }
        ConstIterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }
        ConstIterator &operator--() {
            --_it;
            return *this;
        }
        ConstIterator operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        friend bool operator==(ConstIterator const &a, ConstIterator const &b) { return a._it == b._it; }
        friend bool operator!=(ConstIterator const &a, ConstIterator const &b) { return !(a == b); }

      private:
        vec_iter _it{};
    };

    // ----- Capacity -----
    [[nodiscard]] bool empty() const noexcept { return _map.empty(); }
    size_type          size() const noexcept { return _map.size(); }

    // ----- Iteration (in insertion order) -----
    Iterator      begin() { return Iterator{_order.begin()}; }
    Iterator      end() { return Iterator{_order.end()}; }
    ConstIterator begin() const { return cbegin(); }
    ConstIterator end() const { return cend(); }
    ConstIterator cbegin() const { return ConstIterator{_order.begin()}; }
    ConstIterator cend() const { return ConstIterator{_order.end()}; }

    // ----- Lookup (O(1) via pos_ index) -----
    Iterator find(K const &key) {
        auto pit = _pos.find(key);
        if (pit == _pos.end())
            return end();
        return Iterator{_order.begin() + static_cast<std::ptrdiff_t>(pit->second)};
    }

    ConstIterator find(K const &key) const {
        auto pit = _pos.find(key);
        if (pit == _pos.end())
            return cend();
        return ConstIterator{_order.begin() + static_cast<std::ptrdiff_t>(pit->second)};
    }

    // ----- Insertion -----
    // insert/emplace preserve first-in insertion order
    std::pair<Iterator, bool> insert(value_type const &kv) {
        auto [mit, inserted] = _map.insert(kv);
        if (inserted) {
            _pos[mit->first] = _order.size();
            _order.push_back(mit);
        }
        size_t idx = _pos[mit->first];
        return {Iterator{_order.begin() + static_cast<std::ptrdiff_t>(idx)}, inserted};
    }

    template <class... Args>
    std::pair<Iterator, bool> emplace(Args &&...args) {
        auto [mit, inserted] = _map.emplace(std::forward<Args>(args)...);
        if (inserted) {
            _pos[mit->first] = _order.size();
            _order.push_back(mit);
        }
        size_t idx = _pos[mit->first];
        return {Iterator{_order.begin() + static_cast<std::ptrdiff_t>(idx)}, inserted};
    }

    // like operator[]: inserts default if missing and returns reference
    V &operator[](K const &key) {
        auto [mit, inserted] = _map.try_emplace(key);
        if (inserted) {
            _pos[key] = _order.size();
            _order.push_back(mit);
        }
        return mit->second;
    }

    V       &at(K const &key) { return _map.at(key); }
    V const &at(K const &key) const { return _map.at(key); }

    /// Returns 1 if the key exists, 0 otherwise (compatible with std::map::count).
    size_type count(K const &key) const { return _map.count(key); }

    /// Check whether a key exists.
    bool contains(K const &key) const { return _map.count(key) != 0; }

    // (optional) erase by key; keeps order_ and pos_ in sync
    size_type erase(K const &key) {
        auto pit = _pos.find(key);
        if (pit == _pos.end())
            return 0;
        size_t idx = pit->second;
        // Remove from order_ and update pos_ for shifted elements
        _order.erase(_order.begin() + static_cast<std::ptrdiff_t>(idx));
        _pos.erase(pit);
        // Update indices for all elements after the erased one
        for (size_t i = idx; i < _order.size(); ++i) {
            _pos[_order[i]->first] = i;
        }
        _map.erase(key);
        return 1;
    }
};

} // namespace einsums
