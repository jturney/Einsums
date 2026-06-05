//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <cstdint>
#    include <deque>
#    include <shared_mutex>
#    include <string>
#    include <string_view>
#    include <unordered_map>

namespace einsums::profile {

/// Thread-safe string interning table.
/// Strings are assigned monotonically increasing IDs starting from 0.
/// Lookups (intern) take a shared lock on the fast path (string already interned),
/// and an exclusive lock only when inserting a new string.
class StringTable {
  public:
    /// Intern a string and return its ID. Thread-safe.
    auto intern(std::string_view s) -> uint32_t {
        // Fast path: shared lock read
        {
            std::shared_lock lock(_mutex);
            auto             it = _map.find(s);
            if (it != _map.end())
                return it->second;
        }
        // Slow path: exclusive lock for insert
        std::unique_lock lock(_mutex);
        // Double-check after acquiring exclusive lock
        auto it = _map.find(s);
        if (it != _map.end())
            return it->second;
        auto id = static_cast<uint32_t>(_strings.size());
        _strings.emplace_back(s);
        _map.emplace(_strings.back(), id);
        return id;
    }

    /// Retrieve a string by ID. Thread-safe for reads concurrent with intern().
    auto get(uint32_t id) const -> std::string const & {
        std::shared_lock lock(_mutex);
        return _strings[id];
    }

    /// Number of interned strings.
    auto size() const -> size_t {
        std::shared_lock lock(_mutex);
        return _strings.size();
    }

  private:
    mutable std::shared_mutex                      _mutex;
    std::deque<std::string>                        _strings;
    std::unordered_map<std::string_view, uint32_t> _map;
};

} // namespace einsums::profile

#endif
