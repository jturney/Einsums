//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <waggle/Config.hpp>

WAGGLE_NAMESPACE_BEGIN

/// Thread-safe string interning with ids counting up from 0. Interning a known string takes a
/// shared lock; only a new string takes the exclusive one.
class StringTable {
  public:
    /// Id 0 is the empty string, which doubles as "no string": a caller can test an id against 0.
    StringTable() { intern(""); }

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

    /// Retrieve a string by id. Thread-safe alongside intern().
    ///
    /// An id this table never issued returns @ref unknown_string. Ids arrive from events, and with
    /// a static libEinsums each extension module has its own profiler, so an id can come from
    /// another table; reading it out of range would crash the consumer thread.
    auto get(uint32_t id) const -> std::string const & {
        std::shared_lock lock(_mutex);
        if (id >= _strings.size()) {
            return unknown_string();
        }
        return _strings[id];
    }

    /// What @ref get returns for an id this table never issued: visible in a report, unlike "".
    static auto unknown_string() -> std::string const & {
        static std::string const value{"<unknown>"};
        return value;
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

WAGGLE_NAMESPACE_END
