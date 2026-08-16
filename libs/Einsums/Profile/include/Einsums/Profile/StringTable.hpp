//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <cstdint>
#    include <deque>
#    include <mutex>
#    include <shared_mutex>
#    include <string>
#    include <string_view>
#    include <unordered_map>

EINSUMS_NAMESPACE_BEGIN(profile)

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
    ///
    /// An id this table never issued yields @ref unknown_string rather than
    /// undefined behaviour. That is not defensive padding: every id the
    /// Consumer resolves arrives from an event it popped off a ring buffer, so
    /// the table cannot assume the id is one of its own, and it has no way to
    /// re-derive the string if it is not. Indexing a deque out of range hands
    /// back a reference to whatever those bytes happen to be, and the caller
    /// then uses it as a std::string: Consumer::process_annotate hashes it
    /// straight into a map key, which dereferences a wild pointer and takes the
    /// process down from the consumer thread.
    ///
    /// That is what a static libEinsums produced. Coverage builds fold a
    /// private copy of the profiler into every extension module, so ids minted
    /// by one copy reached another copy's table, and the Linux coverage leg
    /// died in _Hash_bytes under Consumer::process_annotate. Duplicated
    /// profilers are their own problem, but a telemetry consumer must not be
    /// able to kill the program over an annotation it cannot name, whatever
    /// put the id there.
    auto get(uint32_t id) const -> std::string const & {
        std::shared_lock lock(_mutex);
        if (id >= _strings.size()) {
            return unknown_string();
        }
        return _strings[id];
    }

    /// What @ref get returns for an id this table never issued.
    ///
    /// Named rather than empty so an unresolvable id is visible in a report
    /// instead of silently reading as a blank name, and so a test can assert
    /// on it without hardcoding the spelling.
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

EINSUMS_NAMESPACE_END(profile)

#endif
