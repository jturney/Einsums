//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <Waggle/Types.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

WAGGLE_NAMESPACE_BEGIN

/**
 * @brief What libraries add to the profiler's server and session files: themselves, request
 * handlers a viewer calls by name, and sections a session file embeds.
 *
 * The profiler owns the table, not the server, so a library can register at any time, whether or
 * not a server is running yet or ever will. Registration and calls may come from any thread.
 *
 * A handler runs under the table's shared lock, so @ref remove waits for any call of it in
 * progress, and an owner that removes its handlers in its destructor is never called once gone.
 * For the same reason a handler must not register or remove handlers itself.
 */
/// Another copy of the collector, loaded beside this one, which found this one and switched itself
/// off.
struct DuplicateCollector {
    std::string path;      ///< the copy's library file
    uint32_t    abi_major; ///< the C interface version it was built with
    uint32_t    abi_minor;
};

class RequestHandlers {
  public:
    /// Register @p handler for @p method, replacing any earlier one.
    void add(std::string method, RequestHandler handler) {
        std::unique_lock const lock(_mutex);
        _handlers[std::move(method)] = std::move(handler);
    }

    /// Remove the handler for @p method, once any call of it in progress has returned.
    void remove(std::string const &method) {
        std::unique_lock const lock(_mutex);
        _handlers.erase(method);
    }

    /// Call the handler for @p method, or return nothing if none is registered.
    [[nodiscard]] auto call(std::string const &method, std::string const &params) const -> std::optional<std::string> {
        std::shared_lock const lock(_mutex);
        auto                   it = _handlers.find(method);
        if (it == _handlers.end()) {
            return std::nullopt;
        }
        return it->second(params);
    }

    /// The methods handlers are registered for, sorted: what a viewer can ask this program.
    [[nodiscard]] auto methods() const -> std::vector<std::string> {
        std::shared_lock const   lock(_mutex);
        std::vector<std::string> out;
        out.reserve(_handlers.size());
        for (auto const &entry : _handlers) {
            out.push_back(entry.first);
        }
        std::ranges::sort(out);
        return out;
    }

    /// Register @p section under @p key in every session file, replacing any earlier one. Keys are
    /// namespaced by the library, as in ``einsums.compute_graphs``.
    void add_session_section(std::string key, SessionSection section) {
        std::unique_lock const lock(_mutex);
        for (auto &entry : _sections) {
            if (entry.first == key) {
                entry.second = std::move(section);
                return;
            }
        }
        _sections.emplace_back(std::move(key), std::move(section));
    }

    /// Record that another copy of the collector was loaded and switched itself off, so the
    /// zones of the libraries that use it are missing.
    void add_duplicate(DuplicateCollector duplicate) {
        std::unique_lock const lock(_mutex);
        _duplicates.push_back(std::move(duplicate));
    }

    /// The other copies of the collector this one was told about.
    [[nodiscard]] auto duplicates() const -> std::vector<DuplicateCollector> {
        std::shared_lock const lock(_mutex);
        return _duplicates;
    }

    /// Add @p client to the libraries using the profiler.
    void add_client(ClientInfo client) {
        std::unique_lock const lock(_mutex);
        _clients.push_back(std::move(client));
    }

    /// Remove one client named @p name; returns whether there was one.
    auto remove_client(std::string const &name) -> bool {
        std::unique_lock const lock(_mutex);
        // The most recent first, so a library registered twice leaves in the order it came.
        for (auto it = _clients.rbegin(); it != _clients.rend(); ++it) {
            if (it->name == name) {
                _clients.erase(std::next(it).base());
                return true;
            }
        }
        return false;
    }

    /// The libraries using the profiler, in the order they arrived.
    [[nodiscard]] auto clients() const -> std::vector<ClientInfo> {
        std::shared_lock const lock(_mutex);
        return _clients;
    }

    /// Every session section as a key and its JSON value, in registration order.
    [[nodiscard]] auto session_sections() const -> std::vector<std::pair<std::string, std::string>> {
        std::shared_lock const                           lock(_mutex);
        std::vector<std::pair<std::string, std::string>> out;
        out.reserve(_sections.size());
        for (auto const &[key, section] : _sections) {
            out.emplace_back(key, section());
        }
        return out;
    }

  private:
    mutable std::shared_mutex                           _mutex;
    std::unordered_map<std::string, RequestHandler>     _handlers;
    std::vector<std::pair<std::string, SessionSection>> _sections;
    std::vector<ClientInfo>                             _clients;
    std::vector<DuplicateCollector>                     _duplicates;
};

WAGGLE_NAMESPACE_END
