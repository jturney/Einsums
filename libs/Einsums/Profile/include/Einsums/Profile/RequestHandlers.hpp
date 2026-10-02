//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <functional>
#    include <mutex>
#    include <optional>
#    include <shared_mutex>
#    include <string>
#    include <unordered_map>
#    include <utility>
#    include <vector>

EINSUMS_NAMESPACE_BEGIN(profile)

/**
 * @brief What libraries add to the profiler's server and session files: request handlers a viewer
 * calls by name, and sections a session file embeds.
 *
 * The profiler owns the table, not the server, so a library can register at any time, whether or
 * not a server is running yet or ever will. Registration and calls may come from any thread.
 *
 * A handler runs under the table's shared lock, so @ref remove waits for any call of it in
 * progress, and an owner that removes its handlers in its destructor is never called once gone.
 * For the same reason a handler must not register or remove handlers itself.
 */
class RequestHandlers {
  public:
    /// Answers a viewer request: receives the request's params as a JSON object, returns JSON.
    using Handler = std::function<std::string(std::string const &params)>;

    /// Produces one JSON value for a session file.
    using SessionSection = std::function<std::string()>;

    /// Register @p handler for @p method, replacing any earlier one.
    void add(std::string method, Handler handler) {
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

    /// Register @p section under @p key in every session file, replacing any earlier one.
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
    std::unordered_map<std::string, Handler>            _handlers;
    std::vector<std::pair<std::string, SessionSection>> _sections;
};

EINSUMS_NAMESPACE_END(profile)

#endif
