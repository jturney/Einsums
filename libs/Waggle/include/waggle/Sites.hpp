//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <waggle/Config.hpp>

WAGGLE_NAMESPACE_BEGIN

/// One zone in the source: its name, where it is, and which library it belongs to. The strings are
/// string-table ids.
struct Site {
    uint32_t name_id{0};
    uint32_t file_id{0};
    uint32_t func_id{0};
    int      line{0};
    uint32_t domain{0};
};

/**
 * @brief Every call site the process registered, each named by a 32-bit id.
 *
 * Registration is deduplicated by the whole description. A site is a function-local static, and a
 * zone in an inline function or template in a header gets one static per shared object under hidden
 * visibility; deduplication gives every copy the same id. Id 0 is no site.
 */
class SiteTable {
  public:
    SiteTable() { _sites.emplace_back(); }

    /// The id of @p site, registering it if new. Thread-safe.
    auto add(Site const &site) -> uint32_t {
        auto const key = std::make_tuple(site.name_id, site.file_id, site.func_id, site.line, site.domain);
        {
            std::shared_lock const lock(_mutex);
            if (auto it = _ids.find(key); it != _ids.end()) {
                return it->second;
            }
        }
        std::unique_lock const lock(_mutex);
        if (auto it = _ids.find(key); it != _ids.end()) {
            return it->second;
        }
        auto const id = static_cast<uint32_t>(_sites.size());
        _sites.push_back(site);
        _ids.emplace(key, id);
        return id;
    }

    /// The site with id @p id; an id this table never issued gives the empty site. Thread-safe.
    [[nodiscard]] auto get(uint32_t id) const -> Site {
        std::shared_lock const lock(_mutex);
        return id < _sites.size() ? _sites[id] : Site{};
    }

  private:
    using Key = std::tuple<uint32_t, uint32_t, uint32_t, int, uint32_t>;

    mutable std::shared_mutex _mutex;
    std::deque<Site>          _sites;
    std::map<Key, uint32_t>   _ids;
};

/**
 * @brief The libraries that own sites, by name. Id 0 is the unnamed domain, which sites that name
 * none belong to.
 */
class DomainTable {
  public:
    DomainTable() { _names.emplace_back(); }

    /// The id of the domain named @p name, registering it if new; "" is domain 0. Thread-safe.
    auto add(std::string_view name) -> uint32_t {
        if (name.empty()) {
            return 0;
        }
        std::unique_lock const lock(_mutex);
        if (auto it = _ids.find(std::string(name)); it != _ids.end()) {
            return it->second;
        }
        auto const id = static_cast<uint32_t>(_names.size());
        _names.emplace_back(name);
        _ids.emplace(std::string(name), id);
        return id;
    }

    /// The name of domain @p id; "" for domain 0 or an id this table never issued. Thread-safe.
    [[nodiscard]] auto name(uint32_t id) const -> std::string {
        std::scoped_lock const lock(_mutex);
        return id < _names.size() ? _names[id] : std::string{};
    }

  private:
    mutable std::mutex                        _mutex;
    std::deque<std::string>                   _names;
    std::unordered_map<std::string, uint32_t> _ids;
};

WAGGLE_NAMESPACE_END
