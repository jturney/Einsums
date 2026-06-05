//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>

#include <memory>

namespace einsums {
using std::toupper;

EINSUMS_SINGLETON_IMPL(GlobalConfigMap)

GlobalConfigMap::GlobalConfigMap()
    : _str_map{ConfigMap<std::string>::create()}, _int_map{ConfigMap<std::int64_t>::create()}, _double_map{ConfigMap<double>::create()},
      _bool_map{ConfigMap<bool>::create()} {
}

bool GlobalConfigMap::empty() const noexcept {
    return _str_map->get_value().empty() && _int_map->get_value().empty() && _double_map->get_value().empty();
}

size_t GlobalConfigMap::size() const noexcept {
    return _str_map->get_value().size() + _int_map->get_value().size() + _double_map->get_value().size();
}

size_t GlobalConfigMap::max_size() const noexcept {
    return _str_map->get_value().max_size() + _int_map->get_value().max_size() + _double_map->get_value().max_size();
}

std::string GlobalConfigMap::get_string(std::string const &key) const noexcept {
    if (_str_map->get_value().contains(key)) {
        return _str_map->get_value().at(key);
    } else {
        return "";
    }
}

std::string GlobalConfigMap::get_string(std::string const &key, std::string const &default_value) const noexcept {
    if (_str_map->get_value().contains(key)) {
        return _str_map->get_value().at(key);
    } else {
        return default_value;
    }
}

std::int64_t GlobalConfigMap::get_int(std::string const &key, std::int64_t default_value) const noexcept {
    if (_int_map->get_value().contains(key)) {
        return _int_map->get_value().at(key);
    } else {
        return default_value;
    }
}
double GlobalConfigMap::get_double(std::string const &key, double default_value) const noexcept {
    if (_double_map->get_value().contains(key)) {
        return _double_map->get_value().at(key);
    } else {
        return default_value;
    }
}
bool GlobalConfigMap::get_bool(std::string const &key, bool default_value) const noexcept {
    if (_bool_map->get_value().contains(key)) {
        return _bool_map->get_value().at(key);
    } else {
        return default_value;
    }
}

void GlobalConfigMap::set_string(std::string const &key, std::string const &value) {
    (*_str_map).get_value()[key] = value;
}

void GlobalConfigMap::set_int(std::string const &key, std::int64_t value) {
    (*_int_map).get_value()[key] = value;
}

void GlobalConfigMap::set_double(std::string const &key, double value) {
    (*_double_map).get_value()[key] = value;
}

void GlobalConfigMap::set_bool(std::string const &key, bool value) {
    (*_bool_map).get_value()[key] = value;
}

std::shared_ptr<ConfigMap<std::string>> GlobalConfigMap::get_string_map() noexcept {
    return _str_map;
}
std::shared_ptr<ConfigMap<std::int64_t>> GlobalConfigMap::get_int_map() noexcept {
    return _int_map;
}
std::shared_ptr<ConfigMap<double>> GlobalConfigMap::get_double_map() noexcept {
    return _double_map;
}
std::shared_ptr<ConfigMap<bool>> GlobalConfigMap::get_bool_map() noexcept {
    return _bool_map;
}

size_t einsums::hashes::InsensitiveHash<std::string>::operator()(std::string const &str) const noexcept {
    size_t hash = 0;

    // Calculate the mask for the top byte of size_t.
    constexpr size_t mask = static_cast<size_t>(0xFF) << (8 * (sizeof(size_t) - 1));

    for (char const ch : str) {
        char upper = toupper(ch); // NOLINT
        if (upper == '-') {       // Convert dashes to underscores.
            upper = '_';
        }

        hash <<= sizeof(size_t); // Shift left a number of bits equal to the number of bytes in size_t.
        hash += (uint8_t)upper;

        if ((hash & mask) != (size_t)0) {
            hash ^= mask >> (6 * sizeof(size_t));
            hash &= ~mask;
        }
    }
    return hash;
}

size_t einsums::hashes::InsensitiveHash<char *>::operator()(char const *str) const noexcept {
    size_t hash = 0;

    // Calculate the mask for the top byte of size_t.
    constexpr size_t mask       = static_cast<size_t>(0xFF) << (8 * (sizeof(size_t) - 1));
    size_t           curr_index = 0;

    while (str[curr_index] != 0) {
        char upper = std::toupper(str[curr_index]); // NOLINT
        if (upper == '-') {                         // Convert dashes to underscores.
            upper = '_';
        }

        hash <<= sizeof(size_t); // Shift left a number of bits equal to the number of bytes in size_t.
        hash += (uint8_t)upper;

        if ((hash & mask) != (size_t)0) {
            hash ^= mask >> (6 * sizeof(size_t));
            hash &= ~mask;
        }
        curr_index++;
    }
    return hash;
}

void GlobalConfigMap::lock() {
    _str_map->lock();
    _int_map->lock();
    _double_map->lock();
    _bool_map->lock();
}

bool GlobalConfigMap::try_lock() {
    // Roll-back paths use unlock(false): no value was modified, so there's
    // nothing observer-worthy to fire. Avoiding notify here also keeps
    // try_lock free of the lock-order risk that bit GlobalConfigMap::unlock
    // (see the comment there for the full pattern).
    bool res = _str_map->try_lock();

    if (!res) {
        return false;
    }

    res = _int_map->try_lock();

    if (!res) {
        _str_map->unlock(false);
        return false;
    }

    res = _double_map->try_lock();

    if (!res) {
        _str_map->unlock(false);
        _int_map->unlock(false);
        return false;
    }

    res = _bool_map->try_lock();

    if (!res) {
        _str_map->unlock(false);
        _int_map->unlock(false);
        _double_map->unlock(false);
        return false;
    }

    return true;
}

void GlobalConfigMap::unlock(bool notify) {
    // Release all four sub-mutexes FIRST, then fire observers, so any
    // observer callback's implicit conversion-to-T (which re-acquires the
    // sub-Observable's mutex via get_value()) doesn't run while *sibling*
    // sub-Observables are still locked. The previous implementation
    // sequentially called _str_map->unlock(true), which fired the str_map's
    // observers immediately while _int/_double/_bool were still held — TSan
    // caught the resulting M0 <-> M1 cycle (CI run 26690424069, third
    // lock-order finding in this complex). The Observable API was extended
    // with a public notify() to allow this split. See task #20 for the
    // long-term plan to replace this whole aggregate with a single
    // mutex-protected struct.
    _str_map->unlock(false);
    _int_map->unlock(false);
    _double_map->unlock(false);
    _bool_map->unlock(false);
    if (notify) {
        _str_map->notify();
        _int_map->notify();
        _double_map->notify();
        _bool_map->notify();
    }
}

} // namespace einsums