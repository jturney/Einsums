//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/TypeSupport/Observable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace einsums {

namespace hashes {

/**
 * @struct insensitive_hash
 *
 * @brief Hashes a string without regard for letter case, along with some other conversions.
 *
 * This hash object first converts a string to all capital letters and converts hyphens to underscores.
 * Then it computes a hash based on the new string.
 *
 * @versionadded{1.0.1}
 */
template <typename str_type>
struct InsensitiveHash {
  public:
    constexpr InsensitiveHash() = default;

    /**
     * @brief Perform the hashing operation.
     *
     * @param[in] str The string to hash.
     * @return The hash of the string.
     *
     * @versionadded{1.0.1}
     */
    size_t operator()(str_type const &str) const {
        // PJW hash (Peter J. Weinberger's ELF-style rolling hash), applied to
        // the case-folded, dash-normalized bytes. The shift/mask constants are
        // derived from sizeof(size_t) rather than hard-coded so the algorithm
        // keeps its shape on any word size.
        size_t hash = 0;

        // Calculate the mask for the top byte of size_t.
        constexpr size_t mask = static_cast<size_t>(0xFF) << (8 * (sizeof(size_t) - 1));

        for (auto ch : str) {
            decltype(ch) upper = std::toupper(ch);
            if (upper == '-') { // Convert dashes to underscores.
                upper = '_';
            }
            auto const *bytes = reinterpret_cast<uint8_t const *>(std::addressof(upper));

            for (int i = 0; i < sizeof(std::decay_t<decltype(ch)>); i++) {
                hash <<= sizeof(size_t); // Shift left a number of bits equal to the number of bytes in size_t.
                hash += bytes[i];

                if ((hash & mask) != (size_t)0) {
                    hash ^= mask >> (6 * sizeof(size_t));
                    hash &= ~mask;
                }
            }
        }
        return hash;
    }
};

template <>
struct EINSUMS_EXPORT InsensitiveHash<std::string> {
  public:
    constexpr InsensitiveHash() = default;

    size_t operator()(std::string const &str) const noexcept;
};

template <>
struct EINSUMS_EXPORT InsensitiveHash<char *> {
  public:
    constexpr InsensitiveHash() = default;

    size_t operator()(char const *str) const noexcept;
};

} // namespace hashes

namespace detail {

/**
 * @struct insensitive_equals
 *
 * @brief Compares two strings, ignoring letter case and some other things.
 *
 * This will compare the two strings, ignoring letter case. Also, hyphens and underscores will match together.
 * This means that an option like @c buffer-size will be matched with @c BUFFER_SIZE .
 *
 * @versionadded{1.0.1}
 */
template <typename str_type>
struct InsensitiveEquals {
    constexpr InsensitiveEquals() = default;

    /**
     * @brief Compare two strings using the rules stated before.
     *
     * @param[in] a The first string to compare.
     * @param[in] b The second string to compare.
     *
     * @return True if the strings match based on the rules stated before, false if they don't.
     *
     * @versionadded{1.0.1}
     */
    bool operator()(str_type const &a, str_type const &b) const {
        if (a.size() != b.size()) {
            return false;
        }

        for (size_t i = 0; i < a.size(); i++) {
            char a_ch = std::toupper(a[i]), b_ch = std::toupper(b[i]);

            if (a_ch == '-') {
                a_ch = '_';
            }

            if (b_ch == '-') {
                b_ch = '_';
            }

            if (a_ch != b_ch) {
                return false;
            }
        }
        return true;
    }
};

} // namespace detail

/**
 * @typedef config_mapping_type
 *
 * @brief The type of map that underlies the ConfigMap class.
 *
 * @tparam T The value type of the map.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
using ConfigMappingType = std::unordered_map<std::string, T, hashes::InsensitiveHash<std::string>, detail::InsensitiveEquals<std::string>>;

/**
 * @class ConfigMap
 *
 * @brief Holds a mapping of string keys to configuration values.
 *
 * Objects of this type can hold maps of configuration variables. They can also act as a subject,
 * which can attach observers. When a configuration variable is updated, this map will notify its
 * observers with the new information. it has all of the methods and typedefs available from std::map.
 *
 * @tparam Value The type of data to be associated with each key.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{1.0.1}
 *      Keys are now case-insensitive. Hyphens and underscores are also treated as equivalent.
 * @endversion
 */
template <typename Value>
class ConfigMap
    : public std::enable_shared_from_this<ConfigMap<Value>>,
      public design_pats::Observable<
          std::unordered_map<std::string, Value, hashes::InsensitiveHash<std::string>, detail::InsensitiveEquals<std::string>>> {
  private:
    /**
     * @class PrivateType
     *
     * @brief This class allows for a public constructor that can't be used in public contexts.
     *
     * This class helps users to make shared pointers from this class.
     *
     * @versionadded{1.0.0}
     */
    class PrivateType {
      public:
        explicit PrivateType() = default;
    };

  public:
    /**
     * @typedef MappingType
     *
     * @brief Represents the type used to hold the option map.
     *
     * @versionadded{1.0.0}
     * @versionchangeddesc{1.0.1}
     *      Keys are now case-insensitive. Hyphens and underscores are also treated as equivalent.
     * @endversion
     */
    using MappingType =
        std::unordered_map<std::string, Value, hashes::InsensitiveHash<std::string>, detail::InsensitiveEquals<std::string>>;

    /**
     * Public constructor that can only be accessed in private contexts. Used to make shared pointers
     * from this class.
     *
     * @versionadded{1.0.0}
     */
    ConfigMap(PrivateType)
        : design_pats::Observable<
              std::unordered_map<std::string, Value, hashes::InsensitiveHash<std::string>, detail::InsensitiveEquals<std::string>>>() {}

    /**
     * @brief Create a shared pointer from this class.
     *
     * @return A shared pointer to a ConfigMap.
     *
     * @versionadded{1.0.0}
     */
    static std::shared_ptr<ConfigMap<Value>> create() { return std::make_shared<ConfigMap<Value>>(PrivateType()); }

  private:
    /**
     * @brief Default constructor.
     *
     * @versionadded{1.0.0}
     */
    explicit ConfigMap() = default;

    friend class GlobalConfigMap;
};

/**
 * @typedef SharedConfigMap
 *
 * @brief Shared pointer to a ConfigMap.
 *
 * @versionadded{1.0.0}
 */
template <typename Value>
using SharedConfigMap = std::shared_ptr<ConfigMap<Value>>;
// using SharedInfoMap = std::shared_ptr<InfoMap>;

/**
 * @class GlobalConfigMap
 *
 * @brief This is a map that holds global configuration variables.
 *
 * This map holds three ConfigMap's inside. It has one for each of integer values, floating point values,
 * and string values. Observers can observe this map, and depending on the type of the observer, it will
 * be attached to the appropriate sub-map. This class is a singleton.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{1.0.1}
 *      Added support for Boolean values.
 * @endversion
 */
class EINSUMS_EXPORT GlobalConfigMap {
    EINSUMS_SINGLETON_DEF(GlobalConfigMap)
  public:
    /**
     * @brief Checks to see if the map is empty.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Gets the size of the map.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] size_t size() const noexcept;

    /**
     * @brief Gets the maximum number of buckets in the map.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] size_t max_size() const noexcept;

    /**
     * @brief Get the string value stored at the given key.
     *
     * Returns the empty string if the key is not in the map.
     *
     * @param[in] key The key to query.
     *
     * @versionadded{1.0.0}
     * @versionchangeddesc{2.0.0}
     *      Returns a new string instead of a const reference to an underlying string.
     * @endversion
     */
    [[nodiscard]] std::string get_string(std::string const &key) const noexcept;

    /**
     * @brief Get the string value stored at the given key.
     *
     * Returns an optional default value if the key is not in the map.
     *
     * @param[in] key The key to query.
     * @param[in] default_value The default value. If the key is not in the map, this is what will be returned.
     *
     * @versionadded{1.0.1}
     * @versionchangeddesc{2.0.0}
     *      Returns a new string instead of a const reference to an underlying string.
     * @endversion
     */
    [[nodiscard]] std::string get_string(std::string const &key, std::string const &default_value) const noexcept;

    /**
     * @brief Get the integer value stored at the given key.
     *
     * Returns an optional default value if the key is not in the map.
     *
     * @param[in] key The key to query.
     * @param[in] default_value The default value. If the key is not in the map, this is what will be returned.
     *
     * @versionadded{1.0.0}
     * @versionchangeddesc{1.0.1}
     *      Added the ability to set a default return value.
     * @endversion
     */
    [[nodiscard]] std::int64_t get_int(std::string const &key, std::int64_t default_value = 0) const noexcept;

    /**
     * @brief Get the floating point value stored at the given key.
     *
     * Returns an optional default value if the key is not in the map.
     *
     * @param[in] key The key to query.
     * @param[in] default_value The default value. If the key is not in the map, this is what will be returned.
     *
     * @versionadded{1.0.0}
     * @versionchangeddesc{1.0.1}
     *      Added the ability to set a default return value.
     * @endversion
     */
    [[nodiscard]] double get_double(std::string const &key, double default_value = 0) const noexcept;

    /**
     * @brief Get the boolean flag stored at the given key.
     *
     * Returns an optional default value if the key is not in the map.
     *
     * @param[in] key The key to query.
     * @param[in] default_value The default value. If the key is not in the map, this is what will be returned.
     *
     * @versionadded{1.0.1}
     */
    [[nodiscard]] bool get_bool(std::string const &key, bool default_value = false) const noexcept;

    /**
     * @brief Set the string value stored at the given key.
     *
     * @param[in] key The key to query.
     * @param[in] value The new value.
     *
     * @versionadded{2.0.0}
     */
    void set_string(std::string const &key, std::string const &value);

    /**
     * @brief Set the integer value stored at the given key.
     *
     * @param[in] key The key to query.
     * @param[in] value The new value.
     */
    void set_int(std::string const &key, std::int64_t value);

    /**
     * @brief Set the floating point value stored at the given key.
     *
     * @param[in] key The key to query.
     * @param[in] value The new value.
     *
     * @versionadded{2.0.0}
     */
    void set_double(std::string const &key, double value);

    /**
     * @brief Set the Boolean value stored at the given key.
     *
     * @param[in] key The key to query.
     * @param[in] value The new value.
     *
     * @versionadded{2.0.0}
     */
    void set_bool(std::string const &key, bool value);

    /**
     * @brief Returns the map containing string options.
     *
     * @versionadded{1.0.0}
     */
    std::shared_ptr<ConfigMap<std::string>> get_string_map() noexcept;

    /**
     * @brief Returns the map containing integer options.
     *
     * @versionadded{1.0.0}
     */
    std::shared_ptr<ConfigMap<std::int64_t>> get_int_map() noexcept;

    /**
     * @brief Returns the map containing floating point options.
     *
     * @versionadded{1.0.0}
     */
    std::shared_ptr<ConfigMap<double>> get_double_map() noexcept;

    /**
     * @brief Returns the map containing boolean flags.
     *
     * @versionadded{1.0.0}
     */
    std::shared_ptr<ConfigMap<bool>> get_bool_map() noexcept;

    /**
     * @brief Attach an observer to the global configuration map.
     *
     * The observer should be an object derived from ConfigObserver. The template parameter
     * on the ConfigObserver class
     * determines which map or maps the observer will be attached to. The template parameter can
     * be either @c std::string , @c std::int64_t , @c bool or @c double . If the observer derives from
     * multiple of these observers, it will be attached to each map that it is able to.
     *
     * @param[in] obs The observer to attach.
     *
     * @versionadded{1.0.0}
     */
    template <typename T, bool string_requirement = requires(T obs, ConfigMappingType<std::string> map) { obs(map); },
              bool int_requirement    = requires(T obs, ConfigMappingType<std::int64_t> map) { obs(map); },
              bool double_requirement = requires(T obs, ConfigMappingType<double> map) { obs(map); },
              bool bool_requirement   = requires(T obs, ConfigMappingType<bool> map) { obs(map); }>
    void attach(T &obs) {

        if constexpr (string_requirement) {
            _str_map->attach(obs);
        }

        if constexpr (int_requirement) {
            _int_map->attach(obs);
        }

        if constexpr (double_requirement) {
            _double_map->attach(obs);
        }

        if constexpr (bool_requirement) {
            _bool_map->attach(obs);
        }
    }

    /**
     * @brief Detach an observer from the global configuration map.
     *
     * @param[in] obs The observer to remove.
     *
     * @versionadded{1.0.0}
     */
    template <typename T, bool string_requirement = requires(T obs, ConfigMappingType<std::string> map) { obs(map); },
              bool int_requirement    = requires(T obs, ConfigMappingType<std::int64_t> map) { obs(map); },
              bool double_requirement = requires(T obs, ConfigMappingType<double> map) { obs(map); },
              bool bool_requirement   = requires(T obs, ConfigMappingType<bool> map) { obs(map); }>
    void detach(T &obs) {
        if constexpr (string_requirement) {
            _str_map->detach(obs);
        }

        if constexpr (int_requirement) {
            _int_map->detach(obs);
        }

        if constexpr (double_requirement) {
            _double_map->detach(obs);
        }

        if constexpr (bool_requirement) {
            _bool_map->detach(obs);
        }
    }

    /**
     * @brief Lock all of the maps contained in this object.
     *
     * @versionadded{1.1.0}
     */
    void lock();

    /**
     * @brief Try to lock all of the maps contained in this object.
     *
     * @return True if a lock could be obtained for all of the maps, false if any map could not be locked.
     *
     * @versionadded{1.1.0}
     */
    bool try_lock();

    /**
     * @brief Unlock all of the maps contained in this object.
     *
     * @param[in] notify If true, notify all of the observers that an option was changed.
     *
     * @versionadded{1.1.0}
     */
    void unlock(bool notify = true);

  private:
    explicit GlobalConfigMap();

    /**
     * @property str_map_
     *
     * @brief Holds the string valued options.
     *
     * @versionadded{1.0.0}
     */
    std::shared_ptr<ConfigMap<std::string>> _str_map;

    /**
     * @property int_map_
     *
     * @brief Holds the integer valued options.
     *
     * @versionadded{1.0.0}
     */
    std::shared_ptr<ConfigMap<std::int64_t>> _int_map;

    /**
     * @property double_map_
     *
     * @brief Holds the floating-point valued options.
     *
     * @versionadded{1.0.0}
     */
    std::shared_ptr<ConfigMap<double>> _double_map;

    /**
     * @property bool_map_
     *
     * @brief Holds the Boolean flag options.
     *
     * @versionadded{1.0.1}
     */
    std::shared_ptr<ConfigMap<bool>> _bool_map;
};

struct GlobalConfigMapLockScope {
    GlobalConfigMapLockScope() : _singleton(GlobalConfigMap::get_singleton()) { _singleton.lock(); }

    ~GlobalConfigMapLockScope() { _singleton.unlock(); }

  private:
    GlobalConfigMap &_singleton;
};

} // namespace einsums

/**
 * @brief Compare a ConfigMap with an object of the same contained mapping type.
 *
 * @versionadded{1.0.0}
 */
template <class Value>
bool operator==(std::unordered_map<std::string, Value, einsums::hashes::InsensitiveHash<std::string>,
                                   einsums::detail::InsensitiveEquals<std::string>> const &lhs,
                einsums::ConfigMap<Value> const                                           &rhs) {
    return lhs == rhs.get_value();
}

/**
 * @brief Compare a ConfigMap with an object of the same contained mapping type.
 *
 * @versionadded{1.0.0}
 */
template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, std::unordered_map<std::string, Value, einsums::hashes::InsensitiveHash<std::string>,
                                                                         einsums::detail::InsensitiveEquals<std::string>> const &rhs) {
    return lhs.get_value() == rhs;
}

/**
 * @brief Compare two ConfigMaps.
 *
 * @versionadded{1.0.0}
 */
template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, einsums::ConfigMap<Value> const &rhs) {
    return lhs.get_value() == rhs.get_value();
}
