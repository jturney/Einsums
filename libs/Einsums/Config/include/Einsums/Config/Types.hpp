//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/TypeSupport/Observable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace einsums {

namespace hashes {

template <typename str_type>
struct insensitive_hash {
  public:
    constexpr insensitive_hash() = default;

    size_t operator()(str_type const &str) const {
        size_t hash = 0;

        // Calculate the mask. If size_t is N bytes, mask for the top N bits.
        // The first part creates a Mersenne value with the appropriate number of bits.
        // The second shifts it to the top.
        constexpr size_t mask = (((size_t)1 << sizeof(size_t)) - 1) << (7 * sizeof(size_t));

        for (auto ch : str) {
            decltype(ch) upper = std::toupper(ch);
            if (upper == '-') { // Convert dashes to underscores.
                upper = '_';
            }
            uint8_t const *bytes = reinterpret_cast<uint8_t const *>(std::addressof(upper));

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
struct insensitive_hash<std::string> {
  public:
    constexpr insensitive_hash() = default;

    size_t operator()(std::string const &str) const noexcept;
};

} // namespace hashes

namespace detail {

template <typename str_type>
struct insensitive_equals {
    constexpr insensitive_equals() = default;

    bool operator()(str_type const &a, str_type const &b) const {
        if (a.size() != b.size()) {
            return false;
        }

        for (size_t i = 0; i < a.size(); i++) {
            if (std::toupper(a[i]) != std::toupper(b[i])) {
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
 */
template <typename T>
using config_mapping_type =
    std::unordered_map<std::string, T, hashes::insensitive_hash<std::string>, detail::insensitive_equals<std::string>>;

/**
 * @class ConfigMap
 *
 * @brief Holds a mapping of string keys to configuration values.
 *
 * Objects of this type can hold maps of configuration variables. They can also act as a subject,
 * which can attach observers. When a configuration variable is updated, this map will notify its
 * observers with the new information. it has all methods and typedefs available from std::map.
 *
 * @tparam Value The type of data to be associated with each key.
 */
template <typename Value>
class ConfigMap
    : public std::enable_shared_from_this<ConfigMap<Value>>,
      public design_pats::Observable<
          std::unordered_map<std::string, Value, hashes::insensitive_hash<std::string>, detail::insensitive_equals<std::string>>> {
  private:
    /**
     * @class PrivateType
     *
     * @brief This class allows for a public constructor that can't be used in public contexts.
     *
     * This class helps users to make shared pointers from this class.
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
     */
    using MappingType =
        std::unordered_map<std::string, Value, hashes::insensitive_hash<std::string>, detail::insensitive_equals<std::string>>;

    /**
     * Public constructor that can only be accessed in private contexts. Used to make shared pointers
     * from this class.
     */
    ConfigMap(PrivateType)
        : design_pats::Observable<
              std::unordered_map<std::string, Value, hashes::insensitive_hash<std::string>, detail::insensitive_equals<std::string>>>() {}

    /**
     * @brief Create a shared pointer from this class.
     *
     * @return A shared pointer to a ConfigMap.
     */
    static std::shared_ptr<ConfigMap<Value>> create() { return std::make_shared<ConfigMap<Value>>(PrivateType()); }

  private:
    /**
     * @brief Default constructor.
     */
    explicit ConfigMap() = default;

    friend class GlobalConfigMap;
};

/**
 * @typedef SharedConfigMap
 *
 * @brief Shared pointer to a ConfigMap.
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
 */
class EINSUMS_EXPORT GlobalConfigMap {
    EINSUMS_SINGLETON_DEF(GlobalConfigMap)
  public:
    /**
     * @brief Checks to see if the map is empty.
     */
    bool empty() const noexcept;

    /**
     * @brief Gets the size of the map.
     */
    size_t size() const noexcept;

    /**
     * @brief Gets the maximum number of buckets in the map.
     */
    size_t max_size() const noexcept;

    /**
     * @brief Get the string value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     * @param dephault The default value. If the key is not in the map, this is what will be returned.
     */
    std::string const &get_string(std::string const &key, std::string const &dephault = "") const;

    /**
     * @brief Get the integer value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     * @param dephault The default value. If the key is not in the map, this is what will be returned.
     */
    std::int64_t get_int(std::string const &key, std::int64_t dephault = 0) const;

    /**
     * @brief Get the floating point value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     * @param dephault The default value. If the key is not in the map, this is what will be returned.
     */
    double get_double(std::string const &key, double dephault = 0) const;

    /**
     * @brief Get the boolean flag stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     * @param dephault The default value. If the key is not in the map, this is what will be returned.
     */
    bool get_bool(std::string const &key, bool dephaul = false) const;

    /**
     * @brief Returns the map containing string options.
     */
    std::shared_ptr<ConfigMap<std::string>> get_string_map();

    /**
     * @brief Returns the map containing integer options.
     */
    std::shared_ptr<ConfigMap<std::int64_t>> get_int_map();

    /**
     * @brief Returns the map containing floating point options.
     */
    std::shared_ptr<ConfigMap<double>> get_double_map();

    /**
     * @brief Returns the map containing boolean flags.
     */
    std::shared_ptr<ConfigMap<bool>> get_bool_map();

    /**
     * @brief Attach an observer to the global configuration map.
     *
     * The observer should be an object derived from ConfigObserver. The template parameter
     * on the ConfigObserver class
     * determines which map or maps the observer will be attached to. The template parameter can
     * be either @c std::string , @c std::int64_t , @c bool or @c double . If the observer derives from
     * multiple of these observers, it will be attached to each map that it is able to.
     *
     * @param obs The observer to attach.
     */
    template <typename T, bool string_requirement = requires(T obs, config_mapping_type<std::string> map) { obs(map); },
              bool int_requirement    = requires(T obs, config_mapping_type<std::int64_t> map) { obs(map); },
              bool double_requirement = requires(T obs, config_mapping_type<double> map) { obs(map); },
              bool bool_requirement   = requires(T obs, config_mapping_type<bool> map) { obs(map); }>
    void attach(T &obs) {

        if constexpr (string_requirement) {
            str_map_->attach(obs);
        }

        if constexpr (int_requirement) {
            int_map_->attach(obs);
        }

        if constexpr (double_requirement) {
            double_map_->attach(obs);
        }

        if constexpr (bool_requirement) {
            bool_map_->attach(obs);
        }
    }

    /**
     * @brief Detach an observer from the global configuration map.
     *
     * @param obs The observer to remove.
     */
    template <typename T, bool string_requirement = requires(T obs, config_mapping_type<std::string> map) { obs(map); },
              bool int_requirement    = requires(T obs, config_mapping_type<std::int64_t> map) { obs(map); },
              bool double_requirement = requires(T obs, config_mapping_type<double> map) { obs(map); },
              bool bool_requirement   = requires(T obs, config_mapping_type<bool> map) { obs(map); }>
    void detach(T &obs) {
        if constexpr (string_requirement) {
            str_map_->detach(obs);
        }

        if constexpr (int_requirement) {
            int_map_->detach(obs);
        }

        if constexpr (double_requirement) {
            double_map_->detach(obs);
        }

        if constexpr (bool_requirement) {
            bool_map_->detach(obs);
        }
    }

    void lock();

    bool try_lock();

    void unlock(bool notify = true);

  private:
    explicit GlobalConfigMap();

    /**
     * @property str_map_
     *
     * @brief Holds the string valued options.
     */
    std::shared_ptr<ConfigMap<std::string>> str_map_;

    /**
     * @property int_map_
     *
     * @brief Holds the integer valued options.
     */
    std::shared_ptr<ConfigMap<std::int64_t>> int_map_;

    /**
     * @property double_map_
     *
     * @brief Holds the floating-point valued options.
     */
    std::shared_ptr<ConfigMap<double>> double_map_;

    /**
     * @property bool_map_
     *
     * @brief Holds the Boolean flag options.
     */
    std::shared_ptr<ConfigMap<bool>> bool_map_;
};

} // namespace einsums

template <class Value>
bool operator==(std::unordered_map<std::string, Value, einsums::hashes::insensitive_hash<std::string>,
                                   einsums::detail::insensitive_equals<std::string>> const &lhs,
                einsums::ConfigMap<Value> const                                            &rhs) {
    return lhs == rhs.get_value();
}

template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, std::unordered_map<std::string, Value, einsums::hashes::insensitive_hash<std::string>,
                                                                         einsums::detail::insensitive_equals<std::string>> const &rhs) {
    return lhs.get_value() == rhs;
}

template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, einsums::ConfigMap<Value> const &rhs) {
    return lhs.get_value() == rhs.get_value();
}
