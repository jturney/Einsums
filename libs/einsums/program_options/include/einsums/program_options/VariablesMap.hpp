//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/Config.hpp>

#include <any>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>

// clang-format off
#include <einsums/config/WarningsPrefix.hpp>
#include <utility>
// clang-format on

namespace einsums::program_options {

template <typename Char>
class BasicParsedOptions;

class ValueSemantic;
class VariablesMap;

// forward declaractions

/** Stores in 'm' all options that are defined in 'options'.
    If 'm' already has a non-defaulted value of an option, that value
    is not changed, even if 'options' specify some value.
*/
EINSUMS_EXPORT
void store(BasicParsedOptions<char> const &options, VariablesMap &m, bool utf8 = false);

/** Stores in 'm' all options that are defined in 'options'.
    If 'm' already has a non-defaulted value of an option, that value
    is not changed, even if 'options' specify some value.
    This is wide character variant.
*/
EINSUMS_EXPORT
void store(BasicParsedOptions<wchar_t> const &options, VariablesMap &m);

/** Runs all 'notify' function for options in 'm'. */
EINSUMS_EXPORT void notify(VariablesMap &m);

/** Class holding value of option. Contains details about how the
    value is set and allows to conveniently obtain the value.
*/
class EINSUMS_EXPORT VariableValue {
  public:
    VariableValue() : _defaulted(false) {}
    VariableValue(std::any xv, bool xdefaulted) : _v(std::move(xv)), _defaulted(xdefaulted) {}

    /** If stored value if of type T, returns that value. Otherwise,
        throws boost::bad_any_cast exception. */
    template <class T>
    auto as() const -> T const & {
        return std::any_cast<T const &>(_v);
    }
    /** @overload */
    template <class T>
    auto as() -> T & {
        return std::any_cast<T &>(_v);
    }

    /// Returns true if no value is stored.
    [[nodiscard]] auto empty() const -> bool;
    /** Returns true if the value was not explicitly
        given, but has default value. */
    [[nodiscard]] auto defaulted() const -> bool;
    /** Returns the contained value. */
    [[nodiscard]] auto value() const -> std::any const &;

    /** Returns the contained value. */
    auto value() -> std::any &;

  private:
    std::any _v;
    bool     _defaulted;
    // Internal reference to value semantic. We need to run
    // notifications when *final* values of options are known, and
    // they are known only after all sources are stored. By that
    // time options_description for the first source might not
    // be easily accessible, so we need to store semantic here.
    std::shared_ptr<ValueSemantic const> _value_semantic;

    friend EINSUMS_EXPORT void store(BasicParsedOptions<char> const &options, VariablesMap &m, bool);

    friend class EINSUMS_EXPORT variables_map;
};

/** Implements string->string mapping with convenient value casting
    facilities. */
class EINSUMS_EXPORT AbstractVariablesMap {
  public:
    AbstractVariablesMap();
    AbstractVariablesMap(AbstractVariablesMap const *next);

    virtual ~AbstractVariablesMap() {}

    /** Obtains the value of variable 'name', from *this and
        possibly from the chain of variable maps.

        - if there's no value in *this.
            - if there's next variable map, returns value from it
            - otherwise, returns empty value

        - if there's defaulted value
            - if there's next variable map, which has a non-defaulted
              value, return that
            - otherwise, return value from *this

        - if there's a non-defaulted value, returns it.
    */
    auto operator[](std::string const &name) const -> VariableValue const &;

    /** Sets next variable map, which will be used to find
       variables not found in *this. */
    void next(AbstractVariablesMap *next);

  private:
    /** Returns value of variable 'name' stored in *this, or
        empty value otherwise. */
    [[nodiscard]] virtual auto get(std::string const &name) const -> VariableValue const & = 0;

    AbstractVariablesMap const *_next;
};

/** Concrete variables map which store variables in real map.

    This class is derived from std::map<std::string, VariableValue>,
    so you can use all map operators to examine its content.
*/
class EINSUMS_EXPORT VariablesMap : public AbstractVariablesMap, public std::map<std::string, VariableValue> {
  public:
    VariablesMap();
    VariablesMap(AbstractVariablesMap const *next);

    // Resolve conflict between inherited operators.
    auto operator[](std::string const &name) const -> VariableValue const & { return AbstractVariablesMap::operator[](name); }

    // Override to clear some extra fields.
    void clear();

    void notify();

  private:
    /** Implementation of abstract_variables_map::get
        which does 'find' in *this. */
    [[nodiscard]] auto get(std::string const &name) const -> VariableValue const & override;

    /** Names of option with 'final' values \-- which should not
        be changed by subsequence assignments. */
    std::set<std::string> _final;

    friend EINSUMS_EXPORT void store(BasicParsedOptions<char> const &options, VariablesMap &xm, bool utf8);

    /** Names of required options, filled by parser which has
        access to options_description.
        The map values are the "canonical" names for each corresponding option.
        This is useful in creating diagnostic messages when the option is absent. */
    std::map<std::string, std::string> _required;
};

/*
 * Templates/inlines
 */

inline auto VariableValue::empty() const -> bool {
    return !_v.has_value();
}

inline auto VariableValue::defaulted() const -> bool {
    return _defaulted;
}

inline auto VariableValue::value() const -> std::any const & {
    return _v;
}

inline auto VariableValue::value() -> std::any & {
    return _v;
}

} // namespace einsums::program_options

#include <einsums/config/WarningsSuffix.hpp>
