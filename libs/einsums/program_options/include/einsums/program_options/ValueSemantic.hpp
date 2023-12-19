//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/Config.hpp>
#include <einsums/program_options/Errors.hpp>

#include <any>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

namespace einsums::program_options {

/** Class which specifies how the option's value is to be parsed and converted into C++ types. */
class EINSUMS_EXPORT ValueSemantic {
  public:
    /** Returns the name of the option. The name is only meaningful
    for automatic help message.
 */
    [[nodiscard]] virtual auto name() const -> std::string = 0;

    /** The minimum number of tokens for this option that
        should be present on the command line. */
    [[nodiscard]] virtual auto min_tokens() const -> unsigned = 0;

    /** The maximum number of tokens for this option that
        should be present on the command line. */
    [[nodiscard]] virtual auto max_tokens() const -> unsigned = 0;

    /** Returns true if values from different sources should be composed.
        Otherwise, value from the first source is used and values from
        other sources are discarded.
    */
    [[nodiscard]] virtual auto is_composing() const -> bool = 0;

    /** Returns true if value must be given. Non-optional value

    */
    [[nodiscard]] virtual auto is_required() const -> bool = 0;

    /** Parses a group of tokens that specify a value of option.
        Stores the result in 'value_store', using whatever representation
        is desired. May be be called several times if value of the same
        option is specified more than once.
    */
    virtual void parse(std::any &value_store, std::vector<std::string> const &new_tokens, bool utf8) const = 0;

    /** Called to assign default value to 'value_store'. Returns
        true if default value is assigned, and false if no default
        value exists. */
    virtual auto apply_default(std::any &value_store) const -> bool = 0;

    /** Called when final value of an option is determined.
     */
    virtual void notify(std::any const &value_store) const = 0;

    virtual ~ValueSemantic() {}
};

/** Helper class which perform necessary character conversions in the 'parse' method and forwards the data further. */
template <typename Char>
class ValueSemanticCodecvtHelper {
    // Nothing here. Specialization to follow.
};

/** Helper conversion class for values that accept ascii
    strings as input.
    Overrides the 'parse' method and defines new 'xparse'
    method taking std::string. Depending on whether input
    to parse is ascii or UTF8, will pass it to xparse unmodified,
    or with UTF8->ascii conversion.
*/
template <>
class EINSUMS_EXPORT ValueSemanticCodecvtHelper<char> : public ValueSemantic {
  private: // base overrides
    EINSUMS_EXPORT void parse(std::any &value_store, std::vector<std::string> const &new_tokens, bool utf8) const override;

  protected: // interface for derived classes.
    virtual void xparse(std::any &value_store, std::vector<std::string> const &new_tokens) const = 0;
};

/** Helper conversion class for values that accept ascii
    strings as input.
    Overrides the 'parse' method and defines new 'xparse'
    method taking std::wstring. Depending on whether input
    to parse is ascii or UTF8, will recode input to Unicode, or
    pass it unmodified.
*/
template <>
class EINSUMS_EXPORT ValueSemanticCodecvtHelper<wchar_t> : public ValueSemantic {
  private: // base overrides
    EINSUMS_EXPORT void parse(std::any &value_store, std::vector<std::string> const &new_tokens, bool utf8) const override;

  protected: // interface for derived classes.
    virtual void xparse(std::any &value_store, std::vector<std::wstring> const &new_tokens) const = 0;
};

/** Class which specifies a simple handling of a value: the value will
    have string type and only one token is allowed. */
class EINSUMS_EXPORT UntypedValue : public ValueSemanticCodecvtHelper<char> {
  public:
    UntypedValue(bool zero_tokens = false) : _zero_tokens(zero_tokens) {}

    [[nodiscard]] auto name() const -> std::string override;

    [[nodiscard]] auto min_tokens() const -> unsigned override;
    [[nodiscard]] auto max_tokens() const -> unsigned override;

    [[nodiscard]] auto is_composing() const -> bool override { return false; }

    [[nodiscard]] auto is_required() const -> bool override { return false; }

    /** If 'value_store' is already initialized, or new_tokens
        has more than one elements, throws. Otherwise, assigns
        the first string from 'new_tokens' to 'value_store', without
        any modifications.
     */
    void xparse(std::any &value_store, std::vector<std::string> const &new_tokens) const override;

    /** Does nothing. */
    auto apply_default(std::any &) const -> bool override { return false; }

    /** Does nothing. */
    void notify(std::any const &) const override {}

  private:
    bool _zero_tokens;
};

/** Base class for all option that have a fixed type, and are
    willing to announce this type to the outside world.
    Any 'value_semantics' for which you want to find out the
    type can be dynamic_cast-ed to typed_value_base. If conversion
    succeeds, the 'type' method can be called.
*/
class TypedValueBase {
  public:
    // Returns the type of the value described by this
    // object.
    [[nodiscard]] virtual auto value_type() const -> std::type_info const & = 0;
    // Not really needed, since deletion from this
    // class is silly, but just in case.
    virtual ~TypedValueBase() {}
};

/** Class which handles value of a specific type. */
template <class T, class Char = char>
class TypedValue : public ValueSemanticCodecvtHelper<Char>, public TypedValueBase {
  public:
    /** Ctor. The 'store_to' parameter tells where to store
        the value when it's known. The parameter can be nullptr. */
    TypedValue(T *store_to) : _store_to(store_to) {}

    /** Specifies default value, which will be used
        if none is explicitly specified. The type 'T' should
        provide operator<< for ostream.
    */
    auto default_value(T const &v) -> TypedValue * {
        _default_value         = std::any(v);
        _default_value_as_text = std::to_string(v);
        return this;
    }

    /** Specifies default value, which will be used
        if none is explicitly specified. Unlike the above overload,
        the type 'T' need not provide operator<< for ostream,
        but textual representation of default value must be provided
        by the user.
    */
    auto default_value(T const &v, std::string const &textual) -> TypedValue * {
        _default_value         = std::any(v);
        _default_value_as_text = textual;
        return this;
    }

    /** Specifies an implicit value, which will be used
        if the option is given, but without an adjacent value.
        Using this implies that an explicit value is optional,
    */
    auto implicit_value(T const &v) -> TypedValue * {
        _implicit_value         = std::any(v);
        _implicit_value_as_text = std::to_string(v);
        return this;
    }

    /** Specifies the name used to to the value in help message.  */
    auto value_name(std::string const &name) -> TypedValue * {
        _value_name = name;
        return this;
    }

    /** Specifies an implicit value, which will be used
        if the option is given, but without an adjacent value.
        Using this implies that an explicit value is optional, but if
        given, must be strictly adjacent to the option, i.e.: '-ovalue'
        or '--option=value'.  Giving '-o' or '--option' will cause the
        implicit value to be applied.
        Unlike the above overload, the type 'T' need not provide
        operator<< for ostream, but textual representation of default
        value must be provided by the user.
    */
    auto implicit_value(T const &v, std::string const &textual) -> TypedValue * {
        _implicit_value         = std::any(v);
        _implicit_value_as_text = textual;
        return this;
    }

    /** Specifies a function to be called when the final value
        is determined. */
    auto notifier(std::function<void(T const &)> f) -> TypedValue * {
        _notifier = f;
        return this;
    }

    /** Specifies that the value is composing. See the 'is_composing'
        method for explanation.
    */
    auto composing() -> TypedValue * {
        _composing = true;
        return this;
    }

    /** Specifies that the value can span multiple tokens.
     */
    auto multitoken() -> TypedValue * {
        _multitoken = true;
        return this;
    }

    /** Specifies that no tokens may be provided as the value of
        this option, which means that only presence of the option
        is significant. For such option to be useful, either the
        'validate' function should be specialized, or the
        'implicit_value' method should be also used. In most
        cases, you can use the 'bool_switch' function instead of
        using this method. */
    auto zero_tokens() -> TypedValue * {
        _zero_tokens = true;
        return this;
    }

    /** Specifies that the value must occur. */
    auto required() -> TypedValue * {
        _required = true;
        return this;
    }

  public: // value semantic overrides
    [[nodiscard]] auto name() const -> std::string override;

    [[nodiscard]] auto is_composing() const -> bool override { return _composing; }

    [[nodiscard]] auto min_tokens() const -> unsigned override {
        if (_zero_tokens || _implicit_value.has_value()) {
            return 0;
        } else {
            return 1;
        }
    }

    [[nodiscard]] auto max_tokens() const -> unsigned override {
        if (_multitoken) {
            return (std::numeric_limits<unsigned>::max)();
        } else if (_zero_tokens) {
            return 0;
        } else {
            return 1;
        }
    }

    [[nodiscard]] auto is_required() const -> bool override { return _required; }

    /** Creates an instance of the 'validator' class and calls
        its operator() to perform the actual conversion. */
    void xparse(std::any &value_store, std::vector<std::basic_string<Char>> const &new_tokens) const override;

    /** If default value was specified via previous call to
        'default_value', stores that value into 'value_store'.
        Returns true if default value was stored.
    */
    auto apply_default(std::any &value_store) const -> bool override {
        if (!_default_value.has_value()) {
            return false;
        } else {
            value_store = _default_value;
            return true;
        }
    }

    /** If an address of variable to store value was specified
        when creating *this, stores the value there. Otherwise,
        does nothing. */
    void notify(std::any const &value_store) const override;

  public: // typed_value_base overrides
    [[nodiscard]] auto value_type() const -> std::type_info const & override { return typeid(T); }

  private:
    T *_store_to;

    // Default value is stored as std::any and not
    // as boost::optional to avoid unnecessary instantiations.
    std::string                    _value_name;
    std::any                       _default_value;
    std::string                    _default_value_as_text;
    std::any                       _implicit_value;
    std::string                    _implicit_value_as_text;
    bool                           _composing{}, _implicit{}, _multitoken{}, _zero_tokens{}, _required{};
    std::function<void(T const &)> _notifier;
};

/** Creates a typed_value<T> instance. This function is the primary
    method to create value_semantic instance for a specific type, which
    can later be passed to 'option_description' constructor.
    The second overload is used when it's additionally desired to store the
    value of option into program variable.
*/
template <class T>
auto value() -> TypedValue<T> *;

/** @overload
 */
template <class T>
auto value(T *v) -> TypedValue<T> *;

/** Creates a typed_value<T> instance. This function is the primary
    method to create value_semantic instance for a specific type, which
    can later be passed to 'option_description' constructor.
*/
template <class T>
auto wvalue() -> TypedValue<T, wchar_t> *;

/** @overload
 */
template <class T>
auto wvalue(T *v) -> TypedValue<T, wchar_t> *;

/** Works the same way as the 'value<bool>' function, but the created
    value_semantic won't accept any explicit value. So, if the option
    is present on the command line, the value will be 'true'.
*/
EINSUMS_EXPORT auto bool_switch() -> TypedValue<bool> *;

/** @overload
 */
EINSUMS_EXPORT auto bool_switch(bool *v) -> TypedValue<bool> *;

} // namespace einsums::program_options

#include <einsums/program_options/detail/ValueSemantic.hpp>