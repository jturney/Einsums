//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/Config.hpp>
#include <einsums/program_options/Errors.hpp>
#include <einsums/program_options/ValueSemantic.hpp>

#include <cstddef>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <einsums/config/WarningsPrefix.hpp>
// clang-format on

namespace einsums::program_options {

/** Describes one possible command line/config file option. There are two
    kinds of properties of an option. First describe it syntactically and
    are used only to validate input. Second affect interpretation of the
    option, for example default value for it or function that should be
    called  when the value is finally known. Routines which perform parsing
    never use second kind of properties \-- they are side effect free.
    @sa OptionsDescription
*/
class EINSUMS_EXPORT OptionDescription {
  public:
    OptionDescription();

    /** Initializes the object with the passed data.

        Note: it would be nice to make the second parameter auto_ptr,
        to explicitly pass ownership. Unfortunately, it's often needed to
        create objects of types derived from 'ValueSemantic':
           OptionsDescription d;
           d.add_options()("a", parameter<int>("n")->default_value(1));
        Here, the static type returned by 'parameter' should be derived
        from ValueSemantic.

        Alas, derived->base conversion for auto_ptr does not really work,
        see
        http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2000/n1232.pdf
        http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#84

        So, we have to use plain old pointers. Besides, users are not
        expected to use the constructor directly.


        The 'name' parameter is interpreted by the following rules:
        - if there's no "," character in 'name', it specifies long name
        - otherwise, the part before "," specifies long name and the part
        after \-- short name.
    */
    OptionDescription(char const *name, ValueSemantic const *s);

    /** Initializes the class with the passed data.
     */
    OptionDescription(char const *name, ValueSemantic const *s, char const *description);

    virtual ~OptionDescription();

    enum MatchResult { no_match, full_match, approximate_match };

    /** Given 'option', specified in the input source,
        returns 'true' if 'option' specifies *this.
    */
    [[nodiscard]] auto match(std::string const &option, bool approx, bool long_ignore_case, bool short_ignore_case) const -> MatchResult;

    /** Returns the key that should identify the option, in
        particular in the variables_map class.
        The 'option' parameter is the option spelling from the
        input source.
        If option name contains '*', returns 'option'.
        If long name was specified, it's the long name, otherwise
        it's a short name with pre-pended '-'.
    */
    [[nodiscard]] auto key(std::string const &option) const -> std::string const &;

    /** Returns the canonical name for the option description to enable the user to
        recognized a matching option.
        1) For short options ('-', '/'), returns the short name prefixed.
        2) For long options ('--' / '-') returns the first long name prefixed
        3) All other cases, returns the first long name (if present) or the short
           name, un-prefixed.
    */
    [[nodiscard]] auto canonical_display_name(int canonical_option_style = 0) const -> std::string;

    [[nodiscard]] auto long_name() const -> std::string const &;

    [[nodiscard]] auto long_names() const -> std::pair<std::string const *, std::size_t> const;

    /// Explanation of this option
    [[nodiscard]] auto description() const -> std::string const &;

    /// Semantic of option's value
    [[nodiscard]] auto semantic() const -> std::shared_ptr<ValueSemantic const>;

    /// Returns the option name, formatted suitably for usage message.
    [[nodiscard]] auto format_name() const -> std::string;

    /** Returns the parameter name and properties, formatted suitably for
        usage message. */
    [[nodiscard]] auto format_parameter() const -> std::string;

  private:
    auto set_names(char const *name) -> OptionDescription &;

    /**
     * a one-character "switch" name - with its prefix,
     * so that this is either empty or has length 2 (e.g. "-c"
     */
    std::string _short_name;

    /**
     *  one or more names by which this option may be specified
     *  on a command-line or in a config file, which are not
     *  a single-letter switch. The names here are _without_
     * any prefix.
     */
    std::vector<std::string> _long_names;

    std::string _description;

    // shared_ptr is needed to simplify memory management in
    // copy ctor and destructor.
    std::shared_ptr<ValueSemantic const> _value_semantic;
};

class OptionsDescription;

/** Class which provides convenient creation syntax to option_description.
 */
class EINSUMS_EXPORT OptionsDescriptionEasyInit {
  public:
    OptionsDescriptionEasyInit(OptionsDescription *owner);

    auto operator()(char const *name, char const *description) -> OptionsDescriptionEasyInit &;

    auto operator()(char const *name, ValueSemantic const *s) -> OptionsDescriptionEasyInit &;

    auto operator()(char const *name, ValueSemantic const *s, char const *description) -> OptionsDescriptionEasyInit &;

  private:
    OptionsDescription *_owner;
};

/** A set of option descriptions. This provides convenient interface for
    adding new option (the add_options) method, and facilities to search
    for options by name.

    See @ref a_adding_options "here" for option adding interface discussion.
    @sa option_description
*/
class EINSUMS_EXPORT OptionsDescription {
  public:
    static unsigned const DefaultLineLength;

    /** Creates the instance. */
    OptionsDescription(unsigned line_length = DefaultLineLength, unsigned min_description_length = DefaultLineLength / 2);
    /** Creates the instance. The 'caption' parameter gives the name of
        this 'OptionsDescription' instance. Primarily useful for output.
        The 'description_length' specifies the number of columns that
        should be reserved for the description text; if the option text
        encroaches into this, then the description will start on the next
        line.
    */
    OptionsDescription(std::string const &caption, unsigned line_length = DefaultLineLength,
                       unsigned min_description_length = DefaultLineLength / 2);
    /** Adds new variable description. Throws duplicate_variable_error if
        either short or long name matches that of already present one.
    */
    void add(std::shared_ptr<OptionDescription> desc);
    /** Adds a group of option description. This has the same
        effect as adding all option_descriptions in 'desc'
        individually, except that output operator will show
        a separate group.
        Returns *this.
    */
    auto add(OptionsDescription const &desc) -> OptionsDescription &;

    /** Find the maximum width of the option column, including options
        in groups. */
    [[nodiscard]] auto get_option_column_width() const -> std::size_t;

  public:
    /** Returns an object of implementation-defined type suitable for adding
        options to OptionsDescription. The returned object will
        have overloaded operator() with parameter type matching
        'option_description' constructors. Calling the operator will create
        new option_description instance and add it.
    */
    auto add_options() -> OptionsDescriptionEasyInit;

    [[nodiscard]] auto find(std::string const &name, bool approx, bool long_ignore_case = false, bool short_ignore_case = false) const
        -> OptionDescription const &;

    [[nodiscard]] auto find_nothrow(std::string const &name, bool approx, bool long_ignore_case = false,
                                    bool short_ignore_case = false) const -> OptionDescription const *;

    [[nodiscard]] auto options() const -> std::vector<std::shared_ptr<OptionDescription>> const &;

    /** Produces a human readable output of 'desc', listing options,
        their descriptions and allowed parameters. Other OptionsDescription
        instances previously passed to add will be output separately. */
    friend EINSUMS_EXPORT auto operator<<(std::ostream &os, OptionsDescription const &desc) -> std::ostream &;

    /** Outputs 'desc' to the specified stream, calling 'f' to output each
        option_description element. */
    void print(std::ostream &os, std::size_t width = 0) const;

  private:
#if defined(EINSUMS_MSVC) && EINSUMS_MSVC >= 1800
    // prevent warning C4512: assignment operator could not be generated
    OptionsDescription &operator=(OptionsDescription const &);
#endif

    using Name2IndexIterator = std::map<std::string, int>::const_iterator;
    using ApproximationRange = std::pair<Name2IndexIterator, Name2IndexIterator>;

    // approximation_range find_approximation(const std::string& prefix) const;

    std::string       _caption;
    std::size_t const _line_length;
    std::size_t const _min_description_length;

    // Data organization is chosen because:
    // - there could be two names for one option
    // - option_add_proxy needs to know the last added option
    std::vector<std::shared_ptr<OptionDescription>> _options;

    // Whether the option comes from one of declared groups.
    // vector<bool> is buggy there, see
    // http://support.microsoft.com/default.aspx?scid=kb;en-us;837698
    std::vector<char> _belong_to_group;

    std::vector<std::shared_ptr<OptionsDescription>> _groups;
};

/** Class thrown when duplicate option description is found. */
class EINSUMS_ALWAYS_EXPORT DuplicateOptionError : public Error {
  public:
    DuplicateOptionError(std::string const &xwhat) : Error(xwhat) {}
};

} // namespace einsums::program_options

#include <einsums/config/WarningsSuffix.hpp>
