//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/TypeSupport/Casting.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums::cl {

//----------------------------------------------------------------------------------------------
/// Command line option processing entry point.
///
/// Returns true on success. Otherwise, this will print the error message to stderr and exit if
/// \p Errs is not set (nullptr by default), or print the error message to \p Errs and return
/// false if \p Errs is provided.
///
/// If \p EnvVar is not nullptr, command-line options are also parsed from the environment
/// variable named by \p EnvVar. Precedence is given to occurrences from argv. This precedence
/// is currently implemented by parsing argv after the environment variable, so it is only
/// implemented correctly for options that give precedence to later occurrences. If your program
/// supports options that give precedence to earlier occurrences, you will need to extend this
/// function to support it correctly
EINSUMS_EXPORT bool parse_command_line_options(int argc, char const *argv,
                                               std::string_view Overview     = "",
                                               std::ostream    *Errs         = nullptr,
                                               char const      *EnvVar       = nullptr,
                                               bool LongOptionsUseDoubleDash = false);

/// Function pointer type for printing version information.
using VersionPrinterType = std::function<void(std::ostream &)>;

//----------------------------------------------------------------------------------------------
/// Override the default version printer using to print out the version when --version is given
/// on the command line. This allows other programs using the CommandLine utilities to print
/// their own version string.
EINSUMS_EXPORT void set_version_printer(VersionPrinterType func);

//----------------------------------------------------------------------------------------------
/// Add an extra printer to use in addition to the default one. This can be called multiple
/// times, and each time it adds a new function to the list which will be called after the
/// basic Einsums version printing is complete. Each can then add additional information
/// specific to the tool.
EINSUMS_EXPORT void add_extra_version_printer(VersionPrinterType func);

//----------------------------------------------------------------------------------------------
/// Print option values.
/// With -print-options print the difference between option values and defaults.
/// With -print-all-options print all option values.
EINSUMS_EXPORT void print_option_values();

// Forward declarations
struct Option;

/// Add a new option for parsing and provides the option it refers to.
///
/// \param O pointer to the option
/// \param Name the string name for the option to handle during parsing
///
/// Literal options are used by some parsers to register special option values.
EINSUMS_EXPORT void add_literal_option(Option &O, std::string_view Name);

//----------------------------------------------------------------------------------------------
// Flags permitted to be passed to command line arguments
//

enum class NumOccurrences : std::uint8_t {
    // Flags for the number of occurrences allowed
    Optional   = 0x00, // Zero or One occurrences
    ZeroOrMore = 0x01, // Zero or more occurrences allowed
    Required   = 0x02, // One occurrences required
    OneOrMore  = 0x03, // One or more occurrences required

    // Indicates that this option is fed anything that follows the last positional argument
    // required by the application.
    ConsumeAfter = 0x04
};

enum class ValueExpected : std::uint8_t { Optional = 0x01, Required = 0x02, Disallowed = 0x03 };

enum class OptionHidden : std::uint8_t { NotHidden = 0x00, Hidden = 0x01, ReallyHidden = 0x0 };

// This controls special features that the option might have that cause it to be
// parsed differently...
//
// Prefix - This option allows arguments that are otherwise unrecognized to be
// matched by options that are a prefix of the actual value.  This is useful for
// cases like a linker, where options are typically of the form '-lfoo' or
// '-L../../include' where -l or -L are the actual flags.  When prefix is
// enabled, and used, the value for the flag comes from the suffix of the
// argument.
//
// AlwaysPrefix - Only allow the behavior enabled by the Prefix flag and reject
// the Option=Value form.
//

enum class Formatting : std::uint8_t {
    Normal       = 0x00,
    Positional   = 0x01,
    Prefix       = 0x02,
    AlwaysPrefix = 0x03
};

enum class Misc : std::uint8_t {
    CommaSeparated     = 0x01,
    PositionalEatsArgs = 0x02,
    Sink               = 0x04,

    Grouping = 0x08,

    Default = 0x0
};

//----------------------------------------------------------------------------------------------
struct OptionCategory {
  private:
    std::string_view const _name;
    std::string_view const _description;

    EINSUMS_EXPORT void register_category();

  public:
    explicit OptionCategory(std::string_view const name, std::string_view const description = "")
        : _name(name), _description(description) {
        register_category();
    }

    [[nodiscard]] std::string_view name() const { return _name; }
    [[nodiscard]] std::string_view description() const { return _description; }
};

// The general Option Category (used as default category)
EINSUMS_EXPORT OptionCategory &general_category();

//----------------------------------------------------------------------------------------------
struct SubCommand {
  private:
    std::string_view _name;
    std::string_view _description;

  protected:
    EINSUMS_EXPORT void register_subcommand();
    EINSUMS_EXPORT void unregister_subcommand();

  public:
    SubCommand(std::string_view const name, std::string_view const description)
        : _name(name), _description(description) {
        register_subcommand();
    }
    SubCommand() = default;

    // Get the special subcommand representing no subcommand
    EINSUMS_EXPORT static SubCommand &top_level();

    // Get the special subcommand that can be used to put an option into all subcommands.
    EINSUMS_EXPORT static SubCommand &all();

    EINSUMS_EXPORT void reset();

    EINSUMS_EXPORT explicit operator bool() const;

    [[nodiscard]] std::string_view name() const { return _name; }
    [[nodiscard]] std::string_view description() const { return _description; }

    std::vector<Option *>           positional_opts;
    std::vector<Option *>           sink_opts;
    std::map<std::string, Option *> options_map;

    Option *consume_after_opt = nullptr;
};

struct SubCommandGroup {
  private:
    std::vector<SubCommand *> _subs;

  public:
    SubCommandGroup(std::initializer_list<SubCommand *> const IL) : _subs(IL) {}

    [[nodiscard]] std::vector<SubCommand *> subcommands() const { return _subs; }
};

//----------------------------------------------------------------------------------------------
struct EINSUMS_EXPORT Option {
  private:
    friend class alias;

    // Overridden by subclasses to handle the value passed into an argument. Should return true
    // if there was an error processing the argument and the program should exit.
    virtual bool handle_occurrence(unsigned pos, std::string_view argname,
                                   std::string_view arg) = 0;

    [[nodiscard]] virtual ValueExpected value_expected_flag_default() const {
        return ValueExpected::Optional;
    }

    // Out of line virtual function to provide home for the class
    virtual void anchor();

    uint16_t _num_occurrences{0};
    // Occurrences, HiddenFlag, and Formatting are all enum types but to avoid
    // problems with signed enums in bitfields.
    uint16_t _occurrences : 3; // enum NumOccurrencesFlag
    // not using the enum type for 'Value' because zero is an implementation
    // detail representing the non-value
    uint16_t _value : 2 {0};
    uint16_t _hidden_flag : 2;                                            // enum OptionHidden
    uint16_t _formatting : 2 {static_cast<uint16_t>(Formatting::Normal)}; // enum FormattingFlags
    uint16_t _misc : 5 {0};
    uint16_t _fully_initialized : 1 {false}; // Has addArgument been called?
    uint16_t _position{0};                   // Position of the last occurrence of the option
    uint16_t _additional_vals{0};            // Greater than 0 for a multivalued option.

  public:
    std::string_view              argument;
    std::string_view              help;
    std::string_view              value;
    std::vector<OptionCategory *> categories;
    std::set<SubCommand *>        subs;

    [[nodiscard]] NumOccurrences num_occurrences_flag() const {
        return static_cast<NumOccurrences>(_occurrences);
    }

    [[nodiscard]] ValueExpected value_expected_flag() const {
        return _value ? static_cast<ValueExpected>(_value) : value_expected_flag_default();
    }

    [[nodiscard]] OptionHidden option_hidden_flag() const {
        return static_cast<OptionHidden>(_hidden_flag);
    }

    [[nodiscard]] Formatting formatting_flag() const {
        return static_cast<Formatting>(_formatting);
    }

    [[nodiscard]] unsigned misc_flags() const { return _misc; }
    [[nodiscard]] unsigned position() const { return _position; }
    [[nodiscard]] unsigned num_additional_vals() const { return _additional_vals; }

    [[nodiscard]] bool has_argument() const { return !argument.empty(); }
    [[nodiscard]] bool is_positional() const {
        return formatting_flag() == Formatting::Positional;
    }
    [[nodiscard]] bool is_sink() const {
        return misc_flags() & static_cast<unsigned>(Misc::Sink);
    }
    [[nodiscard]] bool is_default_option() const {
        return misc_flags() & static_cast<unsigned>(Misc::Default);
    }

    void set_argument(std::string_view const S);
    void set_description(std::string_view S) { help = S; }
    void set_value(std::string_view S) { value = S; }
    void set_num_occurrences_flag(NumOccurrences Val) {
        _occurrences = static_cast<uint16_t>(Val);
    }
    void set_value_expected_flag(ValueExpected Val) { _value = static_cast<uint16_t>(Val); }
    void set_hidden_flag(OptionHidden Val) { _hidden_flag = static_cast<uint16_t>(Val); }
    void set_formatting_flag(Formatting Val) { _formatting = static_cast<uint16_t>(Val); }
    void set_misc_flag(Misc Val) { _misc |= static_cast<uint16_t>(Val); }
    void set_position(unsigned pos) { _position = pos; }
    void add_category(OptionCategory &C);
    void add_subcommand(SubCommand &S) { subs.insert(&S); }

  protected:
    explicit Option(NumOccurrences OccurrencesFlag, OptionHidden Hidden)
        : _occurrences(static_cast<uint16_t>(OccurrencesFlag)),
          _hidden_flag(static_cast<uint16_t>(Hidden)) {
        categories.push_back(&general_category());
    }

    void set_num_additional_vals(unsigned const n) { _additional_vals = n; }

  public:
    virtual ~Option() = default;

    /// Register this argument with the commandline system.
    void add_argument();

    /// Unregister this option from the commandline system.
    /// \note For testing purposes only.
    void remove_argument();

    /// Return the width of the option tag for printing.
    [[nodiscard]] virtual size_t get_option_width() const = 0;

    /// Print out information about this option
    virtual void print_option_info(size_t GlobalWidth) const              = 0;
    virtual void print_option_value(size_t GlobalWidth, bool Force) const = 0;
    virtual void set_default()                                            = 0;

    /// Prints the help string for an option.
    static void print_help_string(std::string_view HelpStr, size_t Ident,
                                  size_t FirstLineIndentedBy);

    /// Prints the help string for an enum value.
    static void print_enum_value_help_string(std::string_view HelpStr, size_t Indent,
                                             size_t FirstLineIdentedBy);

    virtual void get_extra_option_names(std::vector<std::string_view> &) {}

    /// Wrapper around handle_occurence that enforces Flags
    virtual bool add_occurrence(unsigned pos, std::string_view ArgName, std::string_view Value,
                                bool MultiArg = false);

    /// Prints option name followed by a message. Always returns true.
    // bool error(const)

    [[nodiscard]] int get_num_occurrences() const { return _num_occurrences; }
    void              reset();
};

//----------------------------------------------------------------------------------------------
// Command line option modifiers that can be used to modify the behavior of command line option
// parsers

/// Modifier to set the description shown in the -help output
struct desc { // NOLINT
    std::string_view description;

    desc(std::string_view str) : description(str) {}

    void apply(Option &O) const { O.set_description(description); }
};

/// Modifier to set the value description shown in the -help output
struct value_desc { // NOLINT
    std::string_view description;

    value_desc(std::string_view str) : description(str) {}

    void apply(Option &O) const { O.set_value(description); }
};

/// Specify a default (initial) value for the command line argument, if the default constructor
/// for the argument type does not give you what you want. This is only valid on "opt" arguments
/// not on "list" arguments.
template <typename Type>
struct Initializer {
    Type const &init;
    Initializer(Type const &Val) : init(Val) {}

    template <typename Opt>
    void apply(Opt &O) const {
        O.set_initial_value(init);
    }
};

template <typename Type>
struct ListInitializer {
    std::vector<Type> const &inits;

    ListInitializer(std::vector<Type> const &Vals) : inits(Vals) {}

    template <typename Opt>
    void apply(Opt &O) const {
        O.set_initial_values(inits);
    }
};

template <typename Type>
Initializer<Type> init(Type const &Val) {
    return Initializer<Type>(Val);
}

template <typename Type>
ListInitializer<Type> list_init(std::vector<Type> const &Vals) {
    return ListInitializer<Type>(Vals);
}

template <typename Type>
struct Location {
    Type &location;

    Location(Type &L) : location(L) {}

    template <typename Opt>
    void apply(Opt &O) const {
        O.set_location(O, location);
    }
};

template <typename Type>
Location<Type> location(Type &L) {
    return Location<Type>(L);
}

struct cat { // NOLINT
    OptionCategory &category;

    explicit cat(OptionCategory &c) : category(c) {}

    template <typename Opt>
    void apply(Opt &O) const {
        O.add_category(category);
    }
};

struct sub { // NOLINT
    SubCommand      *subcommand = nullptr;
    SubCommandGroup *group      = nullptr;

    explicit sub(SubCommand &S) : subcommand(&S) {}
    explicit sub(SubCommandGroup &G) : group(&G) {}

    template <typename Opt>
    void apply(Opt &O) const {
        if (subcommand) {
            O.add_subcommand(subcommand);
        } else if (group) {
            for (SubCommand *sc : group->subcommands()) {
                O.add_subcommand(*sc);
            }
        }
    }
};

template <typename R, typename Type>
struct cb { // NOLINT
    std::function<R(Type)> CB;

    explicit cb(std::function<R(Type)> CB) : CB(CB) {}

    template <typename Opt>
    void apply(Opt &O) const {
        O.set_callback(CB);
    }
};

namespace detail {
template <typename F>
struct CallbackTraits : CallbackTraits<decltype(&F::operator())> {};

template <typename R, typename C, typename... Args>
struct CallbackTraits<R (C::*)(Args...) const> {
    using ResultType = R;
    using ArgType    = std::tuple_element_t<0, std::tuple<Args...>>;
    static_assert(sizeof...(Args) == 1,
                  "Callback function must have one and only one parameter");
    static_assert(std::is_same_v<ResultType, void>, "Callback return type must be void");
    static_assert(std::is_lvalue_reference_v<ArgType> &&
                      std::is_const_v<std::remove_reference_t<ArgType>>,
                  "Callback ArgType must be a const lvalue reference");
};

} // namespace detail

template <typename F>
cb<typename detail::CallbackTraits<F>::ResultType, typename detail::CallbackTraits<F>::ArgType>
callback(F CB) {
    using ResultType = detail::CallbackTraits<F>::ResultType;
    using ArgType    = detail::CallbackTraits<F>::ArgType;
    return cb<ResultType, ArgType>(CB);
}

//----------------------------------------------------------------------------------------------

struct EINSUMS_EXPORT GenericOptionValue {
    [[nodiscard]] virtual bool compare(GenericOptionValue const &V) const = 0;

  protected:
    GenericOptionValue()                                      = default;
    GenericOptionValue(GenericOptionValue const &)            = default;
    GenericOptionValue &operator=(GenericOptionValue const &) = default;
    ~GenericOptionValue()                                     = default;

  private:
    virtual void anchor();
};

template <typename Type>
struct OptionValue;

template <typename Type, bool isClass>
struct OptionValueBase : GenericOptionValue {
    using WrapperType = OptionValue<Type>;

    [[nodiscard]] bool has_value() const { return false; }

    Type const &get_value() const { EINSUMS_UNREACHABLE; }

    // Some options may take their value from a different data type
    template <typename DT>
    void set_value(const DT &) {}

    [[nodiscard]] bool compare(Type const &) const { return false; }

    [[nodiscard]] bool compare(GenericOptionValue const &) const override { return false; }

  protected:
    ~OptionValueBase() = default;
};

template <typename Type>
struct OptionValueCopy : GenericOptionValue {
  private:
    Type _value;
    bool _valid = false;

  protected:
    OptionValueCopy(OptionValueCopy const &)            = default;
    OptionValueCopy &operator=(OptionValueCopy const &) = default;
    ~OptionValueCopy()                                  = default;

  public:
    OptionValueCopy() = default;

    [[nodiscard]] bool has_value() const { return _valid; }

    [[nodiscard]] Type const &get_value() const {
        EINSUMS_ASSERT_MSG(_valid, "Invalid option value");
        return _value;
    }

    void set_value(Type const &V) {
        _valid = true;
        _value = V;
    }

    [[nodiscard]] bool compare(Type const &V) const { return _valid && (_value == V); }

    [[nodiscard]] bool compare(GenericOptionValue const &V) const override {
        auto const &vc = static_cast<OptionValueCopy const &>(V);
        if (!vc.has_value())
            return false;
        return compare(vc.get_value());
    }
};

template <typename Type>
struct OptionValueBase<Type, false> : OptionValueCopy<Type> {
    using WrapperType = Type;

  protected:
    OptionValueBase()                                   = default;
    OptionValueBase(OptionValueBase const &)            = default;
    OptionValueBase &operator=(OptionValueBase const &) = default;
    ~OptionValueBase()                                  = default;
};

// Top-level option class
template <typename Type>
struct OptionValue final : OptionValueBase<Type, std::is_class_v<Type>> {
    OptionValue() = default;

    OptionValue(Type const &V) { set_value(V); }

    template <typename DT>
    OptionValue<Type> &operator=(const DT &V) {
        set_value(V);
        return *this;
    }
};

enum boolOrDefault { BOOL_UNSET, BOOL_TRUE, BOOL_FALSE };
template <>
struct EINSUMS_EXPORT OptionValue<boolOrDefault> final : OptionValueCopy<boolOrDefault> {
    using WrapperType = boolOrDefault;

    OptionValue() = default;

    OptionValue(boolOrDefault const &V) { set_value(V); }

    OptionValue &operator=(boolOrDefault const &V) {
        set_value(V);
        return *this;
    }

  private:
    void anchor() override;
};

template <>
struct EINSUMS_EXPORT OptionValue<std::string> final : OptionValueCopy<std::string> {
    using WrapperType = std::string_view;

    OptionValue() = default;

    OptionValue(std::string const &V) { set_value(V); }

    OptionValue &operator=(std::string const &V) {
        set_value(V);
        return *this;
    }

  private:
    void anchor() override;
};

struct OptionEnumValue {
    std::string_view name;
    int              value;
    std::string_view description;
};

#define clEnumVal(ENUMVAL, DESC)                                                                \
    einsums::cl::OptionEnumVal {                                                                \
        #ENUMVAL, int(ENUMVAL), DESC                                                            \
    }
#define clEnumValN(ENUMVAL, FLAGNAME, DESC)                                                     \
    einsums::cl::OptionEnumVal {                                                                \
        #FLAGNAME, int(ENUMVAL), DESC                                                           \
    }

struct ValuesClass {
  private:
    std::vector<OptionEnumValue> _values;

  public:
    ValuesClass(std::initializer_list<OptionEnumValue> options) : _values(options) {}

    template <typename Opt>
    void apply(Opt &O) const {
        for (auto const &value : _values) {
            O.get_parser().add_literal_option(value.name, value.value, value.description);
        }
    }
};

template <typename... OptsType>
ValuesClass values(OptsType... Options) {
    return ValuesClass({Options...});
}

struct EINSUMS_EXPORT GenericParserBase {
  protected:
    Option &owner;

    struct GenericOptionInfo {
        std::string_view name;
        std::string_view help;

        GenericOptionInfo(std::string_view name, std::string_view help)
            : name(name), help(help) {}
    };

  public:
    GenericParserBase(Option &O) : owner(O) {}

    virtual ~GenericParserBase() = default;

    [[nodiscard]] virtual unsigned                  get_num_options() const           = 0;
    [[nodiscard]] virtual std::string_view          get_option(unsigned N) const      = 0;
    [[nodiscard]] virtual std::string_view          get_description(unsigned N) const = 0;
    [[nodiscard]] virtual size_t                    get_option_width(Option const &O) const;
    [[nodiscard]] virtual GenericOptionValue const &get_option_value(unsigned N) const = 0;

    virtual void print_option_info(Option const &O, size_t GlobalWidth) const;
    void         print_generic_option_diff(Option const &O, GenericOptionValue const &V,
                                           GenericOptionValue const &Default, size_t GlobalWidth) const;

    template <typename AnyOptionValue>
    void print_option_diff(Option const &O, AnyOptionValue const &V,
                           AnyOptionValue const &Default, size_t GlobalWidth) const {
        print_generic_option_diff(O, V, Default, GlobalWidth);
    }

    void initialize() {}

    void get_extra_option_names(std::vector<std::string_view> &OptionNames) {
        if (!owner.has_argument()) {
            for (unsigned i = 0, e = get_num_options(); i != e; ++i) {
                OptionNames.push_back(get_option(i));
            }
        }
    }

    ValueExpected get_value_expected_flag_default() const {
        if (owner.has_argument())
            return ValueExpected::Required;
        return ValueExpected::Disallowed;
    }

    unsigned find_option(std::string_view Name);
};

template <typename Type>
struct Parser : GenericParserBase {
  protected:
    struct OptionInfo : GenericOptionInfo {
        OptionValue<Type> value;

        OptionInfo(std::string_view name, Type v, std::string_view help)
            : GenericOptionInfo(name, help), value(v) {}
    };
    std::vector<OptionInfo> values;

  public:
    Parser(Option &O) : GenericParserBase(O) {}

    using ParserDataType = Type;

    // Implement virtual functions needed by GenericParserBase
    [[nodiscard]] unsigned get_num_options() const override {
        return static_cast<unsigned>(values.size());
    }
    [[nodiscard]] std::string_view get_option(unsigned N) const override {
        return values[N].name;
    }
    [[nodiscard]] std::string_view get_description(unsigned N) const override {
        return values[N].help;
    }

    [[nodiscard]] GenericOptionValue const &get_option_value(unsigned N) const override {
        return values[N].value;
    }

    bool parse(Option &O, std::string_view ArgName, std::string_view Arg, Type &V) {
        std::string_view ArgVal;
        if (owner.has_argument())
            ArgVal = Arg;
        else
            ArgVal = ArgName;

        for (size_t i = 0, e = values.size(); i != e; ++i) {
            if (values[i].name == ArgVal) {
                V = values[i].value.get_value();
                return false;
            }
        }

        // TODO: Need to change this line
        return true;
    }

    template <typename DT>
    void add_literal_option(std::string_view Name, const DT &V, std::string_view HelpStr) {
#if !defined(NDEBUG)
        if (find_option(Name) != values.size()) {
            EINSUMS_LOG_ERROR("Option '{}' already exists!", Name);
        }
#endif
        OptionInfo X(Name, static_cast<Type>(V), HelpStr);
        values.push_back(X);
        add_literal_option(owner, Name);
    }

    void remove_literal_option(std::string_view Name) {
        unsigned N = find_option(Name);
        EINSUMS_ASSERT_MSG(N != values.size(), "Option not found!");
        values.erase(values.begin() + N);
    }
};

struct EINSUMS_EXPORT BasicParserImpl {
    BasicParserImpl(Option &) {}

    virtual ~BasicParserImpl() = default;

    [[nodiscard]] virtual ValueExpected get_value_expected_flag_default() const {
        return ValueExpected::Required;
    }

    void get_extra_option_names(std::vector<std::string_view> &) {}

    virtual void initialize() {}

    [[nodiscard]] size_t get_option_width(Option const &O) const;

    void print_option_info(Option const &O, size_t GlobalWidth) const;

    void print_option_no_value(Option const &O, size_t GlobalWidth) const;

    [[nodiscard]] virtual std::string_view get_value_name() const { return "value"; }

    virtual void anchor();

  protected:
    void print_option_name(Option const &O, size_t GlobalWidth) const;
};

template <typename Type>
struct BasicParser : BasicParserImpl {
    using ParserDataType = Type;
    using OptVal         = OptionValue<Type>;

    BasicParser(Option &O) : BasicParserImpl(O) {}
};

extern template class EINSUMS_EXPORT BasicParser<bool>;

template <>
struct EINSUMS_EXPORT Parser<bool> : BasicParser<bool> {
    Parser(Option &O) : BasicParser(O) {}

    bool parse(Option &O, std::string_view ArgName, std::string_view Arg, bool &Val);

    void initialize() override {}

    [[nodiscard]] ValueExpected get_value_expected_flag_default() const override {
        return ValueExpected::Optional;
    }

    [[nodiscard]] std::string_view get_value_name() const override {
        return std::string_view{};
    };

    void print_option_diff(Option const &O, bool V, OptVal Default, size_t GlobalWidth) const;

    void anchor() override;
};

} // namespace einsums::cl
