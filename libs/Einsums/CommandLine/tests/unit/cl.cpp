//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine.hpp>
#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/Utilities/SetEnv.hpp>

#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums::cl;

namespace {

/// Restores the registry, so options declared in a test case cannot leak into
/// the next one through the process-wide registry.
struct CLITestFixture {
    CLITestFixture() { Registry::instance().clear_for_tests(); }
    ~CLITestFixture() { Registry::instance().clear_for_tests(); }
};

/// Sets an environment variable for the lifetime of a scope.
struct ScopedEnv {
    std::string name;

    ScopedEnv(std::string n, std::string const &value) : name(std::move(n)) { einsums::set_env_var(name, value); }
    ScopedEnv(ScopedEnv const &)            = delete;
    ScopedEnv &operator=(ScopedEnv const &) = delete;
    ~ScopedEnv() { einsums::unset_env_var(name); }
};

std::vector<std::string> to_args(std::initializer_list<char const *> il) {
    return {il.begin(), il.end()};
}

} // namespace

TEST_CASE("Defaults and explicit override", "[opt][defaults]") {
    CLITestFixture _;

    OptionCategory Cat{"T1"};
    int            threads_bound = 0;
    Opt<int>       Threads{"t1-threads", {'T'}, "threads", Default(4), Cat, RangeBetween(1, 256), ValueName("N")};
    Threads.OnSet([&](int v) { threads_bound = v; });

    SECTION("No args -> default 4") {
        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Threads.get() == 4);
        REQUIRE(threads_bound == 4);
        REQUIRE(Threads.value_source == Source::Default);
        REQUIRE_FALSE(Threads.was_specified());
    }

    SECTION("Explicit CLI override -> 8") {
        auto args = to_args({"prog", "--t1-threads", "8"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Threads.get() == 8);
        REQUIRE(threads_bound == 8);
        REQUIRE(Threads.value_source == Source::CommandLine);
        REQUIRE(Threads.was_specified());
    }

    SECTION("A setter may also take the source") {
        Source seen = Source::None;
        Threads.OnSet([&](int, Source src) { seen = src; });
        auto args = to_args({"prog", "--t1-threads=8"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(seen == Source::CommandLine);
    }
}

TEST_CASE("Implicit value when present without argument", "[opt][implicit]") {
    CLITestFixture _;

    OptionCategory Cat{"T2"};
    Opt<int>       Level{"t2-level", {'l'}, "level", Cat, ImplicitValue(7), ValueName("N")};

    SECTION("Appears without value -> implicit 7") {
        auto args = to_args({"prog", "--t2-level"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Level.get() == 7);
    }

    SECTION("Appears with value -> explicit wins (9)") {
        auto args = to_args({"prog", "--t2-level=9"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Level.get() == 9);
    }
}

TEST_CASE("Flag default and implicit override", "[flag]") {
    CLITestFixture _;
    OptionCategory Cat{"T3"};
    bool           verbose_bound = false;
    bool           yes_no_test   = false;
    Flag           Verbose{"t3-verbose", {'v'}, "verbose logging", Cat, Default(false), ImplicitValue(true)};
    Flag           YesFlag{"yes", {}, "yes flag", Cat, Location(yes_no_test)};
    Flag           NoFlag{"no", {}, "no flag", Cat, Location(yes_no_test), ImplicitValue(false)};
    Verbose.OnSet([&](bool on) { verbose_bound = on; });
    auto exclusion = make_yes_no(YesFlag, NoFlag);

    SECTION("Default false") {
        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE_FALSE(Verbose.get());
        REQUIRE_FALSE(verbose_bound);
    }

    SECTION("Presence -> true") {
        auto args = to_args({"prog", "-v"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Verbose.get());
        REQUIRE(verbose_bound);
    }

    SECTION("Explicit false via value") {
        auto args = to_args({"prog", "--t3-verbose=false"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE_FALSE(Verbose.get());
    }

    SECTION("Yes flags") {
        auto args1 = to_args({"prog", "--yes"});

        auto pr = parse(args1);
        REQUIRE(pr.ok);
        REQUIRE(yes_no_test);
    }

    SECTION("No flags") {
        auto args2 = to_args({"prog", "--no"});

        auto pr = parse(args2);
        REQUIRE(pr.ok);
        REQUIRE(!yes_no_test);
    }

    SECTION("Both flags") {
        auto args3 = to_args({"prog", "--yes", "--no"});

        auto pr = parse(args3);
        REQUIRE(!pr.ok);
    }
}

TEST_CASE("Bundled short options and attached value", "[short][bundle]") {
    CLITestFixture _;

    OptionCategory Cat{"T4"};
    Flag           A{"t4-a", {'a'}, "flag A", Cat};
    Flag           B{"t4-b", {'b'}, "flag B", Cat};
    Opt<int>       O{"t4-o", {'o'}, "opt O", Cat, ValueName("N")};

    SECTION("-ab sets both; -o12 attaches value") {
        auto args = to_args({"prog", "-ab", "-o12"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(A.get());
        REQUIRE(B.get());
        REQUIRE(O.get() == 12);
    }

    SECTION("-o 34 with space") {
        auto args = to_args({"prog", "-o", "34"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(O.get() == 34);
    }
}

TEST_CASE("Positional list captures multiple tokens", "[positional][list]") {
    CLITestFixture _;

    OptionCategory    Cat{"T5"};
    List<std::string> Inputs{"t5-inputs", Positional{}, "inputs"};

    auto args = to_args({"prog", "a.txt", "b.txt", "c.txt"});
    auto pr   = parse(args);
    REQUIRE(pr.ok);
    auto const &vals = Inputs.values();
    REQUIRE(vals.size() == 3);
    REQUIRE(vals[0] == "a.txt");
    REQUIRE(vals[1] == "b.txt");
    REQUIRE(vals[2] == "c.txt");
    REQUIRE(Inputs.occurrences == 3);

    SECTION("A second parse starts from an empty list") {
        auto args2 = to_args({"prog", "d.txt"});
        auto pr2   = parse(args2);
        REQUIRE(pr2.ok);
        REQUIRE(Inputs.values().size() == 1);
        REQUIRE(Inputs.values()[0] == "d.txt");
    }
}

TEST_CASE("Enum option maps strings to enum values", "[enum]") {
    CLITestFixture _;

    enum struct Mode { Fast, Accurate };

    OptionCategory Cat{"T6"};
    Mode           bound = Mode::Accurate;
    OptEnum<Mode>  M{"t6-mode", {'m'}, Mode::Fast, {{"fast", Mode::Fast}, {"accurate", Mode::Accurate}}, "mode", Cat, Location(bound)};

    SECTION("Default is Fast, and reaches the bound location") {
        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(M.to_string() == "fast");
        REQUIRE(bound == Mode::Fast);
    }

    SECTION("Set to accurate") {
        auto args = to_args({"prog", "--t6-mode", "accurate"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(M.to_string() == "accurate");
        REQUIRE(M.get() == Mode::Accurate);
        REQUIRE(bound == Mode::Accurate);
    }

    SECTION("Invalid value errors and lists the choices") {
        auto args = to_args({"prog", "--t6-mode", "banana"});
        auto pr   = parse(args);
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 1);
    }

    SECTION("A setter fires with the source") {
        Source seen = Source::None;
        M.OnSet([&](Mode const &, Source src) { seen = src; });
        auto args = to_args({"prog", "--t6-mode=accurate"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(seen == Source::CommandLine);
    }
}

TEST_CASE("Numeric option types beyond int", "[opt][types]") {
    CLITestFixture _;

    OptionCategory   Cat{"T6b"};
    Opt<unsigned>    U{"t6b-u", {}, "unsigned", Cat, Default(1U)};
    Opt<float>       F{"t6b-f", {}, "float", Cat, Default(0.0F)};
    Opt<std::size_t> Z{"t6b-z", {}, "size_t", Cat};

    auto args = to_args({"prog", "--t6b-u=42", "--t6b-f=1.5", "--t6b-z=4096"});
    auto pr   = parse(args);
    REQUIRE(pr.ok);
    REQUIRE(U.get() == 42U);
    REQUIRE(F.get() == Catch::Approx(1.5F));
    REQUIRE(Z.get() == 4096U);
}

TEST_CASE("Range validation enforces bounds", "[range]") {
    CLITestFixture _;

    OptionCategory Cat{"T7"};
    Opt<int>       R{"t7-ranged", {'r'}, "ranged", Default(10), Cat, RangeBetween(5, 15)};

    SECTION("In range ok") {
        auto args = to_args({"prog", "--t7-ranged=12"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(R.get() == 12);
    }

    SECTION("Out of range produces error") {
        auto args = to_args({"prog", "--t7-ranged=100"});
        auto pr   = parse(args);
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 1);
    }
}

TEST_CASE("A fractional bound is not truncated to an integer", "[range][float]") {
    CLITestFixture _;

    OptionCategory Cat{"T7b"};
    Opt<double>    D{"t7b-frac", {}, "fraction", Cat, Default(0.5), RangeBetween(0.0, 1.0)};

    SECTION("Inside the fractional range") {
        auto args = to_args({"prog", "--t7b-frac=0.75"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(D.get() == Catch::Approx(0.75));
    }

    SECTION("Just outside is rejected, not rounded into range") {
        auto args = to_args({"prog", "--t7b-frac=1.25"});
        auto pr   = parse(args);
        REQUIRE_FALSE(pr.ok);
    }
}

TEST_CASE("Config precedence: defaults < config < CLI", "[config][precedence]") {
    CLITestFixture _;

    OptionCategory Cat{"T8"};
    int            threads_cfg = 0;
    Opt<int>       T{"t8-threads", {'t'}, "threads", Default(2), Cat, Location<int>(threads_cfg)};

    std::map<std::string, std::string, std::less<>> cfg;
    cfg["t8-threads"] = "6";

    SECTION("Config overrides default") {
        auto                     args = to_args({"prog"});
        std::vector<std::string> unknown;
        auto                     pr = parse_internal(args, "prog", "1.0", &cfg, &unknown);
        REQUIRE(pr.ok);
        REQUIRE(T.get() == 6);
        REQUIRE(threads_cfg == 6);
        REQUIRE(T.value_source == Source::ConfigFile);
    }

    SECTION("CLI overrides config") {
        auto                     args = to_args({"prog", "--t8-threads=9"});
        std::vector<std::string> unknown;
        auto                     pr = parse_internal(args, "prog", "1.0", &cfg, &unknown);
        REQUIRE(pr.ok);
        REQUIRE(T.get() == 9);
        REQUIRE(threads_cfg == 9);
        REQUIRE(T.value_source == Source::CommandLine);
    }
}

TEST_CASE("Environment variable names derive from the option name", "[env][naming]") {
    REQUIRE(derive_env_name("EINSUMS", "einsums:log:level") == "EINSUMS_LOG_LEVEL");
    REQUIRE(derive_env_name("EINSUMS", "einsums:profile:wait-for-viewer") == "EINSUMS_PROFILE_WAIT_FOR_VIEWER");

    SECTION("A name that is not already namespaced gains the prefix") {
        REQUIRE(derive_env_name("EINSUMS", "buffer-size") == "EINSUMS_BUFFER_SIZE");
    }

    SECTION("An already-prefixed name does not pick the prefix up twice") {
        REQUIRE(derive_env_name("EINSUMS", "EINSUMS_LOG_LEVEL") == "EINSUMS_LOG_LEVEL");
    }

    SECTION("A prefix that merely starts the same is not mistaken for it") {
        REQUIRE(derive_env_name("EIN", "einsums:log:level") == "EIN_EINSUMS_LOG_LEVEL");
    }

    SECTION("Runs of separators collapse") {
        REQUIRE(derive_env_name("P", "a--b..c") == "P_A_B_C");
    }

    SECTION("No prefix means no derived name") {
        REQUIRE(derive_env_name("", "einsums:log:level").empty());
    }
}

TEST_CASE("Environment precedence: defaults < config < env < CLI", "[env][precedence]") {
    CLITestFixture _;
    ScopedEnv      env{"T9_THREADS", "9"};

    Registry::instance().set_env_prefix("T9");

    OptionCategory Cat{"T9"};
    int            bound = 0;
    Opt<int>       T{"threads", {'t'}, "threads", Cat, Default(2), Location<int>(bound)};

    SECTION("Environment overrides the default") {
        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(T.get() == 9);
        REQUIRE(bound == 9);
        REQUIRE(T.value_source == Source::Environment);
    }

    SECTION("Environment overrides a config file") {
        std::map<std::string, std::string, std::less<>> cfg{{"threads", "6"}};
        auto                                            args = to_args({"prog"});
        auto                                            pr   = parse_internal(args, "prog", "1.0", &cfg, nullptr);
        REQUIRE(pr.ok);
        REQUIRE(T.get() == 9);
        REQUIRE(T.value_source == Source::Environment);
    }

    SECTION("The command line overrides the environment") {
        auto args = to_args({"prog", "--threads=12"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(T.get() == 12);
        REQUIRE(T.value_source == Source::CommandLine);
    }
}

TEST_CASE("Environment variable opt-in and opt-out", "[env][naming]") {
    CLITestFixture _;

    SECTION("Without a prefix, only an explicit Env() reads the environment") {
        ScopedEnv derived{"T10_ALPHA", "5"};
        ScopedEnv explicitly{"T10_PINNED", "7"};

        OptionCategory Cat{"T10"};
        Opt<int>       A{"alpha", {}, "derived name, no prefix set", Cat, Default(1)};
        Opt<int>       B{"beta", {}, "pinned name", Cat, Default(1), Env("T10_PINNED")};

        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(A.get() == 1);
        REQUIRE(B.get() == 7);
    }

    SECTION("NoEnv exempts an option from a prefix policy") {
        Registry::instance().set_env_prefix("T10");
        ScopedEnv derived{"T10_ALPHA", "5"};
        ScopedEnv exempt{"T10_GAMMA", "5"};

        OptionCategory Cat{"T10"};
        Opt<int>       A{"alpha", {}, "derived", Cat, Default(1)};
        Opt<int>       G{"gamma", {}, "exempt", Cat, Default(1), NoEnv};

        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(A.get() == 5);
        REQUIRE(G.get() == 1);
    }

    SECTION("An explicit Env() beats the derived name") {
        Registry::instance().set_env_prefix("T10");
        ScopedEnv derived{"T10_ALPHA", "5"};
        ScopedEnv pinned{"LEGACY_ALPHA", "7"};

        OptionCategory Cat{"T10"};
        Opt<int>       A{"alpha", {}, "pinned", Cat, Default(1), Env("LEGACY_ALPHA")};

        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(A.get() == 7);
    }
}

TEST_CASE("An empty environment variable counts as unset", "[env]") {
    CLITestFixture _;
    ScopedEnv      env{"T11_ALPHA", ""};
    Registry::instance().set_env_prefix("T11");

    OptionCategory Cat{"T11"};
    Opt<int>       A{"alpha", {}, "alpha", Cat, Default(3)};

    auto args = to_args({"prog"});
    auto pr   = parse(args);
    REQUIRE(pr.ok);
    REQUIRE(A.get() == 3);
    REQUIRE(A.value_source == Source::Default);
}

TEST_CASE("An unparseable environment variable is an error naming the variable", "[env][errors]") {
    CLITestFixture _;
    ScopedEnv      env{"T12_ALPHA", "banana"};
    Registry::instance().set_env_prefix("T12");

    OptionCategory Cat{"T12"};
    Opt<int>       A{"alpha", {}, "alpha", Cat, Default(3)};

    auto args = to_args({"prog"});
    auto pr   = parse(args);
    REQUIRE_FALSE(pr.ok);
    REQUIRE(pr.exit_code == 1);
}

TEST_CASE("A flag reads the environment as presence, not as a raw value", "[env][flag]") {
    CLITestFixture _;
    Registry::instance().set_env_prefix("T13");

    // The shape every negated Einsums flag has: a positive binding, a negative
    // name, and an implicit value of false.
    SECTION("A truthy variable applies the flag's implicit value") {
        ScopedEnv env{"T13_NO_ATTACH", "1"};

        OptionCategory Cat{"T13"};
        bool           attach = true;
        Flag           NoAttach{"no-attach", {}, "do not attach", Cat, Location(attach), Default(true), ImplicitValue(false)};

        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE_FALSE(attach); // "yes, do not attach"
    }

    SECTION("A falsey variable leaves the default alone") {
        ScopedEnv env{"T13_NO_ATTACH", "0"};

        OptionCategory Cat{"T13"};
        bool           attach = false;
        Flag           NoAttach{"no-attach", {}, "do not attach", Cat, Location(attach), Default(true), ImplicitValue(false)};

        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(attach); // the flag was not requested
    }

    SECTION("A plain flag still turns on") {
        ScopedEnv env{"T13_VERBOSE", "yes"};

        OptionCategory Cat{"T13"};
        Flag           Verbose{"verbose", {}, "verbose", Cat, Default(false)};

        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Verbose.get());
    }
}

TEST_CASE("Mutually exclusive options conflict only at the same precedence", "[env][exclusive]") {
    CLITestFixture _;
    Registry::instance().set_env_prefix("T14");

    SECTION("The command line outranks the environment instead of conflicting") {
        ScopedEnv env{"T14_YES", "1"};

        OptionCategory Cat{"T14"};
        bool           choice = false;
        Flag           Yes{"yes", {}, "yes", Cat, Location(choice)};
        Flag           No{"no", {}, "no", Cat, Location(choice), ImplicitValue(false)};
        auto           exclusion = make_yes_no(Yes, No);

        auto args = to_args({"prog", "--no"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE_FALSE(choice);
    }

    SECTION("Two environment variables at the same level do conflict") {
        ScopedEnv yes{"T14_YES", "1"};
        ScopedEnv no{"T14_NO", "1"};

        OptionCategory Cat{"T14"};
        bool           choice = false;
        Flag           Yes{"yes", {}, "yes", Cat, Location(choice)};
        Flag           No{"no", {}, "no", Cat, Location(choice), ImplicitValue(false)};
        auto           exclusion = make_yes_no(Yes, No);

        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE_FALSE(pr.ok);
    }
}

TEST_CASE("Unknown args collection (including after --)", "[unknown]") {
    CLITestFixture _;

    OptionCategory Cat{"T15"};
    Flag           K{"t15-known", {'k'}, "known", Cat};

    auto                     args = to_args({"prog", "--nope", "-z", "--", "pos1", "--still", "-x"});
    std::vector<std::string> unknown;
    auto                     pr = parse_internal(args, "prog", "1.0", nullptr, &unknown);
    REQUIRE(pr.ok);

    REQUIRE_FALSE(K.get()); // known flag not present

    REQUIRE(unknown.size() == 5);
    REQUIRE(unknown[0] == "--nope");
    REQUIRE(unknown[1] == "-z");
    REQUIRE(unknown[2] == "pos1");
    REQUIRE(unknown[3] == "--still");
    REQUIRE(unknown[4] == "-x");
}

TEST_CASE("Builtins: --help and --version exit 0", "[builtins]") {
    CLITestFixture _;

    SECTION("--help exits 0") {
        auto args = to_args({"prog", "--help"});
        auto pr   = parse(args, "prog", "9.9");
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 0);
    }
    SECTION("--version exits 0") {
        auto args = to_args({"prog", "--version"});
        auto pr   = parse(args, "prog", "9.9");
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 0);
    }
    SECTION("--help rejects a value") {
        auto args = to_args({"prog", "--help=please"});
        auto pr   = parse(args, "prog", "9.9");
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 1);
    }
}

TEST_CASE("Repeated parses do not accumulate registry entries", "[registry]") {
    CLITestFixture _;

    OptionCategory Cat{"T16"};
    Flag           F{"t16-flag", {}, "flag", Cat};

    auto const before = Registry::instance().options.size();

    auto args = to_args({"prog"});
    REQUIRE(parse(args).ok);
    auto const after_one = Registry::instance().options.size();

    REQUIRE(parse(args).ok);
    auto const after_two = Registry::instance().options.size();

    // The built-ins join the registry on the first parse and stay there; what
    // must not happen is the count growing on every subsequent parse.
    REQUIRE(after_one >= before);
    REQUIRE(after_two == after_one);
}

TEST_CASE("An option removes itself from the registry when destroyed", "[registry]") {
    CLITestFixture _;

    auto const before = Registry::instance().options.size();
    {
        OptionCategory Cat{"T17"};
        Flag           F{"t17-flag", {}, "flag", Cat};
        REQUIRE(Registry::instance().options.size() == before + 1);
    }
    REQUIRE(Registry::instance().options.size() == before);
}

TEST_CASE("A long name declared twice is reported", "[registry][errors]") {
    CLITestFixture _;

    OptionCategory Cat{"T18"};
    Flag           A{"t18-dup", {}, "first", Cat};
    Flag           B{"t18-dup", {}, "second", Cat};

    auto args = to_args({"prog"});
    auto pr   = parse(args);
    REQUIRE_FALSE(pr.ok);
    REQUIRE(pr.exit_code == 1);
}

TEST_CASE("Help text reports defaults and environment variables", "[help]") {
    CLITestFixture _;
    Registry::instance().set_env_prefix("T19");

    OptionCategory Cat{"T19"};
    int            bound = 0;
    // A bound option still knows its default; before, Location() suppressed it.
    Opt<int>         Threads{"threads", {'t'}, "How many threads to use", Cat, Default(4), Location<int>(bound), ValueName("N")};
    Opt<std::string> Quiet{"quiet", {}, "Quiet mode", Cat, Default(std::string{"no"}), NoEnv};

    auto const help = format_help("prog");

    REQUIRE(help.find("--threads <N>") != std::string::npos);
    REQUIRE(help.find("(default: 4)") != std::string::npos);
    REQUIRE(help.find("[env: T19_THREADS]") != std::string::npos);
    REQUIRE(help.find("(default: no)") != std::string::npos);
    REQUIRE(help.find("[env: T19_QUIET]") == std::string::npos);
}

// ---------------------------------------------------------------------------
// Descriptors: one declaration carrying the name, help, type, and default.
// ---------------------------------------------------------------------------

TEST_CASE("Derived spellings", "[descriptor]") {
    REQUIRE(derive_key("einsums:log:level") == "log-level");
    REQUIRE(derive_key("einsums:buffer-size") == "buffer-size");
    REQUIRE(derive_key("einsums:graph:profile-groups") == "graph-profile-groups");
    REQUIRE(derive_key("bare-name") == "bare-name");

    REQUIRE(derive_negated_name("einsums:debug:attach-debugger") == "einsums:debug:no-attach-debugger");
    REQUIRE(derive_negated_name("bare") == "no-bare");
}

TEST_CASE("A descriptor read before registration yields its declared default", "[descriptor]") {
    CLITestFixture _;

    ConfigOption<bool>        flag = config_flag("t20:unregistered-flag", "never registered", "T20", true);
    ConfigOption<std::string> text = config_opt<std::string>("t20:unregistered-text", "never registered", "T20", "fallback");

    REQUIRE(einsums::config::get(flag) == true);
    REQUIRE(einsums::config::get(text) == "fallback");
    REQUIRE_FALSE(einsums::config::try_get(flag).has_value());
}

TEST_CASE("A registered descriptor reads what the command line supplied", "[descriptor]") {
    CLITestFixture _;

    ConfigOption<bool>         verbose = config_flag("t21:verbose", "chatty", "T21", false);
    ConfigOption<bool>         guard   = config_flag("t21:guard", "on unless told otherwise", "T21", true);
    ConfigOption<std::int64_t> level   = config_opt<std::int64_t>("t21:level", "how loud", "T21", 3, "N", RangeBetween(0, 9));
    ConfigOption<std::string>  name    = config_opt<std::string>("t21:name", "what to call it", "T21", "anonymous");

    register_option(verbose);
    register_option(guard);
    register_option(level);
    register_option(name);

    SECTION("Nothing on the command line -> the declared defaults") {
        auto args = to_args({"prog"});
        REQUIRE(parse(args).ok);
        REQUIRE(einsums::config::get(verbose) == false);
        REQUIRE(einsums::config::get(guard) == true);
        REQUIRE(einsums::config::get(level) == 3);
        REQUIRE(einsums::config::get(name) == "anonymous");
    }

    SECTION("The generated negation turns a default-true flag off") {
        auto args = to_args({"prog", "--t21:no-guard"});
        REQUIRE(parse(args).ok);
        REQUIRE(einsums::config::get(guard) == false);
    }

    SECTION("The positive spelling turns a default-false flag on") {
        auto args = to_args({"prog", "--t21:verbose"});
        REQUIRE(parse(args).ok);
        REQUIRE(einsums::config::get(verbose) == true);
        REQUIRE(einsums::config::try_get(verbose).value_or(false) == true);
    }

    SECTION("Values parse and range-check through the descriptor") {
        auto args = to_args({"prog", "--t21:level", "7", "--t21:name", "einsums"});
        REQUIRE(parse(args).ok);
        REQUIRE(einsums::config::get(level) == 7);
        REQUIRE(einsums::config::get(name) == "einsums");

        auto bad = to_args({"prog", "--t21:level", "42"});
        REQUIRE_FALSE(parse(bad).ok);
    }

    SECTION("Naming both halves of a generated pair is a contradiction") {
        auto args = to_args({"prog", "--t21:guard", "--t21:no-guard"});
        REQUIRE_FALSE(parse(args).ok);
    }
}

TEST_CASE("A descriptor reads the environment under its derived name", "[descriptor][env]") {
    CLITestFixture _;
    Registry::instance().set_env_prefix("T22");

    ConfigOption<std::int64_t> level = config_opt<std::int64_t>("t22:level", "how loud", "T22", 1, "N");
    register_option(level);

    ScopedEnv const env{"T22_LEVEL", "5"};

    auto args = to_args({"prog"});
    REQUIRE(parse(args).ok);
    REQUIRE(einsums::config::get(level) == 5);
}

TEST_CASE("Help lists the positive spelling and points at its negation", "[descriptor][help]") {
    CLITestFixture _;

    ConfigOption<bool> guard = config_flag("t23:guard", "keep the guard on", "T23", true);
    register_option(guard);

    auto const help = format_help("prog");

    REQUIRE(help.find("--t23:guard") != std::string::npos);
    REQUIRE(help.find("(default: true)") != std::string::npos);
    REQUIRE(help.find("[negate: --t23:no-guard]") != std::string::npos);
    // The negation itself is hidden, so the table does not list every flag twice.
    REQUIRE(help.find("  --t23:no-guard ") == std::string::npos);
}

TEST_CASE("A write reaches the slot and whoever asked to hear about it", "[descriptor][observer]") {
    CLITestFixture _;

    ConfigOption<std::string> name  = config_opt<std::string>("t24:name", "what to call it", "T24", "anonymous");
    ConfigOption<bool>        guard = config_flag("t24:guard", "on unless told otherwise", "T24", true);
    register_option(name);
    register_option(guard);

    int         calls = 0;
    std::string seen;
    on_change(name, [&] {
        ++calls;
        seen = einsums::config::get(name);
    });

    einsums::config::set(name, std::string("einsums"));
    REQUIRE(einsums::config::get(name) == "einsums");
    REQUIRE(calls == 1);
    REQUIRE(seen == "einsums");

    einsums::config::set(guard, false);
    REQUIRE(einsums::config::get(guard) == false);
    // The observer belongs to the other option and must not have fired.
    REQUIRE(calls == 1);
}

TEST_CASE("Dynamic keys reach the same slots, spelled loosely", "[descriptor][dynamic]") {
    CLITestFixture _;

    ConfigOption<std::string> name = config_opt<std::string>("t25:name", "what to call it", "T25", "anonymous");
    register_option(name);

    // The derived key for t25:name is t25-name; case and underscores fold.
    einsums::config::set_dynamic<std::string>("T25_NAME", "einsums");
    REQUIRE(einsums::config::get(name) == "einsums");
    REQUIRE(einsums::config::get_dynamic<std::string>("t25-name", "fallback") == "einsums");

    // A key no descriptor claims answers with the caller's default until it is
    // written, and reads of a slot nobody set still answer with the default.
    REQUIRE(einsums::config::get_dynamic<std::int64_t>("t25-unclaimed", 7) == 7);
    einsums::config::set_dynamic<std::int64_t>("t25-unclaimed", 11);
    REQUIRE(einsums::config::get_dynamic<std::int64_t>("t25-unclaimed", 7) == 11);
    REQUIRE(einsums::config::get_dynamic<bool>("t25-unclaimed", true) == true);
}
