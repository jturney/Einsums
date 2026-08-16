..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Options:

===============
Einsums Options
===============

A small C++20 command-line parser in the spirit of LLVM's ``cl::`` utilities, simplified and
dependency-free aside from ``fmt`` for help text.

- **Options**: ``Flag``, ``Opt<T>``, ``List<T>``, ``OptEnum<Enum>``, ``Alias``
- **Named parameter tags**: ``Default(...)``, ``ImplicitValue(...)``, ``ValueName("...")``, ``Env("...")``, ``NoEnv``
- **Bindings**: ``Location<T>`` (write to external storage), ``Setter<T>`` (callback)
- **Precedence**: *defaults* < *config file* < *environment* < *command line*
- **Unknown args**: unrecognized options and everything after ``--`` are collected
- **Built-ins**: ``--help`` and ``--version`` (always available)

Quick Start
-----------

.. code-block:: cpp

   #include <Einsums/Options.hpp>

   using namespace einsums::cl;

   int main(int argc, char** argv) {
     std::vector<std::string> args(argv, argv + argc);

     // 1) Group options into categories (affects help layout)
     OptionCategory General{"General"};

     // 2) Declare options
     Flag Verbose{
       "verbose", {'v'}, "Enable verbose logging",
       General, Default(false), ImplicitValue(true)
     };

     Opt<int> Threads{
       "threads", {'t'}, "Number of threads", Default(4),
       General, RangeBetween(1, 256), ValueName("N")
     };

     Opt<std::string> Output{
       "output", {'o'}, "Output file", Default(std::string{"a.out"}),
       General, ValueName("PATH")
     };

     // Positional "gather the rest"
     List<std::string> Inputs{"inputs", Positional{}, "Input file(s)"};

     // 3) Let every option read an environment variable derived from its name
     Registry::instance().set_env_prefix("EINSUMS");

     // 4) Parse with optional config file and unknown-args collection
     std::vector<std::string> unknown;
     auto pr = parse_with_config(args, "einsums", "2.0.0",
                                 "settings.json", &unknown);
     if (!pr.ok) return pr.exit_code;

     // 5) Use the values
     fmt::print("threads = {}\n", Threads.get());
     fmt::print("output  = {}\n", Output.get());
     fmt::print("verbose = {}\n", Verbose.get());
     for (auto& f : Inputs.values()) fmt::print("input   = {}\n", f);

     if (!unknown.empty()) {
       fmt::print("unknown: ");
       for (auto& u : unknown) fmt::print("{} ", u);
       fmt::print("\n");
     }
     return 0;
   }

Run it:

.. code-block:: console

   $ ./einsums -v -t 8 -o result.bin a.txt b.txt
   threads = 8
   output  = result.bin
   verbose = true
   input   = a.txt
   input   = b.txt

   $ EINSUMS_THREADS=8 ./einsums a.txt
   threads = 8

Built-ins:

.. code-block:: console

   $ ./einsums --help
   Usage: einsums [options] <inputs>
   ...

   $ ./einsums --version
   einsums 2.0.0

Core Concepts
-------------

Categories
~~~~~~~~~~

``OptionCategory`` groups options under a heading in ``--help``.

.. code-block:: cpp

   OptionCategory IO{"I/O"};
   OptionCategory Perf{"Performance"};

Types of Options
~~~~~~~~~~~~~~~~

- ``Flag`` - boolean option. Presence sets true (configurable via ``ImplicitValue``).
- ``Opt<T>`` - single value option. ``T`` can be any integral or floating point type, ``std::string``, or a type of your own (see `Custom value types`_).
- ``List<T>`` - repeated or comma-separated values. As a positional, it *gathers remaining tokens*.
- ``OptEnum<Enum>`` - map string choices to an enum.
- ``Alias`` - forwards to a target option, optionally supplying a preset value.

Named Parameter Tags
~~~~~~~~~~~~~~~~~~~~

- ``Default(value)`` - the value used when no other source supplies one.
- ``ImplicitValue(value)`` - used when the option appears without an explicit value.
- ``ValueName("NAME")`` - placeholder in help, e.g., ``--threads <N>``.
- ``Env("NAME")`` - read this option from a named environment variable.
- ``NoEnv`` - exempt this option from the registry's automatic environment names.

Bindings & Callbacks
~~~~~~~~~~~~~~~~~~~~

- ``Location<T>(ref)`` - write the parsed value directly into external storage.
- ``Setter<T>{...}`` - invoke on assignment. The callable takes either ``(T const&)`` or ``(T const&, Source)``; the two-argument form additionally learns where the value came from.

.. code-block:: cpp

   Opt<int> Batch{
     "batch", {}, Default(32), "Batch size", Tuning,
     Setter<int>{[](int v, Source from){
       fmt::print("batch={} (source: {})\n", v, to_string(from));
     }}
   };

A plain lambda works in the same position, without naming ``Setter<T>``, and
``OnSet(...)`` attaches one after construction.

Occurrences & Visibility
~~~~~~~~~~~~~~~~~~~~~~~~

- ``Occurrence``: ``Optional`` (default), ``Required``, ``ZeroOrMore``, ``OneOrMore``.
- ``Visibility``: ``Normal`` (default), ``Hidden`` (omits from help).

Value Precedence
----------------

Every option resolves through four sources, each free to overwrite the one before it:

.. code-block:: text

   1. Default(...)    the value the programmer compiled in
   2. config file     the committed baseline, via parse_with_config
   3. environment     the per-shell or per-job override
   4. command line    the per-invocation override

``OptionBase::value_source`` records which of them supplied the value that is
in effect, and ``was_specified()`` answers whether anything other than the
default did.

.. code-block:: cpp

   if (Threads.value_source == Source::Environment) {
     fmt::print("threads came from {}\n", Threads.effective_env_name());
   }

Environment Variables
---------------------

An option becomes environment-sensitive in one of two ways.

Naming one explicitly with ``Env("NAME")`` always works, prefix policy or not:

.. code-block:: cpp

   Opt<std::string> Arch{
     "einsums:simd:arch", {}, "Highest SIMD rung to use",
     SimdCat, Env("EINSUMS_SIMD_ARCH")
   };

Installing a prefix on the registry gives *every* option a derived name at once:

.. code-block:: cpp

   Registry::instance().set_env_prefix("EINSUMS");

Derivation upper-cases the long name, turns each run of separators into a
single underscore, and prepends the prefix unless the name already carries it:

.. code-block:: text

   einsums:log:level              ->  EINSUMS_LOG_LEVEL
   einsums:profile:wait-for-viewer -> EINSUMS_PROFILE_WAIT_FOR_VIEWER
   buffer-size                    ->  EINSUMS_BUFFER_SIZE

An explicit ``Env(...)`` overrides the derived name, and ``NoEnv`` opts an
option out of derivation entirely. With no prefix set, only options that name
a variable read the environment.

Rules worth knowing:

- An unset **or empty** variable counts as not specified, so ``FOO=`` leaves the default in place.
- A value that will not parse is an error naming the variable, not a silent fallback: ``error: environment variable EINSUMS_LOG_LEVEL: invalid integer 'abc'``.
- Environment names appear in ``--help`` as ``[env: NAME]``.

Flags read presence, not a value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the command line a ``Flag`` carries its meaning in its presence. A config
file or environment variable has no presence to observe, so its value answers
that question instead: a truthy value applies the flag exactly as if it had
been named on the command line, and a falsey value leaves the default alone.

This is what makes negated flags behave. Given

.. code-block:: cpp

   Flag NoAttach{
     "einsums:debug:no-attach-debugger", {}, "Do not attach a debugger",
     DebugCat, Location(global_bools["attach-debugger"]),
     Default(true), ImplicitValue(false)
   };

``EINSUMS_DEBUG_NO_ATTACH_DEBUGGER=1`` means *yes, do not attach*, which is the
flag's ``ImplicitValue`` of false, rather than the literal ``true`` it parsed.
``=0`` means the flag was not requested, leaving ``attach-debugger`` at its
default. An ``Opt<bool>`` is unaffected by this rule: its variable supplies the
value directly.

Mutual exclusion respects precedence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Options in the same ``ExclusiveCategory`` conflict only when they are resolved
at the *same* precedence level. Passing ``--yes --no`` on one command line is a
contradiction and fails; setting one in the environment and the other on the
command line is not, because the command line simply outranks the environment.

Custom value types
------------------

``parse_value`` is an ADL customization point. Declaring one for your own type
in its own namespace is enough to use it with ``Opt<T>`` and ``List<T>``:

.. code-block:: cpp

   namespace myapp {
     struct Size { std::size_t bytes; };

     inline bool parse_value(std::string_view sv, Size &out, std::string &err) {
       // ... fill out.bytes, or set err and return false
       return true;
     }
   }

   Opt<myapp::Size> Buffer{"buffer", {}, "Buffer size", Cat};

The ``Parsable`` concept expresses this requirement, so a type without a
``parse_value`` is rejected at the point of declaration with a readable
message rather than at link time.

API Reference
-------------

Construction (variadic, named-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Flag
   Flag(std::string_view longName,
        std::initializer_list<char> shorts,
        std::string_view helpText,
        /* extras: */ OptionCategory&, ExclusiveCategory&, Visibility, Occurrence,
                     Location<bool>, Setter<bool>, Default(bool), ImplicitValue(bool),
                     Env("NAME"), NoEnv);

   // Opt<T> (without positional default value)
   template <Parsable T>
   Opt(std::string_view longName,
       std::initializer_list<char> shorts,
       std::string_view helpText,
       /* extras: */ OptionCategory&, ExclusiveCategory&, Visibility, Occurrence,
                    ValueExpected, Range, Location<T>, Setter<T>, Default(T),
                    ImplicitValue(T), ValueName("NAME"), Env("NAME"), NoEnv);

   // Opt<T> (with positional default value)
   template <Parsable T>
   Opt(std::string_view longName,
       std::initializer_list<char> shorts,
       T defaultValue,
       std::string_view helpText,
       /* extras: ... as above ... */);

   // List<T> (named)
   template <Parsable T>
   List(std::string_view longName,
        std::initializer_list<char> shorts,
        std::string_view helpText,
        /* extras: OptionCategory&, ExclusiveCategory&, Visibility, Occurrence,
                   Env("NAME"), NoEnv */);

   // List<T> (positional gather)
   template <Parsable T>
   List(std::string_view positionalName, Positional, std::string_view helpText);

   // OptEnum<Enum>
   template <typename Enum>
   OptEnum(std::string_view longName,
           std::initializer_list<char> shorts,
           Enum defaultValue,
           std::initializer_list<std::pair<std::string_view, Enum>> choices,
           std::string_view helpText,
           /* extras: OptionCategory&, Location<Enum>, Setter<Enum>, ... */);

   // Alias
   Alias(std::string_view longName,
         std::initializer_list<char> shorts,
         OptionBase& target,
         std::string_view helpText,
         /* extras: OptionCategory&, Visibility, Occurrence, a preset value string */);

Parsing
~~~~~~~

.. code-block:: cpp

   struct ParseResult { bool ok; int exit_code; };

   ParseResult parse(std::span<std::string const> args,
                     const char* programName = nullptr,
                     std::string_view version = {},
                     std::vector<std::string>* unknown_args = nullptr);

   ParseResult parse_with_config(std::span<std::string const> args,
                                 const char* programName = nullptr,
                                 std::string_view version = {},
                                 std::string_view config_path = {},
                                 std::vector<std::string>* unknown_args = nullptr);

A ``std::vector<std::string>`` converts to the ``args`` span implicitly, so
existing call sites need no change.

Behavior & Semantics
--------------------

Option Names
~~~~~~~~~~~~

- Long: ``--threads``, optionally ``--threads=8``.
- Short (bundles allowed): ``-v``, ``-abc``, ``-o12`` or ``-o 12``.

Declaring the same long name twice is a programming error and is reported at
parse time rather than letting one option silently shadow the other.

Implicit Values
~~~~~~~~~~~~~~~

For options with ``ValueExpected::ValueRequired``, the parser **only consumes the next token
as a value if it does not look like another option**. Otherwise it passes ``std::nullopt`` to the
option, allowing ``ImplicitValue(...)`` to apply. Examples:

- ``--level``  => if ``ImplicitValue(7)`` set, then 7
- ``--level=9`` => 9
- ``-l`` (last in bundle) => apply implicit value if configured

An option declared ``ValueExpected::ValueDisallowed``, such as ``--help``,
rejects ``--help=x`` rather than trying to parse the value.

Numeric look-ahead rule
~~~~~~~~~~~~~~~~~~~~~~~

Tokens like ``-5`` or ``-3.14`` are treated as **values** (not options) when they are expected
to be consumed as the next value.

Positional List Gathering
~~~~~~~~~~~~~~~~~~~~~~~~~

A positional ``List<T>`` stays "active" and gathers subsequent bare tokens. For example:

.. code-block:: cpp

   List<std::string> Inputs{"inputs", Positional{}, "Input files"};
   // ./app a.txt b.txt c.txt  => Inputs.values() == {"a.txt","b.txt","c.txt"}

Unknown Arguments
~~~~~~~~~~~~~~~~~

- Any unrecognized option (e.g., ``--weird``) or short (e.g., ``-z``) is appended to ``unknown_args``.
- Everything **after** a literal ``--`` is appended to ``unknown_args``.

Config Files
~~~~~~~~~~~~

``parse_with_config`` accepts:

- **Key/Value** (``.env``-ish): lines of ``key = value``; ``#`` starts a comment.
- **Flat JSON object**: ``{"threads": 12, "output": "a.bin", "verbose": true}``.

Keys are matched to long option names (case-insensitive).

Built-in Options
~~~~~~~~~~~~~~~~

- ``--help`` (alias ``-h``)
- ``--version``

These are always present. The parser **short-circuits**:

- On ``--help``: prints help and returns ``{ok=false, exit_code=0}``
- On ``--version``: prints version and returns ``{ok=false, exit_code=0}``

Help Layout
~~~~~~~~~~~

``--help`` prints usage, categories, normal-visibility options, and positional arguments.
Each line reports the option's default and its environment variable, and long help text wraps
to the terminal width with a hanging indent. ``COLUMNS`` overrides the detected width, which is
what a test should pin to get stable output.

``format_help(prog)`` returns the same text as a string, for callers that want to
paginate it or assert on it.

Validation & Errors
~~~~~~~~~~~~~~~~~~~

- Missing required options or invalid values cause ``{ok=false, exit_code=1}`` and a message to ``stderr``.
- ``RangeBetween(min, max)`` checks bounds. Integral and floating point bounds are kept separately, so ``RangeBetween(0.0, 1.0)`` is a genuine fractional bound rather than a truncated one.
- ``OptEnum`` reports invalid choices and lists allowed keys.

Examples
--------

Flags
~~~~~

.. code-block:: cpp

   OptionCategory Log{"Logging"};
   Flag Verbose{
     "verbose", {'v'}, "Enable verbose logging",
     Log, Default(false), ImplicitValue(true)
   };

Integers with Range and Implicit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   OptionCategory Perf{"Performance"};
   Opt<int> Threads{
     "threads", {'t'}, Default(4), "Number of threads",
     Perf, RangeBetween(1, 256), ValueName("N")
   }; // --threads=8 => 8

Strings & Paths
~~~~~~~~~~~~~~~

.. code-block:: cpp

   OptionCategory IO{"I/O"};
   Opt<std::string> Output{
     "output", {'o'}, Default(std::string{"a.out"}), "Output file",
     IO, ValueName("PATH")
   };

Lists (named and positional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   List<std::string> Include{
     "include", {'I'}, "Include directories",
     IO
   }; // --include=inc1,inc2 or --include inc1 --include inc2

   List<std::string> Inputs{"inputs", Positional{}, "Input files"};
   // gathers bare tokens at end: ./app a b c

Enums
~~~~~

.. code-block:: cpp

   enum struct Mode { Fast, Accurate, Debug };
   OptionCategory Modes{"Mode"};

   OptEnum<Mode> ModeOpt{
     "mode", {}, Mode::Fast,
     { {"fast", Mode::Fast}, {"accurate", Mode::Accurate}, {"debug", Mode::Debug} },
     "Execution mode", Modes
   };

Aliases
~~~~~~~

.. code-block:: cpp

   // Make --fast an alias for --mode=fast
   Alias FastAlias{
     "fast", {}, ModeOpt, "Alias: fast mode", Modes, std::string("fast")
   };

Binding and Callbacks
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   OptionCategory Tuning{"Tuning"};

   // Bind to external storage
   int threads_bound = 0;
   Opt<int> Threads{
     "threads", {'t'}, Default(4), "Threads",
     Tuning, Location<int>(threads_bound)
   };

   // Callback on assignment, from any source
   Opt<int> Batch{
     "batch", {}, Default(32), "Batch size",
     Tuning, Setter<int>{[](int v, Source from){
       fmt::print("Reconfiguring batch={} (source: {})\n", v, to_string(from));
     }}
   };

Thread-Safety & Reentrancy
--------------------------

- The design uses a **global registry** and **mutable option state**, so it is **not thread-safe** and **not reentrant** for concurrent parses. Serialize calls to ``parse(...)`` if needed.
- Options add themselves to the registry on construction and remove themselves on destruction, so an option with automatic storage duration is safe to declare inside a scope.
- Each ``parse`` resets the state left by the previous one, so occurrence counts and gathered ``List`` items do not accumulate across repeated parses in one process.
- Reading the environment during ``parse`` is a plain ``std::getenv``. Mutating the environment concurrently from another thread is undefined behavior on every platform; configure before threads exist.

FAQ
---

How do implicit values work with short options?
  If a short option that requires a value is the last in a bundle (``-l``), the parser will consume
  the next token only if it **doesn't look like an option**; otherwise ``ImplicitValue(...)`` applies.

Why are unknown options not an error?
  To emulate LLVM's flexibility and to ease integration with upstream tools, unknown tokens and
  passthrough arguments (after ``--``) are returned to the caller in ``unknown_args`` for further dispatch.

Can I bind directly into application config structs?
  Yes, use ``Location<T>`` to a field that outlives parsing, or use ``Setter<T>`` to translate,
  validate, and write into your own structure.

How do I find out where a value came from?
  Read ``value_source`` on the option, and ``effective_env_name()`` for the variable it consulted.

See the :ref:`API reference <modules_Einsums_Options_api>` of this module for more
details.
