..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-add-an-option:

*********************
Add a Runtime Option
*********************

Every runtime setting in Einsums is a typed descriptor declared once, in the ``Options.hpp`` of the module that reads it.
There is no string-keyed configuration map.
Adding an option means writing a descriptor, registering it, and giving it a paragraph of prose; the command-line spelling, the environment variable, the ``--help`` entry, the negated form, and the Python binding all follow from the descriptor without another table to edit.

.. contents:: On this page
    :local:
    :depth: 1

Declare the descriptor
======================

Put it in ``libs/Einsums/<Module>/include/Einsums/<Module>/Options.hpp``, under namespace ``option``, in the module that reads the value.
Not in a central header: the point is that the declaration sits next to the code that consumes it.

.. code-block:: cpp

    #include <Einsums/Options/Get.hpp>

    EINSUMS_NAMESPACE_BEGIN(option)

    /// Write the text report at shutdown.
    inline constinit cl::ConfigOption<bool> ProfileReport =
        cl::config_flag("einsums:profile:report",
                        "Generate a profile report on exit",
                        "Profile",
                        true);

    EINSUMS_NAMESPACE_END(option)

The four arguments are the long name, the ``--help`` text, the ``--help`` heading to group under, and the default.

Descriptors are ``constinit`` aggregates of literal types, so a namespace-scope one is constant-initialized.
There is no static initialization order to reason about, and a read that happens before registration simply sees the default.

Pick the right declaration helper
=================================

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Helper
      - Use for
    * - ``cl::config_flag``
      - A boolean. The ``no-`` spelling is generated for you.
    * - ``cl::config_opt<T>``
      - A value of type ``std::int64_t``, ``double``, or ``std::string``.
    * - ``cl::config_opt_computed<T>``
      - A default only knowable at run time, such as a temporary directory or a name carrying the process id. Takes a provider function.

``config_opt`` also takes an optional placeholder shown in help and an optional inclusive range enforced after parsing:

.. code-block:: cpp

    inline constinit cl::ConfigOption<std::int64_t> ProfilePort =
        cl::config_opt<std::int64_t>("einsums:profile:port", "Profile server port",
                                     "Profile", 19216, "PORT");

**Never hand-write a negated flag.**
Declare the option in the positive and let registration generate the ``no-`` twin.
A hand-written negation that never parsed is exactly how the argument reference went wrong before.

Register it
===========

Add it to the module's registration function in ``libs/Einsums/<Module>/src/Options.cpp``:

.. code-block:: cpp

    int register_Einsums_Profile_options() {
        cl::register_option(option::ProfileReport);
        cl::register_option(option::ProfilePort);
        // ... one line per option
        return 0;
    }

The header ends with a namespace-scope initializer that calls it:

.. code-block:: cpp

    EINSUMS_EXPORT int register_Einsums_<Module>_options();

    namespace detail {
    [[maybe_unused]] static int const register_options_Einsums_<Module> =
        register_Einsums_<Module>_options();
    }

Register from that initializer and not from the ``register_arguments`` hook.
The hook fires part-way through ``initialize()``, and the Python binding layer enumerates the registry before that point, so an option registered from the hook is missing from Python.

Read it
=======

.. code-block:: cpp

    #include <Einsums/Options/Get.hpp>

    if (config::get(option::ProfileReport)) {
        // ...
    }

No key string, and no fallback argument.
Reads are lock-free atomic slot loads, safe from any thread at any time, and cheap enough to sit in a tensor constructor.
Registration and parsing happen at startup, after which the registry freezes.

Two companions exist for narrower needs.
``config::try_get`` distinguishes an unset option from one sitting at its default, and ``config::set`` writes, which is mostly for tests.

Include the right header.
``Get.hpp`` is what a reader needs and is deliberately light, pulling in neither the parser nor ``fmt``.
``Declare.hpp`` is for declaring, and ``Parse.hpp`` belongs only in the parse driver.

What you get for free
=====================

Everything below is derived from the long name, which is what keeps the spellings from drifting apart:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Surface
      - Derivation
    * - Config key
      - Drop a leading ``einsums:``, turn remaining ``:`` into ``-``. So ``einsums:log:level`` keys on ``log-level``.
    * - Environment variable
      - Upper-case the long name, each run of separators becoming one underscore: ``EINSUMS_PROFILE_REPORT``.
    * - Negated flag
      - The ``no-`` goes on the last segment: ``einsums:profile:no-report``.
    * - ``--help`` entry
      - From the help text and category, with the default in effect for the build.
    * - Python
      - ``einsums.rc`` is generated from the descriptors, so the option reaches Python with no table to edit.

Document it, or the build fails
===============================

``docs/sphinx/user/arguments.rst`` is generated by ``docs/tools/generate_arguments_rst.py`` from two inputs that are checked against each other: the C++ descriptors, and the prose in ``docs/sphinx/user/arguments.yaml``.

Add an entry to the YAML file, keyed by the descriptor's long name without leading dashes:

.. code-block:: yaml

    "einsums:profile:report":
      added: "2.0.0"
      body: |
        Write the profiler's text report when the process exits.

        The file is named by :option:`--einsums:profile:filename` and appended to
        unless :option:`--einsums:profile:no-append` is given.

Do not repeat the name, type, default, category, or placeholder: those come from the descriptor, and stating them twice is what the split exists to prevent.

A descriptor with no YAML entry fails the build, and so does a YAML entry for an option no descriptor declares.
That is the point rather than an inconvenience: adding an option without documenting it is not possible.

The generator also checks every ``--einsums:...`` spelling your prose mentions against the registry, negations included.
Version history is exempt, since it describes names that deliberately no longer exist.

When a dynamic key is justified
===============================

``config::get_dynamic`` and ``config::set_dynamic`` take runtime strings.
They exist for genuinely runtime-constructed keys, of which the optimizer's per-pass flags are the example.

A name known at compile time belongs in a descriptor.
The dynamic path restores the silent-typo failure mode that descriptors were introduced to remove, so reach for it only when the name truly is not known until the program runs.

Checklist
=========

#. Descriptor in the reading module's ``Options.hpp``, under namespace ``option``.
#. ``cl::register_option`` line in that module's ``src/Options.cpp``.
#. Reads go through ``config::get(option::X)``, including ``<Einsums/Options/Get.hpp>``.
#. Entry in ``docs/sphinx/user/arguments.yaml``.
#. Build the docs; the generator will tell you if the two halves disagree.

Next
====

- :ref:`howto-add-a-module` if the option belongs to a module that does not exist yet.
- :doc:`/user/arguments` for the reference the generator produces.
