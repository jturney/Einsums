..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Runtime:

=======
Runtime
=======

This module contains several runtime utilities.

See the :ref:`API reference <modules_Einsums_Runtime_api>` of this module for more
details.

----------
Public API
----------

- :cpp:func:`~einsums::initialize` - initializes the Einsums framework and executes the passed function as if it were ``main``.
- :cpp:func:`~einsums::finalize` - marks that the Einsums library may be torn down.
- :cpp:func:`~einsums::register_arguments` - adds a function that registers module-specific command line arguments to the list of start-up functions.

Runtime configuration
---------------------

The parse driver that collects ``argv``, runs each module's argument registration, and hands the
result to the option system used to be a module of its own. It is small enough now that it lives
here, beside the runtime it configures. The options themselves are declared by the modules that
read them; see :ref:`modules_Einsums_Options`.
