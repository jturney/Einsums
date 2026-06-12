..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Config:

======
Config
======

This module contains several utilities for telling the compiler how to handle things.

See the :ref:`API reference <modules_Einsums_Config_api>` of this module for more
details.

Public Symbols
--------------

Most symbols in this module are private. However, there are some that may be useful to the user.
See the :ref:`API reference <modules_Einsums_Config_api>` for full details.

- :c:macro:`EINSUMS_COMPUTE_CODE` is defined only when Einsums is built with GPU capabilities.
- :c:macro:`EINSUMS_OMP_PARALLEL_FOR` marks a loop as parallelizable.

.. cpp:class:: GlobalConfigMap

    Contains the mappings for all of the options passed to Einsums.