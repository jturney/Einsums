..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_overview:

========
Overview
========

|einsums| is organized into different sub-libraries and those in turn into modules.
The libraries and modules are independent, with clear dependencies and no
cycles. As an end-user, the use of these libraries is completely transparent. If
you use e.g. ``add_einsums_executable`` to create a target in your project you will
automatically get all modules as dependencies. See below for a list of the
available libraries and modules. Currently these are nothing more than an
internal grouping and do not affect usage. They cannot be consumed individually
at the moment.

.. toctree::
   :maxdepth: 2

   /libs/core/modules.rst
   /libs/parallelism/modules.rst
   /libs/full/modules.rst
