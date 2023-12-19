..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

This directory holds modularized libraries einsums is build upon. Those libraries
can be seen as independent modules, with clear dependencies and no cycles.

The tool ```create_module_skeleton.py`` can be used to generate a basic
skeleton. The structure of this skeleton should be as follows:

* ``<lib_name>/``

  * ``README.rst``
  * ``CMakeLists.txt``
  * ``cmake``
  * ``docs/``

    * ``index.rst``

  * ``examples/``

    * ``CMakeLists.txt``

  * ``include/``

    * ``einsums/``

      * ``<lib_name>``

  * ``src/``

    * ``CMakeLists.txt``

  * ``tests/``

    * ``CMakeLists.txt``
    * ``unit/``

      * ``CMakeLists.txt``

    * ``regressions/``

      * ``CMakeLists.txt``

    * ``performance/``

      * ``CMakeLists.txt``

A ``README.rst`` should be always included which explains the basic purpose of
the library and a link to the generated documentation.

The ``include`` directory should contain only headers that other libraries need.
Private headers should be placed under the ``src`` directory. This allows for
clear separation. The ``cmake`` subdirectory may include additional |cmake|_
scripts needed to generate the respective build configurations.

Documentation is placed in the ``docs`` folder. A empty skeleton for the index
is created, which is picked up by the main build system and will be part of the
generated documentation.
