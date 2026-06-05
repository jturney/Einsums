..
    ----------------------------------------------------------------------------------------------
      Copyright (c) The Einsums Developers. All rights reserved.
      Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _create_module_skeleton:

Creating Module Skeletons
=========================

In the ``libs`` directory, there is a Python package called ``create_module_skeleton``. This package
sets up a module skeleton to help you create new modules. This page will act as the documentation
on how to call it and modify it.


Usage
-----

.. code::

    python3 create_module_skeleton [OPTIONS] LIBRARY_NAME MODULE_NAME
    python3 create_module_skeleton --reindex [LIBRARIES, ...]

Options
-------

``-h, --help``: Print a help message.

``--python``: If included, the new package will be set up as a Python extension module.

``--gpu``: If included, the new package will be set up as a GPU-enabled module.

``--python-name PYTHON_NAME``: If included, the name of the library as seen by Python. This
may be different from the name of the library. For instance, the ``Python`` module is given
the Python name of ``_core``.

``--reindex``: Goes through and updates the CMake files and auto-generated documentation to
see the modules that are currently present. If no libraries are given, it will reindex all
libraries in the current directory.

``--rebuild``: Adds new files to the given module that may not have existed when the module was
originally created.

``LIBRARY_NAME``: The name of the top-level library. If it does not exist, it will be created.

``MODULE_NAME``: The name of the module to create within the top-level library.

``LIBRARIES``: A list of libraries to reindex.

Output Files
------------

``.is_python_lib``: If this file is present, the library is considered to be a Python extension module.
It also contains options for updating the library's tree. These options are as follows.

* ``CONFIGURE_PREAMBLE``: If set to a truthy value, the preable file will be updated if the module is ever
  rebuilt.
* ``CONFIGURE_CLOSER``: If set to a truthy value, the closer file will be updated if the module is ever
  rebuilt.

``CMakeLists.txt``: This is the library's CMake file. It is automatically populated when the library is indexed
and should not be edited.

``ExportAll.cpp.in``: (Python library only) Template file for exporting symbols to be seen by Python.
It is automatically generated and populated. It shouldn't need to be generated, but may be if needed.

``modules.rst``: This is the top-level documentation for the library, containing a list of modules within
the library. It is automatically populated when the library is indexed, and should not be edited.

``preamble.txt``: This file contains CMake code to be executed before adding module subdirectories. It
is automatically generated for Python libraries, and may be overwritten by a rebuild. However, it may be changed if needed.
If you do, don't forget to set the ``CONFIGURE_PREAMBLE`` option in the ``.is_python_lib`` file appropriately.

``closer.txt``: This file contains CMake code to be executed after adding module subdirectories. It
is automatically generated for Python libraries, and may be overwritten by a rebuild. However, it may be changed if needed.
If you do, don't forget to set the ``CONFIGURE_CLOSER`` option in the ``.is_python_lib`` file appropriately.

``MODULE/docs/index.rst``: This is the top-level documentation for the module.

``MODULE/examples/CMakeLists.txt``: This is the CMake file that adds and builds example code.

``MODULE/include/LIBRARY/MODULE/.gitkeep``: This file makes sure that the directory doesn't disappear in GitHub,
even when the directory is empty.

``MODULE/include/LIBRARY/MODULE/InitModule.hpp``: This file contains the prototypes to initialize the module. If the
module does not need to be initialized, it can be safely deleted, along with ``MODULE/src/InitModule.cpp``,
and references to it can be removed from ``MODULE/CMakeLists.txt``.

``MODULE/include/LIBRARY/MODULE/ModuleVars.hpp``: This file contains a class that initializes module-global
variables in a way that makes them safe to access, and also ensures that they are properly deleted on program
exit. The class is called ``LIBRARY_MODULE_vars``, and it contains a ``get_singleton()`` static method for accessing
the underlying data. If your module does not define any global variables, this file can be safely deleted,
along with ``MODULE/src/ModuleVars.cpp``, and references to it can be removed from ``MODULE/CMakeLists.txt``.

``MODULE/src/.gitkeep``: This file makes sure that the directory doesn't disappear on GitHub when empty.

``MODULE/src/Export.cpp``: (Python library only) This file contains the code to export symbols from the module to Python.
It shouldn't be deleted, but the export function may be empty.

``MODULE/src/InitModule.cpp``: This file contains the module initialization code. The functions to modify are
``initialize_LIBRARY_MODULE`` and ``finalize_LIBRARY_MODULE``. If your code does not need to be initialized,
you may delete it, along with ``MODULE/include/LIBRARY/MODULE/InitModule.hpp``, and references to it can be removed from ``MODULE/CMakeLists.txt``.

``MODULE/src/ModuleVars.cpp``: This file contains the code to handle the setup and access of module-global variables.
It shouldn't need to be modified. If your module does not have global variables, it can be safely deleted,
along with ``MODULE/include/LIBRARY/MODULE/ModuleVars.hpp``, and references to it can be removed from ``MODULE/CMakeLists.txt``.

``MODULE/tests/performance/CMakeLists.txt``: The CMake file for adding performance tests.

``MODULE/tests/regressions/CMakeLists.txt``: The CMake file for adding regression tests.

``MODULE/tests/unit/CMakeLists.txt``: The CMake file for adding unit tests. Add your test names to the 
line that says ``set(MODULETests)``. For instance, ``set(TensorTests dot)`` will add a test called "dot"
that depends on a file called ``dot.cpp``.

``MODULE/tests/CMakeLists.txt``: Top-level CMake file for tests.

``MODULE/CMakeLists.txt``: Module CMake file. Add your headers and source files to this file. You
can use the provided ``set`` lines. You can also add your own dependencies.

``MODULE/README.rst``: A module-level readme file for documentation.

Input Files
^^^^^^^^^^^

For users who just want to create modules, you don't need anything further. The following is only
for those who want to modify the template files used to generate the outputs above. All files are
interpreted as Python f-strings. As such, any bracket needs to be escaped. The following variable names
are made available.

``module_name``: The name of the module.

``lib_name``: The name of the library.

``lib_symb``: Normally, it is the name of the library. For Python libraries, though, it is ``${EINSUMS_PYTHON_LIB_NAME}``,
which allows the Python version to be attached to the library outputs.

``docs_head``: The name of the library and module, followed by an appropriate amount of equals signs.

``readme_head``: The name of the module, followed by an appropriate amount of equals signs.

``export_source``: If this is a Python module, it will be ``Export.cpp``. Otherwise, it will be empty.

``export_depends``: If the current library is Einsums, it will be empty. Otherwise, it will contain ``Einsums``,
which will link the Einsums target to the library.

``python_footer``: If this is a Python module, it will contain code for adding Python dependencies.

``python_deps``: Contains Python dependencies for tests, if the module is a Python module. Otherwise, it will be empty.

``gpu_head`` and ``gpu_foot``: Puts an if-statement around the CMake code that only evaluates when GPU support
is enabled.

``modules``: Contains a list of modules in a library.

``module_docs``: Contains a list of module documentation paths in a library.

