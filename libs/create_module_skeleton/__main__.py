# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import argparse
import os
import sys

import build_structure
import configure_toplevel


# The package lives at <repo>/libs/create_module_skeleton, so its parent
# directory is the libs/ folder that holds the library trees. Deriving the
# output base from the script location lets the tool run from any working
# directory and still write modules into the right place.
LIBS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def build_new(libs_dir, library, module, args):
    if args.python_name is None:
        args.python_name = library

    # A module exposes Python bindings through the PYBIND keyword on
    # einsums_add_module, not by becoming a separate Python extension library,
    # so the library symbol is always the real library name.
    build_structure.build_structure(
        libs_dir, library, module, lib_symb=library, **vars(args)
    )

    configure_toplevel.configure_cmake(
        libs_dir, library, lib_symb=library, **vars(args)
    )
    configure_toplevel.configure_module_docs(
        libs_dir, library, lib_symb=library, **vars(args)
    )


def reindex(libs_dir, libraries=None):
    if libraries is None or len(libraries) == 0:
        libraries = filter(
            lambda x: os.path.isfile(os.path.join(libs_dir, x, "CMakeLists.txt")),
            os.listdir(libs_dir),
        )

    for lib in libraries:
        configure_toplevel.configure_cmake(libs_dir, lib, lib_symb=lib)
        configure_toplevel.configure_module_docs(libs_dir, lib, lib_symb=lib)


def main():
    parser = argparse.ArgumentParser(
        prog="create_module_skeleton",
        description="Creates a module skeleton for developing in Einsums. All of the template files are treated as Python f strings, so be careful with braces.",
        usage="""
        
%(prog)s --reindex [LIBRARIES, ...]
%(prog)s [OPTIONS] LIBRARY_NAME MODULE_NAME""",
    )

    parser.add_argument(
        "--python",
        help="Set this flag if the module is a Python extension module.",
        action="store_true",
    )
    parser.add_argument(
        "--gpu",
        help="Set this flag if the module is a GPU module.",
        action="store_true",
    )
    parser.add_argument(
        "--python-name", help="The name of the top-level Python module."
    )
    parser.add_argument(
        "--reindex",
        help="Reindex the libraries. The libraries may be specified afterwards. This is incompatible with other options.",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild",
        help="Adds new files that were added to the template but do not exist in the output structure. It also re-indexes.",
        action="store_true",
    )
    parser.add_argument(
        "--libs-dir",
        default=LIBS_DIR,
        help="Directory that holds the library folders. Defaults to the libs/ directory containing this script, so the tool can be run from any working directory.",
    )

    known_args, unknown_args = parser.parse_known_intermixed_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if known_args.reindex:
        reindex(known_args.libs_dir, unknown_args)
    else:
        if len(unknown_args) != 2:
            parser.error(
                "expected exactly two positional arguments: LIBRARY_NAME MODULE_NAME"
            )
        build_new(known_args.libs_dir, unknown_args[0], unknown_args[1], known_args)


if __name__ == "__main__":
    main()
