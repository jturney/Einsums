# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import os
import sys

import cleanup_cmake


def configure_cmake(output_base, lib_name, **kwargs):
    base = os.path.dirname(__file__)

    format = ""

    with open(os.path.join(base, "cmake_template.txt.fstring"), "r") as fp:
        format = fp.read()

    # Check for preamble.
    preamble = ""
    if os.path.exists(os.path.join(output_base, lib_name, "preamble.txt")):
        with open(os.path.join(output_base, lib_name, "preamble.txt"), "r") as fp:
            preamble = fp.read()

    closer = ""
    if os.path.exists(os.path.join(output_base, lib_name, "closer.txt")):
        with open(os.path.join(output_base, lib_name, "closer.txt"), "r") as fp:
            closer = fp.read()

    modules = "\n  ".join(
        sorted(
            filter(
                lambda x: os.path.exists(
                    os.path.join(output_base, lib_name, x, "CMakeLists.txt")
                ),
                os.listdir(os.path.join(output_base, lib_name)),
            )
        )
    )

    modules = modules.rstrip()

    with open(os.path.join(output_base, lib_name, "CMakeLists.txt"), "w+") as fp:
        fp.write(
            cleanup_cmake.cleanup_cmake(
                format.format(
                    modules=modules,
                    preamble=preamble,
                    closer=closer,
                    lib_name=lib_name,
                    **kwargs,
                )
            )
        )


def configure_module_docs(output_base, lib_name, **kwargs):
    base = os.path.dirname(__file__)

    format_str = ""

    with open(os.path.join(base, "modules.rst.fstring"), "r") as fp:
        format_str = fp.read()

    module_docs = "\n    ".join(
        sorted(
            map(
                lambda x: os.path.join("/libs", lib_name, x, "docs", "index.rst"),
                filter(
                    lambda x: os.path.exists(
                        os.path.join(output_base, lib_name, x, "CMakeLists.txt")
                    ),
                    os.listdir(os.path.join(output_base, lib_name)),
                ),
            )
        )
    )

    with open(os.path.join(output_base, lib_name, "modules.rst"), "w+") as fp:
        fp.write(
            format_str.format(
                module_docs=module_docs,
                lib_name=lib_name,
                section_head="".join("=" for i in f"{lib_name} modules"),
                **kwargs,
            )
        )
    
    if not os.path.exists(os.path.join(output_base, lib_name, "todo.rst")) :
        with open(os.path.join(base, "todo.rst.fstring"), "r") as inp, open(os.path.join(output_base, lib_name, "todo.rst"), "w+") as outp :
            outp.write(inp.read().format(lib_name = lib_name, section_head="".join("=" for i in f"{lib_name} Library TODO"), **kwargs))
    
    if not os.path.exists(os.path.join(output_base, lib_name, "index.rst")) :
        with open(os.path.join(base, "index.rst.fstring"), "r") as inp, open(os.path.join(output_base, lib_name, "index.rst"), "w+") as outp :
            outp.write(inp.read().format(lib_name = lib_name, section_head="".join("=" for i in lib_name), **kwargs))