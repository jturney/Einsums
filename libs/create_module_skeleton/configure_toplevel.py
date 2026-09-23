# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import os
import re
import sys

import cleanup_cmake


def ordered_modules(list_file, lib_name, present):
    """The modules to configure, in the order the existing list gives them.

    The list is in dependency order, kept by hand: a module that reads
    something another module sets up at configure time has to come after it
    (Hardware follows SIMD, for one). Sorting it would silently undo that, so
    the existing order is kept, modules whose directory is gone are dropped,
    and new ones are appended with a notice to place them.
    """
    existing = []
    if os.path.exists(list_file):
        with open(list_file, "r") as fp:
            m = re.search(
                r"set\(_" + re.escape(lib_name) + r"_modules\s+(.*?)\)", fp.read(), re.S
            )
        if m:
            existing = m.group(1).split()

    kept = [x for x in existing if x in present]
    new = [x for x in present if x not in existing]
    for x in new:
        print(
            f"note: {lib_name}/{x} was appended to the module list in "
            f"{list_file}; move it earlier if a module listed before it needs it"
        )
    return kept + new


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

    present = sorted(
        filter(
            lambda x: os.path.exists(
                os.path.join(output_base, lib_name, x, "CMakeLists.txt")
            ),
            os.listdir(os.path.join(output_base, lib_name)),
        )
    )
    modules = "\n  ".join(
        ordered_modules(
            os.path.join(output_base, lib_name, "CMakeLists.txt"), lib_name, present
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
    """List every module page of lib_name in the toctree of libs/overview.rst.

    overview.rst is the one page that links the module pages into the site,
    and a page left out of every toctree fails the warnings-as-errors docs
    build. The entries for lib_name are rewritten, sorted, from the modules
    that have a docs/index.rst; the rest of the page is left alone.
    """
    overview = os.path.join(output_base, "overview.rst")
    with open(overview, "r") as fp:
        text = fp.read()

    pages = sorted(
        x
        for x in os.listdir(os.path.join(output_base, lib_name))
        if os.path.exists(os.path.join(output_base, lib_name, x, "docs", "index.rst"))
    )
    entries = "".join(f"    /libs/{lib_name}/{x}/docs/index.rst\n" for x in pages)

    block = re.compile(r"(?:^    /libs/" + re.escape(lib_name) + r"/[^/\n]+/docs/index\.rst\n)+", re.M)
    if not block.search(text):
        sys.exit(f"error: no /libs/{lib_name}/*/docs/index.rst toctree entries found in {overview}")
    with open(overview, "w") as fp:
        fp.write(block.sub(lambda _: entries, text, count=1))
