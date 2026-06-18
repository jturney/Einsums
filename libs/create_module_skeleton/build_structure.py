# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import os
import sys


def parse_template(filename, output_file, **kwargs):
    format_str = ""

    with open(filename, "r") as fp:
        format_str = fp.read()

    with open(output_file, "w+") as fp:
        fp.write(format_str.format(**kwargs))


def build_layer(input_dir, output_dir, **kwargs):
    for item in os.listdir(input_dir):
        item_out = kwargs.get(item, item)

        if os.path.splitext(item_out)[1] == ".fstring" :
            item_out = os.path.splitext(item_out)[0]

        if os.path.isdir(os.path.join(input_dir, item)):
            if not os.path.isdir(os.path.join(output_dir, item_out)):
                os.mkdir(os.path.join(output_dir, item_out))
            build_layer(
                os.path.join(input_dir, item), os.path.join(output_dir, item_out), **kwargs
            )
        elif not os.path.exists(os.path.join(output_dir, item_out)):
            format_str = ""
            try:
                with open(os.path.join(input_dir, item), "r") as fp:
                    format_str = fp.read()
                with open(os.path.join(output_dir, item_out), "w+") as fp:
                    fp.write(format_str.format(**kwargs))
            except KeyError as e:
                print(format_str)
                raise RuntimeError(f"File being parsed was {input_dir}/{item}.") from e
            except ValueError as e:
                print(format_str)
                print(e)
                raise RuntimeError(f"File being parsed was {input_dir}/{item}.") from e


def build_structure(output_base, lib_name, module_name, python=False, **kwargs):
    base = os.path.dirname(__file__)

    if not os.path.exists(os.path.join(output_base, lib_name)):
        os.mkdir(os.path.join(output_base, lib_name))

    if not os.path.exists(os.path.join(output_base, lib_name, module_name)):
        os.mkdir(os.path.join(output_base, lib_name, module_name))

    build_layer(
        os.path.join(base, "template_top"),
        os.path.join(output_base, lib_name, module_name),
        module_name=module_name,
        lib_name=lib_name,
        docs_head="".join("=" for i in lib_name + ' ' + module_name),
        readme_head="".join("=" for i in module_name),
        export_depends="" if lib_name == "Einsums" else "Einsums",
        # ``--python`` adds the PYBIND keyword to einsums_add_module so apiary
        # generates the bindings. It no longer makes the module a separate
        # Python extension library.
        pybind_flag="PYBIND\n  " if python else "",
        python=python,
        python_deps="",
        gpu_head="if(EINSUMS_WITH_GPU_SUPPORT)" if kwargs["gpu"] else "",
        gpu_foot=f"""
        if(EINSUMS_WITH_CUDA)
          foreach(f IN LISTS {module_name}Sources)
            set_source_file_properties("src/${{f}}" PROPERTIES LANGUAGE CUDA)
          endforeach()
        endif()
        endif()
        """ if kwargs["gpu"] else "",
        **kwargs
    )
