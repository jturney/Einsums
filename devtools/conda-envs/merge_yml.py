# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

from ruamel.yaml import YAML
import os
import platform
import argparse

packages_to_filter = [
    'cpptrace' if platform.system() == 'Windows' else None,
]

dir_path = os.path.dirname(os.path.realpath(__file__))


def merge_yaml_files(output_file, *input_files):
    yaml = YAML()
    merged = {"name": None, "channels": [], "dependencies": []}

    for file in input_files:
        with open(os.path.join(dir_path, file), 'r') as f:
            data = yaml.load(f)
            if "name" in data and not merged["name"]:
                merged["name"] = data["name"]
            merged["channels"].extend(data.get("channels", []))
            merged["dependencies"].extend(data.get("dependencies", []))

    # Remove duplicates
    merged["channels"] = list(dict.fromkeys(merged["channels"]))
    merged["dependencies"] = list(dict.fromkeys(merged["dependencies"]))

    for package in packages_to_filter:
        if package:
            merged["dependencies"].remove(package)

    print(merged)
    print("Writing to {}".format(output_file))

    with open(output_file, 'w') as f:
        yaml.dump(merged, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="merge_yml.py",
        description="Creates a conda environment yml file to develop Einsums.",
        usage="""

%(prog)s [OPTIONS] [COMPILER=default] [BLAS=openblas]

Options:
  --docs    Include documentation build dependencies (Sphinx, Doxygen, etc.)""",
    )

    parser.add_argument(
        "--output", help="Name of the output yml file", default="conda.yml"
    )

    parser.add_argument(
        "--docs", help="Install packages needed to build documentation",
        action="store_true"
    )

    parser.add_argument("compiler", choices=["default", "intel", "windows"], default="default", nargs="?",
                        help="The compiler to use (choices: default, intel, Windows). Default implies default for your platform.")

    # Optional positional argument with choices
    parser.add_argument(
        "blas",
        nargs="?",
        choices=["accelerate", "mkl", "openblas"],
        default="openblas",
        help="The BLAS library to use (optional; choices: accelerate, mkl, openblas)",
    )

    args = parser.parse_args()

    if args.compiler == "intel":
        # The only valid option for BLAS is MKL
        print(f"Defaulting to MKL for BLAS.")
        args.blas = "mkl"

    if args.compiler == 'windows':
        # The only valid option for BLAS is MKL
        print(f"Defaulting to MKL for BLAS.")
        args.blas = "mkl"

    if platform.system() == "Darwin":
        # Default to Accelerate for BLAS
        print(f"Defaulting to Accelerate for BLAS on macOS.")
        args.blas = "accelerate"

    snippets = ["snippets/common.yml", f"snippets/compiler/{args.compiler}.yml",
                f"snippets/blas/{args.blas}.yml", f"snippets/os/{platform.system()}.yml"]
    if args.docs:
        snippets.append("snippets/docs.yml")
    merge_yaml_files(args.output, *snippets)
