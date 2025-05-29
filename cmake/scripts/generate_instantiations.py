#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
import sys
import re
from pathlib import Path
import argparse

# Regex to find macros
pattern = re.compile(r'^EINSUMS_TENSOR_EXPORT.*$', re.MULTILINE)


def extract_instantiations(header_path):
    with open(Path("include") / Path(header_path), 'r') as f:
        contents = f.read()
    return pattern.findall(contents)


def generate_cpp(output_dir, header, number, line):
    cpp_path = Path(output_dir) / f"{Path(header).stem}{number}.cpp"
    with open(cpp_path, 'w') as f:
        f.write(f"#include <{header}>\n\n")
        f.write("namespace einsums {\n")
        f.write(f"{line.replace("EXPORT", "DEFINE")}\n")
        f.write("}\n")
    return cpp_path


def main():
    parser = argparse.ArgumentParser(prog="generate_instantiations")
    parser.add_argument("--output_dir", nargs=1, required=True)
    parser.add_argument("--list_files", action="store_true")
    parser.add_argument("headers", nargs="*")

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir[0])
        output_dir.mkdir(parents=True, exist_ok=True)

    current_cpp_files = set()
    expected_names = set()

    # 1. Gather all expected instantiations
    for header in args.headers:
        if "TensorForward" in header:
            continue
        insts = extract_instantiations(header)
        for number, line in enumerate(insts):
            cpp_path = generate_cpp(args.output_dir[0], header, number, line)
            current_cpp_files.add(str(cpp_path))
            expected_names.add(cpp_path.name)

    # 2. Clean up stale .cpp files
    # This would be neat but adding the number to the end doesn't allow this to work properly.
    # for file in output_dir.glob("*.cpp"):
    #     if file.name not in expected_names:
    #         print(f"Removing stale instantiation: {file}")
    #         file.unlink()

    # 3. Write instantiations.cmake
    if args.list_files:
        print("\n".join(sorted(current_cpp_files)))


if __name__ == "__main__":
    main()
