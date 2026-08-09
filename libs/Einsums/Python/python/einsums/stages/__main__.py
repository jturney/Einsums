#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""``python -m einsums.stages`` - the promote and extract tools.

::

    python -m einsums.stages promote mymethod/stages.py --out cpp/
    python -m einsums.stages extract mymethod/solver.py::Solver.build_t2

``promote`` reads the Python stage signatures and contract dataclasses by
runtime introspection - no libclang, no parsing of C++ - and writes the C++
side of every stage in the module that has stated a contract.

It operates on the module, never on one stage. A contract's binding strategy
depends on which direction it crosses the boundary, and that is only knowable
with the whole stage set in hand. A stage name given with ``--stage`` filters
which port skeletons get written; the generated module always covers
everything.

``extract`` is upstream of ``promote``: it turns a method into the stage split
``promote`` operates on. It reports every ``self`` field and helper the method
touches, then refuses to scaffold anything until a cut spec has dispositioned
each one - because the hard half of extracting a method is choosing where to
cut, and a tool that chooses silently emits the 32-field contract this project
has already rejected once.
"""

import argparse
import pathlib
import sys

from ._emit import Emitter, find_formatters
from ._extract import (
    ExtractError,
    analyze,
    load_spec,
    narrowing_summary,
    render_report,
    render_scaffold,
    render_template,
)
from ._promote import PromoteError, build_model, classify_stages, load_stages_module

__all__ = ["main"]


def _promote_parser(sub):
    p = sub.add_parser(
        "promote",
        help="generate the C++ side of a stages module",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "module",
        help="path to the stages module (mymethod/stages.py) or its dotted name",
    )
    p.add_argument(
        "--out",
        default="cpp",
        help="directory to write into (default: cpp)",
    )
    p.add_argument(
        "--namespace",
        help="C++ namespace (default: the top-level Python package name)",
    )
    p.add_argument(
        "--module-name",
        help="name of the compiled pybind module (default: <namespace>_stages)",
    )
    p.add_argument(
        "--stage",
        action="append",
        metavar="NAME",
        help="only scaffold this stage's port skeleton and differential test; "
        "repeatable. The generated module still covers every stage, so the ones "
        "left out will not link until they have a source.",
    )
    p.add_argument(
        "--license-header",
        type=pathlib.Path,
        help="file whose text is emitted as a comment block at the top of every "
        "generated file. Pass this when the target tree has a license hook, so "
        "the hook does not edit generated files out from under their hashes.",
    )
    p.add_argument(
        "--clang-format",
        metavar="BINARY",
        help="clang-format to format the generated C++ with (default: the one on "
        "PATH). Formatting here is what keeps a formatting hook from editing a "
        "generated file and making the next run refuse to touch it.",
    )
    p.add_argument(
        "--cmake-format",
        metavar="BINARY",
        help="cmake-format to format the generated CMakeLists.txt with "
        "(default: the one on PATH)",
    )
    p.add_argument(
        "--no-format",
        action="store_true",
        help="do not format the output. Expect a formatting hook to rewrite it.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite generated files that have been edited by hand, discarding "
        "the edits. Never applies to the port skeleton or the build file, which "
        "promote writes once and never rewrites.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without writing anything",
    )
    p.set_defaults(run=_run_promote)
    return p


def _extract_parser(sub):
    p = sub.add_parser(
        "extract",
        help="analyze a method for promotion and scaffold its extraction",
        description=(
            "Report every `self` field and helper the method touches, write a cut-spec "
            "template, and - once every entry has a disposition - scaffold the extraction: "
            "contract, stage signature, plan and finish skeletons. A static AST pass; no "
            "build and no imports are needed."
        ),
    )
    p.add_argument(
        "target",
        help="the method, as path/to/file.py::Class.method",
    )
    p.add_argument(
        "--also",
        action="append",
        default=[],
        metavar="FILE",
        help="extra source file providing helper definitions (a base class in "
        "another file); repeatable",
    )
    p.add_argument(
        "--spec",
        type=pathlib.Path,
        help="the cut spec (default: <method>.cut.toml in the current directory). "
        "Missing: a TODO template is written there. Present: it is validated and "
        "the scaffold is emitted.",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="directory the scaffold is written into (default: .)",
    )
    p.set_defaults(run=_run_extract)
    return p


def _run_extract(args) -> int:
    analysis = analyze(args.target, also=args.also)
    print(render_report(analysis))

    spec_path = args.spec or pathlib.Path(f"{analysis.method}.cut.toml")
    if not spec_path.exists():
        spec_path.write_text(render_template(analysis))
        print(
            f"\nwrote {spec_path}: fill in every disposition, then rerun the same "
            f"command to scaffold the extraction."
        )
        return 0

    spec = load_spec(spec_path, analysis)
    scaffold_path = args.out / f"extracted_{spec.stage_name}.py"
    if scaffold_path.exists():
        print(
            f"\nextract: {scaffold_path} already exists and is yours; it is never "
            f"rewritten. Move it aside to scaffold again.",
            file=sys.stderr,
        )
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    scaffold_path.write_text(render_scaffold(analysis, spec))
    print(f"\n{narrowing_summary(analysis, spec)}")
    print(f"wrote {scaffold_path}")
    return 0


def _run_promote(args) -> int:
    module = load_stages_module(args.module)
    promotable, skipped = classify_stages(module)
    model = build_model(module, namespace=args.namespace, module_name=args.module_name)

    if args.stage:
        known = {st.name for st in promotable}
        unknown = sorted(set(args.stage) - known)
        if unknown:
            raise PromoteError(
                f"--stage names {', '.join(unknown)}, which {module.__name__} does not declare "
                f"with a contract. Contracted stages: {', '.join(sorted(known)) or 'none'}."
            )

    license_text = args.license_header.read_text() if args.license_header else None
    formatters = find_formatters(
        args.clang_format, args.cmake_format, enabled=not args.no_format
    )

    results = Emitter(
        model,
        args.out,
        license_text=license_text,
        formatters=formatters,
        force=args.force,
        dry_run=args.dry_run,
        skeletons=set(args.stage) if args.stage else None,
    ).run()

    print(f"{model.source_module} -> {args.out}" + ("  (dry run)" if args.dry_run else ""))
    for st in model.stages:
        print(f"  promote {st.name} -> {model.namespace}::{st.name}")
    for st, reason in skipped:
        print(f"  skip    {st.name} ({reason})")
    print()
    for r in results:
        print(f"  {r.action:9} {r.path}" + (f"\n{' ' * 14}{r.detail}" if r.detail else ""))

    refused = [r for r in results if r.refused]
    if refused:
        print(
            f"\n{len(refused)} file(s) refused: they no longer match what promote wrote, so "
            f"someone edited them. Nothing was written for those.",
            file=sys.stderr,
        )
        return 1
    if not args.no_format and formatters.missing:
        print(
            f"\nno {' or '.join(formatters.missing)} found, so that part of the output is "
            f"unformatted. A formatting hook in the target tree will rewrite it, and the next "
            f"promote run will then refuse it as hand-edited. Name the binary explicitly, or "
            f"pass --no-format to accept that.",
            file=sys.stderr,
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m einsums.stages")
    sub = parser.add_subparsers(dest="command", required=True)
    _promote_parser(sub)
    _extract_parser(sub)
    args = parser.parse_args(argv)

    try:
        return args.run(args)
    except (PromoteError, ExtractError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
