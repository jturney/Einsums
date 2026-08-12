#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Tests for ``python -m einsums.stages promote``: the model, the emitters, the guards.

The generator is tested against a stages module written into a temp directory
rather than against a golden file. A committed golden can be regenerated into
agreeing with a bug; a fixture whose Python says what it means cannot.

The one golden that IS worth having lives at the bottom: the DLPNO stage module
is checked in, builds out of tree, and is bit-identical against its Python
backend, so a generator that reproduces it has reproduced something known to
work rather than something that merely compiles.
"""

import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

import einsums  # noqa: F401  (loads the runtime)
from einsums.stages import ContractError, _emit, _promote

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
DLPNO = REPO_ROOT / "examples" / "dlpno"


# ----------------------------------------------------------------------
# A stages module, written out and imported, so nothing here depends on
# whatever else the test session has already registered.
# ----------------------------------------------------------------------
FIXTURE = '''
"""A fixture method with one promotable stage."""

from dataclasses import dataclass

from einsums.stages import TensorD, cmp, contract, stage


@contract(parallel=["row", "col"])
@dataclass(frozen=True)
class Sparsity:
    """Which entries survive screening.

    Structure of arrays, indexed by entry ordinal.
    """

    #: Row index of each surviving entry.
    row: list[int] = cmp.exact()
    col: list[int] = cmp.exact()
    #: Screening threshold the plan was built at.
    cutoff: float = cmp.exact()


@contract
@dataclass(frozen=True)
class Amplitudes:
    """What the solver produces."""

    #: Per pair, the amplitude block.
    t2: list[TensorD] = cmp.close()
    n_pair: list[int] = cmp.exact()


@stage
def build_amplitudes(f: TensorD, plan: Sparsity, shift: float) -> Amplitudes:
    """Build the doubles amplitudes.

    Longer prose that should reach the header.

    Args:
        f: The Fock matrix.
        plan: Which entries survive.
        shift: Denominator level shift.
    """
    return Amplitudes(t2=[], n_pair=[])


@stage(contract=False)
def not_ready(solver) -> None:
    """On the promotion path, but no contract yet."""


@stage(promotable=False)
def write_checkpoint(solver) -> None:
    """Never going to C++."""
'''


@pytest.fixture(scope="module")
def fixture_module(tmp_path_factory):
    root = tmp_path_factory.mktemp("promote_fixture")
    pkg = root / "mymethod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "stages.py").write_text(FIXTURE)
    sys.path.insert(0, str(root))
    try:
        yield _promote.load_stages_module(str(pkg / "stages.py"))
    finally:
        sys.path.remove(str(root))


@pytest.fixture(scope="module")
def model(fixture_module):
    return _promote.build_model(fixture_module)


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------
def test_only_contracted_stages_are_promoted(fixture_module, model):
    assert [st.name for st in model.stages] == ["build_amplitudes"]

    _, skipped = _promote.classify_stages(fixture_module)
    assert dict((st.name, reason) for st, reason in skipped) == {
        "not_ready": "no contract",
        "write_checkpoint": "python-only",
    }


def test_parameters_map_onto_the_closed_type_list(model):
    stage = model.stages[0]
    assert [p.decl for p in stage.params] == [
        "einsums::RuntimeTensor<double> const &f",
        "Sparsity const &plan",
        "double shift",
    ]
    assert stage.return_cpp == "Amplitudes"
    assert stage.unit == "BuildAmplitudes"
    assert stage.entry_point == "stage_build_amplitudes"


def test_contract_direction_decides_the_binding_strategy(model):
    by_name = {c.name: c for c in model.contracts}
    # An argument-position contract gets a caster and cannot be a py::class_:
    # a caster converts INTO C++ and cannot hand an object back.
    assert by_name["Sparsity"].as_argument and not by_name["Sparsity"].as_return
    # A returned one is the mirror image.
    assert by_name["Amplitudes"].as_return and not by_name["Amplitudes"].as_argument


def test_field_prose_comes_from_the_python_source(model):
    sparsity = next(c for c in model.contracts if c.name == "Sparsity")
    fields = {f.name: f for f in sparsity.fields}
    assert fields["row"].doc == ["Row index of each surviving entry."]
    assert fields["col"].doc == []  # no `#:` comment, so no invented one
    assert fields["cutoff"].cpp == "double"
    assert "Structure of arrays" in " ".join(sparsity.doc)


def test_parameter_prose_comes_from_the_args_section(model):
    stage = model.stages[0]
    docs = {p.name: p.doc for p in stage.params}
    assert docs["f"] == ["The Fock matrix."]
    assert docs["shift"] == ["Denominator level shift."]
    assert stage.summary == "Build the doubles amplitudes."
    # The Args: section is consumed into @param and must not be repeated in the
    # body, where it would render as prose with no command in front of it.
    assert "Args:" not in "\n".join(stage.doc)


def test_only_the_args_section_is_consumed_from_a_docstring():
    summary, body, params = _promote.split_docstring(
        """Summary line.

        Opening prose.

        Args:
            a: First.
                Continued.
            b: Second.

        Returns:
            Something worth keeping.

        Note:
            And this.
        """
    )
    assert summary == "Summary line."
    assert params == {"a": ["First.", "Continued."], "b": ["Second."]}
    # Returns and Note have no Doxygen command that would not be a guess, so
    # they stay in the body verbatim rather than being dropped.
    assert "Returns:" in body and "Something worth keeping." in "\n".join(body)
    assert "Note:" in body and "And this." in "\n".join(body)
    assert "Args:" not in body


def test_a_module_with_no_contracted_stage_says_which_ones_it_skipped(tmp_path):
    pkg = tmp_path / "bare"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "stages.py").write_text(
        textwrap.dedent(
            """
            from einsums.stages import stage

            @stage(promotable=False)
            def promote_fixture_python_only(solver) -> None:
                pass
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = _promote.load_stages_module(str(pkg / "stages.py"))
        with pytest.raises(_promote.PromoteError, match="promote_fixture_python_only"):
            _promote.build_model(module)
    finally:
        sys.path.remove(str(tmp_path))


# ----------------------------------------------------------------------
# The emitters
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def generated(model, tmp_path_factory):
    out = tmp_path_factory.mktemp("promote_out")
    _emit.generate(model, out, formatters=_emit.find_formatters())
    return out


#: What each generated file must actually SAY. Existence is not the assertion
#: worth making: a cmake-format invoked wrongly returned an empty string, and
#: the resulting CMakeLists.txt held its banner and not one command. It existed,
#: it matched the other generated copy of itself byte for byte, and it produced
#: a build with no target in it.
EXPECTED_CONTENT = {
    "include/mymethod/Contracts.hpp": ["struct Sparsity {", "struct Amplitudes {"],
    "include/mymethod/BuildAmplitudes.hpp": ["build_amplitudes(", "Sparsity const &plan"],
    "include/mymethod/Casters.hpp": ["type_caster<mymethod::Sparsity>"],
    "bindings.cpp": ["PYBIND11_MODULE(mymethod_stages, m)", "EINSUMS_STAGE_MODULE"],
    "CMakeLists.txt": ["find_package(Einsums", "einsums_add_stage_module"],
    "src/BuildAmplitudes.cpp": ["namespace mymethod {", "TODO: port"],
    "tests/test_diff_build_amplitudes.py": ["def make_inputs()"],
}


def test_every_expected_file_is_written_and_says_something(generated):
    for rel, needles in EXPECTED_CONTENT.items():
        path = generated / rel
        assert path.is_file(), rel
        text = path.read_text()
        for needle in needles:
            assert needle in text, f"{rel}: missing {needle!r}"


def test_the_caster_is_emitted_for_arguments_and_the_class_for_returns(generated):
    casters = (generated / "include/mymethod/Casters.hpp").read_text()
    bindings = (generated / "bindings.cpp").read_text()

    assert "type_caster<mymethod::Sparsity>" in casters
    assert "type_caster<mymethod::Amplitudes>" not in casters
    assert 'py::class_<mymethod::Amplitudes>(m, "Amplitudes")' in bindings
    assert "py::class_<mymethod::Sparsity>" not in bindings


def test_the_caster_checks_the_declared_parallel_group(generated):
    casters = (generated / "include/mymethod/Casters.hpp").read_text()
    assert "must be parallel" in casters
    assert "value.col.size()" in casters
    # cutoff is not a list and is not in the group, so it must not be measured.
    assert "value.cutoff.size()" not in casters


def test_the_handshake_and_the_stage_prefix_are_in_the_bindings(generated):
    bindings = (generated / "bindings.cpp").read_text()
    assert 'EINSUMS_STAGE_MODULE(m, "mymethod_stages");' in bindings
    assert 'm.def("stage_build_amplitudes", &mymethod::build_amplitudes' in bindings
    assert 'py::arg("shift")' in bindings


def test_the_skeleton_declares_what_the_header_declares(generated):
    header = (generated / "include/mymethod/BuildAmplitudes.hpp").read_text()
    skeleton = (generated / "src/BuildAmplitudes.cpp").read_text()
    for param in ("einsums::RuntimeTensor<double> const &f", "Sparsity const &plan", "double shift"):
        assert param in header
        assert param in skeleton
    assert "// TODO: port." in skeleton


# ----------------------------------------------------------------------
# The provenance guard
# ----------------------------------------------------------------------
def test_a_regenerated_file_is_rewritten_and_a_scaffold_is_not(model, tmp_path):
    first = {w.path.name: w.action for w in _emit.generate(model, tmp_path)}
    assert first["Casters.hpp"] == "created"
    assert first["CMakeLists.txt"] == "created"

    second = {w.path.name: w.action for w in _emit.generate(model, tmp_path)}
    assert second["Casters.hpp"] == "unchanged"
    assert second["CMakeLists.txt"] == "kept"


def test_promote_refuses_a_generated_file_someone_edited(model, tmp_path):
    _emit.generate(model, tmp_path)
    casters = tmp_path / "include/mymethod/Casters.hpp"
    edited = casters.read_text() + "\n// a hand-written helper nobody wants to lose\n"
    casters.write_text(edited)

    result = next(w for w in _emit.generate(model, tmp_path) if w.path == casters)
    assert result.refused
    assert casters.read_text() == edited, "a refused file must not be written"

    # And the known-false case: the same guard must NOT fire on a file promote
    # itself wrote, or the refusal is just "this file exists" wearing a hash.
    forced = next(w for w in _emit.generate(model, tmp_path, force=True) if w.path == casters)
    assert forced.action == "updated"
    assert casters.read_text() != edited


def test_the_port_skeleton_is_never_overwritten(model, tmp_path):
    _emit.generate(model, tmp_path)
    port = tmp_path / "src/BuildAmplitudes.cpp"
    port.write_text("// the whole port lives here\n")

    for kwargs in ({}, {"force": True}):
        result = next(w for w in _emit.generate(model, tmp_path, **kwargs) if w.path == port)
        assert result.action == "kept"
        assert port.read_text() == "// the whole port lives here\n"


def test_the_hash_covers_the_whole_file_including_the_license(model, tmp_path):
    _emit.generate(model, tmp_path, license_text="Copyright (c) Someone Else.")
    bindings = tmp_path / "bindings.cpp"
    text = bindings.read_text()
    assert _emit.is_sealed(text)
    assert not _emit.is_sealed(text.replace("Someone Else", "Someone Different"))


def test_dry_run_writes_nothing(model, tmp_path):
    results = _emit.generate(model, tmp_path, dry_run=True)
    assert results and all(w.action == "created" for w in results)
    assert not any(tmp_path.iterdir())


# ----------------------------------------------------------------------
# Contracts nested inside contracts: direction must reach every level
# ----------------------------------------------------------------------
NESTED_FIXTURE = '''
"""A fixture whose contracts nest other contracts, in both directions.

Nesting is legal - `check_type` accepts a contract field of contract type - so
the generator has to classify the nested one by the direction of its CONTAINER:
it crosses the boundary with it. A nested contract that is left out of the
classification still gets a struct (the dependency walk finds it), which is
exactly what makes the hole invisible until the first call fails.
"""

from dataclasses import dataclass

from einsums.stages import TensorD, cmp, contract, stage


@contract
@dataclass(frozen=True)
class Leaf:
    """Two levels down on the argument side."""

    #: Per entry, its weight.
    w: list[float] = cmp.exact()


@contract
@dataclass(frozen=True)
class Mid:
    """One level down, holding a Leaf."""

    leaf: Leaf
    #: Scale applied on top of the leaf weights.
    scale: float = cmp.exact()


@contract
@dataclass(frozen=True)
class Plan:
    """The argument-position container."""

    mid: Mid
    #: Per block, one leaf; a contract reached through list[...] inside a field.
    blocks: list[Leaf] = cmp.exact()


@contract
@dataclass(frozen=True)
class Summary:
    """Nested on the way OUT.

    Only capture-legal output types: a computed float is refused on the return
    path, so the residual norms come back as rank-0 tensors.
    """

    #: Per pair, the residual norm as a rank-0 tensor.
    norms: list[TensorD] = cmp.close()
    #: Per pair, how many iterations it took.
    iters: list[int] = cmp.exact()


@contract
@dataclass(frozen=True)
class Result:
    """The return-position container."""

    summary: Summary
    n_pairs: int = cmp.exact()


@stage
def run_plan(t: TensorD, plan: Plan) -> Result:
    """Run the nested plan.

    Args:
        t: Input tensor.
        plan: The plan, two contracts deep.
    """
    return Result(summary=Summary(norms=[]), n_pairs=0)
'''


@pytest.fixture(scope="module")
def nested_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("promote_nested")
    pkg = root / "nestedmethod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "stages.py").write_text(NESTED_FIXTURE)
    sys.path.insert(0, str(root))
    try:
        yield _promote.build_model(_promote.load_stages_module(str(pkg / "stages.py")))
    finally:
        sys.path.remove(str(root))


@pytest.fixture(scope="module")
def nested_generated(nested_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("promote_nested_out")
    _emit.generate(nested_model, out, formatters=_emit.find_formatters())
    return out


def test_a_nested_contract_inherits_its_containers_direction(nested_model):
    by_name = {c.name: c for c in nested_model.contracts}
    assert set(by_name) == {"Leaf", "Mid", "Plan", "Summary", "Result"}
    # Leaf and Mid cross only inside Plan, so they need a caster exactly as
    # much as Plan does: the Plan caster's load() does .cast<Mid>() and the Mid
    # caster's load() does .cast<Leaf>().
    for name in ("Leaf", "Mid", "Plan"):
        assert by_name[name].as_argument, name
        assert not by_name[name].as_return, name
    # And the mirror image on the way out.
    for name in ("Summary", "Result"):
        assert by_name[name].as_return, name
        assert not by_name[name].as_argument, name


def test_translate_names_every_contract_reachable_through_the_fields(nested_model):
    plan = next(c for c in nested_model.contracts if c.name == "Plan")
    translated = _promote.translate(plan.py_type, where="test")
    # Outermost first, as the Translated.contracts docstring promises, and Leaf
    # named once despite being reachable twice (through Mid and through blocks).
    assert [c.__name__ for c in translated.contracts] == ["Plan", "Mid", "Leaf"]


def test_nested_contracts_are_declared_before_their_containers(nested_generated):
    header = (nested_generated / "include/nestedmethod/Contracts.hpp").read_text()
    for name in ("Leaf", "Mid", "Plan", "Summary", "Result"):
        assert f"struct {name} {{" in header, name
    assert header.index("struct Leaf {") < header.index("struct Mid {") < header.index("struct Plan {")
    assert header.index("struct Summary {") < header.index("struct Result {")


def test_every_argument_position_contract_gets_a_caster(nested_generated):
    casters = (nested_generated / "include/nestedmethod/Casters.hpp").read_text()
    for name in ("Leaf", "Mid", "Plan"):
        assert f"type_caster<nestedmethod::{name}>" in casters, name
    # Return-side contracts cross as py::class_ and must not also get a caster:
    # a caster specialization would shadow the class_ registration on the way
    # out, and the generated one cannot cast in that direction.
    for name in ("Summary", "Result"):
        assert f"type_caster<nestedmethod::{name}>" not in casters, name


def test_the_nested_caster_precedes_the_one_whose_load_uses_it(nested_generated):
    # Plan's load() converts Mid, and Mid's converts Leaf. A full
    # specialization must be declared before anything instantiates the primary
    # template for the same type, so the casters have to come out in the same
    # dependency order the structs do - innermost first.
    casters = (nested_generated / "include/nestedmethod/Casters.hpp").read_text()
    assert (
        casters.index("type_caster<nestedmethod::Leaf>")
        < casters.index("type_caster<nestedmethod::Mid>")
        < casters.index("type_caster<nestedmethod::Plan>")
    )


def test_every_return_position_contract_is_bound_as_a_class(nested_generated):
    bindings = (nested_generated / "bindings.cpp").read_text()
    assert 'py::class_<nestedmethod::Summary>(m, "Summary")' in bindings
    assert 'py::class_<nestedmethod::Result>(m, "Result")' in bindings
    for name in ("Leaf", "Mid", "Plan"):
        assert f"py::class_<nestedmethod::{name}>" not in bindings, name


# ----------------------------------------------------------------------
# A contract crossing in BOTH directions is refused, loudly and early
# ----------------------------------------------------------------------
BOTH_WAYS_FIXTURE = '''
"""A fixture whose contract crosses in both directions, which cannot be bound."""

from dataclasses import dataclass

from einsums.stages import cmp, contract, stage


@contract
@dataclass(frozen=True)
class State:
    """Carried in and handed back."""

    #: Per pair, an iteration count.
    n: list[int] = cmp.exact()


@contract
@dataclass(frozen=True)
class Wrapped:
    """Returned, with the argument contract nested inside."""

    state: State
    converged: bool = cmp.exact()


@stage
def advance(state: State) -> Wrapped:
    """Advance the state by one sweep."""
    return Wrapped(state=state, converged=True)
'''


def test_a_contract_crossing_both_ways_is_refused(tmp_path):
    """One type cannot have both a caster and a py::class_.

    The caster cannot cast back, and generating both does not compile: the
    caster specialization shadows the class_ on the return path. Refusing at
    generation time turns a baffling pybind compile error into a sentence
    naming the contract - and the nested use on the return side must count,
    which is exactly the classification hole the nested-contract fix closed.
    """
    pkg = tmp_path / "bothways"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "stages.py").write_text(BOTH_WAYS_FIXTURE)
    sys.path.insert(0, str(tmp_path))
    try:
        module = _promote.load_stages_module(str(pkg / "stages.py"))
        with pytest.raises(_promote.PromoteError, match="State.*argument and as a return"):
            _promote.build_model(module)
    finally:
        sys.path.remove(str(tmp_path))


# ----------------------------------------------------------------------
# Parameter defaults and prose that needs escaping
# ----------------------------------------------------------------------
EDGE_FIXTURE = r'''
"""A fixture for emission edge cases: parameter defaults, prose full of quotes."""

from typing import Annotated

from einsums.stages import TensorD, index, stage


@stage
def screen_pairs(
    t: TensorD,
    label: str = 'core "frozen" \\ tag',
    cutoff: float = 1e-6,
    tighten: bool = False,
    n_keep: int = -1,
    start: Annotated[int, index] = 0,
) -> None:
    """Screen the "raw" pairs \\ keep the survivors.

    Args:
        t: Amplitudes to screen in place.
        label: Which pair class the cutoff applies to.
        cutoff: Screening threshold.
        tighten: Whether to shrink domains afterwards.
        n_keep: Keep at least this many pairs; -1 keeps all.
        start: First pair ordinal to touch.
    """
'''


@pytest.fixture(scope="module")
def edge_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("promote_edge")
    pkg = root / "edgemethod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "stages.py").write_text(EDGE_FIXTURE)
    sys.path.insert(0, str(root))
    try:
        yield _promote.build_model(_promote.load_stages_module(str(pkg / "stages.py")))
    finally:
        sys.path.remove(str(root))


@pytest.fixture(scope="module")
def edge_generated(edge_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("promote_edge_out")
    _emit.generate(edge_model, out, formatters=_emit.find_formatters())
    return out


ESCAPED_LABEL = r'"core \"frozen\" \\ tag"'


def test_defaults_cross_with_their_cpp_spelling(edge_model):
    defaults = {p.name: p.cpp_default for p in edge_model.stages[0].params}
    assert defaults == {
        "t": None,
        "label": ESCAPED_LABEL,
        "cutoff": "1e-06",
        "tighten": "false",
        "n_keep": "-1",
        "start": "0",
    }


def test_the_header_carries_the_defaults_and_the_skeleton_does_not(edge_generated):
    header = (edge_generated / "include/edgemethod/ScreenPairs.hpp").read_text()
    for decl in ("double cutoff = 1e-06", "bool tighten = false", "std::int64_t n_keep = -1"):
        assert decl in header, decl
    assert ESCAPED_LABEL in header
    # C++ refuses a default argument repeated at the definition, so the
    # skeleton must spell the parameters bare.
    skeleton = (edge_generated / "src/ScreenPairs.cpp").read_text()
    assert "cutoff = " not in skeleton.split("// The Python backend")[0]


def test_the_binding_applies_the_same_defaults(edge_generated):
    bindings = (edge_generated / "bindings.cpp").read_text()
    assert 'py::arg("t"),' in bindings  # required stays bare
    assert 'py::arg("cutoff") = 1e-06' in bindings
    assert 'py::arg("tighten") = false' in bindings
    assert 'py::arg("n_keep") = -1' in bindings
    assert 'py::arg("start") = 0' in bindings
    assert f'py::arg("label") = {ESCAPED_LABEL}' in bindings


def test_docstring_prose_is_escaped_into_the_binding(edge_generated):
    bindings = (edge_generated / "bindings.cpp").read_text()
    assert r'"Screen the \"raw\" pairs \\ keep the survivors."' in bindings
    # The unescaped form would be an unterminated literal; its absence is the
    # actual claim.
    assert '"Screen the "raw"' not in bindings


def test_cpp_string_literal_escapes_what_cpp_cares_about():
    assert _promote.cpp_string_literal('a "b" \\ c') == r'"a \"b\" \\ c"'
    assert _promote.cpp_string_literal("line\nbreak\ttab") == r'"line\nbreak\ttab"'


def test_defaults_outside_the_scalar_types_are_refused_at_decoration():
    from typing import Annotated

    from einsums.stages import TensorD, index, stage

    with pytest.raises(ContractError, match="only int, float, bool and str"):

        @stage
        def edge_bad_tensor_default(t: TensorD = None) -> None:
            """A tensor default has no C++ spelling."""

    with pytest.raises(ContractError, match="finite sentinel"):

        @stage
        def edge_bad_inf_default(x: float = float("inf")) -> None:
            """Non-finite floats have no portable literal."""

    with pytest.raises(ContractError, match="not an int"):

        @stage
        def edge_bad_typed_default(x: int = "five") -> None:
            """The default must be an instance of the annotated type."""

    with pytest.raises(ContractError, match="std::size_t"):

        @stage
        def edge_bad_index_default(i: Annotated[int, index] = -1) -> None:
            """An index crosses as std::size_t, so a negative default wraps."""


def test_a_required_parameter_after_a_defaulted_one_is_refused():
    from einsums.stages import stage

    # Legal Python behind the keyword-only marker, but C++ has no such marker
    # and its defaults are trailing only, so the two sides could never agree.
    with pytest.raises(ContractError, match="trailing"):

        @stage
        def edge_bad_default_order(a: int = 1, *, b: int) -> None:
            """A required parameter after a defaulted one."""


# ----------------------------------------------------------------------
# Contract-level: the parallel declaration is enforced on both sides
# ----------------------------------------------------------------------
def test_parallel_is_checked_when_the_contract_is_built(fixture_module):
    Sparsity = fixture_module.Sparsity
    Sparsity(row=[0, 1], col=[2, 3], cutoff=1e-6)
    with pytest.raises(ContractError, match="parallel"):
        Sparsity(row=[0, 1], col=[2], cutoff=1e-6)


def test_parallel_rejects_a_name_that_is_not_a_list_field():
    from dataclasses import dataclass

    from einsums.stages import cmp, contract

    with pytest.raises(ContractError, match="only list fields"):

        @contract(parallel=["a", "b"])
        @dataclass(frozen=True)
        class Bad:
            a: list[int] = cmp.exact()
            b: int = 0

    with pytest.raises(ContractError, match="not a field"):

        @contract(parallel=["a", "nope"])
        @dataclass(frozen=True)
        class AlsoBad:
            a: list[int] = cmp.exact()
            b: list[int] = cmp.exact()


# ----------------------------------------------------------------------
# The DLPNO regression: promote reproduces the checked-in stage module
# ----------------------------------------------------------------------
def _formatters_or_skip():
    missing = [t for t in ("clang-format", "cmake-format") if shutil.which(t) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not on PATH; generated output would be unformatted")
    return _emit.find_formatters()


@pytest.mark.skipif(not DLPNO.is_dir(), reason="examples/dlpno is not in this tree")
def test_promote_reproduces_the_checked_in_dlpno_module(tmp_path):
    """The definition of done for M4, as a diff.

    Everything promote claims to generate must come back byte for byte out of
    an empty directory. The port is excluded on purpose - it is the one file
    promote does not write - and is checked on its signature instead, which is
    the part the generator IS responsible for.
    """
    formatters = _formatters_or_skip()

    # Both formatters discover their config from the OUTPUT path, which is the
    # right behavior - generated code should land in the developer's own style.
    # It also means a bare temp directory would be formatted at the tools'
    # defaults and could never match anything. Copying the repo's configs makes
    # this a test of the generator instead of a test of config discovery.
    shutil.copy(REPO_ROOT / ".clang-format", tmp_path / ".clang-format")
    shutil.copy(REPO_ROOT / ".cmake-format.py", tmp_path / ".cmake-format.py")

    sys.path.insert(0, str(DLPNO))
    try:
        module = _promote.load_stages_module(str(DLPNO / "dlpno" / "stages.py"))
        model = _promote.build_model(module)
        _emit.generate(
            model,
            tmp_path,
            license_text=(REPO_ROOT / "devtools" / "LicenseHeader.txt").read_text(),
            formatters=formatters,
        )
    finally:
        sys.path.remove(str(DLPNO))

    checked_in = DLPNO / "cpp"
    regenerated = {
        "include/dlpno/Contracts.hpp": ["struct CouplingPlan {", "struct PnoOverlaps {"],
        "include/dlpno/ComputePnoOverlaps.hpp": ["compute_pno_overlaps(", "CouplingPlan const &plan"],
        "include/dlpno/Casters.hpp": ["type_caster<dlpno::CouplingPlan>", "must be parallel"],
        "bindings.cpp": ['EINSUMS_STAGE_MODULE(m, "dlpno_stages")', "stage_compute_pno_overlaps"],
    }
    # Scaffolds are the developer's files after the first write, and the DLPNO
    # ones HAVE been edited since (d39bade8f made the diff tests' module load
    # idempotent), so byte-identity holds only for the regenerated set. Judging
    # scaffolds by content keeps this a test of the generator rather than a ban
    # on ever exercising the ownership rule the generator itself promises.
    scaffolded = {
        "CMakeLists.txt": ["find_package(Einsums", "einsums_add_stage_module"],
        "tests/test_diff_compute_pno_overlaps.py": ["def make_inputs()"],
    }
    for rel, needles in {**regenerated, **scaffolded}.items():
        generated_text = (tmp_path / rel).read_text()
        # Content first, byte-identity second. Two identically empty files are
        # byte-identical, and that is how an empty CMakeLists.txt got past this.
        for needle in needles:
            assert needle in generated_text, f"{rel}: missing {needle!r}"
        if rel in regenerated:
            assert generated_text == (checked_in / rel).read_text(), rel

    # The port: compared on its signature, since it is 252 lines of algorithm
    # promote is not trying to write. A definition that no longer matches the
    # generated declaration is exactly the drift M7's `check` will catch.
    declaration = (tmp_path / "src/ComputePnoOverlaps.cpp").read_text()
    port = (checked_in / "src/ComputePnoOverlaps.cpp").read_text()
    signature = declaration[declaration.index("PnoOverlaps compute_pno_overlaps") : declaration.index(") {")]
    assert signature in port


@pytest.mark.skipif(not DLPNO.is_dir(), reason="examples/dlpno is not in this tree")
def test_promote_cli_leaves_the_checked_in_tree_clean():
    """Running the documented command must be a no-op on a clean tree."""
    _formatters_or_skip()
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "einsums.stages",
            "promote",
            str(DLPNO / "dlpno" / "stages.py"),
            "--out",
            str(DLPNO / "cpp"),
            "--license-header",
            str(REPO_ROOT / "devtools" / "LicenseHeader.txt"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # unchanged for everything regenerated, kept for the scaffolds. A "refused"
    # here means the checked-in files no longer match what promote writes.
    assert "refused" not in done.stdout, done.stdout
    assert "updated" not in done.stdout, done.stdout
