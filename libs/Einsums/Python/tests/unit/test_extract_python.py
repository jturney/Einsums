#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Tests for ``python -m einsums.stages extract``: the analysis, the spec, the scaffold.

The analysis is tested against a synthetic solver written into a temp
directory, small enough that every access kind it exercises is named in the
source next to the assertion that checks it. The one real-world case at the
bottom runs the analysis over the DLPNO example, which is where the tool's
numbers have to agree with the ones the design document records.
"""

import textwrap
import pathlib

import pytest

import einsums  # noqa: F401  (loads the runtime)
from einsums.stages import _extract
from einsums.stages.__main__ import main

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
DLPNO = REPO_ROOT / "examples" / "dlpno"


# ----------------------------------------------------------------------
# A solver split over two files, like a method on a base class.
# ----------------------------------------------------------------------
BASE = '''
class SolverBase:
    @property
    def n_pairs(self):
        return len(self.pair_index)

    def _prepare(self, ij):
        return self.integrals[ij] * self.cfg.screening

    def _log(self, message):
        if self.verbose:
            print(message)
'''

SOLVER = '''
from .base import SolverBase

class Solver(SolverBase):
    def build_t2(self):
        total = 0.0
        runner = self._run          # bound-method reference, no call
        for ij in range(self.n_pairs):
            k = self._prepare(ij)
            self.t2[ij] = k / self.denom[ij]     # subscript store
            self.history.append(k)               # mutator call
            total += self.cfg.shift              # chain read
        self.e_corr = total                      # plain write
        self.iterations += 1                     # augmented: read and write
        self._log("done")
        return runner

    def _run(self, graph):
        self.executor.execute(graph)
'''


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("extract_fixture")
    pkg = root / "mymethod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "base.py").write_text(BASE)
    (pkg / "solver.py").write_text(SOLVER)
    return pkg


@pytest.fixture(scope="module")
def analysis(sources):
    return _extract.analyze(
        f"{sources / 'solver.py'}::Solver.build_t2", also=[sources / "base.py"]
    )


# ----------------------------------------------------------------------
# The analysis
# ----------------------------------------------------------------------
def test_field_set_and_classification(analysis):
    fields = analysis.fields
    assert set(fields) == {
        "pair_index", "integrals", "cfg", "verbose", "n_pairs", "denom",
        "t2", "history", "e_corr", "iterations", "executor",
    }
    reads = {f.name for f in analysis.reads}
    writes = {f.name for f in analysis.writes}
    # A subscript store and a mutator call are writes, which is the direction
    # the report script this grew from was documented to get wrong.
    assert {"t2", "history", "e_corr", "iterations"} <= writes
    assert {"cfg", "denom", "integrals", "n_pairs", "executor"} <= reads


def test_property_is_state_and_still_followed(analysis):
    # n_pairs is a @property: it needs a disposition like any field...
    assert "n_pairs" in analysis.fields
    assert "n_pairs" not in analysis.helpers
    # ...and its body was followed, so the field it reads is evidence.
    assert "pair_index" in analysis.fields
    assert analysis.fields["pair_index"].via == ["n_pairs"]


def test_helpers_followed_transitively(analysis):
    assert set(analysis.helpers) == {"_prepare", "_log", "_run"}
    # _run was reached through a bound-method reference without a call.
    assert analysis.fields["executor"].via == ["_run"]
    # Fields reached only inside helpers say so.
    assert analysis.fields["verbose"].via == ["_log"]


def test_chain_reads_are_recorded(analysis):
    assert set(analysis.fields["cfg"].chains) == {"cfg.screening", "cfg.shift"}


def test_bad_targets_are_refused(sources):
    with pytest.raises(_extract.ExtractError, match="Class.method"):
        _extract.analyze(str(sources / "solver.py"))
    with pytest.raises(_extract.ExtractError, match="not found"):
        _extract.analyze(f"{sources / 'solver.py'}::Solver.no_such")


# ----------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------
def _template_path(analysis, tmp_path):
    path = tmp_path / "cut.toml"
    path.write_text(_extract.render_template(analysis))
    return path


def test_template_refuses_until_filled(analysis, tmp_path):
    path = _template_path(analysis, tmp_path)
    with pytest.raises(_extract.ExtractError, match="TODO"):
        _extract.load_spec(path, analysis)


GOOD_SPEC = '''
[stage]
name = "build_t2"
contract = "T2Result"
extra_returns = ["residual_norm"]

[fields.integrals]
disposition = "param"

[fields.denom]
disposition = "param"
params = ["denominators"]

[fields.cfg]
disposition = "plan"
params = ["screening", "shift"]
note = "two scalars in disguise"

[fields.pair_index]
disposition = "plan"

[fields.n_pairs]
disposition = "plan"

[fields.t2]
disposition = "returns"

[fields.e_corr]
disposition = "returns"

[fields.history]
disposition = "finish"

[fields.iterations]
disposition = "finish"

[fields.verbose]
disposition = "finish"

[fields.executor]
disposition = "finish"

[helpers._prepare]
disposition = "stage"

[helpers._run]
disposition = "stage"

[helpers._log]
disposition = "finish"
'''


def test_good_spec_loads(analysis, tmp_path):
    path = tmp_path / "cut.toml"
    path.write_text(GOOD_SPEC)
    spec = _extract.load_spec(path, analysis)
    # Parameter order is spec order, with a param entry's default name filled.
    assert spec.params == ["integrals", "denominators", "screening", "shift"]
    assert spec.return_fields == ["t2", "e_corr", "residual_norm"]
    assert spec.contract_name == "T2Result"


@pytest.mark.parametrize(
    "edit, complaint",
    [
        # A field the method does not touch is stale, not silently ignored.
        ('\n[fields.ghost]\ndisposition = "plan"\n', "stale"),
        # Dropping a touched field is refused, which is the tool's whole point.
        (("[fields.verbose]", "[fields.verbose_gone]"), "does not mention"),
        # An output cannot cross inward, an input cannot come back.
        (('disposition = "param"\n\n[fields.denom]',
          'disposition = "returns"\n\n[fields.denom]'), "never writes"),
        (('[fields.t2]\ndisposition = "returns"',
          '[fields.t2]\ndisposition = "param"'), "only writes"),
        # Two entries fighting over one parameter name.
        (('params = ["denominators"]', 'params = ["integrals"]'), "declared by both"),
        # params on a disposition that crosses nothing.
        (('[fields.history]\ndisposition = "finish"',
          '[fields.history]\ndisposition = "finish"\nparams = ["h"]'),
         "does not cross"),
        # extra_returns colliding with a returns field.
        (('extra_returns = ["residual_norm"]',
          'extra_returns = ["t2"]'), "already returns"),
        # A misspelled disposition names the vocabulary.
        (('disposition = "finish"\n\n[helpers._prepare]',
          'disposition = "later"\n\n[helpers._prepare]'), "not one of"),
    ],
)
def test_spec_problems_are_refused(analysis, tmp_path, edit, complaint):
    text = GOOD_SPEC
    if isinstance(edit, tuple):
        assert edit[0] in text
        text = text.replace(edit[0], edit[1], 1)
    else:
        text += edit
    path = tmp_path / "cut.toml"
    path.write_text(text)
    with pytest.raises(_extract.ExtractError, match=complaint):
        _extract.load_spec(path, analysis)


def test_all_problems_reported_at_once(analysis, tmp_path):
    text = GOOD_SPEC.replace('[fields.verbose]', '[fields.verbose_gone]', 1) \
        + '\n[fields.ghost]\ndisposition = "plan"\n'
    path = tmp_path / "cut.toml"
    path.write_text(text)
    with pytest.raises(_extract.ExtractError) as err:
        _extract.load_spec(path, analysis)
    message = str(err.value)
    assert "ghost" in message and "verbose" in message


# ----------------------------------------------------------------------
# The scaffold
# ----------------------------------------------------------------------
@pytest.fixture()
def scaffold(analysis, tmp_path):
    path = tmp_path / "cut.toml"
    path.write_text(GOOD_SPEC)
    spec = _extract.load_spec(path, analysis)
    return spec, _extract.render_scaffold(analysis, spec)


def test_scaffold_has_the_cut(scaffold):
    spec, text = scaffold
    assert "def build_t2(" in text
    assert "class T2Result:" in text
    assert "def plan_build_t2(self):" in text
    assert "def _finish_build_t2(self, result):" in text
    # Parameters appear in spec order, each traceable to its origin field.
    assert text.index("    integrals,") < text.index("    denominators,")
    assert "(from self.denom)" in text
    # The narrowing is stated, not buried.
    assert "11 fields" in text and "4 parameters" in text and "3 return fields" in text


def test_scaffold_does_not_import(scaffold, tmp_path):
    # Deliberate: TODO annotations mean the contract is not stated yet, and a
    # scaffold that imports cleanly reads as more finished than it is. Under
    # lazy annotations the refusal comes from @contract's hint resolution
    # rather than the name lookup, so pin the message rather than the type.
    _, text = scaffold
    target = tmp_path / "extracted_build_t2.py"
    target.write_text(text)
    with pytest.raises(Exception, match="TODO"):
        exec(compile(text, str(target), "exec"), {})


# ----------------------------------------------------------------------
# The command line
# ----------------------------------------------------------------------
def test_cli_roundtrip(sources, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = f"{sources / 'solver.py'}::Solver.build_t2"
    also = str(sources / "base.py")

    # First run: report plus template, and a TODO template does not scaffold.
    assert main(["extract", target, "--also", also]) == 0
    template = tmp_path / "build_t2.cut.toml"
    assert template.exists()
    assert "11 fields" in capsys.readouterr().out
    assert main(["extract", target, "--also", also]) == 1
    assert "TODO" in capsys.readouterr().err

    # Filled spec: the scaffold is written, once and only once.
    template.write_text(GOOD_SPEC)
    assert main(["extract", target, "--also", also]) == 0
    scaffold = tmp_path / "extracted_build_t2.py"
    assert scaffold.exists()
    assert "4 parameters" in capsys.readouterr().out
    assert main(["extract", target, "--also", also]) == 1
    assert "never" in capsys.readouterr().err


# ----------------------------------------------------------------------
# The real thing
# ----------------------------------------------------------------------
@pytest.mark.skipif(not DLPNO.exists(), reason="dlpno example not present")
def test_dlpno_lmp2_iterations_analyzes(capsys):
    """The remaining wide DLPNO method analyzes, and it is still wide.

    The design records lmp2_iterations touching ~45 fields, which is its
    argument that the phase is cut at the wrong boundary for promotion. The
    exact count moves as mp2.py evolves; what this pins is that the analysis
    runs on the real code and keeps reporting the width honestly.
    """
    analysis = _extract.analyze(
        f"{DLPNO / 'dlpno' / 'mp2.py'}::DLPNOMP2.lmp2_iterations",
        also=[DLPNO / "dlpno" / "base.py"],
    )
    assert len(analysis.fields) > 30
    assert not analysis.unresolved
