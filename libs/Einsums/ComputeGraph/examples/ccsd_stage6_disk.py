# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""CCSD tutorial, stage 6: the integrals and the amplitudes in files.

Stage 5 with the inputs moved out of memory. The integrals are written to ints.etn once and the
graph reads them from there; t1 and t2 live in amps.etn, which each replay reads and writes back.
iterate() and energy() are unchanged: what changed is where the tensors they are handed live.
"""

from __future__ import annotations

import os
import tempfile

import einsums.rc

# The runtime reads the budget when it starts, so it is set before einsums is imported.
einsums.rc.max_memory = "16MB"

import einsums  # noqa: E402
import einsums.graph as cg  # noqa: E402
from ccsd_equations import annotate, converge, energy, integrals, iterate, register_spaces  # noqa: E402
from ccsd_problem import REFERENCE_CCSD, load  # noqa: E402

p = load()
eager = integrals(p)
on_disk = ("oovv", "ooov", "oovo", "oooo", "ovvo", "ovov", "ovvv", "vvvo", "ovoo", "Bvv")

with tempfile.TemporaryDirectory() as scratch:
    # [doc:files]
    ints_path, amps_path = os.path.join(scratch, "ints.etn"), os.path.join(scratch, "amps.etn")
    for name in on_disk:
        einsums.io.write(ints_path, name, eager[name])
    t2_guess = einsums.zeros([p.nocc, p.nocc, p.nvir, p.nvir], name="t2")
    einsums.linalg.direct_division(1.0, eager["oovv"], eager["Dijab"], 0.0, t2_guess)
    einsums.io.write(amps_path, "t1", einsums.zeros([p.nocc, p.nvir], name="t1"))
    einsums.io.write(amps_path, "t2", t2_guess)

    graph = cg.Graph("ccsd")
    files, amplitudes = einsums.io.EtnFile(ints_path), einsums.io.EtnFile(amps_path)
    ints = {name: graph.declare_tensor(name, backing=files.entry(name)) for name in on_disk}
    ints.update({name: eager[name] for name in ("Foo", "Fvv", "Dia", "Dijab")})
    t1 = graph.declare_tensor("t1", backing=amplitudes.entry("t1", update=True))
    t2 = graph.declare_tensor("t2", backing=amplitudes.entry("t2", update=True))
    # [doc:files-end]
    e = einsums.zeros([1], name="e")

    register_spaces()

    def new(name, shape):
        return graph.scratch(name, list(shape), "float64")

    with cg.capture(graph):
        annotate(t1, t2, ints)
        iterate(t1, t2, ints, new)
        energy(e, t1, t2, ints, new)
    graph.optimize()

    # [doc:report]
    for decision in graph.decisions:
        if decision.code in ("axis-tiling.sliced", "slab-streaming.streamed-box", "slab-streaming.streamed-read"):
            print(decision.text)
    # [doc:report-end]

    iterations, ms = converge(graph.execute, lambda: float(e[0]))

    # [doc:measured]
    run = graph.last_run()
    print(f"the last replay held at most {run.peak_bytes} bytes")
    for backing in run.backings:
        print(f"  {backing.tensor}: {backing.bytes_read} bytes read in {backing.reads} read(s), {backing.bytes_written} written")
    # [doc:measured-end]

print(f"CCSD correlation energy = {float(e[0]):.10f}   ({iterations} iterations, {ms:.2f} ms each)")
assert abs(float(e[0]) - REFERENCE_CCSD) < 1e-10
