# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""ComputeGraph structure as the profiler reports it, and TaskPool metrics.

``get_compute_graphs`` (``ComputeGraph/src/Graph/Report.cpp``) answers with
``{"graphs": [...]}``; ``--einsums:profile:save`` stores the same list as the session's
``einsums.compute_graphs`` extension. Each graph has ``tensors`` (id, name, dims, dtype), ``nodes``
(id, kind, label, target, stream_id, inputs/outputs as tensor ids, optional timing_ms
and einsum indices) and ``edges`` (from, to, tensor_id). No Textual here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphTensor:
    id: int
    name: str
    dims: list[int]
    dtype: str = ""
    intermediate: bool = False

    @property
    def shape(self) -> str:
        return "x".join(str(d) for d in self.dims) if self.dims else "scalar"


@dataclass
class GraphNode:
    id: int
    kind: str
    label: str
    target: str = ""
    stream_id: int = 0
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    timing_ms: float | None = None
    spec: str = ""  # "ij <- ik,kj" for an einsum

    @property
    def title(self) -> str:
        return f"{self.kind} {self.label}".strip() if self.label != self.kind else self.kind


@dataclass
class Graph:
    name: str
    context: str  # pipeline / workspace / stage, when the graph ran inside one
    tensors: dict[int, GraphTensor]
    nodes: list[GraphNode]
    edges: list[tuple[int, int, int]]  # (from node, to node, tensor id)

    @property
    def total_ms(self) -> float:
        return sum(n.timing_ms or 0.0 for n in self.nodes)

    def predecessors(self, node_id: int) -> list[int]:
        return sorted({src for src, dst, _ in self.edges if dst == node_id})

    def tensor_name(self, tid: int) -> str:
        t = self.tensors.get(tid)
        return f"{t.name} [{t.shape}]" if t else f"tensor {tid}"


def parse_graph(data: dict[str, Any]) -> Graph:
    tensors = {
        t["id"]: GraphTensor(t["id"], t.get("name", ""), list(t.get("dims", [])), t.get("dtype", ""), t.get("is_intermediate", False))
        for t in data.get("tensors", [])
    }
    nodes = []
    for n in data.get("nodes", []):
        spec = ""
        if n.get("c_indices") is not None or n.get("a_indices") is not None:
            operands = ",".join(x for x in (n.get("a_indices"), n.get("b_indices")) if x)
            spec = f"{n.get('c_indices', '')} <- {operands}"
        nodes.append(
            GraphNode(
                id=n["id"],
                kind=n.get("kind", ""),
                label=n.get("label", ""),
                target=n.get("target", ""),
                stream_id=n.get("stream_id", 0),
                inputs=list(n.get("inputs", [])),
                outputs=list(n.get("outputs", [])),
                timing_ms=n.get("timing_ms"),
                spec=spec,
            )
        )
    context = " / ".join(data[k] for k in ("pipeline_name", "workspace_name", "stage_name") if data.get(k))
    edges = [(e["from"], e["to"], e.get("tensor_id", -1)) for e in data.get("edges", [])]
    return Graph(data.get("name", "graph"), context, tensors, nodes, edges)


def parse_graphs(payload: Any) -> list[Graph]:
    """The graphs in a ``get_compute_graphs`` reply or a session's ``einsums.compute_graphs`` list."""
    if isinstance(payload, dict):
        payload = payload.get("graphs", [])
    return [parse_graph(g) for g in payload or [] if isinstance(g, dict)]


def graph_summary(graph: Graph, top: int = 8) -> str:
    """Rich markup describing a graph: size, total timing, and its slowest nodes."""
    from rich.markup import escape

    lines = [f"[bold]{escape(graph.name)}[/bold]"]
    if graph.context:
        lines.append(f"  {escape(graph.context)}")
    lines.append(f"  {len(graph.nodes)} nodes, {len(graph.tensors)} tensors, {len(graph.edges)} edges")
    timed = [n for n in graph.nodes if n.timing_ms is not None]
    if timed:
        lines.append(f"  measured node time: {graph.total_ms:.3f} ms")
        lines.append("  [bold]Slowest nodes:[/bold]")
        for n in sorted(timed, key=lambda n: n.timing_ms or 0.0, reverse=True)[:top]:
            share = (n.timing_ms or 0.0) / graph.total_ms * 100.0 if graph.total_ms > 0 else 0.0
            lines.append(f"    {n.timing_ms:10.3f} ms {share:5.1f}%  #{n.id} {escape(n.title)}")
    else:
        lines.append("  (no node timings: the graph has not been executed with the profiler on)")
    return "\n".join(lines)


def node_summary(graph: Graph, node: GraphNode) -> str:
    from rich.markup import escape

    lines = [f"[bold]#{node.id} {escape(node.title)}[/bold]"]
    if node.spec:
        lines.append(f"  spec: {escape(node.spec)}")
    where = [x for x in (f"target {node.target}" if node.target else "", f"stream {node.stream_id}") if x]
    lines.append("  " + ", ".join(where))
    if node.timing_ms is not None:
        share = node.timing_ms / graph.total_ms * 100.0 if graph.total_ms > 0 else 0.0
        lines.append(f"  time: {node.timing_ms:.3f} ms ({share:.1f}% of the graph)")
    lines.append("  [bold]reads:[/bold] " + (", ".join(escape(graph.tensor_name(t)) for t in node.inputs) or "-"))
    lines.append("  [bold]writes:[/bold] " + (", ".join(escape(graph.tensor_name(t)) for t in node.outputs) or "-"))
    after = graph.predecessors(node.id)
    lines.append("  [bold]after:[/bold] " + (", ".join(f"#{p}" for p in after) or "- (a root)"))
    return "\n".join(lines)


@dataclass
class TaskPoolMetrics:
    submitted: int
    completed: int
    steals: int
    active: int
    workers: int
    executed: list[int]
    stolen: list[int]


def parse_taskpool(data: dict[str, Any]) -> TaskPoolMetrics | None:
    """A ``get_taskpool_metrics`` reply, or None for an error (no TaskPool in the program)."""
    if not isinstance(data, dict) or "error" in data or "num_workers" not in data:
        return None
    return TaskPoolMetrics(
        data.get("total_submitted", 0),
        data.get("total_completed", 0),
        data.get("total_steals", 0),
        data.get("active_workers", 0),
        data.get("num_workers", 0),
        list(data.get("per_worker_executed", [])),
        list(data.get("per_worker_stolen", [])),
    )


def taskpool_text(m: TaskPoolMetrics | None, error: str = "") -> str:
    if m is None:
        return f"[bold]TaskPool[/bold]\n  {error or 'this program has no TaskPool (it answered: unknown method)'}"
    pending = m.submitted - m.completed
    lines = [
        f"[bold]TaskPool[/bold]  {m.workers} workers, {m.active} active   submitted {m.submitted:,}  "
        f"completed {m.completed:,}  pending {pending:,}  steals {m.steals:,}",
    ]
    peak = max(m.executed or [0]) or 1
    idle = 0
    for i, (done, stole) in enumerate(zip(m.executed, m.stolen + [0] * (len(m.executed) - len(m.stolen)))):
        if done == 0 and stole == 0:
            idle += 1  # a 48-core machine is mostly idle rows otherwise
            continue
        bar = "█" * int(done / peak * 30)
        lines.append(f"  worker {i:3d} {bar:<30} {done:>10,} run  {stole:>8,} stolen")
    if idle:
        lines.append(f"  ({idle} worker(s) have run nothing yet)")
    return "\n".join(lines)
