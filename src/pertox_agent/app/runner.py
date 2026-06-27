"""Shared PersAgent report execution entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pertox_agent.reporting.terminal_output import stream_report


def build_report_graph() -> Any:
    from pertox_agent.workflow.graph import build_graph

    return build_graph()


def run_persagent_report(
    initial_state: dict[str, Any],
    output_path: Path,
    *,
    graph: Any | None = None,
) -> dict[str, Any]:
    active_graph = graph if graph is not None else build_report_graph()
    return stream_report(active_graph, initial_state, output_path)
