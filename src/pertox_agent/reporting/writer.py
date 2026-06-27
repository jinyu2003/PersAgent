"""Final report JSON writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_final_output(final_state: dict[str, Any]) -> dict[str, Any]:
    final_output = final_state.get("final_output")
    if final_output is None:
        from pertox_agent.reporting.formatter import format_to_json

        final_output = {"json": format_to_json(final_state["draft_report"])}
        final_state["final_output"] = final_output
    return final_output


def write_final_report(final_state: dict[str, Any], output_path: Path) -> None:
    final_output = ensure_final_output(final_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_output["json"]["json"], encoding="utf-8")
