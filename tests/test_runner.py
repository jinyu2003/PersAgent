from pathlib import Path
from unittest.mock import Mock

from pertox_agent.app import runner


def test_run_persagent_report_uses_shared_graph_and_stream(monkeypatch) -> None:
    graph = object()
    state = {"raw_patient_info": {}, "raw_drug_info": {}}
    output_path = Path("results/final_report.json")
    final_state = {"final_output": {"json": {}}}

    build_graph = Mock(return_value=graph)
    stream_report = Mock(return_value=final_state)
    monkeypatch.setattr(runner, "build_report_graph", build_graph)
    monkeypatch.setattr(runner, "stream_report", stream_report)

    result = runner.run_persagent_report(state, output_path)

    assert result is final_state
    build_graph.assert_called_once_with()
    stream_report.assert_called_once_with(graph, state, output_path)
