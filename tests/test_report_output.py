from pathlib import Path
from unittest.mock import Mock

from pertox_agent.reporting import terminal_output


class FakeGraph:
    def stream(self, _state, stream_mode=None):
        assert stream_mode == "updates"
        yield {
            "format_output": {
                "trace": ["parsed", "formatted"],
                "final_output": {
                    "json": {
                        "payload": {
                            "drug_entity": {"primary_name": "demo"},
                            "admet_profile": [],
                            "known_ade_profile": [],
                            "mechanism_chains": [],
                            "persade_contextual_evidence": [],
                            "universal_toxicity_report": {"general_toxicity": []},
                            "personalized_toxicity_report": {"personalized_toxicity": []},
                        },
                        "json": "{}",
                    }
                },
            }
        }


def test_stream_report_writes_final_output_then_trace(monkeypatch, capsys) -> None:
    write_mock = Mock()
    monkeypatch.setattr(terminal_output, "write_final_report", write_mock)

    final_state = terminal_output.stream_report(FakeGraph(), {"trace": []}, Path("results/final_report_demo.json"))

    output = capsys.readouterr().out
    assert output.index("=== Final Output ===") < output.index("=== PersAgent Trace ===")
    assert "JSON report:" in output
    assert "- formatted" in output
    assert final_state["trace"] == ["parsed", "formatted"]
    write_mock.assert_called_once()
