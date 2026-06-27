from unittest.mock import Mock

from pertox_agent.app import cli


def test_load_state_from_text() -> None:
    input_path = Mock()
    input_path.read_text.return_value = """
    [patient]
    patient_id: demo-warfarin-001
    age: 65
    sex: female
    genotypes.CYP2C9: *2/*3
    medical_history: K74.6 cirrhosis; I48 atrial fibrillation
    concomitant_medications: amiodarone
    organ_function.LVEF: 55%
    exposure.route: oral

    [drug]
    name: warfarin
    drugbank_id: DB00682
    frequency: daily
    """

    state = cli.load_state_from_text(input_path)

    assert state["raw_patient_info"]["age"] == 65
    assert state["raw_patient_info"]["genotypes"] == {"CYP2C9": "*2/*3"}
    assert state["raw_patient_info"]["medical_history"] == ["K74.6 cirrhosis", "I48 atrial fibrillation"]
    assert state["raw_patient_info"]["organ_function"] == {"LVEF": "55%"}
    assert state["raw_patient_info"]["exposure"] == {"route": "oral"}
    assert state["raw_drug_info"]["name"] == "warfarin"
    assert state["raw_drug_info"]["drugbank_id"] == "DB00682"


def test_load_state_from_json_accepts_demo_shape() -> None:
    input_path = Mock()
    input_path.read_text.return_value = """
    {
      "raw_patient_info": {"patient_id": "demo", "age": 65, "sex": "female"},
      "raw_drug_info": {"name": "warfarin", "drugbank_id": "DB00682"}
    }
    """

    state = cli.load_state_from_json(input_path)

    assert state["raw_patient_info"]["patient_id"] == "demo"
    assert state["raw_drug_info"]["name"] == "warfarin"
    assert state["messages"] == []
    assert state["trace"] == []


def test_main_uses_shared_report_runner() -> None:
    state = {"raw_patient_info": {"age": 65}, "raw_drug_info": {"name": "warfarin"}}
    calls = {}

    original_load_state_from_json = cli.load_state_from_json
    original_run_persagent_report = cli.run_persagent_report
    cli.load_state_from_json = Mock(return_value=state)
    cli.run_persagent_report = Mock(side_effect=lambda state_arg, path: calls.update(state=state_arg, path=path))

    try:
        assert cli.main(["--json", "input.json"]) == 0

        cli.load_state_from_json.assert_called_once()
        cli.run_persagent_report.assert_called_once()
    finally:
        cli.load_state_from_json = original_load_state_from_json
        cli.run_persagent_report = original_run_persagent_report
    assert calls["state"] is state
    assert calls["path"].name == "final_report_warfarin.json"
