from pertox_agent.workflow.nodes import _merge_drug_exposure


def test_merge_drug_exposure_uses_patient_exposure_as_fallback() -> None:
    raw_drug = {"name": "amiodarone", "dose": "unspecified"}
    raw_patient = {"exposure": {"dose": "200 mg/day", "route": "oral", "frequency": "daily", "form": "tablet"}}

    merged = _merge_drug_exposure(raw_drug, raw_patient)

    assert merged == {
        "name": "amiodarone",
        "dose": "200 mg/day",
        "route": "oral",
        "frequency": "daily",
        "form": "tablet",
    }
    assert raw_drug == {"name": "amiodarone", "dose": "unspecified"}


def test_merge_drug_exposure_does_not_override_drug_fields() -> None:
    raw_drug = {"name": "amiodarone", "dose": "100 mg/day", "route": "intravenous"}
    raw_patient = {"exposure": {"dose": "200 mg/day", "route": "oral", "frequency": "daily"}}

    merged = _merge_drug_exposure(raw_drug, raw_patient)

    assert merged["dose"] == "100 mg/day"
    assert merged["route"] == "intravenous"
    assert merged["frequency"] == "daily"
