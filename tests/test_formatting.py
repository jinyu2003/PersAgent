from pertox_agent.reporting.formatter import format_to_json


def test_format_to_json_filters_stage1_context_to_liver_and_heart() -> None:
    report = {
        "drug_entity": {"primary_name": "ExampleDrug"},
        "patient_features": {"patient_id": "demo"},
        "structure_profile": {"descriptors": {"MW": 100}},
        "admet_profile": [
            {"endpoint": "label_DILI_t", "value": "1", "soc": "Hepatobiliary disorders"},
            {"endpoint": "label_hERG", "value": "1", "mechanism_group": "hERG/QT"},
            {"endpoint": "label_Neural_t", "value": "1", "soc": "Nervous system disorders"},
            {"endpoint": "label_from_attribution", "value": "1", "soc": None},
        ],
        "known_ade_profile": [
            {"ade_id": "heart-1", "ade_name": "Thrombosis", "soc": "Cardiac disorders", "organ_system": "heart"},
            {"ade_id": "skin-1", "ade_name": "Rash", "soc": "Skin and subcutaneous tissue disorders", "organ_system": "skin"},
        ],
        "mechanism_chains": [
            {"chain_id": "liver-chain", "soc": "Hepatobiliary disorders", "organ_system": "liver"},
            {"chain_id": "skin-chain", "soc": "Skin and subcutaneous tissue disorders", "organ_system": "skin"},
        ],
        "persade_contextual_evidence": [
            {"ade_id": "heart-1", "soc": None},
            {"ade_id": "skin-1", "soc": None},
        ],
        "universal_report": {
            "general_toxicity": [
                {
                    "soc": "Hepatobiliary disorders",
                    "baseline_probability": 0.42,
                    "attribution": {
                        "admet_endpoint": [{"endpoint": "label_from_attribution"}],
                        "molecular_attribution": [],
                    },
                }
            ]
        },
        "personalized_report": {"personalized_toxicity": []},
        "verification_status": "PASS",
        "verification_report": {"status": "PASS"},
        "final_decision": {"status": "PASS"},
    }

    payload = format_to_json(report)["payload"]

    assert [item["endpoint"] for item in payload["admet_profile"]] == [
        "label_DILI_t",
        "label_hERG",
        "label_from_attribution",
    ]
    assert [item["ade_id"] for item in payload["known_ade_profile"]] == ["heart-1"]
    assert [item["chain_id"] for item in payload["mechanism_chains"]] == ["liver-chain"]
    assert [item["ade_id"] for item in payload["persade_contextual_evidence"]] == ["heart-1"]
    assert payload["patient_features"] == {"patient_id": "demo"}
    assert payload["universal_toxicity_report"] == report["universal_report"]
