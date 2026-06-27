"""Run the warfarin personalized toxicity example."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pertox_agent.app.runner import run_persagent_report


def build_warfarin_state() -> dict:
    return {
        "raw_patient_info": {
            "patient_id": "demo-warfarin-001",
            "age": 65,
            "sex": "female",
            "weight_kg": 58,
            "alt_u_l": 68,
            "ast_u_l": 74,
            "bilirubin_mg_dl": 1.8,
            "child_pugh": "B",
            "creatinine_mg_dl": 1.3,
            "egfr_ml_min": 45,
            "genotypes": {"CYP2C9": "*2/*3"},
            "hla_types": [],
            "medical_history": ["K74.6 cirrhosis", "I48 atrial fibrillation"],
            "concomitant_medications": ["amiodarone"],
            "organ_function": {"LVEF": "55%"},
            "exposure": {"route": "oral", "frequency": "daily"},
            "pregnancy_status": "not_pregnant",
        },
        "raw_drug_info": {
            "name": "warfarin",
            "drugbank_id": "DB00682",
            "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
            # "dose": "5 mg/day",
            # "route": "oral",
            # "frequency": "daily",
            # "known_toxicities": ["bleeding", "skin necrosis"],
        },
        "messages": [],
        "trace": [],
    }


def main() -> None:
    output_path = PROJECT_ROOT / "results" / "final_report_warfarin.json"
    run_persagent_report(build_warfarin_state(), output_path)


if __name__ == "__main__":
    main()

