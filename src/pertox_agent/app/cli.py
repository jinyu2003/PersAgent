"""Command-line interface for running PersAgent from structured input files.

Supported text format:

    [patient]
    patient_id: case-warfarin-001
    age: 65
    sex: female
    weight_kg: 58
    alt_u_l: 68
    ast_u_l: 74
    bilirubin_mg_dl: 1.8
    child_pugh: B
    creatinine_mg_dl: 1.3
    egfr_ml_min: 45
    genotypes.CYP2C9: *2/*3
    medical_history: K74.6 cirrhosis; I48 atrial fibrillation
    concomitant_medications: amiodarone
    organ_function.LVEF: 55%
    exposure.route: oral
    exposure.frequency: daily
    pregnancy_status: not_pregnant

    [drug]
    name: warfarin
    drugbank_id: DB00682
    smiles: CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O
    dose: 5 mg/day
    route: oral
    frequency: daily

Supported JSON format:

    {
      "raw_patient_info": { "...": "..." },
      "raw_drug_info": { "...": "..." }
    }
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pertox_agent.app.runner import run_persagent_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LIST_FIELDS = {
    "medical_history",
    "concomitant_medications",
    "hla_types",
    "missing_modalities",
    "known_toxicities",
}


def load_state_from_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object.")

    patient = payload.get("raw_patient_info") or payload.get("patient") or payload.get("patient_info")
    drug = payload.get("raw_drug_info") or payload.get("drug") or payload.get("drug_info")
    if not isinstance(patient, dict) or not isinstance(drug, dict):
        raise ValueError("JSON input must contain raw_patient_info and raw_drug_info objects.")

    return _state(patient, drug)


def load_state_from_text(path: Path) -> dict[str, Any]:
    sections: dict[str, dict[str, Any]] = {"patient": {}, "drug": {}}
    current_section: str | None = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        section_match = re.fullmatch(r"\[(patient|drug)\]", line, flags=re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1).lower()
            continue

        if current_section is None:
            raise ValueError(f"Line {line_number}: expected [patient] or [drug] section before fields.")

        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            raise ValueError(f"Line {line_number}: expected 'key: value'.")

        _assign_field(sections[current_section], key.strip(), value.strip())

    return _state(sections["patient"], sections["drug"])


def _state(patient: dict[str, Any], drug: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_patient_info": patient,
        "raw_drug_info": drug,
        "messages": [],
        "trace": [],
    }


def _assign_field(target: dict[str, Any], key: str, raw_value: str) -> None:
    if not key:
        raise ValueError("Empty field name in text input.")

    parts = [part.strip() for part in key.split(".") if part.strip()]
    if not parts:
        raise ValueError("Empty field name in text input.")

    value = _parse_field_value(parts[0], raw_value)
    cursor = target
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            raise ValueError(f"Cannot assign nested field under non-object key '{part}'.")
        cursor = nested
    cursor[parts[-1]] = value


def _parse_field_value(field: str, value: str) -> Any:
    if field in LIST_FIELDS:
        return [item.strip() for item in re.split(r"[;|,]", value) if item.strip()]

    lowered = value.lower()
    if lowered in {"null", "none", "na", "n/a", ""}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?\d+\.\d+", value):
        return float(value)
    return value


def _default_output_path(state: dict[str, Any]) -> Path:
    drug = state.get("raw_drug_info", {})
    name = str(drug.get("name") or drug.get("drug_name") or drug.get("drug") or "drug")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_") or "drug"
    return PROJECT_ROOT / "results" / f"final_report_{slug}.json"


def _load_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.text:
        return load_state_from_text(args.text)
    if args.json:
        return load_state_from_json(args.json)
    raise ValueError("Use either --text or --json.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PersAgent from structured text or JSON input.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=Path, help="Structured text input file with [patient] and [drug] sections.")
    input_group.add_argument("--json", type=Path, help="JSON input file containing raw_patient_info and raw_drug_info.")
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults to results/final_report_<drug>.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    state = _load_state(args)
    output_path = args.output or _default_output_path(state)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    run_persagent_report(state, output_path)
    return 0
