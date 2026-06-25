"""Randomly sample admetSAR molecules and write cardiac attribution text.

This batch script keeps the runtime path separate from ``main.py``. Each sampled
molecule is passed to the agents as name + SMILES only; DrugBank IDs and
InChIKeys are left for the local tools to resolve internally.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agents.brain_agent as brain_module
from agents.brain_agent import BrainAgent
from agents.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from config import get_model_config
from models.schemas import DrugInfo, GeneralToxicityItem, PatientInfo
from tool.common import ADMETSAR, resolve_drug


CARDIAC_SOC = "Cardiac disorders"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "20_drug_analysis.md"


def _read_admetsar_smiles(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        handle.readline()
        smiles = []
        for line in handle:
            value = line.split("\t", 1)[0].strip()
            if value:
                smiles.append(value)
    return smiles


def sample_named_admetsar_drugs(
    *,
    count: int,
    seed: int,
    max_attempts: int,
    source_path: Path = ADMETSAR,
) -> list[dict[str, str]]:
    smiles_values = _read_admetsar_smiles(source_path)
    rng = random.Random(seed)
    rng.shuffle(smiles_values)

    samples: list[dict[str, str]] = []
    seen_names: set[str] = set()
    seen_smiles: set[str] = set()
    attempts = 0

    for smiles in smiles_values:
        if len(samples) >= count:
            break
        attempts += 1
        if attempts > max_attempts:
            break
        if smiles in seen_smiles:
            continue

        entity = resolve_drug({"smiles": smiles})
        name = str(entity.get("name") or "").strip()
        if not name or name == smiles:
            continue

        normalized_name = name.lower()
        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        seen_smiles.add(smiles)
        samples.append({"name": name, "smiles": smiles})

    if len(samples) < count:
        raise RuntimeError(
            f"Only found {len(samples)} named admetSAR molecules after {attempts} attempts; "
            f"requested {count}. Increase --max-attempts or use another --seed."
        )
    return samples


def _cardiac_item(items: list[GeneralToxicityItem]) -> GeneralToxicityItem:
    for item in items:
        if item.soc == CARDIAC_SOC:
            return item
    raise ValueError(f"{CARDIAC_SOC} row was not produced.")


def _narrative_retry_context(
    *,
    drug: DrugInfo,
    cardiac: GeneralToxicityItem,
) -> dict[str, Any]:
    return {
        "drug": {
            "name": drug.name,
            "smiles": drug.smiles,
        },
        "organ_system": "heart",
        "soc": cardiac.soc,
        "baseline_risk": {
            "risk_level": cardiac.baseline_risk_level,
            "probability": cardiac.baseline_probability,
            "uncertainty": cardiac.uncertainty,
        },
        "probability_audit": {
            "main_drivers": [
                driver.get("driver")
                for driver in cardiac.attribution.molecular_attribution
                if isinstance(driver, dict) and driver.get("driver")
            ],
            "evidence_summary": [
                ref.get("summary")
                for driver in cardiac.attribution.molecular_attribution
                if isinstance(driver, dict)
                for ref in driver.get("evidence_refs", [])
                if isinstance(ref, dict) and ref.get("summary")
            ],
        },
        "method_availability": {
            "structural_alert_matching": bool(cardiac.attribution.structural),
            "smarts": any(bool(item.smarts) for item in cardiac.attribution.structural),
            "gnn_attention": False,
            "shap": False,
        },
    }


def _narrative_retry_payload(cardiac: GeneralToxicityItem) -> dict[str, Any]:
    return {
        "attribution_explanation": cardiac.attribution.attribution_explanation,
        "molecular_attribution": cardiac.attribution.molecular_attribution,
        "attribution_limitations": cardiac.attribution.attribution_limitations,
    }


def _retry_attribution_narrative(
    *,
    brain: BrainAgent,
    drug: DrugInfo,
    cardiac: GeneralToxicityItem,
    retries: int,
    sleep_seconds: float,
) -> str:
    narrative = cardiac.attribution.attribution_narrative or ""
    context = _narrative_retry_context(drug=drug, cardiac=cardiac)
    attribution = _narrative_retry_payload(cardiac)
    for attempt in range(max(0, retries)):
        if narrative:
            break
        if attempt:
            time.sleep(sleep_seconds)
        narrative = brain._generate_attribution_narrative_with_llm(context, attribution) or ""  # type: ignore[attr-defined]
    return narrative


def run_one_drug(
    index: int,
    sample: dict[str, str],
    *,
    narrative_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, str]:
    patient = PatientInfo(patient_id=f"admetsar-20-analysis-{index:02d}", age=60, sex="unknown")
    drug = DrugInfo(
        name=sample["name"],
        smiles=sample["smiles"],
        dose="unspecified",
        route="unspecified",
    )

    retriever = KnowledgeRetrievalAgent()
    brain = BrainAgent()
    evidence = retriever.retrieve(
        query={
            "purpose": "universal_toxicity",
            "scope": "admetsar_random_20_cardiac",
            "organ_system": "heart",
        },
        patient_info=patient,
        drug_info=drug,
    )
    universal = brain.build_universal_report(patient, drug, evidence)
    cardiac = _cardiac_item(universal.general_toxicity)
    attribution = cardiac.attribution
    narrative = _retry_attribution_narrative(
        brain=brain,
        drug=drug,
        cardiac=cardiac,
        retries=narrative_retries,
        sleep_seconds=retry_sleep_seconds,
    )

    return {
        "index": str(index),
        "name": sample["name"],
        "smiles": sample["smiles"],
        "attribution_explanation": attribution.attribution_explanation or "",
        "attribution_narrative": narrative,
    }


def run_analysis(
    *,
    count: int,
    seed: int,
    max_attempts: int,
    workers: int,
    narrative_retries: int,
    retry_sleep_seconds: float,
) -> list[dict[str, str]]:
    samples = sample_named_admetsar_drugs(count=count, seed=seed, max_attempts=max_attempts)
    print(f"Sampled {len(samples)} named molecules from {ADMETSAR}.")

    original_modeled_organs = set(brain_module.MODELED_ORGANS)
    brain_module.MODELED_ORGANS.clear()
    brain_module.MODELED_ORGANS.add("heart")

    results: list[dict[str, str] | None] = [None] * len(samples)
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_index = {
                executor.submit(
                    run_one_drug,
                    index,
                    sample,
                    narrative_retries=narrative_retries,
                    retry_sleep_seconds=retry_sleep_seconds,
                ): index - 1
                for index, sample in enumerate(samples, start=1)
            }
            for completed, future in enumerate(as_completed(future_to_index), start=1):
                result_index = future_to_index[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - report the exact failed molecule.
                    sample = samples[result_index]
                    raise RuntimeError(
                        f"Failed while analyzing {sample['name']} ({sample['smiles']}): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                results[result_index] = row
                print(f"[{completed}/{len(samples)}] Completed {row['name']}")
    finally:
        brain_module.MODELED_ORGANS.clear()
        brain_module.MODELED_ORGANS.update(original_modeled_organs)

    return [row for row in results if row is not None]


def write_markdown(rows: list[dict[str, str]], output_path: Path, *, include_smiles: bool = False) -> None:
    lines = ["# 20 Drug Cardiac Attribution Analysis", ""]
    for row in rows:
        section = [
            f"## {row['index']}. {row['name']}",
            "",
        ]
        if include_smiles:
            section.extend(
                [
                    f"SMILES: `{row['smiles']}`",
                    "",
                ]
            )
        section.extend(
            [
                "Attribution Explanation:",
                "",
                row["attribution_explanation"] or "No cardiac attribution explanation available.",
                "",
                "Attribution Narrative:",
                "",
                row["attribution_narrative"] or "No cardiac attribution narrative available.",
                "",
            ]
        )
        lines.extend(
            section
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8-sig")


def _parse_args() -> argparse.Namespace:
    config = get_model_config()
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample named molecules from admetSAR3 and write cardiac "
            "Attribution Explanation/Narrative to a Markdown report."
        )
    )
    parser.add_argument("--count", type=int, default=20, help="Number of drugs to analyze.")
    parser.add_argument("--seed", type=int, default=20260624, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5000,
        help="Maximum shuffled admetSAR rows to try while finding named molecules.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, config.attribution_parallelism),
        help="Number of drugs to process concurrently.",
    )
    parser.add_argument(
        "--narrative-retries",
        type=int,
        default=3,
        help="Extra narrative-only LLM attempts when the first BrainAgent pass returns no narrative.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep between extra narrative-only retry attempts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown report path.",
    )
    parser.add_argument(
        "--include-smiles",
        action="store_true",
        help="Include each sampled input SMILES in the Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_analysis(
        count=args.count,
        seed=args.seed,
        max_attempts=args.max_attempts,
        workers=args.workers,
        narrative_retries=args.narrative_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    write_markdown(rows, args.output, include_smiles=args.include_smiles)
    narrative_count = sum(1 for row in rows if row["attribution_narrative"])
    print("\n=== 20 Drug Analysis Complete ===")
    print(f"Rows: {len(rows)}")
    print(f"Rows with Attribution Narrative: {narrative_count}")
    print(f"Markdown report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
