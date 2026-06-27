"""Randomly sample admetSAR molecules and write cardiac attribution explanations.

This batch script keeps the runtime path separate from ``main.py``. Each sampled
molecule is passed to the agents as name + SMILES only; DrugBank IDs and
InChIKeys are left for the local tools to resolve internally.
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pertox_agent.agents.toxicity_orchestrator_agent as orchestrator_module
from pertox_agent.agents.toxicity_orchestrator_agent import ToxicityOrchestratorAgent
from pertox_agent.agents.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from pertox_agent.settings import get_model_config
from pertox_agent.schemas import DrugInfo, GeneralToxicityItem, PatientInfo
from pertox_agent.tools.shared.common import ADMETSAR, resolve_drug


CARDIAC_SOC = "Cardiac disorders"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "20_drug_analysis.md"


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


def run_one_drug(index: int, sample: dict[str, str]) -> dict[str, str]:
    patient = PatientInfo(patient_id=f"admetsar-20-analysis-{index:02d}", age=60, sex="unknown")
    drug = DrugInfo(
        name=sample["name"],
        smiles=sample["smiles"],
        dose="unspecified",
        route="unspecified",
    )

    retriever = KnowledgeRetrievalAgent()
    orchestrator = ToxicityOrchestratorAgent()
    evidence = retriever.retrieve(
        query={
            "purpose": "universal_toxicity",
            "scope": "admetsar_random_20_cardiac",
            "organ_system": "heart",
        },
        patient_info=patient,
        drug_info=drug,
    )
    universal = orchestrator.build_universal_report(patient, drug, evidence)
    cardiac = _cardiac_item(universal.general_toxicity)
    attribution = cardiac.attribution

    return {
        "index": str(index),
        "name": sample["name"],
        "smiles": sample["smiles"],
        "attribution_explanation": attribution.attribution_explanation or "",
    }


def run_analysis(
    *,
    count: int,
    seed: int,
    max_attempts: int,
    workers: int,
) -> list[dict[str, str]]:
    samples = sample_named_admetsar_drugs(count=count, seed=seed, max_attempts=max_attempts)
    print(f"Sampled {len(samples)} named molecules from {ADMETSAR}.")

    original_modeled_organs = set(orchestrator_module.MODELED_ORGANS)
    orchestrator_module.MODELED_ORGANS.clear()
    orchestrator_module.MODELED_ORGANS.add("heart")

    results: list[dict[str, str] | None] = [None] * len(samples)
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_index = {
                executor.submit(
                    run_one_drug,
                    index,
                    sample,
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
        orchestrator_module.MODELED_ORGANS.clear()
        orchestrator_module.MODELED_ORGANS.update(original_modeled_organs)

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
            "Attribution Explanation to a Markdown report."
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
    )
    write_markdown(rows, args.output, include_smiles=args.include_smiles)
    print("\n=== 20 Drug Analysis Complete ===")
    print(f"Rows: {len(rows)}")
    print(f"Markdown report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
