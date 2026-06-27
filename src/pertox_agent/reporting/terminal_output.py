"""Shared terminal and JSON report output helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from pertox_agent.reporting.writer import ensure_final_output, write_final_report


STREAM_TARGET_SOCS = {"Hepatobiliary disorders", "Cardiac disorders"}


def stream_report(graph: Any, initial_state: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """Run a compiled graph and print report sections as each node finishes."""
    final_state = dict(initial_state)
    rendered_final_output = False

    for node_name, update in _stream_graph_updates(graph, initial_state):
        if isinstance(update, dict):
            final_state.update(update)
        rendered_final_output = _render_stream_update(node_name, final_state, output_path) or rendered_final_output

    if not rendered_final_output and "draft_report" in final_state:
        _render_final_output(final_state, output_path)

    _print_trace(final_state)
    return final_state


def print_report_summary(final_state: dict[str, Any], output_path: Path) -> None:
    final_output = ensure_final_output(final_state)
    schema_payload = final_output["json"]["payload"]

    print("\n=== PersAgent Trace ===")
    for item in final_state.get("trace", []):
        print(f"- {item}")

    evidence = final_state["evidence_package"]
    print("\n=== Knowledge Retrieval ===")
    print(f"Evidence package purpose: {evidence.query_purpose}")
    print(f"Tools called: {', '.join(evidence.tool_results.keys())}")
    print(f"Evidence items: {len(evidence.evidence_items)}")
    if evidence.conflicts:
        print("Conflicts:")
        for conflict in evidence.conflicts:
            print(f"- {conflict}")
    else:
        print("Conflicts: none")

    universal = final_state["universal_report"]
    print("\n=== Stage 1 Universal Toxicity ===")
    print(f"Drug: {universal.drug.name} ({universal.drug.drugbank_id})")
    _print_stage1_attribution_cards(schema_payload.get("universal_attribution_cards", []))

    personalized = final_state["personalized_report"]
    print("\n=== Stage 2 Personalized Toxicity ===")
    for item in personalized.personalized_toxicity:
        print(
            f"- {item.soc}: baseline={_fmt_nullable_float(item.baseline.probability)}, "
            f"personalized={_fmt_nullable_float(item.personalized_probability)}, "
            f"shift={_fmt_nullable_float(item.risk_shift, signed=True)}, "
            f"CTCAE grade={_fmt_nullable_text(item.ctcae_grade_predicted)}, "
            f"modifiers={len(item.patient_attribution)}"
        )

    print("\n=== Recommendations ===")
    seen_recommendations = set()
    for item in personalized.personalized_toxicity:
        rec = item.clinical_recommendation
        if rec is None:
            continue
        if rec.text in seen_recommendations:
            continue
        print(f"- [{rec.action}] {rec.text}")
        seen_recommendations.add(rec.text)

    verification = final_state["verification_report"]
    print("\n=== Verification ===")
    print(f"Status: {verification.status}")
    print(f"Summary: {verification.summary}")
    for issue in verification.issues:
        print(f"- L{issue.layer} {issue.severity} {issue.code}: {issue.message}")

    write_final_report(final_state, output_path)

    print("\n=== Final Output ===")
    print(f"JSON report: {output_path.resolve()}")
    print(f"JSON top-level keys: {', '.join(schema_payload.keys())}")
    print(
        "General toxicity rows: "
        f"{len(schema_payload['universal_toxicity_report']['general_toxicity'])}"
    )
    print(
        "Personalized toxicity rows: "
        f"{len(schema_payload['personalized_toxicity_report']['personalized_toxicity'])}"
    )


def _stream_graph_updates(graph: Any, initial_state: dict[str, Any]):
    try:
        yield from _iter_graph_stream(graph.stream(initial_state, stream_mode="updates"))
    except TypeError:
        yield from _iter_graph_stream(graph.stream(initial_state))


def _iter_graph_stream(stream: Any):
    for chunk in stream:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            chunk = chunk[1]
        if not isinstance(chunk, dict):
            continue
        if _looks_like_state_update(chunk):
            yield "state_update", chunk
            continue
        for node_name, update in chunk.items():
            yield str(node_name), update


def _looks_like_state_update(chunk: dict[str, Any]) -> bool:
    state_keys = {
        "raw_patient_info",
        "raw_drug_info",
        "patient_info",
        "drug_info",
        "universal_report",
        "personalized_report",
        "final_output",
        "trace",
    }
    return bool(state_keys.intersection(chunk.keys()))


def _render_stream_update(node_name: str, state: dict[str, Any], output_path: Path) -> bool:
    if node_name == "orchestrator_parse_input":
        _render_input_parsed(state)
    elif node_name == "orchestrator_stage1_plan_retrieval":
        _render_retrieval_plan("Stage 1 Retrieval Plan", state.get("stage1_retrieval_plan"))
    elif node_name == "knowledge_retrieval_node":
        _render_knowledge_retrieved(state)
    elif node_name == "orchestrator_stage1_reason":
        _render_universal_toxicity(state)
        _render_stage1_attribution(state)
    elif node_name == "orchestrator_standardize_patient":
        _render_patient_features(state)
    elif node_name == "orchestrator_stage2_plan_retrieval":
        _render_retrieval_plan("Stage 2 Retrieval Plan", state.get("stage2_retrieval_plan"))
    elif node_name == "orchestrator_stage2_reason":
        _render_personalized_toxicity(state)
        _render_recommendations(state)
    elif node_name == "safety_verifier_node":
        _render_verification(state)
    elif node_name == "orchestrator_revise_output":
        _render_final_decision(state)
    elif node_name == "format_output":
        _render_final_output(state, output_path)
        return True
    return False


def _render_input_parsed(state: dict[str, Any]) -> None:
    patient = state.get("patient_info")
    drug = state.get("drug_info")
    print("\n=== Input Parsed ===")
    print(f"Patient: {_get_value(patient, 'patient_id')}")
    print(
        "Patient context: "
        f"age={_fmt_nullable_text(_get_value(patient, 'age'))}, "
        f"sex={_fmt_nullable_text(_get_value(patient, 'sex'))}"
    )
    print(
        "Drug: "
        f"{_fmt_nullable_text(_get_value(drug, 'name'))} "
        f"({_fmt_nullable_text(_get_value(drug, 'drugbank_id'))})"
    )
    print(
        "Exposure: "
        f"dose={_fmt_nullable_text(_get_value(drug, 'dose'))}, "
        f"route={_fmt_nullable_text(_get_value(drug, 'route'))}, "
        f"frequency={_fmt_nullable_text(_get_value(drug, 'frequency'))}"
    )


def _render_retrieval_plan(title: str, plan: Any) -> None:
    plan = plan or {}
    query = _get_value(plan, "query", {}) or {}
    print(f"\n=== {title} ===")
    print(f"Goal: {_fmt_nullable_text(_get_value(plan, 'goal'))}")
    needs = _get_value(query, "needs", []) or []
    if needs:
        print("Needs:")
        for need in needs:
            print(f"- {need}")


def _render_knowledge_retrieved(state: dict[str, Any]) -> None:
    evidence = state.get("latest_evidence_package")
    purpose = _get_value(evidence, "query_purpose") or _get_value(_get_value(evidence, "query", {}), "purpose")
    title = "Stage 2 Knowledge Retrieved" if purpose == "personalized_modifiers" else "Stage 1 Knowledge Retrieved"
    tool_results = _get_value(evidence, "tool_results", {}) or {}
    print(f"\n=== {title} ===")
    print(f"Evidence package purpose: {_fmt_nullable_text(purpose)}")
    print(f"Tools called: {', '.join(tool_results.keys()) if tool_results else 'none'}")
    print(f"Evidence items: {len(_get_value(evidence, 'evidence_items', []) or [])}")
    conflicts = _get_value(evidence, "conflicts", []) or []
    print(f"Conflicts: {len(conflicts)}")
    if purpose == "universal_toxicity":
        _print_stage1_tool_counts(tool_results)


def _print_stage1_tool_counts(tool_results: dict[str, Any]) -> None:
    admet = _get_value(tool_results.get("admetsar_predict", {}), "admet_profile", []) or []
    known = _get_value(tool_results.get("persade_drug_profile", {}), "known_ade_profile", []) or []
    chains = _get_value(tool_results.get("mechanism_chains_lookup", {}), "mechanism_chains", []) or []
    context = _get_value(tool_results.get("persade_subgroup_scores", {}), "persade_contextual_evidence", []) or []
    print(
        "Raw evidence rows: "
        f"admet={len(admet)}, known_ade={len(known)}, "
        f"mechanism_chains={len(chains)}, persade_context={len(context)}"
    )


def _render_universal_toxicity(state: dict[str, Any]) -> None:
    universal = state.get("universal_report")
    print("\n=== Stage 1 Universal Toxicity ===")
    print(f"Drug: {_fmt_nullable_text(_get_value(_get_value(universal, 'drug'), 'name'))}")
    for item in _target_toxicity_items(_get_value(universal, "general_toxicity", []) or []):
        print(
            f"- {_get_value(item, 'soc')}: "
            f"baseline={_fmt_nullable_text(_get_value(item, 'baseline_risk_level'))}, "
            f"probability={_fmt_nullable_float(_get_value(item, 'baseline_probability'))}, "
            f"uncertainty={_fmt_nullable_float(_get_value(item, 'uncertainty'))}, "
            f"CTCAE grade={_fmt_nullable_text(_get_value(item, 'ctcae_grade_predicted'))}"
        )


def _render_stage1_attribution(state: dict[str, Any]) -> None:
    from pertox_agent.reporting.formatter import build_universal_attribution_cards

    cards = build_universal_attribution_cards({"universal_report": state.get("universal_report")})
    _print_stage1_attribution_cards(_target_cards(cards))


def _render_patient_features(state: dict[str, Any]) -> None:
    features = state.get("patient_features")
    organ_classes = _get_value(features, "organ_function_classes", {}) or {}
    print("\n=== Patient Standardized ===")
    print(
        "Patient features: "
        f"age_group={_fmt_nullable_text(_get_value(features, 'age_group'))}, "
        f"elderly={_fmt_nullable_text(_get_value(features, 'elderly'))}, "
        f"sex={_fmt_nullable_text(_get_value(features, 'sex'))}"
    )
    print(
        "Organ classes: "
        f"renal={_fmt_nullable_text(_get_value(organ_classes.get('renal'), 'klass'))}, "
        f"hepatic={_fmt_nullable_text(_get_value(organ_classes.get('hepatic'), 'klass'))}, "
        f"cardiac={_fmt_nullable_text(_get_value(organ_classes.get('cardiac'), 'klass'))}"
    )
    print(f"PGx phenotypes: {len(_get_value(features, 'pgx_phenotypes', []) or [])}")


def _render_personalized_toxicity(state: dict[str, Any]) -> None:
    personalized = state.get("personalized_report")
    print("\n=== Stage 2 Personalized Toxicity ===")
    for item in _target_toxicity_items(_get_value(personalized, "personalized_toxicity", []) or []):
        baseline = _get_value(item, "baseline")
        print(
            f"- {_get_value(item, 'soc')}: "
            f"baseline={_fmt_nullable_float(_get_value(baseline, 'probability'))}, "
            f"personalized={_fmt_nullable_float(_get_value(item, 'personalized_probability'))}, "
            f"shift={_fmt_nullable_float(_get_value(item, 'risk_shift'), signed=True)}, "
            f"CTCAE grade={_fmt_nullable_text(_get_value(item, 'ctcae_grade_predicted'))}, "
            f"modifiers={len(_get_value(item, 'patient_attribution', []) or [])}"
        )


def _render_recommendations(state: dict[str, Any]) -> None:
    personalized = state.get("personalized_report")
    print("\n=== Recommendations ===")
    seen_recommendations = set()
    for item in _get_value(personalized, "personalized_toxicity", []) or []:
        rec = _get_value(item, "clinical_recommendation")
        text = _get_value(rec, "text")
        if not text or text in seen_recommendations:
            continue
        print(f"- [{_get_value(rec, 'action')}] {text}")
        seen_recommendations.add(text)
    if not seen_recommendations:
        print("- No clinical recommendations generated.")


def _render_verification(state: dict[str, Any]) -> None:
    verification = state.get("verification_report")
    print("\n=== Verification ===")
    print(f"Status: {_fmt_nullable_text(_get_value(verification, 'status'))}")
    print(f"Summary: {_fmt_nullable_text(_get_value(verification, 'summary'))}")
    for issue in _get_value(verification, "issues", []) or []:
        print(
            f"- L{_get_value(issue, 'layer')} "
            f"{_get_value(issue, 'severity')} {_get_value(issue, 'code')}: "
            f"{_get_value(issue, 'message')}"
        )


def _render_final_decision(state: dict[str, Any]) -> None:
    decision = _get_value(state.get("draft_report", {}), "final_decision", {}) or {}
    print("\n=== Final Decision ===")
    print(f"Status: {_fmt_nullable_text(_get_value(decision, 'status'))}")
    print(f"Message: {_fmt_nullable_text(_get_value(decision, 'message'))}")


def _render_final_output(state: dict[str, Any], output_path: Path) -> None:
    final_output = ensure_final_output(state)
    schema_payload = final_output["json"]["payload"]
    write_final_report(state, output_path)
    print("\n=== Final Output ===")
    print(f"JSON report: {output_path.resolve()}")
    print(f"JSON top-level keys: {', '.join(schema_payload.keys())}")
    print(
        "General toxicity rows: "
        f"{len(schema_payload['universal_toxicity_report']['general_toxicity'])}"
    )
    print(
        "Personalized toxicity rows: "
        f"{len(schema_payload['personalized_toxicity_report']['personalized_toxicity'])}"
    )
    print(
        "Filtered evidence rows: "
        f"admet={len(schema_payload.get('admet_profile') or [])}, "
        f"known_ade={len(schema_payload.get('known_ade_profile') or [])}, "
        f"mechanism_chains={len(schema_payload.get('mechanism_chains') or [])}, "
        f"persade_context={len(schema_payload.get('persade_contextual_evidence') or [])}"
    )


def _print_trace(state: dict[str, Any]) -> None:
    print("\n=== PersAgent Trace ===")
    for item in state.get("trace", []):
        print(f"- {item}")


def _target_toxicity_items(items: Any) -> list[Any]:
    return [item for item in items if _get_value(item, "soc") in STREAM_TARGET_SOCS]


def _target_cards(cards: list[dict]) -> list[dict]:
    return [card for card in cards if card.get("soc") in STREAM_TARGET_SOCS]


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _fmt_nullable_float(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "null"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _fmt_nullable_text(value: object) -> str:
    return "null" if value is None else str(value)


def _clip_text(value: object, limit: int = 150) -> str:
    text = _fmt_nullable_text(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _print_wrapped(label: str, value: object, *, indent: str = "  ", width: int = 100) -> None:
    text = _fmt_nullable_text(value).replace("\n", " ").strip()
    print(f"{indent}{label}:")
    for line in textwrap.wrap(text, width=width) or ["null"]:
        print(f"{indent}  {line}")


def _impact_label(value: object) -> str:
    labels = {
        "direct_probability_support": "probability",
        "mechanistic_context_only": "context",
        "uncertainty_or_gap": "uncertainty",
    }
    return labels.get(str(value or ""), "context")


def _card_generation_method(cards: list[dict]) -> str:
    methods = {card.get("generation_method") for card in cards if card.get("generation_method")}
    if not methods:
        return "none"
    return next(iter(methods)) if len(methods) == 1 else "mixed"


def _print_card_driver_group(title: str, drivers: list[dict], empty_text: str) -> None:
    print(f"  {title}")
    if not drivers:
        print(f"    - {empty_text}")
        return
    for index, driver in enumerate(drivers[:5], start=1):
        prefix = f"{index}." if title == "Probability drivers" else "-"
        print(f"    {prefix} {_fmt_nullable_text(driver.get('label', 'retrieved evidence'))}")
        if "support" in driver:
            detail = (
                f"direction={driver.get('direction', 'unknown')} | "
                f"support={driver.get('support', 'n/a')} | "
                f"ref={driver.get('ref') or '[no trace ref]'}"
            )
        else:
            refs = driver.get("refs") or []
            detail = (
                f"direction={driver.get('direction', 'unknown')} | "
                f"confidence={driver.get('confidence', 'n/a')} | "
                f"ref={refs[0] if refs else '[no trace ref]'}"
            )
        print(f"       {detail}")
        if title != "Probability drivers":
            interpretation = driver.get("interpretation")
            if interpretation:
                print(f"       {interpretation}")
            role = driver.get("mechanistic_role") or driver.get("limitations")
            if role:
                print(f"       {_clip_text(role, 130)}")


def _print_stage1_attribution_cards(cards: list[dict]) -> None:
    print(f"\n=== Stage 1 Universal Toxicity Attribution [{_card_generation_method(cards)}] ===")
    for index, card in enumerate(cards, start=1):
        print(
            f"\n[{index}] {card.get('soc')} | "
            f"organ={card.get('organ_system', 'unknown')} | "
            f"modeled={str(bool(card.get('modeled'))).lower()}"
        )
        if not card.get("modeled"):
            print(f"  {card.get('not_modeled_reason', 'Not actively modeled in Stage 1.')}")
            continue

        risk = card.get("risk_judgement", {})
        print("Risk judgement")
        print(f"  Baseline risk : {_fmt_nullable_text(risk.get('baseline_risk_level')).upper()}")
        print(f"  Probability   : p={_fmt_nullable_float(risk.get('baseline_probability'))}")
        print(f"  Uncertainty   : {_fmt_nullable_float(risk.get('uncertainty'))}")
        _print_wrapped("Judgement", risk.get("judgement"))
        if risk.get("uncertainty_summary"):
            _print_wrapped("Uncertainty summary", risk.get("uncertainty_summary"))
        uncertainty_drivers = risk.get("uncertainty_drivers") or []
        if uncertainty_drivers:
            print("  Uncertainty drivers:")
            for driver in uncertainty_drivers[:5]:
                for line_index, line in enumerate(textwrap.wrap(str(driver), width=98) or [""]):
                    prefix = "    - " if line_index == 0 else "      "
                    print(f"{prefix}{line}")

        print("\nEvidence chain")
        chain = card.get("evidence_chain", {})
        print(f"  Chain type    : {chain.get('chain_type', 'evidence_assembled_chain')}")
        if chain.get("mechanism_summary"):
            _print_wrapped("Mechanism summary", chain.get("mechanism_summary"))
        print("  Readable path :")
        nodes = chain.get("nodes") or []
        if nodes:
            for offset, node in enumerate(nodes):
                prefix = "    " if offset == 0 else "      -> "
                print(f"{prefix}{_clip_text(node.get('line'), 130)}")
                if node.get("impact_note"):
                    _print_wrapped(
                        f"Impact [{_impact_label(node.get('impact_type'))}]",
                        node.get("impact_note"),
                        indent="         ",
                        width=92,
                    )
            for line in chain.get("readable_path") or []:
                if str(line).startswith("SOC:"):
                    print(f"      -> {line}")
        else:
            for offset, line in enumerate(chain.get("readable_path") or []):
                prefix = "    " if offset == 0 else "      -> "
                print(f"{prefix}{_clip_text(line, 130)}")

        driver_layers = card.get("driver_layers", {})
        print("\nDriver layers")
        _print_card_driver_group(
            "Probability drivers",
            driver_layers.get("probability_drivers") or [],
            "No explicit probability driver returned for this SOC.",
        )
        _print_card_driver_group(
            "Mechanistic context",
            driver_layers.get("mechanistic_context") or [],
            "No separate mechanistic context driver returned.",
        )
        _print_card_driver_group(
            "Population context",
            driver_layers.get("population_context") or [],
            "No population-signal context returned.",
        )

        print("\nUncertainty & traceability")
        uncertainty = card.get("uncertainty_and_traceability", {})
        print("  Limitations")
        for limitation in (uncertainty.get("limitations") or ["No additional attribution limitations reported."])[:5]:
            print(f"    - {_clip_text(limitation, 130)}")
        print("  Trace refs")
        for ref in (uncertainty.get("trace_refs") or ["[no trace ref]"])[:6]:
            print(f"    - {ref}")
