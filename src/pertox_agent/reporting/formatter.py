"""Runtime formatting helpers for PersAgent reports."""


from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pertox_agent.reporting.filters import (
    filter_admet_profile,
    filter_known_ade_profile,
    filter_mechanism_chains,
    filter_persade_contextual_evidence,
    target_ade_scope,
)


def to_plain_dict(value: Any) -> Any:
    """Convert Pydantic models and datetimes into JSON-ready Python values."""
    if hasattr(value, "model_dump"):
        return to_plain_dict(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return to_plain_dict(value.dict())
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def format_to_json(report: Dict[str, Any]) -> Dict[str, Any]:
    """Format the agent draft into the public JSON payload shape."""
    universal_attribution_cards = build_universal_attribution_cards(report)
    report_payload = to_plain_dict(report)
    known_ade_profile = filter_known_ade_profile(report_payload.get("known_ade_profile"))
    ade_scope = target_ade_scope(report_payload.get("known_ade_profile"))
    payload = {
        "drug_entity": report_payload.get("drug_entity"),
        "patient_features": report_payload.get("patient_features"),
        "structure_profile": report_payload.get("structure_profile"),
        "admet_profile": filter_admet_profile(
            report_payload.get("admet_profile"),
            report_payload.get("universal_report"),
        ),
        "known_ade_profile": known_ade_profile,
        "mechanism_chains": filter_mechanism_chains(report_payload.get("mechanism_chains")),
        "persade_contextual_evidence": filter_persade_contextual_evidence(
            report_payload.get("persade_contextual_evidence"),
            ade_scope,
        ),
        "baseline_organ_risk": report_payload.get("baseline_organ_risk"),
        "attribution_explanations": report_payload.get("attribution_explanations"),
        "universal_attribution_cards": universal_attribution_cards,
        "universal_toxicity_report": report_payload.get("universal_report"),
        "personalized_toxicity_report": report_payload.get("personalized_report"),
        "verification_status": report_payload.get("verification_status"),
        "verification_report": report_payload.get("verification_report"),
        "final_decision": report_payload.get("final_decision"),
    }
    return {
        "content_type": "application/json",
        "payload": payload,
        "json": json.dumps(payload, ensure_ascii=False, indent=2),
    }


ORGAN_BY_SOC = {
    "Hepatobiliary disorders": "liver",
    "Cardiac disorders": "heart",
    "Renal and urinary disorders": "kidney",
    "Blood and lymphatic system disorders": "hematologic",
    "Immune system disorders": "immune",
    "Skin and subcutaneous tissue disorders": "skin",
    "Nervous system disorders": "neurologic",
    "Gastrointestinal disorders": "gastrointestinal",
}


def build_universal_attribution_cards(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build stage-1-only per-SOC attribution cards for JSON and terminal display."""
    universal_report = report.get("universal_report")
    toxicity_rows = _as_list(_get_value(universal_report, "general_toxicity", []))
    baseline_by_soc = {
        item.get("soc"): item
        for item in _as_list(report.get("baseline_organ_risk", []))
        if isinstance(item, dict)
    }
    summaries = report.get("stage1_card_summaries", {}) or {}
    cards: List[Dict[str, Any]] = []
    for item in toxicity_rows:
        soc = _get_value(item, "soc")
        attribution = _get_value(item, "attribution", {}) or {}
        probability = _get_value(item, "baseline_probability")
        if probability is None:
            cards.append(
                {
                    "soc": soc,
                    "organ_system": ORGAN_BY_SOC.get(soc, "unknown"),
                    "modeled": False,
                    "not_modeled_reason": "Not actively modeled in Stage 1; retained as null placeholder.",
                }
            )
            continue

        audit = baseline_by_soc.get(soc, {})
        llm_summary = summaries.get(soc, {}) if isinstance(summaries, dict) else {}
        molecular_drivers = _as_list(_get_value(attribution, "molecular_attribution", []))
        probability_drivers = _audit_probability_drivers(audit) or _fallback_probability_drivers(molecular_drivers)
        mechanism_context = [
            _render_driver(driver)
            for driver in molecular_drivers
            if _driver_role(driver) in {"mechanistic_context", "structural_context"}
        ][:5]
        population_context = [
            _render_driver(driver)
            for driver in molecular_drivers
            if _driver_role(driver) == "population_context"
        ][:5]
        limitations = _display_limitations(_as_list(_get_value(attribution, "attribution_limitations", [])))

        cards.append(
            {
                "soc": soc,
                "organ_system": ORGAN_BY_SOC.get(soc, "unknown"),
                "modeled": True,
                "generation_method": _get_value(attribution, "attribution_generation_method"),
                "risk_judgement": {
                    "baseline_risk_level": _get_value(item, "baseline_risk_level"),
                    "baseline_probability": probability,
                    "uncertainty": _get_value(item, "uncertainty"),
                    "judgement": llm_summary.get("judgement") or _template_judgement(audit, probability_drivers, attribution),
                    "uncertainty_summary": llm_summary.get("uncertainty_summary"),
                    "uncertainty_drivers": limitations[:6],
                },
                "evidence_chain": {
                    **_render_evidence_chain(
                        item,
                        attribution,
                        llm_summary.get("node_impact_notes", []),
                        probability_drivers,
                    ),
                    "mechanism_summary": llm_summary.get("mechanism_summary"),
                },
                "driver_layers": {
                    "probability_drivers": probability_drivers[:6],
                    "mechanistic_context": [
                        {
                            **driver,
                            "interpretation": "supports mechanism explanation; not counted as independent probability lift",
                        }
                        for driver in mechanism_context
                    ],
                    "population_context": [
                        {
                            **driver,
                            "interpretation": "statistical association; not direct causal proof",
                        }
                        for driver in population_context
                    ],
                },
                "uncertainty_and_traceability": {
                    "uncertainty": _get_value(item, "uncertainty"),
                    "limitations": limitations[:6],
                    "trace_refs": _trace_refs(item, probability_drivers),
                },
            }
        )
    return cards


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value) if isinstance(value, tuple) else [value]


def _display_limitations(limitations: List[Any]) -> List[str]:
    display: List[str] = []
    for limitation in limitations:
        text = str(limitation).strip()
        if not text:
            continue
        lower = text.lower()
        if "live llm returned" in lower or "local code wrapped" in lower:
            continue
        if "llm evidence path could not be verified" in lower:
            text = "Some LLM-selected evidence references could not be verified against tool outputs."
        display.append(text)
    return list(dict.fromkeys(display))


def _template_judgement(audit: Dict[str, Any], drivers: List[Dict[str, Any]], attribution: Any) -> str:
    main_drivers = _as_list(audit.get("main_drivers", []))
    if main_drivers:
        return "Baseline probability is supported by " + "; ".join(str(driver) for driver in main_drivers[:5]) + "."
    labels = [str(driver.get("label")) for driver in drivers[:4] if driver.get("label")]
    if labels:
        return "Baseline probability is supported by " + "; ".join(labels) + "."
    return _get_value(attribution, "attribution_explanation") or _get_value(attribution, "mechanism_summary") or ""


def _driver_role(driver: Any) -> str:
    role = str(_get_value(driver, "contribution_role") or "").strip().lower()
    if role in {"probability_driver", "mechanistic_context", "structural_context", "population_context"}:
        return role
    driver_type = str(_get_value(driver, "driver_type") or "").strip().lower()
    direction = str(_get_value(driver, "direction") or "").strip().lower()
    if driver_type == "population_signal":
        return "population_context"
    if direction in {"increase", "decrease", "up", "down"}:
        return "probability_driver"
    return "mechanistic_context"


def _audit_probability_drivers(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    drivers_by_label: Dict[str, Dict[str, Any]] = {}
    for evidence in _as_list(audit.get("evidence_summary", []))[:10]:
        if not isinstance(evidence, dict):
            continue
        label = str(evidence.get("detail") or evidence.get("source") or "baseline evidence")
        driver = {
            "label": label,
            "direction": "increase",
            "support": evidence.get("support"),
            "ref": _audit_ref_label(evidence),
            "source": evidence.get("source"),
            "evidence_type": evidence.get("evidence_type"),
        }
        existing = drivers_by_label.get(label)
        if existing is None:
            drivers_by_label[label] = driver
            continue
        existing_support = _float_or_none(existing.get("support")) or 0.0
        candidate_support = _float_or_none(driver.get("support")) or 0.0
        if "PMID" in str(driver.get("ref")) or candidate_support > existing_support:
            drivers_by_label[label] = driver
    return list(drivers_by_label.values())


def _fallback_probability_drivers(drivers: List[Any]) -> List[Dict[str, Any]]:
    return [_render_driver(driver) for driver in drivers if _driver_role(driver) == "probability_driver"][:6]


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _audit_ref_label(evidence: Dict[str, Any]) -> str:
    source = evidence.get("source") or "baseline_fusion"
    evidence_type = evidence.get("evidence_type") or "evidence"
    detail = evidence.get("detail") or "detail"
    pubmed = _as_list(evidence.get("pubmed", []))
    parts = [_clip_text(source, 42), _clip_text(evidence_type, 32), _clip_text(detail, 86)]
    if pubmed:
        parts.append("PMID " + ", ".join(str(item) for item in pubmed[:2]))
    return "[" + " | ".join(parts) + "]"


def _render_driver(driver: Any) -> Dict[str, Any]:
    refs = [_render_driver_ref(ref) for ref in _as_list(_get_value(driver, "evidence_refs", []))]
    return {
        "label": _get_value(driver, "driver") or _get_value(driver, "driver_type") or "retrieved evidence",
        "driver_type": _get_value(driver, "driver_type"),
        "direction": _get_value(driver, "direction"),
        "confidence": _get_value(driver, "confidence"),
        "mechanistic_role": _get_value(driver, "mechanistic_role"),
        "refs": [ref for ref in refs if ref],
        "limitations": _get_value(driver, "limitations"),
    }


def _render_driver_ref(ref: Any) -> str:
    tool = _get_value(ref, "tool_name") or _get_value(ref, "source") or "evidence"
    field = _get_value(ref, "field") or _get_value(ref, "evidence_path")
    summary = _get_value(ref, "summary")
    parts = [str(tool)]
    if field:
        parts.append(str(field))
    if summary:
        parts.append(str(summary))
    return "[" + " | ".join(parts) + "]"


def _render_evidence_chain(
    item: Any,
    attribution: Any,
    node_impact_notes: Any = None,
    probability_drivers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    note_by_node = _node_impact_note_lookup(node_impact_notes)
    probability_drivers = probability_drivers or []
    chains = _as_list(_get_value(attribution, "mechanism_chains", []))
    if chains:
        chain = max(chains, key=lambda candidate: float(_get_value(candidate, "chain_score", 0) or 0))
        nodes = sorted(
            _as_list(_get_value(chain, "nodes", [])),
            key=lambda node: int(_get_value(node, "order", 0) or 0),
        )
        rendered_nodes = [
            _render_chain_node(
                node,
                note_by_node.get(_node_key(_get_value(node, "node_type"), _get_value(node, "label")))
                or _fallback_node_impact_note(node, probability_drivers),
            )
            for node in nodes[:8]
        ]
        return {
            "chain_type": _mechanism_chain_type([chain]),
            "readable_path": [
                node["line"] for node in rendered_nodes if node.get("line")
            ] + [f"SOC: {_get_value(item, 'soc')}"],
            "nodes": rendered_nodes,
            "chain_id": _get_value(chain, "chain_id"),
            "chain_score": _get_value(chain, "chain_score"),
            "chain_confidence": _get_value(chain, "chain_confidence"),
        }
    target_pathway = _as_list(_get_value(attribution, "target_pathway", []))
    if target_pathway:
        path_lines = []
        for path in target_pathway[:5]:
            target = _get_value(path, "target", {}) or {}
            pathway = _get_value(path, "pathway", {}) or {}
            ade = _get_value(path, "ade", {}) or {}
            pubmed = _as_list(_get_value(path, "pubmed", []))
            bits = [
                _get_value(target, "gene") or _get_value(target, "protein") or _get_value(target, "id"),
                _get_value(pathway, "name") or _get_value(pathway, "id"),
                _get_value(ade, "name") or _get_value(ade, "id"),
            ]
            label = "target/pathway evidence: " + (" -> ".join(str(bit) for bit in bits if bit) or "available evidence")
            if pubmed:
                label += " PMID " + ", ".join(str(ref) for ref in pubmed[:2])
            path_lines.append(label)
        path_lines.append(f"SOC: {_get_value(item, 'soc')}")
        return {"chain_type": "evidence_assembled_chain", "readable_path": path_lines, "nodes": []}
    summary = _get_value(attribution, "mechanism_summary") or "No curated mechanism chain mapped to this SOC."
    return {
        "chain_type": "evidence_assembled_chain",
        "readable_path": ["evidence summary: " + str(summary), f"SOC: {_get_value(item, 'soc')}"],
        "nodes": [],
    }


def _render_chain_node(node: Any, impact_note: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    extras = []
    role = _get_value(node, "role")
    enzymes = _as_list(_get_value(node, "enzymes", []))
    binding_role = _get_value(node, "binding_role")
    confidence = _get_value(node, "confidence")
    if role:
        extras.append(str(role))
    if enzymes:
        extras.append("enz=" + "/".join(str(item) for item in enzymes[:3]))
    if binding_role:
        extras.append(f"binding={binding_role}")
    if confidence is not None:
        extras.append(f"conf={_format_float(confidence)}")
    evidence_refs = [
        ref
        for ref in (_render_mechanism_evidence_ref(evidence) for evidence in _as_list(_get_value(node, "evidence", []))[:2])
        if ref
    ]
    extras.extend(evidence_refs)
    node_type = _get_value(node, "node_type")
    label = _get_value(node, "label")
    line = f"{node_type}: {label}"
    if extras:
        line += " (" + "; ".join(_clip_text(item, 80) for item in extras) + ")"
    rendered = {
        "node_type": node_type,
        "label": label,
        "confidence": confidence,
        "evidence_refs": evidence_refs,
        "line": line,
    }
    if impact_note:
        rendered["impact_type"] = impact_note.get("impact_type")
        rendered["impact_note"] = impact_note.get("impact_note")
        rendered["impact_source"] = impact_note.get("impact_source")
    return rendered


def _node_impact_note_lookup(value: Any) -> Dict[tuple[str, str], Dict[str, Any]]:
    notes: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        node_type = item.get("node_type")
        label = item.get("label")
        impact_note = item.get("impact_note")
        if not node_type or not label or not impact_note:
            continue
        notes[_node_key(node_type, label)] = {
            "impact_type": item.get("impact_type") or "mechanistic_context_only",
            "impact_note": impact_note,
            "impact_source": "live_llm",
        }
    return notes


def _node_key(node_type: Any, label: Any) -> tuple[str, str]:
    return (str(node_type or "").strip().lower(), str(label or "").strip().lower())


def _fallback_node_impact_note(node: Any, probability_drivers: List[Dict[str, Any]]) -> Dict[str, str]:
    node_type = str(_get_value(node, "node_type") or "").strip()
    label = str(_get_value(node, "label") or "this node").strip()
    matched_driver = _matching_probability_driver(label, probability_drivers)
    if matched_driver:
        support = matched_driver.get("support")
        support_text = f" with support={_format_float(support)}" if support is not None else ""
        return {
            "impact_type": "direct_probability_support",
            "impact_source": "template_fallback",
            "impact_note": (
                f"Retrieved probability-audit evidence maps to {label}{support_text}. "
                "This node therefore contributes direct support to the baseline probability rather than only explaining the mechanism."
            ),
        }

    evidence_source = _node_evidence_source(node)
    if node_type == "metabolism":
        return {
            "impact_type": "mechanistic_context_only",
            "impact_source": "template_fallback",
            "impact_note": (
                f"{evidence_source or 'Retrieved metabolism knowledge'} places the drug in a metabolic context. "
                "Because no metabolism-specific probability driver was found in the baseline audit, it supports plausibility but does not independently raise the probability."
            ),
        }
    if node_type == "active_or_toxic_species":
        return {
            "impact_type": "uncertainty_or_gap",
            "impact_source": "template_fallback",
            "impact_note": (
                "The chain keeps the active or toxic species explicit, but the retrieved evidence did not resolve a specific toxic metabolite. "
                "This preserves the causal route while adding uncertainty rather than direct probability support."
            ),
        }
    if evidence_source:
        return {
            "impact_type": "mechanistic_context_only",
            "impact_source": "template_fallback",
            "impact_note": (
                f"{evidence_source} supports {label} as part of the mechanism chain. "
                "It is treated as context unless the same evidence appears in the probability audit as a scored driver."
            ),
        }
    return {
        "impact_type": "mechanistic_context_only",
        "impact_source": "template_fallback",
        "impact_note": (
            "This node structures the retrieved mechanism chain, but no separate scored probability driver was matched to it. "
            "It explains biological plausibility without independently changing the baseline probability."
        ),
    }


def _matching_probability_driver(label: str, probability_drivers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    normalized_label = _normalize_match_text(label)
    if not normalized_label:
        return None
    for driver in probability_drivers:
        haystack = " ".join(
            str(part)
            for part in (
                driver.get("label"),
                driver.get("ref"),
                driver.get("source"),
                driver.get("evidence_type"),
            )
            if part
        )
        normalized_haystack = _normalize_match_text(haystack)
        if normalized_label and normalized_label in normalized_haystack:
            return driver
    return None


def _normalize_match_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _node_evidence_source(node: Any) -> Optional[str]:
    evidence = _as_list(_get_value(node, "evidence", []))
    if not evidence:
        return None
    labels: List[str] = []
    for item in evidence[:2]:
        source = _get_value(item, "source")
        ref = _get_value(item, "ref")
        if source and ref:
            labels.append(f"{source} ({ref})")
        elif source or ref:
            labels.append(str(source or ref))
    if not labels:
        return None
    return "Retrieved evidence from " + "; ".join(labels)


def _render_mechanism_evidence_ref(evidence: Any) -> Optional[str]:
    source = _get_value(evidence, "source")
    ref = _get_value(evidence, "ref")
    tier = _get_value(evidence, "tier")
    if not source and not ref:
        return None
    parts = [str(source or "evidence")]
    if tier is not None:
        parts.append(f"tier {tier}")
    if ref:
        parts.append(str(ref))
    return "[" + " | ".join(parts) + "]"


def _mechanism_chain_type(chains: List[Any]) -> str:
    if not chains:
        return "evidence_assembled_chain"
    chain = chains[0]
    summary = str(_get_value(chain, "summary", "") or "").lower()
    evidence = " ".join(str(_get_value(item, "ref", "")) for item in _as_list(_get_value(chain, "evidence", []))).lower()
    if "fallback" in summary or "fallback" in evidence:
        return "fallback_chain"
    if _get_value(chain, "chain_complete", False):
        return "curated_mechanism_chain"
    return "evidence_assembled_chain"


def _trace_refs(item: Any, probability_drivers: List[Dict[str, Any]]) -> List[str]:
    refs = [str(driver.get("ref")) for driver in probability_drivers if driver.get("ref")]
    for evidence in _as_list(_get_value(item, "evidence", [])):
        rendered = _render_general_evidence_ref(evidence)
        if rendered:
            refs.append(rendered)
    return list(dict.fromkeys(refs))[:10]


def _render_general_evidence_ref(evidence: Any) -> Optional[str]:
    source = _get_value(evidence, "source")
    tier = _get_value(evidence, "tier")
    ref = _get_value(evidence, "ref")
    if not source and not ref:
        return None
    parts = [str(source or "evidence")]
    if tier is not None:
        parts.append(f"tier {tier}")
    if ref:
        parts.append(str(ref))
    return "[" + " | ".join(parts) + "]"


def _clip_text(value: Any, limit: int = 150) -> str:
    text = "null" if value is None else str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "null"

