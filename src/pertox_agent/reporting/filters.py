"""Target-scope filters for readable Stage 1 final reports."""

from __future__ import annotations

from typing import Any, Dict


TARGET_STAGE1_SOCS = {"Hepatobiliary disorders", "Cardiac disorders"}
TARGET_STAGE1_ORGANS = {"liver", "heart"}
TARGET_STAGE1_MECHANISM_GROUPS = {"hepatotoxicity", "mitochondrial", "cardiotoxicity", "herg/qt"}


def filter_admet_profile(value: Any, universal_report: Any) -> Any:
    items = _as_list(value)
    if not isinstance(value, list):
        return value
    target_endpoints = target_admet_endpoints(universal_report)
    return [
        item
        for item in items
        if is_target_related(item)
        or _normalize_match_text(_get_value(item, "endpoint")) in target_endpoints
        or _normalize_match_text(_get_value(item, "mechanism_group")) in TARGET_STAGE1_MECHANISM_GROUPS
    ]


def filter_known_ade_profile(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [item for item in value if is_target_related(item)]


def filter_mechanism_chains(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [item for item in value if is_target_related(item) or has_target_node(item)]


def filter_persade_contextual_evidence(value: Any, ade_scope: Dict[str, Dict[str, Any]]) -> Any:
    if not isinstance(value, list):
        return value
    filtered = []
    for item in value:
        if is_target_related(item):
            filtered.append(item)
            continue
        ade_id = str(_get_value(item, "ade_id") or "").strip()
        if ade_id and ade_id in ade_scope:
            filtered.append(item)
    return filtered


def target_ade_scope(known_ade_profile: Any) -> Dict[str, Dict[str, Any]]:
    scoped: Dict[str, Dict[str, Any]] = {}
    for item in _as_list(known_ade_profile):
        if not isinstance(item, dict) or not is_target_related(item):
            continue
        ade_id = str(item.get("ade_id") or "").strip()
        if ade_id:
            scoped[ade_id] = item
    return scoped


def target_admet_endpoints(universal_report: Any) -> set[str]:
    endpoints: set[str] = set()
    for item in _as_list(_get_value(universal_report, "general_toxicity", [])):
        if _get_value(item, "soc") not in TARGET_STAGE1_SOCS:
            continue
        attribution = _get_value(item, "attribution", {}) or {}
        for endpoint in _as_list(_get_value(attribution, "admet_endpoint", [])):
            endpoint_name = _normalize_match_text(_get_value(endpoint, "endpoint"))
            if endpoint_name:
                endpoints.add(endpoint_name)
    return endpoints


def is_target_related(value: Any) -> bool:
    soc = _get_value(value, "soc")
    if soc in TARGET_STAGE1_SOCS:
        return True

    organ = _normalize_match_text(_get_value(value, "organ_system") or _get_value(value, "organ"))
    if organ in TARGET_STAGE1_ORGANS:
        return True

    organ_systems = _as_list(_get_value(value, "organ_systems", []))
    if any(_normalize_match_text(item) in TARGET_STAGE1_ORGANS for item in organ_systems):
        return True

    mechanism_group = _normalize_match_text(_get_value(value, "mechanism_group"))
    return mechanism_group in TARGET_STAGE1_MECHANISM_GROUPS


def has_target_node(value: Any) -> bool:
    nodes = _as_list(_get_value(value, "nodes", [])) + _as_list(_get_value(value, "chain", []))
    return any(is_target_related(node) for node in nodes)


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_match_text(value: Any) -> str:
    return str(value or "").strip().lower()
