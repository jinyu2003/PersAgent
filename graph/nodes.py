"""LangGraph node functions for the PersAgent workflow."""

from __future__ import annotations

from typing import Any, Dict

from agents.input_parser import normalize_drug_input
from agents.brain_agent import BrainAgent
from agents.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from agents.verifier_agent import VerifierAgent
from models.formatting import format_to_json
from models.state import AgentState


brain_agent = BrainAgent()
knowledge_agent = KnowledgeRetrievalAgent()
verifier_agent = VerifierAgent()


def _trace(state: AgentState, message: str) -> None:
    state.setdefault("trace", [])
    state["trace"].append(message)


def brain_parse_input(state: AgentState) -> AgentState:
    patient = state.get("patient_info") or brain_agent.parse_patient_info(state.get("raw_patient_info", {}))
    raw_drug_info = state.get("raw_drug_info", {})
    drug = state.get("drug_info") or brain_agent.parse_drug_info(raw_drug_info)
    state["normalized_drug_input"] = normalize_drug_input(
        {
            "name": drug.name,
            "smiles": drug.smiles,
            "drugbank_id": drug.drugbank_id,
            "inchi_key": drug.inchi_key,
        }
    )
    state["patient_info"] = patient
    state["drug_info"] = drug
    state["next_step"] = "stage1_plan_retrieval"
    _trace(state, f"Brain parsed patient={patient.patient_id}, drug={drug.name}.")
    return state


def brain_stage1_plan_retrieval(state: AgentState) -> AgentState:
    """Stage 1 Brain planning: decide universal toxicity knowledge needs."""
    query = {
        "purpose": "universal_toxicity",
        "drug": state["drug_info"].name,
        "patient_id": state["patient_info"].patient_id,
        "planning_stage": "stage1_universal_toxicity_retrieval",
        "needs": [
            "Drug Card",
            "ADMET endpoints",
            "DTI/mechanism/pathway",
            "PersADE/FAERS population baseline and signal",
        ],
    }
    state["stage1_retrieval_plan"] = {
        "goal": "Retrieve evidence needed to estimate population-level SOC toxicity baseline.",
        "query": query,
    }
    state["pending_knowledge_query"] = query
    state["return_to"] = "brain_stage1_reason"
    state["next_step"] = "knowledge"
    _trace(state, "Brain planned Stage 1 universal toxicity retrieval and sent query to Knowledge Retrieval.")
    return state


def knowledge_retrieval_node(state: AgentState) -> AgentState:
    """Knowledge Retrieval runs exactly once for the currently pending Brain query."""
    query = state.get("pending_knowledge_query") or {"purpose": "comprehensive"}
    evidence = knowledge_agent.retrieve(
        query=query,
        patient_info=state["patient_info"],
        drug_info=state["drug_info"],
    )
    state["latest_evidence_package"] = evidence

    purpose = query.get("purpose")
    if purpose == "universal_toxicity":
        state["knowledge_stage1_done"] = True
        state["stage1_evidence_package"] = evidence
    elif purpose == "personalized_modifiers":
        state["knowledge_stage2_done"] = True
        state["stage2_evidence_package"] = evidence

    state["pending_knowledge_query"] = None
    state["next_step"] = state.get("return_to", "brain_stage1_reason")
    _trace(
        state,
        f"Knowledge Retrieval returned {len(evidence.evidence_items)} evidence items for {purpose}; returning to Brain.",
    )
    return state


def brain_stage1_reason(state: AgentState) -> AgentState:
    """Brain receives Stage 1 evidence and produces UniversalToxicityReport."""
    state["evidence_package"] = brain_agent.merge_evidence_packages(
        state.get("evidence_package"),
        state["latest_evidence_package"],
    )
    universal_report = brain_agent.build_universal_report(
        patient_info=state["patient_info"],
        drug_info=state["drug_info"],
        evidence_package=state["evidence_package"],
    )
    state["universal_report"] = universal_report
    state["next_step"] = "stage2_plan_retrieval"
    _trace(state, "Brain received Stage 1 evidence and generated UniversalToxicityReport.")
    return state


def brain_stage2_plan_retrieval(state: AgentState) -> AgentState:
    """Stage 2 Brain planning: decide personalized modifier knowledge needs."""
    query = {
        "purpose": "personalized_modifiers",
        "drug": state["drug_info"].name,
        "patient_id": state["patient_info"].patient_id,
        "planning_stage": "stage2_personalized_toxicity_retrieval",
        "baseline_soc_count": len(state["universal_report"].general_toxicity),
        "needs": [
            "PGx/CPIC",
            "DDI",
            "HLA",
            "patient-context retrieval",
            "similar cases",
            "cohort modifiers",
        ],
    }
    state["stage2_retrieval_plan"] = {
        "goal": "Retrieve patient-specific PGx, DDI, comorbidity, and organ-function modifiers.",
        "query": query,
    }
    state["pending_knowledge_query"] = query
    state["return_to"] = "brain_stage2_reason"
    state["next_step"] = "knowledge"
    _trace(state, "Brain planned Stage 2 personalized modifier retrieval and sent query to Knowledge Retrieval.")
    return state


def brain_stage2_reason(state: AgentState) -> AgentState:
    """Brain receives Stage 2 evidence and produces PersonalizedToxicityReport."""
    state["evidence_package"] = brain_agent.merge_evidence_packages(
        state.get("evidence_package"),
        state["latest_evidence_package"],
    )
    personalized_report = brain_agent.build_personalized_report(
        patient_info=state["patient_info"],
        drug_info=state["drug_info"],
        universal_report=state["universal_report"],
        evidence_package=state["evidence_package"],
    )
    state["personalized_report"] = personalized_report
    state["draft_report"] = brain_agent.synthesize_draft_report(
        patient_info=state["patient_info"],
        drug_info=state["drug_info"],
        universal_report=state["universal_report"],
        personalized_report=personalized_report,
        evidence_package=state["evidence_package"],
    )
    state["next_step"] = "verify"
    _trace(state, "Brain received Stage 2 evidence and generated PersonalizedToxicityReport draft.")
    return state


def verifier_node(state: AgentState) -> AgentState:
    verification = verifier_agent.verify(
        draft_report=state["draft_report"],
        patient_info=state["patient_info"],
        drug_info=state["drug_info"],
        evidence_package=state["evidence_package"],
    )
    state["verification_report"] = verification
    state["next_step"] = "revise"
    _trace(state, f"Verifier completed with status={verification.status}.")
    return state


def brain_revise_output(state: AgentState) -> AgentState:
    revised = brain_agent.revise_with_verification(
        draft_report=state["draft_report"],
        verification_report=state["verification_report"],
    )
    state["draft_report"] = revised
    state["personalized_report"] = revised["personalized_report"]
    state["next_step"] = "format"
    _trace(state, "Brain integrated verifier feedback.")
    return state


def format_output(state: AgentState) -> AgentState:
    report: Dict[str, Any] = state["draft_report"]
    state["final_output"] = {
        "json": format_to_json(report),
    }
    state["next_step"] = "done"
    _trace(state, "Formatted final JSON output using the requested Stage 1 and Stage 2 schemas.")
    return state


def route_after_knowledge(state: AgentState) -> str:
    next_step = state.get("next_step")
    if next_step == "brain_stage2_reason":
        return "stage2_reason"
    return "stage1_reason"
