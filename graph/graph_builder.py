"""Build the LangGraph state graph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from graph.nodes import (
    brain_parse_input,
    brain_revise_output,
    brain_stage1_reason,
    brain_stage1_plan_retrieval,
    brain_stage2_reason,
    brain_stage2_plan_retrieval,
    format_output,
    knowledge_retrieval_node,
    route_after_knowledge,
    verifier_node,
)
from models.state import AgentState


def build_graph() -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("brain_parse_input", brain_parse_input)
    workflow.add_node("brain_stage1_plan_retrieval", brain_stage1_plan_retrieval)
    workflow.add_node("brain_stage1_reason", brain_stage1_reason)
    workflow.add_node("knowledge_retrieval_node", knowledge_retrieval_node)
    workflow.add_node("brain_stage2_plan_retrieval", brain_stage2_plan_retrieval)
    workflow.add_node("brain_stage2_reason", brain_stage2_reason)
    workflow.add_node("verifier_node", verifier_node)
    workflow.add_node("brain_revise_output", brain_revise_output)
    workflow.add_node("format_output", format_output)

    workflow.set_entry_point("brain_parse_input")
    workflow.add_edge("brain_parse_input", "brain_stage1_plan_retrieval")
    workflow.add_edge("brain_stage1_plan_retrieval", "knowledge_retrieval_node")
    workflow.add_conditional_edges(
        "knowledge_retrieval_node",
        route_after_knowledge,
        {
            "stage1_reason": "brain_stage1_reason",
            "stage2_reason": "brain_stage2_reason",
        },
    )
    workflow.add_edge("brain_stage1_reason", "brain_stage2_plan_retrieval")
    workflow.add_edge("brain_stage2_plan_retrieval", "knowledge_retrieval_node")
    workflow.add_edge("brain_stage2_reason", "verifier_node")
    workflow.add_edge("verifier_node", "brain_revise_output")
    workflow.add_edge("brain_revise_output", "format_output")
    workflow.add_edge("format_output", END)
    

    return workflow.compile()
