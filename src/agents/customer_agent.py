from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, END
from agents.nodes import CustomerServiceNodes

class CustomerServiceState(TypedDict):
    messages: Annotated[List, add_messages]
    customer_id: Optional[str]
    intent: Optional[str]
    confidence_score: float
    context: dict
    escalation_needed: bool
    resolved: bool
    conversation_history: List[dict]
    retrieved_documents: List[dict]

def create_customer_service_graph():
    """Create the main customer service workflow"""

    nodes = CustomerServiceNodes()

    # Create the graph
    workflow = StateGraph(CustomerServiceState)

    # Add nodes
    workflow.add_node("classify_intent", nodes.classify_intent)
    workflow.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    workflow.add_node("generate_response", nodes.generate_response)
    workflow.add_node("check_escalation", nodes.check_escalation)
    workflow.add_node("escalation_to_human", nodes.escalate_to_human)

    # Define the workflow
    workflow.set_entry_point("classify_intent")

    # Add conditional edges
    workflow.add_edge("classify_intent", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "generate_response")
    workflow.add_edge("generate_response", "check_escalation")

    # Conditional routing based on escalation need
    workflow.add_conditional_edges(
        "check_escalation",
        should_escalate,
        {
            "escalate":"escalate_to_human",
            "continue": END
        }
    )

    workflow.add("escalate_to_human", END)

    return workflow.compile()

def should_escalate(state: CustomerServiceState) -> str:
    """Determine routing based on escalation need"""
    return "escalate" if state["escalation_needed"] else "continue"