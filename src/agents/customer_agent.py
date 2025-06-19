from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages

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
    