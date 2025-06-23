from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from langchain_core.messages import HumanMessage
from agents.customer_agent import CustomerServiceState, create_customer_service_graph

app = FastAPI(title="Customer Service Agent API")

# Store active conversations in memory (use Redis in production)
active_conversations = {}

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    respone: str
    session_id: str
    escalation_needed: bool
    resolved: bool

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Get or create conversation state
    if session_id in active_conversations:
        state = active_conversations[session_id]
    else:
        state = CustomerServiceState(
            messages=[],
            customer_id=request.customer_id,
            intent=None,
            confidence_score=0.0,
            context={},
            escalation_needed=False,
            resolved=False,
            conversation_history=[],
            retrieved_documents=[],
        )
    
    # Add user message
    state["messages"].append(HumanMessage(content=request.message))

    # Process throught LangGraph
    graph = create_customer_service_graph()
    result = graph.invoke(state)

    # Store updated state
    active_conversations[session_id] = result

    # Get AI repsonse
    ai_response = result["messages"][-1].content

    return ChatResponse(
        respone=ai_response,
        session_id=session_id,
        escalation_needed=result["escalation_needed"],
        resolved=result["resolved"]
    )

@app.get("/health")
async def health_check():
    return {"status":"healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)