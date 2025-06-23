from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from utils.database import KnowledgeBase, CustomerServiceState
import time

class CustomerServiceNodes:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.kb = KnowledgeBase()
    
    def classify_intent(self, state: CustomerServiceState) -> CustomerServiceState:
        """Classify the intent of the latest message"""
        latest_message = state["messages"][-1].content

        system_prompt = """
        You are an intent classifier for customer service. 
        Classify the customer's message into one of these categories:
        - greeting: General greetings, hello
        - product_inquiry: Questions about products, features, availability
        - order_status: Questions about existing orders, tracking
        - shipping_info: Questions about shipping, delivery
        - return_refund: Return, refund, or exchange requests
        - technical_support: Technical issues, troubleshooting
        - billing: Payment, billing, invoice questions
        - complaint: Complaints, negative feedback
        - other: Anything else
        
        Also provide a confidence score (0-1) for your classification.
        
        Respond in JSON format:
        {"intent": "category", "confidence": 0.85}   
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Customer message: {latest_message}")
        ]

        response = self.llm.invoke(messages)
        result = json.loads(response.content)

        state["intent"] = result["intent"]
        result["confidence_score"] = result["confidence"]

        return state
    
    def retrieve_knowledge(self, state:CustomerServiceState) -> CustomerServiceState:
        """Retrieve relevant information from knowledge base"""

        latest_message = state["messages"][-1].content

        #Search knowledge base
        search_results = self.kb.search(latest_message, n_results=3)

        retrieve_docs = []
        if search_results['documents']:
            for i, doc in enumerate(search_results['documents'][0]):
                retrieve_docs.append({
                    "content":docs,
                    "metadata": search_results["metadatas"][0][i],
                    "distance": search_results['distances'][0][i]
                })
        
        state["retrieved_documents"] = retrieve_docs

        return state
    
    def generate_response(self, state: CustomerServiceState) -> CustomerServiceState:
        """Generate response using retrieved knowledge"""

        latest_message = state["messages"][-1].content
        retrieved_docs = state["retrieved_documents"]
        intent = state["intent"]

        #Build context from retrieved documents
        context = "\n".join([doc["content"] for doc in retrieve_docs[:2]])

        system_prompt = f"""
        You are a helpful customer service representative.
        
        Customer Intent: {intent}

        Relevent Information:
        {context}

        Guidelines:
            - Be friendly and professional
            - Use the provided information to answer accurately
            - If you cannot answer with the given information, say so politely
            - Keep responses concise but complete
            - If the issue seems complex, suggest escalation
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Customer: {latest_message}")
        ]

        response = self.llm.invoke(messages)

        # Add AI response to messages

        from langchain_core.messages import AIMessage
        state["messages"].append(AIMessage(content=response.content))

        return state

    def check_escalation(self, state:CustomerServiceState) -> CustomerServiceState:
        """Determine if escalation to human is needed"""
        escalation_triggers = [
            state["confidence_score"] < 0.6,
            state["intent"] == "complaint"
            "manager" in state["messages"][-2].content.lower(),
            len(state["conversation_history"]) > 10,
        ]

        # Check if conversation is going in circles
        recent_messages = [msg.content for msg in state["messages"][-6:]]
        if len(set(recent_messages)) < len(recent_messages) *0.7:
            escalation_triggers.append(True)
        
        state["escalation_needed"] = any(escalation_triggers)

        return state
    
    def escalate_to_human(self, state: CustomerServiceState) -> CustomerServiceState:
        """Handle escalation to human agent"""

        escalation_message = """
        I understand this requires additional assistance. Let me connect you with one of our human representatives who can better help you with this matter. 
        
        Please hold on while I transfer your conversation. A human agent will be with you shortly.
        
        Reference ID: CS-{customer_id}-{timestamp}
        """.format(
            customer_id = state.get("customer_id","GUEST"),
            timestamp = int(time.time())
        )

        from langchain_core.messages import AIMessage
        state["messages"].append(AIMessage(content=escalation_message))

        # Log escalation (implement logging logic)
        self._log_escalation(state)

        return state
    
    def _log_escalation(self, state: CustomerServiceState) -> CustomerServiceState:
        """Log escalation for human agents"""
        #Implement logging to database or queue system
        pass
    


