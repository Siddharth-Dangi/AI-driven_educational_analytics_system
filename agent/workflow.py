import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# Import local modules
from agent.state import AgentState, AssessmentReport
from agent.rag import get_faiss_retriever

load_dotenv()

# Initialize Gemini Model
# We expect GEMINI_API_KEY to be in the environment natively or via .env
import os
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2,
    api_key=os.getenv("GEMINI_API_KEY")
)
structured_llm = llm.with_structured_output(AssessmentReport)

# ---------------------------------------------------------
# NODES
# ---------------------------------------------------------

def retrieve_guidelines(state: AgentState):
    """Retrieves relevant pedagogy guidelines using FAISS/RAG."""
    query = state.get("query", "")
    context = state.get("assessment_context", "")
    search_str = f"{query} {context}"
    
    try:
        retriever = get_faiss_retriever()
        docs = retriever.invoke(search_str)
        retrieved_text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        retrieved_text = "Standard pedagogical guidelines apply. (FAISS index lookup failed or not found)"
    
    return {"retrieved_pedagogy": retrieved_text}

def analyze_and_generate(state: AgentState):
    """Generates the structured assessment report using Gemini."""
    query = state.get("query", "")
    context = state.get("assessment_context", "")
    pedagogy = state.get("retrieved_pedagogy", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Educational Assessment Designer and Pedagogy Agent. "
                   "Your goal is to evaluate exam questions and provide highly actionable, structured feedback.\n\n"
                   "Follow these Pedagogy Guidelines strictly:\n{pedagogy}\n\n"
                   "You must format your response exactly to the requested structured output. "
                   "Provide a holistic summary, identify specific gaps/flaws, suggest concrete improvements, "
                   "cite the pedagogy rules used, and include a standard ethical disclaimer."),
        ("human", "Educator Query: {query}\n\nAssessment Context/Questions:\n{context}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "pedagogy": pedagogy,
            "query": query,
            "context": context
        })
        return {"final_report": result}
    except Exception as e:
        # Fallback empty structure in case of parsing error or missing API key
        return {"errors": str(e)}

# ---------------------------------------------------------
# GRAPH DEFINITION
# ---------------------------------------------------------

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_guidelines)
workflow.add_node("generate", analyze_and_generate)

# Add edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile graph
app_workflow = workflow.compile()
