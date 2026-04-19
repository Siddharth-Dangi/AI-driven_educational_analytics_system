from typing import TypedDict, List
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# STRUCTURED OUTPUT SCHEMA (For LLM to adhere to)
# ---------------------------------------------------------
class AssessmentReport(BaseModel):
    summary: str = Field(description="Summary of Assessment Quality & Difficulty Dist.")
    gaps: List[str] = Field(description="List of Identified Learning Gaps or flaws in the question design.")
    advice: List[str] = Field(description="List of Recommended Improvements for the questions.")
    refs: List[str] = Field(description="List of Pedagogical References justifying the advice.")
    disclaimer: str = Field(description="Educational/Ethical notice indicating AI assistance.")

# ---------------------------------------------------------
# LANGGRAPH STATE
# ---------------------------------------------------------
class AgentState(TypedDict):
    # Input
    query: str
    assessment_context: str # The question text or contextual info
    
    # RAG Retrieval
    retrieved_pedagogy: str
    
    # Model Predictions (if numeric data provided)
    numeric_model_difficulty: str
    
    # Output
    final_report: AssessmentReport
    errors: str
