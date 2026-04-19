# System Architecture Diagram & Input-Output Specification

## 1. System Architecture Diagram

```mermaid
graph TD
    subgraph Frontend [Streamlit Application]
        UI[User Interface]
        INP1[Batch CSV Upload] --> UI
        INP2[Single Q Input] --> UI
        INP3[Agentic Query + Context] --> UI
    end

    subgraph Backend_ML [Milestone 1: Predictive Model]
        Pipeline[Sklearn Pipeline]
        TFIDF[TF-IDF Text Features] --> Pipeline
        Scaler[Standard Scaler Numeric] --> Pipeline
        LogReg[Logistic Regression] --> Pipeline
    end

    subgraph Backend_Agent [Milestone 2: LangGraph Agent]
        State[(Agent State Dict)]
        RAG[FAISS Vector Store]
        Node1[Node 1: retrieve_guidelines]
        Node2[Node 2: analyze_and_generate]
        Gemini[Google Gemini LLM]
    end

    UI --> |Numeric/Text Data| Pipeline
    Pipeline --> |Predictions & Probas| UI

    UI --> |Educator Query & Qs| State
    State --> Node1
    Node1 --> |Similarity Search| RAG
    RAG --> |Retrieved Pedagogy| Node1
    Node1 --> State
    
    State --> Node2
    Node2 --> |State + Prompt| Gemini
    Gemini --> |Structured JSON Response| Node2
    Node2 --> State
    
    State --> |Final AssessmentReport| UI
```

## 2. Input-Output Specification (Agent Module)

### Inputs
The system expects an initial state containing string inputs from the Streamlit frontend.
- `query` (string): The general question or prompt from the educator (e.g., "Melt these questions down and tell me if they are fair.")
- `assessment_context` (string): The raw text of the drafted questions to be evaluated.

### Core Processing
1. **RAG Search Strategy**: Concatenates `{query} {assessment_context}` to generate an embedding. Queries FAISS local index for top `K=3` relevant chunks from `pedagogy_guidelines.txt`.
2. **Generative Processing**: LangChain's `with_structured_output` mechanism forces the LLM to map its reasoning tokens into a predefined JSON schema (via Pydantic).

### Output
The system yields a `final_report` which maps exactly to the `AssessmentReport` schema.

```json
{
  "summary": "String detailing the holistic overview.",
  "gaps": ["Array", "of", "strings", "detailing", "flaws."],
  "advice": ["Array", "of", "actionable", "recommendations."],
  "refs": ["Array", "of", "pedagogical", "citations."],
  "disclaimer": "String indicating AI generation."
}
```
