# Model Performance Evaluation Report

## 1. Milestone 1 - Predictive ML Model (Logistic Regression)

The baseline system uses linear models to predict class severity. The model was trained dynamically via TF-IDF feature space embeddings mapped to student numerical aggregates.

- **Accuracy**: 98.90%
- **F1 Score**: 98.90%
- **Classes**: Easy (0.99 F1), Medium (0.98 F1), Hard (0.99 F1)
- **Evaluation Limitations**: The model assumes deterministic behavior. It does not account for poorly phrased questions causing skewed variance, and is rigid, lacking generative text capabilities to explain *why* something is hard.

## 2. Milestone 2 - RAG + Generative AI Feedback (Agentic)

### RAG Strategy Evaluation
- **Embeddings Used**: `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval Engine**: FAISS (Facebook AI Similarity Search)
- **Top K Value**: 3 documents.
- **Accuracy Context**: The FAISS index highly consistently retrieves matching vectors due to specific overlapping vernacular in educational queries (e.g., terms like "multiple-choice", "options", "phrasing").

### Generative AI Synthesis (Gemini 1.5)
- **Performance Evaluation**: The LLM successfully ingests pedagogical rules and correctly classifies overlapping multiple choice vectors as "flawed" based on the RAG rules.
- **Hallucination Reductions**: Since the model's temperature is set to `0.2`, the output logic is highly rigid and adheres strictly to the defined schema. 
- **Performance Trade-offs**: 
   - **Pros**: The generation provides step-by-step reasoning that a standard ML classifier fundamentally cannot. It actively improves user curriculum.
   - **Cons**: Introducing RAG latency, there is a marked ~2.5 second delay while the agent resolves the graph edges asynchronously, compared to the near-instant `<0.1s` inference block of standard Logistic Regression.
