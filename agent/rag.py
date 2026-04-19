import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
DATA_PATH = os.path.join(BASE_DIR, "data", "pedagogy_guidelines.txt")

def get_embeddings():
    # Use lightweight local embeddings from HuggingFace
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_faiss_index():
    """Builds and saves the FAISS index to disk."""
    print("Loading documents...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("Creating embeddings and FAISS index...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    print(f"Saving FAISS index to {INDEX_PATH}...")
    vectorstore.save_local(INDEX_PATH)
    print("Done!")

def get_faiss_retriever():
    """Loads the FAISS index from disk and returns a retriever."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please build it first.")
    
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    build_faiss_index()
