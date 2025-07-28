# backend/embedder.py

from backend.chunker import chunk_documents
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_or_load_vectorstore(documents):
    # Step 1: Chunk documents
    chunks = chunk_documents(documents)

    # Step 2: Load HuggingFace embeddings WITHOUT manually setting device
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 3: Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
