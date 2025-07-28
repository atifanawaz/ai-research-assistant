# embedder.py

from backend.chunker import chunk_documents
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


def create_or_load_vectorstore(documents):
    # Step 1: Chunk the documents
    chunks = chunk_documents(documents)

    # Step 2: Use free sentence-transformer for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 3: Create vectorstore using FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
