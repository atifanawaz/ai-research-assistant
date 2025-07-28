# embedder.py

from backend.chunker import chunk_documents
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

def create_or_load_vectorstore(documents):
    # Step 1: Chunk the documents
    chunks = chunk_documents(documents)

    # Step 2: Manually load the model to avoid device issues
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = HuggingFaceEmbeddings(model=model)

    # Step 3: Create vectorstore using FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
