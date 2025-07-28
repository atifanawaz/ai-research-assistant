# backend/embedder.py

from backend.chunker import chunk_documents
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class CustomEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False)

def create_or_load_vectorstore(documents):
    # Step 1: Chunk documents
    chunks = chunk_documents(documents)

    # Step 2: Use the custom embedding wrapper
    embeddings = CustomEmbedding()

    # Step 3: Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
