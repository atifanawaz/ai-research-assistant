from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    Splits Document objects into smaller chunks while preserving metadata.

    Args:
        documents (List[Document]): List of LangChain Documents with metadata.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks for better context.

    Returns:
        List[Document]: List of split Document chunks with metadata.
    """

    # Sentence-aware splitting for clean context boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]  # most natural first
    )

    # Split documents while preserving metadata
    chunks = text_splitter.split_documents(documents)

    # Optional debug log
    print(f"[chunker.py] Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks
