import fitz  # PyMuPDF
import docx2txt
import requests
from langchain_core.documents import Document
import os

def extract_text_from_pdf_with_pages(file_path, source_name):
    """Extracts PDF text page by page and returns Document objects with page numbers."""
    doc = fitz.open(file_path)
    documents = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": i
                }
            ))
    return documents

def extract_text_from_docx(file_path, source_name):
    text = docx2txt.process(file_path).strip()
    return [Document(
        page_content=text,
        metadata={
            "source": source_name,
            "page": "N/A"
        }
    )]

def extract_text_from_txt(file_path, source_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return [Document(
        page_content=text,
        metadata={
            "source": source_name,
            "page": "N/A"
        }
    )]

def extract_text_from_url(url):
    """Very basic placeholder, returns whole text as a Document."""
    if "arxiv.org" in url:
        arxiv_id = url.strip().split("/")[-1]
        response = requests.get(f'https://export.arxiv.org/api/query?id_list={arxiv_id}')
        text = response.text.strip()
        return [Document(
            page_content=text,
            metadata={
                "source": url,
                "page": "N/A"
            }
        )]
    elif "pubmed.ncbi.nlm.nih.gov" in url:
        return [Document(
            page_content=f"This is a PubMed link: {url}",
            metadata={
                "source": url,
                "page": "N/A"
            }
        )]
    else:
        return [Document(
            page_content=f"Unsupported URL: {url}",
            metadata={
                "source": url,
                "page": "N/A"
            }
        )]

def load_documents(uploaded_files, urls):
    documents = []

    # Process uploaded files
    for file in uploaded_files:
        file_path = f"data/uploads/{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            documents.extend(extract_text_from_pdf_with_pages(file_path, file.name))
        elif file.name.endswith(".docx"):
            documents.extend(extract_text_from_docx(file_path, file.name))
        elif file.name.endswith(".txt"):
            documents.extend(extract_text_from_txt(file_path, file.name))

    # Process URLs
    for url in urls:
        if url.strip():
            documents.extend(extract_text_from_url(url.strip()))

    return documents
