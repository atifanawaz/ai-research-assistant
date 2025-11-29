# AI Research Assistant with Accurate Citation Support

A fully featured, citation-aware AI research assistant powered by **LangChain**, **Groq (LLaMA 3)**, and **FAISS**. It allows users to upload academic papers or link to online research (like **arXiv** or **PubMed**), ask questions, and receive grounded, page-referenced answers — with optional inline citation tags like `[1]`.

---

## Live Demo

Try it live here:  
[https://askmyresearch.streamlit.app/](https://citation-insight.streamlit.app/)

---

## Features

- Upload research papers in **PDF**, **DOCX**, or **TXT** formats
- Paste **public links** from arXiv or PubMed (auto-fetch and parsing supported)
- Ask **custom research questions** related to uploaded papers or URLs
- Get **contextual, grounded answers** using LLaMA 3 via Groq API
- Automatically extract **citations** including:
  - Page number (`Page X`)
  - Matching content snippet
  - Source (file name or URL)
- Injects **inline citation tags** like `[1]` if content from a document is used
- Citations are only shown if the answer overlaps the source content
- Works on **any research domain** — no fixed keywords or filters
- Processes multiple files and URLs together

---

## Technologies Used

- Streamlit — for building the interactive web interface
- LangChain — for orchestrating the retrieval-augmented generation (RAG) pipeline
- Groq (LLaMA 3) — used for generating language model responses with high speed and accuracy
- FAISS — for storing and retrieving semantic document chunks using vector similarity
- Hugging Face Sentence Transformers — used for generating document embeddings (`all-MiniLM-L6-v2`)
- PyMuPDF and docx2txt — for extracting text from PDF and DOCX files
- tiktoken — used for token-aware chunking of long texts to fit LLM context
- Regex Matching and String Inference — for inline citation injection based on content similarity

---

## How It Works

1. **Upload Files or Paste Links**  
   - Supports PDF, DOCX, TXT, arXiv.org and PubMed URLs.

2. **Document Parsing and Chunking**  
   - Text is extracted and chunked intelligently using sentence boundaries and token-aware limits.

3. **Embedding and Vector Storage**  
   - Each chunk is embedded with `all-MiniLM-L6-v2` and stored in FAISS.

4. **Question Answering with Citations**  
   - Your question is matched to relevant chunks using max marginal relevance.
   - Answer is generated with Groq's LLaMA 3 and cited **only if** source content overlaps.

5. **Citation Injection**  
   - Citation numbers like `[1]` are shown inline if document content matches.
   - A full citation summary is appended at the end (page number + snippet + source).

---

## Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key in environment (or config.py)
export GROQ_API_KEY=your_groq_api_key

# 5. Run the app
streamlit run app.py
