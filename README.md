# AI Research Assistant with Citation Support

An intelligent, citation-aware research assistant powered by LangChain, LLaMA 3 (via Groq), and FAISS. It allows users to upload academic papers or link online documents (like arXiv or PubMed), ask research questions, and receive contextual answers with inline citations.

---

## 🔗 Live Demo

Try it here:  
[https://ai-research-assistant-dro7f3ajgp4fq5vdjqiffd.streamlit.app/](https://ai-research-assistant-dro7f3ajgp4fq5vdjqiffd.streamlit.app/)

---

## 🔍 Features

- Upload PDF, DOCX, or TXT academic documents  
- Paste links to arXiv or PubMed papers  
- Ask research-oriented questions and get contextual answers  
- Inline citation support ([1], [2], etc.)  
- View citation details including page, content snippet, and source  
- Powered by Retrieval-Augmented Generation (RAG)

---

## ⚙️ Technologies Used

| Tool/Library                  | Purpose                                  |
|-------------------------------|------------------------------------------|
| **Streamlit**                 | Web app UI                               |
| **LangChain**                 | Retrieval and chaining framework         |
| **FAISS**                     | Vector store for document similarity     |
| **Hugging Face Embeddings**   | Free embedding model for documents       |
| **Groq (LLaMA 3)**            | Fast large language model via API        |
| **PyMuPDF, pypdf, docx2txt**  | Document parsing (PDF, DOCX, TXT)        |
| **tiktoken**                  | Token-aware chunking                     |

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
