import os
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

from backend.loader import load_documents
from backend.embedder import create_or_load_vectorstore
from backend.rag_chain import get_answer_with_citations
from citations.citation_formatter import format_citations_grouped

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# <CHANGE> Updated color theme to coral/salmon and dark navy with beautiful modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    :root {
        --bg-primary: #0f0f23;
        --bg-secondary: #1a1a2e;
        --bg-tertiary: #16213e;
        --accent-primary: #ff6b6b;
        --accent-secondary: #ff5252;
        --text-primary: #f5f5f7;
        --text-secondary: #a1a1aa;
        --border-color: #374151;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
    }
    
    [data-testid="stAppViewContainer"] > div:first-child {
        background: transparent;
    }
    
    .main .block-container {
        background: transparent;
        padding: 2rem 1rem;
    }
    
    /* Header styling */
    .main h1 {
        background: linear-gradient(135deg, #ff6b6b, #ff5252);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main h2 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1.5rem 0;
        border-bottom: 2px solid #ff6b6b;
        padding-bottom: 0.75rem;
        display: inline-block;
    }
    
    .main h3 {
        color: #ff6b6b;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Subtitle text */
    .main .stMarkdown {
        text-align: center;
    }
    
    .main .stMarkdown p {
        color: #a1a1aa;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    /* Main content wrapper */
    .main-card {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid #374151;
        border-radius: 1.5rem;
        padding: 2.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        margin: 2rem 0;
    }
    
    /* File uploader */
    .stFileUploader {
        background: transparent;
        border: none;
        padding: 0;
    }
    
    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #ff6b6b;
        background: rgba(255, 107, 107, 0.05);
        border-radius: 1rem;
        padding: 2rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff5252;
        background: rgba(255, 107, 107, 0.1);
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.2);
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] p {
        color: #f5f5f7 !important;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: #a1a1aa !important;
        font-size: 0.9rem;
    }
    
    /* Text area */
    .stTextArea {
        margin: 1.5rem 0;
    }
    
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .stTextArea textarea {
        background: #16213e !important;
        border: 1px solid #374151 !important;
        border-radius: 0.75rem !important;
        color: #f5f5f7 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ff6b6b !important;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.2) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #a1a1aa !important;
    }
    
    /* Text input */
    .stTextInput {
        margin: 1.5rem 0;
    }
    
    .stTextInput label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .stTextInput input {
        background: #16213e !important;
        border: 1px solid #374151 !important;
        border-radius: 0.75rem !important;
        color: #f5f5f7 !important;
        font-size: 1rem !important;
        padding: 0.875rem 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #ff6b6b !important;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.2) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder {
        color: #a1a1aa !important;
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #ff6b6b, #ff5252) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.875rem 1.5rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px rgba(255, 107, 107, 0.3) !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #ff5555, #ff4444) !important;
        box-shadow: 0 15px 40px rgba(255, 107, 107, 0.4) !important;
        transform: translateY(-2px);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 82, 82, 0.1)) !important;
        border: 1px solid rgba(255, 107, 107, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.15) !important;
    }
    
    .stSuccess .stMarkdown, .stSuccess .stMarkdown p {
        color: #ff6b6b !important;
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 82, 82, 0.1)) !important;
        border: 1px solid rgba(255, 107, 107, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.15) !important;
    }
    
    .stWarning .stMarkdown, .stWarning .stMarkdown p {
        color: #ffa5a5 !important;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Citation items */
    .citation-item {
        background: rgba(26, 26, 46, 0.6);
        border-left: 4px solid #ff6b6b;
        border: 1px solid #374151;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #f5f5f7;
        transition: all 0.3s ease;
    }
    
    .citation-item:hover {
        border-left-color: #ff5252;
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.2);
        transform: translateX(5px);
    }
    
    .citation-item strong {
        color: #ff6b6b;
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ff6b6b, #ff5252);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ff5252, #ff6b6b);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {display: none;}
    footer {display: none;}
    header {display: none;}
    
    .viewerBadge_container__1QSob {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("AI Research Assistant")
st.markdown("Upload research papers (**PDF**, **DOCX**, **TXT**) or paste academic paper links (arXiv, PubMed), then ask your question")

# Main container
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Two column layout for uploads and links
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload research documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        for file in uploaded_files:
            st.markdown(f'<div style="background: rgba(255, 107, 107, 0.1); border: 1px solid rgba(255, 107, 107, 0.3); border-radius: 0.5rem; padding: 0.75rem; margin: 0.5rem 0; color: #f5f5f7; font-size: 0.9rem;">📎 {file.name}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("🔗 Paper Links")
    urls = st.text_area(
        "Paste paper links",
        placeholder="https://arxiv.org/abs/1234.5678\nhttps://pubmed.ncbi.nlm.nih.gov/...",
        height=150,
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Question section
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("⚡ Your Research Question")
question = st.text_input(
    "Ask your question",
    placeholder="e.g., What are recent deep learning methods in medical imaging?",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Submit button
col_button = st.columns([1, 4, 1])
with col_button[0]:
    submit_btn = st.button("Search", use_container_width=True)

# Process inputs
if submit_btn and (uploaded_files or urls.strip()) and question.strip():
    with st.spinner("Processing documents and generating your answer..."):
        try:
            documents = load_documents(uploaded_files, urls.splitlines())
            vectorstore = create_or_load_vectorstore(documents)
            answer, citations = get_answer_with_citations(question, vectorstore)
            
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.subheader("✨ Answer")
            st.success(answer)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.subheader("📚 Citations")
            grouped_citations_text = format_citations_grouped(citations)
            lines = grouped_citations_text.split('\n')
            citation_counter = 1
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('**'):
                    if line.startswith('- '):
                        line = line[2:]
                    elif line.startswith('* '):
                        line = line[2:]
                    
                    if line:
                        st.markdown(f'<div class="citation-item"><strong>[{citation_counter}]</strong> {line}</div>', unsafe_allow_html=True)
                        citation_counter += 1
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif submit_btn and question and not (uploaded_files or urls.strip()):
    st.warning("Please upload at least one document or provide academic links to get started.")