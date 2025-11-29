import os
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

from backend.loader import load_documents
from backend.embedder import create_or_load_vectorstore
from backend.rag_chain import get_answer_with_citations
from citations.citation_formatter import format_citations_grouped

st.set_page_config(
    page_title="AI Research Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# <CHANGE> Complete redesign with coral/salmon and dark navy theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0f0f1a;
        --bg-secondary: #1a1a2e;
        --bg-tertiary: #242438;
        --accent-coral: #ff7f6b;
        --accent-coral-light: #ff9f8f;
        --accent-teal: #2a5d7d;
        --accent-teal-light: #3a7d9d;
        --text-primary: #ffffff;
        --text-secondary: #b0b0c0;
        --border-color: #383854;
        --success-color: #10b981;
        --shadow-sm: 0 4px 12px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 12px 40px rgba(255, 127, 107, 0.15);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-tertiary) 100%);
        color: var(--text-primary);
    }
    
    /* Main container */
    .main .block-container {
        background: rgba(26, 26, 46, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-lg);
        max-width: 900px;
    }
    
    /* Title styling */
    .main h1 {
        color: var(--text-primary);
        text-align: center;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--accent-coral), var(--accent-coral-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle styling */
    .main .stMarkdown > div:first-child > p {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Subheader styling */
    .main h2 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--accent-coral);
    }
    
    .main h3 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* File uploader */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(42, 93, 125, 0.1), rgba(255, 127, 107, 0.05));
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed var(--accent-coral);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 1rem 0;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-coral-light);
        background: linear-gradient(135deg, rgba(42, 93, 125, 0.15), rgba(255, 127, 107, 0.1));
        box-shadow: var(--shadow-sm);
        transform: translateY(-2px);
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Text area styling */
    .stTextArea {
        margin: 1.2rem 0;
    }
    
    .stTextArea label {
        color: var(--text-primary) !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stTextArea textarea {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent-coral) !important;
        box-shadow: 0 0 0 3px rgba(255, 127, 107, 0.1) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.6;
    }
    
    /* Text input styling */
    .stTextInput {
        margin: 1.2rem 0;
    }
    
    .stTextInput label {
        color: var(--text-primary) !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stTextInput input {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        padding: 0.9rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent-coral) !important;
        box-shadow: 0 0 0 3px rgba(255, 127, 107, 0.1) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.6;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, var(--accent-coral), var(--accent-coral-light)) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 10px 30px rgba(255, 127, 107, 0.3) !important;
    }
    
    .stSuccess .stMarkdown {
        color: white !important;
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, var(--accent-teal), var(--accent-teal-light)) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 10px 30px rgba(42, 93, 125, 0.3) !important;
    }
    
    .stWarning .stMarkdown {
        color: white !important;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        margin: 2rem 0;
    }
    
    .stSpinner > div {
        border-color: var(--accent-coral) !important;
        border-right-color: transparent !important;
    }
    
    /* Citation items - Individual blocks */
    .citation-item {
        background: linear-gradient(135deg, rgba(42, 93, 125, 0.1), rgba(255, 127, 107, 0.05));
        border-left: 4px solid var(--accent-coral);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.9rem 0;
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-coral);
    }
    
    .citation-item:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow-lg);
        border-left-color: var(--accent-coral-light);
        background: linear-gradient(135deg, rgba(42, 93, 125, 0.15), rgba(255, 127, 107, 0.1));
    }
    
    .citation-item strong {
        color: var(--accent-coral);
        font-weight: 700;
    }
    
    .citation-item p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Answer heading */
    .main h3 {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Divider */
    .main hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-coral), transparent);
        margin: 2.5rem 0;
        opacity: 0.3;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-coral), var(--accent-coral-light));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-coral-light), var(--accent-coral));
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.5rem;
            border-radius: 16px;
        }
        
        .main h1 {
            font-size: 1.8rem;
        }
        
        .main h2 {
            font-size: 1.1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title("AI Research Assistant")
st.markdown("📚 Upload research papers or paste academic links, then ask your question with full citations")

# Main sections
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload research documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

with col2:
    st.subheader("🔗 Paste Links")
    urls = st.text_area(
        "Academic paper links",
        placeholder="https://arxiv.org/abs/1234.5678\nhttps://pubmed.ncbi.nlm.nih.gov/...",
        height=100
    )

# Question section
st.subheader("❓ Ask Your Question")
question = st.text_input(
    "Research question",
    placeholder="e.g. What are recent deep learning methods in medical imaging?"
)

# Process inputs
if (uploaded_files or urls.strip()) and question.strip():
    with st.spinner("🔄 Processing documents and generating your answer..."):
        # Step 1: Load documents
        documents = load_documents(uploaded_files, urls.splitlines())
        
        # Step 2: Create or load vector store
        vectorstore = create_or_load_vectorstore(documents)
        
        # Step 3: Get answer with citations
        answer, citations = get_answer_with_citations(question, vectorstore)
        
        # Step 4: Display answer
        st.subheader("💡 Answer")
        st.success(answer)
        
        # Step 5: Display citations
        st.subheader("📖 Citations")
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
                    st.markdown(f"""
                    <div class="citation-item">
                        <strong>[{citation_counter}]</strong> {line}
                    </div>
                    """, unsafe_allow_html=True)
                    citation_counter += 1

elif question and not (uploaded_files or urls.strip()):
    st.warning("⚠️ Please upload at least one document or provide academic links to proceed.")