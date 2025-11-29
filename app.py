import os
import streamlit as st
from backend.loader import load_documents
from backend.embedder import create_or_load_vectorstore
from backend.rag_chain import get_answer_with_citations
from citations.citation_formatter import format_citations_grouped

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Enhanced Dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    :root {
        --bg-primary: #0a0e27;
        --bg-secondary: #111633;
        --bg-tertiary: #1a1f3a;
        --bg-card: rgba(26, 31, 58, 0.6);
        --accent-primary: #6366f1;
        --accent-secondary: #a78bfa;
        --accent-tertiary: #ec4899;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --border-color: #334155;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-tertiary) 100%);
        color: var(--text-primary);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        background: transparent;
        padding: 2rem 1.5rem;
    }
    
    /* Title styling */
    .main h1 {
        color: var(--text-primary);
        text-align: center;
        font-weight: 700;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary), var(--accent-tertiary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    /* Subtitle */
    .main .stMarkdown > div:first-child > p {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Subheaders */
    .main h2 {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 1.5rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .main h2::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 1.5rem;
        background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 2px;
    }
    
    /* Input sections container */
    .input-section {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .input-section:hover {
        border-color: var(--accent-primary);
        background: rgba(26, 31, 58, 0.8);
    }
    
    /* File uploader */
    .stFileUploader {
        background: transparent !important;
        border: 2px dashed var(--accent-primary) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-secondary) !important;
        background: rgba(99, 102, 241, 0.05) !important;
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
        min-height: 100px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        outline: none !important;
    }
    
    .stTextArea label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Text input */
    .stTextInput input {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        outline: none !important;
    }
    
    .stTextInput label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.1)) !important;
        border: 1px solid var(--success-color) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 2rem 0 !important;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1) !important;
    }
    
    .stSuccess .stMarkdown p {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(251, 191, 36, 0.1)) !important;
        border: 1px solid var(--warning-color) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 2rem 0 !important;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.1) !important;
    }
    
    .stWarning .stMarkdown p {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 1.05rem !important;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        margin: 3rem 0;
    }
    
    /* Answer section */
    .answer-section {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
        border: 1px solid var(--accent-primary);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.1);
    }
    
    .answer-section h3 {
        color: var(--accent-primary);
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* Citations styling */
    .citations-section {
        margin-top: 3rem;
    }
    
    .citations-section h3 {
        color: var(--accent-primary);
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    .citation-item {
        background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        border-left: 4px solid var(--accent-primary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-primary);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        border: 1px solid var(--border-color);
        backdrop-filter: blur(5px);
    }
    
    .citation-item:hover {
        transform: translateX(8px);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.2);
        border-left-color: var(--accent-secondary);
        border-color: var(--accent-primary);
    }
    
    .citation-number {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 0.8rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-secondary), var(--accent-primary));
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main h1 {
            font-size: 2.5rem;
        }
        
        .main h2 {
            font-size: 1.3rem;
        }
        
        .main .block-container {
            padding: 1rem;
        }
        
        .input-section {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🖋 CiteForgeAI: Automated Research & Citation Support")
st.markdown("Upload research papers (**PDF**, **DOCX**, **TXT**) or paste academic paper links (arXiv, PubMed), then ask your question below!")

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.subheader("📄 Upload Files or Provide Paper Links")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    uploaded_files = st.file_uploader(
        "Upload research documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with col2:
    urls = st.text_area(
        "Paste paper links (arXiv / PubMed, one per line)",
        placeholder="https://arxiv.org/abs/1234.5678\nhttps://pubmed.ncbi.nlm.nih.gov/...",
        height=100,
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Question Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.subheader("Ask a Research Question")
question = st.text_input(
    "Your question:",
    placeholder="e.g. What are recent deep learning methods in medical imaging?",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Process inputs
if (uploaded_files or urls.strip()) and question.strip():
    with st.spinner("🔍 Processing documents and generating your answer..."):
        # Step 1: Load documents from files and links
        documents = load_documents(uploaded_files, urls.splitlines())
        
        # Step 2: Create or load vector store
        vectorstore = create_or_load_vectorstore(documents)
        
        # Step 3: Ask the question using retrieval + LLM
        answer, citations = get_answer_with_citations(question, vectorstore)
        
        # Step 4: Display the answer with inline references
        st.markdown('<div class="answer-section">', unsafe_allow_html=True)
        st.markdown('### 💡 Answer')
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 5: Display citations in separate blocks
        st.markdown('<div class="citations-section">', unsafe_allow_html=True)
        st.markdown('### 📚 Citations')
        
        # Parse the grouped citations and display each in separate blocks
        grouped_citations_text = format_citations_grouped(citations)
        
        # Split by lines and process each citation
        lines = grouped_citations_text.split('\n')
        citation_counter = 1
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('**'):
                # Remove markdown list formatting
                if line.startswith('- '):
                    line = line[2:]
                elif line.startswith('* '):
                    line = line[2:]
                
                # Display each citation in a separate styled block
                if line:
                    st.markdown(f"""
                    <div class="citation-item">
                        <span class="citation-number">{citation_counter}</span>{line}
                    </div>
                    """, unsafe_allow_html=True)
                    citation_counter += 1
        
        st.markdown('</div>', unsafe_allow_html=True)

elif question and not (uploaded_files or urls.strip()):
    st.warning("⚠️ Please upload at least one document or provide academic links to proceed.")