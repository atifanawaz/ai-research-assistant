import os
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

from backend.loader import load_documents
from backend.embedder import create_or_load_vectorstore
from backend.rag_chain import get_answer_with_citations
from citations.citation_formatter import format_citations_grouped

st.set_page_config(
    page_title="AI Research Assistant", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme with coral accents */
    :root {
        --bg-primary: #0f1419;
        --bg-secondary: #1a2332;
        --bg-tertiary: #252d3d;
        --accent-coral: #d97563;
        --accent-coral-light: #ff7f6b;
        --accent-teal: #00d9d9;
        --text-primary: #ffffff;
        --text-secondary: #b0b8c1;
        --border-color: #3a4556;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: transparent;
    }
    
    .main .block-container {
        background: transparent;
        padding: 2rem 4rem;
        max-width: 1000px;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .viewerBadge_container__r0smo { display: none; }
    
    /* Title and headers */
    .main h1 {
        color: var(--text-primary);
        text-align: center;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .main .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .main h2 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.3rem;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .main h3 {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 1rem;
        margin: 0;
    }
    
    /* Markdown text */
    .main .stMarkdown p {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Input containers - Card styling */
    .input-card {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .input-card:hover {
        border-color: var(--accent-coral);
        box-shadow: 0 10px 40px rgba(217, 117, 99, 0.15);
    }
    
    .input-card h3 {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* File uploader */
    .stFileUploader {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stFileUploader > div {
        background: linear-gradient(135deg, rgba(217, 117, 99, 0.1), rgba(0, 217, 217, 0.05));
        border: 2px dashed var(--accent-coral);
        border-radius: 12px;
        padding: 2rem !important;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        background: linear-gradient(135deg, rgba(217, 117, 99, 0.15), rgba(0, 217, 217, 0.1));
        border-color: var(--accent-coral-light);
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: var(--bg-tertiary) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent-coral) !important;
        box-shadow: 0 0 0 3px rgba(217, 117, 99, 0.1) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.6;
    }
    
    .stTextArea label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Text input */
    .stTextInput input {
        background: var(--bg-tertiary) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent-coral) !important;
        box-shadow: 0 0 0 3px rgba(217, 117, 99, 0.1) !important;
    }
    
    .stTextInput input::placeholder {
        color: var(--text-secondary) !important;
    }
    
    .stTextInput label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-coral), var(--accent-coral-light));
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(217, 117, 99, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(0, 217, 217, 0.1)) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
    }
    
    .stSuccess .stMarkdown {
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        line-height: 1.6;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, rgba(217, 117, 99, 0.15), rgba(255, 127, 107, 0.1)) !important;
        border: 1px solid rgba(217, 117, 99, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
    }
    
    .stWarning .stMarkdown {
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Separator */
    .separator {
        text-align: center;
        color: var(--text-secondary);
        margin: 1.5rem 0;
        font-size: 0.9rem;
        position: relative;
    }
    
    .separator::before,
    .separator::after {
        content: '';
        position: absolute;
        top: 50%;
        width: 35%;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    }
    
    .separator::before {
        left: 0;
    }
    
    .separator::after {
        right: 0;
    }
    
    /* Feature cards */
    .feature-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: var(--accent-coral);
        transform: translateY(-4px);
        box-shadow: 0 10px 30px rgba(217, 117, 99, 0.15);
    }
    
    .feature-card-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: var(--accent-coral);
    }
    
    .feature-card h4 {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .feature-card p {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Citations styling */
    .citation-item {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border-left: 4px solid var(--accent-coral);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .citation-item:hover {
        transform: translateX(5px);
        border-left-color: var(--accent-coral-light);
        box-shadow: 0 8px 25px rgba(217, 117, 99, 0.15);
    }
    
    .citation-item strong {
        color: var(--accent-coral);
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-coral), var(--accent-teal));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-coral-light), var(--accent-teal));
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1>Your Intelligent Research Companion</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload research papers, provide academic links, and get instant insights with accurate citations</p>',
        unsafe_allow_html=True
    )

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<h3>📄 Upload Documents</h3>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload research documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)
if uploaded_files:
    st.markdown(f'<p style="color: #10b981; font-size: 0.9rem; margin-top: 0.5rem;">{len(uploaded_files)} file(s) uploaded</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="separator">or</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<h3>🔗 Paste Paper Links</h3>', unsafe_allow_html=True)
urls = st.text_area(
    "Paste paper links",
    placeholder="https://arxiv.org/abs/1234.5678\nhttps://pubmed.ncbi.nlm.nih.gov/...",
    height=100,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<h3>❓ Your Research Question</h3>', unsafe_allow_html=True)
question = st.text_input(
    "Your question",
    placeholder="e.g. What are recent deep learning methods in medical imaging?",
    label_visibility="collapsed"
)
st.button("🔍 Ask Your Question", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="feature-cards">', unsafe_allow_html=True)
st.markdown("""
    <div class="feature-card">
        <div class="feature-card-icon">📄</div>
        <h4>Multiple Formats</h4>
        <p>Support for PDF, DOCX, and TXT files</p>
    </div>
    <div class="feature-card">
        <div class="feature-card-icon">🔗</div>
        <h4>Academic Links</h4>
        <p>Direct access to arXiv and PubMed</p>
    </div>
    <div class="feature-card">
        <div class="feature-card-icon">⭐</div>
        <h4>Instant Citations</h4>
        <p>Get answers with source references</p>
    </div>
</div>
""", unsafe_allow_html=True)

if (uploaded_files or urls.strip()) and question.strip():
    with st.spinner("Processing documents and generating your answer..."):
        
        documents = load_documents(uploaded_files, urls.splitlines())
        vectorstore = create_or_load_vectorstore(documents)
        answer, citations = get_answer_with_citations(question, vectorstore)
        
        st.markdown("### Answer")
        st.success(answer)
        
        st.markdown("### Citations")
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
    st.warning("Please upload at least one document or provide academic links.")
