import os
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

from backend.loader import load_documents
from backend.embedder import create_or_load_vectorstore
from backend.rag_chain import get_answer_with_citations
from citations.citation_formatter import format_citations_grouped

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# <CHANGE> Complete redesign with coral/salmon and dark navy theme matching page.tsx
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Color scheme */
    :root {
        --bg-primary: #0f0f23;
        --bg-secondary: #1a1a2e;
        --bg-tertiary: #16213e;
        --accent: #ff6b6b;
        --accent-alt: #ff5252;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --border-color: #374151;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit header and footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0;
    }
    
    /* Header */
    .header-container {
        border-bottom: 1px solid #374151;
        background: rgba(26, 26, 46, 0.5);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        margin-bottom: 3rem;
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #ff6b6b, #ff5252);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    .header-text h1 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    
    .header-text p {
        color: #a1a1aa;
        font-size: 0.875rem;
        margin: 0.25rem 0 0 0;
    }
    
    /* Main content wrapper */
    .content-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .hero-section h2 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 1rem;
    }
    
    .hero-section p {
        color: #a1a1aa;
        font-size: 1.125rem;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Main card */
    .main-card {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid #374151;
        border-radius: 1.5rem;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        margin-bottom: 3rem;
    }
    
    /* Form sections */
    .form-section {
        margin-bottom: 2rem;
    }
    
    .form-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .form-label-icon {
        color: #ff6b6b;
        font-size: 1.25rem;
    }
    
    /* File upload */
    .file-upload-area {
        border: 2px dashed rgba(255, 107, 107, 0.4);
        border-radius: 0.75rem;
        padding: 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background: transparent;
    }
    
    .file-upload-area:hover {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.05);
    }
    
    .file-upload-icon {
        font-size: 2rem;
        color: #ff6b6b;
        margin-bottom: 0.5rem;
    }
    
    .file-upload-text {
        color: #ffffff;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    .file-upload-hint {
        color: #a1a1aa;
        font-size: 0.875rem;
    }
    
    /* File list */
    .file-list-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(22, 33, 62, 0.5);
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        color: #f5f5f7;
        margin-top: 1rem;
    }
    
    .file-list-icon {
        color: #ff6b6b;
        font-size: 1rem;
    }
    
    /* Divider */
    .divider-section {
        position: relative;
        margin: 2rem 0;
    }
    
    .divider-line {
        border-top: 1px solid #374151;
    }
    
    .divider-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(26, 26, 46, 0.6);
        padding: 0 0.75rem;
        color: #a1a1aa;
        font-size: 0.875rem;
    }
    
    /* Textarea */
    .stTextArea textarea {
        background: #16213e !important;
        border: 1px solid #374151 !important;
        border-radius: 0.75rem !important;
        color: #f5f5f7 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ff6b6b !important;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.2) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #a1a1aa !important;
        opacity: 0.7;
    }
    
    /* Text input */
    .stTextInput input {
        background: #16213e !important;
        border: 1px solid #374151 !important;
        border-radius: 0.75rem !important;
        color: #f5f5f7 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #ff6b6b !important;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.2) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder {
        color: #a1a1aa !important;
        opacity: 0.7;
    }
    
    /* Submit button */
    .submit-btn {
        width: 100%;
        background: linear-gradient(to right, #ff6b6b, #ff5252);
        color: #ffffff;
        font-weight: 600;
        padding: 0.75rem;
        border: none;
        border-radius: 0.75rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 1rem;
        margin-top: 1rem;
        box-shadow: 0 0 30px rgba(255, 107, 107, 0.2);
    }
    
    .submit-btn:hover {
        background: linear-gradient(to right, #ff5555, #ff4444);
        box-shadow: 0 0 40px rgba(255, 107, 107, 0.4);
    }
    
    .submit-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* Info/Warning messages */
    .info-message {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: #ffa5a5;
        margin-top: 1rem;
    }
    
    .stWarning {
        background: rgba(255, 107, 107, 0.1) !important;
        border: 1px solid rgba(255, 107, 107, 0.3) !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
        color: #ffa5a5 !important;
    }
    
    /* Answer section */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1.5rem !important;
        color: #ffffff !important;
        margin: 1rem 0 !important;
    }
    
    /* Citations */
    .citations-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .citation-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #374151;
        border-left: 4px solid #ff6b6b;
        border-radius: 0.75rem;
        padding: 1.5rem;
        color: #ffffff;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .citation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.2);
        border-left-color: #ff5252;
    }
    
    .citation-number {
        color: #ff6b6b;
        font-weight: 700;
        font-size: 0.875rem;
        margin-right: 0.5rem;
    }
    
    /* Features grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 3rem;
    }
    
    .feature-card {
        background: rgba(26, 26, 46, 0.4);
        border: 1px solid #374151;
        border-radius: 0.75rem;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #ff6b6b;
        background: rgba(26, 26, 46, 0.6);
    }
    
    .feature-icon {
        width: 48px;
        height: 48px;
        background: rgba(255, 107, 107, 0.2);
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #ff6b6b;
    }
    
    .feature-title {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #a1a1aa;
        font-size: 0.875rem;
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.25rem;
        margin: 2rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header-icon {
        color: #ff6b6b;
        font-size: 1.5rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #ff6b6b !important;
        border-top-color: rgba(255, 107, 107, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# <CHANGE> Header with icon and branding
st.markdown("""
<div class="header-container">
    <div class="header-content">
        <div class="header-icon">📚</div>
        <div class="header-text">
            <h1>AI Research Assistant</h1>
            <p>Powered by advanced AI</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# <CHANGE> Main content wrapper
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# <CHANGE> Hero section
st.markdown("""
<div class="hero-section">
    <h2>Your Intelligent Research Companion</h2>
    <p>Upload research papers, provide academic links, and get instant insights with citations</p>
</div>
""", unsafe_allow_html=True)

# <CHANGE> Main card container
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# <CHANGE> Upload section with improved styling
st.markdown("""
<div class="form-section">
    <div class="form-label">
        <span class="form-label-icon">📄</span>
        Upload Documents
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop files or click to upload",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Display uploaded files
if uploaded_files:
    for file in uploaded_files:
        st.markdown(f"""
        <div class="file-list-item">
            <span class="file-list-icon">📋</span>
            <span>{file.name}</span>
        </div>
        """, unsafe_allow_html=True)

# <CHANGE> Divider
st.markdown("""
<div class="divider-section">
    <div class="divider-line"></div>
    <div class="divider-text">or</div>
</div>
""", unsafe_allow_html=True)

# <CHANGE> URL section
st.markdown("""
<div class="form-section">
    <div class="form-label">
        <span class="form-label-icon">🔗</span>
        Paste Paper Links
    </div>
</div>
""", unsafe_allow_html=True)

urls = st.text_area(
    "Paste paper links",
    placeholder="https://arxiv.org/abs/1234.5678\nhttps://pubmed.ncbi.nlm.nih.gov/...",
    height=100,
    label_visibility="collapsed"
)

# <CHANGE> Question section
st.markdown("""
<div class="form-section">
    <div class="form-label">
        <span class="form-label-icon">⚡</span>
        Your Research Question
    </div>
</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Your question",
    placeholder="e.g., What are recent deep learning methods in medical imaging?",
    label_visibility="collapsed"
)

# <CHANGE> Submit button logic
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button(
        "Ask Your Question",
        use_container_width=True,
        type="primary" if (uploaded_files or urls.strip()) and question.strip() else "secondary"
    )

# <CHANGE> Info message
if not uploaded_files and not urls.strip():
    st.markdown("""
    <div class="info-message">
        Please upload at least one document or provide academic links to get started.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main card

# ... existing code ...

# Process inputs if all are ready
if submit_button and (uploaded_files or urls.strip()) and question.strip():
    with st.spinner("Processing documents and generating your answer..."):
        
        # Step 1: Load documents from files and links
        documents = load_documents(uploaded_files, urls.splitlines())
        
        # Step 2: Create or load vector store
        vectorstore = create_or_load_vectorstore(documents)
        
        # Step 3: Ask the question using retrieval + LLM
        answer, citations = get_answer_with_citations(question, vectorstore)
        
        # <CHANGE> Display answer section with new styling
        st.markdown("""
        <div style="margin-top: 3rem;">
            <div class="section-header">
                <span class="section-header-icon">✓</span>
                Answer
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(answer)
        
        # <CHANGE> Display citations in card grid
        st.markdown("""
        <div class="section-header">
            <span class="section-header-icon">📚</span>
            Citations
        </div>
        <div class="citations-container">
        """, unsafe_allow_html=True)
        
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
                    <div class="citation-card">
                        <span class="citation-number">[{citation_counter}]</span>{line}
                    </div>
                    """, unsafe_allow_html=True)
                    citation_counter += 1
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close citations container
        
        # <CHANGE> Features section
        st.markdown("""
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Multiple Formats</div>
                <div class="feature-description">Support for PDF, DOCX, and TXT files</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔗</div>
                <div class="feature-title">Academic Links</div>
                <div class="feature-description">Direct access to arXiv and PubMed</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Instant Citations</div>
                <div class="feature-description">Get answers with source references</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif question and not (uploaded_files or urls.strip()):
    st.warning("Please upload at least one document or provide academic links.")

st.markdown('</div>', unsafe_allow_html=True)  # Close content wrapper