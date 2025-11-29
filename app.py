import os
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
from backend.loader import load_documents
from backend.embedder import create_or_load_vectorstore
from backend.rag_chain import get_answer_with_citations
from citations.citation_formatter import format_citations_grouped

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# Beautiful Modern CSS with Coral Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Color Palette */
    :root {
        --coral: #E8695B;
        --coral-dark: #D85A48;
        --navy: #2C3E50;
        --navy-light: #34495E;
        --cream: #F5F3F0;
        --white: #FFFFFF;
        --text-dark: #1A1A1A;
        --text-light: #6B7280;
        --gold: #F4B942;
        --success: #10B981;
        --warning: #F59E0B;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, var(--cream) 0%, #FAFAF8 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        max-width: 900px;
        padding: 2.5rem 2rem;
    }
    
    /* Title Styling */
    .main h1 {
        color: var(--navy);
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle */
    .main .stMarkdown p {
        color: var(--text-light);
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2.5rem;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Subheader */
    .main h2 {
        color: var(--navy);
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        position: relative;
        padding-left: 1rem;
    }
    
    .main h2::before {
        content: '';
        position: absolute;
        left: 0;
        width: 4px;
        height: 28px;
        background: var(--coral);
        border-radius: 2px;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: var(--white);
        border: 2px solid #E5E7EB;
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .stFileUploader:hover {
        border-color: var(--coral);
        box-shadow: 0 12px 30px rgba(232, 105, 91, 0.15);
    }
    
    .stFileUploader label {
        color: var(--navy) !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(232, 105, 91, 0.05), rgba(244, 185, 66, 0.05));
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text-light) !important;
        font-size: 0.95rem;
    }
    
    /* Text Area */
    .stTextArea label {
        color: var(--navy) !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    .stTextArea textarea {
        background: var(--white) !important;
        border: 2px solid #E5E7EB !important;
        border-radius: 12px !important;
        color: var(--text-dark) !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--coral) !important;
        box-shadow: 0 0 0 4px rgba(232, 105, 91, 0.1) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-light) !important;
    }
    
    /* Text Input */
    .stTextInput label {
        color: var(--navy) !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    .stTextInput input {
        background: var(--white) !important;
        border: 2px solid #E5E7EB !important;
        border-radius: 12px !important;
        color: var(--text-dark) !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--coral) !important;
        box-shadow: 0 0 0 4px rgba(232, 105, 91, 0.1) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder {
        color: var(--text-light) !important;
    }
    
    /* Success Message - Answer Box */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(244, 185, 66, 0.05)) !important;
        border: 2px solid var(--success) !important;
        border-radius: 16px !important;
        padding: 1.8rem !important;
        margin: 2rem 0 !important;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.1) !important;
    }
    
    .stSuccess .stMarkdown {
        color: var(--text-dark) !important;
        font-size: 1.05rem !important;
        line-height: 1.8 !important;
    }
    
    /* Warning Message */
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05), rgba(244, 185, 66, 0.05)) !important;
        border: 2px solid var(--warning) !important;
        border-radius: 16px !important;
        padding: 1.8rem !important;
        margin: 2rem 0 !important;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.1) !important;
    }
    
    .stWarning .stMarkdown {
        color: var(--text-dark) !important;
        font-size: 1.05rem !important;
        font-weight: 500;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        margin: 2.5rem 0;
    }
    
    .stSpinner > div > div {
        border-top-color: var(--coral) !important;
    }
    
    /* Citation Items */
    .citation-item {
        background: var(--white);
        border-left: 5px solid var(--coral);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-dark);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 1px solid #E5E7EB;
    }
    
    .citation-item:hover {
        transform: translateX(6px);
        box-shadow: 0 8px 25px rgba(232, 105, 91, 0.15);
        border-left-color: var(--gold);
    }
    
    /* Citation Counter Badge */
    .citation-item strong {
        color: var(--coral);
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Divider */
    .main hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #D4CCC8, transparent);
        margin: 3rem 0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--cream);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--coral), var(--gold));
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--coral-dark), var(--coral));
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 AI Research Assistant")
st.markdown("Upload research papers (**PDF**, **DOCX**, **TXT**) or paste academic paper links (arXiv, PubMed), then ask your question below!")

# Main UI
st.subheader("📤 Upload Files or Provide Paper Links")

# Upload PDF, DOCX, or TXT files
uploaded_files = st.file_uploader(
    "Upload research documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Input paper links (arXiv, PubMed)
urls = st.text_area(
    "Paste paper links (arXiv / PubMed, one per line)",
    placeholder="https://arxiv.org/abs/1234.5678\nhttps://pubmed.ncbi.nlm.nih.gov/..."
)

# Ask a research question
st.subheader("❓ Ask a Research Question")
question = st.text_input(
    "Your question:",
    placeholder="e.g. What are recent deep learning methods in medical imaging?"
)

# Process inputs if all are ready
if (uploaded_files or urls.strip()) and question.strip():
    with st.spinner("🔍 Processing documents and generating your answer..."):
        
        # Step 1: Load documents from files and links
        documents = load_documents(uploaded_files, urls.splitlines())
        
        # Step 2: Create or load vector store
        vectorstore = create_or_load_vectorstore(documents)
        
        # Step 3: Ask the question using retrieval + LLM
        answer, citations = get_answer_with_citations(question, vectorstore)
        
        # Step 4: Display the answer with inline references
        st.markdown("### 💡 Answer")
        st.success(answer)
        
        # Step 5: Display citations in separate blocks
        st.markdown("### 📖 Citations")
        
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
                        <strong>[{citation_counter}]</strong> {line}
                    </div>
                    """, unsafe_allow_html=True)
                    citation_counter += 1

elif question and not (uploaded_files or urls.strip()):
    st.warning("⚠️ Please upload at least one document or provide academic links.")