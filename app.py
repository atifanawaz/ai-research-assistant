import streamlit as st
from config import GROQ_API_KEY
from backend.loader import load_documents
from backend.embedder import create_or_load_vectorstore
from backend.rag_chain import get_answer_with_citations
from citations.citation_formatter import format_citations_grouped

st.set_page_config(page_title="AI Research Assistant", layout="centered")

# Dark theme CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme variables */
    :root {
        --bg-primary: #0f0f23;
        --bg-secondary: #1a1a2e;
        --bg-tertiary: #16213e;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --border-color: #374151;
        --success-color: #10b981;
        --warning-color: #f59e0b;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-tertiary) 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        background: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
    }
    
    /* Headers styling */
    .main h1 {
        color: var(--text-primary);
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main h2, .main h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .main h2 {
        border-bottom: 2px solid var(--accent-primary);
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    .main h3 {
        color: var(--accent-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Markdown text */
    .main .stMarkdown p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        line-height: 1.6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* File uploader */
    .stFileUploader {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px dashed var(--accent-primary);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-secondary);
        background: var(--bg-tertiary);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text-secondary);
    }
    
    /* Text area */
    .stTextArea {
        margin: 1rem 0;
    }
    
    .stTextArea label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    .stTextArea textarea {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.7;
    }
    
    /* Text input */
    .stTextInput {
        margin: 1rem 0;
    }
    
    .stTextInput label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    .stTextInput input {
        background: var(--bg-secondary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.7;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, var(--success-color), #34d399) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stSuccess .stMarkdown {
        color: white !important;
        font-weight: 500;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, var(--warning-color), #fbbf24) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3) !important;
    }
    
    .stWarning .stMarkdown {
        color: white !important;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        margin: 2rem 0;
    }
    
    .stSpinner > div {
        border-color: var(--accent-primary) !important;
    }
    
    /* Citations styling - Individual blocks */
    .citation-item {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border-left: 4px solid var(--accent-primary);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: var(--text-primary);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        border: 1px solid var(--border-color);
    }
    
    .citation-item:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.2);
        border-left-color: var(--accent-secondary);
    }
    
    /* Footer */
    .main hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
        margin: 3rem 0 2rem 0;
    }
    
    .main .stCaption {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.8;
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
    
    /* Custom glow effects */
    .glow-text {
        text-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
    }
    
    /* File upload success indicator */
    .upload-success {
        background: linear-gradient(135deg, var(--success-color), #34d399);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Research Assistant with Citation Support")
st.markdown("Upload research papers (**PDF**, **DOCX**, **TXT**) or paste academic paper links (arXiv, PubMed), then ask your question below!")

# Main UI
st.subheader("Upload Files or Provide Paper Links")

# Upload PDF, DOCX, or TXT files
uploaded_files = st.file_uploader(
    "Upload research documents", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True)

# Input paper links (arXiv, PubMed)
urls = st.text_area("Paste paper links (arXiv / PubMed, one per line)", placeholder="https://arxiv.org/abs/1234.5678")

# Ask a research question
st.subheader("Ask a Research Question")
question = st.text_input("Your question:", placeholder="e.g. What are recent deep learning methods in medical imaging?")

# Process inputs if all are ready
if (uploaded_files or urls.strip()) and question.strip():
    with st.spinner("Processing documents and generating your answer..."):
        
        # Step 1: Load documents from files and links
        documents = load_documents(uploaded_files, urls.splitlines())
        
        # Step 2: Create or load vector store
        vectorstore = create_or_load_vectorstore(documents)
        
        # Step 3: Ask the question using retrieval + LLM
        answer, citations = get_answer_with_citations(question, vectorstore)
        
        # Step 4: Display the answer with inline references
        st.markdown("### Answer")
        st.success(answer)
        
        # Step 5: Display citations in separate blocks
        st.markdown("### Citations")
        
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
    st.warning("Please upload at least one document or provide academic links.")

