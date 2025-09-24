# app.py (Updated for V2)
import streamlit as st
from src.parser import extract_text
import time
import tempfile
import uuid
from pathlib import Path
from src.vector_store import VectorStore
from src.qa_chain import QASystem, LLMResponse
# from pdf_processor import create_vector_db_from_pdf # We can add error handling here
# from rag_pipeline import create_rag_chain_v2

# --- UI Setup ---
st.set_page_config(page_title="Chat with PDFs", layout="wide")
st.title("Chat with Your PDFs")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(collection_name=st.session_state.session_id)
if "qa_system" not in st.session_state:
    st.session_state.qa_system = QASystem()
if "pdfs" not in st.session_state:
    st.session_state.pdfs = []

# --- Sidebar for File Upload & Processing ---

with st.sidebar:
    st.header("Upload Your PDF")
    pdf_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)

    if pdf_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # UI Status Update: Show a spinner during processing
                with st.spinner("Processing PDFs... This may take a moment."):
                    for pdf_file in pdf_files:
                        if pdf_file.name in st.session_state.pdfs:
                            continue
                        temp_file_path = Path(temp_dir) / f"{pdf_file.name}"
                        with open(temp_file_path, "wb") as f:
                            f.write(pdf_file.getbuffer())
                        text = extract_text(temp_file_path) # Assuming this is V1's processor
                
                        if text:
                            st.session_state.vector_store.add_text(text, pdf_file.name)
                            st.session_state.pdfs.append(pdf_file.name)
                            st.info(f"{pdf_file.name} processed! Ask your questions now.")
            except Exception as e:
                st.error(f"An error occurred: {e}")


# --- Chat Interface ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Citations", expanded=True):
                for idx, doc in enumerate(message["citations"]):
                        st.markdown(f"{idx + 1}. {doc}")
        # Display interactive citations if they exist
        if "chunks" in message and message["chunks"]:
            with st.expander("Context", expanded=False):
                for idx, doc in enumerate(message["chunks"]):
                        st.write(f"Source {idx + 1}: {doc['metadata']['source']}")
                        st.caption(doc['text'])
                        st.divider()


# Main chat logic
if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store.empty:
        st.warning("Please upload and process a PDF file first.")
        st.stop()
        
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and thinking..."):
            # Retrieve documents first to use for interactive citations
            retrieved_docs = st.session_state.vector_store.search_and_rerank(prompt)

            # Handling No Chunks Found
            if not retrieved_docs:
                response = "Based on the provided document, I cannot answer this question."
                citations = []
                references = []
            else:
                response = st.session_state.qa_system.answer_query(prompt, retrieved_docs)
                answer = response.answer
                citations = response.citations
                if answer in ["I don't know.", "I could not find the answer in the provided document or our conversation history."]:
                    references = []
                else:
                    references = retrieved_docs
            
            st.markdown(answer)
            if citations:
                with st.expander("Citations", expanded=True):
                    for idx, doc in enumerate(citations):
                        st.markdown(f"{idx + 1}. {doc}")
            # Interactive Citations
            if retrieved_docs:
                with st.expander("Context", expanded=False):
                    for idx, doc in enumerate(retrieved_docs):
                        st.write(f"Source {idx + 1}: {doc['metadata']['source']}")
                        st.caption(doc['text'])
                        st.divider()

    # Add assistant response and citations to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "chunks": references
    })