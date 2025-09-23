# PDF Chat Application

**Task:** Build a Streamlit app that lets a user upload any PDF and chat with it.

**Flow:** PDF upload → chunking → embeddings → store in vector DB → retrieve → Q&A with citations

## Features

- Upload and process multiple PDF documents
- Chat interface for asking questions about the document content
- Automatic handling of both text-based and scanned PDFs
- Citation of sources for answers
- Persistent chat history within the session

## Setup

### Prerequisites

- Tesseract OCR (for processing scanned PDFs)
- OpenAI API key

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Install Tesseract OCR:

   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
5. Create a `.env` file in the project root with your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
6. Run the application:

   💡 **Note:** The first time you run the application, it may take a few minutes to start. This is because it needs to download the necessary embedding and re-ranking models (e.g., `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2`). This is a one-time process; subsequent launches will be much faster.

   ```bash
   streamlit run app.py
   ```

## Models and Technologies

### Language Model

- **Provider**: OpenAI
- **Model**: gpt-4o-mini
- **Size:** approx. 8b
- **Temperature**: 0.5
- **Reason:** This model was selected for its optimal balance of performance, cost, and speed. While more powerful models exist, `gpt-4o-mini` provides high-quality, relevant answers for RAG tasks where the necessary information is contained within the provided context. A `temperature` of **0.5** is a deliberate choice to ensure the model’s responses are factual and grounded in the source material, rather than being overly creative or prone to hallucination.

### Embedding Model

- **Model**: all-MiniLM-L6-v2 (default ChromaDB embedding model)
- **Vector Database**: ChromaDB (persistent on-disk storage)
- **Reason:**
  - ChromaDB was chosen for its simplicity and suitability for local, file-based applications. It runs entirely on the local machine and stores the data persistently on disk, meeting the requirement for a local vector store without needing to set up a separate server or cloud service. This makes the application easy to set up and run with a simple `pip install`.
  - The Embedding model is a strong, open-source choice. It is highly efficient for its size and provides excellent performance for semantic search tasks. It's often the default and recommended embedding model for lightweight vector databases like ChromaDB due to its small footprint and good out-of-the-box results, eliminating the need for a separate API and associated costs.

### **Chunking Strategy**:

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Splitter**: RecursiveCharacterTextSplitter from LangChain
- **Reason:** The choice of the `RecursiveCharacterTextSplitter` is a standard and highly effective strategy in RAG. This method intelligently splits text by trying to break it down using a list of delimiters (like `\n\n`, `\n`, `.` etc.). The goal is to keep semantically related sentences and paragraphs together in a single chunk. The **1000-character chunk size** and **200-character overlap** are common, well-tested values that balance granularity with contextual awareness. The overlap helps prevent the loss of information at the boundaries of the chunks.

### **Re-ranking Model**

- **Model**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Technology**: CrossEncoder from the sentence-transformers library
- **Reason**: To improve the quality of answers, this application uses a two-stage retrieval process.
- **Retrieval (Recall)**: First, the vector database quickly retrieves a large set of documents (e.g., 20) that are semantically similar to the query.
- **Re-ranking (Precision)**: Next, the Re-ranking model takes this smaller set of documents and the original query to compute a more accurate relevance score. It re-orders the documents to place the most contextually relevant ones at the top. This ensures that the context provided to the LLM is of the highest possible quality, significantly reducing noise and leading to more accurate, cited answers.

### PDF Processing

- **Text Extraction**: PyMuPDF (fitz) for text-based PDFs
- **OCR Fallback**: Tesseract OCR for scanned/image-based PDFs
  - **DPI**: 300 (for better OCR accuracy)
  - **Language**: English
- **Reason:** The two-step approach is designed for robustness.
  - **PyMuPDF** is an extremely fast and reliable library for extracting text from PDFs that have a native text layer. It is the primary method because it's non-resource intensive and highly accurate.
  - **Tesseract** is used as a fallback. For scanned documents or PDFs that are essentially just images, PyMuPDF's text extraction will fail. Tesseract performs Optical Character Recognition (OCR) on these documents by converting each page into an image and then extracting the text. Using a **DPI of 300** is a best practice for OCR to ensure higher resolution images, leading to better text recognition accuracy.

## Conversation History

The application maintains conversation history in the following ways:

1. **Session-based History**:

   - Each session is assigned a unique ID
   - Conversation history is stored in `st.session_state.messages`
   - History includes both user questions and AI responses
2. **Context Window**:

   - The full conversation history is passed to the LLM with each query
   - This allows the model to maintain context and reference previous interactions
3. **Vector Store**:

   - Each session creates a new ChromaDB collection
   - Document chunks and their embeddings are stored persistently
   - The collection is named using the session ID for isolation

## Known Limitations

1. **Document Size**:

   - Large documents may take significant time to process
   - Memory usage increases with document size
2. **OCR Limitations**:

   - Scanned documents require OCR processing which is slower
   - OCR accuracy depends on document quality and formatting
   - Complex layouts may not be processed correctly
3. **Context Window**:

   - The LLM has a limited context window
   - Very long conversations may exceed this limit
4. **File Types**:

   - Only PDF files are supported
   - Password-protected PDFs cannot be processed
5. **Performance**:

   - Initial processing of large documents may be slow
   - Response time depends on the complexity of the query and document size

## Troubleshooting

- **Missing Dependencies**: Ensure all Python packages are installed from `requirements.txt`
- **Tesseract Not Found**: Verify Tesseract is installed and in your system PATH
- **API Key Issues**: Check that your OpenAI API key is correctly set in the `.env` file
- **PDF Processing Errors**: Try with a different PDF file if one fails to process
