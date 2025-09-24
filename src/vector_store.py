import chromadb
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from src.qa_chain import HypotheticalQACache
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    """
    A class to handle text chunking, embedding, and storage in a ChromaDB
    vector database, as well as retrieval.
    """
    def __init__(self, db_path: str = "storage", collection_name: str = "pdf_documents"):
        """
        Initializes the VectorStore.

        Args:
            db_path: The directory path to persist the ChromaDB database.
            collection_name: The name of the collection to store documents in.
        """
        logging.info(f"Initializing VectorStore with database path: {db_path}")
        # Use a persistent client to save the database to disk
        self.client = chromadb.PersistentClient(path=db_path)
        self.empty = True
        
        # Get or create the collection. ChromaDB uses a default embedding model
        # (all-MiniLM-L6-v2) which is great for general-purpose use.
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # Provides context between chunks
            length_function=len,
        )

        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logging.info("CrossEncoder model for re-ranking initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize CrossEncoder model: {e}")
            self.reranker = None
        logging.info("VectorStore initialized successfully.")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding for a given text string using the
        collection's configured embedding model.

        This is useful for tasks like semantic similarity checks without
        needing to store the text in the database.

        Args:
            text: The string to embed.

        Returns:
            The embedding as a list of floats, or an empty list if it fails.
        """
        if not text or not text.strip():
            logging.warning("embed_text called with an empty string.")
            return []

        try:
            # The collection's embedding function expects a batch (list) of texts.
            # We access it, pass a list containing our single text, and get a list
            # of embeddings back. We then return the first (and only) one.
            embedding = self.collection._embedding_function([text])
            return embedding[0]
        except Exception as e:
            logging.error(f"Failed to generate embedding for text: '{text[:50]}...'. Error: {e}")
            return []

    def add_text(self, text: str, pdf_file_name: str):
        """
        Chunks the given text and stores the chunks in the vector database
        with metadata.

        Args:
            text: The text content extracted from a PDF.
            pdf_file_name: The name of the source PDF file for metadata.
        """
        if not text:
            logging.warning(f"No text provided for '{pdf_file_name}'. Skipping storage.")
            return

        logging.info(f"Processing and storing text from '{pdf_file_name}'...")
        chunks = self.text_splitter.split_text(text)

        # Create metadata for each chunk
        metadatas = [{"source": pdf_file_name} for _ in chunks]

        # Create unique IDs for each chunk to prevent duplicates
        ids = [f"{pdf_file_name}_chunk_{i}" for i in range(len(chunks))]

        try:
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logging.info(f"Successfully added {len(chunks)} chunks from '{pdf_file_name}'.")
            self.empty = False
        except Exception as e:
            logging.error(f"Failed to add document chunks to ChromaDB: {e}")
    
    def add_qas(self, qa_cache: HypotheticalQACache):
        """
        Adds a QAWithCitation object to the vector database.

        Args:
            qa: The QAWithCitation object to add.
        """
        try:

            if len(qa_cache.qas) == 0:
                logging.warning("No QAWithCitation object provided. Skipping storage.")
                return

            chunks = []
            metadatas = []
            ids = []
            
            for qa in qa_cache.qas:
                chunks.append(qa.question)
                metadatas.append({"source_chunks": json.dumps(qa.source_chunks), "citations": json.dumps(qa.citations), "answer": qa.answer})
                ids.append(f"qa_{uuid.uuid4()}")

            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )

            logging.info(f"Successfully added QAWithCitation object to ChromaDB.")
        except Exception as e:
            logging.error(f"Failed to add QAWithCitation object to ChromaDB: {e}")
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Retrieves all document chunks from the vector database.

        Returns:
            A list of dictionaries, each containing the chunk text,
            metadata, and similarity score (distance).
        """
        try:
            results = self.collection.get()

            documents = []
            if results and results.get('documents'):
                docs = results['documents']
                metas = results['metadatas']
                
                for doc, meta in zip(docs, metas):

                    documents.append({
                        "text": doc,
                        "metadata": meta
                    })

            return documents
        except Exception as e:
            logging.error(f"Failed to retrieve document chunks from ChromaDB: {e}")
            return []

    def search(self, query: str, n_results: int = 5, similarity_threshold: float | None = None) -> List[Dict[str, Any]]:
        """
        Searches the vector database for the most relevant text chunks.

        Args:
            query: The search query string.
            n_results: The number of results to return.

        Returns:
            A list of dictionaries, each containing the chunk text,
            metadata, and similarity score (distance).
        """
        logging.info(f"Searching for query: '{query}'...")
        try:
            results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                )

            # Format the results for easier use
            formatted_results = []
            if results and results.get('documents'):
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                dists = results['distances'][0]

                for doc, meta, dist in zip(docs, metas, dists):
                    if similarity_threshold and dist > similarity_threshold:
                        continue
                    
                    formatted_results.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist
                    })

            logging.info(f"Found {len(formatted_results)} relevant chunks.")
            return formatted_results
        except Exception as e:
            logging.error(f"Failed to perform search in ChromaDB: {e}")
            return []

    def search_and_rerank(
        self,
        query: str,
        n_initial_results: int = 20,
        n_final_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs a vector search and then re-ranks the results for higher relevance.

        Args:
            query: The search query string.
            n_initial_results: The number of documents to retrieve initially.
            n_final_results: The final number of top documents to return after re-ranking.

        Returns:
            A sorted list of the most relevant document chunks.
        """
        if not self.reranker:
            logging.warning("Re-ranker not initialized. Falling back to standard search.")
            return self.search(query, n_results=n_final_results)

        logging.info(f"Performing search and re-ranking for query: '{query}'...")
        
        # 1. Initial Retrieval (Recall Stage)
        # We retrieve a larger number of documents initially.
        initial_results = self.search(query, n_results=n_initial_results)
        if not initial_results:
            return []

        # 2. Re-ranking Stage (Precision Stage)
        # Create pairs of [query, document_text] for the cross-encoder
        doc_texts = [result['text'] for result in initial_results]
        pairs = [[query, doc_text] for doc_text in doc_texts]

        # Predict relevance scores
        logging.info(f"Re-ranking {len(initial_results)} initial results...")
        scores = self.reranker.predict(pairs)

        # 3. Combine results with new scores and sort
        for result, score in zip(initial_results, scores):
            result['relevance_score'] = float(score)

        # Sort results by the new relevance score in descending order
        reranked_results = sorted(initial_results, key=lambda x: x['relevance_score'], reverse=True)

        # 4. Return the top N final results
        final_results = reranked_results[:n_final_results]
        logging.info(f"Returning {len(final_results)} re-ranked results.")
        
        return final_results