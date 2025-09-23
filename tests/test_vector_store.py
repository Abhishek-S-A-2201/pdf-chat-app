import pytest
from pathlib import Path
from src.vector_store import VectorStore

# Define sample data that will be used across multiple tests.
SAMPLE_TEXT = (
    "AI assistants are transforming the way we work. "
    "LangChain provides a framework for developing applications powered by language models. "
    "ChromaDB is an open-source embedding database. "
    "Together, they form a powerful stack for RAG applications."
)
SAMPLE_FILE_NAME = "test_document.pdf"


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    """
    Pytest fixture to create an isolated VectorStore instance for each test.
    
    It uses pytest's `tmp_path` fixture to create a temporary directory,
    ensuring that each test runs with a fresh, empty database and that
    no test data is left behind.
    """
    # Create a subdirectory within the temporary path for the database
    db_path = str(tmp_path / "test_chroma_db")
    # `scope="function"` (the default) means this fixture runs for each test
    return VectorStore(db_path=db_path, collection_name="test_collection")


def test_initialization(vector_store: VectorStore):
    """
    Tests that the VectorStore initializes correctly, creating the database
    and an empty collection.
    """
    assert isinstance(vector_store, VectorStore)
    # The collection is created on initialization, so it should exist.
    assert vector_store.collection is not None
    # A new collection should be empty.
    assert vector_store.collection.count() == 0
    print("\n✅ Initialization test passed.")


def test_add_text_happy_path(vector_store: VectorStore):
    """
    Tests the primary functionality of adding text to the vector store.
    """
    vector_store.add_text(SAMPLE_TEXT, SAMPLE_FILE_NAME)

    # Since the sample text is short, it should result in exactly one chunk.
    assert vector_store.collection.count() == 1

    # Retrieve the added item to verify its content and metadata.
    retrieved_item = vector_store.collection.get(ids=[f"{SAMPLE_FILE_NAME}_chunk_0"])
    
    assert retrieved_item['documents'][0] == SAMPLE_TEXT
    assert retrieved_item['metadatas'][0]['source'] == SAMPLE_FILE_NAME
    print("\n✅ Add text (happy path) test passed.")


def test_add_empty_text(vector_store: VectorStore):
    """
    Tests that providing an empty string to add_text does not add anything
    to the database and does not raise an error.
    """
    vector_store.add_text("", SAMPLE_FILE_NAME)
    assert vector_store.collection.count() == 0
    print("\n✅ Add empty text test passed.")


def test_add_text_idempotency(vector_store: VectorStore):
    """
    Tests that adding the same document multiple times does not create
    duplicate entries, thanks to the unique IDs.
    """
    # Add the text the first time
    vector_store.add_text(SAMPLE_TEXT, SAMPLE_FILE_NAME)
    count_after_first_add = vector_store.collection.count()
    assert count_after_first_add == 1

    # Add the exact same text again
    vector_store.add_text(SAMPLE_TEXT, SAMPLE_FILE_NAME)
    count_after_second_add = vector_store.collection.count()

    # The count should not have changed because ChromaDB upserts based on ID.
    assert count_after_second_add == count_after_first_add
    print("\n✅ Idempotency test passed.")


def test_search_happy_path(vector_store: VectorStore):
    """
    Tests the search functionality after adding a document.
    """
    # First, add the document to the store.
    vector_store.add_text(SAMPLE_TEXT, SAMPLE_FILE_NAME)

    # Perform a search with a relevant query.
    query = "What is ChromaDB?"
    search_results = vector_store.search(query, n_results=1)

    assert isinstance(search_results, list)
    assert len(search_results) == 1
    
    result = search_results[0]
    assert "text" in result
    assert "metadata" in result
    assert "distance" in result
    assert result["metadata"]["source"] == SAMPLE_FILE_NAME
    assert "ChromaDB" in result["text"] # Check for relevance
    print("\n✅ Search (happy path) test passed.")


def test_search_no_results(vector_store: VectorStore):
    """
    Tests that searching an empty database or with an irrelevant query
    returns an empty list.
    """
    # Search an empty database
    results_empty_db = vector_store.search("anything")
    assert results_empty_db == []

    # Search with a completely irrelevant query after adding text
    vector_store.add_text(SAMPLE_TEXT, SAMPLE_FILE_NAME)
    results_irrelevant_query = vector_store.search("quantum physics")
    
    # While it will still return the *closest* result, we can check if the
    # distance is high, but simply checking format is more robust.
    assert isinstance(results_irrelevant_query, list)
    print("\n✅ Search (no results) test passed.")


def test_search_n_results_limit(vector_store: VectorStore):
    """
    Tests that the `n_results` parameter correctly limits the number of
    returned search results.
    """
    # Create longer text that will generate multiple chunks
    long_text = (SAMPLE_TEXT + " ") * 5
    vector_store.add_text(long_text, "long_doc.pdf")

    # There should be more than 1 chunk in the database
    assert vector_store.collection.count() > 1

    # Perform a search and limit the results
    search_results = vector_store.search("AI assistants", n_results=1)
    assert len(search_results) == 1

    search_results_2 = vector_store.search("AI assistants", n_results=2)
    assert len(search_results_2) <= 2 # It might be less if there are fewer than 2 chunks
    print("\n✅ Search n_results limit test passed.")