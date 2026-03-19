"""
Implementation for our search and web retrieval nodes
J. A. Moreno
2026
"""

from state import JobPrepState
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.messages import HumanMessage
from langchain_chroma import Chroma
from langgraph.types import Command, Literal
from langchain_community.document_loaders.url import UnstructuredURLLoader

# Load environment variables
load_dotenv()

# Define the vector store location
vector_store_path = Path(__file__).absolute().parents[2] / "./data/embeddings/"

# Define constants
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 100
EMBEDDINGS_MODEL = "nomic-embed-text-v2-moe"
LLM_MODEL = "qwen3.5:9b"
NUM_CTX = 8000

# Initialize components
llm = ChatOllama(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
vector_store = Chroma(persist_directory=vector_store_path.__str__(),
                      embedding_function=embeddings)

# Search in vector store
# TODO: Check how to turn this into a proper node
def search_user_db(state: JobPrepState):
    """
    Retrieves the relevant information for the query
    """
    query = state.get('user_query')
    retrieved_docs = vector_store.as_retriever().invoke(query, k=10)
    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}" +
                    f"\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    # Return serialized and raw documents
    return {"serialized_documents": serialized,
            "retrieved_documents": retrieved_docs}

# Scrap the job posting URL
# TODO: Check how to turn this into a proper node
def scrap_job_posting(state: JobPrepState):
    """
    Scraps the job post and loads it as a document 
    using the UnstructuredURLLoader.
    """
    job_url = state.get('job_post_url')
    # Create the url loader
    loader = UnstructuredURLLoader([job_url])
    # Load data
    retrieved_post = loader.load()
    # Serialize
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}" +
                    f"\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_post
    )
    # Return serialized and raw docs
    return {"serialized_job_post": serialized,
            "job_post_documents": retrieved_post}
