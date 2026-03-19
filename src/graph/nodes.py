"""
Implementation for our workflow nodes
J. A. Moreno
2026
"""

from pathlib import Path
from state import JobPrepState
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.messages import HumanMessage
from langchain_chroma import Chroma
from langgraph.types import Command, Literal
from langchain_community.document_loaders.url import UnstructuredURLLoader

import llm_setup

# Load environment variables
load_dotenv()

# Initialize LLM
llm = llm_setup.LLM()

### DATA NODES
# Search in vector store
# TODO: Check how to turn this into a proper node
def search_user_db(state: JobPrepState):
    """
    Retrieves the relevant information for the query
    """
    query = state.get('user_query')
    retrieved_docs = llm.retriever.invoke(query, k=10)
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


## USER INPUT PROCESSING NODES


## ANSWER GENERATION NODES
