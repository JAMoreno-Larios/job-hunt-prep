"""
tools.py

Here we define all tools to be used in our agent

J. A. Moreno
2026
"""

from typing import List, Tuple
from langchain.tools import tool
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.documents import Document

from . import llm_setup

# Initialize retriever
retriever = llm_setup.Retriever().retriever

# Define tools
@tool
def search_user_db(query: str, k: int = 5) -> Tuple[str, List[Document]]:
    """
    Semantically searches a vector store containing user's personal information
    matching the query.

    Args:
        query: Search terms to look for
        k: The maximum amount of desired results

    Returns:
        serialized: Serialized documents from the vector store
        retrieved_docs: Retrieved Documents from the vector store
    """
    try:
        retrieved_docs = retriever.invoke(str(query), k=k)
        # Serialize documents for the model
        serialized = "\n\n".join(
            (
                f"Source: {doc.metadata.get('source', 'Unknown')}"
                + f"\n\nContent: {doc.page_content}"
            )
            for doc in retrieved_docs
        )
        # Return serialized and raw documents
        return (serialized, retrieved_docs)

    except Exception:
        Exception("Search turned no elemments")
        return ("No elements found", [])


@tool
def scrap_job_posting(job_post_url: str) -> Tuple[str, List[Document]]:
    """
    Scraps the job post located in job_post_url and loads
    as a list of Documents using SeleniumURLLoader.

    Args:
        job_post_url: A valid URL containing a job post

    Returns:
        serialized: Serialized documents from the job post
        retrieved_post: Retrieved Documents from the job post

    """
    # Create the url loader
    loader = SeleniumURLLoader(
        [job_post_url],
    )
    # Load data
    retrieved_post = loader.load()
    # Serialize
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}"
            f"\n\nTitle: {doc.metadata.get('title', 'Unknown')}"
            + f"\n\nContent: {doc.page_content}"  # Specific to SeleniumURLLoader
        )
        for doc in retrieved_post
    )
    # Return serialized and raw docs
    return (serialized, retrieved_post)
