"""
tools.py

Here we define all tools to be used in our agent

J. A. Moreno
2026
"""

from langchain.tools import tool, ToolRuntime
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.messages import ToolMessage
from langgraph.types import Command

from . import llm_setup

# Initialize retriever
retriever = llm_setup.Retriever().retriever

# Define tools
@tool
def search_user_db(runtime: ToolRuntime) -> Command:
    """
    Semantically searches a vector store containing user's personal information
    matching the query.

    Sets:
        serialized_documents: Serialized documents from the vector store
        retrieved_documents: Retrieved Documents from the vector store
    """

    query = runtime.state["distilled_query"]
    try:
        retrieved_docs = retriever.invoke(str(query), k=5)
        # Serialize documents for the model
        serialized = "\n\n".join(
            (
                f"Source: {doc.metadata.get('source', 'Unknown')}"
                + f"\n\nContent: {doc.page_content}"
            )
            for doc in retrieved_docs
        )
        # Return serialized and raw documents
        return Command(
                update={
                    "serialized_documents": serialized,
                    "retrieved_documents": retrieved_docs,
                    "messages": [
                    ToolMessage(
                        content="Retrieved docs!",
                        tool_call_id=runtime.tool_call_id
                    )
                    ]
                }
        )

    except Exception:
        Exception("Search turned no elemments")
        return Command(
                update={
                    "serialized_documents": None,
                    "retrieved_documents": None,
                    "messages": [
                    ToolMessage(
                        content="Failed to retrieve docs",
                        tool_call_id=runtime.tool_call_id
                    )
                    ]
                }
        )


@tool
def scrap_job_posting(runtime: ToolRuntime) -> Command:
    """
    Scraps the job post located in job_post_url and loads
    as a list of Documents using SeleniumURLLoader.

    Sets the state as follows:
        serialized_job_post: Serialized documents from the job post
        job_post_documents: Retrieved Documents from the job post

    """
    job_post_url = runtime.state["job_post_url"]
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
    return Command(
        update={
        "serialized_job_post": serialized,
        "job_post_documents": retrieved_post,
        "messages": [
            ToolMessage(
                    content="Failed to retrieve docs",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )
