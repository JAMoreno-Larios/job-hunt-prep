"""
tools.py

Here we define all tools to be used in our agent

J. A. Moreno
2026
"""

from langchain.tools import ToolRuntime
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.types import Command

from . import llm_setup

# Initialize retriever
retriever = llm_setup.Retriever().retriever


class Tools:
    def __init__(self, retriever: BaseRetriever) -> None:
        self._retriever = retriever

    # Define tools - Register them with the tool function
    def search_user_db(self, query: str, runtime: ToolRuntime) -> Command:
        """
        Semantically searches a vector store containing user's personal information
        matching the query.

        Sets:
            serialized_documents: Serialized documents from the vector store
            retrieved_documents: Retrieved Documents from the vector store
        """

        try:
            retrieved_docs = self._retriever.invoke(str(query), k=5)
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
                            content=serialized, tool_call_id=runtime.tool_call_id
                        )
                    ],
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
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

    def scrap_job_posting(self, job_post_url: str, runtime: ToolRuntime) -> Command:
        """
        Scraps the job post located in job_post_url and loads
        as a list of Documents using SeleniumURLLoader.

        Arguments:
            - job_post_url: The URL to scrap

        Sets the state as follows:
            job_post_url: The scrapped URL
            serialized_job_post: Serialized documents from the job post
            job_post_documents: Retrieved Documents from the job post

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
        return Command(
            update={
                "job_post_url": job_post_url,
                "serialized_job_post": serialized,
                "job_post_documents": retrieved_post,
                "messages": [
                    ToolMessage(content=serialized, tool_call_id=runtime.tool_call_id)
                ],
            }
        )
