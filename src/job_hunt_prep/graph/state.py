"""
state.py

We will define this graph's state here.
J. A. Moreno
2026
"""


from typing import TypedDict
from langchain_core.documents import Document
from langgraph.graph import MessagesState


class ParsedUserInputSchema(TypedDict):
    """
    We take a natural-language user input that contains the
    URL for job_post_url and a job related user_query.
    """
    job_post_url: str | None
    user_query: str | None


class DistilledQuerySchema(TypedDict):
    """
    Used to define the output for our distilled_query node
    """
    distilled_query: str | None

class JobPrepState(ParsedUserInputSchema, DistilledQuerySchema, MessagesState):
    
    # Raw user query
    raw_query: str

    # Raw search results
    retrieved_documents: list[Document] | None  # Raw docs
    serialized_documents: str | None  # Serialized content
    job_post_documents: list[Document] | None
    serialized_job_post: str | None # From web page

    # Generated content
    draft_response: str | None
