"""
state.py

We will define this graph's state here.
J. A. Moreno
2026
"""


from typing import TypedDict
from langchain_core.documents import Document
from langgraph.graph import MessagesState


class InputState(MessagesState):
    """
    We take a natural-language user input that contains the
    URL for job_post_url and a job related user_query.
    """
    raw_query: str


class OutputState(MessagesState):
    draft_response: str | None

class JobPrepState(MessagesState):
    
    # Raw user query
    raw_query: str
    
    # Parsed query
    user_query: str
    job_post_url: str

    # Raw search results
    retrieved_documents: list[Document] | None  # Raw docs
    serialized_documents: str | None  # Serialized content
    job_post_documents: list[Document] | None
    serialized_job_post: str | None # From web page

    # Generated content
    distilled_query: str | None
    draft_response: str | None
