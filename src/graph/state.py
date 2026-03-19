"""
state.py

We will define this graph's state here.
J. A. Moreno
2026
"""

from typing import TypedDict

from langchain_core.documents import Document

class JobPrepState(TypedDict):
    
    # Raw user query
    user_query: str
    job_post_url: str

    # Raw search results
    retrieved_documents: list[Document] | None  # Raw docs
    serialized_documents: str | None  # Serialized content
    job_post_documents: list[Document] | None
    serialized_job_post: str | None # From web page

    # Generated content
    distilled_query: str
    draft_response: str | None
    messages: list[str] | None
