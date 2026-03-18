"""
state.py

We will define this graph's state here.
J. A. Moreno
2026
"""

from typing import TypedDict

class JobPrepState(TypedDict):
    
    # Raw user query
    user_query: str
    job_post_url: str

    # Raw search results
    vector_store_search_results: list[str] | None  # Raw document chunks
    job_post_contents: list[str] | None # Same as above

    # Generated content
    draft_response: str | None
    messages: list[str] | None
