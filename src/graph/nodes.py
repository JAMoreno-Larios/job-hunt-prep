"""
Implementation for our workflow nodes
J. A. Moreno
2026
"""

from state import JobPrepState
from dotenv import load_dotenv
from langgraph.types import Command, Literal
from langchain_community.document_loaders.url import UnstructuredURLLoader

import llm_setup
import prompts

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
    query = state.get('distilled_query')
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

def distill_search_query(state: JobPrepState):
    """
    Generates a search query based on user input and job post,
    then retrieves relevant user information.
    """

    # Format the prompt used to obtain the search terms
    formatted_prompt = prompts.distill_query.format(
        job_post=state['serialized_job_post'],
        user_query=state['user_query']
    )
    distilled_query = llm.llm.invoke(formatted_prompt)

    # Now we use can use the refined query to extract user information.
    return {"distilled_query": distilled_query}

## ANSWER GENERATION NODES

def draft_answer(state: JobPrepState):
    formatted_prompt = prompts.draft_answer.format(
        job_post=state['serialized_job_post'],
        user_info=state['serialized_documents'],
        user_query=state['user_query']
    )
    response = llm.llm.invoke(formatted_prompt)
    return {"draft_response": response}
