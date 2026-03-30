"""
Implementation for our workflow nodes
J. A. Moreno
2026
"""

from logging import exception

from langgraph.prebuilt import ToolNode
from .state import OutputState, JobPrepState
from dotenv import load_dotenv
from langchain_community.document_loaders import SeleniumURLLoader

from . import llm_setup, prompts, tools

# Load environment variables
load_dotenv()

# Initialize LLM
llm = llm_setup.LLM()

# Initialize retriever
retriever = llm_setup.Retriever()

## USER INPUT PROCESSING NODES


def process_user_input(state: JobPrepState) -> JobPrepState:
    """
    Extracts the user query and job post from the user input.
    """
    # Format the propt to obtain the user query and
    # job post URL
    raw_query = state["messages"][-1].content

    # formatted_prompt = prompts.raw_query.format(raw_query=raw_query)
    formatted_prompt = prompts.raw_query.invoke({"raw_query": raw_query})

    response = llm.llm.with_structured_output(
        prompts.ProcessUserInputSchema,
        strict=True,
        method="json_schema",
        include_raw=True,
    ).invoke(state["messages"] + formatted_prompt.to_messages())

    user_query = response["parsed"]["user_query"]
    job_post_url = response["parsed"]["job_post_url"]
    user_instructions = response["parsed"]["user_instructions"]
    messages = response["raw"]

    # Now we use can use the refined query to extract user information.
    return {
        "raw_query": raw_query,
        "user_query": user_query,
        "job_post_url": job_post_url,
        "user_instructions": user_instructions,
        "messages": messages,
    }


### DATA NODES


# Scrap the job posting URL
def scrap_job_posting(state: JobPrepState) -> JobPrepState:
    """
    Scraps the job post and loads it as a document
    using the UnstructuredURLLoader.
    """
    print("\nScrapping job post\n")
    job_url = state.get("job_post_url")

    # Create the url loader
    loader = SeleniumURLLoader(
        [job_url],
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
    return {"serialized_job_post": serialized, "job_post_documents": retrieved_post}


def distill_search_query(state: JobPrepState) -> JobPrepState:
    """
    Generates a search query based on user input and job post,
    then retrieves relevant user information.
    """

    print("\nDistilling search query\n")
    # Format the prompt used to obtain the search terms
    formatted_prompt = prompts.distill_query.format(
        job_post=state["serialized_job_post"], user_query=state["user_query"]
    )
    response = llm.llm.with_structured_output(
        prompts.DistilledQuerySchema,
        strict=True,
        method="json_schema",
    ).invoke(formatted_prompt)

    # Now we use can use the refined query to extract user information.
    return {"distilled_query": response}


# Search in vector store
def search_user_db(state: JobPrepState) -> JobPrepState:
    """
    Retrieves the relevant information for the query
    """
    query = state.get("distilled_query")
    try:
        retrieved_docs = retriever.retriever.invoke(str(query), k=5)
        # Serialize documents for the model
        serialized = "\n\n".join(
            (
                f"Source: {doc.metadata.get('source', 'Unknown')}"
                + f"\n\nContent: {doc.page_content}"
            )
            for doc in retrieved_docs
        )
        # Return serialized and raw documents
        return {
            "serialized_documents": serialized,
            "retrieved_documents": retrieved_docs,
        }
    except Exception:
        exception("Search turned no elemments")
        return state


## ANSWER GENERATION NODES


def draft_answer(state: JobPrepState) -> OutputState:
    formatted_prompt = prompts.draft_answer.format(
        job_post=state["serialized_job_post"],
        user_info=state["serialized_documents"],
        user_instructions=state["user_instructions"],
        user_query=state["user_query"],
    )
    response = llm.llm.invoke(formatted_prompt)
    return {"draft_response": response}


## TOOL NODES

url_scrap_tool = ToolNode(
    [
        tools.scrap_job_posting,
    ]
)

search_user_db_tool = ToolNode(
    [
        tools.search_user_db,
    ]
)
