""" graph.py

Here we wire all our nodes into a graph

J. A. Moreno
2026
"""

from . import nodes
from typing import Any, Iterator
from .state import InputState, JobPrepState, OutputState
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import CachePolicy


checkpointer = InMemorySaver()


class Graph:
    def __init__(self) -> None:
        """
        Creates the graph when called.
        """
        workflow = StateGraph(
            JobPrepState, input_schema=InputState, output_schema=OutputState
        )

        # Add nodes
        workflow.add_node("process_user_input", nodes.process_user_input)
        workflow.add_node("scrap_job_posting", nodes.scrap_job_posting,
                          cache_policy=CachePolicy())
        workflow.add_node("distill_query", nodes.distill_search_query)
        workflow.add_node("search_user_data", nodes.search_user_db)
        workflow.add_node("draft_answer", nodes.draft_answer)

        # Declare edges
        workflow.add_edge(START, "process_user_input")
        workflow.add_edge("process_user_input", "scrap_job_posting")
        workflow.add_edge("scrap_job_posting", "distill_query")
        workflow.add_edge("distill_query", "search_user_data")
        workflow.add_edge("search_user_data", "draft_answer")
        workflow.add_edge("draft_answer", END)

        # Compile the graph
        self._app = workflow.compile(cache=InMemoryCache(),
                                     checkpointer=checkpointer)

    @property
    def get_graph(self):
        return self._app.get_graph()

    def draw_mermaid_png(self, output_file_path="workflow.png"):
        self._app.get_graph().draw_mermaid_png(output_file_path=output_file_path)

    def run_agent(self, query) -> Iterator[dict[str, Any] | Any]:
        """
        Runs the agent workflow, forms the user input
        """
    
    # Define configuration
        config = {
            "configurable": {
                "thread_id" : "1"  # Change later
            }
        }
        # Form input
        messages = [{"role": "user", "content": query}]
        # Invoke the graph as a stream
        return self._app.stream(
            {"messages": messages},
            config=config,
            stream_mode=["messages"],
            version="v2"
        )

    def run_agent_streamlit(self, query):
        """
        Runs the agent workflow, forms the user input
        """
        # Invoke the graph as a stream
        for part in self.run_agent(query):
            if part["type"] == "messages":
                # MessagesStreamPart — (message_chunk, metadata) from LLM calls
                msg, metadata = part["data"]
                yield msg.content
