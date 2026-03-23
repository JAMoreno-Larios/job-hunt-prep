"""
graph.py

Here we wire all our nodes into a graph

J. A. Moreno
2026
"""


import nodes
from state import InputState, JobPrepState, OutputState
from langgraph.graph import StateGraph, START, END

class Graph:

    def __init__(self) -> None:
        """
        Creates the graph when called.
        """
        workflow = StateGraph(JobPrepState,
                              input_schema=InputState,
                              output_schema=OutputState)

        # Add nodes
        workflow.add_node("process_user_input", nodes.process_user_input)
        workflow.add_node("scrap_job_posting", nodes.scrap_job_posting)
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
        self._app = workflow.compile()


    @property
    def get_graph(self):
        return self._app.get_graph()


    def draw_mermaid_png(self, output_file_path="workflow.png"):
        self._app.get_graph().draw_mermaid_png(output_file_path=output_file_path)


    def run_agent(self, query) -> None:
        """
        Runs the agent workflow, forms the user input
        """
        # Form input
        messages = [{"role": "user", "content": query}]
        # Invoke the graph
        for part in self._app.stream(
            {
                "messages": messages
            },
            stream_mode=["messages"],
            version="v2"):
            if part["type"] == "messages":
                # MessagesStreamPart — (message_chunk, metadata) from LLM calls
                msg, metadata = part["data"]
                print(msg.content, end="", flush=True)

