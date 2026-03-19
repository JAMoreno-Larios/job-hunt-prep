"""
graph.py

Here we wire all our nodes into a graph

J. A. Moreno
2026
"""


import nodes
from state import JobPrepState
from langgraph.graph import StateGraph, START, END

def create_graph():
    """
    Creates the graph when called.
    """
    workflow = StateGraph(JobPrepState)

    # Add nodes
    workflow.add_node("scrap_job_posting", nodes.scrap_job_posting)
    workflow.add_node("distill_query", nodes.distill_search_query)
    workflow.add_node("search_user_data", nodes.search_user_db)
    workflow.add_node("draft_answer", nodes.draft_answer)

    # Declare edges
    workflow.add_edge(START, "scrap_job_posting")
    workflow.add_edge("scrap_job_posting", "distill_query")
    workflow.add_edge("distill_query", "search_user_data")
    workflow.add_edge("search_user_data", "draft_answer")
    workflow.add_edge("draft_answer", END)

    # Compile the graph
    return workflow.compile()

graph = create_graph()
