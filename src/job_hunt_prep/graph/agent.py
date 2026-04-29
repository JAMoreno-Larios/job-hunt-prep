"""agent.py

Here we define our agent, we provide tools and middleware

J. A. Moreno
2026
"""

from langchain.chat_models import BaseChatModel
from langchain.tools import tool
from langchain_core.retrievers import BaseRetriever
from langgraph.graph.state import RunnableConfig
from typing import Any, Iterator, List
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
from langchain_community.agent_toolkits import FileManagementToolkit

from . import prompts
from .tools import Tools
from .state import JobPrepState

checkpointer = InMemorySaver()

root_dir = Path(__file__).absolute().parents[3] / "./output/"

class Agent:
    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever) -> None:
        """
        Initialize class. Inject LLM and Retriever to be used
        """
        self._llm = llm
        self._retriever = retriever
        self._Tools = Tools(retriever)

        # Initialize file management toolkit, get write_file tool into list
        tools = FileManagementToolkit(
            root_dir=str(root_dir), selected_tools=["write_file"]
        ).get_tools()

        # Register custom tools
        tools.append(tool(self._Tools.scrap_job_posting))
        tools.append(tool(self._Tools.search_user_db))

        # Create agent
        self._app = create_agent(
            model=self._llm,
            tools=tools,
            middleware=[TodoListMiddleware(), SummarizationMiddleware(model=self._llm)],
            checkpointer=InMemorySaver(),
            system_prompt=prompts.agent_prompt,
            state_schema=JobPrepState,
        )

    @property
    def get_graph(self):
        return self._app.get_graph()

    def draw_mermaid_png(self, output_file_path="workflow.png"):
        self._app.get_graph().draw_mermaid_png(output_file_path=output_file_path)

    def run_agent(
        self,
        query,
        config=RunnableConfig(
            {
                "configurable": {
                    "thread_id": "1"  # Change later
                }
            }
        ),
    ) -> Iterator[dict[str, Any] | Any]:
        """
        Runs the agent workflow, forms the user input
        """

        # Form input
        messages = [{"role": "user", "content": query}]
        # Invoke the graph
        return self._app.invoke(
            {"messages": messages},
            config=config,
        )

    def stream_agent(
        self,
        query,
        config=RunnableConfig(
            {
                "configurable": {
                    "thread_id": "1"  # Change later
                }
            }
        ),
    ) -> Iterator[dict[str, Any] | Any]:
        """
        Runs the agent workflow, forms the user input
        """

        # Define configuration
        config = RunnableConfig(
            {
                "configurable": {
                    "thread_id": "1"  # Change later
                }
            }
        )
        # Form input
        messages = [{"role": "user", "content": query}]
        # Invoke the graph as a stream
        return self._app.stream(
            {"messages": messages},
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        )

    def run_agent_streamlit(self, session_messages: List[str], config: RunnableConfig):
        """
        Runs the agent workflow, expect state messages
        """
        # Invoke the graph
        return self._app.invoke(
            {"messages": session_messages},
            config=config,
        )
