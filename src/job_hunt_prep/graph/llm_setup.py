"""
llm_setup.py

Here we instantiate our LLMs.

J. A. Moreno
2026
"""

from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Define the vector store location
vector_store_path = Path(__file__).absolute().parents[3] / "./data/embeddings/"

# Define constants
EMBEDDINGS_MODEL = "nomic-embed-text-v2-moe"
# LLM_MODEL = "qwen3:8b"
LLM_MODEL = "qwen3.5:9b"
NUM_CTX = 4096
NUM_THREADS = None
NUM_GPU = 999  # Uses all GPUs installed
REASONING = False

class BaseLLMConfig(ABC):
    """
    Base contract for any LLM setup
    """
    
    @property
    @abstractmethod
    def llm(self) -> ChatOllama:
        """
        Use to get the LLM instance
        """
        pass
    


class BaseRetrieverConfig(ABC):
    """
    Base contract for any retriever LLM setup
    """
    @property
    @abstractmethod
    def retriever(self) -> VectorStoreRetriever:
        """
        Use to get a retriever from the vector store
        """
        pass


class LLM(BaseLLMConfig):

    """
    Service class to set up all LLM-related parameters.
    """
    def __init__(self) -> None:
        self._llm = ChatOllama(model=LLM_MODEL,
                               num_ctx=NUM_CTX,
                               reasoning=REASONING,
                               num_gpu=NUM_GPU,
                               num_thread=NUM_THREADS)

    @property
    def llm(self) -> ChatOllama:
        return self._llm
    


class Retriever(BaseRetrieverConfig):

    """
    Service class to set up all LLM-related parameters.
    """
    def __init__(self) -> None:
        self._embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
        self._vector_store = Chroma(persist_directory=str(vector_store_path),
                      embedding_function=self._embeddings)

        self._retriever = self._vector_store.as_retriever()

    @property
    def retriever(self) -> VectorStoreRetriever:
        return self._retriever
