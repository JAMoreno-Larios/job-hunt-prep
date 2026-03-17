"""
Ingestion pipeline that will convert user documents into langchain docs,
to be embedded in ChromaDb.

J. A. Moreno
2026
"""

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Load environmental variables
load_dotenv()
