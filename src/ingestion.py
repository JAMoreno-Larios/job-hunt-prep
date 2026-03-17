"""
Ingestion pipeline that will convert user documents into langchain docs,
to be embedded in ChromaDb.

J. A. Moreno
2026
"""

from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader,
    PyPDFDirectoryLoader, TextLoader)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Load environmental variables
load_dotenv()

# Define constants
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 100
EMBEDDINGS_MODEL = "nomic-embed-text-v2-moe"
NUM_CTX = 8000

# Define an enumeration for file types
class Filetypes(Enum):
    PDF = 1
    TXT = 2
    OTHER = 3


# Define the user's data path
document_path = Path("./data/user-data/").resolve()

# Define the vector store location
vector_store_path = Path("./data/embeddings/").resolve()

# Instantiate character splitter.
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)

# Create embeddings
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, num_gpu=1, num_ctx=NUM_CTX)

# Create vector store
vector_store = Chroma(persist_directory=vector_store_path.__str__(),
                      embedding_function=embeddings)

# Define data ingestion pipeline methods.

def document_load_split_embed(path: str, type=Filetypes.OTHER):
    """
    Method used to load, split, and embed documents.
    Provide type as in "pdf", "txt", or "other".
    """    

    # Instantiate the appropriate loader
    match type:
        case Filetypes.PDF:
            print("Loading PDF files")
            loader = PyPDFDirectoryLoader(path)
        case Filetypes.TXT:
            print("Loading text files")
            loader = DirectoryLoader(path=path, glob='**/*.txt',
                                     loader_cls=TextLoader,
                                     use_multithreading=True)
        case Filetypes.OTHER:
            print("Loading other types of files")
            loader = DirectoryLoader(path=path,
                                     exclude=['**/*.pdf', '**/*.txt',
                                              '**/README.md'],
                                     use_multithreading=True)
    # Load documents
    documents = loader.load()
    # Split into chunks
    print ("Splitting...")
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Ingest
    print("Ingesting...")
    vector_store.from_documents(chunks, embeddings)
    print("Done ingesting.")



# Define the ingestion pipeline
def run_ingestion_pipeline(document_path: Path):
    """
    Method that runs a ingestion pipeline where we will load all
    documents found in the input document_path and create 
    a local vector store with Chroma.
    """
    document_load_split_embed(document_path.__str__(),
                              Filetypes.PDF)
    document_load_split_embed(document_path.__str__(),
                              Filetypes.TXT)
    document_load_split_embed(document_path.__str__(),
                              Filetypes.OTHER)


if __name__ == "__main__":
    run_ingestion_pipeline(document_path)
