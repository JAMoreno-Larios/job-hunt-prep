"""
Ingestion pipeline that will convert user documents into langchain docs,
to be embedded in ChromaDb.

J. A. Moreno
2026
"""

import asyncio
from typing import List
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader,
    PyPDFDirectoryLoader, TextLoader)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Load environmental variables
load_dotenv()

# Define constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDINGS_MODEL = "nomic-embed-text-v2-moe"
BATCH_SIZE = 50

# Define an enumeration for file types
class Filetypes(Enum):
    PDF = 1
    TXT = 2
    OTHER = 3


# Define the user's data path
document_path = Path(__file__).absolute().parents[1] / "data/user-data"

# Define the vector store location
vector_store_path = Path(__file__).absolute().parents[1] / "./data/embeddings/"

# Instantiate character splitter.
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)

# Create embeddings
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, num_gpu=1)

# Create vector store
vector_store = Chroma(persist_directory=vector_store_path.__str__(),
                      embedding_function=embeddings)

# Define data ingestion pipeline methods.

async def index_documents_async(documents: List[Document],
                                batch_size: int = BATCH_SIZE):
    """Process documents in batches asynchronously"""

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(
            0, len(documents), batch_size
                   )
    ]

    print(f"Vector store indexing: Split into {len(batches)} of {batch_size} docs each")

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            print(
                f"VectorStore Indexing: Successfully added {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            print(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successfull = sum(1 for result in results if result is True)

    if successfull == len(batches):
        print(
            f"VectorStore Indexing: All batches processed successfully! ({successfull}/{len(batches)})"
        )
    else:
        print(
            f"VectorStore Indexing: Processed {successfull}/{len(batches)} batches successfully"
        )

async def document_load_split_embed(path: str, type=Filetypes.OTHER):
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
    # Load documents concurrently
    documents = await loader.aload()
    # Split into chunks
    print ("Splitting...")
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Ingest
    print("Ingesting...")
    await index_documents_async(chunks)
    print("Done ingesting.")



# Define the ingestion pipeline
async def run_ingestion_pipeline(document_path: Path):
    """
    Method that runs a ingestion pipeline where we will load all
    documents found in the input document_path and create 
    a local vector store with Chroma.
    """
    await document_load_split_embed(str(document_path),
                              Filetypes.PDF)
    await document_load_split_embed(str(document_path),
                              Filetypes.TXT)
    await document_load_split_embed(str(document_path),
                              Filetypes.OTHER)


if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline(document_path))
