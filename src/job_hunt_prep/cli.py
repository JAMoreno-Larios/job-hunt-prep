"""
cli.py

We implement a Typer-based CLI for this project

J. A. Moreno
2026
"""

import typer
import subprocess
import asyncio
from pathlib import Path
from job_hunt_prep.ingestion import run_ingestion_pipeline

# Instantiate the Typer app
app = typer.Typer(help="Job Prep AI Assitant")

@app.command()
def run():
    """Start the Streamlit application."""
    subprocess.run(["streamlit", "run", "./src/job_hunt_prep/main.py"])

@app.command()
def ingest(path: Path = Path("./data/user-data")):
    """Ingest documents into the vector store."""
    asyncio.run(run_ingestion_pipeline(path))

@app.command()
def check():
    """Check environment variables"""
    subprocess.run(["./scripts/check_env.sh"])

@app.command()
def setup_models():
    """Pull required Ollama models."""
    subprocess.run(["ollama", "pull", "qwen3.5:9b"])
    subprocess.run(["ollama", "pull", "nomic-embed-text-v2-moe"])

if __name__ == "__main__":
    app()
