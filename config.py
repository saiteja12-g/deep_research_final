import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "research123")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directories
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
PAPERS_DIR = os.getenv("PAPERS_DIR", "./papers")
PAPERS_SUMMARY_DIR = os.getenv("PAPERS_SUMMARY_DIR", "./papers_summary")

# Create directories if they don't exist
for directory in [OUTPUT_DIR, PAPERS_DIR, PAPERS_SUMMARY_DIR]:
    os.makedirs(directory, exist_ok=True)