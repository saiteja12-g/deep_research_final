# Research Paper Assistant 📚

A comprehensive system for extracting, analyzing, and generating review papers from academic research using AI.

## Overview

This application provides a complete workflow for academic research paper processing:

1. **Research Paper Extraction**: Automatically fetch papers from arXiv based on your query and follow citation networks
2. **Knowledge Base Integration**: Load extracted papers into a Neo4j graph database and vector database
3. **Contextual Analysis**: Process papers to extract key themes, methodologies, strengths, and limitations
4. **Review Paper Generation**: Generate comprehensive review papers using AI agents

## Features

- **Intelligent Paper Discovery**: BFS traversal of citation networks starting from initial query results
- **Graph-based Knowledge Representation**: Store papers and their relationships in Neo4j 
- **Semantic Search**: Find related papers using vector embeddings in ChromaDB
- **Image Processing**: Extract and analyze figures from research papers
- **Citation Mapping**: Identify and map citations between papers
- **AI-Powered Review Generation**: Generate structured review papers with proper citations using LLM agents
- **Interactive UI**: Streamlit-based frontend for easy interaction

## Setup and Installation

### Prerequisites

- Python 3.9+ 
- Docker (for running Neo4j)
- OpenAI API key

### Installation

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Running the App

### Using the Workflow Manager (Recommended)

The `workflow.py` script provides a simplified way to run the complete pipeline or individual steps:

1. Run the complete workflow:
   ```bash
   python workflow.py --query "Single image to 3D" --full-workflow
   ```

2. Extract papers only:
   ```bash
   python workflow.py --query "Single image to 3D" --extract-only --max-depth 2 --max-papers 5
   ```

3. Load extracted papers to database:
   ```bash
   python workflow.py --load-only
   ```

4. Generate a review paper:
   ```bash
   python workflow.py --query "Single image to 3D" --generate-review
   ```

5. Continue a previously started paper:
   ```bash
   python workflow.py --generate-review --continue
   ```

### Using the Streamlit UI

1. Start the Streamlit app:
   ```bash
   streamlit run frontend.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. In the Streamlit interface:
   - Configure your environment settings
   - Start the Neo4j database
   - Run paper extraction, loading, or review generation processes

### Running Components Individually

Alternatively, you can run each component separately:

1. Start Neo4j (required for knowledge storage):
   ```bash
   docker run -p 7474:7474 -p 7687:7687 --env NEO4J_AUTH=neo4j/research123 neo4j:latest
   ```

2. Extract papers from arXiv:
   ```bash
   python papers_extractor_bfs.py
   ```

3. Process and load papers into knowledge base:
   ```bash
   python knowledge_base.py
   ```

4. Generate a review paper:
   ```bash
   python main.py --query "Your research topic"
   ```

## Running with Docker (Not Tested)

The application includes Docker support for easy deployment:

1. Build the Docker image:
   ```bash
   docker build -t research-paper-assistant .
   ```

2. Run using docker-compose (handles both the app and Neo4j):
   ```bash
   docker-compose up
   ```

3. Access the Streamlit interface at `http://localhost:8501`

## Project Structure

- `frontend.py`: Streamlit application
- `workflow.py`: Complete workflow manager
- `papers_extractor_bfs.py`: ArXiv paper extraction with BFS traversal
- `knowledge_base.py`: Database and knowledge storage integrations
- `citation_mapper.py`: Handles paper citations
- `processing_pipeline.py`: Text and image processing
- `review_writer.py`: AI-powered paper generation
- `main.py`: Command-line interface for review generation

## Folder Structure

- `/papers` - Downloaded PDF files
- `/papers_summary` - Extracted metadata in JSON format
- `/output` - Generated review papers and figures
- `/chroma_db` - Vector embeddings database
- `/neo4j` - Graph database files

## Troubleshooting

- **Docker Issues**: Ensure Docker is running and you have permission to create containers
- **API Rate Limits**: If you encounter OpenAI API rate limits, add waiting periods or implement retries
- **Memory Issues**: Reduce batch sizes in the extraction and processing pipelines for lower memory usage
- **Neo4j Connection**: Ensure the Neo4j container is running before running knowledge base operations

## Flowcharts
### Model Architecture
![alt text](files\image-1.png)

### Agent Workflow
![alt text](files\image.png)

### Video Demo
[![Research Paper Assistant Demo](https://img.youtube.com/vi/CLq7E8Y3Pk4/0.jpg)](https://youtu.be/CLq7E8Y3Pk4)

Click the image above to watch the demo video of the Research Paper Assistant in action.
## Acknowledgements

- [ArXiv API](https://arxiv.org/help/api/index) for paper access
- [OpenAI](https://openai.com/) for natural language processing
- [Neo4j](https://neo4j.com/) for graph database functionality
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Streamlit](https://streamlit.io/) for the user interface