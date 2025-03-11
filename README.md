# Research Paper Assistant (Development) 📚

A Streamlit application for extracting, analyzing, and generating review papers from academic research.

## Overview

This application provides a web interface for the comprehensive research paper processing workflow, including:

1. **Research Paper Extraction**: Automatically fetch papers from arXiv based on your query and follow citation networks
2. **Knowledge Base Integration**: Load extracted papers into a Neo4j graph database and vector database
3. **Review Paper Generation**: Generate comprehensive review papers using AI agents

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

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Running the App Locally

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. In the Streamlit interface:
   - Configure your environment settings
   - Start the Neo4j database
   - Run paper extraction, loading, or review generation processes

## Deploying to Streamlit Cloud

1. Push your code to a GitHub repository (make sure not to include your `.env` file)

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in

3. Create a new app, selecting your GitHub repository

4. Set environment variables in the Streamlit Cloud dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key

5. Deploy the app

**Note**: The Neo4j Docker container functionality will not work on Streamlit Cloud. For deployment, consider using a hosted Neo4j instance and updating the connection details.

## Using an External Neo4j Database

For Streamlit Cloud deployment, follow these steps to use an external Neo4j instance:

1. Create a Neo4j Aura instance or set up a Neo4j server elsewhere

2. Modify the `_init_neo4j` method in `knowledge_base.py` to use your database connection:
   ```python
   def _init_neo4j(self):
       return GraphDatabase.driver(
           os.getenv("NEO4J_URI", "bolt://localhost:7687"),
           auth=(
               os.getenv("NEO4J_USER", "neo4j"), 
               os.getenv("NEO4J_PASSWORD", "research123")
           ),
           encrypted=True
       )
   ```

3. Add the Neo4j credentials to your Streamlit Cloud environment variables:
   - `NEO4J_URI`: Your Neo4j instance URI
   - `NEO4J_USER`: Username
   - `NEO4J_PASSWORD`: Password

## File Structure

- `app.py`: Streamlit application
- `final.py`: Main workflow manager
- `papers_extractor_bfs.py`: ArXiv paper extraction logic
- `knowledge_base.py`: Database and knowledge storage
- `citation_mapper.py`: Handles paper citations
- `processing_pipeline.py`: Text and image processing
- `review_writer.py`: AI-powered paper generation
- `main.py`: Command-line interface for review generation

## Troubleshooting

- **Docker Issues**: Ensure Docker is running and you have permission to create containers
- **API Rate Limits**: If you encounter OpenAI API rate limits, add waiting periods or implement retries
- **Memory Issues**: Reduce batch sizes in the extraction and processing pipelines for lower memory usage

## License

[Your License Here]




```
docker run `
    -p 7474:7474 `
    -p 7687:7687 `
    -v ${PWD}/neo4j/data:/data `
    -v ${PWD}/neo4j/import:/import `
    --env NEO4J_AUTH=neo4j/research123 `
    neo4j:latest
```

```
python .\papers_extractor_bfs.py
python .\update_metadata.py
python .\knowledge_base.py
```
