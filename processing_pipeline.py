import json
import base64
import logging
from typing import Set, List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import hashlib
import os
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --------------------- CONFIGURATION ---------------------
FILE_PATH = r"downloads\1706.03762v7.pdf"
OUTPUT_DIR = r"output"
BATCH_SIZE = 3  # For parallel processing
MAX_CHUNK_LENGTH = 3000  # Characters per chunk for processing

# --------------------- DATA MODELS ---------------------
class PaperMetadata(BaseModel):
    key_themes: Set[str] = Field(default_factory=set, description="Core technical topics")
    methodology: Set[str] = Field(default_factory=set, description="Specific techniques/algorithms")
    domain: Set[str] = Field(default_factory=set, description="Application domains/fields")
    strengths: Set[str] = Field(default_factory=set, description="Technical contributions")
    limitations: Set[str] = Field(default_factory=set, description="Technical shortcomings")

    @field_validator('*')
    def normalize_terms(cls, values, info):
        """Normalize technical terms"""
        term_map = {
            "nlp": "natural language processing",
            "ai": "artificial intelligence",
            "dl": "deep learning"
        }
        
        if isinstance(values, str):
            return term_map.get(values.lower(), values)
        return values

    def model_dump(self) -> dict:
        """Convert sets to lists for serialization"""
        return {
            "key_themes": list(self.key_themes),
            "methodology": list(self.methodology),
            "domain": list(self.domain),
            "strengths": list(self.strengths),
            "limitations": list(self.limitations)
        }

    def dict(self, **kwargs):
        data = super().dict(**kwargs)
        # Convert sets to lists for JSON serialization
        for field in ["key_themes", "methodology", "domain", "strengths", "limitations"]:
            data[field] = list(data[field])
        return data

# --------------------- VALIDATION ---------------------
def validate_pdf(file_path: str):
    """Ensure PDF is valid research paper format"""
    if not file_path.endswith(".pdf"):
        raise ValueError("Invalid file format")
    # Check page count
    from pypdf import PdfReader
    with open(file_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        page_count = len(reader.pages)

    if page_count > 20:
        logger.warning(f"Deleting PDF file with {page_count} pages: {file_path}")
        os.remove(file_path)
        # Remove corresponding JSON file if it exists
        pdf_filename = os.path.basename(file_path)
        json_filename = os.path.splitext(pdf_filename)[0] + ".json"
        json_path = os.path.join("papers_summary", json_filename)
        if os.path.exists(json_path):
            logger.warning(f"Removing corresponding JSON file: {json_path}")
            os.remove(json_path)
        raise ValueError(f"PDF has {page_count} pages, which exceeds the 20-page limit")
    with open(file_path, "rb") as f:
        header = f.read(4)
        if header != b"%PDF":
            raise ValueError("Corrupted or invalid PDF file")

# --------------------- PDF PROCESSING ---------------------
def extract_text_tables_images(file_path: str) -> tuple:
    """Extract structured content from PDF with enhanced error handling"""
    try:
        validate_pdf(file_path)
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000
        )

        tables, texts, images = [], [], []
        seen_images = set()

        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            elif "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                # Extract unique images
                for el in chunk.metadata.orig_elements:
                    if "Image" in str(type(el)) and el.metadata.image_base64 not in seen_images:
                        images.append(el.metadata.image_base64)
                        seen_images.add(el.metadata.image_base64)

        return texts, tables, images

    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise

# --------------------- METADATA EXTRACTION ---------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_chunk(chunk: str, chain) -> Optional[PaperMetadata]:
    """Process individual text chunk with retries"""
    try:
        result = chain.invoke({"chunk": chunk[:MAX_CHUNK_LENGTH]})
        logger.info(f"Result type: {type(result)}")
        
        # If result is already a PaperMetadata object
        if isinstance(result, PaperMetadata):
            return result
            
        # If result is a dict, convert to PaperMetadata
        elif isinstance(result, dict):
            logger.info(f"Converting dict to PaperMetadata:")
            try:
                # Ensure all required fields are present
                for field in ["key_themes", "methodology", "domain", "strengths", "limitations"]:
                    if field not in result:
                        result[field] = []
                        
                # Convert lists to sets
                for field in ["key_themes", "methodology", "domain", "strengths", "limitations"]:
                    if isinstance(result[field], list):
                        result[field] = set(result[field])
                        
                return PaperMetadata(**result)
            except Exception as e:
                logger.error(f"Error converting dict to PaperMetadata: {str(e)}")
                # Create an empty PaperMetadata if conversion fails
                return PaperMetadata()
        else:
            logger.error(f"Unexpected result type: {type(result)}")
            return None
    except Exception as e:
        logger.warning(f"Chunk processing failed: {str(e)}")
        return None

def extract_metadata(texts: List) -> PaperMetadata:
    """Enhanced metadata extraction with parallel processing"""
    logger.info("Starting metadata extraction...")
    
    # Configure extraction pipeline
    parser = JsonOutputParser(pydantic_object=PaperMetadata)
    prompt_template = """Analyze this technical text chunk:
    {chunk}

    Extract as JSON:
    - key_themes: 3-5 core technical topics (e.g., "transformer architecture")
    - methodology: Specific techniques/algorithms (e.g., "self-attention mechanisms")
    - domain: Practical application areas (e.g., "machine translation")
    - strengths: Technical contributions (e.g., "parallel computation efficiency")
    - limitations: Technical shortcomings (e.g., "quadratic memory complexity")

    Rules:
    - Use exact technical terms from text
    - Avoid generic terms like "deep learning"
    - Prioritize novel concepts

    {format_instructions}"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chunk"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model = ChatOpenAI(temperature=0.1, model="gpt-4o")
    chain = prompt | model | parser

    # Process in parallel batches
    metadata = PaperMetadata()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        batch_results = []

        # Process batch
        for text_chunk in batch:
            result = process_chunk(str(text_chunk), chain)
            if result:
                batch_results.append(result)

        # Aggregate results
        for result in batch_results:
            if not isinstance(result, PaperMetadata):
                logger.error("Invalid result type, skipping")
                continue
            metadata.key_themes.update(result.key_themes)
            metadata.methodology.update(result.methodology)
            metadata.domain.update(result.domain)
            metadata.strengths.update(result.strengths)
            metadata.limitations.update(result.limitations)

    logger.info("Metadata extraction complete")
    return metadata

# --------------------- CONTENT SUMMARIZATION ---------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def summarize_text_chunk(chunk: str) -> str:
    """Technical text summarization with error handling"""
    prompt_text = """Summarize this research paper section focusing on technical content:
    {element}
    
    Include:
    - Key formulas/equations
    - Novel architectural details
    - Benchmark results (include metrics)
    - Comparison with prior work
    - Technical limitations mentioned"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.3, model="gpt-4o")
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"element": chunk})

def summarize_text(texts: List) -> List[str]:
    """Batch process text summarization"""
    logger.info("Starting text summarization...")
    return [summarize_text_chunk(str(t)) for t in texts]

def summarize_table(table: str) -> str:
    """Technical table analysis"""
    prompt_text = """Analyze this technical table:
    {element}
    
    1. Identify table type:
       - Results comparison
       - Model architecture
       - Benchmark metrics
       - Hyperparameters
    2. Summarize key numerical findings
    3. Note any statistical significance markers"""
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.2, model="gpt-4o")
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"element": table})

def summarize_tables(tables: List) -> List[str]:
    """Process tables with parallel execution"""
    logger.info("Starting table summarization...")
    tables_html = [t.metadata.text_as_html for t in tables]
    return [summarize_table(t) for t in tables_html]

def summarize_image(image_data: str, output_dir: str) -> tuple:
    """Technical image analysis with local storage"""
    try:
        # Save image
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_data))

        # Generate description
        prompt_text = """Analyze this technical figure:
        - Architecture diagrams: components and connections
        - Graphs: axes, trends, key data points
        - Mathematical visualizations: symbols and relationships
        - Flowcharts: decision points and processes"""
        
        messages = [
            ("user", [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ])
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()
        description = chain.invoke({})

        return description, filepath

    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return "", ""

def summarize_images(images: List[str]) -> tuple:
    """Process images with error handling"""
    logger.info("Starting image processing...")
    output_dir = os.path.join(OUTPUT_DIR, "images")
    results = [summarize_image(img, output_dir) for img in images]
    descriptions, paths = zip(*results) if results else ([], [])
    return descriptions, paths

def extract_basic_info(pdf_path):
    import fitz  # PyMuPDF
    
    """Extracts metadata like title, authors, DOI, and journal from a PDF."""
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    title = metadata.get("title", "Unknown Title")
    authors = metadata.get("author", "Unknown Author").split(", ") if metadata.get("author") else ["Unknown Author"]
    publication_year = metadata.get("modDate", "Unknown Year")[2:6] if metadata.get("modDate") else "Unknown Year"
    source = pdf_path  # Placeholder for now, can be replaced with a proper URL later

    return {
        "title": title,
        "authors": authors,
        "publication_year": publication_year,
        "source": source
    }
# --------------------- DATA STORAGE ---------------------
def store_extracted_data(
    text_summaries: List[str],
    table_summaries: List[str],
    image_summaries: tuple,
    metadata: PaperMetadata,
    basic_info, output_path
) -> str:
    """Store processed data with enhanced structure"""
    try:
        # Prepare content structure
        content = {
            "technical_summary": {
                "sections": {
                    "introduction": text_summaries[0] if len(text_summaries) > 0 else "",
                    "methodology": text_summaries[1] if len(text_summaries) > 1 else "",
                    "results": text_summaries[2] if len(text_summaries) > 2 else ""
                },
                "tables": [{"summary": t} for t in table_summaries],
                "figures": [
                    {"description": d, "path": p}
                    for d, p in zip(image_summaries[0], image_summaries[1])
                ] if image_summaries else []
            },
            "metadata": metadata.model_dump(),
            "basic_info": basic_info
        }

        # Save to file
        # output_path = os.path.join(output_path, "technical_analysis.json")
        with open(output_path, "r", encoding="utf-8") as f:
            existing_content = json.load(f)
            for key, val in content.items():
                if key not in existing_content:
                    existing_content[key] = val
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_content, f, indent=4, ensure_ascii=False)

        logger.info(f"Data successfully saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Data storage failed: {str(e)}")
        raise

# --------------------- MAIN PIPELINE ---------------------
def process_research_paper(file_path: str, output_json_path: str) -> dict:
    """End-to-end processing pipeline"""
    try:
        logger.info(f"Starting processing: {file_path}")
        
        # Extraction
        texts, tables, images = extract_text_tables_images(file_path)
        
        # Summarization
        text_summaries = summarize_text(texts)
        table_summaries = summarize_tables(tables)
        image_summaries = summarize_images(images)
        
        # Metadata
        metadata = extract_metadata(texts)
        basic_info = ""
        # Storage
        output_path = store_extracted_data(
            text_summaries,
            table_summaries,
            image_summaries,
            metadata, basic_info, output_json_path
        )

        return {
            "status": "success",
            "output_path": output_path,
            # "stats": {
            #     "text_sections": len(text_summaries),
            #     "tables": len(table_summaries),
            #     "figures": len(image_summaries[0]) if image_summaries else 0
            # }
        }

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    result = process_research_paper(FILE_PATH, './')
    print(json.dumps(result, indent=2))