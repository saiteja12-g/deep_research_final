import json
import base64
import logging
import re
from typing import Set, List, Dict, Optional, Tuple
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

    if page_count > 80:
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

def identify_document_structure(texts: List) -> Dict[str, List]:
    """Identify document sections from text chunks"""
    section_patterns = {
        "abstract": r"(?:^|\s)abstract(?:$|\s)",
        "introduction": r"(?:^|\s)(?:introduction|background|overview)(?:$|\s)",
        "methodology": r"(?:^|\s)(?:methodology|method|approach|experiment|implementation)(?:$|\s)",
        "results": r"(?:^|\s)(?:results|evaluation|performance|findings)(?:$|\s)",
        "discussion": r"(?:^|\s)(?:discussion|analysis|implications)(?:$|\s)",
        "conclusion": r"(?:^|\s)(?:conclusion|future work|summary)(?:$|\s)"
    }
    
    structured_document = {}
    
    for section, pattern in section_patterns.items():
        structured_document[section] = []
    
    # Categorize chunks into sections
    for i, chunk in enumerate(texts):
        chunk_text = str(chunk)
        chunk_data = {
            "text": chunk_text,
            "index": i,
        }
        
        # Try to identify chunk's section
        section_found = False
        for section, pattern in section_patterns.items():
            # Check if the chunk title or first paragraph matches a section pattern
            lines = chunk_text.split('\n')
            title = lines[0] if lines else ""
            first_para = '\n'.join(lines[:3]) if len(lines) > 1 else ""
            
            if re.search(pattern, title.lower()) or re.search(pattern, first_para.lower()):
                structured_document[section].append(chunk_data)
                section_found = True
                break
        
        # If no section was matched, place in "other"
        if not section_found:
            if "other" not in structured_document:
                structured_document["other"] = []
            structured_document["other"].append(chunk_data)
    
    return structured_document

def extract_citations_from_text(text: str) -> List[str]:
    """Extract citation references from text"""
    # Pattern for [Author, Year] or [Number] style citations
    patterns = [
        r'\[([^,\]]+(?:, \d{4}|, et al\., \d{4}))\]',  # [Author, 2021] or [Author, et al., 2021]
        r'\[(\d+(?:,\s*\d+)*)\]',  # [1] or [1, 2, 3]
    ]
    
    all_citations = []
    for pattern in patterns:
        citations = re.findall(pattern, text)
        all_citations.extend(citations)
        
    return all_citations

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

# --------------------- IMAGE PROCESSING ---------------------
def process_images(images: List[str], output_dir: str, max_images: int = 10) -> List[Dict]:
    """
    Process images with error handling and generate descriptions
    Uses LLM to determine image importance directly
    """
    logger.info(f"Starting image processing (limiting to {max_images} images)...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort images by size initially (to start with likely more important diagrams)
    image_sizes = []
    for idx, img_data in enumerate(images):
        try:
            img_size = len(base64.b64decode(img_data))
            image_sizes.append((idx, img_size))
        except:
            # Skip invalid images
            continue
    
    # Take top 2*max_images candidates by size for further analysis
    sorted_images = sorted(image_sizes, key=lambda x: x[1], reverse=True)[:max_images*2]
    candidate_indices = [idx for idx, _ in sorted_images]
    
    logger.info(f"Analyzing {len(candidate_indices)} candidate images from {len(images)} total")
    
    results = []
    for i, idx in enumerate(candidate_indices):
        try:
            img_data = images[idx]
            
            # Save image
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(img_data))
            
            logger.info(f"Processing image {i+1}/{len(candidate_indices)}...")
            
            # Generate description and importance score together
            prompt_text = """Analyze this technical figure from a research paper:

1. Provide a concise yet comprehensive description of what this figure shows, focusing on:
   - Architecture diagrams and components
   - Graphs, charts, and data visualizations
   - Mathematical formulas or concepts illustrated
   - Algorithm flowcharts or processes
   - Results or findings being presented

2. On a scale of 1-10, rate the IMPORTANCE of this figure to understanding the research, where:
   - 10: Critical figure showing main results, core architecture, or central innovation
   - 7-9: Important visualization of methodology, key results, or significant components
   - 4-6: Supplementary information, examples, or secondary findings
   - 1-3: Minor illustration, decorative element, or peripheral information

Format your response as:
IMPORTANCE: [number 1-10]
DESCRIPTION: [your detailed description]"""
            
            messages = [
                ("user", [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ])
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()
            response = chain.invoke({})
            
            # Parse importance score and description
            importance_score = 5  # Default mid-range score
            description = response
            
            importance_match = re.search(r'IMPORTANCE:\s*(\d+)', response)
            if importance_match:
                importance_score = int(importance_match.group(1))
                
            description_match = re.search(r'DESCRIPTION:\s*(.*)', response, re.DOTALL)
            if description_match:
                description = description_match.group(1).strip()
            
            results.append({
                "path": filepath,
                "description": description,
                "importance": importance_score
            })
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
    
    # Sort results by importance score (descending)
    results = sorted(results, key=lambda x: x.get("importance", 0), reverse=True)
    
    logger.info(f"Selected top {min(max_images, len(results))} images by importance score")
    return results[:max_images]  # Ensure we return at most max_images

# --------------------- DATA STORAGE ---------------------
def prepare_raw_chunks(texts: List) -> List[Dict]:
    """Prepare raw text chunks for storage with section identification"""
    # Identify document structure
    structure = identify_document_structure(texts)
    
    raw_chunks = []
    for section_name, chunks in structure.items():
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            citations = extract_citations_from_text(chunk_text)
            
            raw_chunks.append({
                "text": chunk_text,
                "section": section_name,
                "section_idx": idx,
                "citations": citations
            })
    
    return raw_chunks

def store_extracted_data(
    texts: List,
    figures: List[Dict],
    metadata: PaperMetadata,
    output_path: str
) -> str:
    """Store processed data with enhanced structure"""
    try:
        # Prepare raw chunks with structure
        raw_chunks = prepare_raw_chunks(texts)
        
        # Prepare content structure
        content = {
            "raw_chunks": raw_chunks,
            "figures": figures,
            "metadata": metadata.model_dump(),
        }
        
        # Load existing content if file exists
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_content = json.load(f)
                for key, val in content.items():
                    if key not in existing_content:
                        existing_content[key] = val
        except (FileNotFoundError, json.JSONDecodeError):
            # File doesn't exist or is invalid
            existing_content = content
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_content, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Data successfully saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Data storage failed: {str(e)}")
        raise

def extract_basic_info(pdf_path: str) -> Dict:
    """Extracts metadata like title, authors, DOI, and journal from a PDF."""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        title = metadata.get("title", "Unknown Title")
        authors = metadata.get("author", "Unknown Author").split(", ") if metadata.get("author") else ["Unknown Author"]
        publication_year = metadata.get("modDate", "Unknown Year")[2:6] if metadata.get("modDate") else "Unknown Year"
        source = pdf_path  # Placeholder for now, can be replaced with a proper URL later
        
        # If title is empty or "untitled", try to extract from first page
        if not title or title.lower() == "untitled":
            first_page = doc[0]
            text = first_page.get_text()
            lines = text.split('\n')
            # Assume the first non-empty line might be the title
            for line in lines:
                if line.strip() and len(line) > 5:  # Basic check to avoid headers/page numbers
                    title = line.strip()
                    break
        
        return {
            "title": title,
            "authors": authors,
            "published_year": publication_year,
            "source": source,
            "references": []  # Will be populated later
        }
    except Exception as e:
        logger.error(f"Error extracting basic info: {str(e)}")
        return {
            "title": "Unknown Title",
            "authors": ["Unknown Author"],
            "published_year": "Unknown Year",
            "source": pdf_path,
            "references": []
        }

# --------------------- MAIN PIPELINE ---------------------
def process_research_paper(file_path: str, output_json_path: str) -> dict:
    """End-to-end processing pipeline focusing on raw chunk storage"""
    try:
        logger.info(f"Starting processing: {file_path}")
        
        # Extract basic info first
        # basic_info = extract_basic_info(file_path)
        
        # Extract content
        texts, tables, images = extract_text_tables_images(file_path)
        
        # Process images
        output_dir = os.path.join(OUTPUT_DIR, "images")
        figures = process_images(images, output_dir)
        
        # Extract metadata
        metadata = extract_metadata(texts)
        
        # Store with raw chunks instead of summaries
        output_path = store_extracted_data(
            texts,
            figures,
            metadata,
            output_json_path
        )
        
        return {
            "status": "success",
            "output_path": output_path,
            "metadata": metadata.model_dump(),
            "raw_chunks_count": len(texts),
            "figures_count": len(figures)
        }
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    result = process_research_paper(FILE_PATH, './')
    print(json.dumps(result, indent=2))