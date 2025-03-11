import re
from typing import Dict, List, Optional, Set, Tuple
import PyPDF2  # Replacing fitz (PyMuPDF)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationMapper:
    """
    A class to map numeric citations to their corresponding reference entries
    and enhance the knowledge base with detailed citation information.
    """
    
    def __init__(self, knowledge_base=None):
        """
        Initialize the CitationMapper with an optional knowledge base.
        
        Args:
            knowledge_base: Instance of ResearchKnowledgeBase
        """
        self.kb = knowledge_base
        
    def extract_references_section(self, pdf_path: str) -> str:
        """
        Extract the references section from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            String containing the references section text
        """
        try:
            full_text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    full_text += page.extract_text() or ""
            
            # Find references section using multiple possible headers
            references_start = re.search(
                r'(?:^|\n)(References|Bibliography|Cited Works|Reference List|Literature Cited)',
                full_text, re.IGNORECASE
            )
            
            if not references_start:
                logger.warning("No references section found in PDF")
                return ""
                
            # Extract text from references start to the end
            references_text = full_text[references_start.start():]
            
            # Try to determine the end of the references section (look for next section headers)
            next_section = re.search(
                r'\n(?:Appendix|Acknowledgements|Figure Legends|Abbreviations|Supplementary|Conflicts of Interest)',
                references_text, re.IGNORECASE
            )
            
            if next_section:
                references_text = references_text[:next_section.start()]
            
            # Limit to reasonable size if needed
            if len(references_text) > 50000:
                logger.warning("References section unusually large, truncating")
                references_text = references_text[:50000]
                
            return references_text
            
        except Exception as e:
            logger.error(f"Error extracting references: {e}")
            return ""
    
    def parse_numeric_references(self, references_text: str) -> Dict[str, str]:
        """
        Parse references formatted with numeric identifiers like [1], [2], etc.
        
        Args:
            references_text: Text of the references section
            
        Returns:
            Dictionary mapping citation numbers to reference text
        """
        citation_map = {}
        
        # Try different reference formats
        
        # Format [1] Author et al.
        reference_pattern1 = r'\[(\d+)\](.*?)(?=\[\d+\]|\Z)'
        reference_matches1 = re.findall(reference_pattern1, references_text, re.DOTALL)
        
        # Format 1. Author et al.
        reference_pattern2 = r'(?:^|\n)(\d+)\.\s+(.*?)(?=\n\d+\.|\Z)'
        reference_matches2 = re.findall(reference_pattern2, references_text, re.DOTALL)
        
        # Choose the pattern that found more references
        if len(reference_matches1) > len(reference_matches2):
            for num, ref_text in reference_matches1:
                citation_map[num.strip()] = ref_text.strip()
        else:
            for num, ref_text in reference_matches2:
                citation_map[num.strip()] = ref_text.strip()
        
        logger.info(f"Extracted {len(citation_map)} numeric references")
        return citation_map
    
    def parse_author_year_references(self, references_text: str) -> Dict[str, str]:
        """
        Parse references for author-year style citations like [Smith, 2020].
        
        Args:
            references_text: Text of the references section
            
        Returns:
            Dictionary mapping author-year keys to reference text
        """
        citation_map = {}
        
        # Split references by newlines and process each potential reference
        lines = references_text.split('\n')
        current_ref = ""
        
        for line in lines:
            # Check if line looks like a new reference entry
            if re.match(r'^\s*\w+', line) and len(line.strip()) > 20:
                if current_ref:  # Save previous reference if exists
                    # Try to extract author and year
                    author_match = re.search(r'([A-Z][a-z]+)(?:,?\s+et\s+al\.?)?,?\s+\(?(\d{4})\)?', current_ref)
                    if author_match:
                        author = author_match.group(1)
                        year = author_match.group(2)
                        key = f"{author}, {year}"
                        citation_map[key] = current_ref.strip()
                
                # Start new reference
                current_ref = line
            else:
                # Continue current reference
                current_ref += " " + line.strip()
        
        # Don't forget the last reference
        if current_ref:
            author_match = re.search(r'([A-Z][a-z]+)(?:,?\s+et\s+al\.?)?,?\s+\(?(\d{4})\)?', current_ref)
            if author_match:
                author = author_match.group(1)
                year = author_match.group(2)
                key = f"{author}, {year}"
                citation_map[key] = current_ref.strip()
        
        logger.info(f"Extracted {len(citation_map)} author-year references")
        return citation_map
    
    def map_citations_in_paper(self, paper_id: str, pdf_path: str) -> Dict:
        """
        Process all chunks of a paper to map citations to their reference entries.
        This updates the citation information in the knowledge base.
        
        Args:
            paper_id: ID of the paper to process
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with mapping results
        """
        if not self.kb:
            logger.error("Knowledge base not provided")
            return {"status": "error", "message": "Knowledge base not provided"}
        
        try:
            # Extract all chunks for the paper
            paper_chunks = self.kb.get_paper_chunks(paper_id)
            if not paper_chunks:
                return {"status": "error", "message": "No chunks found for paper"}
            
            # Extract references section
            references_text = self.extract_references_section(pdf_path)
            if not references_text:
                return {"status": "error", "message": "Failed to extract references"}
            
            # Parse references
            numeric_refs = self.parse_numeric_references(references_text)
            author_year_refs = self.parse_author_year_references(references_text)
            
            # Process each chunk
            updated_chunks = 0
            resolved_citations = 0
            
            for chunk in paper_chunks:
                chunk_id = chunk["chunk_id"]
                citations = chunk.get("citations", [])
                
                # Resolve each citation to reference text
                for citation in citations:
                    # For numeric citations
                    if citation.isdigit() and citation in numeric_refs:
                        reference_text = numeric_refs[citation]
                        resolved_citations += 1
                        
                        # Try to find a matching paper ID
                        cited_paper_id = self._extract_paper_id_from_reference(reference_text)
                        
                        # Create or update citation relationship in Neo4j
                        if cited_paper_id:
                            # ADD THIS NEW CODE HERE:
                            citation_info = {
                                "citation_type": "numeric",
                                "citation_number": citation,
                                "reference_text": reference_text
                            }
                            
                            with self.kb.graph_db.session() as session:
                                session.execute_write(
                                    self.kb._create_chunk_citation_relationship,
                                    chunk_id,
                                    cited_paper_id,
                                    citation_info  # Pass the citation info as third argument
                                )
                    
                    # For author-year citations
                    elif citation in author_year_refs:
                        # Add similar code here for author-year citations
                        reference_text = author_year_refs[citation]
                        resolved_citations += 1
                        
                        # Try to find a matching paper ID
                        cited_paper_id = self._extract_paper_id_from_reference(reference_text)
                        
                        # Create or update citation relationship in Neo4j
                        if cited_paper_id:
                            # ADD THIS NEW CODE HERE TOO:
                            citation_info = {
                                "citation_type": "author_year",
                                "citation_key": citation,
                                "reference_text": reference_text
                            }
                            
                            with self.kb.graph_db.session() as session:
                                session.execute_write(
                                    self.kb._create_chunk_citation_relationship,
                                    chunk_id,
                                    cited_paper_id,
                                    citation_info  # Pass the citation info as third argument
                                )    
                updated_chunks += 1
                
            return {
                "status": "success",
                "paper_id": paper_id,
                "chunks_processed": updated_chunks,
                "citations_resolved": resolved_citations,
                "numeric_references": len(numeric_refs),
                "author_year_references": len(author_year_refs)
            }
            
        except Exception as e:
            logger.error(f"Error processing citations: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def _extract_paper_id_from_reference(self, reference_text: str) -> Optional[str]:
        """
        Try to extract an arXiv ID or DOI from reference text.
        
        Args:
            reference_text: Text of the reference
            
        Returns:
            Paper ID if found, None otherwise
        """
        # Look for arXiv ID
        arxiv_match = re.search(r'arXiv:(\d+\.\d+v?\d*)', reference_text)
        if arxiv_match:
            return arxiv_match.group(1)
        
        # Look for DOI
        doi_match = re.search(r'doi:?\s*([^\s,]+)', reference_text, re.IGNORECASE)
        if doi_match:
            return doi_match.group(1)
        
        return None
    
    def generate_citation_text(self, citation_id: str, json_path: str, format_style="apa") -> str:
        """
        Generate formatted citation text for a given citation ID.
        
        Args:
            citation_id: Citation identifier
            json_path: Path to the paper's JSON metadata file
            format_style: Citation style (apa, mla, etc.)
            
        Returns:
            Formatted citation text
        """
        try:
            import json
            
            # Load paper metadata
            with open(json_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Get references
            if "basic_info" in paper_data and "references" in paper_data["basic_info"]:
                references = paper_data["basic_info"]["references"]
                
                # If citation is numeric, use it as an index
                if citation_id.isdigit():
                    idx = int(citation_id) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(references):
                        ref_id = references[idx]
                        
                        # Try to find reference details
                        if self.kb and ref_id:
                            with self.kb.graph_db.session() as session:
                                result = session.run("""
                                    MATCH (p:Paper {id: $paper_id})
                                    RETURN p.title, p.authors, p.year
                                """, {"paper_id": ref_id}).single()
                                
                                if result:
                                    title = result["p.title"]
                                    authors = result["p.authors"]
                                    year = result["p.year"]
                                    
                                    # Format according to style
                                    if format_style == "apa":
                                        authors_text = ", ".join(authors)
                                        return f"{authors_text} ({year}). {title}."
                                    else:
                                        return f"{', '.join(authors)}. \"{title}.\" {year}."
            
            # If no specific reference found, return the citation ID
            return f"[{citation_id}]"
            
        except Exception as e:
            logger.error(f"Error generating citation text: {e}")
            return f"[{citation_id}]"
    
    def enhance_json_with_citations(self, json_path: str, pdf_path: str) -> Dict:
        """
        Enhance a paper's JSON metadata with detailed citation information.
        
        Args:
            json_path: Path to the paper's JSON metadata file
            pdf_path: Path to the paper's PDF file
            
        Returns:
            Status dictionary
        """
        try:
            import json
            
            # Load paper metadata
            with open(json_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Extract paper ID
            paper_id = paper_data.get("basic_info", {}).get("paper_id", "")
            if not paper_id:
                return {"status": "error", "message": "Paper ID not found in JSON"}
            
            # Extract references section
            references_text = self.extract_references_section(pdf_path)
            if not references_text:
                return {"status": "error", "message": "Failed to extract references"}
            
            # Parse references
            numeric_refs = self.parse_numeric_references(references_text)
            author_year_refs = self.parse_author_year_references(references_text)
            
            # Create a detailed references section
            detailed_references = {}
            
            # Add numeric references
            for num, ref_text in numeric_refs.items():
                ref_id = f"ref_{num}"
                detailed_references[ref_id] = {
                    "text": ref_text,
                    "type": "numeric",
                    "number": num,
                    "arxiv_id": self._extract_paper_id_from_reference(ref_text)
                }
            
            # Add author-year references
            for key, ref_text in author_year_refs.items():
                ref_id = f"ref_{key.replace(' ', '_').replace(',', '')}"
                detailed_references[ref_id] = {
                    "text": ref_text,
                    "type": "author_year",
                    "key": key,
                    "arxiv_id": self._extract_paper_id_from_reference(ref_text)
                }
            
            # Update paper data with detailed references
            paper_data["detailed_references"] = detailed_references
            
            # Update raw chunks with citation information if present
            if "raw_chunks" in paper_data:
                for chunk in paper_data["raw_chunks"]:
                    if "citations" in chunk:
                        # Map each citation to its detailed reference
                        resolved_citations = []
                        for citation in chunk["citations"]:
                            if citation.isdigit() and citation in numeric_refs:
                                resolved_citations.append({
                                    "citation_id": citation,
                                    "reference_id": f"ref_{citation}",
                                    "reference_text": numeric_refs[citation]
                                })
                            elif citation in author_year_refs:
                                ref_id = f"ref_{citation.replace(' ', '_').replace(',', '')}"
                                resolved_citations.append({
                                    "citation_id": citation,
                                    "reference_id": ref_id,
                                    "reference_text": author_year_refs[citation]
                                })
                            else:
                                resolved_citations.append({
                                    "citation_id": citation,
                                    "reference_id": None,
                                    "reference_text": "Unknown reference"
                                })
                        
                        # Add resolved citations to chunk
                        chunk["resolved_citations"] = resolved_citations
            
            # Save updated paper data
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=4, ensure_ascii=False)
            
            return {
                "status": "success",
                "paper_id": paper_id,
                "references_added": len(detailed_references)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing JSON with citations: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

# Function to replace citations in text with their full references
def replace_citations_in_text(text: str, citation_map: Dict[str, str]) -> str:
    """
    Replace numeric citations like [1] with their full reference text.
    
    Args:
        text: Text containing citations
        citation_map: Dictionary mapping citation IDs to reference text
        
    Returns:
        Text with citations replaced
    """
    # Replace numeric citations
    for num, ref_text in citation_map.items():
        # Replace [num] with (Author et al., Year)
        # Extract author and year from reference text if possible
        author_year = ref_text
        author_match = re.search(r'([A-Z][a-z]+)(?:,?\s+et\s+al\.?)?,?\s+\(?(\d{4})\)?', ref_text)
        if author_match:
            author = author_match.group(1)
            year = author_match.group(2)
            if "et al" in ref_text:
                author_year = f"({author} et al., {year})"
            else:
                author_year = f"({author}, {year})"
                
        # Replace the citation
        text = re.sub(r'\[' + num + r'\]', author_year, text)
    
    return text

# Integration function for the papers_extractor_bfs.py script
def process_and_map_citations(paper_id: str, pdf_path: str, json_path: str, kb=None):
    """
    Process a paper to extract and map citations to references.
    Intended to be called from the papers_extractor_bfs.py script.
    
    Args:
        paper_id: ID of the paper
        pdf_path: Path to the PDF file
        json_path: Path to the JSON metadata file
        kb: Optional knowledge base instance
        
    Returns:
        Status dictionary
    """
    try:
        mapper = CitationMapper(kb)
        
        # Enhance the JSON with detailed citations
        result = mapper.enhance_json_with_citations(json_path, pdf_path)
        
        # If knowledge base is provided, also update the citation relationships
        if kb and result["status"] == "success":
            mapping_result = mapper.map_citations_in_paper(paper_id, pdf_path)
            result["kb_mapping"] = mapping_result
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing citations: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}