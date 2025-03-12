import os
import json
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
import psutil
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import gc
from chromadb.config import Settings
import openai  # Updated import
import re
import torch
import time
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# Load environment variables for API keys
load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResearchKnowledgeBase:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vector_db = self._init_chroma()
        self.graph_db = self._init_neo4j()
        self.image_db = self._init_image_collection()
        self.text_embedder = SentenceTransformer('BAAI/bge-base-en-v1.5').to(device)


    def _init_chroma(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=10000000000000  # ~10GB
            )
        client = chromadb.PersistentClient(path="./chroma_db", settings=settings)
        client.heartbeat()
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-base-en-v1.5',
            device=device
        )
        return client.get_or_create_collection(
            name="research_papers",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def _init_neo4j(self):
        return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        encrypted=False
    )

    def _init_image_collection(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=100000000000  # ~10GB
        )

        client = chromadb.PersistentClient(path="./image_db")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-base-en-v1.5',
            device=device
        )
        return client.get_or_create_collection(
            name="research_images",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_paper(self, paper_data: Dict):
        gc.collect()
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss // 1024 ** 2} MB")
        paper_id = paper_data["basic_info"]["paper_id"]
        
        try:
            print(f"Storing text chunks for paper {paper_id} in Chroma DB...")
            chunk_ids = self._store_in_chroma(paper_id, paper_data)
            paper_data["content_chunks"] = chunk_ids
            print(f"Text storage successful: {len(chunk_ids)} chunks stored")
        except Exception as e:
            import traceback
            print(f"Error storing text in Chroma: {str(e)}")
            print(traceback.format_exc())
            # Continue with other steps even if this one fails
        
        try:
            print(f"Storing paper {paper_id} in Neo4j...")
            self._store_in_neo4j(paper_id, paper_data)
            print("Neo4j storage successful")
        except Exception as e:
            import traceback
            print(f"Error storing in Neo4j: {str(e)}")
            print(traceback.format_exc())
        
        try:
            print(f"Storing images for paper {paper_id}...")
            self._store_images(paper_data)
            print("Image storage successful")
        except Exception as e:
            import traceback
            print(f"Error storing images: {str(e)}")
            print(traceback.format_exc())
        
        print(f"Finished processing paper {paper_id}")
        # time.sleep(5)
        print("======================================================")
        
        return paper_data  # Return updated paper data with chunk IDs


    def _store_in_chroma(self, paper_id: str, data: Dict) -> List[Dict]:
        """Store raw text chunks in ChromaDB and return chunk information"""
        # Get raw chunks from paper data
        raw_chunks = data.get("raw_chunks", [])
        
        if not raw_chunks:
            print(f"No raw chunks found for paper {paper_id}")
            return []
            
        print(f"Processing {len(raw_chunks)} chunks for ChromaDB storage")
        stored_chunks = []
        
        # Process in smaller batches
        BATCH_SIZE = 10
        for batch_start in range(0, len(raw_chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(raw_chunks))
            print(f"Processing batch {batch_start//BATCH_SIZE + 1}: chunks {batch_start+1}-{batch_end}")
            
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            # Process batch
            for idx in range(batch_start, batch_end):
                chunk = raw_chunks[idx]
                chunk_id = f"{paper_id}_chunk_{idx}"
                chunk_text = chunk["text"]
                # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
                # chunked_text = text_splitter.split_documents(documents)
                # Extract citations from text if they exist
                citations = self._extract_citations(chunk_text)
                embedding = self.text_embedder.encode(chunk_text, normalize_embeddings=True)
                documents.append(chunk_text)
                metadata = {
                    "paper_id": paper_id,
                    "year": data["basic_info"]["published_year"],
                    "themes": ",".join(data["metadata"]["key_themes"]) if "metadata" in data and "key_themes" in data["metadata"] else "",
                    "methods": ",".join(data["metadata"]["methodology"]) if "metadata" in data and "methodology" in data["metadata"] else "",
                    "chunk_idx": idx,
                    "section": chunk.get("section", "unknown"),
                    "citations": ",".join(citations)
                }
                metadatas.append(metadata)
                ids.append(chunk_id)
                embeddings.append(embedding.tolist())
                # Store chunk info to return
                stored_chunks.append({
                    "chunk_id": chunk_id,
                    "section": chunk.get("section", "unknown"),
                    "citations": citations
                })
            
            # Add this batch to ChromaDB
            try:
                self.vector_db.add(documents=documents,embeddings=embeddings, metadatas=metadatas, ids=ids)

            except Exception as e:
                print(f"ERROR ADDING BATCH: {str(e)}")
                import traceback
                traceback.print_exc()
                # print(f"Failed on document length: {[len(d) for d in documents]}")
        
        return stored_chunks
    def get_citation_text(self, chunk_id: str, citation_id: str) -> str:
        """
        Get the full text for a citation referenced in a chunk.
        
        Args:
            chunk_id: ID of the chunk containing the citation
            citation_id: ID of the citation (e.g., "1" for [1])
            
        Returns:
            Formatted citation text
        """
        try:
            # First try to get citation from citation relationships
            with self.graph_db.session() as session:
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:CITES]->(p:Paper)
                    WHERE c.id = $chunk_id
                    RETURN p.id, p.title, p.authors, p.year
                """, {"chunk_id": chunk_id}).single()
                
                if result:
                    paper_id = result["p.id"]
                    title = result["p.title"]
                    authors = result["p.authors"]
                    year = result["p.year"]
                    
                    # Format as APA style
                    authors_text = ", ".join(authors) if isinstance(authors, list) else authors
                    return f"{authors_text} ({year}). {title}."
            
            # If not found in graph, try to get from chunk metadata
            chunk_data = self.get_chunks_by_id([chunk_id])
            if chunk_data and len(chunk_data) > 0:
                chunk = chunk_data[0]
                if "metadata" in chunk and "citations" in chunk["metadata"]:
                    citations = chunk["metadata"]["citations"].split(",")
                    if citation_id in citations:
                        # Extract paper ID
                        paper_id = chunk["metadata"]["paper_id"]
                        
                        # Look up the paper's JSON file
                        import os
                        json_path = os.path.join("papers_summary", f"{paper_id}.json")
                        
                        # Use CitationMapper to get formatted citation
                        from citation_mapper import CitationMapper
                        mapper = CitationMapper(self)
                        return mapper.generate_citation_text(citation_id, json_path)
            
            # Default return if citation not found
            return f"[{citation_id}]"
        
        except Exception as e:
            import traceback
            print(f"Error getting citation text: {str(e)}")
            print(traceback.format_exc())
            return f"[{citation_id}]"

# Also add a method to replace citations in text:

    def replace_citations_in_text(self, text: str, chunk_id: str, detailed_references: Dict = None) -> str:
        """
        Replace numeric citations like [1] with their full reference text using detailed_references.
        
        Args:
            text: Text containing citations
            chunk_id: ID of the chunk containing the citations
            detailed_references: Optional paper's detailed references dictionary
            
        Returns:
            Text with citations replaced
        """
        try:
            # If no detailed_references provided, try to fetch it
            if detailed_references is None:
                # Get paper_id from chunk metadata
                chunks = self.get_chunks_by_id([chunk_id])
                if not chunks:
                    return text
                
                paper_id = chunks[0].get("metadata", {}).get("paper_id")
                if not paper_id:
                    return text
                
                # Load paper data from JSON file
                try:
                    paper_path = os.path.join("papers_summary", f"{paper_id}.json")
                    with open(paper_path, 'r', encoding='utf-8') as f:
                        paper_data = json.load(f)
                        detailed_references = paper_data.get("detailed_references", {})
                except Exception as e:
                    print(f"Error loading paper data: {str(e)}")
                    return text
            
            # If still no detailed_references, return original text
            if not detailed_references:
                return text
            
            # Extract numeric citations using regex - pattern like [1] or [1, 2, 3]
            numeric_pattern = r'\[(\d+)\]'
            citations = re.findall(numeric_pattern, text)
            
            # Track all citations to add to the references section later
            found_citations = {}
            
            # Replace each citation
            for citation_id in citations:
                # Look for the reference in detailed_references
                ref_key = f"ref_{citation_id}"
                if ref_key in detailed_references:
                    reference = detailed_references[ref_key]
                    
                    # Extract key citation information in a shorter format for inline citation
                    # This creates Harvard style citations like (Smith et al., 2019)
                    title = reference.get("text", "")
                    # Extract author name if possible
                    author_match = re.search(r'([A-Za-z]+),\s+[A-Za-z\.]+', title)
                    author = author_match.group(1) if author_match else "Author"
                    
                    # Extract year if possible
                    year_match = re.search(r'(\d{4})', title)
                    year = year_match.group(1) if year_match else "Year"
                    
                    # Create inline citation
                    inline_citation = f"({author}, {year})"
                    
                    # Replace the citation in the text
                    text = text.replace(f"[{citation_id}]", inline_citation)
                    
                    # Store full reference for bibliography
                    found_citations[ref_key] = reference["text"]
                
            # Also handle author-year style citations if present
            # This would require similar parsing as above
            
            return text
            
        except Exception as e:
            import traceback
            print(f"Error replacing citations: {str(e)}")
            print(traceback.format_exc())
            return text

    def _extract_citations(self, text: str) -> List[str]:
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

    def _chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """Proper text chunking using BGE's tokenizer with pre-splitting for long texts"""
        # First split by paragraphs to avoid tokenizing extremely long texts at once
        paragraphs = text.split('\n\n')
        chunks = []
        
        current_chunk = ""
        for paragraph in paragraphs:
            # Try to tokenize the current paragraph
            try:
                # Check if adding this paragraph would exceed max tokens
                test_tokens = self.text_embedder.tokenizer.encode(current_chunk + " " + paragraph)
                if len(test_tokens) > max_tokens and current_chunk:
                    # If it would exceed and we already have content, save current chunk
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Otherwise add to current chunk
                    current_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
            except Exception as e:
                print(f"Tokenization error with paragraph: {e}")
                # If a single paragraph is too long, split it by sentences
                sentences = paragraph.split('.')
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    try:
                        test_tokens = self.text_embedder.tokenizer.encode(current_chunk + " " + sentence + ".")
                        if len(test_tokens) > max_tokens and current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = sentence + "."
                        else:
                            current_chunk = current_chunk + " " + sentence + "." if current_chunk else sentence + "."
                    except Exception as e:
                        print(f"Skipping sentence due to tokenization error: {e}")
                        continue
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Final check: if any chunk is still too long, force split it
        final_chunks = []
        for chunk in chunks:
            try:
                tokens = self.text_embedder.tokenizer.encode(chunk)
                if len(tokens) <= max_tokens:
                    final_chunks.append(chunk)
                else:
                    # If still too long, force split by token count
                    for i in range(0, len(tokens), max_tokens):
                        sub_tokens = tokens[i:i+max_tokens]
                        sub_chunk = self.text_embedder.tokenizer.decode(
                            sub_tokens, 
                            skip_special_tokens=True
                        )
                        final_chunks.append(sub_chunk)
            except Exception as e:
                print(f"Final chunking error: {e}")
                continue
                
        return final_chunks

    def _store_images(self, paper_data: Dict):
        paper_id = paper_data["basic_info"]["paper_id"]
        cnt_img = 0
        for img in paper_data.get("figures", []):
            path = Path(img["path"])
            img_id = path.stem
            description = img["description"]
            documents = []
            metadatas = []
            ids = []
            if cnt_img > 10:
                break
            # Simplify - just limit by character count if needed
            if len(description) > 500:  # Arbitrary limit to prevent very long descriptions
                description = description[:500] + "..."
            documents.append(description)
            metadatas.append({
                "paper_id": paper_id,
                "path": img['path'],
                "themes": ",".join(paper_data["metadata"]["key_themes"]) if "metadata" in paper_data and "key_themes" in paper_data["metadata"] else "",
                "methods": ",".join(paper_data["metadata"]["methodology"]) if "metadata" in paper_data and "methodology" in paper_data["metadata"] else ""
            })
            ids.append(img_id)
            cnt_img += 1
            self.image_db.add(documents=documents, metadatas=metadatas, ids=ids)

    def _store_in_neo4j(self, paper_id: str, data: Dict):
        with self.graph_db.session() as session:
            # Create paper node
            session.execute_write(
                self._create_paper_node,
                paper_id,
                data["basic_info"]
            )
            
            # Create metadata relationships if they exist
            if "metadata" in data:
                if "key_themes" in data["metadata"]:
                    for theme in data["metadata"]["key_themes"]:
                        session.execute_write(
                            self._create_theme_relationship,
                            paper_id,
                            theme
                        )
                
                if "methodology" in data["metadata"]:
                    for method in data["metadata"]["methodology"]:
                        session.execute_write(
                            self._create_methodology_relationship,
                            paper_id,
                            method
                        )
                
                if "strengths" in data["metadata"]:
                    for strength in data["metadata"]["strengths"]:
                        session.execute_write(
                            self._create_strength_relationship,
                            paper_id,
                            strength
                        )
                
                if "limitations" in data["metadata"]:
                    for limitation in data["metadata"]["limitations"]:
                        session.execute_write(
                            self._create_limitation_relationship,
                            paper_id,
                            limitation
                        )
                
                if "domain" in data["metadata"]:
                    for domain in data["metadata"]["domain"]:
                        session.execute_write(
                            self._create_domain_relationship,
                            paper_id,
                            domain
                        )
            
            # Create reference relationships
            if "basic_info" in data and "references" in data["basic_info"]:
                for reference in data["basic_info"]["references"]:
                    session.execute_write(
                        self._create_reference_relationship,
                        paper_id,
                        reference
                    )
            
            # Store content chunks
            if "content_chunks" in data:
                for chunk in data["content_chunks"]:
                    chunk_id = chunk["chunk_id"]
                    section = chunk.get("section", "unknown")
                    
                    # Create chunk node and relationship to paper
                    session.execute_write(
                        self._create_chunk_node,
                        paper_id,
                        chunk_id,
                        section
                    )
                    
                    # Create citation relationships from chunks
                    if "citations" in chunk and chunk["citations"]:
                        for citation in chunk["citations"]:
                            # Try to match citation to a paper ID
                            cited_paper_id = self._resolve_citation(citation, data["basic_info"].get("references", []))
                            if cited_paper_id:
                                session.execute_write(
                                    self._create_chunk_citation_relationship,
                                    chunk_id,
                                    cited_paper_id
                                )

    def _resolve_citation(self, citation: str, references: List[str]) -> Optional[str]:
        """Try to match a citation string to a paper ID from the references list"""
        # If citation is just a number, try to use it as an index
        if citation.isdigit():
            idx = int(citation) - 1  # Convert to 0-indexed
            if 0 <= idx < len(references):
                return references[idx]
        
        # Otherwise try fuzzy matching (not implemented here)
        # This would require more complex logic to match citation text to paper IDs
        
        return None

    @staticmethod
    def _create_paper_node(tx, paper_id: str, basic_info: Dict):
        tx.run("""
            MERGE (p:Paper {id: $paper_id})
            SET p.title = $title,
                p.authors = $authors,
                p.year = $year
        """, {
            "paper_id": paper_id,
            "title": basic_info.get("title", "Unknown Title"),
            "authors": basic_info.get("authors", ["Unknown"]),
            "year": basic_info.get("published_year", "Unknown Year"),
        })

    @staticmethod
    def _create_chunk_node(tx, paper_id: str, chunk_id: str, section: str):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.section = $section
            MERGE (p)-[:CONTAINS]->(c)
        """, {
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "section": section
        })

    @staticmethod
    def _create_chunk_citation_relationship(tx, chunk_id: str, cited_paper_id: str, citation_info=None):
        """
        Create or update a citation relationship between a chunk and a paper.
        
        Args:
            tx: Neo4j transaction
            chunk_id: ID of the citing chunk
            cited_paper_id: ID of the cited paper
            citation_info: Optional dictionary with additional citation metadata
        """
        # Base query to create the relationship
        query = """
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (cited:Paper {id: $cited_paper_id})
            MERGE (c)-[r:CITES]->(cited)
            SET r.strength = coalesce(r.strength, 0) + 1
        """
        
        # Parameters for the query
        params = {
            "chunk_id": chunk_id,
            "cited_paper_id": cited_paper_id
        }
        
        # If we have additional citation information, add it to the relationship
        if citation_info:
            properties = []
            for key, value in citation_info.items():
                if key not in ["chunk_id", "cited_paper_id"]:
                    property_name = key.replace(" ", "_").lower()
                    params[property_name] = value
                    properties.append(f"r.{property_name} = ${property_name}")
            
            if properties:
                # Add additional SET clauses for the properties
                property_string = ", ".join(properties)
                query = query.replace("SET r.strength = coalesce(r.strength, 0) + 1", 
                                f"SET r.strength = coalesce(r.strength, 0) + 1, {property_string}")
        
        # Execute the query
        tx.run(query, params)

    @staticmethod
    def _create_theme_relationship(tx, paper_id: str, theme: str):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (t:Theme {name: $theme})
            MERGE (p)-[r:DISCUSSES_THEME]->(t)
            SET r.strength = coalesce(r.strength, 0) + 1
        """, {"paper_id": paper_id, "theme": theme})

    @staticmethod
    def _create_methodology_relationship(tx, paper_id: str, method: str):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (m:Methodology {name: $method})
            MERGE (p)-[r:USES_METHOD]->(m)
            SET r.strength = coalesce(r.strength, 0) + 1
        """, {"paper_id": paper_id, "method": method})

    @staticmethod
    def _create_strength_relationship(tx, paper_id: str, strength: str):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (s:Strength {name: $strength})
            MERGE (p)-[r:HAS_STRENGTH]->(s)
            SET r.strength = coalesce(r.strength, 0) + 1
        """, {"paper_id": paper_id, "strength": strength})

    @staticmethod
    def _create_limitation_relationship(tx, paper_id: str, limitation: str):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (l:Limitation {name: $limitation})
            MERGE (p)-[r:HAS_LIMITATION]->(l)
            SET r.strength = coalesce(r.strength, 0) + 1
        """, {"paper_id": paper_id, "limitation": limitation})

    @staticmethod
    def _create_domain_relationship(tx, paper_id: str, domain: str):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (d:Domain {name: $domain})
            MERGE (p)-[r:BELONGS_TO_DOMAIN]->(d)
            SET r.strength = coalesce(r.strength, 0) + 1
        """, {"paper_id": paper_id, "domain": domain})
    
    @staticmethod
    def _create_reference_relationship(tx, paper_id: str, reference: str):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (cited:Paper {id: $reference})
            MERGE (p)-[r:CITES]->(cited)
            SET r.strength = coalesce(r.strength, 0) + 1
        """, {"paper_id": paper_id, "reference": reference})

    @staticmethod
    def _create_image_nodes(tx, paper_id: str, image: Dict):
        tx.run("""
            MATCH (p:Paper {id: $paper_id})
            MERGE (i:Image {path: $path})
            SET i.description = $desc
            MERGE (p)-[:HAS_IMAGE]->(i)
        """, {
            "paper_id": paper_id,
            "path": image["path"],
            "desc": image["description"]
        })
        for theme in image.get("themes", []):
            tx.run("""
                MATCH (i:Image {path: $path})
                MERGE (t:Theme {name: $theme})
                MERGE (i)-[:ILLUSTRATES]->(t)
            """, {"path": image["path"], "theme": theme})
        for method in image.get("methods", []):
            tx.run("""
                MATCH (i:Image {path: $path})
                MERGE (m:Methodology {name: $method})
                MERGE (i)-[:DEMONSTRATES]->(m)
            """, {"path": image["path"], "method": method})

    def hybrid_search(self, query: str, top_k: int = 5, include_images: bool = True) -> Dict:
        """Enhanced hybrid search leveraging chunk-based storage and Neo4j relationships"""
        # Text search on chunks
        chunk_results = self.vector_db.query(
            query_texts=[query],
            n_results=top_k*3,  # Get extra chunks for aggregation
            include=["metadatas", "documents", "distances"]
        )
        print("done vector search")
        # print(chunk_results)
        # Aggregate chunks by paper
        paper_chunks = {}
        for doc, meta, dist in zip(chunk_results["documents"][0],
                                chunk_results["metadatas"][0],
                                chunk_results["distances"][0]):
            paper_id = meta["paper_id"]
            if paper_id not in paper_chunks:
                paper_chunks[paper_id] = {
                    "chunks": [],
                    "min_distance": float('inf'),
                    "themes": set(),
                    "methods": set(),
                    "domains": set(),
                    "strengths": set(),
                    "limitations": set(),
                    "citations": set()
                }
            
            paper_chunks[paper_id]["chunks"].append({
                "text": doc,
                "distance": dist,
                "chunk_id": meta.get("chunk_id", ""),
                "chunk_idx": meta.get("chunk_idx", 0),
                "section": meta.get("section", "unknown")
            })
            
            paper_chunks[paper_id]["min_distance"] = min(
                paper_chunks[paper_id]["min_distance"], 
                dist
            )
            
            if "themes" in meta and meta["themes"]:
                paper_chunks[paper_id]["themes"].update(meta["themes"].split(','))
            if "methods" in meta and meta["methods"]:
                paper_chunks[paper_id]["methods"].update(meta["methods"].split(','))
            if "domain" in meta and meta["domain"]:
                paper_chunks[paper_id]["domains"].update(meta["domain"].split(','))
            if "strengths" in meta and meta["strengths"]:
                paper_chunks[paper_id]["strengths"].update(meta["strengths"].split(','))
            if "limitations" in meta and meta["limitations"]:
                paper_chunks[paper_id]["limitations"].update(meta["limitations"].split(','))
            if "citations" in meta and meta["citations"]:
                paper_chunks[paper_id]["citations"].update(meta["citations"].split(','))
        
        # Sort papers by minimum chunk distance and take top_k
        sorted_papers = sorted(
            paper_chunks.items(),
            key=lambda x: x[1]["min_distance"]
        )[:top_k]
        
        # Enrich with graph data and process chunks
        papers = []
        for paper_id, data in sorted_papers:
            try:
                print("started graph seach")
                # Get base paper info from Neo4j
                with self.graph_db.session() as session:
                    graph_data = session.execute_read(
                        self._get_enriched_paper_details,
                        paper_id
                    )
                    
                # Process chunks
                sorted_chunks = sorted(
                    data["chunks"], 
                    key=lambda x: x["distance"]
                )
                
                # Get raw text for top chunks
                top_chunks = []
                for chunk in sorted_chunks[:3]:  # Take top 3 chunks
                    top_chunks.append({
                        "text": chunk["text"],
                        "section": chunk["section"],
                        "distance": chunk["distance"]
                    })
                
                papers.append({
                    "id": paper_id,
                    "chunks": top_chunks,
                    "themes": list(data["themes"]),
                    "methods": list(data["methods"]),
                    "domains": list(data["domains"]),
                    "strengths": list(data["strengths"]),
                    "limitations": list(data["limitations"]),
                    "citations": list(data["citations"]),
                    "graph_data": graph_data,
                    "chunk_count": len(data["chunks"]),
                    "min_distance": data["min_distance"]
                })
            except Exception as e:
                print(f"Error processing paper {paper_id}: {str(e)}")
                continue
        print("starting images")
        # Image search with theme/method alignment
        images = []
        if include_images:
            try:
                image_results = self.image_db.query(
                    query_texts=[query],
                    n_results=top_k*2,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Filter images related to found papers or themes/methods/domains
                found_themes = set(t for p in papers for t in p["themes"])
                found_methods = set(m for p in papers for m in p["methods"])
                found_domains = set(d for p in papers for d in p.get("domains", []))
                
                for doc, meta, dist in zip(image_results["documents"][0],
                                image_results["metadatas"][0],
                                image_results["distances"][0]):
                    if (meta["paper_id"] in paper_chunks or
                        any(t in found_themes for t in meta["themes"].split(',') if meta.get("themes")) or
                        any(m in found_methods for m in meta["methods"].split(',') if meta.get("methods")) or
                        any(d in found_domains for d in meta.get("domain", "").split(',') if meta.get("domain"))):
                        
                        images.append({
                            "path": meta["path"],
                            "paper_id": meta["paper_id"],
                            "description": doc,
                            "themes": meta["themes"].split(',') if meta.get("themes") else [],
                            "methods": meta["methods"].split(',') if meta.get("methods") else [],
                            "domains": meta.get("domain", "").split(',') if meta.get("domain") else [],
                            "distance": dist
                        })
                    if len(images) >= top_k:
                        break
            except Exception as e:
                print(f"Error processing images: {str(e)}")
        for paper in papers:
        # Load paper data once for each paper to access detailed_references
            try:
                paper_id = paper["id"]
                paper_path = os.path.join("papers_summary", f"{paper_id}.json")
                with open(paper_path, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Extract detailed_references from paper data for citation replacement
                detailed_references = paper_data.get("detailed_references", {})
                
                # Add content field with combined chunks for the review paper writer
                content = []
                for chunk in paper["chunks"]:
                    # Replace citations in each chunk
                    chunk_text = self.replace_citations_in_text(
                        chunk["text"], 
                        chunk.get("chunk_id", ""), 
                        detailed_references
                    )
                    content.append(chunk_text)
                
                # Add the combined content to the paper
                paper["content"] = "\n\n".join(content)
            except Exception as e:
                print(f"Error processing paper {paper_id}: {str(e)}")
                paper["content"] = "\n\n".join([chunk["text"] for chunk in paper["chunks"]])
        
        return {
            "papers": papers,
            "images": images
        }
        

    @staticmethod
    def _get_enriched_paper_details(tx, paper_id: str):
        """Get extended paper relationships from Neo4j including all metadata and chunks."""
        result = tx.run("""
            MATCH (p:Paper{id: $paper_id})

            // Get basic paper info immediately - no need for related papers or collections if that's all we need
            WITH p, p.id AS paper_id, p.title AS title, p.authors AS authors, p.year AS year

            // Get chunks with LIMIT to prevent collecting too many
            OPTIONAL MATCH (p)-[:CONTAINS]->(c:Chunk)
            WITH p, paper_id, title, authors, year, 
                collect(DISTINCT c.id)[..20] AS chunk_ids,  // Limit to 20 chunks
                collect(DISTINCT c.section)[..10] AS chunk_sections

            // Get citations with LIMIT
            OPTIONAL MATCH (p)-[:CONTAINS]->(:Chunk)-[:CITES]->(cited:Paper)
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections,
                collect(DISTINCT cited.id)[..15] AS chunk_citations  // Limit to 15 citations

            // Split the query into multiple parts - get themes and theme-related papers
            OPTIONAL MATCH (p)-[:DISCUSSES_THEME]->(t:Theme)
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations,
                collect(DISTINCT t.name)[..10] AS themes  // Limit to 10 themes

            // Get theme-related papers separately with LIMIT
            OPTIONAL MATCH (p)-[:DISCUSSES_THEME]->(t:Theme)<-[:DISCUSSES_THEME]-(theme_related:Paper)
            WHERE theme_related <> p
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, themes,
                collect(DISTINCT theme_related.id)[..5] AS theme_related_papers  // Limit to 5 related papers

            // Get methods with LIMIT
            OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Methodology)
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers,
                collect(DISTINCT m.name)[..8] AS methods  // Limit to 8 methods

            // Get method-related papers separately with LIMIT
            OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Methodology)<-[:USES_METHOD]-(method_related:Paper)
            WHERE method_related <> p
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers, methods,
                collect(DISTINCT method_related.id)[..5] AS method_related_papers  // Limit to 5 related papers

            // Get domains with LIMIT
            OPTIONAL MATCH (p)-[:BELONGS_TO_DOMAIN]->(d:Domain)
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers, methods, method_related_papers,
                collect(DISTINCT d.name)[..5] AS domains  // Limit to 5 domains

            // Get domain-related papers separately with LIMIT
            OPTIONAL MATCH (p)-[:BELONGS_TO_DOMAIN]->(d:Domain)<-[:BELONGS_TO_DOMAIN]-(domain_related:Paper)
            WHERE domain_related <> p
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers, methods, method_related_papers, domains,
                collect(DISTINCT domain_related.id)[..5] AS domain_related_papers  // Limit to 5 related papers

            // Get strengths with LIMIT
            OPTIONAL MATCH (p)-[:HAS_STRENGTH]->(s:Strength)
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers, methods, method_related_papers, domains, domain_related_papers,
                collect(DISTINCT s.name)[..8] AS strengths  // Limit to 8 strengths

            // Get strength-related papers separately with LIMIT
            OPTIONAL MATCH (p)-[:HAS_STRENGTH]->(s:Strength)<-[:HAS_STRENGTH]-(strength_related:Paper)
            WHERE strength_related <> p
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers, methods, method_related_papers, domains, domain_related_papers,
                strengths,
                collect(DISTINCT strength_related.id)[..5] AS strength_related_papers  // Limit to 5 related papers

            // Get limitations with LIMIT
            OPTIONAL MATCH (p)-[:HAS_LIMITATION]->(l:Limitation)
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers, methods, method_related_papers, domains, domain_related_papers,
                strengths, strength_related_papers,
                collect(DISTINCT l.name)[..8] AS limitations  // Limit to 8 limitations

            // Get limitation-related papers separately with LIMIT
            OPTIONAL MATCH (p)-[:HAS_LIMITATION]->(l:Limitation)<-[:HAS_LIMITATION]-(limitation_related:Paper)
            WHERE limitation_related <> p
            WITH p, paper_id, title, authors, year, chunk_ids, chunk_sections, chunk_citations, 
                themes, theme_related_papers, methods, method_related_papers, domains, domain_related_papers,
                strengths, strength_related_papers, limitations,
                collect(DISTINCT limitation_related.id)[..5] AS limitation_related_papers  // Limit to 5 related papers

            // Final RETURN statement with all collected data
            RETURN 
                paper_id, title, authors, year,
                chunk_ids, chunk_sections, chunk_citations,
                themes, methods, domains, strengths, limitations,
                theme_related_papers, method_related_papers, domain_related_papers,
                strength_related_papers, limitation_related_papers

        """, {"paper_id": paper_id})
        
        record = result.single()
        if record is None:
            return {"paper_id": paper_id, "found": False}
        return dict(record)

    
    def get_chunks_by_id(self, chunk_ids: List[str]) -> List[Dict]:
        """Retrieve full chunk data by IDs"""
        if not chunk_ids:
            return []
            
        results = []
        for chunk_id in chunk_ids:
            try:
                # Fetch chunk from ChromaDB
                chunk_data = self.vector_db.get(
                    ids=[chunk_id],
                    include=["metadatas", "documents"]
                )
                
                if chunk_data and chunk_data["documents"]:
                    results.append({
                        "chunk_id": chunk_id,
                        "text": chunk_data["documents"][0],
                        "metadata": chunk_data["metadatas"][0] if chunk_data["metadatas"] else {}
                    })
            except Exception as e:
                print(f"Error retrieving chunk {chunk_id}: {str(e)}")
                
        return results

    def get_paper_chunks(self, paper_id: str) -> List[Dict]:
        """Get all chunks for a specific paper"""
        # Query ChromaDB for chunks with matching paper_id
        results = self.vector_db.get(
            where={"paper_id": paper_id},
            include=["metadatas", "documents", "embeddings"]
        )
        
        chunks = []
        if results and results["documents"]:
            for doc, meta, embedding in zip(results["documents"], results["metadatas"], results["embeddings"]):
                chunks.append({
                    "chunk_id": meta.get("chunk_id", ""),
                    "text": doc,
                    "section": meta.get("section", "unknown"),
                    "citations": meta.get("citations", "").split(",") if meta.get("citations") else [],
                    "embedding": embedding
                })
                
        return chunks

    def is_db_populated(self) -> bool:
        paper_count = self.vector_db.count()
        image_count = self.image_db.count()
        return paper_count > 0 and image_count > 0
    def generate_review_paper(self, query: str, top_k: int = 10) -> Dict:
        """Generate content for a review paper based on the provided query"""
        # First, perform a hybrid search to find relevant papers and chunks
        search_results = self.hybrid_search(query, top_k=top_k)
        
        # Extract key papers and their most relevant chunks
        relevant_papers = search_results["papers"]
        
        # Organize information for the review
        review_data = {
            "query": query,
            "papers": [],
            "themes": self._aggregate_themes(relevant_papers),
            "methods": self._aggregate_methods(relevant_papers),
            "domains": self._aggregate_domains(relevant_papers),
            "strengths": self._aggregate_strengths(relevant_papers),
            "limitations": self._aggregate_limitations(relevant_papers),
            "figures": search_results["images"]
        }
        
        # Process each paper to include its chunks
        for paper in relevant_papers:
            paper_data = {
                "id": paper["id"],
                "title": paper.get("graph_data", {}).get("title", "Unknown Title"),
                "authors": paper.get("graph_data", {}).get("authors", ["Unknown"]),
                "year": paper.get("graph_data", {}).get("year", "Unknown Year"),
                "chunks": paper["chunks"],
                "citations": paper["citations"],
                "themes": paper["themes"],
                "methods": paper["methods"],
                "strengths": paper["strengths"],
                "limitations": paper["limitations"]
            }
            review_data["papers"].append(paper_data)
        
        # Use citation graph to suggest structure
        review_data["citation_network"] = self._analyze_citation_network(relevant_papers)
    
        return review_data
    def _aggregate_themes(self, papers: List[Dict]) -> List[Dict]:
        """Aggregate and rank themes from the most relevant papers"""
        theme_counts = {}
        for paper in papers:
            for theme in paper["themes"]:
                if theme in theme_counts:
                    theme_counts[theme] += 1
                else:
                    theme_counts[theme] = 1
        
        # Sort by frequency and return
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"name": theme, "count": count} for theme, count in sorted_themes]
    
    def _aggregate_methods(self, papers: List[Dict]) -> List[Dict]:
        """Aggregate and rank methods from the most relevant papers"""
        method_counts = {}
        for paper in papers:
            for method in paper["methods"]:
                if method in method_counts:
                    method_counts[method] += 1
                else:
                    method_counts[method] = 1
        
        # Sort by frequency and return
        sorted_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"name": method, "count": count} for method, count in sorted_methods]
    def _aggregate_strengths(self, papers: List[Dict]) -> List[Dict]:
        """Aggregate and rank strengths from the most relevant papers"""
        strength_counts = {}
        for paper in papers:
            for strength in paper.get("strengths", []):
                if strength in strength_counts:
                    strength_counts[strength] += 1
                else:
                    strength_counts[strength] = 1
        
        # Sort by frequency and return
        sorted_strengths = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"name": strength, "count": count} for strength, count in sorted_strengths]
    
    def _aggregate_limitations(self, papers: List[Dict]) -> List[Dict]:
        """Aggregate and rank limitations from the most relevant papers"""
        limitation_counts = {}
        for paper in papers:
            for limitation in paper.get("limitations", []):
                if limitation in limitation_counts:
                    limitation_counts[limitation] += 1
                else:
                    limitation_counts[limitation] = 1
        
        # Sort by frequency and return
        sorted_limitations = sorted(limitation_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"name": limitation, "count": count} for limitation, count in sorted_limitations]

    def _aggregate_domains(self, papers: List[Dict]) -> List[Dict]:
        """Aggregate and rank domains from the most relevant papers"""
        domain_counts = {}
        for paper in papers:
            for domain in paper.get("domains", []):
                if domain in domain_counts:
                    domain_counts[domain] += 1
                else:
                    domain_counts[domain] = 1
        
        # Sort by frequency and return
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"name": domain, "count": count} for domain, count in sorted_domains]
    
    def _analyze_citation_network(self, papers: List[Dict]) -> Dict:
        """Analyze the citation network among relevant papers to suggest structure"""
        # Identify foundational papers (heavily cited)
        paper_ids = [p["id"] for p in papers]
        citation_counts = {}
        for paper in papers:
            for citation in paper["citations"]:
                if citation in paper_ids:  # Only count citations to other papers in our set
                    if citation in citation_counts:
                        citation_counts[citation] += 1
                    else:
                        citation_counts[citation] = 1
        
        # Group papers by theme and method
        theme_groups = {}
        method_groups = {}
        strength_groups = {}
        limitation_groups = {}

        
        for paper in papers:
            for theme in paper["themes"]:
                if theme not in theme_groups:
                    theme_groups[theme] = []
                theme_groups[theme].append(paper["id"])
            
            for method in paper["methods"]:
                if method not in method_groups:
                    method_groups[method] = []
                method_groups[method].append(paper["id"])
            
            for strength in paper.get("strengths", []):
                if strength not in strength_groups:
                    strength_groups[strength] = []
                strength_groups[strength].append(paper["id"])

            for limitation in paper.get("limitations", []):
                if limitation not in limitation_groups:
                    limitation_groups[limitation] = []
                limitation_groups[limitation].append(paper["id"])

        
        return {
            "most_cited": sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "theme_groups": theme_groups,
            "method_groups": method_groups,
            "strength_groups": strength_groups,
            "limitation_groups": limitation_groups
        }

def ingest_json_directory(kb: ResearchKnowledgeBase, json_dir: str = "papers_summary"):
    json_files = list(Path(json_dir).glob("*.json"))
    successful = 0
    failed = 0
    
    for json_path in tqdm(json_files, desc="Ingesting papers"):
        try:
            print(f"\nProcessing {json_path.name}")
            with open(json_path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
            
            updated_data = kb.ingest_paper(paper_data)
            
            # Save the updated data with chunk IDs back to the file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, indent=4, ensure_ascii=False)
                
            successful += 1
            print(f"Successfully processed {json_path.name}")
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in {json_path.name}: {str(e)}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"Error processing {json_path.name}: {str(e)}")
            print(traceback.format_exc())
            failed += 1
            
    print(f"\nIngestion complete: {successful} files processed successfully, {failed} files failed")
    return successful > 0

def get_review_topics(kb: ResearchKnowledgeBase, num_topics=5) -> List[Dict]:
    try:
        papers = kb.vector_db.get(include=["embeddings", "metadatas"])
        if not papers or "embeddings" not in papers or len(papers["embeddings"]) == 0:
            print("No papers found in the database")
            return []
            
        method_vectors = np.array(papers["embeddings"], dtype=np.float32)
        if method_vectors.size == 0:
            print("No valid embeddings found for clustering")
            return []
            
        try:
            kmeans = KMeans(n_clusters=min(3, len(method_vectors)), random_state=0, n_init=10)
            kmeans.fit(method_vectors)
            cluster_labels = kmeans.labels_
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            return []
            
        # Process clusters
        method_clusters = {}
        for i in range(kmeans.n_clusters):
            methods = []
            for idx, meta in enumerate(papers["metadatas"]):
                if cluster_labels[idx] == i and "methods" in meta:
                    methods.extend(meta["methods"].split(','))
            unique_methods = list(set(methods))
            method_clusters[f"cluster_{i}"] = unique_methods
            
        # Query Neo4j for emerging themes, strengths, limitations and interdisciplinary connections
        with kb.graph_db.session() as session:
            emerging_themes = session.execute_read(
                lambda tx: tx.run("""
                MATCH (t:Theme)<-[:DISCUSSES_THEME]-(p:Paper)
                WHERE p.year >= 2022
                WITH t.name AS theme, COUNT(p) AS paper_count
                ORDER BY paper_count DESC
                LIMIT 10
                RETURN theme, paper_count
                """).data()
            )
            
            # Get common strengths from recent papers
            common_strengths = session.execute_read(
                lambda tx: tx.run("""
                MATCH (s:Strength)<-[:HAS_STRENGTH]-(p:Paper)
                WHERE p.year >= 2022
                WITH s.name AS strength, COUNT(p) AS paper_count
                ORDER BY paper_count DESC
                LIMIT 8
                RETURN strength, paper_count
                """).data()
            )
            
            # Get common limitations from recent papers
            common_limitations = session.execute_read(
                lambda tx: tx.run("""
                MATCH (l:Limitation)<-[:HAS_LIMITATION]-(p:Paper)
                WHERE p.year >= 2022
                WITH l.name AS limitation, COUNT(p) AS paper_count
                ORDER BY paper_count DESC
                LIMIT 8
                RETURN limitation, paper_count
                """).data()
            )
            
            # Get improvement opportunities (strengths addressing others' limitations)
            improvement_opportunities = session.execute_read(
                lambda tx: tx.run("""
                MATCH (p1:Paper)-[:HAS_STRENGTH]->(s:Strength)
                MATCH (p2:Paper)-[:HAS_LIMITATION]->(l:Limitation)
                WHERE p1 <> p2 AND p1.year >= p2.year
                WITH s.name AS strength, l.name AS limitation, COUNT(DISTINCT p1) AS strength_count, COUNT(DISTINCT p2) AS limitation_count
                ORDER BY (strength_count + limitation_count) DESC
                LIMIT 6
                RETURN strength, limitation, strength_count, limitation_count
                """).data()
            )
            
            interdisciplinary = session.execute_read(
                lambda tx: tx.run("""
                MATCH (t1:Theme)<-[:DISCUSSES_THEME]-(p:Paper)-[:DISCUSSES_THEME]->(t2:Theme)
                WHERE t1 <> t2 AND p.year >= 2020
                WITH t1.name AS theme1, t2.name AS theme2, COUNT(p) AS connection_strength
                ORDER BY connection_strength DESC
                LIMIT 5
                RETURN theme1 + " & " + theme2 AS combined_theme, connection_strength
                """).data()
            )
        return _synthesize_topics(
            emerging_themes=emerging_themes,
            method_clusters=method_clusters,
            interdisciplinary=interdisciplinary,
            common_strengths=common_strengths,
            common_limitations=common_limitations,
            improvement_opportunities=improvement_opportunities,
            num_topics=num_topics
        )
    except Exception as e:
        print(f"Error generating topics: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

def _synthesize_topics(emerging_themes, method_clusters, interdisciplinary, common_strengths, common_limitations, improvement_opportunities, num_topics=5) -> List[Dict]:
    themes_text = "\n".join([f"- {item['theme']} ({item['paper_count']} papers)" for item in emerging_themes]) if emerging_themes else "No emerging themes found"
    methods_text = "\n".join([f"Cluster {cid}: {', '.join(methods)}" for cid, methods in method_clusters.items()]) if method_clusters else "No method clusters found"
    interdisciplinary_text = "\n".join([f"- {item['combined_theme']} ({item['connection_strength']} connections)" for item in interdisciplinary]) if interdisciplinary else "No interdisciplinary connections found"
    strengths_text = "\n".join([f"- {item['strength']} ({item['paper_count']} papers)" for item in common_strengths]) if common_strengths else "No common strengths found"
    limitations_text = "\n".join([f"- {item['limitation']} ({item['paper_count']} papers)" for item in common_limitations]) if common_limitations else "No common limitations found"
    
    improvement_text = ""
    if improvement_opportunities:
        improvement_text = "\n".join([f"- Strength '{item['strength']}' ({item['strength_count']} papers) could address limitation '{item['limitation']}' ({item['limitation_count']} papers)" 
                                   for item in improvement_opportunities])
    else:
        improvement_text = "No clear improvement opportunities found"

    prompt = f"""
    Generate {num_topics} research review paper topics based on:

    Emerging Themes:
    {themes_text}

    Methodology Clusters:
    {methods_text}

    Interdisciplinary Connections:
    {interdisciplinary_text}

    Common Strengths in Recent Papers:
    {strengths_text}
    
    Common Limitations in Recent Papers:
    {limitations_text}
    
    Potential Improvement Opportunities:
    {improvement_text}


    Guidelines:
    - Combine 2-3 concepts from different areas
    - Focus on recent developments (post-2020)
    - Format: "Advances in [TECH] for [DOMAIN]: [SPECIFIC FOCUS]"
    - Include both theoretical and applied aspects

    Return a JSON list with: "title", "focus", "themes", "methods", "key_strengths", "addressing_limitations"
    """
    
    try:
        # Using new OpenAI API format (v1.0.0+)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research advisor specialized in identifying novel research directions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        # Extract content from new response format
        content = response.choices[0].message.content.strip()
        return json.loads(content)[:num_topics]
    except Exception as e:
        print(f"Error generating topics with LLM: {str(e)}")
        return []

if __name__ == "__main__":
    kb = ResearchKnowledgeBase()
    if not kb.is_db_populated():
        ingest_json_directory(kb)
        print(f"Papers in vector DB: {kb.vector_db.count()}")
        print(f"Images in vector DB: {kb.image_db.count()}")
    else:
        print("Databases already populated. Skipping ingestion.")
    
    print("Generating review topics...")
    topics = get_review_topics(kb)
    print(json.dumps(topics, indent=2))
    with open("review_topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)
    search_result = kb.hybrid_search("Tri-hybrid Representation in 3D Reconstruction: Navigating through Recent Innovations", include_images = True)
    with open("search_result.json", "w", encoding="utf-8") as f:
        json.dump(search_result, f, indent=2)