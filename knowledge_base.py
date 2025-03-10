import os
import json
import hashlib
from typing import Dict, List, Optional
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
import psutil
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import gc
from chromadb.config import Settings
import openai  # Updated import

# Load environment variables for API keys
load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResearchKnowledgeBase:
    def __init__(self):
        self.vector_db = self._init_chroma()
        self.graph_db = self._init_neo4j()
        self.image_db = self._init_image_collection()
        self.text_embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')


    def _init_chroma(self):
        settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=10000000000  # ~10GB
            )
        client = chromadb.PersistentClient(path="./chroma_db")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-base-en-v1.5'
        )
        return client.get_or_create_collection(
            name="research_papers",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def _init_neo4j(self):
        return GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "research123"),
            encrypted=False
        )

    def _init_image_collection(self):
        settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=10000000000  # ~10GB
        )

        client = chromadb.PersistentClient(path="./chroma_db")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-base-en-v1.5'
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
            print(f"Storing text for paper {paper_id} in Chroma DB...")
            self._store_in_chroma(paper_id, paper_data)
            print("Text storage successful")
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
        print("======================================================")


    def _store_in_chroma(self, paper_id: str, data: Dict):
        content = "\n".join([
            data["technical_summary"]["sections"]["introduction"],
            data["technical_summary"]["sections"]["methodology"],
            data["technical_summary"]["sections"]["results"]
        ])
        
        # Chunk text for BGE model (max 512 tokens)
        text_chunks = self._chunk_text(
            text=content,
            max_tokens=500,  # Leave room for special tokens
            )
        documents = []
        metadatas = []
        ids = []
        # Store each chunk with metadata
        for idx, chunk in enumerate(text_chunks):
            documents.append(chunk)
            metadatas.append({
                "paper_id": paper_id,
                "year": data["basic_info"]["published_year"],
                "themes": ",".join(data["metadata"]["key_themes"]),
                "methods": ",".join(data["metadata"]["methodology"]),
                "chunk": idx
            })
            ids.append(f"{paper_id}_text_{idx}")
        self.vector_db.add(documents=documents, metadatas=metadatas, ids=ids)


    def _store_images(self, paper_data: Dict):
        paper_id = paper_data["basic_info"]["paper_id"]
        cnt_img = 0
        for img in paper_data["technical_summary"]["figures"]:
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
                "themes": ",".join(paper_data["metadata"]["key_themes"]),
                "methods": ",".join(paper_data["metadata"]["methodology"])
            })
            ids.append(img_id)
            # self.image_db.add(
            #     documents=[description],
            #     metadatas=[{
            #         "paper_id": paper_id,
            #         "path": img["path"],
            #         "themes": ",".join(paper_data["metadata"]["key_themes"]),
            #         "methods": ",".join(paper_data["metadata"]["methodology"])
            #     }],
            #     ids=[img_id]
            # )
            cnt_img += 1
            self.image_db.add(documents=documents, metadatas=metadatas, ids=ids)

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

    def _store_in_neo4j(self, paper_id: str, data: Dict):
        with self.graph_db.session() as session:
            session.execute_write(
                self._create_paper_node,
                paper_id,
                data["basic_info"]
            )
            for theme in data["metadata"]["key_themes"]:
                session.execute_write(
                    self._create_theme_relationship,
                    paper_id,
                    theme
                )
            for method in data["metadata"]["methodology"]:
                session.execute_write(
                    self._create_methodology_relationship,
                    paper_id,
                    method
                )
            
            for strength in data["metadata"]["strengths"]:
                session.execute_write(
                    self._create_strength_relationship,
                    paper_id,
                    strength
                )
            for limitations in data["metadata"]["limitations"]:
                session.execute_write(
                    self._create_methodology_relationship,
                    paper_id,
                    limitations
                )
            for domain in data["metadata"]["domain"]:
                session.execute_write(
                    self._create_domain_relationship,
                    paper_id,
                    domain
                )
            for reference in data["basic_info"]["references"]:
                session.execute_write(
                    self._create_reference_relationship,
                    paper_id,
                    reference
                )

    @staticmethod
    def _create_paper_node(tx, paper_id: str, basic_info: Dict):
        tx.run("""
            MERGE (p:Paper {id: $paper_id})
            SET p.title = $title,
                p.authors = $authors,
                p.year = $year
        """, {
            "paper_id": paper_id,
            "title": basic_info["title"],
            "authors": basic_info["authors"],
            "year": basic_info["published_year"],
        })

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
        """Enhanced hybrid search leveraging Chroma collections and Neo4j relationships"""
        # Text search with chunk aggregation
        text_results = self.vector_db.query(
            query_texts=[query],
            n_results=top_k*3,  # Get extra chunks for aggregation
            include=["metadatas", "documents", "distances"]
        )
        
        # Aggregate text chunks by paper
        paper_chunks = {}
        for doc, meta, dist in zip(text_results["documents"][0],
                                text_results["metadatas"][0],
                                text_results["distances"][0]):
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
                    "references": set()
                }
            
            paper_chunks[paper_id]["chunks"].append({
                "text": doc,
                "distance": dist,
                "chunk_idx": meta["chunk"]
            })
            paper_chunks[paper_id]["min_distance"] = min(
                paper_chunks[paper_id]["min_distance"], 
                dist
            )
            if "themes" in meta:
                paper_chunks[paper_id]["themes"].update(meta["themes"].split(','))
            if "methods" in meta:
                paper_chunks[paper_id]["methods"].update(meta["methods"].split(','))
            if "domain" in meta:
                paper_chunks[paper_id]["domains"].update(meta["domain"].split(','))
            if "strengths" in meta:
                paper_chunks[paper_id]["strengths"].update(meta["strengths"].split(','))
            if "limitations" in meta:
                paper_chunks[paper_id]["limitations"].update(meta["limitations"].split(','))
            if "references" in meta:
                paper_chunks[paper_id]["basic_info"]["references"].update(meta["references"].split(','))
        
        # Sort papers by minimum chunk distance and take top_k
        sorted_papers = sorted(
            paper_chunks.items(),
            key=lambda x: x[1]["min_distance"]
        )[:top_k]
        
        # Enrich with graph data and process chunks
        papers = []
        for paper_id, data in sorted_papers:
            try:
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
                combined_text = "\n\n".join([c["text"] for c in sorted_chunks[:3]])  # Top 3 chunks
                
                papers.append({
                    "id": paper_id,
                    "content": combined_text,
                    "themes": list(data["themes"]),
                    "methods": list(data["methods"]),
                    "domains": list(data["domains"]),
                    "strengths": list(data["strengths"]),
                    "limitations": list(data["limitations"]),
                    "references": list(data["references"]),
                    "graph_data": graph_data,
                    "chunk_count": len(data["chunks"]),
                    "min_distance": data["min_distance"]
                })
            except Exception as e:
                print(f"Error processing paper {paper_id}: {str(e)}")
                continue
        
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
                        any(t in found_themes for t in meta["themes"].split(',')) or
                        any(m in found_methods for m in meta["methods"].split(',')) or
                        any(d in found_domains for d in meta.get("domain", "").split(','))):
                        
                        images.append({
                            "path": meta["path"],
                            "paper_id": meta["paper_id"],
                            "description": doc,
                            "themes": meta["themes"].split(','),
                            "methods": meta["methods"].split(','),
                            "domains": meta.get("domain", "").split(',') if meta.get("domain") else [],
                            "distance": dist
                        })
                    if len(images) >= top_k:
                        break
            except Exception as e:
                print(f"Error processing images: {str(e)}")
        
        return {
            "papers": papers,
            "images": images
        }

    @staticmethod
    def _get_enriched_paper_details(tx, paper_id: str):
        """Get extended paper relationships from Neo4j including all metadata."""
        result = tx.run("""
            MATCH (p:Paper{id: $paper_id})
            OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)

            OPTIONAL MATCH (p)-[:DISCUSSES_THEME]->(:Theme)<-[:DISCUSSES_THEME]-(theme_related:Paper)
            WHERE theme_related <> p

            OPTIONAL MATCH (p)-[:USES_METHOD]->(:Methodology)<-[:USES_METHOD]-(method_related:Paper)
            WHERE method_related <> p

            OPTIONAL MATCH (p)-[:BELONGS_TO_DOMAIN]->(:Domain)<-[:BELONGS_TO_DOMAIN]-(domain_related:Paper)
            WHERE domain_related <> p

            RETURN 
                p.id AS paper_id,
                p.title AS title,
                p.authors AS authors,
                COLLECT(DISTINCT cited.id) AS citations,
                COLLECT(DISTINCT theme_related.id) AS theme_related_papers,
                COLLECT(DISTINCT method_related.id) AS method_related_papers,
                COLLECT(DISTINCT domain_related.id) AS domain_related_papers
        """, {"paper_id": paper_id})
        
        record = result.single()
        if record is None:
            return {"paper_id": paper_id, "found": False}
        return dict(record)

    def is_db_populated(self) -> bool:
        paper_count = self.vector_db.count()
        image_count = self.image_db.count()
        return paper_count > 0 and image_count > 0

def ingest_json_directory(kb: ResearchKnowledgeBase, json_dir: str = "papers_summary"):
    json_files = list(Path(json_dir).glob("*.json"))
    successful = 0
    failed = 0
    
    for json_path in tqdm(json_files, desc="Ingesting papers"):
        try:
            print(f"\nProcessing {json_path.name}")
            with open(json_path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
            
            kb.ingest_paper(paper_data)
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
            
        # Rest of the function is unchanged
        # ...existing code...
        method_clusters = {}
        for i in range(kmeans.n_clusters):
            methods = []
            for idx, meta in enumerate(papers["metadatas"]):
                if cluster_labels[idx] == i and "methods" in meta:
                    methods.extend(meta["methods"].split(','))
            unique_methods = list(set(methods))
            method_clusters[f"cluster_{i}"] = unique_methods
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
            num_topics=num_topics
        )
    except Exception as e:
        print(f"Error generating topics: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

def _synthesize_topics(emerging_themes, method_clusters, interdisciplinary, num_topics=5) -> List[Dict]:
    themes_text = "\n".join([f"- {item['theme']} ({item['paper_count']} papers)" for item in emerging_themes]) if emerging_themes else "No emerging themes found"
    methods_text = "\n".join([f"Cluster {cid}: {', '.join(methods)}" for cid, methods in method_clusters.items()]) if method_clusters else "No method clusters found"
    interdisciplinary_text = "\n".join([f"- {item['combined_theme']} ({item['connection_strength']} connections)" for item in interdisciplinary]) if interdisciplinary else "No interdisciplinary connections found"
    
    prompt = f"""
    Generate {num_topics} research review paper topics based on:

    Emerging Themes:
    {themes_text}

    Methodology Clusters:
    {methods_text}

    Interdisciplinary Connections:
    {interdisciplinary_text}

    Guidelines:
    - Combine 2-3 concepts from different areas
    - Focus on recent developments (post-2020)
    - Format: "Advances in [TECH] for [DOMAIN]: [SPECIFIC FOCUS]"
    - Include both theoretical and applied aspects

    Return a JSON list with: "title", "focus", "themes", "methods"
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
    # topics = get_review_topics(kb)
    # # print(json.dumps(topics, indent=2))
    # with open("review_topics.json", "w", encoding="utf-8") as f:
    #     json.dump(topics, f, indent=2)
    search_result = kb.hybrid_search("Advances in Attention Mechanisms for Artificial Intelligence: Enhancing Deep Learning Efficiency", include_images = True)
    with open("search_result.json", "w", encoding="utf-8") as f:
        json.dump(search_result, f, indent=2)