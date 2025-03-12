import arxiv
from dotenv import load_dotenv
import pymupdf as fitz 
from openai import OpenAI
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from tqdm import tqdm
import time
import requests
import json
import os
import traceback
from citation_mapper import CitationMapper, process_and_map_citations


from processing_pipeline import process_research_paper

# Load environment variables
load_dotenv()

class ArxivReferenceExplorer:
    def __init__(self, query, initial_results=5, similarity_threshold=0.7,
                 llm_model="gpt-4o", max_tokens=3000):
        """
        Initialize the explorer with the query and settings
        
        Args:
            query: Search query for arXiv
            initial_results: Number of initial papers to retrieve
            similarity_threshold: Threshold for considering papers as similar
            llm_model: LLM model to use for reference extraction
            max_tokens: Maximum tokens for LLM input
        """
        # Initialize parameters
        self.query = query
        self.similarity_threshold = similarity_threshold
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.client = OpenAI()
        self.arxiv_client = arxiv.Client()
        self.visited = set()  # Stores paper titles to avoid duplicates
        self.queue = deque()  # Stores tuples of (paper_id, depth)
        self.paper_cache = {}  # Maps paper IDs to full paper data
        
        # Create necessary directories
        os.makedirs("./papers", exist_ok=True)
        os.makedirs("./papers_summary", exist_ok=True)

        # Perform initial arXiv search
        print(f"Performing initial search for: {query}")
        search = arxiv.Search(
            query=query,
            max_results=initial_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Add initial papers to the queue
        for result in self.arxiv_client.results(search):
            self._add_paper(result, depth=0)
            
        print(f"Added {len(self.queue)} initial papers to the queue")

    def _add_paper(self, paper, depth):
        """
        Add a paper to the processing queue and initialize its metadata
        
        Args:
            paper: arXiv paper object
            depth: Depth in the citation graph
        """
        if paper.title not in self.visited:
            print(f"Adding paper: {paper.title}")
            self.visited.add(paper.title)
            self.queue.append((paper.entry_id, depth))
            
            # Extract paper ID from arXiv URL
            paper_id = paper.entry_id.split("arxiv.org/abs/")[-1]
            
            # Cache paper metadata
            self.paper_cache[paper.entry_id] = {
                'title': paper.title,
                'authors': [a.name for a in paper.authors],
                'abstract': paper.summary,
                'pdf_url': paper.pdf_url,
                'depth': depth,
                'score': self._calculate_similarity(paper.summary)
            }
            
            # Download PDF file
            try:
                pdf_response = requests.get(paper.pdf_url)
                pdf_path = f'./papers/{paper_id}.pdf'
                
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_response.content)
                    
                print(f"Downloaded PDF to {pdf_path}")
            except Exception as e:
                print(f"Error downloading PDF: {str(e)}")
            
            # Initialize JSON metadata file
            try:
                paper_metadata = {
                    "basic_info": {
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "paper_id": paper_id,
                        "published_year": paper.published.year,
                        "references": []
                    }
                }
                
                json_path = f'./papers_summary/{paper_id}.json'
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(paper_metadata, json_file, indent=4)
                    
                print(f"Created initial metadata at {json_path}")
            except Exception as e:
                print(f"Error creating metadata file: {str(e)}")

    def _calculate_similarity(self, text):
        """
        Calculate cosine similarity between query and text
        
        Args:
            text: Text to compare against the query
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            vectors = self.vectorizer.fit_transform([self.query, text])
            return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def _extract_reference_titles(self, pdf_content):
        """
        Extract paper titles from references section using LLM
        
        Args:
            pdf_content: Text content of the PDF
            
        Returns:
            List of extracted paper titles
        """
        # Find references section using multiple possible headers
        references_start = re.search(
            r'(References|Bibliography|Cited Works|Reference List|Literature Cited)',
            pdf_content, re.IGNORECASE
        )
        
        if not references_start:
            print("No references section found")
            return []
            
        references_text = pdf_content[references_start.start():]
        
        # Truncate to avoid context overflow
        truncated_text = references_text[:5000]
        
        try:
            print("Extracting references using LLM...")
            response = self.client.chat.completions.create(
                model=self.llm_model,
                temperature=0.1,
                messages=[{
                    "role": "system",
                    "content": """You are a research paper reference extractor. Follow these rules:
                    1. Extract ONLY formal academic paper titles from references
                    2. Exclude books, websites, patents, technical reports, and non-scholarly items
                    3. Return titles in their original capitalization exactly as written
                    4. Format as a plain list with one title per line
                    5. No numbering, bullets, quotes, or markdown
                    6. If no papers found, return empty string

                    Example input:
                    References
                    1. Smith, J. (2020). "Deep Learning Approaches". Journal of AI.
                    2. Lee, H. et al. (2022) arXiv preprint arXiv:2201.12345

                    Example output:
                    Deep Learning Approaches
                    Attention Is All You Need"""
                }, {
                    "role": "user",
                    "content": f"Extract paper titles from:\n{truncated_text}"
                }]
            )

            # Process response
            raw_output = response.choices[0].message.content.strip()
            if not raw_output:
                print("No references extracted")
                return []
                
            # Clean and validate titles
            titles = []
            for line in raw_output.split('\n'):
                line = line.strip()
                # Filter out non-title lines and page numbers
                if line and len(line) > 10 and not line.isdigit():
                    # Remove trailing publication years (e.g., " (2023)")
                    clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', line)
                    titles.append(clean_title)
            
            print(f"Extracted {len(titles)} reference titles")        
            return titles[:50]  # Limit to first 50 titles
            
        except Exception as e:
            print(f"LLM extraction failed: {str(e)}")
            return []
    

    def _process_citations(self, paper_id, pdf_path, json_path):
        """Process citations for a paper and map them to references"""
        print(f"Processing citations for paper: {paper_id}")
        result = process_and_map_citations(paper_id, pdf_path, json_path)
        if result["status"] == "success":
            print(f"✓ Successfully processed citations: {result.get('references_added', 0)} references added")
        else:
            print(f"✗ Citation processing failed: {result.get('message', 'Unknown error')}")
        return result

    def _search_arxiv_by_title(self, title):
        """
        Search arXiv for a paper by title
        
        Args:
            title: Paper title to search for
            
        Returns:
            arXiv paper object or None if not found
        """
        try:
            # Clean title for search
            clean_title = re.sub(r'[^a-zA-Z0-9 ]', '', title)[:3000]
            print(f"Searching arXiv for: {clean_title}")
            
            search = arxiv.Search(
                query=f"ti:\"{clean_title}\"",
                max_results=3
            )
            
            # Try to find a matching paper
            for result in self.arxiv_client.results(search):
                title_lower = title.lower()
                result_lower = result.title.lower()
                
                if title_lower in result_lower or result_lower in title_lower:
                    print(f"Found matching paper: {result.title}")
                    return result
                    
            print(f"No match found for: {title}")
            return None
        except Exception as e:
            print(f"arXiv search error: {str(e)}")
            return None

    def bfs_traversal(self, max_depth=2, max_per_level=[5, 3], progress_callback=None):
        """
        Traverse the paper citation network using breadth-first search
        
        Args:
            max_depth: Maximum depth to traverse
            max_per_level: List of maximum papers to process at each depth level
            
        Returns:
            List of collected papers
        """
        # Calculate total expected papers for progress bar
        expected_total = sum(max_per_level[:max_depth]) if len(max_per_level) >= max_depth else sum(max_per_level) + max_per_level[-1] * (max_depth - len(max_per_level))
        
        with tqdm(total=expected_total, desc="Collecting papers") as pbar:
            papers_processed = 0
            
            while self.queue:
                # Get next paper from queue
                paper_id, depth = self.queue.popleft()
                current_paper = self.paper_cache[paper_id]
                
                # Extract paper ID from arXiv URL
                curr_paper_id = paper_id.split("arxiv.org/abs/")[-1]
                pdf_path = f'./papers/{curr_paper_id}.pdf'
                json_path = f'./papers_summary/{curr_paper_id}.json'
                citation_result = self._process_citations(curr_paper_id, pdf_path, json_path)
                print(f"  - Citations: {citation_result.get('references_added', 0)}")
                print(f"\n{'='*60}")
                print(f"Processing paper: {current_paper['title']} (Depth {depth})")
                print(f"{'='*60}")
                
                # Process current PDF and update JSON metadata
                try:
                    # Process the paper to extract chunks, metadata, etc.
                    result = process_research_paper(pdf_path, json_path)
                    
                    if result["status"] == "success":
                        print(f"✓ Successfully processed paper: {curr_paper_id}")
                        print(f"  - Raw chunks: {result.get('raw_chunks_count', 0)}")
                        print(f"  - Figures: {result.get('figures_count', 0)}")
                        
                        # Load existing data to update references
                        with open(json_path, 'r', encoding='utf-8') as json_file:
                            existing_data = json.load(json_file)
                    else:
                        print(f"✗ Processing failed: {result.get('message', 'Unknown error')}")
                        continue
                    
                    # Update progress bar
                    papers_processed += 1
                    pbar.update(1)
                    
                    # Stop traversing this branch if we've reached max depth
                    if depth >= max_depth:
                        print(f"Reached maximum depth ({max_depth}), not traversing further")
                        continue
                        
                    # Get max papers for current depth
                    current_max = max_per_level[depth] if depth < len(max_per_level) else max_per_level[-1]
                    print(f"Searching for up to {current_max} cited papers")
                    
                    # Extract references from the PDF
                    with fitz.open(pdf_path) as doc:
                        pdf_text = ""
                        for page in doc:
                            pdf_text += page.get_text()
                    
                    # Extract reference titles using LLM
                    reference_titles = self._extract_reference_titles(pdf_text)
                    candidates = set()
                    
                    # Search for each reference on arXiv
                    for title in reference_titles:
                        print(f"Looking up reference: {title[:50]}...")
                        found_paper = self._search_arxiv_by_title(title)
                        
                        if found_paper and found_paper.title not in self.visited:
                            print(f"Adding candidate: {found_paper.title}")
                            candidates.add((found_paper.entry_id, found_paper.title, found_paper.published))
                            
                            # Cache the paper data
                            self.paper_cache[found_paper.entry_id] = {
                                'title': found_paper.title,
                                'authors': [a.name for a in found_paper.authors],
                                'abstract': found_paper.summary,
                                'pdf_url': found_paper.pdf_url,
                                'score': self._calculate_similarity(found_paper.summary)
                            }
                        # Sleep to respect arXiv rate limits
                        time.sleep(1.5)
                    
                    print(f"Found {len(candidates)} candidate papers to process")
                    
                    # Score and select top candidates
                    scored = []
                    for candidate_id, candidate_title, candidate_published in candidates:
                        candidate_data = self.paper_cache.get(candidate_id)
                        if candidate_data:
                            scored.append((
                                candidate_data['score'], 
                                candidate_id, 
                                candidate_title, 
                                candidate_published
                            ))
                    
                    # Sort by relevance score and take top papers for this level
                    scored.sort(reverse=True, key=lambda x: x[0])
                    selected_candidates = scored[:current_max]
                    
                    print(f"Selected {len(selected_candidates)} papers to process at depth {depth+1}")
                    
                    # Process each selected paper
                    for score, c_id, c_title, c_published in selected_candidates:
                        if c_title not in self.visited:
                            print(f"\nProcessing cited paper: {c_title}")
                            print(f"Relevance score: {score:.2f}")
                            
                            # Get paper ID from arXiv URL
                            child_paper_id = c_id.split("arxiv.org/abs/")[-1]
                            
                            # Download PDF
                            try:
                                candidate_pdf_response = requests.get(self.paper_cache[c_id]['pdf_url'])
                                child_pdf_path = f'./papers/{child_paper_id}.pdf'
                                
                                with open(child_pdf_path, 'wb') as f:
                                    f.write(candidate_pdf_response.content)
                                    
                                print(f"Downloaded PDF to {child_pdf_path}")
                            except Exception as e:
                                print(f"Error downloading PDF: {str(e)}")
                                continue
                            
                            # Mark as visited and add to queue
                            self.visited.add(c_title)
                            self.queue.append((c_id, depth+1))
                            
                            # Create initial JSON metadata
                            child_json_path = f'./papers_summary/{child_paper_id}.json'
                            paper_metadata = {
                                "basic_info": {
                                    "title": self.paper_cache[c_id]['title'],
                                    "authors": self.paper_cache[c_id]['authors'],
                                    "paper_id": child_paper_id,
                                    "published_year": c_published.year,
                                    "references": []
                                }
                            }
                            
                            # Add to parent's references
                            if "basic_info" in existing_data and "references" in existing_data["basic_info"]:
                                if child_paper_id not in existing_data["basic_info"]["references"]:
                                    existing_data["basic_info"]["references"].append(child_paper_id)
                            
                            # Save child paper metadata
                            with open(child_json_path, 'w', encoding='utf-8') as json_file:
                                json.dump(paper_metadata, json_file, indent=4)
                                
                            print(f"Created initial metadata at {child_json_path}")
                    
                    # Save updated parent metadata with new references
                    with open(json_path, 'w', encoding='utf-8') as json_file:
                        json.dump(existing_data, json_file, indent=4)
                    
                    print(f"Updated references in {json_path}")
                    if progress_callback:
                        progress_callback(depth, max_depth, self.paper_cache)
                    
                except Exception as e:
                    print(f"\n⚠️ Error processing paper: {str(e)}")
                    traceback.print_exc()
                    continue
            
            print(f"\nCompleted BFS traversal, processed {papers_processed} papers")
            return list(self.paper_cache.values())

if __name__ == "__main__":
    # Usage example with error handling
    try:
        print("=" * 80)
        print("Starting ArXiv Reference Explorer")
        print("=" * 80)
        
        # Initialize explorer with search query
        explorer = ArxivReferenceExplorer(
            query="Single image to 3d",
            initial_results=3,
            similarity_threshold=0.65
        )
        
        print("\n" + "=" * 80)
        print("Beginning BFS traversal of paper citation network")
        print("=" * 80)
        
        # Start traversal
        papers = explorer.bfs_traversal(max_depth=2, max_per_level=[2, 1])
    
        # Display results
        print("\n" + "=" * 80)
        print(f"Total papers collected: {len(explorer.paper_cache)}")
        print("=" * 80)
        
        print("\nTop papers by relevance:")
        for paper in sorted(explorer.paper_cache.values(), 
                        key=lambda x: x['score'], 
                        reverse=True)[:5]:
            print(f"- {paper['title']}")
            print(f"  Score: {paper['score']:.2f}, Depth: {paper.get('depth', 0)}")
            print()
            
    except Exception as e:
        print(f"\n⚠️ Fatal error: {str(e)}")
        traceback.print_exc()