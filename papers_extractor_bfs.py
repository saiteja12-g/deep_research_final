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

from processing_pipeline import process_research_paper

load_dotenv()
class ArxivReferenceExplorer:
    def __init__(self, query, initial_results=5, similarity_threshold=0.7,
                 llm_model="gpt-4-1106-preview", max_tokens=3000):
        # ... previous init code ...
        self.query = query
        self.similarity_threshold = similarity_threshold
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.client = OpenAI()
        self.arxiv_client = arxiv.Client()
        self.visited = set()  # Stores paper IDs instead of objects
        self.queue = deque()  # Stores tuples of (paper_id, depth)
        self.paper_cache = {}  # Maps paper IDs to full paper data

        search = arxiv.Search(
            query=query,
            max_results=initial_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for result in self.arxiv_client.results(search):
            self._add_paper(result, depth=0)

    def _add_paper(self, paper, depth):
        if paper.title not in self.visited:
            # formatted_id = paper.entry_id[:-2]
            self.visited.add(paper.title)
            self.queue.append((paper.entry_id, depth))
            self.paper_cache[paper.entry_id] = {
                'title': paper.title,
                'authors': [a.name for a in paper.authors],
                'abstract': paper.summary,
                'pdf_url': paper.pdf_url,
                'depth': depth,
                'score': self._calculate_similarity(paper.summary)
            }
            pdf_response = requests.get(paper.pdf_url)
            temp = paper.entry_id.split("arxiv.org/abs/")[-1]
            with open(f'./papers/{temp}.pdf', 'wb') as f:
                f.write(pdf_response.content)
            
            os.makedirs("./papers_summary", exist_ok=True)

            # Write metadata to JSON file
            paper_metadata = {
                "basic_info": {
                    "title": self.paper_cache[paper.entry_id]['title'],
                    "authors": self.paper_cache[paper.entry_id]['authors'],
                    "paper_id": temp,
                    "published_year": paper.published.year,
                    "references": []
                }
            }
            # Save to JSON file using paper_id as filename
            with open(f'./papers_summary/{temp}.json', 'w', encoding='utf-8') as json_file:
                json.dump(paper_metadata, json_file, indent=4)

    def _calculate_similarity(self, text):
        vectors = self.vectorizer.fit_transform([self.query, text])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def _extract_reference_titles(self, pdf_content):
        # Find references section using multiple possible headers
        references_start = re.search(
            r'(References|Bibliography|Cited Works|Reference List|Literature Cited)',
            pdf_content, re.IGNORECASE
        )
        
        if not references_start:
            return []
            
        references_text = pdf_content[references_start.start():]
        
        # Truncate to avoid context overflow and non-reference content
        truncated_text = references_text[:5000]  # Adjust based on model context window
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                temperature=0.1,  # Reduce creativity
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
                    
            return titles[:50]  # Limit to first 50 titles
            
        except Exception as e:
            print(f"LLM extraction failed: {str(e)}")
            return []

    def _search_arxiv_by_title(self, title):
        try:
            clean_title = re.sub(r'[^a-zA-Z0-9 ]', '', title)[:3000]
            search = arxiv.Search(
                query=f"ti:\"{clean_title}\"",
                max_results=3
            )
            
            for result in self.arxiv_client.results(search):
                if result.title.lower() in title.lower() or title.lower() in result.title.lower():
                    return result
            return None
        except Exception as e:
            print(f"arXiv search error: {str(e)}")
            return None

    def bfs_traversal(self, max_depth=3, max_per_level=[5, 3]):
        with tqdm(total=sum(max_per_level[:max_depth]), desc="Collecting papers") as pbar:
            while self.queue:
                paper_id, depth = self.queue.popleft()  # Get paper ID from queue
                current_paper = self.paper_cache[paper_id]  # Get actual paper data
                # processing current pdf
                curr_paper_id = paper_id.split("arxiv.org/abs/")[-1]
                pdf_path = r'./papers/' + curr_paper_id + '.pdf'
                json_path = r'./papers_summary/' + curr_paper_id + '.json'
                result = process_research_paper(pdf_path, json_path)
                print(json.dumps(result, indent=2))
                # Update JSON file with processing results
                try:
                    with open(json_path, 'r', encoding='utf-8') as json_file:
                        existing_data = json.load(json_file)
                    
                except Exception as e:
                    print(f"Error opening JSON file {json_path}: {str(e)}")

                if depth >= max_depth:
                    continue

                try:
                    pdf_response = requests.get(current_paper['pdf_url'])

                    # Extracting children based on references
                    with fitz.open(stream=pdf_response.content, filetype="pdf") as doc:
                        pdf_text = ""
                        for page in doc:
                            pdf_text += page.get_text()
                    
                    reference_titles = self._extract_reference_titles(pdf_text)
                    candidates = set()
                    
                    for title in reference_titles:
                        found_paper = self._search_arxiv_by_title(title)
                        if found_paper and found_paper.title not in self.visited:
                            candidates.add((found_paper.entry_id, found_paper.title, found_paper.published))
                            # Cache the paper data immediately
                            self.paper_cache[found_paper.entry_id] = {
                                'title': found_paper.title,
                                'authors': [a.name for a in found_paper.authors],
                                'abstract': found_paper.summary,
                                'pdf_url': found_paper.pdf_url,
                                'score': self._calculate_similarity(found_paper.summary)
                            }
                        time.sleep(1.5)  # arXiv rate limit
                    
                    # Score using cached data
                    scored = []
                    for candidate_id, candidate_title, candidate_published in candidates:
                        candidate_data = self.paper_cache.get(candidate_id)
                        if candidate_data:
                            scored.append((candidate_data['score'], candidate_id, candidate_title, candidate_published))
                    
                    # Sort and select top papers
                    scored.sort(reverse=True, key=lambda x: x[0])
                    for score, c_id, c_title, c_published in scored[:max_per_level[depth]]:
                        if c_title not in self.visited:
                            temp = c_id.split("arxiv.org/abs/")[-1]
                            formatted_c_id = c_id[:-2]
                            candidate_pdf_response = requests.get(self.paper_cache[c_id]['pdf_url'])
                            with open(f'./papers/{temp}.pdf', 'wb') as f:
                                f.write(candidate_pdf_response.content)
                            self.visited.add(c_title)
                            self.queue.append((c_id, depth+1))
                            pbar.update(1)

                            # Ensure the directory exists
                            os.makedirs("./papers_summary", exist_ok=True)

                            # Write metadata to JSON file
                            paper_metadata = {"basic_info": {
                                    "title": self.paper_cache[c_id]['title'],
                                    "authors": self.paper_cache[c_id]['authors'],
                                    "paper_id": temp,
                                    "published_year": c_published.year,
                                    "references": []
                                }}
                            existing_data["basic_info"]["references"].append(temp)
                            # Save to JSON file using paper_id as filename
                            with open(f'./papers_summary/{temp}.json', 'w', encoding='utf-8') as json_file:
                                json.dump(paper_metadata, json_file, indent=4)
                    
                        with open(json_path, 'w', encoding='utf-8') as json_file:
                            json.dump(existing_data, json_file, indent=4)
                        
                except Exception as e:
                    print(f"Error processing {current_paper.get('title', 'Unknown paper')}: {str(e)}")

        return list(self.paper_cache.values())

# Usage example with error handling
try:
    explorer = ArxivReferenceExplorer(
        query="Generative AI Advancements",
        initial_results=5,
        similarity_threshold=0.65
    )
    papers = explorer.bfs_traversal(max_depth=2, max_per_level=[5, 3, 2])

    print(f"\nTotal papers collected: {len(explorer.paper_cache)}")
    print("Top papers:")
    for paper in sorted(explorer.paper_cache.values(), 
                    key=lambda x: x['score'], 
                    reverse=True)[:5]:
        print(f"- {paper['title']} (Score: {paper['score']:.2f}, Depth: {paper.get('depth', 0)})")
        
except Exception as e:
    print(f"Fatal error: {str(e)}")