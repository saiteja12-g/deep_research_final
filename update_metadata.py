# import os
# import json
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import AgglomerativeClustering
# import numpy as np
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()
# # Initialize the OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Function to load JSON files from a directory
# def load_json_files(directory):
#     json_files = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.json'):
#             with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 json_path = os.path.join(directory, filename)
#                 yield json_path, data

# # Embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # LLM query for canonical terms
# def get_canonical_term(terms):
#     prompt = f"Provide one concise canonical term for the following similar concepts:\n{', '.join(terms)}"
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2
#     )
#     return response.choices[0].message.content.strip()

# # Function to cluster and generate canonical terms
# def cluster_and_canonicalize(terms, distance_threshold=1.0):
#     embeddings = model.encode(terms)
#     clustering = AgglomerativeClustering(
#         n_clusters=None, 
#         distance_threshold=distance_threshold,
#         linkage='average'
#     )
#     clustering_labels = clustering.fit_predict(embeddings)

#     clusters = {}
#     for term, label in zip(terms, clustering.labels_):
#         clusters.setdefault(label, []).append(term)

#     canonical_mapping = {}
#     for cluster_terms in clusters.values():
#         canonical_term = get_canonical_term(cluster_terms)
#         for term in cluster_terms:
#             canonical_mapping[term] = canonical_term

#     return canonical_mapping

# # Directory containing JSON files
# directory = './papers_summary'

# # Collect all themes and methodologies
# all_themes, all_methods, all_strengths, all_limitations, all_domain = set(), set(), set(), set(), set()
# for json_path, data in load_json_files(directory):
#     metadata = data['metadata']
#     all_themes.update(metadata.get('key_themes', []))
#     all_methods.update(metadata.get('methodology', []))
#     all_strengths.update(metadata.get('strengths', []))
#     all_limitations.update(metadata.get('limitations', []))
#     all_domain.update(metadata.get('domain', []))


# # Generate canonical mappings
# canonical_themes = cluster_and_canonicalize(list(all_themes), distance_threshold=1.2)
# canonical_methods = cluster_and_canonicalize(list(all_methods), distance_threshold=1.2)
# canonical_strengths = cluster_and_canonicalize(list(all_strengths), distance_threshold=1.2)
# canonical_limitations = cluster_and_canonicalize(list(all_limitations), distance_threshold=1.2)
# canonical_domain = cluster_and_canonicalize(list(all_domain), distance_threshold=1.2)



# # Update JSON files
# for json_path, data in load_json_files(directory):
#     metadata = data['metadata']
#     metadata['key_themes'] = list(set(canonical_themes.get(term, term) for term in metadata.get('key_themes', [])))
#     metadata['methodology'] = list(set(canonical_methods.get(term, term) for term in metadata.get('methodology', [])))
#     metadata['strengths'] = list(set(canonical_strengths.get(term, term) for term in metadata.get('strengths', [])))
#     metadata['limitations'] = list(set(canonical_limitations.get(term, term) for term in metadata.get('limitations', [])))
#     metadata['domain'] = list(set(canonical_domain.get(term, term) for term in metadata.get('domain', [])))

#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4)

#     print(f"Updated {json_path}")

# print("All JSON files updated successfully.")

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI
from dotenv import load_dotenv
import time
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the embedding model (smaller/faster model)
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded")

# Function to load JSON files from a directory
def load_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_path = os.path.join(directory, filename)
                    yield json_path, data
            except Exception as e:
                print(f"Error loading {filename}: {e}")

# LLM query for canonical terms with rate limiting
def get_canonical_term(terms, retries=3):
    if not terms:
        return ""
    
    # Use the most frequent term directly if less than 3 terms
    if len(terms) <= 2:
        return max(terms, key=len)
    
    # Create a concise description for this cluster
    prompt = f"Provide ONE short canonical term (max 3-4 words) that represents these concepts:\n{', '.join(terms[:10])}"
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=20
            )
            result = response.choices[0].message.content.strip()
            # Remove quotes if present
            result = result.strip('"\'')
            if result:
                return result
        except Exception as e:
            print(f"Error getting canonical term (attempt {attempt+1}): {e}")
            time.sleep(2)  # Rate limiting
    
    # Fallback to most frequent term
    return max(terms, key=len)

# Function to cluster and generate canonical terms
def cluster_and_canonicalize(terms, distance_threshold=1.2):
    if not terms:
        return {}
    
    print(f"Clustering {len(terms)} terms")
    
    # Remove any empty strings and duplicates
    terms = [t for t in terms if t.strip()]
    terms = list(set(terms))
    
    if not terms:
        return {}
    
    # Compute embeddings
    embeddings = model.encode(terms)
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        linkage='average'
    )
    clustering_labels = clustering.fit_predict(embeddings)
    
    # Group terms by cluster
    clusters = {}
    for term, label in zip(terms, clustering_labels):
        clusters.setdefault(label, []).append(term)
    
    print(f"Created {len(clusters)} clusters")
    
    # Generate canonical terms for each cluster
    canonical_mapping = {}
    for i, cluster_terms in enumerate(clusters.values()):
        # Progress update
        if i % 10 == 0:
            print(f"Processing cluster {i}/{len(clusters)}")
        
        canonical_term = get_canonical_term(cluster_terms)
        for term in cluster_terms:
            canonical_mapping[term] = canonical_term
    
    return canonical_mapping

# Filter terms by frequency and importance
def filter_metadata_terms(term_lists, max_terms_per_type=5):
    # Count term frequency across all papers
    term_counters = {
        "key_themes": Counter(),
        "methodology": Counter(),
        "strengths": Counter(),
        "limitations": Counter(), 
        "domain": Counter()
    }
    
    # Count occurrences of each term
    for terms_dict in term_lists:
        for term_type, terms in terms_dict.items():
            if term_type in term_counters:
                term_counters[term_type].update(terms)
    
    # Create a mapping of which terms to keep for each paper
    filtered_lists = []
    for terms_dict in term_lists:
        filtered_dict = {}
        for term_type, terms in terms_dict.items():
            if term_type in term_counters:
                # Keep the most frequent terms for this paper
                kept_terms = []
                # Sort the paper's terms by their global frequency
                sorted_terms = sorted(terms, key=lambda t: term_counters[term_type].get(t, 0), reverse=True)
                kept_terms = sorted_terms[:max_terms_per_type]
                filtered_dict[term_type] = kept_terms
            else:
                filtered_dict[term_type] = terms
        filtered_lists.append(filtered_dict)
    
    return filtered_lists

# Compute paper embeddings for fast similarity calculation
def compute_paper_embeddings(directory, metadata_lists):
    paper_embeddings = {}
    
    print("Computing paper embeddings...")
    for i, (json_path, metadata) in enumerate(zip(directory, metadata_lists)):
        paper_id = os.path.basename(json_path).split('.')[0]
        
        # Create a combined metadata text
        metadata_text = ' '.join([
            ' '.join(metadata.get('key_themes', [])),
            ' '.join(metadata.get('methodology', [])),
            ' '.join(metadata.get('strengths', [])),
            ' '.join(metadata.get('limitations', [])),
            ' '.join(metadata.get('domain', []))
        ])
        
        # Compute embedding
        embedding = model.encode(metadata_text)
        paper_embeddings[paper_id] = embedding.tolist()
        
        # Progress update
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1} paper embeddings")
    
    # Save embeddings
    with open('paper_embeddings.json', 'w') as f:
        json.dump(paper_embeddings, f)
    
    print(f"Saved embeddings for {len(paper_embeddings)} papers")
    return paper_embeddings

# Main function
def update_metadata(directory='./papers_summary'):
    start_time = time.time()
    print(f"Starting metadata optimization...")
    
    # Collect all terms and papers
    all_terms = {
        "key_themes": set(),
        "methodology": set(), 
        "strengths": set(),
        "limitations": set(),
        "domain": set()
    }
    
    # Track all JSON paths and their metadata
    json_paths = []
    metadata_lists = []
    
    print(f"Reading JSON files from {directory}")
    for json_path, data in load_json_files(directory):
        metadata = data.get('metadata', {})
        json_paths.append(json_path)
        metadata_lists.append(metadata)
        
        # Collect all terms
        for term_type in all_terms.keys():
            all_terms[term_type].update(metadata.get(term_type, []))
    
    print(f"Found {len(json_paths)} JSON files")
    for term_type, terms in all_terms.items():
        print(f"Total {term_type}: {len(terms)}")
    
    # Generate canonical mappings
    canonical_mappings = {}
    for term_type, terms in all_terms.items():
        print(f"Canonicalizing {term_type}...")
        canonical_mappings[term_type] = cluster_and_canonicalize(list(terms), distance_threshold=1.2)
    
    # Apply canonical mappings to metadata
    print("Applying canonical mappings...")
    for metadata in metadata_lists:
        for term_type in all_terms.keys():
            if term_type in metadata:
                metadata[term_type] = [
                    canonical_mappings[term_type].get(term, term) 
                    for term in metadata.get(term_type, [])
                ]
                # Remove duplicates
                metadata[term_type] = list(set(metadata[term_type]))
    
    # Filter metadata to keep only important terms
    print("Filtering metadata to reduce term count...")
    filtered_metadata = filter_metadata_terms(metadata_lists, max_terms_per_type=5)
    
    # Compute and save paper embeddings for fast similarity
    compute_paper_embeddings(json_paths, filtered_metadata)
    
    # Update JSON files with optimized metadata
    print("Updating JSON files...")
    for json_path, metadata in zip(json_paths, filtered_metadata):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update metadata
            data['metadata'] = metadata
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
        except Exception as e:
            print(f"Error updating {json_path}: {e}")
    
    end_time = time.time()
    print(f"Metadata optimization completed in {end_time - start_time:.2f} seconds")
    print("All JSON files updated successfully.")

if __name__ == "__main__":
    update_metadata()