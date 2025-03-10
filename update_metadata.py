import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to load JSON files from a directory
def load_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_path = os.path.join(directory, filename)
                yield json_path, data

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# LLM query for canonical terms
def get_canonical_term(terms):
    prompt = f"Provide one concise canonical term for the following similar concepts:\n{', '.join(terms)}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Function to cluster and generate canonical terms
def cluster_and_canonicalize(terms, distance_threshold=1.0):
    embeddings = model.encode(terms)
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        linkage='average'
    )
    clustering_labels = clustering.fit_predict(embeddings)

    clusters = {}
    for term, label in zip(terms, clustering.labels_):
        clusters.setdefault(label, []).append(term)

    canonical_mapping = {}
    for cluster_terms in clusters.values():
        canonical_term = get_canonical_term(cluster_terms)
        for term in cluster_terms:
            canonical_mapping[term] = canonical_term

    return canonical_mapping

# Directory containing JSON files
directory = './papers_summary'

# Collect all themes and methodologies
all_themes, all_methods, all_strengths, all_limitations, all_domain = set(), set(), set(), set(), set()
for json_path, data in load_json_files(directory):
    metadata = data['metadata']
    all_themes.update(metadata.get('key_themes', []))
    all_methods.update(metadata.get('methodology', []))
    all_strengths.update(metadata.get('strengths', []))
    all_limitations.update(metadata.get('limitations', []))
    all_domain.update(metadata.get('domain', []))


# Generate canonical mappings
canonical_themes = cluster_and_canonicalize(list(all_themes), distance_threshold=1.2)
canonical_methods = cluster_and_canonicalize(list(all_methods), distance_threshold=1.2)
canonical_strengths = cluster_and_canonicalize(list(all_strengths), distance_threshold=1.2)
canonical_limitations = cluster_and_canonicalize(list(all_limitations), distance_threshold=1.2)
canonical_domain = cluster_and_canonicalize(list(all_domain), distance_threshold=1.2)



# Update JSON files
for json_path, data in load_json_files(directory):
    metadata = data['metadata']
    metadata['key_themes'] = list(set(canonical_themes.get(term, term) for term in metadata.get('key_themes', [])))
    metadata['methodology'] = list(set(canonical_methods.get(term, term) for term in metadata.get('methodology', [])))
    metadata['strengths'] = list(set(canonical_strengths.get(term, term) for term in metadata.get('strengths', [])))
    metadata['limitations'] = list(set(canonical_limitations.get(term, term) for term in metadata.get('limitations', [])))
    metadata['domain'] = list(set(canonical_domain.get(term, term) for term in metadata.get('domain', [])))

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"Updated {json_path}")

print("All JSON files updated successfully.")