import os
import json
import arxiv
import time
import argparse
from pathlib import Path


def update_json_with_arxiv_data(json_path):
    """
    Update a JSON file with arXiv metadata based on the filename (arXiv ID)
    """
    try:
        # Extract arXiv ID from filename
        arxiv_id = Path(json_path).stem
        
        print(f"Processing {arxiv_id}...")
        
        # Load existing JSON data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Check if data already has required fields
        if "basic_info" in data and "paper_id" in data["basic_info"] and "published_year" in data["basic_info"]:
            print(f"File {json_path} already has required metadata")
            return
        
        # Initialize arXiv client
        client = arxiv.Client()
        
        # Search for the paper
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        
        if not results:
            print(f"No results found for arXiv ID: {arxiv_id}")
            return
        
        paper = results[0]
        
        # Ensure basic_info exists
        if "basic_info" not in data:
            data["basic_info"] = {}
        
        # Update with paper_id and published_year
        data["basic_info"]["paper_id"] = arxiv_id
        data["basic_info"]["published_year"] = paper.published.year
        
        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        print(f"Updated {json_path} with arXiv metadata")
        
        # Respect arXiv API rate limits
        time.sleep(1)
        
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Update JSON files with arXiv metadata")
    parser.add_argument("--dir", default="./papers_summary", help="Directory containing JSON files")
    parser.add_argument("--file", help="Process a specific JSON file")
    args = parser.parse_args()
    
    if args.file:
        # Process a single file
        update_json_with_arxiv_data(args.file)
    else:
        # Process all JSON files in directory
        directory = args.dir
        os.makedirs(directory, exist_ok=True)
        
        json_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                     if f.endswith('.json')]
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            update_json_with_arxiv_data(json_file)


if __name__ == "__main__":
    main()