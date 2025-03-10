import json
from knowledge_base import ResearchKnowledgeBase, ingest_json_directory, get_review_topics
from review_writer import ResearchPaperOrchestrator, ChatOpenAI
import os

def main():
    # Initialize knowledge base
    kb = ResearchKnowledgeBase()
    
    if not kb.is_db_populated():
        print("Ingesting papers...")
        ingest_json_directory(kb)
    
    print("\nLoading review topics...")
    try:
        with open("review_topics.json", "r") as f:
            topics = json.load(f)
    except FileNotFoundError:
        print("Review topics file not found. Generating new topics...")
        topics = get_review_topics(kb)
        with open("review_topics.json", "w") as f:
            json.dump(topics, f, indent=2)
    
    # Select first topic
    selected_topic = topics[0]
    print(f"\nSelected Topic: {selected_topic['title']}")
    
    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    
    # Get relevant sources from knowledge base
    search_results = kb.hybrid_search(
        query=selected_topic['focus'],
        top_k=10,
        include_images=True
    )

    # Process sources for orchestrator
    sources = [
        {
            "title": paper["graph_data"]["title"],
            "authors": paper["graph_data"].get("authors", ["Unknown"]),
            "year": paper["graph_data"]["year"],
            "content": paper["content"]
        }
        for paper in search_results["papers"]
    ]
    
    # Print debug information
    print("\nFound papers:")
    for paper in search_results["papers"]:
        print(f"- {paper['graph_data']['title']} ({paper['graph_data']['year']})")
    
    print("\nFound images:")
    for image in search_results["images"]:
        print(f"- {image['path']} from {image['paper_id']}")

    # Initialize and run orchestrator
    orchestrator = ResearchPaperOrchestrator(
        topic=selected_topic["title"],
        sources=sources,  # Use processed sources
        llm=llm
    )
    orchestrator.topic_details = selected_topic
    if search_results["papers"]:
        orchestrator.graph_context = search_results["papers"][0]["graph_data"]
    
    try:
        final_paper = orchestrator.run_pipeline()
        os.makedirs("output", exist_ok=True)
        with open("output/review_paper.tex", "w") as f:
            f.write(final_paper)
        print("\nPaper generated successfully!")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()