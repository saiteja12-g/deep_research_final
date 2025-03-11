#!/usr/bin/env python3
import os
import sys
import argparse
import json
from dotenv import load_dotenv

# Import the updated version of PaperGenerator and KnowledgeBaseConnector
from review_writer import PaperGenerator, KnowledgeBaseConnector

# import sys
# try:
#     from fix_encoding import *
#     print("Applied encoding fix for Windows.")
# except ImportError:
#     print("Warning: Encoding fix not found. Some characters may not display correctly.")

# Optional: Import the actual knowledge base if available
try:
    from knowledge_base import ResearchKnowledgeBase
    has_kb = True
except ImportError:
    has_kb = False

def main():
    """Main entry point for scientific paper generation."""
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
        print("Please set this variable and try again.")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a scientific paper using CrewAI agents")
    parser.add_argument("--query", type=str, help="Initial research query")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--continue", dest="continue_state", action="store_true", help="Continue from saved state")
    args = parser.parse_args()
    
    # Initialize knowledge base connector
    if has_kb:
        kb = ResearchKnowledgeBase()
        kb_connector = KnowledgeBaseConnector(kb)
        print("Using actual Research Knowledge Base")
    else:
        kb_connector = KnowledgeBaseConnector()
        print("Using mock Knowledge Base (sample data)")
    
    # Initialize paper generator
    generator = PaperGenerator(kb_connector)
    
    # If continuing from a saved state
    if args.continue_state:
        if generator.load_state():
            print(f"Loaded state with title: {generator.paper_state['title']}")
            print(f"Current section: {generator.paper_state['current_section_index'] + 1}/{len(generator.paper_state['sections'])}")
            
            # Get completed and remaining sections
            completed = [s for s in generator.paper_state['sections'] if s['status'] == 'APPROVED']
            remaining = [s for s in generator.paper_state['sections'] if s['status'] != 'APPROVED']
            
            print(f"Completed sections: {len(completed)}")
            print(f"Remaining sections: {len(remaining)}")
            
            if args.query:
                print("Warning: Query parameter ignored when continuing from saved state")
        else:
            print("No saved state found. Starting fresh.")
            if not args.query:
                args.query = input("Enter research query: ")
    elif not args.query:
        args.query = input("Enter research query: ")
    
    if not args.query and not args.continue_state:
        print("Error: Research query is required when not continuing from saved state")
        sys.exit(1)
    
    # Run the paper generation pipeline
    if args.continue_state and generator.paper_state["title"]:
        # Continue from current section
        current_index = generator.paper_state["current_section_index"]
        if current_index < len(generator.paper_state["sections"]):
            print(f"\nContinuing with section {current_index + 1}: {generator.paper_state['sections'][current_index]['title']}")
            
            while current_index < len(generator.paper_state["sections"]):
                section = generator.paper_state["sections"][current_index]
                print(f"\n[WRITING] Writing section: {section['title']}...")
                result = generator.write_section(current_index)
                
                if not args.non_interactive:
                    print("\nSection preview:")
                    print("=" * 40)
                    preview = result["content"]
                    if len(preview) > 500:
                        preview = preview[:500] + "..."
                    print(preview)
                    print("=" * 40)
                    
                    approve = input("\nApprove this section? (yes/no): ").lower()
                    if approve != "yes":
                        print("Section not approved. Exiting.")
                        return
                
                generator.approve_section(current_index)
                print(f"[APPROVED] Section approved: {section['title']}")
                current_index = generator.paper_state["current_section_index"]
            
            # Generate LaTeX
            print("\n[DOCUMENT] Generating LaTeX document...")
            latex_result = generator.generate_latex()
            
            if "error" in latex_result:
                print(f"[ERROR] Error: {latex_result['error']}")
            else:
                print(f"\n[APPROVED] Paper generation complete!")
                print(f"LaTeX document saved to: {latex_result['output_path']}")
        else:
            print("All sections are already completed. Generating LaTeX...")
            latex_result = generator.generate_latex()
            
            if "error" in latex_result:
                print(f"[ERROR] Error: {latex_result['error']}")
            else:
                print(f"\n[APPROVED] Paper generation complete!")
                print(f"LaTeX document saved to: {latex_result['output_path']}")
    else:
        # Start fresh with a new query
        generator.run_pipeline(args.query, not args.non_interactive)

if __name__ == "__main__":
    main()