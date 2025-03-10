import os
import json
import re
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class SectionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    APPROVED = "approved"

class PaperGenerator:
    def __init__(self, kb_connector):
        """Initialize the Paper Generator with a knowledge base connector."""
        self.kb_connector = kb_connector
        self.paper_state = {
            "title": "",
            "abstract": "",
            "sections": [],
            "references": {},
            "figures": [],
            "current_section_index": 0
        }
        
        # Initialize agents
        self.agents = self._create_agents()
        
    def _create_agents(self):
        """Create the agent crew for paper generation."""
        # Base model
        base_model = ChatOpenAI(temperature=0.2, model="gpt-4-turbo")
        creative_model = ChatOpenAI(temperature=0.5, model="gpt-4-turbo")
        
        # Create agents
        researcher = Agent(
            role="Research Specialist",
            goal="Find and analyze relevant scientific literature",
            backstory="""You are an expert researcher who can query knowledge bases, 
            analyze scientific papers, and extract key information. You understand 
            scientific methodology and can identify important research findings.""",
            verbose=True,
            allow_delegation=False,
            llm=base_model
        )
        
        writer = Agent(
            role="Scientific Writer",
            goal="Write high-quality scientific content with appropriate citations",
            backstory="""You are a professional scientific writer with experience 
            publishing in top academic journals. You can synthesize complex information 
            into clear, structured prose following academic conventions.""",
            verbose=True,
            allow_delegation=False,
            llm=creative_model
        )
        
        editor = Agent(
            role="Publication Editor",
            goal="Ensure paper meets academic standards and format in LaTeX",
            backstory="""You are a seasoned academic editor who ensures papers are 
            properly structured, cited, and formatted. You have extensive experience 
            with LaTeX and academic publishing requirements.""",
            verbose=True,
            allow_delegation=False,
            llm=base_model
        )
        
        return {
            "researcher": researcher,
            "writer": writer,
            "editor": editor
        }
    
    def _query_knowledge_base(self, query, top_k=5):
        """Query the knowledge base for relevant information."""
        return self.kb_connector.search(query, top_k)
    
    def _extract_references(self, papers):
        """Extract references from paper data."""
        references = {}
        for paper in papers:
            paper_id = paper["id"]
            title = paper.get("graph_data", {}).get("title", "Unknown Title")
            authors = paper.get("graph_data", {}).get("authors", ["Unknown"])
            year = paper.get("graph_data", {}).get("year", "Unknown Year")
            
            references[paper_id] = {
                "title": title,
                "authors": authors,
                "year": year,
                "paper_id": paper_id
            }
        
        return references
    
    def _extract_figures(self, images):
        """Extract figures from image data."""
        figures = []
        for idx, image in enumerate(images):
            figures.append({
                "path": image["path"],
                "description": image["description"],
                "paper_id": image["paper_id"],
                "figure_id": f"fig_{idx + 1}"
            })
        
        return figures
    
    def save_state(self, filename="paper_state.json"):
        """Save the current paper state to a file."""
        with open(filename, "w") as f:
            json.dump(self.paper_state, f, indent=2)
    
    def load_state(self, filename="paper_state.json"):
        """Load the paper state from a file."""
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.paper_state = json.load(f)
            return True
        return False
    
    def initial_research(self, query):
        """Perform initial research based on user query."""
        # Create research task
        research_task = Task(
            description=f"""
            Research the topic: "{query}"
            
            1. Query the knowledge base to find relevant papers
            2. Analyze the key themes, methods, and findings
            3. Identify gaps or opportunities for a novel review
            4. Recommend a specific paper title that would be valuable to the field
            5. Provide a comprehensive abstract for this proposed paper
            6. Identify key sections that should be included in this paper
            
            Your output should be structured as:
            TITLE: [Your recommended title]
            ABSTRACT: [A comprehensive abstract]
            KEY SECTIONS: [List of 5-7 recommended sections for the paper]
            KEY FINDINGS: [Summary of important findings from the literature]
            
            Query: {query}
            """,
            expected_output="A structured research proposal with title, abstract, and sections",
            agent=self.agents["researcher"]
        )
        
        # Create a crew to execute the task
        research_crew = Crew(
            agents=[self.agents["researcher"]],
            tasks=[research_task],
            verbose=True
        )
        
        # Execute the research task
        result = research_crew.kickoff()
        
        # Get the string result from the CrewOutput object
        if hasattr(result, 'raw_output'):
            result_text = result.raw_output
        else:
            # Try to get the result as a string representation
            result_text = str(result)
        
        # Parse the results
        title_match = re.search(r"TITLE:(.*?)(?=ABSTRACT:|$)", result_text, re.DOTALL)
        abstract_match = re.search(r"ABSTRACT:(.*?)(?=KEY SECTIONS:|$)", result_text, re.DOTALL)
        sections_match = re.search(r"KEY SECTIONS:(.*?)(?=KEY FINDINGS:|$)", result_text, re.DOTALL)
        
        if title_match:
            self.paper_state["title"] = title_match.group(1).strip()
        
        if abstract_match:
            self.paper_state["abstract"] = abstract_match.group(1).strip()
        
        if sections_match:
            section_text = sections_match.group(1).strip()
            section_list = [s.strip() for s in re.split(r'\d+\.\s*|\n-\s*|\n\s*', section_text) if s.strip()]
            
            # Ensure we have standard sections if they're missing
            standard_sections = ["Introduction", "Literature Review", "Methodology", 
                                "Results", "Discussion", "Conclusion"]
            
            existing_sections = [s.lower() for s in section_list]
            
            # Add missing standard sections
            for section in standard_sections:
                if not any(section.lower() in s for s in existing_sections):
                    section_list.append(section)
            
            # Initialize sections
            self.paper_state["sections"] = [
                {"title": section, "content": "", "status": SectionStatus.PENDING.value}
                for section in section_list
            ]
        
        # Query the knowledge base for relevant papers and images
        search_results = self._query_knowledge_base(query, top_k=10)
        
        # Extract references and figures
        self.paper_state["references"].update(
            self._extract_references(search_results.get("papers", []))
        )
        self.paper_state["figures"].extend(
            self._extract_figures(search_results.get("images", []))
        )
        
        # Save the state
        self.save_state()
        
        return {
            "title": self.paper_state["title"],
            "abstract": self.paper_state["abstract"],
            "sections": [s["title"] for s in self.paper_state["sections"]],
            "references": len(self.paper_state["references"]),
            "figures": len(self.paper_state["figures"])
        }
    
    def write_section(self, section_index=None):
        """Write content for a specific section or the current section."""
        if section_index is None:
            section_index = self.paper_state["current_section_index"]
        
        if section_index >= len(self.paper_state["sections"]):
            return {"error": "Section index out of bounds"}
        
        section = self.paper_state["sections"][section_index]
        
        # Query for additional information specific to this section
        section_query = f"{self.paper_state['title']} {section['title']}"
        search_results = self._query_knowledge_base(section_query, top_k=5)
        
        # Extract chunks from papers
        chunks = []
        for paper in search_results.get("papers", []):
            for chunk in paper.get("chunks", []):
                chunks.append({
                    "text": chunk.get("text", ""),
                    "paper_id": paper.get("id", ""),
                    "section": chunk.get("section", "unknown")
                })
        
        # Update references and figures
        self.paper_state["references"].update(
            self._extract_references(search_results.get("papers", []))
        )
        
        new_figures = self._extract_figures(search_results.get("images", []))
        existing_paths = [fig["path"] for fig in self.paper_state["figures"]]
        for fig in new_figures:
            if fig["path"] not in existing_paths:
                self.paper_state["figures"].append(fig)
        
        # Create the writing task
        previous_sections = []
        for i in range(section_index):
            prev_section = self.paper_state["sections"][i]
            # Truncate long sections for context
            content = prev_section["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            previous_sections.append(f"{prev_section['title']}: {content}")
        
        # Format some chunk and reference information to include in the prompt
        chunk_info = "\n\n".join([
            f"CHUNK {i+1} (from {chunk['paper_id']}):\n{chunk['text'][:300]}..." 
            for i, chunk in enumerate(chunks[:5])
        ])
        
        ref_info = "\n".join([
            f"[{ref_id}] {ref_data['title']} by {', '.join(ref_data['authors'])} ({ref_data['year']})"
            for ref_id, ref_data in list(self.paper_state["references"].items())[:10]
        ])
        
        fig_info = "\n".join([
            f"FIGURE {i+1}: {fig['description'][:100]}... (from paper {fig['paper_id']})"
            for i, fig in enumerate(self.paper_state["figures"][:3])
        ])
        
        writing_task = Task(
            description=f"""
            Write the '{section['title']}' section for a scientific paper.
            
            Paper Title: {self.paper_state['title']}
            Abstract: {self.paper_state['abstract']}
            
            You have the following reference material to incorporate:
            
            RELEVANT TEXT CHUNKS:
            {chunk_info}
            
            KEY REFERENCES:
            {ref_info}
            
            AVAILABLE FIGURES:
            {fig_info}
            
            Guidelines:
            1. This is section {section_index + 1} of {len(self.paper_state['sections'])}
            2. Write in a formal, academic style appropriate for publication
            3. Use inline citations in the format [AuthorYear] or [paper_id]
            4. Incorporate content from the provided chunks
            5. Reference figures where appropriate using the format (Fig. X)
            6. Ensure the section flows logically and fits with the overall paper structure
            
            Previous sections context:
            {' '.join(previous_sections) if previous_sections else 'This is the first section.'}
            """,
            expected_output="A complete, well-written section with inline citations",
            agent=self.agents["writer"]
        )
        
        # Create a crew to execute the task
        writing_crew = Crew(
            agents=[self.agents["writer"]],
            tasks=[writing_task],
            verbose=True
        )
        
        # Execute the writing task
        result = writing_crew.kickoff()
        
        # Get the string result from the CrewOutput object
        if hasattr(result, 'raw_output'):
            content = result.raw_output
        else:
            # Try to get the result as a string representation
            content = str(result)
        
        # Update the section content
        section["content"] = content
        section["status"] = SectionStatus.COMPLETED.value
        self.paper_state["sections"][section_index] = section
        
        # Save the state
        self.save_state()
        
        return {
            "section_title": section["title"],
            "content": content,
            "section_index": section_index,
            "status": section["status"]
        }
    
    def approve_section(self, section_index=None):
        """Approve a section and move to the next one."""
        if section_index is None:
            section_index = self.paper_state["current_section_index"]
        
        if section_index >= len(self.paper_state["sections"]):
            return {"error": "Section index out of bounds"}
        
        # Approve the section
        section = self.paper_state["sections"][section_index]
        section["status"] = SectionStatus.APPROVED.value
        self.paper_state["sections"][section_index] = section
        
        # Move to the next section if current one was approved
        if section_index == self.paper_state["current_section_index"]:
            self.paper_state["current_section_index"] += 1
        
        # Save the state
        self.save_state()
        
        return {
            "section_title": section["title"],
            "section_index": section_index,
            "status": section["status"],
            "next_section_index": self.paper_state["current_section_index"]
        }
    
    def generate_latex(self):
        """Generate a LaTeX document from the paper state."""
        # Check if all sections are approved
        not_approved = [s for s in self.paper_state["sections"] if s["status"] != SectionStatus.APPROVED.value]
        if not_approved:
            section_names = [s["title"] for s in not_approved]
            return {
                "error": f"Not all sections are approved. Pending sections: {', '.join(section_names)}"
            }
        
        # Format section content and references for the task description
        sections_content = "\n\n".join([
            f"SECTION {i+1}: {section['title']}\n\n{section['content'][:500]}..."
            for i, section in enumerate(self.paper_state["sections"])
        ])
        
        references_formatted = "\n".join([
            f"[{ref_id}] {ref_data['title']} by {', '.join(ref_data['authors'])} ({ref_data['year']})"
            for ref_id, ref_data in list(self.paper_state["references"].items())[:20]
        ])
        
        figures_formatted = "\n".join([
            f"FIGURE {i+1}: {fig['figure_id']} - {fig['description'][:100]}... (from paper {fig['paper_id']})"
            for i, fig in enumerate(self.paper_state["figures"][:5])
        ])
        
        # Create the LaTeX generation task
        latex_task = Task(
            description=f"""
            Generate a complete LaTeX document for a scientific paper.
            
            Paper Title: {self.paper_state['title']}
            Abstract: {self.paper_state['abstract']}
            
            The paper has {len(self.paper_state['sections'])} sections and includes 
            {len(self.paper_state['figures'])} figures and {len(self.paper_state['references'])} references.
            
            SECTION CONTENT:
            {sections_content}
            
            KEY REFERENCES:
            {references_formatted}
            
            FIGURES:
            {figures_formatted}
            
            Requirements:
            1. Use standard LaTeX article class with appropriate packages
            2. Include a title, authors (placeholder), and abstract
            3. Set up proper section headings for all sections
            4. Include all content from each section
            5. Set up figure references correctly
            6. Create a proper bibliography with all references
            7. Ensure the document compiles correctly
            
            Output the complete LaTeX document including preamble, document environment, and all content.
            """,
            expected_output="A complete LaTeX document ready for compilation",
            agent=self.agents["editor"]
        )
        
        # Create a crew to execute the task
        latex_crew = Crew(
            agents=[self.agents["editor"]],
            tasks=[latex_task],
            verbose=True
        )
        
        # Execute the LaTeX generation task
        result = latex_crew.kickoff()
        
        # Get the string result from the CrewOutput object
        if hasattr(result, 'raw_output'):
            latex_document = result.raw_output
        else:
            # Try to get the result as a string representation
            latex_document = str(result)
        
        # Save the LaTeX document
        output_path = Path("output")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "paper.tex", "w", encoding="utf-8") as f:
            f.write(latex_document)
        
        # Create a figures directory and README
        figures_path = output_path / "figures"
        figures_path.mkdir(exist_ok=True)
        
        with open(figures_path / "README.txt", "w", encoding="utf-8") as f:
            f.write("Copy the following images to this directory:\n\n")
            for fig in self.paper_state["figures"]:
                f.write(f"{fig['path']} -> {fig['figure_id']}.jpg\n")
        
        return {
            "latex_document": latex_document,
            "output_path": str(output_path / "paper.tex")
        }
    
    def run_pipeline(self, query, interactive=True):
        """Run the complete paper generation pipeline."""
        # Step 1: Initial research
        print("📚 Performing initial research...")
        research_results = self.initial_research(query)
        
        print(f"\nTitle: {research_results['title']}")
        print(f"Sections: {', '.join(research_results['sections'])}")
        print(f"Found {research_results['references']} references and {research_results['figures']} figures")
        
        if interactive:
            proceed = input("\nContinue with this outline? (yes/no): ").lower()
            if proceed != "yes":
                print("Exiting. You can modify the paper_state.json file and restart.")
                return
        
        # Step 2: Write each section
        for i, section in enumerate(self.paper_state["sections"]):
            print(f"\n📝 Writing section {i+1}/{len(self.paper_state['sections'])}: {section['title']}...")
            result = self.write_section(i)
            
            if interactive:
                print("\nSection preview:")
                print("=" * 40)
                preview = result["content"]
                if len(preview) > 500:
                    preview = preview[:500] + "..."
                print(preview)
                print("=" * 40)
                
                approve = input("\nApprove this section? (yes/no): ").lower()
                if approve != "yes":
                    print("Exiting. You can modify the paper_state.json file and restart.")
                    return
            
            # Approve the section
            self.approve_section(i)
            print(f"✅ Section approved: {section['title']}")
        
        # Step 3: Generate LaTeX
        print("\n📄 Generating LaTeX document...")
        latex_result = self.generate_latex()
        
        if "error" in latex_result:
            print(f"❌ Error: {latex_result['error']}")
            return
        
        print(f"\n✅ Paper generation complete!")
        print(f"LaTeX document saved to: {latex_result['output_path']}")
        print("Figure information saved to: output/figures/README.txt")
        
        return latex_result


# Knowledge Base Connector
class KnowledgeBaseConnector:
    """Simple connector to the Research Knowledge Base."""
    
    def __init__(self, kb=None):
        """Initialize with an optional knowledge base instance."""
        self.kb = kb
    
    def search(self, query, top_k=5):
        """Search the knowledge base for a query."""
        if self.kb:
            return self.kb.hybrid_search(query, top_k=top_k)
        else:
            # Return dummy data for testing
            return self._get_dummy_data(query, top_k)
    
    def _get_dummy_data(self, query, top_k):
        """Generate dummy data for testing without a real knowledge base."""
        # You would replace this with actual knowledge base integration
        sample_papers = []
        sample_images = []
        
        # Load sample data from a file if available
        try:
            with open("search_result.json", "r") as f:
                sample_data = json.load(f)
                return sample_data
        except:
            # Create minimal dummy data if no file is available
            for i in range(min(3, top_k)):
                paper_id = f"paper_{i+1}"
                sample_papers.append({
                    "id": paper_id,
                    "chunks": [
                        {
                            "text": f"Sample text for paper {i+1} related to {query}. This would be actual content from the paper.",
                            "section": "introduction",
                            "distance": 0.5
                        }
                    ],
                    "graph_data": {
                        "title": f"Sample Paper {i+1} on {query}",
                        "authors": ["Author A", "Author B"],
                        "year": "2023"
                    }
                })
                
                sample_images.append({
                    "path": f"sample_image_{i+1}.jpg",
                    "description": f"A sample image related to {query}",
                    "paper_id": paper_id
                })
            
            return {"papers": sample_papers, "images": sample_images}