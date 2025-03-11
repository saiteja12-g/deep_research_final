import json
import re
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import os
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
        
        REQUIREMENTS:
        1. Write a comprehensive, in-depth section (at least 800-1000 words)
        2. Use formal academic style appropriate for publication
        3. IMPORTANT: Use proper in-text citations in Harvard format (Author, Year)
           For example: (Smith, 2019) or (Smith et al., 2019) for multiple authors
        4. Incorporate content from the provided chunks and references thoroughly
        5. Reference figures where appropriate using the format (Figure X)
        6. Ensure the section flows logically and builds a complete narrative
        7. Include relevant technical details and methodology discussions
        
        You have the following reference material to incorporate:
        
        RELEVANT TEXT CHUNKS:
        {chunk_info}
        
        KEY REFERENCES:
        {ref_info}
        
        AVAILABLE FIGURES:
        {fig_info}
        """,
            expected_output="A comprehensive, well-written section (800-1000+ words) with numerous inline citations",
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
    
    def validate_section(self, section_index):
        """Validate a section for quality, length, and cohesion with the rest of the paper."""
        section = self.paper_state["sections"][section_index]
        
        # Get the previous 5 sections for context (or as many as available)
        start_idx = max(0, section_index - 5)
        prev_sections = self.paper_state["sections"][start_idx:section_index]
        
        # Format previous sections for context
        prev_context = "\n\n".join([
            f"SECTION: {s['title']}\n{s['content'][:300]}..." 
            for s in prev_sections
        ])
        
        # Create a validation task
        validation_task = Task(
            description=f"""
            Validate the quality and completeness of the following section for a scientific paper:
            
            PAPER TITLE: {self.paper_state['title']}
            CURRENT SECTION: {section['title']}
            
            SECTION CONTENT:
            {section['content']}
            
            PREVIOUS SECTIONS CONTEXT:
            {prev_context}
            
            VALIDATION CRITERIA:
            1. LENGTH: Section should be comprehensive (800+ words). Is it sufficiently detailed?
            2. CITATIONS: At least 5-7 citations should be included. Are there enough?
            3. TECHNICAL DEPTH: Content should be technically sound and thorough. Is it?
            4. COHESION: Does it flow well from previous sections?
            5. FIGURE USAGE: Does it reference relevant figures? Are they integrated well?
            6. COMPLETENESS: Does it cover all aspects needed for this section?
            
            INSTRUCTIONS:
            - Evaluate the section against each criterion
            - Provide a pass/fail status for each criterion
            - Give specific recommendations for improvement
            - Provide an overall assessment (APPROVE or REVISE)
            - If REVISE, list the top 3 most critical issues to fix
            
            OUTPUT FORMAT:
            LENGTH: [PASS/FAIL] - [Specific comments]
            CITATIONS: [PASS/FAIL] - [Specific comments]
            TECHNICAL DEPTH: [PASS/FAIL] - [Specific comments]
            COHESION: [PASS/FAIL] - [Specific comments]
            FIGURE USAGE: [PASS/FAIL] - [Specific comments]
            COMPLETENESS: [PASS/FAIL] - [Specific comments]
            
            OVERALL: [APPROVE or REVISE]
            
            [If REVISE, list specific improvements needed]
            """,
            expected_output="A detailed section validation report with specific feedback",
            agent=self.agents["researcher"]  # Using researcher as validator
        )
        
        # Create a crew and execute
        validation_crew = Crew(
            agents=[self.agents["researcher"]],
            tasks=[validation_task],
            verbose=True
        )
        
        result = validation_crew.kickoff()
        
        # Get the string result
        if hasattr(result, 'raw_output'):
            validation_text = result.raw_output
        else:
            validation_text = str(result)
            
        # Determine if the section is approved
        approved = "APPROVE" in validation_text.upper()
        
        return {
            "approved": approved,
            "feedback": validation_text
        }    
    def create_outline(self, query):
        """Create a more focused outline for the paper based on the research."""
        # Create a task to generate a focused outline
        outline_task = Task(
            description=f"""
            Create a focused outline for a scientific paper on the topic: "{query}"
            
            Paper Title: {self.paper_state['title']}
            Abstract: {self.paper_state['abstract']}
            
            REQUIREMENTS:
            1. Create a streamlined outline with 5-7 sections maximum (not including References)
            2. Follow standard scientific paper structure 
            3. Merge related topics to create more comprehensive sections
            4. Each section should have 2-3 subsections
            5. Consider the available research data and themes
            
            Current themes and topics found in research:
            {', '.join([ref['title'] for ref_id, ref in list(self.paper_state['references'].items())[:10]])}
            
            OUTPUT FORMAT:
            Provide a hierarchical outline with:
            - Main sections (5-7 total)
            - Subsections (2-3 per main section)
            - Brief description of content for each
            """,
            expected_output="A focused paper outline with 5-7 main sections and subsections",
            agent=self.agents["researcher"]
        )
        
        # Create a crew and execute
        outline_crew = Crew(
            agents=[self.agents["researcher"]],
            tasks=[outline_task],
            verbose=True
        )
        
        result = outline_crew.kickoff()
        
        # Get the string result
        if hasattr(result, 'raw_output'):
            outline_text = result.raw_output
        else:
            outline_text = str(result)
            
        # Parse the outline to extract sections
        sections = []
        main_section = None
        
        # Use regex to find main sections and subsections
        # Pattern looks for numbered sections like "1. Introduction" or "Section 1: Introduction"
        main_pattern = r'(?:^|\n)(?:\d+\.\s+|Section\s+\d+:\s*)([A-Za-z\s]+)'
        
        matches = re.finditer(main_pattern, outline_text, re.MULTILINE)
        for match in matches:
            section_title = match.group(1).strip()
            if section_title.lower() not in ['references', 'bibliography']:
                sections.append({"title": section_title, "content": "", "status": SectionStatus.PENDING.value})
                
        # Fallback if regex didn't find enough sections
        if len(sections) < 5:
            # Use standard scientific paper sections
            standard_sections = [
                "Introduction", 
                "Literature Review",
                "Methodology",
                "Results",
                "Discussion", 
                "Conclusion"
            ]
            
            # Add any missing standard sections
            existing = [s["title"].lower() for s in sections]
            for section in standard_sections:
                if section.lower() not in existing:
                    sections.append({"title": section, "content": "", "status": SectionStatus.PENDING.value})
        
        # Update paper state with new sections
        self.paper_state["sections"] = sections[:7]  # Limit to at most 7 sections
        self.save_state()
        
        return [s["title"] for s in self.paper_state["sections"]]
    
    # def generate_latex(self):
    #     """Generate a LaTeX document from the paper state."""
    #     # Check if all sections are approved
    #     not_approved = [s for s in self.paper_state["sections"] if s["status"] != SectionStatus.APPROVED.value]
    #     if not_approved:
    #         section_names = [s["title"] for s in not_approved]
    #         return {
    #             "error": f"Not all sections are approved. Pending sections: {', '.join(section_names)}"
    #         }
        
    #     # Format section content
    #     sections_content = []
    #     for section in self.paper_state["sections"]:
    #         # Convert inline citations (Author, Year) to LaTeX \cite commands
    #         content = section["content"]
    #         # Match patterns like (Author, Year) or (Author et al., Year)
    #         citation_pattern = r'\(([A-Za-z]+)(?:\s+et\s+al\.?)?,\s*(\d{4})\)'
            
    #         # Find all citation matches
    #         matches = re.findall(citation_pattern, content)
            
    #         # Create citation keys and replace inline citations
    #         for author, year in matches:
    #             cite_key = f"{author.lower()}{year}"
    #             content = content.replace(f"({author}, {year})", f"\\cite{{{cite_key}}}")
    #             content = content.replace(f"({author} et al., {year})", f"\\cite{{{cite_key}}}")
            
    #         # Add formatted section to sections content
    #         sections_content.append(f"\\section{{{section['title']}}}\n\n{content}")
        
    #     # Join all sections
    #     all_sections = "\n\n".join(sections_content)
        
    #     # Format references in BibTeX format with proper keys
    #     references_formatted = []
    #     for ref_id, ref_data in self.paper_state["references"].items():
    #         # Create a unique citation key from author and year
    #         authors = ref_data.get("authors", ["Unknown"])
    #         first_author = authors[0].split()[-1] if authors else "Unknown"
    #         year = ref_data.get("year", "2000")
    #         cite_key = f"{first_author.lower()}{year}"
            
    #         # Get other reference details
    #         title = ref_data.get("title", "Unknown Title")
            
    #         # Create BibTeX entry with required fields
    #         bibtex_entry = f"""@article{{{cite_key},
    # title = {{{title}}},
    # author = {{{' and '.join(authors)}}},
    # year = {{{year}}},
    # journal = {{Journal of Research}},
    # volume = {{1}},
    # number = {{1}},
    # pages = {{1--10}},
    # doi = {{10.0000/journal.0000}}
    # }}"""
    #         references_formatted.append(bibtex_entry)
        
    #     references_text = "\n\n".join(references_formatted)
        
    #     # Format figures
    #     figures_text = ""
    #     for i, fig in enumerate(self.paper_state["figures"]):
    #         # Create a clean filename for the figure
    #         clean_name = f"figure_{i+1}"
    #         fig_desc = fig.get("description", "")
    #         if len(fig_desc) > 100:
    #             fig_desc = fig_desc[:100] + "..."
            
    #         # Create LaTeX figure environment
    #         figure = f"""\\begin{{figure}}[htbp]
    #     \\centering
    #     \\includegraphics[width=0.8\\columnwidth]{{{clean_name}}}
    #     \\caption{{{fig_desc}}}
    #     \\label{{fig:{clean_name}}}
    # \\end{{figure}}"""
    #         figures_text += figure + "\n\n"
        
    #     # Create figures information for README
    #     figures_info = []
    #     for i, fig in enumerate(self.paper_state["figures"]):
    #         clean_name = f"figure_{i+1}"
    #         source_path = fig.get("path", "unknown_path")
    #         figures_info.append(f"{source_path} -> {clean_name}.jpg")
        
    #     # Create the LaTeX generation task
    #     latex_task = Task(
    #         description=f"""
    #         Generate a complete LaTeX document for a scientific paper following standard academic publishing format.
            
    #         Paper Title: {self.paper_state['title']}
    #         Abstract: {self.paper_state['abstract']}
            
    #         REQUIREMENTS:
    #         1. Use the standard scientific article class with two-column layout (IEEEtran or similar)
    #         2. Include proper title, authors, and abstract
    #         3. Use the following author names: John Doe, Jane Smith, and Alex Johnson (with placeholder affiliations)
    #         4. Format all sections properly with correct hierarchy
    #         5. IMPORTANT: The LaTeX document should be COMPLETE and COMPILABLE
    #         6. Use the natbib package for citation formatting with author-year style
    #         7. Include only necessary packages (no need for complex customizations)
    #         8. Generate a references section at the end using BibTeX style
            
    #         The paper has {len(self.paper_state['sections'])} sections and includes 
    #         {len(self.paper_state['figures'])} figures and {len(self.paper_state['references'])} references.
            
    #         SECTION CONTENT (already formatted with LaTeX commands):
    #         {all_sections}
            
    #         BIBLIOGRAPHY ENTRIES (in BibTeX format):
    #         {references_text}
            
    #         FIGURES TO INCLUDE (already formatted with LaTeX commands):
    #         {figures_text}
            
    #         Important Notes:
    #         - The bibliography entries are already in BibTeX format, but need to be properly integrated
    #         - All inline citations have been converted to LaTeX \cite{{key}} commands
    #         - Create a complete document with proper preamble and document structure
    #         - The document should be ready to compile with minimal modifications
    #         """,
    #         expected_output="A complete, professional LaTeX document with proper sections, figures, and bibliography",
    #         agent=self.agents["editor"]
    #     )
        
    #     # Create a crew to execute the task
    #     latex_crew = Crew(
    #         agents=[self.agents["editor"]],
    #         tasks=[latex_task],
    #         verbose=True
    #     )
        
    #     # Execute the LaTeX generation task
    #     result = latex_crew.kickoff()
        
    #     # Get the string result from the CrewOutput object
    #     if hasattr(result, 'raw_output'):
    #         latex_document = result.raw_output
    #     else:
    #         # Try to get the result as a string representation
    #         latex_document = str(result)
        
    #     # Save the LaTeX document and BibTeX file
    #     output_path = Path("output")
    #     output_path.mkdir(exist_ok=True)
        
    #     # Save main LaTeX document
    #     with open(output_path / "paper.tex", "w", encoding="utf-8") as f:
    #         f.write(latex_document)
        
    #     # Save BibTeX file
    #     with open(output_path / "references.bib", "w", encoding="utf-8") as f:
    #         f.write(references_text)
        
    #     # Create a figures directory and README
    #     figures_path = output_path / "figures"
    #     figures_path.mkdir(exist_ok=True)
        
    #     # Create README with figure copying instructions
    #     with open(figures_path / "README.txt", "w", encoding="utf-8") as f:
    #         f.write("Copy the following images to this directory:\n\n")
    #         for info in figures_info:
    #             f.write(f"{info}\n")
        
    #     # Also copy figures to the output directory if possible
    #     try:
    #         for i, fig in enumerate(self.paper_state["figures"]):
    #             source_path = fig.get("path", "")
    #             if os.path.exists(source_path):
    #                 target_path = figures_path / f"figure_{i+1}.jpg"
    #                 import shutil
    #                 shutil.copy2(source_path, target_path)
    #                 print(f"Copied figure: {source_path} -> {target_path}")
    #     except Exception as e:
    #         print(f"Error copying figures: {str(e)}")
        
    #     return {
    #         "latex_document": latex_document,
    #         "output_path": str(output_path / "paper.tex"),
    #         "bib_path": str(output_path / "references.bib"),
    #         "figures_path": str(figures_path)
    #     }
    
    def generate_latex(self):
        """Generate a LaTeX document from the paper state."""
        # Check if all sections are approved
        not_approved = [s for s in self.paper_state["sections"] if s["status"] != SectionStatus.APPROVED.value]
        if not_approved:
            section_names = [s["title"] for s in not_approved]
            return {
                "error": f"Not all sections are approved. Pending sections: {', '.join(section_names)}"
            }
        
        # Format section content
        sections_content = []
        for section in self.paper_state["sections"]:
            # Convert inline citations (Author, Year) to LaTeX \cite commands
            content = section["content"]
            # Match patterns like (Author, Year) or (Author et al., Year)
            citation_pattern = r'\(([A-Za-z]+)(?:\s+et\s+al\.?)?,\s*(\d{4})\)'
            
            # Find all citation matches
            matches = re.findall(citation_pattern, content)
            
            # Create citation keys and replace inline citations
            for author, year in matches:
                cite_key = f"{author.lower()}{year}"
                content = content.replace(f"({author}, {year})", f"\\cite{{{cite_key}}}")
                content = content.replace(f"({author} et al., {year})", f"\\cite{{{cite_key}}}")
            
            # Add formatted section to sections content
            sections_content.append(f"\\section{{{section['title']}}}\n\n{content}")
        
        # Join all sections
        all_sections = "\n\n".join(sections_content)
        
        # Format references in BibTeX format with proper keys
        references_formatted = []
        for ref_id, ref_data in self.paper_state["references"].items():
            # Create a unique citation key from author and year
            authors = ref_data.get("authors", ["Unknown"])
            first_author = authors[0].split()[-1] if authors else "Unknown"
            year = ref_data.get("year", "2000")
            cite_key = f"{first_author.lower()}{year}"
            
            # Get other reference details
            title = ref_data.get("title", "Unknown Title")
            
            # Create BibTeX entry with required fields
            bibtex_entry = f"""@article{{{cite_key},
            title = {{{title}}},
            author = {{{' and '.join(authors)}}},
            year = {{{year}}},
            journal = {{Journal of Research}},
            volume = {{1}},
            number = {{1}},
            pages = {{1--10}},
            doi = {{10.0000/journal.0000}}
            }}"""
            references_formatted.append(bibtex_entry)
        
        references_text = "\n\n".join(references_formatted)
        
        # Format figures
        figures_text = ""
        for i, fig in enumerate(self.paper_state["figures"]):
            # Create a clean filename for the figure
            clean_name = f"figure_{i+1}"
            fig_desc = fig.get("description", "")
            if len(fig_desc) > 100:
                fig_desc = fig_desc[:100] + "..."
            
            # Create LaTeX figure environment
            figure = f"""\\begin{{figure}}[htbp]
                \\centering
                \\includegraphics[width=0.8\\columnwidth]{{{clean_name}}}
                \\caption{{{fig_desc}}}
                \\label{{fig:{clean_name}}}
            \\end{{figure}}"""
            figures_text += figure + "\n\n"
        
        # Create figures information for README
        figures_info = []
        for i, fig in enumerate(self.paper_state["figures"]):
            clean_name = f"figure_{i+1}"
            source_path = fig.get("path", "unknown_path")
            figures_info.append(f"{source_path} -> {clean_name}.jpg")
        
        # Create the LaTeX document directly
        paper_title = self.paper_state["title"]
        paper_abstract = self.paper_state["abstract"]
        
        # Build the complete LaTeX document with sections inline
        latex_document = f"""\\documentclass[conference]{{IEEEtran}}
            \\usepackage{{graphicx}}
            \\usepackage{{natbib}}
            \\usepackage{{amsmath,amssymb,amsfonts}}
            \\usepackage{{algorithmic}}
            \\usepackage{{textcomp}}
            \\usepackage{{xcolor}}
            \\usepackage{{hyperref}}

            \\title{{{paper_title}}}

            \\author{{
            \\IEEEauthorblockN{{John Doe}}
            \\IEEEauthorblockA{{
            Department of Computer Science\\\\
            University Example\\\\
            City, Country\\\\
            email@example.com}}
            \\and
            \\IEEEauthorblockN{{Jane Smith}}
            \\IEEEauthorblockA{{
            Department of Research\\\\
            University Sample\\\\
            City, Country\\\\
            email2@example.com}}
            \\and
            \\IEEEauthorblockN{{Alex Johnson}}
            \\IEEEauthorblockA{{
            Research Institute\\\\
            Organization Name\\\\
            City, Country\\\\
            email3@example.com}}
            }}

            \\begin{{document}}

            \\maketitle

            \\begin{{abstract}}
            {paper_abstract}
            \\end{{abstract}}

            {all_sections}

            % Bibliography section
            \\bibliographystyle{{unsrtnat}}
            \\bibliography{{references}}

            \\end{{document}}
    """
        
        # Save the LaTeX document and BibTeX file
        output_path = Path("output")
        output_path.mkdir(exist_ok=True)
        
        # Save main LaTeX document
        with open(output_path / "paper.tex", "w", encoding="utf-8") as f:
            f.write(latex_document)
        
        # Save BibTeX file
        with open(output_path / "references.bib", "w", encoding="utf-8") as f:
            f.write(references_text)
        
        # Create a figures directory and README
        figures_path = output_path / "figures"
        figures_path.mkdir(exist_ok=True)
        
        # Create README with figure copying instructions
        with open(figures_path / "README.txt", "w", encoding="utf-8") as f:
            f.write("Copy the following images to this directory:\n\n")
            for info in figures_info:
                f.write(f"{info}\n")
        
        # Also copy figures to the output directory if possible
        try:
            for i, fig in enumerate(self.paper_state["figures"]):
                source_path = fig.get("path", "")
                if os.path.exists(source_path):
                    target_path = figures_path / f"figure_{i+1}.jpg"
                    import shutil
                    shutil.copy2(source_path, target_path)
                    print(f"Copied figure: {source_path} -> {target_path}")
        except Exception as e:
            print(f"Error copying figures: {str(e)}")
        
        return {
            "latex_document": latex_document,
            "output_path": str(output_path / "paper.tex"),
            "bib_path": str(output_path / "references.bib"),
            "figures_path": str(figures_path)
        }
    def run_pipeline(self, query, interactive=True):
        """Run the complete paper generation pipeline."""
        # Step 1: Initial research
        print("📚 Performing initial research...")
        research_results = self.initial_research(query)
        
        print(f"\nTitle: {research_results['title']}")
        print(f"Found {research_results['references']} references and {research_results['figures']} figures")
        
        # Step 2: Create a focused outline
        print("\n📝 Creating focused outline...")
        sections = self.create_outline(query)
        print(f"Generated outline with {len(sections)} sections:")
        for i, section in enumerate(sections, 1):
            print(f"{i}. {section}")
        
        if interactive:
            proceed = input("\nContinue with this outline? (yes/no): ").lower()
            if proceed != "yes":
                print("Exiting. You can modify the paper_state.json file and restart.")
                return
        
        # Step 3: Write each section
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
                
            else:
                # In non-interactive mode, validate the section automatically
                print("\n🔍 Validating section quality...")
                validation = self.validate_section(i)
                
                if validation["approved"]:
                    print("✅ Section validation passed")
                else:
                    print("❌ Section validation failed. Rewriting section...")
                    print("\nFeedback:")
                    print("-" * 40)
                    print(validation["feedback"])
                    print("-" * 40)
                    
                    # Try rewriting once
                    print("\n📝 Rewriting section based on feedback...")
                    result = self.write_section(i)
                    
                    # Validate again
                    validation = self.validate_section(i)
                    if validation["approved"]:
                        print("✅ Rewritten section validation passed")
                    else:
                        print("⚠️ Validation still failed but continuing with best version")
            
            # Approve the section
            self.approve_section(i)
            print(f"✅ Section completed: {section['title']}")
        
        # Step 4: Generate LaTeX
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