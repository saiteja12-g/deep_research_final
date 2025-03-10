from crewai import Agent, Task, Crew
from textwrap import dedent
from dotenv import load_dotenv
import json

load_dotenv()
# Initialize agents with specialized roles
class ReviewPaperCrew:
    def __init__(self, papers):
        self.llm = "gpt-4"  # Use appropriate LLM
        self.papers = papers
    
    def _format_paper_references(self):
        """Helper to format paper references dynamically"""
        return "\n".join(
            f"- {paper['id']}: {paper['content'][:100]}..."
            for paper in self.papers['papers']
        )

    def run(self):
        # Define agents
        outline_generator = Agent(
            role='Senior Research Architect',
            goal='Create comprehensive paper outline',
            backstory=dedent("""\
                Expert in structuring academic papers with deep understanding of 
                attention mechanism literature and IEEE paper standards"""),
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

        section_writer = Agent(
            role='Lead Technical Writer',
            goal='Draft detailed paper sections',
            backstory=dedent("""\
                Experienced academic writer specializing in deep learning architectures
                with 10+ years in top AI journals"""),
            verbose=True,
            llm=self.llm
        )

        literature_agent = Agent(
            role='Literature Review Specialist',
            goal='Ensure proper citation integration',
            backstory=dedent("""\
                PhD holder in Computer Science with expertise in bibliometric analysis
                and citation management"""),
            verbose=True,
            llm=self.llm
        )

        editor_in_chief = Agent(
            role='Chief Editor',
            goal='Enforce academic standards',
            backstory=dedent("""\
                Former editor of NeurIPS with strict quality control standards
                and attention to technical detail"""),
            verbose=True,
            llm=self.llm
        )

        # Define tasks
        outline_task = Task(
            description=dedent(f"""\
                Create detailed outline for review paper titled:
                'Advances in Attention Mechanisms for Artificial Intelligence: 
                Enhancing Deep Learning Efficiency'
                
                Incorporate these key papers: {[paper['id'] for paper in self.papers['papers']]}
                """),
            agent=outline_generator,
            expected_output="Markdown formatted outline with section hierarchy"
        )

        writing_task = Task(
            description=dedent(f"""\
                Draft full paper sections based on the approved outline.
                Include technical details from papers:
                {self._format_paper_references()}
                Maintain academic tone with inline citations using paper IDs"""),
            agent=section_writer,
            expected_output="Full draft in LaTeX format with citations",
            context=[outline_task]
        )

        
        citation_task = Task(
            description=dedent("""\
                Validate all citations and add reference section.
                Ensure proper citation format: (Author et al., Year) [ID].
                Cross-check references with provided papers list"""),
            agent=literature_agent,
            expected_output="Verified paper with complete references",
            context=[writing_task]
        )

        editing_task = Task(
            description=dedent("""\
                Perform final edit for:
                - Technical accuracy
                - Academic style compliance
                - Citation consistency
                - Figure/table integration from image analysis"""),
            agent=editor_in_chief,
            expected_output="Camera-ready paper in IEEE double-column format",
            context=[citation_task]
        )

        # Create and run crew
        crew = Crew(
            agents=[outline_generator, section_writer, literature_agent, editor_in_chief],
            tasks=[outline_task, writing_task, citation_task, editing_task],
            verbose=True,
            process='sequential'  # Ensures proper workflow
        )

        return crew.kickoff()

# Usage
if __name__ == "__main__":
    with open("search_result.json", "r") as f:
        papers = json.load(f)
    paper_crew = ReviewPaperCrew(papers=papers)
    result = paper_crew.run()
    # print(result)
    # Process the CrewAI output
    if hasattr(result, 'output'):
        formatted_content = result.output
    elif hasattr(result, 'raw_output'):
        formatted_content = result.raw_output
    else:
        formatted_content = str(result)
    
    # Write the formatted content to the markdown file
    with open("attention_mechanisms_review.md", "w", encoding="utf-8") as f:
        f.write(formatted_content)
    
    print(f"Review paper has been saved to attention_mechanisms_review.md")