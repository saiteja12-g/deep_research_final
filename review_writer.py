from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
from knowledge_base import ResearchKnowledgeBase

# Initialize knowledge base
kb = ResearchKnowledgeBase()

# Define agents
research_manager = Agent(
    role="Research Manager",
    goal="Understand the user's research needs and find the most relevant papers",
    backstory="Expert at analyzing research queries and identifying key themes",
    verbose=True,
    allow_delegation=False,
    tools=[kb.hybrid_search]
)

research_architect = Agent(
    role="Research Architect",
    goal="Create a comprehensive outline for scientific review papers",
    backstory="Experienced academic who understands scientific paper structure",
    verbose=True,
    allow_delegation=False
)

section_researcher = Agent(
    role="Section Content Specialist",
    goal="Create comprehensive, well-cited content for each section",
    backstory="PhD researcher with expertise in synthesizing information from multiple sources",
    verbose=True,
    allow_delegation=True,
    tools=[kb.hybrid_search]
)

visualization_specialist = Agent(
    role="Visualization and Figure Specialist",
    goal="Select and describe appropriate figures to support the text",
    backstory="Data visualization expert with experience in scientific publications",
    verbose=True,
    tools=[kb.get_images]
)

citation_manager = Agent(
    role="Citation and Bibliography Manager",
    goal="Ensure proper citation and reference formatting",
    backstory="Academic librarian with expertise in citation styles and reference management",
    verbose=True
)

latex_specialist = Agent(
    role="LaTeX Document Specialist",
    goal="Create a professionally formatted LaTeX document",
    backstory="Technical writer with extensive experience in LaTeX and scientific publishing",
    verbose=True
)

# Define tasks
initial_research_task = Task(
    description="""
    Analyze the user query and perform an initial search to identify relevant papers, key themes, and potential areas to explore.
    Output should include: 
    1. List of most relevant papers
    2. Key themes identified
    3. Suggested scope for the review
    """,
    agent=research_manager,
    expected_output="Detailed analysis of search results and suggested scope",
)

create_outline_task = Task(
    description="""
    Based on the initial research, create a comprehensive outline for a scientific review paper.
    The outline should include:
    1. Abstract
    2. Introduction
    3. Background/Related Work
    4. Main content sections (based on key themes)
    5. Discussion
    6. Conclusion
    7. References
    Justify your selection of sections and explain how they connect to the research query.
    """,
    agent=research_architect,
    expected_output="Detailed outline with sections and subsections"
)

# We'll create section-specific tasks after the outline is approved
def create_section_tasks(outline):
    tasks = []
    sections = parse_outline(outline)  # A function to extract sections from the outline
    
    for section in sections:
        tasks.append(Task(
            description=f"""
            Create content for the '{section}' section of the review paper.
            Your content should:
            1. Synthesize information from relevant papers
            2. Include proper inline citations
            3. Suggest relevant figures or tables
            4. Be comprehensive yet concise
            Query the knowledge base as needed for specific information.
            """,
            agent=section_researcher,
            expected_output=f"Complete content for {section} section with citations"
        ))
    
    return tasks

image_selection_task = Task(
    description="""
    Review all available figures from the retrieved papers.
    For each section of the paper:
    1. Identify the most relevant figures
    2. Create proper captions
    3. Explain how each figure supports the section content
    """,
    agent=visualization_specialist,
    expected_output="List of selected figures with captions and placement recommendations"
)

bibliography_task = Task(
    description="""
    Create a complete bibliography for the paper:
    1. Ensure all inline citations have corresponding references
    2. Format all references according to the specified style
    3. Create BibTeX entries for each reference
    """,
    agent=citation_manager,
    expected_output="Complete bibliography in BibTeX format"
)

latex_document_task = Task(
    description="""
    Create a complete LaTeX document incorporating all approved content:
    1. Use appropriate document class and preamble
    2. Include all sections with proper formatting
    3. Incorporate figures with proper placement
    4. Include complete bibliography
    5. Ensure all cross-references work correctly
    """,
    agent=latex_specialist,
    expected_output="Complete LaTeX document ready for compilation"
)

# Crew setup
crew = Crew(
    agents=[
        research_manager,
        research_architect,
        section_researcher,
        visualization_specialist,
        citation_manager,
        latex_specialist
    ],
    tasks=[
        initial_research_task,
        create_outline_task,
        # Section tasks will be added after outline approval
        image_selection_task,
        bibliography_task,
        latex_document_task
    ],
    verbose=2,
    process=Process.sequential  # Tasks execute in sequence
)

# Run the crew
def run_review_paper_generation(user_query):
    print(f"Starting review paper generation for query: {user_query}")
    
    # Run initial tasks
    result = crew.kickoff(inputs={"query": user_query})
    
    # After outline approval, create and add section tasks
    # This would require some user interaction in a real implementation
    section_tasks = create_section_tasks(result["create_outline_task"])
    
    # Update crew with new tasks
    crew.tasks.extend(section_tasks)
    
    # Run the remaining tasks
    final_result = crew.kickoff()
    
    return final_result