import platform
import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
import subprocess
from dotenv import load_dotenv

# Import custom modules
from papers_extractor_bfs import ArxivReferenceExplorer
try:
    from knowledge_base import ResearchKnowledgeBase, ingest_json_directory, get_review_topics
    has_kb = True
except ImportError:
    has_kb = False
    st.error("Knowledge base module not found. Some features will be disabled.")

try:
    from review_writer import PaperGenerator, KnowledgeBaseConnector
    has_review_writer = True
except ImportError:
    has_review_writer = False
    st.error("Review writer module not found. Some features will be disabled.")

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Error: OPENAI_API_KEY not found in environment variables or .env file. Please set this variable and restart the app.")

# App title and description
st.title("Scientific Paper Analysis and Generation")
st.markdown("This application helps researchers extract papers from arXiv, build a knowledge base, and generate review papers.")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Extract Papers", "Knowledge Base", "Generate Review Paper"])

# Tab 1: Extract Papers
with tab1:
    st.header("Extract Papers from arXiv")
    
    # Input parameters for paper extraction
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input("Research Query", "Single image to 3D")
        initial_results = st.number_input("Initial Results", min_value=1, max_value=10, value=3)
    
    with col2:
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
        max_depth = st.number_input("Max Traversal Depth", min_value=1, max_value=5, value=2)
    
    # Dynamic input for max papers per level
    max_papers = []
    st.subheader("Max Papers Per Level")
    cols = st.columns(max_depth)
    for i in range(max_depth):
        with cols[i]:
            max_papers.append(st.number_input(f"Level {i+1}", min_value=1, max_value=5, value=2 if i == 0 else 1))
    
    # Run extraction
    if st.button("Extract Papers"):
        try:
            with st.spinner("Initializing arXiv explorer..."):
                explorer = ArxivReferenceExplorer(
                    query=query,
                    initial_results=initial_results,
                    similarity_threshold=similarity_threshold
                )
                st.success("Initialization successful")
            
            with st.spinner(f"Traversing citation network (max depth: {max_depth})..."):
                progress_bar = st.progress(0)
                
                # Custom callback to update progress
                def progress_callback(current_level, max_level, papers_collected):
                    progress = current_level / max_level
                    progress_bar.progress(progress)
                    st.write(f"Level {current_level}/{max_level}: {len(papers_collected)} papers collected")
                
                explorer.bfs_traversal(max_depth=max_depth, max_per_level=max_papers, progress_callback=progress_callback)
                
                # Display results
                st.success(f"Extraction complete! Total papers collected: {len(explorer.paper_cache)}")
                
                # Show paper cache
                if st.checkbox("Show collected papers"):
                    for i, paper in enumerate(explorer.paper_cache.values()):
                        with st.expander(f"{i+1}. {paper['title']}"):
                            st.write(f"**Authors:** {', '.join(paper['authors'])}")
                            st.write(f"**Abstract:** {paper['abstract']}")
                            st.write(f"**URL:** {paper['url']}")
        
        except Exception as e:
            st.error(f"Error during paper extraction: {str(e)}")

# Tab 2: Knowledge Base
with tab2:
    st.header("Knowledge Base Management")
    
    if not has_kb:
        st.warning("Knowledge base module not available. Please make sure knowledge_base.py is in your project directory.")
    else:
        # Initialize Knowledge Base
        if st.button("Initialize/Check Knowledge Base"):
            try:
                with st.spinner("Initializing knowledge base..."):
                    kb = ResearchKnowledgeBase()
                    # Check if Neo4j Docker container is running
                    try:
                        
                        # Function to check if Neo4j container is running
                        def is_neo4j_running():
                            try:
                                result = subprocess.run(
                                    ["docker", "ps", "--filter", "name=neo4j", "--format", "{{.Names}}"],
                                    capture_output=True,
                                    text=True,
                                    check=False
                                )
                                return "neo4j" in result.stdout
                            except Exception:
                                return False
                        
                        # Function to start Neo4j container
                        def start_neo4j():
                            try:
                                subprocess.run(
                                    ["docker", "run", "--name", "neo4j", "-p7474:7474", "-p7687:7687",
                                     "-e", "NEO4J_AUTH=neo4j/password", "-d", "neo4j:latest"],
                                    check=True
                                )
                                return True
                            except subprocess.CalledProcessError:
                                # Container might already exist but is stopped
                                try:
                                    subprocess.run(["docker", "start", "neo4j"], check=True)
                                    return True
                                except subprocess.CalledProcessError:
                                    return False
                        
                        # Function to stop Neo4j container
                        def stop_neo4j():
                            try:
                                subprocess.run(["docker", "stop", "neo4j"], check=True)
                                return True
                            except subprocess.CalledProcessError:
                                return False
                        
                        # Check current status
                        neo4j_running = is_neo4j_running()
                        
                        # Display Neo4j status
                        if neo4j_running:
                            st.success("Neo4j is running")
                            st.info("Neo4j browser available at: http://localhost:7474")
                            if st.button("Stop Neo4j"):
                                if stop_neo4j():
                                    st.success("Neo4j stopped successfully")
                                    st.rerun()
                                else:
                                    st.error("Failed to stop Neo4j")
                        else:
                            st.warning("Neo4j is not running")
                            if st.button("Start Neo4j"):
                                if start_neo4j():
                                    st.success("Neo4j started successfully")
                                    st.info("Neo4j browser available at: http://localhost:7474")
                                    st.rerun()
                                else:
                                    st.error("Failed to start Neo4j")
                        
                        st.write("---")

                    except Exception as e:
                        st.error(f"Error checking Neo4j status: {str(e)}")
                        st.info("Make sure Docker is installed and running")
                    # Check if DB is populated
                    is_populated = kb.is_db_populated()
                    
                    if is_populated:
                        st.success("Knowledge base is already populated.")
                        st.write(f"Papers in vector DB: {kb.vector_db.count()}")
                        st.write(f"Images in vector DB: {kb.image_db.count() if hasattr(kb, 'image_db') else 'N/A'}")
                    else:
                        st.warning("Knowledge base is empty. Ready for ingestion.")
            except Exception as e:
                st.error(f"Error initializing knowledge base: {str(e)}")
        
        # Ingest data
        if st.button("Ingest Data to Knowledge Base"):
            try:
                with st.spinner("Ingesting data to knowledge base..."):
                    kb = ResearchKnowledgeBase()
                    ingest_json_directory(kb)
                    st.success("Data ingestion complete!")
                    st.write(f"Papers in vector DB: {kb.vector_db.count()}")
                    st.write(f"Images in vector DB: {kb.image_db.count() if hasattr(kb, 'image_db') else 'N/A'}")
            except Exception as e:
                st.error(f"Error during data ingestion: {str(e)}")
        
        # Generate review topics
        if st.button("Generate Review Topics"):
            try:
                with st.spinner("Generating review topics..."):
                    kb = ResearchKnowledgeBase()
                    topics = get_review_topics(kb)
                    
                    # Save to file
                    with open("review_topics.json", "w", encoding="utf-8") as f:
                        json.dump(topics, f, indent=2)
                    
                    # Display topics
                    st.success("Review topics generated and saved to review_topics.json")
                    st.json(topics)
            except Exception as e:
                st.error(f"Error generating review topics: {str(e)}")
        
        # Search functionality
        st.subheader("Search Knowledge Base")
        search_query = st.text_input("Search Query", "3D Reconstruction")
        include_images = st.checkbox("Include Images in Results", value=True)
        section_context = st.selectbox(
            "Section Context (Optional)", 
            [
                "None", 
                "Introduction", 
                "Literature Review", 
                "Methodology", 
                "Results", 
                "Discussion", 
                "Conclusion"
            ],
            index=0
        )
        
        if st.button("Search"):
            try:
                with st.spinner("Searching knowledge base..."):
                    kb = ResearchKnowledgeBase()
                    
                    # Here's the fixed line - self is already passed so we just need query and top_k
                    results = kb.hybrid_search(query=search_query, top_k=3, include_images=include_images)
                    
                    # Display results
                    st.success(f"Found {len(results['papers'])} papers and {len(results['images'])} images")
                    
                    # Display paper results
                    st.subheader("Papers")
                    for i, paper in enumerate(results['papers']):
                        with st.expander(f"Paper {i+1}: {paper.get('graph_data', {}).get('title', 'Untitled')}"):
                            st.write(f"**Paper ID:** {paper.get('id', 'Unknown')}")
                            st.write(f"**Authors:** {', '.join(paper.get('graph_data', {}).get('authors', ['Unknown']))}")
                            st.write(f"**Year:** {paper.get('graph_data', {}).get('year', 'Unknown')}")
                            
                            # Display strengths and limitations
                            if paper.get('strengths'):
                                st.write("**Strengths:**")
                                for strength in paper.get('strengths', [])[:5]:
                                    st.write(f"- {strength}")
                            
                            if paper.get('limitations'):
                                st.write("**Limitations:**")
                                for limitation in paper.get('limitations', [])[:5]:
                                    st.write(f"- {limitation}")
                            
                            # Show content preview
                            st.write("**Content Preview:**")
                            for chunk in paper.get('chunks', [])[:2]:
                                st.text_area(f"Chunk from {chunk.get('section', 'unknown section')}", 
                                            chunk.get('text', 'No text'), 
                                            height=100)
                    
                    # Display image results
                    if include_images and 'images' in results:
                        st.subheader("Images")
                        for i, img in enumerate(results['images'][:5]):
                            with st.expander(f"Image {i+1} from paper {img.get('paper_id', 'Unknown')}"):
                                st.write(f"**Description:** {img.get('description', 'No description')}")
                                st.write(f"**Path:** {img.get('path', 'Unknown path')}")
                                
                                # If possible, display the image
                                img_path = img.get('path')
                                if img_path and os.path.exists(img_path):
                                    st.image(img_path, caption=img.get('description', '')[:50])
            except Exception as e:
                st.error(f"Error searching knowledge base: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
# Tab 3: Generate Review Paper
with tab3:
    st.header("Generate Review Paper")
    
    if not has_review_writer or not has_kb:
        st.warning("Review writer or knowledge base modules not available. Please make sure all required modules are in your project directory.")
    else:
        # Create two columns for layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Check for existing state
            has_state = os.path.exists("paper_state.json")
            
            # Options based on state
            if has_state:
                st.info("Found existing paper state. You can continue from where you left off or start a new paper.")
                
                # Load state to show info
                try:
                    with open("paper_state.json", "r", encoding="utf-8") as f:
                        state = json.load(f)
                    
                    st.write(f"**Existing paper:** {state.get('title', 'Untitled')}")
                    st.write(f"**Current section:** {state.get('current_section_index', 0) + 1}/{len(state.get('sections', []))}")
                    
                    # Get completed and remaining sections
                    completed = [s for s in state.get('sections', []) if s.get('status') == 'approved']
                    remaining = [s for s in state.get('sections', []) if s.get('status') != 'approved']
                    
                    st.write(f"**Completed sections:** {len(completed)}")
                    st.write(f"**Remaining sections:** {len(remaining)}")
                    
                    # Option to continue
                    continue_paper = st.checkbox("Continue from saved state", value=True)
                except Exception as e:
                    st.error(f"Error loading paper state: {str(e)}")
                    continue_paper = False
            else:
                continue_paper = False
            
            # Input for new paper
            if not continue_paper:
                research_query = st.text_input("Research Query for New Paper", "Recent advancements in Single Image to 3D reconstruction")
            
            # Interactive mode option
            interactive_mode = st.checkbox("Interactive Mode (Review each section before proceeding)", value=True)
            
            # Use enhanced knowledge base connector
            use_enhanced_kb = st.checkbox("Use Enhanced Knowledge Base (Context-aware searches)", value=True)
            
            # Generate button
            generate_button = st.button("Generate Paper")
        
        # Output Viewer
        with col2:
            # LaTeX File Viewer Tab
            latex_viewer_tab, pdf_viewer_tab = st.tabs(["LaTeX Code", "PDF Preview"])
            
            with latex_viewer_tab:
                # Find the most recent LaTeX file in output folder
                output_folder = "output"
                latex_files = []
                if os.path.exists(output_folder):
                    latex_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.tex')]
                
                if latex_files:
                    # Sort by modification time, most recent first
                    latest_latex = sorted(latex_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
                    st.write(f"**Viewing:** {latest_latex}")
                    
                    # Read and display the LaTeX file
                    try:
                        with open(latest_latex, 'r', encoding='utf-8') as f:
                            latex_content = f.read()
                        st.text_area("LaTeX Source", latex_content, height=600)
                        
                        # Add download button
                        st.download_button(
                            label="Download LaTeX File",
                            data=latex_content,
                            file_name=os.path.basename(latest_latex),
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error reading LaTeX file: {str(e)}")
                else:
                    st.info("No LaTeX files found. Generate a paper first.")
            
            with pdf_viewer_tab:
                # Find the most recent PDF file
                pdf_files = [f for f in os.listdir() if f.endswith('.pdf')]
                if pdf_files:
                    # Sort by modification time, most recent first
                    latest_pdf = sorted(pdf_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
                    st.write(f"**Viewing:** {latest_pdf}")
                    
                    # Display PDF file
                    try:
                        # Create a download button for the PDF
                        with open(latest_pdf, "rb") as f:
                            pdf_bytes = f.read()
                        
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=latest_pdf,
                            mime="application/pdf"
                        )
                        
                        # Embed PDF viewer
                        st.write("PDF preview not available in Streamlit. Please download the PDF to view it.")
                        # Alternative: Display the first page as an image if you have the capability
                    except Exception as e:
                        st.error(f"Error reading PDF file: {str(e)}")
                else:
                    st.info("No PDF files found. Generate a paper and compile it to PDF first.")
        
        # Compilation section
        st.subheader("Compile LaTeX to PDF")
        compile_col1, compile_col2 = st.columns([1, 3])
        
        with compile_col1:
            # Pick LaTeX file to compile
            output_folder = "output"  # Define the output folder
            latex_files = [f for f in os.listdir(output_folder) if f.endswith('.tex')] if os.path.exists(output_folder) else []
            if latex_files:
                selected_tex = st.selectbox("Select LaTeX file to compile", latex_files)
                compile_button = st.button("Compile to PDF")
                
                if compile_button:
                    with st.spinner("Compiling LaTeX to PDF..."):
                        try:
                            # Use subprocess to run pdflatex
                            import subprocess
                            
                            # Change to output directory
                            orig_dir = os.getcwd()
                            os.chdir(output_folder)
                            
                            # Run pdflatex
                            result = subprocess.run(
                                ["pdflatex", selected_tex], 
                                capture_output=True, 
                                text=True, 
                                check=False
                            )
                            
                            # Change back to original directory
                            os.chdir(orig_dir)
                            
                            if result.returncode == 0:
                                st.success(f"Successfully compiled {selected_tex} to PDF")
                                # Refresh the page to show the new PDF
                                st.rerun()
                            else:
                                # Show compilation errors
                                st.error("Compilation failed")
                                with st.expander("Show compilation log"):
                                    st.code(result.stdout + "\n" + result.stderr)
                        except Exception as e:
                            st.error(f"Error during compilation: {str(e)}")
                            st.info("Make sure pdflatex is installed and available in your PATH")
            else:
                st.info("No LaTeX files found to compile.")
        
        # Paper generation process
        if generate_button:
            try:
                with st.spinner("Initializing paper generator..."):
                    # Initialize knowledge base
                    kb = ResearchKnowledgeBase()
                    
                    # Initialize the appropriate knowledge base connector
                    if use_enhanced_kb:
                        st.info("Using enhanced knowledge base connector with context-aware search...")
                        # Import the enhanced version (assuming it's been implemented)
                        from review_writer import KnowledgeBaseConnector
                        kb_connector = KnowledgeBaseConnector(kb)
                    else:
                        # Use the original version
                        kb_connector = KnowledgeBaseConnector(kb)
                    
                    # Initialize paper generator
                    generator = PaperGenerator(kb_connector)
                    
                    if continue_paper and has_state:
                        # Continue from current state
                        if generator.load_state():
                            st.success(f"Loaded state with title: {generator.paper_state['title']}")
                            
                            current_index = generator.paper_state["current_section_index"]
                            
                            if current_index < len(generator.paper_state["sections"]):
                                progress_container = st.empty()
                                preview_container = st.empty()
                                approval_container = st.empty()
                                
                                while current_index < len(generator.paper_state["sections"]):
                                    section = generator.paper_state["sections"][current_index]
                                    progress_container.info(f"Writing section {current_index + 1}/{len(generator.paper_state['sections'])}: {section['title']}...")
                                    
                                    # Write section
                                    result = generator.write_section(current_index)
                                    
                                    if interactive_mode:
                                        # Show preview
                                        preview = result["content"]
                                        preview_container.text_area("Section Preview", preview, height=300)
                                        
                                        # Ask for approval
                                        if approval_container.button("Approve Section", key=f"approve_{current_index}"):
                                            generator.approve_section(current_index)
                                            st.success(f"Section approved: {section['title']}")
                                            current_index = generator.paper_state["current_section_index"]
                                        else:
                                            st.stop()
                                    else:
                                        # Auto-approve in non-interactive mode
                                        generator.approve_section(current_index)
                                        st.success(f"Section auto-approved: {section['title']}")
                                        current_index = generator.paper_state["current_section_index"]
                                
                                # Generate LaTeX at the end
                                st.info("Generating LaTeX document...")
                                latex_result = generator.generate_latex()
                                
                                if "error" in latex_result:
                                    st.error(f"Error: {latex_result['error']}")
                                else:
                                    st.success(f"Paper generation complete! LaTeX document saved to: {latex_result['output_path']}")
                                    # Refresh the page to show the new LaTeX file
                                    st.rerun()
                            else:
                                st.info("All sections are already completed. Generating LaTeX...")
                                latex_result = generator.generate_latex()
                                
                                if "error" in latex_result:
                                    st.error(f"Error: {latex_result['error']}")
                                else:
                                    st.success(f"Paper generation complete! LaTeX document saved to: {latex_result['output_path']}")
                                    # Refresh the page to show the new LaTeX file
                                    st.rerun()
                        else:
                            st.error("Failed to load saved state.")
                    else:
                        # Start fresh with new query
                        if not research_query:
                            st.error("Research query is required for a new paper.")
                            st.stop()
                        
                        st.info(f"Starting new paper generation for query: {research_query}")
                        generator.run_pipeline(research_query, interactive_mode)
                        
                        if not interactive_mode:
                            st.success("Paper generation complete!")
                            latex_result = generator.generate_latex()
                            
                            if "error" in latex_result:
                                st.error(f"Error: {latex_result['error']}")
                            else:
                                st.success(f"LaTeX document saved to: {latex_result['output_path']}")
                                # Refresh the page to show the new LaTeX file
                                st.rerun()
            except Exception as e:
                st.error(f"Error during paper generation: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("Scientific Paper Analysis System | Built with Streamlit")