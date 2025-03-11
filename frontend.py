import streamlit as st
import subprocess
import re
import os
import sys
import time
import json
import threading
from pathlib import Path
import pandas as pd
import glob
import docker
from auto_refresh_component import timed_auto_refresh

# Set page configuration
st.set_page_config(
    page_title="Research Paper System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .terminal {
        background-color: black;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        height: 300px;
        overflow-y: auto;
        margin-bottom: 20px;
        white-space: pre-wrap;
    }
    .progress-container {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .status-running {
        background-color: #ffff99;
    }
    .status-success {
        background-color: #d4edda;
    }
    .status-error {
        background-color: #f8d7da;
    }
    .nav-link {
        cursor: pointer;
        text-decoration: none;
        color: #4b7bec;
        font-weight: bold;
    }
    .nav-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'terminal_output' not in st.session_state:
    st.session_state.terminal_output = ""
if 'process_running' not in st.session_state:
    st.session_state.process_running = False
if 'docker_running' not in st.session_state:
    st.session_state.docker_running = False
if 'current_process' not in st.session_state:
    st.session_state.current_process = None
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = None
if 'progress_status' not in st.session_state:
    st.session_state.progress_status = ""
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0
if 'neo4j_container_id' not in st.session_state:
    st.session_state.neo4j_container_id = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Paper Extraction"

# Docker client
docker_client = docker.from_env()

def update_session_state(key, value):
    st.session_state[key] = value
    
def run_docker_neo4j():
    """Start the Neo4j Docker container"""
    try:
        # Check if container is already running
        containers = docker_client.containers.list(filters={"name": "neo4j-research"})
        if containers:
            st.session_state.docker_running = True
            st.session_state.neo4j_container_id = containers[0].id
            return "Neo4j container is already running."
        
        # Run new container
        container = docker_client.containers.run(
            "neo4j:latest",
            name="neo4j-research",
            ports={'7474/tcp': 7474, '7687/tcp': 7687},
            volumes={
                f"{os.getcwd()}/neo4j/data": {'bind': '/data', 'mode': 'rw'},
                f"{os.getcwd()}/neo4j/import": {'bind': '/import', 'mode': 'rw'}
            },
            environment=["NEO4J_AUTH=neo4j/research123"],
            detach=True
        )
        
        st.session_state.docker_running = True
        st.session_state.neo4j_container_id = container.id
        return "Neo4j Docker container started successfully."
    
    except docker.errors.APIError as e:
        st.session_state.docker_running = False
        return f"Error starting Neo4j Docker container: {str(e)}"

def stop_docker_neo4j():
    """Stop the Neo4j Docker container"""
    try:
        if st.session_state.neo4j_container_id:
            container = docker_client.containers.get(st.session_state.neo4j_container_id)
            container.stop()
            container.remove()
            st.session_state.docker_running = False
            st.session_state.neo4j_container_id = None
            return "Neo4j Docker container stopped."
        return "No Neo4j container running."
    except Exception as e:
        return f"Error stopping Neo4j container: {str(e)}"

def parse_tqdm_output(line):
    """Parse tqdm progress bar output to extract progress information"""
    # Match patterns like: "23%|██▊       | 23/100 [00:05<00:18,  4.15it/s]"
    match = re.search(r'(\d+)%\|.*?\| (\d+)/(\d+)', line)
    if match:
        percent = int(match.group(1))
        current = int(match.group(2))
        total = int(match.group(3))
        return percent / 100, f"Processing {current}/{total}"
    return None, None

def stream_output(process, terminal_placeholder, progress_bar, progress_text):
    """Stream the output from the subprocess to the Streamlit terminal"""
    # Reset terminal output in session state only
    st.session_state.terminal_output = ""
    
    while True:
        # Read output line by line
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break
        
        if output_line:
            # Remove ANSI escape sequences (used for terminal formatting)
            clean_line = re.sub(r'\x1B\[[0-9;]*[mK]', '', output_line.strip())
            
            # Update ONLY the session state - don't try to update UI directly from thread
            st.session_state.terminal_output += clean_line + "\n"
            
            # Try to extract progress information from tqdm output
            progress_value, status_text = parse_tqdm_output(clean_line)
            if progress_value is not None:
                # Just update the session state values
                st.session_state.progress_value = progress_value
                st.session_state.progress_status = status_text
    
    # Process has completed
    st.session_state.process_running = False
    
    # Get return code
    return_code = process.poll()
    if return_code == 0:
        st.session_state.terminal_output += "\n✅ Process completed successfully.\n"
    else:
        st.session_state.terminal_output += f"\n❌ Process failed with return code {return_code}.\n"

def run_command(command, terminal_placeholder, progress_bar, progress_text):
    """Run a command and stream its output to the terminal"""
    if st.session_state.process_running:
        st.warning("A process is already running. Please wait for it to complete.")
        return
    
    # Reset terminal output
    st.session_state.terminal_output = f"Running command: {' '.join(command)}\n\n"
    terminal_placeholder.text(st.session_state.terminal_output)
    
    # Reset progress
    progress_bar.progress(0)
    progress_text.text("Starting process...")
    
    try:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        st.session_state.process_running = True
        st.session_state.current_process = process
        
        # Stream output in a separate thread
        thread = threading.Thread(
            target=stream_output, 
            args=(process, terminal_placeholder, progress_bar, progress_text),
            daemon=True
        )
        thread.start()
        
        # Enable auto-refresh while process is running
        timed_auto_refresh()
        
    except Exception as e:
        st.session_state.terminal_output += f"Error starting process: {str(e)}\n"
        terminal_placeholder.text(st.session_state.terminal_output)
        st.session_state.process_running = False

def stop_current_process():
    """Stop the currently running process"""
    if st.session_state.process_running and st.session_state.current_process:
        st.session_state.current_process.terminate()
        st.session_state.process_running = False
        return "Process terminated."
    return "No process running."

def get_papers_count():
    """Get the count of extracted papers"""
    pdf_count = len(glob.glob("papers/*.pdf"))
    json_count = len(glob.glob("papers_summary/*.json"))
    return pdf_count, json_count

def get_figures_count():
    """Get the count of extracted figures"""
    return len(glob.glob("output/images/*.jpg"))

def display_papers_data():
    """Display information about the extracted papers"""
    pdf_count, json_count = get_papers_count()
    figures_count = get_figures_count()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PDF Files", pdf_count)
    with col2:
        st.metric("JSON Metadata Files", json_count)
    with col3:
        st.metric("Extracted Figures", figures_count)
    
    # Show papers list
    if json_count > 0:
        st.subheader("Extracted Papers")
        
        papers_data = []
        for json_file in glob.glob("papers_summary/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    basic_info = data.get("basic_info", {})
                    papers_data.append({
                        "Paper ID": basic_info.get("paper_id", "Unknown"),
                        "Title": basic_info.get("title", "Unknown"),
                        "Year": basic_info.get("published_year", "Unknown"),
                        "Authors": ", ".join(basic_info.get("authors", ["Unknown"]))[:50] + "..." if len(", ".join(basic_info.get("authors", ["Unknown"]))) > 50 else ", ".join(basic_info.get("authors", ["Unknown"])),
                        "References": len(basic_info.get("references", [])),
                        "Chunks": len(data.get("raw_chunks", [])),
                        "Figures": len(data.get("figures", []))
                    })
            except Exception as e:
                st.error(f"Error loading {json_file}: {str(e)}")
        
        if papers_data:
            df = pd.DataFrame(papers_data)
            st.dataframe(df, use_container_width=True)

def display_review_topics():
    """Display information about available review topics"""
    try:
        if os.path.exists("review_topics.json"):
            with open("review_topics.json", 'r', encoding='utf-8') as f:
                topics = json.load(f)
            
            st.subheader("Available Review Topics")
            for i, topic in enumerate(topics):
                with st.expander(f"{i+1}. {topic.get('title', 'Untitled Topic')}"):
                    st.write(f"**Focus:** {topic.get('focus', 'No focus defined')}")
                    st.write("**Themes:**")
                    for theme in topic.get('themes', []):
                        st.write(f"- {theme}")
                    st.write("**Methods:**")
                    for method in topic.get('methods', []):
                        st.write(f"- {method}")
        else:
            st.info("No review topics found. Run knowledge_base.py to generate topics.")
    except Exception as e:
        st.error(f"Error loading review topics: {str(e)}")

# Sidebar navigation
st.sidebar.title("Research Paper System")
st.sidebar.image("https://img.icons8.com/color/96/000000/literature.png", width=100)

# Navigation
def set_active_tab(tab_name):
    st.session_state.active_tab = tab_name

# Navigation buttons
st.sidebar.markdown("### Navigation")
if st.sidebar.button("Paper Extraction", key="nav_extraction"):
    set_active_tab("Paper Extraction")
if st.sidebar.button("Knowledge Base", key="nav_kb"):
    set_active_tab("Knowledge Base")
if st.sidebar.button("Review Generation", key="nav_review"):
    set_active_tab("Review Generation")
if st.sidebar.button("System Status", key="nav_status"):
    set_active_tab("System Status")

# Sidebar status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

# Docker status
docker_status = "✅ Running" if st.session_state.docker_running else "❌ Stopped"
st.sidebar.markdown(f"**Neo4j Docker:** {docker_status}")

# Process status
process_status = "🔄 Running" if st.session_state.process_running else "⏹️ Idle"
st.sidebar.markdown(f"**Current Process:** {process_status}")

# Paper counts
pdf_count, json_count = get_papers_count()
st.sidebar.markdown(f"**Extracted Papers:** {json_count}")
st.sidebar.markdown(f"**PDF Files:** {pdf_count}")
st.sidebar.markdown(f"**Figures:** {get_figures_count()}")

# Main content based on active tab
if st.session_state.active_tab == "Paper Extraction":
    st.title("📄 Paper Extraction")
    st.markdown("""
    Extract research papers from arXiv based on your query. This will download PDFs, 
    extract text, figures, and metadata, and prepare them for the knowledge base.
    """)
    
    # Input form
    with st.form("extraction_form"):
        query = st.text_input("Research Query", placeholder="e.g., Single image to 3d")
        col1, col2 = st.columns(2)
        with col1:
            initial_results = st.number_input("Initial Results", min_value=1, max_value=10, value=3)
        with col2:
            max_depth = st.number_input("Max Depth", min_value=1, max_value=3, value=2)
        
        col3, col4 = st.columns(2)
        with col3:
            max_per_level_1 = st.number_input("Max Papers (Level 1)", min_value=1, max_value=5, value=2)
        with col4:
            max_per_level_2 = st.number_input("Max Papers (Level 2)", min_value=1, max_value=5, value=1)
        
        submitted = st.form_submit_button("Start Extraction")
    
    # Create terminal output area
    terminal_placeholder = st.empty()
    
    # Progress indicators
    progress_container = st.container()
    with progress_container:
        progress_text = st.empty()
        progress_bar = st.progress(0)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Stop Extraction", key="stop_extraction"):
            result = stop_current_process()
            st.info(result)
    
    # Run extraction process
    if submitted:
        command = [
            "python", "papers_extractor_bfs.py",
            "--query", query,
            "--initial_results", str(initial_results),
            "--max_depth", str(max_depth),
            "--max_per_level", f"{max_per_level_1},{max_per_level_2}"
        ]
        run_command(command, terminal_placeholder, progress_bar, progress_text)
    else:
        terminal_placeholder.text(st.session_state.terminal_output)
        progress_bar.progress(st.session_state.progress_value)
        progress_text.text(st.session_state.progress_status)
    
    # Show extracted papers
    display_papers_data()

elif st.session_state.active_tab == "Knowledge Base":
    st.title("🧠 Knowledge Base Creation")
    st.markdown("""
    This step creates a knowledge base from the extracted papers using Neo4j and ChromaDB. 
    The knowledge base is used for generating review papers.
    """)
    
    # Neo4j Docker controls
    st.subheader("Step 1: Start Neo4j Database")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Neo4j Docker", key="start_docker"):
            result = run_docker_neo4j()
            st.info(result)
    with col2:
        if st.button("Stop Neo4j Docker", key="stop_docker"):
            result = stop_docker_neo4j()
            st.info(result)
    
    # Neo4j status
    st.info("Neo4j Status: " + ("Running ✅" if st.session_state.docker_running else "Stopped ❌"))
    
    # Knowledge Base Creation
    st.subheader("Step 2: Create Knowledge Base")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update Metadata", key="update_metadata"):
            terminal_placeholder = st.empty()
            progress_container = st.container()
            with progress_container:
                progress_text = st.empty()
                progress_bar = st.progress(0)
            
            command = ["python", "update_metadata.py"]
            run_command(command, terminal_placeholder, progress_bar, progress_text)
    
    with col2:
        if st.button("Build Knowledge Base", key="build_kb"):
            if not st.session_state.docker_running:
                st.error("Neo4j database is not running. Please start it first.")
            else:
                terminal_placeholder = st.empty()
                progress_container = st.container()
                with progress_container:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                
                command = ["python", "knowledge_base.py"]
                run_command(command, terminal_placeholder, progress_bar, progress_text)
    
    # Terminal output
    st.subheader("Process Output")
    terminal_placeholder = st.empty()
    terminal_placeholder.text(st.session_state.terminal_output)
    
    # Progress indicators
    progress_container = st.container()
    with progress_container:
        progress_text = st.empty()
        progress_text.text(st.session_state.progress_status)
        progress_bar = st.progress(st.session_state.progress_value)
    
    if st.button("Stop Current Process", key="stop_kb_process"):
        result = stop_current_process()
        st.info(result)
    
    # Show review topics if available
    display_review_topics()

elif st.session_state.active_tab == "Review Generation":
    st.title("📝 Review Paper Generation")
    st.markdown("""
    Generate a review paper based on the knowledge base. Select a query or use one of the generated topics.
    """)
    
    # Check if knowledge base exists
    kb_missing = not (os.path.exists("./chroma_db") and os.path.exists("./review_topics.json"))
    if kb_missing:
        st.warning("Knowledge base not found. Please complete the 'Knowledge Base Creation' step first.")
    
    # Query options
    query_option = st.radio(
        "Select query option:",
        ["Use generated topic", "Custom query"]
    )
    
    if query_option == "Use generated topic":
        try:
            if os.path.exists("review_topics.json"):
                with open("review_topics.json", 'r', encoding='utf-8') as f:
                    topics = json.load(f)
                
                topic_titles = [topic.get('title', f"Topic {i}") for i, topic in enumerate(topics)]
                selected_topic_idx = st.selectbox("Select a topic:", range(len(topic_titles)), format_func=lambda i: topic_titles[i])
                
                selected_topic = topics[selected_topic_idx]
                st.info(f"Focus: {selected_topic.get('focus', 'No focus defined')}")
                
                query = selected_topic.get('focus', "")
            else:
                st.error("No review topics found. Run knowledge_base.py to generate topics.")
                query = ""
        except Exception as e:
            st.error(f"Error loading review topics: {str(e)}")
            query = ""
    else:
        query = st.text_input("Enter custom query:")
    
    # Interactive mode option
    interactive_mode = st.checkbox("Interactive mode", value=True, help="If checked, you'll be asked to approve each section.")
    
    # Generate paper
    generate_paper = st.button("Generate Review Paper", disabled=kb_missing or not query)
    
    if generate_paper:
        st.subheader("Generating Review Paper...")
        
        terminal_placeholder = st.empty()
        progress_container = st.container()
        with progress_container:
            progress_text = st.empty()
            progress_bar = st.progress(0)
        
        command = ["python", "main.py", "--query", '"' +query + '"', "--continue"]
        if not interactive_mode:
            command.append("--non-interactive")
        
        run_command(command, terminal_placeholder, progress_bar, progress_text)
    else:
        # Display existing terminal output
        terminal_placeholder = st.empty()
        terminal_placeholder.text(st.session_state.terminal_output)
        
        progress_container = st.container()
        with progress_container:
            progress_text = st.empty()
            progress_text.text(st.session_state.progress_status)
            progress_bar = st.progress(st.session_state.progress_value)
    
    # Control buttons
    if st.button("Stop Generation", key="stop_generation"):
        result = stop_current_process()
        st.info(result)
    
    # Show generated papers
    st.subheader("Generated Papers")
    output_path = Path("output")
    
    if output_path.exists():
        tex_files = list(output_path.glob("*.tex"))
        
        if tex_files:
            for tex_file in tex_files:
                with st.expander(f"Review Paper: {tex_file.name}"):
                    try:
                        with open(tex_file, 'r', encoding='utf-8') as f:
                            tex_content = f.read()
                        st.code(tex_content, language="latex")
                        
                        # Download button
                        st.download_button(
                            label="Download LaTeX file",
                            data=tex_content,
                            file_name=tex_file.name,
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error loading {tex_file.name}: {str(e)}")
        else:
            st.info("No generated papers found yet.")
    else:
        st.info("Output directory not found.")

elif st.session_state.active_tab == "System Status":
    st.title("🔄 System Status")
    
    # System directories status
    st.subheader("Directory Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PDF Files", len(glob.glob("papers/*.pdf")))
    with col2:
        st.metric("JSON Metadata Files", len(glob.glob("papers_summary/*.json")))
    with col3:
        st.metric("Extracted Figures", len(glob.glob("output/images/*.jpg")))
    
    # Database status
    st.subheader("Database Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        neo4j_status = "Running" if st.session_state.docker_running else "Stopped"
        st.metric("Neo4j Database", neo4j_status)
        if st.button("Start Neo4j"):
            result = run_docker_neo4j()
            st.info(result)
        if st.button("Stop Neo4j"):
            result = stop_docker_neo4j()
            st.info(result)
    
    with col2:
        chroma_status = "Created" if os.path.exists("./chroma_db") else "Not Created"
        st.metric("Chroma Vector DB", chroma_status)
    
    # Running processes
    st.subheader("Process Status")
    
    if st.session_state.process_running:
        st.warning(f"Process running: {st.session_state.current_process}")
        if st.button("Stop Running Process"):
            result = stop_current_process()
            st.info(result)
    else:
        st.success("No processes currently running")
    
    # View logs
    st.subheader("Terminal Output")
    st.code(st.session_state.terminal_output, language="bash")
    
    # Clear logs button
    if st.button("Clear Terminal Output"):
        st.session_state.terminal_output = ""
        st.success("Terminal output cleared")