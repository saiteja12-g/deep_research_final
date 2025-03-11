import streamlit as st
import os
import sys
import json
import time
import base64
from pathlib import Path
import subprocess
from dotenv import load_dotenv

# Import the paper generator components
try:
    from review_writer import PaperGenerator, KnowledgeBaseConnector
    has_review_writer = True
except ImportError:
    has_review_writer = False

# Try to import the knowledge base
try:
    from knowledge_base import ResearchKnowledgeBase
    has_kb = True
except ImportError:
    has_kb = False

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Research Review Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    pass

# Initialize state
if 'review_started' not in st.session_state:
    st.session_state.review_started = False
if 'extraction_started' not in st.session_state:
    st.session_state.extraction_started = False
if 'extraction_complete' not in st.session_state:
    st.session_state.extraction_complete = False
if 'current_section' not in st.session_state:
    st.session_state.current_section = None
if 'paper_log' not in st.session_state:
    st.session_state.paper_log = []
if 'paper_state' not in st.session_state:
    st.session_state.paper_state = None
if 'display_final_paper' not in st.session_state:
    st.session_state.display_final_paper = False
if 'agent_output' not in st.session_state:
    st.session_state.agent_output = []
if 'extraction_progress' not in st.session_state:
    st.session_state.extraction_progress = 0
if 'latex_result' not in st.session_state:
    st.session_state.latex_result = None

# Header
def render_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://i.imgur.com/IeqQyWx.png", width=100)
    with col2:
        st.title("Research Review Paper Generator")
        st.markdown("Generate comprehensive review papers based on academic research")

# Functions
def run_paper_extraction(query):
    """Run the paper extraction process using papers_extractor_bfs.py"""
    st.session_state.paper_log.append(f"🔍 Starting paper extraction for query: '{query}'")
    st.session_state.extraction_started = True
    st.session_state.extraction_progress = 0
    
    try:
        # Run the extractor process
        command = [sys.executable, "papers_extractor_bfs.py"]
        
        # Create a process with the query as input
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Send the query to the process
        process.stdin.write(query + "\n")
        process.stdin.flush()
        
        # Process output in real-time
        log_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Track progress markers
        total_steps = 10  # Estimate of total steps
        current_step = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_line = output.strip()
                st.session_state.paper_log.append(log_line)
                
                # Update progress based on specific markers in the output
                if "Starting arXiv Reference Explorer" in log_line:
                    current_step = 1
                elif "Beginning BFS traversal" in log_line:
                    current_step = 3
                elif "Processing paper:" in log_line:
                    current_step += 0.2  # Increment for each paper
                elif "Completed BFS traversal" in log_line:
                    current_step = 9
                
                # Update progress
                progress = min(current_step / total_steps, 0.99)
                st.session_state.extraction_progress = progress
                progress_bar.progress(progress)
                
                # Update log display
                log_text = "\n".join(st.session_state.paper_log[-20:])  # Show last 20 lines
                log_placeholder.text_area("Extraction Log", value=log_text, height=400, disabled=True)
        
        # Finalize progress
        progress_bar.progress(1.0)
        st.session_state.extraction_progress = 1.0
        
        rc = process.poll()
        if rc != 0:
            error = process.stderr.read()
            st.session_state.paper_log.append(f"❌ Error in extraction process: {error}")
            return False
        
        st.session_state.paper_log.append("✅ Paper extraction complete!")
        st.session_state.extraction_complete = True
        return True
        
    except Exception as e:
        st.session_state.paper_log.append(f"❌ Error running extraction: {str(e)}")
        return False

def initialize_generators():
    """Initialize the knowledge base and paper generator"""
    if has_kb:
        kb = ResearchKnowledgeBase()