#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
import signal
import json
from pathlib import Path
from datetime import datetime

class ResearchWorkflow:
    """Manages the complete workflow for research paper processing and review generation."""
    
    def __init__(self):
        """Initialize the workflow manager."""
        self.processes = {}
        self.docker_running = False
        self.output_dir = "./output"
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(self.output_dir, f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
    def log(self, message, level="INFO", print_console=True):
        """Log a message to both log file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{level}] {timestamp} - {message}"
        
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
            
        if print_console:
            if level == "ERROR":
                print(f"\033[91m{message}\033[0m")  # Red text for errors
            elif level == "SUCCESS":
                print(f"\033[92m{message}\033[0m")  # Green text for success
            elif level == "WARNING":
                print(f"\033[93m{message}\033[0m")  # Yellow text for warnings
            else:
                print(message)
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        self.log("Checking dependencies...", print_console=True)
        
        # Check Python packages
        try:
            import arxiv
            import pymupdf
            import openai
            import tqdm
            import neo4j
            import chromadb
            import langchain_openai
            import dotenv
            import crewai
            self.log("All required Python packages are installed", level="SUCCESS")
        except ImportError as e:
            self.log(f"Missing Python package: {str(e)}", level="ERROR")
            self.log("Please install required packages using: pip install -r requirements.txt", level="ERROR")
            return False
            
        # Check if Docker is installed
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log("Docker is installed")
            else:
                self.log("Docker is not properly installed", level="ERROR")
                return False
        except FileNotFoundError:
            self.log("Docker is not installed", level="ERROR")
            self.log("Please install Docker: https://docs.docker.com/get-docker/", level="ERROR")
            return False
            
        # Check environment variables
        if not os.getenv("OPENAI_API_KEY"):
            self.log("OPENAI_API_KEY environment variable not found", level="ERROR")
            self.log("Please create a .env file with your OpenAI API key or set it in the environment", level="ERROR")
            return False
            
        return True
    
    def start_neo4j(self):
        """Start Neo4j Docker container."""
        self.log("Starting Neo4j Docker container...", print_console=True)
        
        try:
            # Check if container is already running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=neo4j-research", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            
            if "neo4j-research" in result.stdout:
                self.log("Neo4j container is already running", level="SUCCESS")
                self.docker_running = True
                return True
                
            # Start the container with appropriate settings
            subprocess.run([
                "docker", "run", "--name", "neo4j-research",
                "-p", "7474:7474", "-p", "7687:7687",
                "-e", "NEO4J_AUTH=neo4j/research123",
                "-d", "neo4j:latest"
            ], check=True)
            
            # Wait for Neo4j to start up
            self.log("Waiting for Neo4j to start (this may take a moment)...", print_console=True)
            time.sleep(10)  # Allow time for Neo4j to initialize
            
            self.log("Neo4j container started successfully", level="SUCCESS")
            self.docker_running = True
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to start Neo4j container: {str(e)}", level="ERROR")
            return False
    
    def stop_neo4j(self):
        """Stop and remove the Neo4j Docker container."""
        if not self.docker_running:
            return
            
        self.log("Stopping Neo4j Docker container...", print_console=True)
        
        try:
            # Stop the container
            subprocess.run(["docker", "stop", "neo4j-research"], check=True)
            
            # Remove the container
            subprocess.run(["docker", "rm", "neo4j-research"], check=True)
            
            self.log("Neo4j container stopped and removed", level="SUCCESS")
            self.docker_running = False
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to stop Neo4j container: {str(e)}", level="ERROR")
    
    def extract_papers(self, query, max_depth=2, max_papers=5, non_verbose=True):
        """Run paper extraction with arxiv_reference_explorer."""
        self.log(f"Starting paper extraction for query: '{query}'", print_console=True)
        
        try:
            # Create command with proper arguments
            cmd = [
                sys.executable, "papers_extractor_bfs.py",
                "--query", query
            ]
            
            if max_depth:
                cmd.extend(["--max-depth", str(max_depth)])
            
            if max_papers:
                cmd.extend(["--max-papers", str(max_papers)])
                
            # Use filter to reduce output
            env = os.environ.copy()
            if non_verbose:
                env["PYTHONIOENCODING"] = "utf-8"
                
            # Run the extraction process
            if non_verbose:
                # Capture output but only show progress
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                    bufsize=1
                )
                
                # Show only important updates
                for line in iter(process.stdout.readline, ""):
                    if any(x in line for x in ["Processing paper:", "Adding paper:", "Downloaded PDF", "Extracted", "Searching arXiv", "✓", "✗", "Error"]):
                        print(line.strip())
                    
                process.wait()
                return_code = process.returncode
            else:
                # Show all output
                return_code = subprocess.call(cmd, env=env)
            
            if return_code == 0:
                self.log("Paper extraction completed successfully", level="SUCCESS")
                return True
            else:
                self.log("Paper extraction failed", level="ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error during paper extraction: {str(e)}", level="ERROR")
            return False
    
    def load_data_to_db(self, non_verbose=True):
        """Load extracted paper data into the knowledge base."""
        self.log("Loading paper data into knowledge base...", print_console=True)
        
        try:
            # Create command
            cmd = [sys.executable, "knowledge_base.py"]
                
            # Run the process
            if non_verbose:
                # Capture output but only show important updates
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Show only important updates
                for line in iter(process.stdout.readline, ""):
                    if any(x in line for x in ["Using device:", "Papers in vector DB:", "Images in vector DB:", "Ingestion complete", "Successfully processed", "✓", "✗", "Error"]):
                        print(line.strip())
                    
                process.wait()
                return_code = process.returncode
            else:
                # Show all output
                return_code = subprocess.call(cmd)
            
            if return_code == 0:
                self.log("Data loading completed successfully", level="SUCCESS")
                return True
            else:
                self.log("Data loading failed", level="ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error during data loading: {str(e)}", level="ERROR")
            return False
    
    def generate_review_paper(self, query, interactive=True, continue_state=False, non_verbose=True):
        """Generate a review paper using the main.py script."""
        self.log(f"Generating review paper for query: '{query}'", print_console=True)
        
        try:
            # Create command with proper arguments
            cmd = [sys.executable, "main.py"]
            
            if query and not continue_state:
                cmd.extend(["--query", query])
                
            if not interactive:
                cmd.append("--non-interactive")
                
            if continue_state:
                cmd.append("--continue")
                
            # Run the process
            if non_verbose:
                # For paper generation, we don't want to lose important information
                # so we'll show more of the output
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Show output except very verbose status messages
                for line in iter(process.stdout.readline, ""):
                    # Filter out agent thinking logs but keep section updates
                    if not line.strip().startswith("Agent"):
                        print(line.strip())
                    
                process.wait()
                return_code = process.returncode
            else:
                # Show all output
                return_code = subprocess.call(cmd)
            
            if return_code == 0:
                self.log("Review paper generation completed successfully", level="SUCCESS")
                return True
            else:
                self.log("Review paper generation failed", level="ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error during review paper generation: {str(e)}", level="ERROR")
            return False
    
    def run_complete_workflow(self, query, interactive=True, skip_extraction=False, skip_loading=False, continue_writing=False, max_depth=2, max_papers=5, non_verbose=True):
        """Run the complete workflow from paper extraction to review generation."""
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False
            
        try:
            # Step 2: Start Neo4j (required for both loading and review generation)
            if not skip_loading or not skip_extraction:
                if not self.start_neo4j():
                    return False
            
            # Step 3: Extract papers (unless skipped)
            if not skip_extraction:
                if not self.extract_papers(query, max_depth, max_papers, non_verbose):
                    return False
            
            # Step 4: Load data into knowledge base (unless skipped)
            if not skip_loading:
                if not self.load_data_to_db(non_verbose):
                    return False
            
            # Step 5: Generate review paper
            if not self.generate_review_paper(query, interactive, continue_writing, non_verbose):
                return False
                
            self.log("Complete workflow executed successfully!", level="SUCCESS")
            return True
            
        finally:
            # Clean up: Stop Neo4j container if we started it
            if self.docker_running:
                self.stop_neo4j()
    
    def cleanup(self):
        """Clean up all resources."""
        # Stop Neo4j if running
        if self.docker_running:
            self.stop_neo4j()
            
        # Kill any remaining processes
        for proc_name, proc in self.processes.items():
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    self.log(f"Terminated process: {proc_name}")
                except:
                    pass

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Research Paper Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run complete workflow with interactive review writing
        python workflow.py --query "Advances in transformer models" --full-workflow
        
        # Extract papers only
        python workflow.py --query "Quantum computing algorithms" --extract-only
        
        # Load extracted papers to database only
        python workflow.py --load-only
        
        # Generate review with non-interactive mode (use defaults)
        python workflow.py --query "Neural networks" --generate-review --non-interactive
        
        # Continue writing a previously started paper
        python workflow.py --generate-review --continue
        """
    )
    
    # Main workflow options
    workflow_group = parser.add_argument_group("Workflow Options")
    workflow_group.add_argument("--full-workflow", action="store_true", help="Run the complete workflow from extraction to review")
    workflow_group.add_argument("--extract-only", action="store_true", help="Only run paper extraction")
    workflow_group.add_argument("--load-only", action="store_true", help="Only load papers to database")
    workflow_group.add_argument("--generate-review", action="store_true", help="Only generate review paper")
    
    # Query options
    query_group = parser.add_argument_group("Query Options")
    query_group.add_argument("--query", type=str, help="Research query for paper extraction and review")
    
    # Extraction options
    extract_group = parser.add_argument_group("Extraction Options")
    extract_group.add_argument("--max-depth", type=int, default=2, help="Maximum citation depth for paper extraction")
    extract_group.add_argument("--max-papers", type=int, default=5, help="Maximum papers per level for extraction")
    
    # Review generation options
    review_group = parser.add_argument_group("Review Options")
    review_group.add_argument("--non-interactive", action="store_true", help="Run review generation non-interactively")
    review_group.add_argument("--continue", dest="continue_state", action="store_true", help="Continue from saved state")
    
    # Output control options
    output_group = parser.add_argument_group("Output Control")
    output_group.add_argument("--verbose", action="store_true", help="Show all output (default: show only important updates)")
    
    # Skip options
    skip_group = parser.add_argument_group("Skip Options")
    skip_group.add_argument("--skip-extraction", action="store_true", help="Skip paper extraction step")
    skip_group.add_argument("--skip-loading", action="store_true", help="Skip loading papers to database")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full_workflow, args.extract_only, args.load_only, args.generate_review]):
        parser.error("One workflow option must be specified (--full-workflow, --extract-only, --load-only, or --generate-review)")
        
    if args.full_workflow or args.extract_only:
        if not args.query:
            parser.error("--query is required for paper extraction")
    
    if args.generate_review and not args.continue_state and not args.query:
        parser.error("--query is required for review generation (unless using --continue)")
    
    # Initialize workflow manager
    workflow = ResearchWorkflow()
    
    try:
        if args.full_workflow:
            workflow.run_complete_workflow(
                args.query,
                not args.non_interactive,
                args.skip_extraction,
                args.skip_loading,
                args.continue_state,
                args.max_depth,
                args.max_papers,
                not args.verbose
            )
        elif args.extract_only:
            # Start Neo4j and run extraction
            workflow.start_neo4j()
            workflow.extract_papers(args.query, args.max_depth, args.max_papers, not args.verbose)
        elif args.load_only:
            # Start Neo4j and load data
            workflow.start_neo4j()
            workflow.load_data_to_db(not args.verbose)
        elif args.generate_review:
            # Generate review paper
            workflow.start_neo4j()
            workflow.generate_review_paper(args.query, not args.non_interactive, args.continue_state, not args.verbose)
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
    finally:
        workflow.cleanup()

if __name__ == "__main__":
    main()