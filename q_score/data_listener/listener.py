import os
import json
import time
import queue
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define a queue for file paths to be analyzed
analysis_queue = queue.Queue()

def process_json_file(file_path):
    """
    Process a JSON file containing a list of DICOM file paths to be analyzed

    Args:
        file_path: Path to JSON file
    """

    try:
        with open(file_path, 'r') as json_file:

            data = json.load(json_file)

            # Get unique parent directories from the file paths
            parent_directories = list(set([Path(file_path).parent for file_path in data['dicoms']]))

            # Add the file paths to the analysis queue
            for parent_directory in parent_directories:
                analysis_queue.put(parent_directory)

            # Remove the JSON file
            os.remove(file_path)

    except Exception as e:
        print(f"Error processing JSON file {file_path}: {str(e)}")

# Define a custom event handler to watch for file creation in /incoming
class JSONFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.json'):
            print("NEW JSON FILE DETECTED")
            process_json_file(event.src_path)

def start_listener(json_directory: Path):
    """
    Start listening for JSON files in the incoming directory and add them to the analysis queue

    Args:
        json_directory: Path contaning to where JSON files will be written
    """

    event_handler = JSONFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=json_directory, recursive=False)
    observer.start()
    
    try:
        while True:
            # Check if there are files in the analysis queue to process
            if not analysis_queue.empty():  

                # Get the dicom directory path to analyze
                paths_to_analyze = analysis_queue.get()

                # Make call to analysis script
                call_analysis_script(paths_to_analyze)

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def call_analysis_script(dicom_file_path):
    """
    Call the analysis script to analyze a DICOM file

    Args:
        dicom_file_path: Path to DICOM file to be analyzed
    """

    print(f"Analyzing {dicom_file_path}")

    q_score_path = str(os.environ.get("QSCORE_PATH", '/app/qscore'))
    path_to_script = f'{q_score_path}/run_analysis.py'

    command = ["python3", path_to_script, dicom_file_path]

    # Run command
    try:
        print("RUNNING COMMAND")
        subprocess.run(command, text=True)
        print("AFTER COMMAND COMPLETION")
    except:
        raise Exception(f"Error running command {command}")
