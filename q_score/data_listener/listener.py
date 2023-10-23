import os
import json
import time
import queue
import logging
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from q_score import run_analysis

# Define a queue for file paths to be analyzed
analysis_queue = queue.Queue()

# Set up logging
logging.basicConfig(level=logging.INFO)

def process_json_file(file_path):
    """
    Process a JSON file containing a list of DICOM file paths to be analyzed

    Args:
        file_path: Path to JSON file
    """

    try:
        with open(file_path, 'r') as json_file:

            data = json.load(json_file)
            analysis_queue.put(data)

            # Remove the JSON file
            os.remove(file_path)

    except Exception as e:
        logging.error(f"Error processing JSON file {file_path}: {str(e)}")

# Define a custom event handler to watch for file creation in /incoming
class JSONFileHandler(FileSystemEventHandler):
    """
    Watchdog class for handling events in JSON listener directory
    """            
    def on_closed(self, event):
        """
        Handle file creation events for JSONs
        """

        if event.is_directory:
            return None
        if event.src_path.endswith('.json'):
            logging.info("New JSON file detected")
            process_json_file(event.src_path)

def start_listener(json_directory: Path):
    """
    Start listening for JSON files in the incoming directory and add them to the analysis queue

    Args:
        json_directory: Path containing to where JSON files will be written
    """

    event_handler = JSONFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=json_directory, recursive=False)
    observer.start()
    
    logging.info(f"Started listening for JSON files in {json_directory}")

    try:
        while True:
            # Check if there are files in the analysis queue to process
            if not analysis_queue.empty():  

                # Get the dicom directory path to analyze
                data_to_analyze = analysis_queue.get()

                logging.info(f"Analyzing dicom files for series: {data_to_analyze['SeriesNumber']}")

                # Make call to analysis script
                run_analysis.run(data_to_analyze)

                logging.info(f"Finished analysis for series {data_to_analyze['SeriesNumber']}")

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
