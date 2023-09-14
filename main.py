"""
Script for entrance point to q_score application
"""
from q_score.data_listener import listener
from pathlib import Path
import os
import subprocess

if __name__ == "__main__":
    
    # File location for listening for JSO
    # TODO: make this an environment variable derived from Dockerfile
    json_directory = Path(os.environ.get("JSON_DIR", "/incoming"))

    listener.start_listener(json_directory)
