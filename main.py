from q_score.data_listener import listener
from pathlib import Path
import os

if __name__ == "__main__":
    
    # File location for listening for incoming json files
    json_directory = Path(os.environ.get("JSON_DIR", "/incoming"))

    listener.start_listener(json_directory)
