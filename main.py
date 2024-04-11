from q_score.data_listener import listener
from pathlib import Path
import os

if __name__ == "__main__":
    
    # File location for listening for incoming nifti files
    nifti_directory = Path(os.environ.get("NIFTI_DIR", "/incoming"))

    listener.start_listener(nifti_directory)
