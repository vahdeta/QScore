import os
import uuid
import logging
import argparse
from pathlib import Path
from permutations.permutations import Permutations
from permutations.utils import read_dicoms, post_q_score

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run permutation analysis on a design file')
    parser.add_argument('dicom_data_path', type=str, help='Path to data file')
    parser.add_argument('--task_type', default="object_naming", type=str, help='Type of task to run (motor or object_naming)')
    parser.add_argument('--condition', default="CON1", type=str, help='Condition to run')
    
    # Parse arguments
    args = parser.parse_args()

    # The input directory will be the directory that contains the DICOM files
    input_directory = Path(args.dicom_data_path)

    # Set logging level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # This output directory will need to have a randomly generated name
    session_uuid = uuid.uuid4()
    output_directory = Path(f"/tmp/{session_uuid}")
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Set output directory: {output_directory}")

    # Read dicoms to NIFTI format
    dicom_read_status = read_dicoms(input_directory, output_directory)

    if dicom_read_status is not None:
        logging.error("Error reading dicoms")
        exit(1)

    # base folder is current directory
    base_folder = Path(os.environ.get("QSCORE_PATH", "/app/q_score"))

    # Start permutation analysis
    permutations = Permutations(
                    base_folder = base_folder,
                    output_data_path = output_directory,
                    task_type= args.task_type
                )
    
    permutations.start_permutations()
    q_score = permutations.get_q_score()

    # Send q score to localhost for other Docker container to pick up
    logging.info(f"Permutations complete. Got q score of : {q_score}")

    # Send q score to localhost for other Docker container to pick up
    logging.info("Sending q score to localhost")
    post_q_score(q_score, args.condition)

    # Remove output directory
    os.system(f"rm -rf {output_directory}")


        