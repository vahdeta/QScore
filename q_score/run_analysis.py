import argparse
from permutations.utils import read_dicoms, post_q_score
from permutations.permutations import Permutations
from pathlib import Path
import uuid
import os

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run permutation analysis on a design file')
    parser.add_argument('dicom_data_path', type=str, help='Path to data file')
    parser.add_argument('--task_type', default="object_naming", type=str, help='Type of task to run (motor or object_naming)')
    parser.add_argument('--condition', default="CON1", type=str, help='Condition to run')
    
    # Parse arguments
    args = parser.parse_args()

    print("INSIDE OF RUN_ANALYSIS.PY")
    # The input directory will be the directory that contains the DICOM files
    input_directory = Path(args.dicom_data_path)

    # This output directory will need to have a randomly generated name
    print("Creating output directory")
    session_uuid = uuid.uuid4()
    output_directory = Path(f"/tmp/{session_uuid}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Read dicoms to NIFTI format
    print("Reading dicoms")
    dicom_read_status = read_dicoms(input_directory, output_directory)

    if dicom_read_status is not None:
        # TODO: send error message to logger
        print(dicom_read_status)
        exit(1)

    # base folder is current directory
    base_folder = Path(os.environ.get("QSCORE_PATH", "/app/q_score"))

    # Start permutation analysis
    permutations = Permutations(
                    base_folder = base_folder,
                    output_data_path = output_directory,
                    task_type= args.task_type
                )
    
    print("Starting permutations")
    permutations.start_permutations()
    print("Getting q score")
    q_score = permutations.get_q_score()

    # Send q score to localhost for other Docker container to pick up
    print(q_score)

    # Send q score to localhost for other Docker container to pick up
    post_q_score(q_score, args.condition)

    # Run ls -l command on output directory to make sure it exists
    os.system(f"ls -l {output_directory}")

    # Remove output directory
    os.system(f"rm -rf {output_directory}")


        