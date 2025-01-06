import os
import time
import uuid
import logging
import argparse
from pathlib import Path
from nipype.interfaces import fsl
from concurrent.futures import ThreadPoolExecutor, as_completed
from permutations.permutations import Permutations
from permutations.utils import post_score

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run permutation analysis on a design file')
    parser.add_argument('nifti_data_path', type=str, help='Path to data file')
    parser.add_argument('--task_type', default="object_naming", type=str, help='Type of task to run (motor or object_naming)')
    parser.add_argument('--condition', default="CON1", type=str, help='Condition to run')
    
    # Parse arguments
    args = parser.parse_args()

    # Input file is a nifti file
    nifti_file = Path(args.nifti_data_path)

    # Set logging level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # This output directory will need to have a randomly generated name
    session_uuid = uuid.uuid4()
    output_directory = Path(f"/tmp/{session_uuid}")
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Set output directory: {output_directory}")

    # base folder is current directory
    base_folder = Path(os.environ.get("QSCORE_PATH", "/app/q_score"))

    # Start permutation analysis
    try:
        permutations = Permutations(
                        base_folder = base_folder,
                        output_data_path = output_directory,
                        original_nifti = nifti_file
                    )

        permutations.start_analysis()

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks and keep track of corresponding futures
            futures[executor.submit(permutations.run_feat, 
                                permutations.num_frames,
                                permutations.tr_time,
                                permutations.output_data_path / "truncated_bet_mcf.nii.gz",
                                permutations.design_file_path,
                                permutations.analysis_path
                            )] = "q_score"

            for future in as_completed(futures):
                # Still need to compute the Q score
                q_score = permutations.get_q_score()
                logging.info(f"Q score: {q_score}")
                post_score(permutations.series_number, q_score)
    except Exception as e:
        logging.error(f"Error setting up permutations: {e}")
        post_score(permutations.series_number, -1)

    # Remove output directory
    os.system(f"rm -rf {output_directory}")