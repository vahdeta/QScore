import os
import time
import uuid
import logging
import argparse
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from q_score.permutations.permutations import Permutations
from q_score.permutations.utils import convert_dicoms, get_series_description, post_score


def run(
    dicom_data
):
    # Set logging level
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # This output directory will need to have a randomly generated name
    session_uuid = uuid.uuid4()
    output_directory = Path(f"/tmp/{session_uuid}")
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Set output directory: {output_directory}")

    # Read dicoms to NIFTI format
    dicom_read_status = convert_dicoms(dicom_data["Dicoms"], output_directory)

    if dicom_read_status is not None:
        logging.error("Error reading dicoms")
        exit(1)

    # Get the Series Description from the DICOM file
    task_name = get_series_description(dicom_data["Dicoms"])

    # base folder is current directory
    base_folder = Path(os.environ.get("QSCORE_PATH", "/app/q_score"))

    # Start permutation analysis
    permutations = Permutations(
                    base_folder = base_folder,
                    output_data_path = output_directory,
                    task_type= task_name
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

        futures[executor.submit(permutations.get_compliance_score)] = "compliance_score"

        for future in as_completed(futures):
            # Get the name of the function that was run
            metric = futures[future]

            if metric == "q_score":
                # Still need to compute the Q score
                q_score = permutations.get_q_score()
                logging.info(f"Q score: {q_score}")
                post_score(task_name, metric, q_score)
            elif metric == "compliance_score":
                compliance_score = future.result()
                logging.info(f"Compliance score: {compliance_score}")
                post_score(task_name, metric, compliance_score)

    # Remove output directory
    os.system(f"rm -rf {output_directory}")
