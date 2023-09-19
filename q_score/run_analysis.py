import os
from typing import List, Optional
import uuid
import logging
import argparse
from pathlib import Path
from q_score.permutations.permutations import Permutations
from q_score.permutations.utils import read_dicoms, post_q_score


def run(
    dicom_data,
    *,
    task_type: str = "object_naming",
    condition: str = "CON1",
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
    dicom_read_status = read_dicoms(dicom_data["Dicoms"], output_directory)

    if dicom_read_status is not None:
        logging.error("Error reading dicoms")
        exit(1)

    # base folder is current directory
    base_folder = Path(os.environ.get("QSCORE_PATH", "/app/q_score"))

    # Start permutation analysis
    permutations = Permutations(
        base_folder=base_folder,
        output_data_path=output_directory,
        task_type=task_type,
    )

    permutations.start_permutations()
    q_score = permutations.get_q_score()

    # Send q score to localhost for other Docker container to pick up
    logging.info(f"Permutations complete. Got q score of : {q_score}")

    # Send q score to localhost for other Docker container to pick up
    logging.info("Sending q score to localhost")
    post_q_score(dicom_data["SeriesNumber"], q_score)

    # Remove output directory
    os.system(f"rm -rf {output_directory}")
