import os
import logging
import pydicom
import requests
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_series_description(nifti_path: Path):
    """
    nifti_path: Path to nifti file
    
    Returns: task_name: str
    """

    nifti_file_name = nifti_path.name
    series_number, series_description = nifti_file_name.split("_", 1)
    lc_series_description = series_description.lower()

    # Make sure that task name is either objnam or motor
    if "motor" in lc_series_description:
        task_name = "motor"
    elif "objnam" in lc_series_description:
        task_name = "objnam"
    else:
        logging.warning("Task name not objnam or motor, defaulting to objnam")
        task_name = "objnam"

    return task_name

def post_score(task_name: str, metric_name: str, result: int):
    """
    task_name: Name of task, either objnam, motor, or invalid if error was encountered
    metric_name: Name of metric to post, either q_score or compliance_score

    Returns: Status code of post request
    """

    url = f"http://localhost:5000/data/{metric_name}"

    data = {}
    if task_name == "invalid":
        data = {
            "q_score": 0
        }
    else:
        data = {
            metric_name: result
        }

    response = requests.post(url, json=data)

    return response.status_code