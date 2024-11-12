import json
import logging
import requests
import numpy as np
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

def load_scaling_params(path_to_params: Path):
    """
    path_to_params: Path to json file that contains scaling parameters
    
    Returns: params dict: A dictionary with keys 'data_min', 'data_max', 'scaled_mean', and 'scaled_std', 
        e.g., {"data_min": 2.8, "data_max": 11.2, "scaled_mean": 56.39, "scaled_std": 39.03}.
    """

    with open(path_to_params, 'r') as f:
        params = json.load(f)
    return params

def get_scaled_score(unscaled_score, params):
    """
    unscaled_score: QScore integer value without any scale applied to it
    params: dictionary containing relevant parameters for scale for that task

    Returns: scaled_score: integer representing QScore, with range of 1-100
    """
    scaled_score = ((unscaled_score - params['data_min']) / (params['data_max'] - params['data_min'])) * 100
    scaled_score = int(np.clip(scaled_score, 1, 100))
    return scaled_score

def post_score(task_name: str, metric_name: str, result: int):
    """
    task_name: Name of task, either objnam, motor, or invalid if error was encountered
    metric_name: Name of metric to post, either q_score or compliance_score

    Returns: Status code of post request
    """
    if task_name == "invalid":
        response = post_score('', 'q_score', -1)
        response = post_score('', 'compliance_score', -1)
        return response
    else:
        data = {
            metric_name: result
        }

        url = f"http://localhost:5000/data/{metric_name}"

        response = requests.post(url, json=data)

        return response.status_code