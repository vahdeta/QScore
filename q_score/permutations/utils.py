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
    series_description = nifti_file_name.split("_")
    task_name = series_description[1].lower()

    # Make sure that task name is either objnam or motor
    if task_name not in ["objnam", "motor"]:
        logging.error("Task name not objnam or motor, defaulting to objnam")
        return "error"

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
    scaled_score = round(np.clip(scaled_score, 1, 100))
    return scaled_score

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
            "task_name": "invalid",
            "q_score": 0
        }
    else:
        data = {
            "task_name": task_name,
            metric_name: result
        }

    response = requests.post(url, json=data)

    return response.status_code