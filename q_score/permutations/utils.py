import logging
import requests
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
        logging.warning("Task name not objnam or motor, defaulting to invalid")
        task_name = "invalid"

    return series_number, task_name

def post_score(series_number: int, task_name: str, metric_name: str, result: int):
    """
    series_number: the series number the score is associated with
    task_name: Name of task, either objnam, motor, or invalid if error was encountered
    metric_name: Name of metric to post, either q_score or compliance_score

    Returns: Status code of post request
    """
    if task_name == "invalid":
        response = post_score('ERROR', '', 'q_score', -1)
        response = post_score('ERROR', '', 'compliance_score', -1)
        return response
    else:
        data = {
            'series_number': series_number,
            metric_name: result
        }

        url = f"http://localhost:5000/data/{metric_name}"
        logging.critical(f'Sending {data} to {url}')

        response = requests.post(url, json=data)
        logging.critical(f'got response {response}')

        return response.status_code