import json
import os
import logging
import shutil
from typing import List
import uuid
import pydicom
import requests
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)


def convert_dicoms(dicom_path_list: List[Path], output_directory: Path) -> str:
    """
    input_directory: Path to directory containing DICOM files
    output_directory: Path to directory where NIFTI files will be output

    Returns: None if successful, error message if not
    """

    # Generate a temporary directory to hold the dicoms for this series
    temp_dicom_dir = Path(f"/tmp/{uuid.uuid4()}")
    temp_dicom_dir.mkdir()

    for dicom_file_path in dicom_path_list:
        shutil.copy(dicom_file_path, temp_dicom_dir)

    # Make sure output directory exists
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    logging.info("Running dcm2niix on dicom path")
    # Convert dicoms to NIFTI and output to directory
    command = f"dcm2niix -o {output_directory} {temp_dicom_dir}"

    # Run command
    try:
        subprocess.run(command, shell=True, check=True)
    except:
        dicom_processing_error = "Error running dcm2niix command"
        return dicom_processing_error

    # Remove output directory
    os.system(f"rm -rf {temp_dicom_dir}")

    return None

def get_series_description(dicom_path_list: List[Path]):
    """
    input_directory: Path to directory containing DICOM files
    
    Returns: task_name: str
    """

    # Get the first dicom file in the directory
    dicom_file_path = dicom_path_list[0]

    # Read the DICOM file
    try:
        dicom_data = pydicom.dcmread(dicom_file_path)
    except:
        logging.error("Error reading DICOM file during series description extraction")
        raise Exception("Error reading DICOM file")

    series_description = dicom_data.SeriesDescription.split("_")
    task_name = series_description[1].lower()
    
    # Make sure that task name is either objnam or motor
    if task_name not in ["objnam", "motor"]:
        logging.error("Task name not objnam or motor, defaulting to objnam")
        task_name = "objnam"

    return task_name

def get_nifti_name(input_directory: Path):
    """
    input_directory: Path to directory containing NIFTI files

    Returns: Name of NIFTI file
    """
    # Get list of files in directory
    files = os.listdir(input_directory)

    # Get the nifti file from the directory
    # Just gets the first instance bc there should only be one
    try:
        nifti_file = [file for file in files if file.endswith(".nii")][0]
    except:
        logging.error("No NIFTI file found in directory")
        raise Exception("No NIFTI file found in directory")

    nifti_file_name = Path(nifti_file).name

    return nifti_file_name

def post_score(series_number: str, metric_name: str, result: int):
    """
    series_number: Series number that data originated from
    metric_name: Name of metric to post, either q_score or compliance_score

    Returns: Status code of post request
    """

    url = f"http://localhost:5000/data/{metric_name}"
    data = {
        "SeriesNumber": series_number,
        metric_name: result
    }

    response = requests.post(url, json=data)
    logging.info("Request Response: " + response.text)

    return response.status_code
