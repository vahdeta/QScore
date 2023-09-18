import os
import logging
import requests
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def read_dicoms(input_directory: Path, output_directory: Path):
    """
    input_directory: Path to directory containing DICOM files
    output_directory: Path to directory where NIFTI files will be output

    Returns: None if successful, error message if not
    """

    # Make sure output directory exists
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    logging.info("Running dcm2niix on dicom path")
    # Convert dicoms to NIFTI and output to directory
    command = f"dcm2niix -o {output_directory} {input_directory}"

    # Run command
    try:
        subprocess.run(command, shell=True, check=True)
    except:
        dicom_processing_error = "Error running dcm2niix command"
        return dicom_processing_error
    
    return None

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

def post_q_score(series_number: str, q_score: int):
    """
    series_number: Series number that data originated from
    q_scr: Q score to be posted

    Returns: Status code of post request
    """

    url = "http://localhost:5000/data/q_score"
    data = {
        "SeriesNumber": series_number,
        "QScore": q_score
    }

    response = requests.post(url, json=data)

    return response.status_code