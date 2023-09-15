# Q Score Analysis

## Docker Build + Run

```bash
cd QScore
docker build -t . q_score_container
docker run -t -v path/to/incoming:/incoming --name q_score -e PYTHONUNBUFFERED=1 q_score_container
```


The Dockerfile expects a mount to /incoming, as two of the ENV variables point to it (JSON_DIR and DATA_DIR) 
* Note: if the JSON output is not going to be to /incoming and instead a different mount, please change the Dockerfile's JSON_DIR ENV variable and also make sure the location is passed in as a volume during the docker run

PYTHONUNBUFFERED is set to 1 to allow output to the console for debug purposes. 

## Usage

Once the container is running, you can start feeding it data. To kick off an analysis, you first need to place a JSON file (with the following format) into the local file location that /incoming is mounted to. This will trigger the JSON listener to start an analysis.

``` json
{
    "dicoms": [
        "/incoming/test/MR-0001.dcm",
        "/incoming/test/MR-0002.dcm",
        "/incoming/test/MR-0003.dcm"
     ]
}

```
The program will then generate a single NIFTI file from all of the /incoming/test DICOMs. Then, the rest of the permutations and feat analyses will occur on the NIFTI file.

The number of permutations is set in the q_score/permutations/permutations.py file. 

Eventually, the program outputs a q score which is then posted to "http://localhost:5000/data/q_score". 
