import math
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import distance
from scipy.interpolate import interp1d
import nibabel as nib
from nipype.interfaces import fsl
from q_score.permutations.utils import get_nifti_name

class Permutations:
    def __init__(self, base_folder: Path, output_data_path: Path, task_type: str):
        """
        base_folder: Path to folder to python package
        output_data_path: Path to output data directory, where FEAT and NIFTI files will be written
        task_type: String representing the type of task (e.g. object_naming, motor)
        num_cpus: Number of CPUs (or threads) to use for analysis
        """        

        self.base_folder = base_folder
        self.task_type = task_type
        self.design_file_path = self.base_folder / f"design/{self.task_type}/{self.task_type}.fsf"
        self.design_file_with_confound_path = self.base_folder / f"design/{self.task_type}/{self.task_type}_with_confounds.fsf"

        self.output_data_path = output_data_path
        self.analysis_path = self.output_data_path / "real"

        # Make sure analysis path directory exists
        if not self.analysis_path.exists():
            self.analysis_path.mkdir(parents=True, exist_ok=True)

        self.filtered_data = self.output_data_path / "real.feat/filtered_func_data"

        self.num_threads = 10
        self.iterations = int(os.environ['Q_SCORE_PERMUTATIONS'])
        self.percent_to_permute = .1
        self.contrast_num = 1

        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def start_permutations(self):
        """
        Method to start permutation thread pool and run feat
        """

        logging.info("Starting permutations")

        # Look into data file path and get the name of the nifti file
        nifti_name = get_nifti_name(self.output_data_path)

        # Load the original data as a nifti with nibabel
        original_image_file_path = self.output_data_path / nifti_name
        original_image = nib.load(original_image_file_path)
        tr_time = original_image.header['pixdim'][4]

        # Truncate the original data by 4 frames
        truncated_image = original_image.slicer[:,:,:,:-4]
        truncated_image_file_path = self.output_data_path / "truncated.nii.gz"
        nib.save(truncated_image, truncated_image_file_path)
        num_frames = truncated_image.shape[3]

        # Run brain extraction on truncated data
        bet_image_file_path = self.output_data_path / "truncated_bet.nii.gz"
        bet = fsl.BET()
        bet.inputs.in_file = truncated_image_file_path
        bet.inputs.out_file = bet_image_file_path
        bet.inputs.functional = True
        bet.run()

        # Run mcflirt on truncated data
        mcflirt_image_file_path = self.output_data_path / "truncated_bet_mcf.nii.gz"
        mcflirt = fsl.MCFLIRT()
        mcflirt.inputs.in_file = bet_image_file_path
        mcflirt.inputs.out_file = mcflirt_image_file_path
        mcflirt.run()

        # Run feat on the truncated data
        self.run_feat(num_frames, tr_time, mcflirt_image_file_path, self.design_file_path, self.analysis_path)

        num_permuted_frames = int(math.ceil(self.percent_to_permute * num_frames))

        # Start the permutations in a thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i in range(self.iterations):
                iteration_path = self.output_data_path / f"permutation_{i}"
                executor.submit(self.run_permutations_with_feat, num_permuted_frames, num_frames, tr_time, iteration_path)


    def run_permutations_with_feat(self, num_permuted_frames: int, num_frames: int, tr_time: float, iteration_path: Path):
        """
        Run permutation on single iteration with FSL feat

        Args:
            num_permuted_frames: Number of frames to permute
            num_frames: Number of frames in the data file
            tr_time: TR time of the data file
            iteration_path: Path to directory where permutation data will be written
        """

        logging.info(f"Running permutation for {iteration_path}")

        # Randomly permute the frames, and select a subset of the randomly selected frames
        random_frame_order = np.random.permutation(num_frames)
        frames_to_permute = random_frame_order[:num_permuted_frames]

        # Create confound vector
        confound_vector = np.zeros((num_frames, num_permuted_frames), dtype=int)

        # Set the confound vector to 1 for the frames that are permuted
        for f in range(len(frames_to_permute)):
            confound_vector[frames_to_permute[f]][f] = 1

        confound_file_path = self.output_data_path/ f"{iteration_path}_confound.txt"

        # Save the confound vector to a file
        np.savetxt(confound_file_path, confound_vector, delimiter=' ', fmt='%d')

        self.run_feat(num_frames, tr_time, self.filtered_data, self.design_file_with_confound_path, iteration_path, confound_file_path)


    def run_feat(self, num_frames: int, tr_time: float, data_file_path: Path, design_file_path: Path, analysis_path: Path, confound_file_path=None):
        """
        Adjust design file as needed and launch FSL feat

        Args:
            num_frames: Number of frames in the data file
            tr_time: TR time of the data file
            data_file_path: Path to NIFTI file that will be analyzed
            design_file_path: Path to .fsf file that will be used for analysis
            analysis_path: Path to directory where FEAT files will be output 
            confound_file_path: Path to confound file that will be used for analysis, if applicable
        """
        logging.info(f"Running feat on {data_file_path}")

        new_design_file_path = f"{analysis_path}_new.fsf"

        search_replace_mapping = {
            'set fmri(outputdir) ': f'set fmri(outputdir) "{analysis_path}"',
            'set feat_files(1) ': f'set feat_files(1) "{data_file_path}"',
            'set confoundev_files(1) ': f'set confoundev_files(1) "{confound_file_path}"',
            'set fmri(tr) ': f'set fmri(tr) {tr_time}',
            'set fmri(npts) ': f'set fmri(npts) {num_frames}',
        }

        with open(design_file_path, 'r') as old_design_file, open(new_design_file_path, 'w') as new_design_file:
            for line in old_design_file:
                found = False
                for search_string, replacement_text in search_replace_mapping.items():
                    if search_string in line:
                        new_design_file.write(replacement_text)
                        found = True
                if not found:
                    new_design_file.write(line)

        logging.info(f"Running feat on new design file {new_design_file_path}")
        feat = fsl.FEAT()
        feat.inputs.fsf_file = new_design_file_path
        feat.run()
    
    def get_q_score(self):
        """
        Calculate the q score from the permuted data

        Returns:
            q_score: The q score for the permutation test
        """

        # Load the mask from the feat file
        mask_nii = nib.load(f'{self.output_data_path}/real.feat/mask.nii.gz')
        mask = mask_nii.get_fdata().astype(bool)

        # Initialize a matrix to store permuted data
        all_data_mat = np.zeros((np.sum(mask), self.iterations))

        # Loop through all iterations and load permuted data
        for i in range(self.iterations):
            permuted_image = f"{self.output_data_path}/permutation_{i}.feat/stats/zstat{self.contrast_num}.nii.gz"
            permuted_data_nii = nib.load(permuted_image)
            permuted_data = permuted_data_nii.get_fdata()
            all_data_mat[:, i] = permuted_data[mask]

        # Calculate pairwise correlation between all iterations
        similarity_mat = 1 - distance.cdist(all_data_mat.T, all_data_mat.T, 'correlation')

        # Extract the lower triangular values
        iteration_similarity_lowertri = np.tril_indices(self.iterations, k=-1)
        similarities = similarity_mat[iteration_similarity_lowertri]

        # Calculate the mean similarity divided by the standard deviation of similarities
        q_score = int(np.mean(similarities) / np.std(similarities))

        return q_score