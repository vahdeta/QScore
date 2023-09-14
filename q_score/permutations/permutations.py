import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial import distance
import nibabel as nib
from nipype.interfaces import fsl
from permutations.utils import get_nifti_name

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
        self.design_file_path = self.base_folder / f"design/{self.task_type}.fsf"
        self.design_file_with_confound_path = self.base_folder / f"design/{self.task_type}_with_confounds.fsf"
        
        self.output_data_path = output_data_path
        self.analysis_path = self.output_data_path / "real"
        self.filtered_data = self.output_data_path / "real.feat/filtered_func_data"

        # TODO: put the following in a toml file
        self.num_threads = 10
        self.iterations = 10
        self.percent_to_permute = .1
        self.contrast_num = 1


    def start_permutations(self):
        print("PERMUTATIONS STARTED")
        # Make sure analysis path directory exists
        if not self.analysis_path.exists():
            self.analysis_path.mkdir(parents=True, exist_ok=True)

        # Look into data file path and get the name of the nifti file
        nifti_name = get_nifti_name(self.output_data_path)

        # Load the original data as a nifti with nibabel
        original_image_file_path = self.output_data_path / nifti_name
        original_image = nib.load(original_image_file_path)

        # Get the number of frames in the original data
        num_frames = original_image.shape[3]

        # Run FSL feat on new design file
        print("Running feat on new design file")
        self.run_feat(original_image_file_path, self.design_file_path, self.analysis_path)

        num_permuted_frames = int(math.ceil(self.percent_to_permute * num_frames))

        print("STARTING THREAD POOL")
        # Start the permutations in a thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i in range(self.iterations):
                iteration_path = self.output_data_path / f"permutation_{i}"
                executor.submit(self.run_permutations_with_feat, num_permuted_frames, num_frames, iteration_path)


    def run_permutations_with_feat(self, num_permuted_frames: int, num_frames: int, iteration_path: Path):
        """
        permuted_amt: Percentage of frames to permute
        num_permuted_frames: Number of frames to permute
        iteration_path: Path to directory where permutation data will be written
        """
        print(f"RUNNING PERMUTATION FOR {iteration_path}")

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

        self.run_feat(self.filtered_data, self.design_file_with_confound_path, iteration_path, confound_file_path)



    def run_feat(self, data_file_path, design_file_path, analysis_path, confound_file_path=None):
        """
        data_file_path: Path to NIFTI file that will be analyzed
        design_file_path: Path to .fsf file that will be used for analysis
        analysis_path: Path to directory where FEAT files will be output 
        confound_file_path: Path to confound file that will be used for analysis, if applicable
        """

        new_design_file_path = f"{analysis_path}_new.fsf"

        search_replace_mapping = {
            'set fmri(outputdir) ': f'set fmri(outputdir) "{analysis_path}"',
            'set feat_files(1) ': f'set feat_files(1) "{data_file_path}"',
            'set confoundev_files(1) ': f'set confoundev_files(1) "{confound_file_path}"',
            'set fmri(custom1) ': f'set fmri(custom1) "{self.base_folder}/design/{self.task_type}.txt"',
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

        # Run feat on new design file
        print("Running feat on new design file")
        feat = fsl.FEAT()
        feat.inputs.fsf_file = new_design_file_path
        feat.run()

    
    def get_q_score(self):

        #iteration_similarity_lowertri = np.tril(np.ones((self.iterations, self.iterations)), k=-1)
        iteration_similarity_lowertri = np.tril_indices(self.iterations, k=-1)

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
        similarities = similarity_mat[iteration_similarity_lowertri]

        # Calculate the mean similarity divided by the standard deviation of similarities
        q_score = int(np.mean(similarities) / np.std(similarities))

        return q_score