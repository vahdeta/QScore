import os
import math
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from nipype.interfaces import fsl
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor
from permutations.utils import get_series_description, load_scaling_params, get_scaled_score

class Permutations:
    def __init__(self, base_folder: Path, output_data_path: Path, original_nifti: Path):
        """
        base_folder: Path to folder to python package
        output_data_path: Path to output data directory, where FEAT and NIFTI files will be written
        original_nifti: Path to original NIFTI file
        """        

        self.base_folder = base_folder
        self.original_nifti = original_nifti
        self.task_type = get_series_description(original_nifti)
        if self.task_type == "error":
            raise Exception("Error reading task type from NIFTI file name")

        self.design_file_path = self.base_folder / f"design/{self.task_type}/{self.task_type}.fsf"
        self.design_file_with_confound_path = self.base_folder / f"design/{self.task_type}/{self.task_type}_with_confounds.fsf"

        self.output_data_path = output_data_path
        self.analysis_path = self.output_data_path / "real"

        # Make sure analysis path directory exists
        if not self.analysis_path.exists():
            self.analysis_path.mkdir(parents=True, exist_ok=True)

        self.filtered_data = self.output_data_path / "truncated_bet_mcf.nii.gz"
        self.num_threads = int(os.environ.get("Q_SCORE_THREADS", 10))
        self.iterations = int(os.environ.get("Q_SCORE_PERMUTATIONS", 20))
        self.percent_to_permute = .1
        self.num_frames = 161
        self.tr_time = 1.8

        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def start_analysis(self):
        """
        Method to start permutation thread pool and run feat
        """

        logging.info("Starting first level analysis")
        
        # Load the original data as a nifti with nibabel
        original_image = nib.load(self.original_nifti)
        self.tr_time = original_image.header['pixdim'][4]

        # Truncate the original data by 4 frames
        truncated_image = original_image.slicer[:,:,:,:-4]
        truncated_image_file_path = self.output_data_path / "truncated.nii.gz"
        nib.save(truncated_image, truncated_image_file_path)
        self.num_frames = truncated_image.shape[3]

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
    
    def do_registration(self, num_contrasts):
        """
        Register the example_func to standard space

        Args:
            num_contrasts: Number of contrasts
        """

        # Register the example_func to standard space

        fslroi = fsl.ExtractROI()

        fslroi.inputs.in_file = f"{self.output_data_path}/real.feat/filtered_func_data.nii.gz"
        fslroi.inputs.roi_file = f"{self.output_data_path}/example_func.nii.gz"
        fslroi.inputs.t_min = 78
        fslroi.inputs.t_size = 1

        fslroi.run()

        # Run standard space brain extraction
        fslmaths = fsl.ImageMaths()
        fslmaths.inputs.in_file = f"{self.base_folder}/design/standard/MNI152_T1_4mm_brain.nii.gz"
        fslmaths.inputs.out_file = f"{self.output_data_path}/standard.nii.gz"

        fslmaths.run()

        # Run flirt
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = f"{self.output_data_path}/example_func.nii.gz"
        flirt.inputs.reference = f"{self.output_data_path}/standard.nii.gz"
        flirt.inputs.out_file = f"{self.output_data_path}/example_func2standard.nii.gz"
        flirt.inputs.out_matrix_file = f"{self.output_data_path}/example_func2standard.mat"
        flirt.inputs.cost = "corratio"
        flirt.inputs.dof = 12
        flirt.inputs.searchr_x = [-90, 90]
        flirt.inputs.searchr_y = [-90, 90]
        flirt.inputs.searchr_z = [-90, 90]
        flirt.inputs.interp = "trilinear"

        flirt.run()

        # Run convert_xfm
        convert_xfm_node = fsl.ConvertXFM()
        convert_xfm_node.inputs.in_file = f"{self.output_data_path}/example_func2standard.mat"
        convert_xfm_node.inputs.out_file = f"{self.output_data_path}/standard2example_func.mat"
        convert_xfm_node.inputs.invert_xfm = True

        convert_xfm_node.run()

        for contrast in range(1, num_contrasts+1):  
            # Register the zstat image to standard space
            zstat_flirt = fsl.FLIRT()

            # Set input parameters
            zstat_flirt.inputs.in_file = self.output_data_path / f"real.feat/stats/zstat{contrast}.nii.gz"
            zstat_flirt.inputs.reference = f"{self.base_folder}/design/standard/MNI152_T1_4mm_brain.nii.gz"
            zstat_flirt.inputs.apply_xfm = True
            zstat_flirt.inputs.in_matrix_file = self.output_data_path / f"example_func2standard.mat"

            # Define the path to the 4x4x4 zstat image
            zstat_image = self.output_data_path / f"zstat{contrast}_registered.nii.gz"
            zstat_flirt.inputs.out_file = zstat_image
            
            # Run flirt
            zstat_flirt.run()
        
    def get_dice_coefficient(self, thresholded_image, template_map_data):
        """
        Calculate the dice coefficient between the thresholded image and the template map

        Args:
            thresholded_image: The binarized thresholded image in numpy array format
            template_map_data: The binarized template map in numpy array format
        """
            
        # Get overlap between thresholded image and template map
        overlap = np.sum(np.logical_and(thresholded_image, template_map_data))
        area_thresholded_image = np.sum(thresholded_image)
        area_template_map = np.sum(template_map_data)

        # Calculate dice coefficient
        dice_coefficient = (2 * overlap) / (area_thresholded_image + area_template_map)

        return dice_coefficient

    def get_q_score(self):
        """
        Calculate the q score. Gets the dice coefficient for the thresholded zstat image and the template map.

        Returns:
            q_score: The q score for the permutation test
        """

        q_scores = []

        # Get the number of contrasts
        num_contrasts = 1 if self.task_type == "objnam" else 2

        # Register the zstat image to standard space
        self.do_registration(num_contrasts)
        
        for contrast in range(1, num_contrasts+1):  
            
            if self.task_type == "objnam":
                template_map_path = self.base_folder / f"design/{self.task_type}/template_mask.nii.gz"
            else:
                template_map_path = self.base_folder / f"design/{self.task_type}/template_contrast{contrast}_mask.nii.gz"
            
            zstat_image = self.output_data_path / f"zstat{contrast}_registered.nii.gz"

            # Load z stat image
            zstat_nii = nib.load(zstat_image)
            zstat_data = zstat_nii.get_fdata()
            zstat_data = np.nan_to_num(zstat_data)

            # Load the standard space template and ventricle mask to mask out the ventricles and other non-brain areas
            standard_space_template_data = nib.load(self.base_folder / "design/standard/MNI152_T1_4mm_brain.nii.gz").get_fdata()
            ventricle_mask = nib.load(self.base_folder / "design/standard/MNI152_T1_ventricles_4mm.nii.gz").get_fdata()
            zstat_data = np.where((standard_space_template_data != 0) & (ventricle_mask == 0), zstat_data, 0)

            # Load the template map for comparison
            template_map = nib.load(template_map_path)
            template_map_data = template_map.get_fdata()
            template_map_data = np.nan_to_num(template_map_data)
            
            # Calculate the threshold values
            threshold_percents = [99.5, 99, 97, 95, 90]
            dice_coefficients = []

            # Calculate the dice coefficient for each threshold
            for threshold in threshold_percents:
                percentile = np.percentile(zstat_data, threshold)
                logging.info(f"Percentile for contrast {contrast}: {threshold}: {percentile}")
                dice = self.get_dice_coefficient(np.where(zstat_data > percentile, 1, 0), template_map_data)
                logging.info(f"Dice coefficient for contrast {contrast}: {threshold}: {dice}")
                dice_coefficients.append(dice)

            # Pick the best dice coefficient
            best_dice = max(dice_coefficients)
            logging.info(f"Best dice coefficient for contrast {contrast}: {best_dice}")
            q_scores.append(min(int((best_dice) * 100 / (0.8)), 100))

        unscaled_q_score = int(np.mean(q_scores))

        # Get the scaling parameters for this specific task
        scaling_params = load_scaling_params(path_to_params=Path(self.base_folder / f"design/scales/{self.task_type}_scaling_params.json"))
        q_score = get_scaled_score(unscaled_q_score, scaling_params)

        return q_score

    def get_compliance_score(self):
        """
        Calculate the compliance score by permuting data and calculating the similarity between the permuted data and the real data.

        Returns:
            compliance_score: The compliance score for the permutation test
        """

        num_permuted_frames = int(math.ceil(self.percent_to_permute * self.num_frames))

        # Load the mask from the feat file
        mask_nii = nib.load(f'{self.output_data_path}/truncated_bet_mask.nii.gz')
        mask = mask_nii.get_fdata().astype(bool)

        # Get the number of contrasts
        num_contrast = 1 if self.task_type == "objnam" else 2
        compliance_scores = []

        # Start the permutations in a thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {}
            for i in range(self.iterations):
                iteration_path = self.output_data_path / f"permutation_{i}"
                futures[executor.submit(self.run_permutations_with_feat, num_permuted_frames, self.num_frames, self.tr_time, iteration_path)] = [i]

        # Loop through all iterations and load permuted data for each contrast
        for contrast in range(1, num_contrast+1):
            # Initialize a matrix to store permuted data
            all_data_mat = np.zeros((np.sum(mask), self.iterations))
            for i in range(self.iterations):
                permuted_image = f"{self.output_data_path}/permutation_{i}.feat/stats/zstat{contrast}.nii.gz"
                permuted_data_nii = nib.load(permuted_image)
                permuted_data = permuted_data_nii.get_fdata()
                all_data_mat[:, i] = permuted_data[mask]

            # Calculate pairwise correlation between all iterations
            similarity_mat = 1 - distance.cdist(all_data_mat.T, all_data_mat.T, 'correlation')

            # Extract the lower triangular values
            iteration_similarity_lowertri = np.tril_indices(self.iterations, k=-1)
            similarities = similarity_mat[iteration_similarity_lowertri]

            # Calculate the mean similarity divided by the standard deviation of similarities
            compliance_scores.append(int(np.mean(similarities) / np.std(similarities)))

        # Take the average of the compliance scores across contrasts
        compliance_score = int(np.mean(compliance_scores))

        return compliance_score


