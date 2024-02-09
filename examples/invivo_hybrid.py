import os

from dipy.io.image import load_nifti

import fastmf.inference as inf
import pickle

import numpy as np

if __name__ == '__main__':

    base_path = r"C:\Users\quent\Documents\Github\FastMF_python\tests\dataPaper"
    include_csf = False
    session_id = "2Mtraining"
    run_id = "0"
    task_name = "paper"
    training_sample = 2000000
    orientation_estimate = 'CSD'

    model_path = os.path.join(base_path, 'training', 'type-standard', f'ses-{session_id}')
    base_name_Hybrid = f"type-standard_task-{task_name}_ses-{session_id}_orientation-{orientation_estimate}_trainSamples-{training_sample}"

    metadata_Hybrid = os.path.join(model_path, "Hybrid", base_name_Hybrid + '_metadata.json')
    model_state_dict_Hybrid = os.path.join(model_path, "Hybrid", base_name_Hybrid + '_modelstatedict.pt')

    scaler_path = os.path.join(base_path, 'scaler', 'scaler-minmax_ses-{0}_NNLS.pickle'.format(session_id))
    dic_path = "..\\data\\dictionaries\\dictionary-fixedraddist_scheme-HCPMGH.mat"

    invivo_fitter = inf.Hybrid_Model(metadata_Hybrid, model_state_dict_Hybrid, scaling_fn='MinMax',
                                     dictionary_path=dic_path, scaling_fn_path=scaler_path, device='cpu')

    # Define patient data
    folder_path = r"C:\Users\quent\Documents\Github\FastMF_python\tests\ElikoPyHCPv2"
    patient_path = "sub-1002"
    data_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz"
    bvals = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval"
    bvecs = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec"
    wm_mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_wm_mask_FSL_T1.nii.gz'

    # Load peaks
    odf_csd_path = folder_path + '/subjects/' + patient_path + "/dMRI/ODF/CSD"
    peaks_path = odf_csd_path + '/' + patient_path + '_CSD_peaks.nii.gz'
    peaks_values_path = odf_csd_path + '/' + patient_path + '_CSD_values.nii.gz'
    peaks, _ = load_nifti(peaks_path)
    peaks_values, _ = load_nifti(peaks_values_path)

    numfasc = np.sum(peaks_values[:, :, :, 0] > 0.05) + np.sum(
        peaks_values[:, :, :, 1] > 0.05)

    # Normalize peaks with numfasc > 2
    peaks[numfasc >= 2] = peaks[numfasc >= 2] / np.linalg.norm(peaks[numfasc >= 2], axis=-1)[..., None]

    # Mask with WM mask and only voxels with at least 2 fascicles
    wm_mask, _ = load_nifti(wm_mask_path)
    mask = np.logical_and(wm_mask, numfasc >= 2)
    # Reduce number of TRUE in mask to speedup computation during testing
    mask[0:120, 0:110, :] = False

    print("Number of voxels: ", np.sum(mask))
    print("Number of total voxels: ", np.prod(mask.shape), " (", mask.shape, ")")
    print("Percentage of voxels to be processed: ", np.sum(mask) / np.prod(mask.shape) * 100)

    # Fit the model
    HybridModelFitted = invivo_fitter.fit(data_path, mask, peaks,
                                          bvals=bvals, bvecs=bvecs,
                                          verbose=4, M0_estimation=True)

    # Save the model
    output_folder = os.path.join(folder_path, 'subjects', patient_path, 'dMRI', 'microstructure', 'Hybrid')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, f"{patient_path}_Hybrid.pickle"), 'wb') as handle:
        pickle.dump(HybridModelFitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(HybridModelFitted.write_nifti(os.path.join(output_folder, f"{patient_path}_Hybrid.nii.gz")))

    # Save the model weights
    HybridModelFitted.write_weights(os.path.join(output_folder, f"{patient_path}_Hybrid_weights"))
