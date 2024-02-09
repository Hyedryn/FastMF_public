import os
import fastmf.inference as inf
import pickle

if __name__ == '__main__':

    base_path = r"C:\Users\quent\Documents\Github\FastMF_python\tests\dataV1"
    include_csf = False
    session_id = "2Mtraining"
    run_id = "0"
    task_name = "paper"
    training_sample = 2000000

    model_path = os.path.join(base_path, 'training', 'type-standard', f'ses-{session_id}')
    base_name_Full = f"type-standard_task-{task_name}_ses-{session_id}_trainSamples-{training_sample}"
    metadata_Full = os.path.join(model_path, "FullyLearnedSWAP", base_name_Full + '_metadata.json')
    model_state_dict_Full = os.path.join(model_path, "FullyLearnedSWAP", base_name_Full + '_modelstatedict.pt')
    scaler_path = os.path.join(base_path, 'scaler', 'scaler-minmax_ses-{0}_SH.pickle'.format(session_id))
    dic_path = "..\\data\\dictionaries\\dictionary-fixedraddist_scheme-HCPMGH.mat"

    invivo_fitter = inf.FullyLearned_Model(metadata_Full, model_state_dict_Full, scaling_fn='MinMax', dictionary_path=dic_path, scaling_fn_path=scaler_path, device='cpu')

    # Define patient data
    folder_path = r"C:\Users\quent\Documents\Github\FastMF_python\tests\ElikoPyHCPv2"
    patient_path = "sub-1002"
    data_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.nii.gz"
    bvals = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval"
    bvecs = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec"
    wm_mask_path = folder_path + '/subjects/' + patient_path + "/masks/" + patient_path + '_wm_mask_FSL_T1.nii.gz'

    # Fit the model
    FullyLearnedModelFitted = invivo_fitter.fit(data_path, wm_mask_path,
                                      bvals=bvals, bvecs=bvecs,
                                      verbose=4, M0_estimation=True)

    # Save the model
    output_folder = os.path.join(folder_path, 'subjects', patient_path, 'dMRI', 'microstructure', 'FullyLearned')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, f"{patient_path}_FL.pickle"), 'wb') as handle:
        pickle.dump(FullyLearnedModelFitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(FullyLearnedModelFitted.write_nifti(os.path.join(output_folder, f"{patient_path}_FL.nii.gz")))
