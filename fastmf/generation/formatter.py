import os
import pickle
import json
import numpy as np
import fastmf.utils.mf_utils as mfu
from tqdm import tqdm

class DataFormatter():
    def __init__(self, base_path, task_name, session_id, 
                 reference_dic_path, run_ids, type_, set_size):
        self.base_path = base_path #path to the folder containing the 'synthetizer' and 'generator' folders
        self.task_name = task_name
        self.session_id = session_id
        self.reference_dic = mfu.loadmat(reference_dic_path)
        self.reference_dic["fasc_propnames"] = [s.strip() for s in self.reference_dic['fasc_propnames']]
        self.run_ids = run_ids
        self.type = type_

        self.set_size = set_size
        assert len(set_size) == 3, "Set size must be a list of length 3."

        self.num_fasc = 2

    def __genTarget(self, architecture,
                    min_max_scaling, orientation_estimate=None):

        assert architecture in ["NNLS", "SphericalHarmonic"], "Architecture not supported"
        if architecture == "NNLS":
            assert orientation_estimate in ["CSD", "MSMTCSD", "GROUNDTRUTH"], "Invalid orientation estimate type."

        data_path = os.path.join(self.base_path, "synthetizer", f"type-{self.type}", "raw")

        num_samples = 0
        num_fasc = self.num_fasc
        fasc_propnames = self.reference_dic["fasc_propnames"]

        metadata_runs = {}
        for run_id in self.run_ids:
            filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_raw"
            with open(os.path.join(data_path, filename + ".json"), 'r') as outfile:
                json_metadata = json.load(outfile)
            if (run_id == self.run_ids[0]):  # Check if all generator data include csf
                include_csf = json_metadata['include_csf']
            elif (include_csf != json_metadata['include_csf']):
                raise ValueError('The initial run had include_csf = {0} but we found run_id {1} with include_csf = {2}'.format(include_csf, run_id, json_metadata['include_csf']))

            metadata_runs[run_id] = json_metadata
            num_samples = num_samples + json_metadata["num_samples"]

        assert self.set_size[0] + self.set_size[1] + self.set_size[2] <= num_samples, "Set size {0} is larger than the number of samples {1}.".format(self.set_size[0] + self.set_size[1] + self.set_size[2], num_samples)
        
        
        if architecture == "NNLS":
            if(include_csf):
                target = np.zeros((num_samples, num_fasc * (1 + len(fasc_propnames)) + 1)) #1 is for nus ; len(fasc_propnames) if for the number of properties used to generate the dictionary (MC simulations)
            else:
                target = np.zeros((num_samples, num_fasc * (1 + len(fasc_propnames))))
            output_path = os.path.join(self.base_path, "formatter", f"type-{self.type}", "NNLS")
            est_orientations_swapped_global = np.zeros((num_samples))
        elif architecture == "SphericalHarmonic":
            if(include_csf):
                target = np.zeros((num_samples, num_fasc * (1 + 3 + len(fasc_propnames)) + 1))
            else:
                target = np.zeros((num_samples, num_fasc * (1 + 3 + len(fasc_propnames))))
            output_path = os.path.join(self.base_path, "formatter", f"type-{self.type}", "SH")

        idx = 0
        for run_id in self.run_ids:
            filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_raw"

            with open(os.path.join(data_path, filename + ".pickle"), 'rb') as handle:
                run_dic = pickle.load(handle)

            if architecture == "NNLS":
                #[0, len(fasc_propnames) + 1] : 0 is for first subdictionary, len(fasc_propnames) + 1
                #so prop are stored in the following way : nu1, prop1_1, prop1_2,.... nu2, prop2_1, prop_2_2....
                gen_path = os.path.join(self.base_path, "generator", f"type-{self.type}", "NNLS")
                gen_filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_orientation-{orientation_estimate}_NNLS"
                with open(os.path.join(gen_path, gen_filename + ".pickle"), 'rb') as handle:
                    gen_run_data = pickle.load(handle)
                est_orientations_swapped = gen_run_data['est_orientations_swapped']

                n_samples = metadata_runs[run_id]["num_samples"]
                target[idx:idx+n_samples, [0, len(fasc_propnames) + 1]] = run_dic['nus']

                prop_num = 1
                for prop_name in fasc_propnames:
                    #rund_dic['IDs'] : indexes of the atoms used to generate the signal
                    #[prop_num, 1 + len(fasc_propnames) + prop_num]] : prop_num is for the first subdictionary, 1 + len(...) + prop_num is for the second subdictionary
                    target[idx:idx+n_samples, [prop_num, 1 + len(fasc_propnames) + prop_num]] = self.reference_dic[prop_name][run_dic['IDs']]
                    prop_num = prop_num + 1
                    
                if(include_csf):
                    n_samples = metadata_runs[run_id]["num_samples"]
                    target[idx:idx+n_samples, 6] = run_dic['nuscsf'][:]

                est_orientations_swapped_global[idx:idx+n_samples] = est_orientations_swapped[:]
                    
            elif architecture == "SphericalHarmonic":

                target[idx:idx+metadata_runs[run_id]["num_samples"], [0, 4 + len(fasc_propnames)]] = run_dic['nus']

                target[idx:idx+metadata_runs[run_id]["num_samples"], [1, 4 + len(fasc_propnames) + 1]] = run_dic['orientations'][:, :, 0]
                target[idx:idx+metadata_runs[run_id]["num_samples"], [2, 4 + len(fasc_propnames) + 2]] = run_dic['orientations'][:, :, 1]
                target[idx:idx+metadata_runs[run_id]["num_samples"], [3, 4 + len(fasc_propnames) + 3]] = run_dic['orientations'][:, :, 2]

                prop_num = 1
                for prop_name in fasc_propnames:
                    target[idx:idx+metadata_runs[run_id]["num_samples"], [prop_num + 3, 4 + len(fasc_propnames) + prop_num + 3]] = self.reference_dic[prop_name][run_dic['IDs']]
                    prop_num = prop_num + 1
                if(include_csf):
                    n_samples = metadata_runs[run_id]["num_samples"]
                    target[idx:idx+n_samples, -1] = run_dic['nuscsf'][:]
                    
            idx = idx + metadata_runs[run_id]["num_samples"]
            del run_dic


        if architecture == "NNLS":
            # Performs swapping of first three element with next three depending on est_orientations_swapped
            props_len = (1 + len(fasc_propnames))
            assert num_fasc == 2, "Only two fascicles are supported for NNLS."
            est_orientations_swapped_global = est_orientations_swapped_global.astype(bool)
            (target[est_orientations_swapped_global, 0:props_len],
             target[est_orientations_swapped_global, props_len:props_len * 2]) = \
                (target[est_orientations_swapped_global, props_len:2 * props_len],
                 target[est_orientations_swapped_global, 0:props_len])

        if min_max_scaling:
            if architecture == "NNLS":
                scaler_Hybrid = pickle.load(open(os.path.join(self.base_path, "scaler", "scaler-minmax_ses-{0}_NNLS.pickle".format(self.session_id)), 'rb'))
                target = scaler_Hybrid.transform(target)  # scaler: (num_samples, num_features)
            elif architecture == "SphericalHarmonic":
                scaler_Full = pickle.load(open(os.path.join(self.base_path, "scaler", "scaler-minmax_ses-{0}_SH.pickle".format(self.session_id)), 'rb'))
                target = scaler_Full.transform(target)

        # Divide into train, validation and test

        training_set = target[:self.set_size[0], :]
        validation_set = target[self.set_size[0]:self.set_size[0]+self.set_size[1], :]
        testing_set = target[self.set_size[0]+self.set_size[1]:self.set_size[0]+self.set_size[1]+self.set_size[2], :]

        # Save the data

        output_filename = f"type-{self.type}_task-{self.task_name}_ses-{self.session_id}"

        if architecture == "NNLS":
            output_filename = output_filename + f"_orientation-{orientation_estimate}"

        if min_max_scaling:
            output_filename = output_filename + "_scaler-minmax"
        else:
            output_filename = output_filename + "_scaler-none"

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(os.path.join(output_path, output_filename + "_set-training" + "_target.npy"), training_set)
        np.save(os.path.join(output_path, output_filename + "_set-validation" + "_target.npy"), validation_set)
        np.save(os.path.join(output_path, output_filename + "_set-testing" + "_target.npy"), testing_set)


        return include_csf


    def genNNLSTarget(self, min_max_scaling=True, orientation_estimate="CSD"):
        """ Generate target data."""

        assert orientation_estimate in ["CSD", "MSMTCSD", "GROUNDTRUTH"], "Invalid orientation estimate type."

        scaler_path = os.path.join(self.base_path, "scaler")
        scaler_filename = "scaler-minmax_ses-{0}_NNLS.pickle".format(self.session_id)
        prop_names = self.reference_dic['fasc_propnames']
        num_prop = len(prop_names)
        
        #if not os.path.exists(os.path.join(scaler_path, scaler_filename)):
        minRange = [0]
        maxRange = [1]
        print('Formatting NNLS target')
        for i in tqdm(range(num_prop)):
            mini = np.min(self.reference_dic[prop_names[i]])
            maxi = np.max(self.reference_dic[prop_names[i]])
            minRange.append(mini)
            maxRange.append(maxi)
            # minRange = [0, 6e-10, 6e-02, 0, 6e-10, 6e-02]
            # maxRange = [1, 2.4e-09, 8e-01, 1, 2.4e-09, 8e-01]

        # Check if csf needs tp be included
        include_csf = None
        data_path = os.path.join(self.base_path, "synthetizer", f"type-{self.type}", "raw")
        for run_id in self.run_ids:
            filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_raw"
            with open(os.path.join(data_path, filename + ".json"), 'r') as outfile:
                json_metadata = json.load(outfile)
            if (run_id == self.run_ids[0]):  # Check if all generator data include csf
                include_csf = json_metadata['include_csf']
            elif (include_csf != json_metadata['include_csf']):
                raise ValueError('The initial run had include_csf = {0} but we found run_id {1} with include_csf = {2}'.format(include_csf, run_id, json_metadata['include_csf']))

        maxRange = maxRange * self.num_fasc
        minRange = minRange * self.num_fasc
        if (include_csf):
            minRange.append(0)
            maxRange.append(1)
        self.__generateMinMaxScaler(minRange, maxRange,
                                    scaler_path, scaler_filename
                                    )

        self.__genTarget("NNLS", min_max_scaling, orientation_estimate=orientation_estimate)



    def genSphericalHarmonicTarget(self, min_max_scaling=True):
        """ Generate target data."""

        scaler_path = os.path.join(self.base_path, "scaler")
        scaler_filename = "scaler-minmax_ses-{0}_SH.pickle".format(self.session_id)
        prop_names = self.reference_dic['fasc_propnames']
        num_prop = len(prop_names)
        
        
        minRange = [0,-1,-1,0]
        maxRange = [1,1,1,1]
        print('Formatting Spherical Harmonic target')
        for i in tqdm(range(num_prop)):
            mini = np.min(self.reference_dic[prop_names[i]])
            maxi = np.max(self.reference_dic[prop_names[i]])
            minRange.append(mini)
            maxRange.append(maxi)
            # minRange = [0, 6e-10, 6e-02, 0, 6e-10, 6e-02]
            # maxRange = [1, 2.4e-09, 8e-01, 1, 2.4e-09, 8e-01]

        # Check if csf needs tp be included
        include_csf = None
        data_path = os.path.join(self.base_path, "synthetizer", f"type-{self.type}", "raw")
        for run_id in self.run_ids:
            filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_raw"
            with open(os.path.join(data_path, filename + ".json"), 'r') as outfile:
                json_metadata = json.load(outfile)
            if (run_id == self.run_ids[0]):  # Check if all generator data include csf
                include_csf = json_metadata['include_csf']
            elif (include_csf != json_metadata['include_csf']):
                raise ValueError('The initial run had include_csf = {0} but we found run_id {1} with include_csf = {2}'.format(include_csf, run_id, json_metadata['include_csf']))

        maxRange = maxRange * self.num_fasc
        minRange = minRange * self.num_fasc
        if (include_csf):
            minRange.append(0)
            maxRange.append(1)

        self.__generateMinMaxScaler(minRange, maxRange, scaler_path, scaler_filename)

        self.__genTarget("SphericalHarmonic", min_max_scaling)

    def genNNLSInput(self, normalization="None", orientation_estimate="CSD"):

        assert normalization in ["None", "SumToOne", "Standard"], "Normalization not supported."
        assert orientation_estimate in ["CSD", "MSMTCSD", "GROUNDTRUTH"], "Invalid orientation estimate type."

        num_samples = 0
        num_fasc = self.num_fasc
        num_atoms = self.reference_dic["num_atom"]

        data_path = os.path.join(self.base_path, "generator", f"type-{self.type}", "NNLS")

        metadata_NNLS = {}
        for run_id in self.run_ids:
            data_filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_orientation-{orientation_estimate}_NNLS"
            with open(os.path.join(data_path, data_filename + ".json"), 'r') as outfile:
                json_metadata = json.load(outfile)
                if(run_id == self.run_ids[0]): #Check if all generator data include csf
                    include_csf = json_metadata['include_csf']
                elif(include_csf!=json_metadata['include_csf']):
                    raise ValueError('The initial run had include_csf = {0} but we found run_id {1} with include_csf = {2}'.format(include_csf,run_id,json_metadata['include_csf']))
                    
            
            metadata_NNLS[run_id] = json_metadata
            num_samples = num_samples + json_metadata["num_samples"]
            del json_metadata
            
            if(include_csf):
                latent_space_NNLS = np.zeros((num_samples, num_fasc * (num_atoms) + 1), dtype = 'float32')
            else:
                latent_space_NNLS = np.zeros((num_samples, num_fasc * (num_atoms)))

        idx = 0
        print('Formatting NNLS input')
        for run_id in tqdm(self.run_ids):
            data_filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_orientation-{orientation_estimate}_NNLS"

            if metadata_NNLS[run_id]["sparse_storage"]:
                with open(os.path.join(data_path, data_filename + ".pickle"), 'rb') as handle:
                    NNLS_weights_dic = pickle.load(handle)
                    
                if(include_csf):
                    NNLS_weights = np.zeros((metadata_NNLS[run_id]["num_samples"], 
                                             num_fasc * self.reference_dic["num_atom"] + 1))
                else:
                    NNLS_weights = np.zeros((metadata_NNLS[run_id]["num_samples"], 
                                             num_fasc * self.reference_dic["num_atom"]))
                        
                isbatch = ((NNLS_weights_dic['w_idx'][:, 0] >= 0) &
                           (NNLS_weights_dic['w_idx'][:, 0] < metadata_NNLS[run_id]["num_samples"]))
                chk = (np.sum(isbatch) ==
                       np.sum(NNLS_weights_dic['nnz_hist'][0:metadata_NNLS[run_id]["num_samples"]]))
                assert chk, ("Mismatch non-zero elements in samples "
                             "%d (incl.) to %d (excl.)" % (0, metadata_NNLS[run_id]["num_samples"]))
                w_idx = NNLS_weights_dic['w_idx'][isbatch, :]
                NNLS_weights[w_idx[:, 0] - 0,
                w_idx[:, 1]] = NNLS_weights_dic['w_data'][isbatch]
            else:
                NNLS_weights = np.load(os.path.join(data_path, data_filename + ".npy"))

            if(include_csf):
                assert NNLS_weights.shape[1] == num_fasc * num_atoms + 1, "Mismatch in number of fascicles or atoms"
            else:
                assert NNLS_weights.shape[1] == num_fasc * num_atoms , "Mismatch in number of fascicles or atoms"

            latent_space_NNLS[idx:idx+metadata_NNLS[run_id]["num_samples"], :] = NNLS_weights
            idx = idx + metadata_NNLS[run_id]["num_samples"]
            del NNLS_weights_dic
        # Input Normalization

        if normalization == "SumToOne":
            latent_space_NNLS = latent_space_NNLS / np.sum(latent_space_NNLS, axis=1)[:, np.newaxis]
        elif normalization == "Standard":
            latent_space_NNLS = (latent_space_NNLS - np.mean(latent_space_NNLS, axis=1)[:, np.newaxis])
            std_NNLS = latent_space_NNLS / np.std(latent_space_NNLS, axis=1)[:, np.newaxis]
            idx_pos_std = np.where(std_NNLS > 0)[0]
            latent_space_NNLS[idx_pos_std, :] = (latent_space_NNLS[idx_pos_std, :] / std_NNLS[idx_pos_std][:, np.newaxis])

            del idx_pos_std, std_NNLS


        # Divide into train, validation and test

        training_set = latent_space_NNLS[:self.set_size[0], :]
        validation_set = latent_space_NNLS[self.set_size[0]:self.set_size[0]+self.set_size[1], :]
        testing_set = latent_space_NNLS[self.set_size[0]+self.set_size[1]:self.set_size[0]+self.set_size[1]+self.set_size[2], :]

        # Save the data

        output_path = os.path.join(self.base_path, "formatter", f"type-{self.type}", "NNLS")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_filename = f"type-{self.type}_task-{self.task_name}_ses-{self.session_id}_orientation-{orientation_estimate}_normalization-{normalization}"

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(os.path.join(output_path, output_filename + "_set-training" + "_NNLS.npy"), training_set)
        np.save(os.path.join(output_path, output_filename + "_set-validation" + "_NNLS.npy"), validation_set)
        np.save(os.path.join(output_path, output_filename + "_set-testing" + "_NNLS.npy"), testing_set)

    def genSphericalHarmonicInput(self):

        num_samples = 0
        n_shells = None
        basis_size = None

        data_path = os.path.join(self.base_path, "generator", f"type-{self.type}", "SH")

        metadata_SH = {}
        for run_id in self.run_ids:
            data_filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_SH"
            with open(os.path.join(data_path, data_filename + ".json"), 'r') as outfile:
                json_metadata = json.load(outfile)

            metadata_SH[run_id] = json_metadata
            num_samples = num_samples + json_metadata["num_samples"]

            if n_shells is None:
                n_shells = len(json_metadata["shells"])
            if basis_size is None:
                basis_size = json_metadata["basis_size"]

            assert n_shells == len(json_metadata["shells"]), "Mismatch in number of shells"
            assert basis_size == json_metadata["basis_size"], "Mismatch in basis size"

        SH_input = np.zeros((num_samples, n_shells, basis_size))

        idx = 0
        print('Formatting Spherical Harmonic input')
        for run_id in tqdm(self.run_ids):
            data_filename = f"type-{self.type}_task-{self.task_name}_run-{run_id}_SH"

            SH_weights = np.load(os.path.join(data_path, data_filename + ".npy"))
            num_samples = num_samples + SH_weights.shape[0]

            SH_input[idx:idx+metadata_SH[run_id]["num_samples"], :, :] = SH_weights

            idx = idx + metadata_SH[run_id]["num_samples"]

        # Divide into train, validation and test

        training_set = SH_input[:self.set_size[0], :, :]
        validation_set = SH_input[self.set_size[0]:self.set_size[0]+self.set_size[1], :, :]
        testing_set = SH_input[self.set_size[0]+self.set_size[1]:self.set_size[0]+self.set_size[1]+self.set_size[2], :, :]

        # Save the data

        output_path = os.path.join(self.base_path, "formatter", f"type-{self.type}", "SH")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_filename = f"type-{self.type}_task-{self.task_name}_ses-{self.session_id}"

        np.save(os.path.join(output_path, output_filename + "_normalization-None_set-training" + "_SH.npy"), training_set)
        np.save(os.path.join(output_path, output_filename + "_normalization-None_set-validation" + "_SH.npy"), validation_set)
        np.save(os.path.join(output_path, output_filename + "_normalization-None_set-testing" + "_SH.npy"), testing_set)


    def __generateMinMaxScaler(self, minRange, maxRange, 
                               scaler_path, scaler_filename,):
        from sklearn.preprocessing import MinMaxScaler

        rangeMinMax = np.array([minRange, maxRange])
        scaler = MinMaxScaler()
        target = scaler.fit_transform(rangeMinMax)  # scaler: (num_samples, num_features)

        if not os.path.exists(scaler_path):
            os.makedirs(scaler_path)

        with open(os.path.join(scaler_path, scaler_filename), 'wb') as f:
            pickle.dump(scaler, f)
