import json
import pickle
import numpy as np
import os
from tqdm import tqdm

import fastmf.utils.mf_utils as mfu

from matplotlib import pyplot as plt


class Synthetizer:
    """Class to generate synthetic data from a given scheme and dictionary."""
    def __init__(self, scheme_path, bvals_path, dictionary_path, 
                 task_name="default", 
                 include_csf = False, M0_random = True):
        """Initialize the synthetizer with a given scheme and dictionary."""

        # Load DW-MRI protocol
        bvals = np.loadtxt(bvals_path) # NOT in SI units, in s/mm^2
        scheme = np.loadtxt(scheme_path, skiprows=1)  # only DWI, no b0s

        # Load MF dictionary
        dictionary_structure = mfu.loadmat(dictionary_path)

        self.__init_scheme(scheme, bvals)
        self.__init_MF_dictionary(dictionary_structure)

        self.num_fasc = 2
        self.task_name = task_name
        
        self.include_csf = include_csf
        
        self.M0_random = M0_random

    def __init_MF_dictionary(self, dictionary_structure):
        self.MF_dict = dictionary_structure
        self.MF_dict["fasc_propnames"] = [s.strip() for s in dictionary_structure['fasc_propnames']]

        self.interpolator = mfu.init_PGSE_multishell_interp(
            dictionary_structure['dictionary'],
            dictionary_structure['sch_mat'],
            dictionary_structure['orientation'])

    def __init_scheme(self, scheme, bvals):
        ind_b0 = np.where(bvals <= 1e-16)[0]
        ind_b = np.where(bvals > 1e-16)[0]
        num_B0 = ind_b0.size
        sch_mat_b0 = np.zeros((scheme.shape[0] + num_B0, scheme.shape[1]))
        sch_mat_b0[ind_b0, 4:] = scheme[0, 4:]
        sch_mat_b0[ind_b, :] = scheme
        self.scheme = sch_mat_b0
        self.TE = np.mean(self.scheme[:, 6])
        self.num_mris = sch_mat_b0.shape[0]

    def __generateRandomDirections(self, crossangle_min, dir1='fixed'):
        crossangle_min_rad = crossangle_min * np.pi / 180
        if dir1 == 'fixed':
            # fixed direction (do not take direction in the Z axis orientation)
            cyldir_1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        elif dir1 == 'random':
            norm1 = -1
            while norm1 <= 0:
                cyldir_1 = np.random.randn(3)
                norm1 = np.linalg.norm(cyldir_1, 2)
            cyldir_1 = cyldir_1 / norm1  # get unit vector
            if cyldir_1[2] < 0:
                # Force half-sphere with positive z
                cyldir_1 = -cyldir_1
        else:
            raise ValueError('dir1 should be either fixed or random')

        # cyldir2 - Enforce min crossing angle
        cyldir_2 = cyldir_1.copy()
        while np.abs(np.dot(cyldir_1, cyldir_2)) > np.cos(crossangle_min_rad):
            norm2 = -1
            while norm2 <= 0:
                cyldir_2 = np.random.randn(3)
                norm2 = np.linalg.norm(cyldir_2, 2)
            cyldir_2 = cyldir_2 / norm2
            if cyldir_2[2] < 0:
                # Force half-sphere with positive z
                cyldir_2 = - cyldir_2
        crossang = np.arccos(np.abs(np.dot(cyldir_1, cyldir_2))) * 180 / np.pi

        return cyldir_1, cyldir_2, crossang

    def generateStandardSet(self, num_samples, run_id=0, SNR_min=20, SNR_max=100, 
                            SNR_dist='uniform', nu_min=0.15, nu_csf_max = 0,
                            crossangle_min=30, nu1_dominant=True, random_seed=None):
        np.random.seed(random_seed)
        nus1 = []
        nuscsf = []
        SNRs = []
        nu_max = 1 - nu_min
        for i in range(num_samples):
            if nu1_dominant:
                nus1.append(1 - (nu_min + (nu_max - nu_min) / 2 * np.random.rand()))
            else:
                nus1.append((nu_min + (nu_max - nu_min) * np.random.rand()))
            if(self.include_csf):
                nuscsf.append(nu_csf_max * np.random.rand())
            
            if SNR_dist == 'triangular':
                SNR = np.random.triangular(SNR_min, SNR_min, SNR_max, 1)
            elif SNR_dist == 'uniform':
                SNR = np.random.uniform(SNR_min, SNR_max, 1)
            else:
                raise ValueError("Unknown SNR distribution %s" % SNR_dist)
            SNRs.append(SNR)

        data_dic = self.__generator(num_samples, nus1, nuscsf, SNRs, 
                                    crossangle_min, random_seed = random_seed,)

        data_dic["parameters"]["type"] = "standard"
        data_dic["parameters"]["run_id"] = run_id

        data_dic["parameters"]["SNR_dist"] = SNR_dist
        data_dic["parameters"]["SNR_min"] = SNR_min
        data_dic["parameters"]["SNR_max"] = SNR_max
        data_dic["parameters"]["nu_min"] = nu_min
        data_dic["parameters"]["nu_max"] = 1-nu_min
        data_dic["parameters"]["nu1_dominant"] = nu1_dominant
        data_dic["parameters"]["random_seed"] = random_seed
        data_dic['parameters']['nu_csf_max'] = nu_csf_max
        data_dic['parameters']['include_csf'] = self.include_csf

        return SynthetizerFit(data_dic)

    def generateStructuredSet(self, nu1_values=[0.5, 0.6, 0.7, 0.8, 0.9], 
                              nucsf_values = np.linspace(0.05,0.6,8),
                              include_csf = False,
                              SNR_values=[30, 50, 100], 
                              repetition=500, 
                              run_id=0, 
                              crossangle_min=30, 
                              random_seed=None):
        
        if(not(include_csf == (len(nucsf_values)>0))):
            raise ValueError('include_csf is {0} but len(nucsf_values) is {1}.'.format(include_csf, len(nucsf_values)))
        np.random.seed(random_seed)
        num_samples = 0
        SNRs = []
        nus1 = []
        nuscsf = []
        if(self.include_csf):
            for nu1 in nu1_values:
                for nucsf in nucsf_values:
                    for SNR in SNR_values:
                        for k in range(repetition):
                            num_samples += 1
                            nus1.append(nu1)
                            SNRs.append(SNR)
                            nuscsf.append(nucsf)
        else:
            for nu1 in nu1_values:
                for SNR in SNR_values:
                    for k in range(repetition):
                        num_samples += 1
                        nus1.append(nu1)
                        SNRs.append(SNR)
            

        data_dic = self.__generator(num_samples, nus1, nuscsf, SNRs, crossangle_min, random_seed = random_seed)

        data_dic["parameters"]["type"] = "structured"
        data_dic["parameters"]["run_id"] = run_id

        data_dic["parameters"]["nu1_values"] = nu1_values
        if(len(nucsf_values)>0):
            data_dic["parameters"]["nucsf_values"] = nucsf_values.tolist()
        data_dic["parameters"]["include_csf"] = include_csf
        data_dic["parameters"]["SNR_values"] = SNR_values
        data_dic["parameters"]["repetition"] = repetition
        data_dic["parameters"]["random_seed"] = random_seed

        data_dic["parameters"]["nu_min"] = min(1-np.max(nu1_values), np.min(nu1_values))
        data_dic["parameters"]["nu_max"] = 1 - data_dic["parameters"]["nu_min"]

        return SynthetizerFit(data_dic)

    def __generator(self, num_samples, 
                    nus1, nuscsf, 
                    SNRs, crossangle_min, 
                    random_seed = None):
        np.random.seed(random_seed)
        assert num_samples == len(nus1), "num_samples should be equal to the length of nus1"
        assert num_samples == len(SNRs), "num_samples should be equal to the length of SNRs"
        assert (num_samples == len(nuscsf)) or (0 == len(nuscsf)), "the number of nus for csf should be equal to 0 or num_samples"
        
        M0 = 500 #Only used if not(M0_random)
        num_coils = 1
        dir1_type = 'random'
        S0_max = np.max(self.MF_dict["S0_fasc"])

        # Prepare output arrays
        IDs = np.zeros((num_samples, self.num_fasc), dtype=np.int32)
        nus = np.zeros((num_samples, self.num_fasc))
        orientations = np.zeros((num_samples, self.num_fasc, 3))
        crossangles = np.zeros((num_samples))
        M0s = np.zeros(num_samples)

        DWI_image_store = np.zeros((self.num_mris, num_samples))
        DWI_noisy_store = np.zeros((self.num_mris, num_samples))

        if(len(nuscsf)>0):
            sig_csf = self.MF_dict['sig_csf']
            T2_csf = self.MF_dict['T2_csf']
            TE = self.TE

        print('Generating Voxels...')
        for i in tqdm(range(num_samples)):

            # 1. Generate random fasciles properties based on dictionary.

            nu1 = nus1[i]
            nu2 = 1 - nu1
            

            fasc_dir_1, fasc_dir_2, crossangle = self.__generateRandomDirections(crossangle_min, dir1=dir1_type)


            ID_1 = np.random.randint(0, self.MF_dict["num_atom"])
            ID_2 = np.random.randint(0, self.MF_dict["num_atom"])

            # 2. Generate noise less raw signal

            sig_fasc1 = mfu.interp_PGSE_from_multishell(self.scheme, ordir=self.MF_dict['orientation'], newdir=fasc_dir_1, sig_ms=self.MF_dict["dictionary"][:, ID_1], sch_mat_ms=self.MF_dict["sch_mat"])
            sig_fasc2 = mfu.interp_PGSE_from_multishell(self.scheme, ordir=self.MF_dict['orientation'], newdir=fasc_dir_2, sig_ms=self.MF_dict["dictionary"][:, ID_2], sch_mat_ms=self.MF_dict["sch_mat"])

            if(len(nuscsf) == 0):
                DWI_image = nu1 * sig_fasc1 + nu2 * sig_fasc2
                DWI_image_store[:, i] = DWI_image
            else:
                nucsf = nuscsf[i]
                DWI_image = (1-nucsf) * (nu1 * sig_fasc1 + nu2 * sig_fasc2) + nucsf * sig_csf #np.exp(-self.TE/T2_csf) * np.exp(-self.MF_dict['bvals'] * sig_csf)
                DWI_image_store[:, i] = DWI_image

            # 3. Add noise to raw signal

            # M0 random
            if(self.M0_random):
                M0 = float(np.random.randint(500, 5000))

            sigma_g = S0_max / SNRs[i]
            DWI_image_noisy = mfu.gen_SoS_MRI(DWI_image, sigma_g, num_coils)
            DWI_noisy_store[:, i] = M0*DWI_image_noisy 

            # Store remaining parameters
            IDs[i, :] = np.array([ID_1, ID_2])
            nus[i, :] = np.array([nu1, nu2])
            M0s[i] = M0
            orientations[i, 0, :] = fasc_dir_1
            orientations[i, 1, :] = fasc_dir_2
            crossangles[i] = crossangle

        # Create dictionary containing all the information

        DWI_dict = {'DWI_image_store': DWI_image_store,
                    'DWI_noisy_store': DWI_noisy_store,
                    'M0s': M0s,
                    'IDs': IDs,
                    'nus': nus,
                    'nuscsf' : np.array(nuscsf), #Storing the csf volume fractions
                    'orientations': orientations,
                    'SNRs': SNRs,
                    'crossangles': crossangles,
                    'parameters': {
                        'task_name': self.task_name,
                        'M0_random': self.M0_random,
                        'dir1_type': dir1_type,
                        'crossangle_min': crossangle_min,
                        'num_samples': num_samples,
                        'num_coils': num_coils,
                        'num_fasc': self.num_fasc,
                        'scheme': self.scheme,
                        'MF_dict': self.MF_dict,
                        },
                    }

        return DWI_dict


class SynthetizerFit:

    def __init__(self, data_dic):
        self.data_dic = data_dic

    def save(self, basepath, force_overwrite = False):
        type_ = self.data_dic['parameters']['type']
        task_name = self.data_dic['parameters']['task_name']
        run_id = self.data_dic['parameters']['run_id']

        output_path = os.path.join(basepath, "synthetizer", f"type-{type_}", "raw")
        # Create folder if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = f"type-{type_}_task-{task_name}_run-{run_id}_raw"
        
        if(not(force_overwrite) and os.path.exists(os.path.join(output_path, filename + ".pickle"))):
            raise ValueError('An identical synthetizer already exists. If you want to overwrite it, specify force_overwrite = True') 
        with open(os.path.join(output_path, filename + ".pickle"), 'wb') as handle:
            pickle.dump(self.data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = self.data_dic['parameters'].copy()

        del metadata['MF_dict']
        del metadata['scheme']

        with open(os.path.join(output_path,filename+'.json'), 'w') as fp:
            json.dump(metadata, fp, indent=4)

    def saveQCPlot(self, basepath):
        type_ = self.data_dic['parameters']['type']
        task_name = self.data_dic['parameters']['task_name']
        run_id = self.data_dic['parameters']['run_id']

        output_path = os.path.join(basepath, "synthetizer", f"type-{type_}","raw")
        # Create folder if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = f"type-{type_}_task-{task_name}_run-{run_id}_raw.png"

        self.__plotQC(os.path.join(output_path,filename))

    def __plotQC(self, filename):

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle('Quality control', fontsize=20)

        # Plot SNR distribution
        axs[0, 0].hist(self.data_dic['SNRs'], bins=20, density=True)
        axs[0, 0].set_title('SNR distribution', fontsize=16)
        axs[0, 0].set_xlabel('SNR', fontsize=16)
        axs[0, 0].set_ylabel('Density', fontsize=16)

        # Plot M0 distribution
        axs[0, 1].hist(self.data_dic['M0s'], bins=20, density=True)
        axs[0, 1].set_title('M0 distribution', fontsize=16)
        axs[0, 1].set_xlabel('M0', fontsize=16)
        axs[0, 1].set_ylabel('Density', fontsize=16)

        # Plot nu1 distribution
        axs[1, 0].hist(self.data_dic['nus'][:, 0], bins=20, density=True)
        axs[1, 0].set_title('nu1 distribution', fontsize=16)
        axs[1, 0].set_xlabel('nu1', fontsize=16)
        axs[1, 0].set_ylabel('Density', fontsize=16)

        # Plot crossing angle distribution
        axs[1, 1].hist(self.data_dic['crossangles'], bins=20, density=True)
        axs[1, 1].set_title('Crossing angle distribution', fontsize=16)
        axs[1, 1].set_xlabel('Crossing angle', fontsize=16)
        axs[1, 1].set_ylabel('Density', fontsize=16)

        plt.savefig(filename)
        plt.close()
