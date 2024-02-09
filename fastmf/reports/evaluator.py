import os
import json
import matplotlib.pyplot as plt
import pickle
import torch
import fastmf.models.MLP_Split as MLP_Split
import fastmf.models.MLP_FullyLearned as MLP_Full
from sklearn.linear_model import LinearRegression

import numpy as np

def MinMaxScaler(x, minis, maxis, inverse = False):
    if(maxis.shape[0]>x.shape[1]):
        maxis = maxis[:-1]
        minis = minis[:-1]
    if(inverse):
        a = maxis - minis
        b = minis
    else:
        a = 1/(maxis - minis)
        b = - minis * a
    
    return x*a + b

class Evaluator:
    def __init__(self, base_path, task_name, session_id, run_id, output_path=None, metadata_Hybrid=None,
                 model_state_dict_Hybrid=None, metadata_Full=None, model_state_dict_Full=None, device='cpu',
                 scaling_fn = None, normalization_set= None, orientation_estimate = "CSD"):

        self.base_path = base_path
        self.task_name = task_name
        if output_path is None:
            output_path = os.path.join(base_path,"evaluator",f"ses-{session_id}")
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        assert normalization_set is None or (metadata_Hybrid is None and metadata_Full is None), "normalization_set is mutually exclusive with metadata_Hybrid and ùetadata_Full"
        assert (metadata_Hybrid is not None and model_state_dict_Hybrid is not None) or (metadata_Hybrid is None and model_state_dict_Hybrid is None), "metadata_Hybrid and model_state_dict_Hybrid must be both None or both not None"
        assert (metadata_Full is not None and model_state_dict_Full is not None) or (metadata_Full is None and model_state_dict_Full is None), "metadata_Full and model_state_dict_Full must be both None or both not None"
        assert orientation_estimate in ["CSD", "MSMTCSD", "GROUNDTRUTH"], "orientation_estimate must be either CSD, MSMTCSD or GROUNDTRUTH"
        self.orientation_estimate = orientation_estimate

        if normalization_set:
            self.in_normalization_Hybrid = normalization_set["in_normalization_Hybrid"]
            self.target_normalization_Hybrid = normalization_set["target_normalization_Hybrid"]
            self.in_normalization_Full = normalization_set["in_normalization_Full"]
            self.target_normalization_Full = normalization_set["target_normalization_Full"]


        if metadata_Hybrid is not None:
            with open(metadata_Hybrid, 'r') as f:
                self.metadata_Hybrid = json.load(f)

            assert orientation_estimate == self.metadata_Hybrid["orientation"], "orientation_estimate must be the same as the one used to train the model"

            self.in_normalization_Hybrid = self.metadata_Hybrid["in_normalization"]
            self.target_normalization_Hybrid = self.metadata_Hybrid["target_normalization"]

            self.model_Hybrid = MLP_Split.Network(self.metadata_Hybrid['split_architecture'],
                                                  self.metadata_Hybrid['final_architecture'])

        else:
            self.metadata_Hybrid = None

        if metadata_Full is not None:
            with open(metadata_Full, 'r') as f:
                self.metadata_Full = json.load(f)
            self.in_normalization_Full = self.metadata_Full["in_normalization"]
            self.target_normalization_Full = self.metadata_Full["target_normalization"]

            self.model_Full = MLP_Full.Network(self.metadata_Full['architecture'])
        else:
            self.metadata_Full = None

        self.session_id = session_id  # I prefer this way rather than self.metadata_Hybrid["session_id"], the purpose of which is tracking the data used to train the model
        self._loadStructuredTestSet(task_name, run_id)  # Contains MF loading

        if self.metadata_Hybrid is not None:
            self.model_Hybrid.load_state_dict(torch.load(model_state_dict_Hybrid))
            self.model_Hybrid.eval()
            self.model_Hybrid.to(device)

            self.pred_hybrid = self.model_Hybrid(torch.from_numpy(self.input_NNLS.astype('float32'))).detach().numpy()

        if self.metadata_Full is not None:
            self.model_Full.load_state_dict(torch.load(model_state_dict_Full))
            self.model_Full.eval()
            self.model_Full.to(device)

            self.pred_full = self.model_Full(torch.from_numpy(
                self.input_SH.reshape(self.input_SH.shape[0], self.input_SH.shape[1] * self.input_SH.shape[2]).astype(
                    'float32'))).detach().numpy()

        if(scaling_fn is None):
            self.scaling_fn_full = lambda *args: args[0]
            self.scaling_fn_hybrid = lambda *args: args[0]
        elif(scaling_fn == 'MinMax'):
            scaler_path = os.path.join(base_path, 'scaler', 'scaler-minmax_ses-{0}_SH.pickle'.format(session_id))
            with open( scaler_path, 'rb') as file:
                scaler = pickle.load(file)

            self.maxis_full = scaler.data_max_
            self.minis_full = scaler.data_min_

            self.scaling_fn_full = lambda x, mi, ma, inverse:MinMaxScaler(x, mi, ma, inverse = inverse)
            
            scaler_path = os.path.join(base_path, 'scaler', 'scaler-minmax_ses-{0}_NNLS.pickle'.format(session_id))
            with open( scaler_path, 'rb') as file:
                scaler = pickle.load(file)

            self.maxis_hybrid = scaler.data_max_
            self.minis_hybrid = scaler.data_min_
            
            self.scaling_fn_hybrid = lambda x, mi, ma,  inverse:MinMaxScaler(x, mi, ma, inverse = inverse)
        

        self.scaled_target_NNLS = self.scaling_fn_hybrid(self.target_NNLS[:,:],
                                    self.minis_hybrid[:],
                                    self.maxis_hybrid[:],
                                    True)
        if metadata_Hybrid is not None:
            self.scaled_pred_hybrid = self.scaling_fn_hybrid(self.pred_hybrid[:,:],
                                        self.minis_hybrid[:],
                                        self.maxis_hybrid[:],
                                        True)


        self.scaled_target_SH = self.scaling_fn_full(self.target_SH[:, :],
                                                     self.minis_full[:],
                                                     self.maxis_full[:],
                                                     True)
        if metadata_Full is not None:
            self.scaled_pred_full = self.scaling_fn_full(self.pred_full[:,:],
                                        self.minis_full[:],
                                        self.maxis_full[:],
                                        True)

            self.SH_output_swapField = np.mean(np.abs(self.pred_full[:, 0:6] - self.target_SH[:, 0:6]) +
                                            np.abs(self.pred_full[:, 6:] - self.target_SH[:, 6:]), axis=1) > \
                                       np.mean(np.abs(self.pred_full[:, 0:6] - self.target_SH[:, 6:]) +
                                            np.abs(self.pred_full[:, 6:] - self.target_SH[:, 0:6]), axis=1)


    def swapField(self, input, swapField, props_len=None):

        if props_len is None:
            # Performs swapping of first three element with next three depending on est_orientations_swapped
            props_len = (1 + len(self.MF_dict["fasc_propnames"]))

        input = input.copy()
        swapField = swapField.astype(bool)
        (input[swapField, 0:props_len],
         input[swapField, props_len:props_len * 2]) = \
            (input[swapField, props_len:2 * props_len],
             input[swapField, 0:props_len])

        return input

    def __compute_MF_metrics_from_Generator(self, generator_Dict, include_csf = False):

        num_fasc = 2
        num_props = len(self.MF_dict["fasc_propnames"])

        #Structure: [nu_1, fvf_1, Diff_ex_1, nu_2, fvf_2, Diff_ex_2, csf]
        if include_csf:
            output_MF = np.zeros((len(generator_Dict["weights_non_negative"]), (num_fasc * (num_props + 1)) + 1))
        else:
            output_MF = np.zeros((len(generator_Dict["weights_non_negative"]), (num_fasc * (num_props + 1)) + 0))

        for i, (w_nnz, ind_subdic, ind_totdic) in enumerate(zip(generator_Dict["weights_non_negative"], generator_Dict["indices_subdict"], generator_Dict["indices_totdict"])):
            M0_vox = np.sum(w_nnz)
            if np.abs(M0_vox) > 0:
                # abs could be dropped bc w_nnz[k] >= 0 for all k
                nu = w_nnz / M0_vox
            else:
                nu = w_nnz
            if(len(nu)==3): #If csf is included : len(nu) == num_fasc + 1
                nu[0:2] = nu[0:2]/(1-nu[2])
            r = 0
            for k in range(0,num_fasc):
                # 0, 4
                output_MF[i, r] = nu[k]
                r=r+1
                for j in range(0,num_props):
                    output_MF[i, r] = self.MF_dict[self.MF_dict["fasc_propnames"][j]][ind_subdic[k]]
                    r = r + 1
            if include_csf:
                output_MF[i, r] = nu[-1]
                r = r + 1
            
        return output_MF

    def __plotBlandAltman(self, vec1, vec2, ax):
        diff = vec1 - vec2
        means = 0.5 * (vec1 + vec2)
        ax.plot(means,diff)
    def _loadStructuredTestSet(self, task_name, run_id):
        set_type = 'structured'
        
        # Load MF dict
        synthetizer_path = os.path.join(self.base_path, "synthetizer", f"type-{set_type}", "raw")
        filename_synthetizer = f"type-{set_type}_task-{task_name}_run-{run_id}_raw.pickle"
        with open(os.path.join(synthetizer_path, filename_synthetizer), 'rb') as handle:
            data_dict = pickle.load(handle)
        self.MF_dict = data_dict["parameters"]["MF_dict"]
        
        #Load synthetizer
        synth_path = os.path.join(self.base_path, "synthetizer", f"type-{set_type}", "raw")
        synth_name = 'type-{0}_task-{1}_run-{2}_raw'.format(set_type, task_name, run_id)
        with open(os.path.join(synth_path, synth_name + ".json"), 'rb') as handle:
            self.synth_metadata = json.load(handle)
        with open(os.path.join(synth_path, synth_name + ".pickle"), 'rb') as handle:
            self.synth = pickle.load(handle)
        include_csf = self.synth_metadata['include_csf']
        
        # Load Generator MF result
        MF_path = os.path.join(self.base_path, "generator", f"type-{set_type}", "MF")
        filename_MF_GROUNDTRUTH = f"type-{set_type}_task-{task_name}_run-{run_id}_orientation-GROUNDTRUTH_MF"
        filename_MF_MSMTCSD = f"type-{set_type}_task-{task_name}_run-{run_id}_orientation-MSMTCSD_MF"
        filename_MF_CSD = f"type-{set_type}_task-{task_name}_run-{run_id}_orientation-CSD_MF"
        with open(os.path.join(MF_path, filename_MF_GROUNDTRUTH + ".pickle"), 'rb') as handle:
            self.MF_GROUNDTRUTH = pickle.load(handle)
        if os.path.exists(os.path.join(MF_path, filename_MF_MSMTCSD + ".pickle")):
            with open(os.path.join(MF_path, filename_MF_MSMTCSD + ".pickle"), 'rb') as handle:
                self.MF_MSMTCSD = pickle.load(handle)
        else:
            self.MF_MSMTCSD = None
        with open(os.path.join(MF_path, filename_MF_CSD + ".pickle"), 'rb') as handle:
            self.MF_CSD = pickle.load(handle)
        
        self.MF_GROUNDTRUTH_output = self.__compute_MF_metrics_from_Generator(self.MF_GROUNDTRUTH, include_csf = include_csf)
        if self.MF_MSMTCSD is not None:
            self.MF_MSMTCSD_output = self.__compute_MF_metrics_from_Generator(self.MF_MSMTCSD, include_csf = include_csf)
        self.MF_CSD_output = self.__compute_MF_metrics_from_Generator(self.MF_CSD, include_csf=include_csf)
        if self.MF_MSMTCSD is not None:
            self.MF_MSMTCSD_output_swapField = self.MF_MSMTCSD["est_orientations_swapped"]
        self.MF_CSD_output_swapField = self.MF_CSD["est_orientations_swapped"]

        # Load Generator NoisyMF result
        NoisyMF_path = os.path.join(self.base_path, "generator", f"type-{set_type}", "NoisyMF")
        filename_NoisyMF_GROUNDTRUTH = f"type-{set_type}_task-{task_name}_run-{run_id}_orientation-GROUNDTRUTH_NoisyMF"
        filename_NoisyMF_MSMTCSD = f"type-{set_type}_task-{task_name}_run-{run_id}_orientation-MSMTCSD_NoisyMF"
        filename_NoisyMF_CSD = f"type-{set_type}_task-{task_name}_run-{run_id}_orientation-CSD_NoisyMF"
        if os.path.exists(os.path.join(NoisyMF_path, filename_NoisyMF_GROUNDTRUTH + ".pickle")):
            with open(os.path.join(NoisyMF_path, filename_NoisyMF_GROUNDTRUTH + ".pickle"), 'rb') as handle:
                self.NoisyMF_GROUNDTRUTH = pickle.load(handle)
            self.NoisyMF_GROUNDTRUTH_output = self.__compute_MF_metrics_from_Generator(self.NoisyMF_GROUNDTRUTH,
                                                                                  include_csf=include_csf)
        else:
            self.NoisyMF_GROUNDTRUTH = None
        if os.path.exists(os.path.join(NoisyMF_path, filename_NoisyMF_MSMTCSD + ".pickle")):
            with open(os.path.join(NoisyMF_path, filename_NoisyMF_MSMTCSD + ".pickle"), 'rb') as handle:
                self.NoisyMF_MSMTCSD = pickle.load(handle)
            self.NoisyMF_MSMTCSD_output = self.__compute_MF_metrics_from_Generator(self.NoisyMF_MSMTCSD,
                                                                            include_csf=include_csf)
            self.NoisyMF_MSMTCSD_output_swapField = self.NoisyMF_MSMTCSD["est_orientations_swapped"]
        else:
            self.NoisyMF_MSMTCSD = None
        if os.path.exists(os.path.join(NoisyMF_path, filename_NoisyMF_CSD + ".pickle")):
            with open(os.path.join(NoisyMF_path, filename_NoisyMF_CSD + ".pickle"), 'rb') as handle:
                self.NoisyMF_CSD = pickle.load(handle)
            self.NoisyMF_CSD_output = self.__compute_MF_metrics_from_Generator(self.NoisyMF_CSD,
                                                                            include_csf=include_csf)
            self.NoisyMF_CSD_output_swapField = self.NoisyMF_CSD["est_orientations_swapped"]
        else:
            self.NoisyMF_MSMTCSD = None

        # Load generator NNLS swap result

        NNLS_path = os.path.join(self.base_path, "generator", f"type-{set_type}", "NNLS")
        filename_NNLS = f"type-{set_type}_task-{task_name}_run-{run_id}_orientation-{self.orientation_estimate}_NNLS"
        with open(os.path.join(NNLS_path, filename_NNLS + ".pickle"), 'rb') as handle:
            self.NNLS_generator = pickle.load(handle)
        self.NNLS_output_swapField = self.NNLS_generator["est_orientations_swapped"]


        # Load Formatter result
        SH_path = os.path.join(self.base_path, "formatter", f"type-{set_type}", "SH")
        NNLS_path = os.path.join(self.base_path, "formatter", f"type-{set_type}", "NNLS")

        filename_target = f"type-{set_type}_task-{self.task_name}_ses-{self.session_id}_scaler-{self.target_normalization_Full}_set-testing_target"
        self.target_SH = np.load(os.path.join(SH_path, filename_target + ".npy"))
        filename_target = f"type-{set_type}_task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_scaler-{self.target_normalization_Hybrid}_set-testing_target"
        self.target_NNLS = np.load(os.path.join(NNLS_path, filename_target + ".npy"))

        filename_input = f"type-{set_type}_task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_normalization-{self.in_normalization_Hybrid}_set-testing"
        self.input_NNLS = np.load(os.path.join(NNLS_path, filename_input + "_NNLS.npy"))
        filename_input = f"type-{set_type}_task-{self.task_name}_ses-{self.session_id}_normalization-{self.in_normalization_Full}_set-testing"
        self.input_SH = np.load(os.path.join(SH_path, filename_input + "_SH.npy"))

    def _meanAbsErrByNu(self, vec1, vec2, ranges):
        '''For each r in ranges, compare vec1[r,:] and vec2[r,:] '''
        
        if(len(vec1.shape) == 1 and len(vec2.shape)==1):
            vec1 = vec1[:,np.newaxis]
            vec2 = vec2[:,np.newaxis]
            
        meanAbsErrByNuVec = np.zeros((ranges.shape[0], vec1.shape[1]), dtype = 'float32')
        for k,r in enumerate(ranges):
            meanAbsErrByNuVec[k,:] = np.mean(np.abs(vec1[r,:] - vec2[r,:]), axis = 0)
            
        return meanAbsErrByNuVec
    
    def _D2ByNu(self, pred, target, ranges):
        '''For each r in ranges, compare vec1[r,:] and vec2[r,:] '''
        
        if(len(pred.shape) == 1 and len(target.shape)==1):
            pred = pred[:,np.newaxis]
            target = target[:,np.newaxis]
            
        D2ByNuVec = np.zeros((ranges.shape[0], pred.shape[1]), dtype = 'float32')
        gt_medians = np.median(target, axis = 0)
        for k,r in enumerate(ranges):

            num = np.mean(np.abs(pred[r,:] -target[r,:]), axis = 0)
            den = np.mean(np.abs(target[r,:] - gt_medians[np.newaxis,:]))
            D2ByNuVec[k,:] = 1 -  num/den 
            # D2ByNuVec[k,:] = d2_absolute_error_score(target[r,:], pred[r,:])
            
        return D2ByNuVec
        
    def __plot_Distributions(self):
        
        prop_names = ['$\\nu$', '$D_{ex}$', '$fvf$']
        nprops = len(prop_names)
        idx_prop_full = [0,4,5,6,10,11]
        n = self.target_NNLS.shape[0]
        ylims = [[0,5],[0,2e9],[0,2.5]]

        if self.metadata_Hybrid is not None:
            #Hybrid method
            fig, axs = plt.subplots(nrows=2, ncols=nprops, sharex=False)
            fig.suptitle('Hybrid method')

            tar = self.swapField(self.scaled_target_NNLS, self.NNLS_output_swapField)
            pred = self.scaled_pred_hybrid

            for k in range(nprops*2):
                prop_gt_values = np.unique(tar[:,k])
                bins_edges = prop_gt_values - (prop_gt_values[1]-prop_gt_values[0])*0.5
                bins_edges = np.concatenate((bins_edges,
                                            np.array([prop_gt_values[-1]+(prop_gt_values[1]-prop_gt_values[0])*0.5])))
                hist = np.histogram(tar[:,k], bins = bins_edges)
                bin_middles = (hist[1][1:] + hist[1][0:-1])/2
                bin_size = hist[1][1] - hist[1][0]

                if(k//3 == 0):
                    axs[k//3,k%3].set_title(prop_names[k%3])

                if(k==3):
                    axs[k//3,k%3].set_xlabel('Parameter value')


                axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'k-h', label = 'Groundtruth')

                hist = np.histogram(pred[:,k], bins = bins_edges )
                bin_middles = (hist[1][1:] + hist[1][0:-1])/2
                bin_size = hist[1][1] - hist[1][0]

                axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'b-h', label = 'Prediction')
                axs[k//3,k%3].set_ylim(ylims[k%3])


                if(k == 5):
                    axs[1,2].legend(loc = (0.2,-0.5))
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_method-Hybrid_distrib.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_method-Hybrid_distrib.eps"),
                        bbox_inches='tight', format='eps')


        #SH method
        if self.metadata_Full is not None:
            fig, axs = plt.subplots(nrows=2, ncols=nprops, sharex=False)
            fig.suptitle('FullyLearned method')
            prop_idx = [0,4,5,6,10,11]
            tar = self.scaled_target_SH[:,prop_idx]
            pred = self.scaled_pred_full[:,prop_idx]

            for k in range(nprops*2):
                prop_gt_values = np.unique(tar[:,k])
                bins_edges = prop_gt_values - (prop_gt_values[1]-prop_gt_values[0])*0.5
                bins_edges = np.concatenate((bins_edges,
                                            np.array([prop_gt_values[-1]+(prop_gt_values[1]-prop_gt_values[0])*0.5])))
                hist = np.histogram(tar[:,k], bins = bins_edges)
                bin_middles = (hist[1][1:] + hist[1][0:-1])/2
                bin_size = hist[1][1] - hist[1][0]

                if(k//3 == 0):
                    axs[k//3,k%3].set_title(prop_names[k%3])

                if(k==3):
                    axs[k//3,k%3].set_xlabel('Parameter value')

                axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'k-*', label = 'Groundtruth')

                hist = np.histogram(pred[:,k], bins = bins_edges )
                bin_middles = (hist[1][1:] + hist[1][0:-1])/2
                bin_size = hist[1][1] - hist[1][0]

                axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'b-*', label = 'Prediction')
                axs[k//3,k%3].set_ylim(ylims[k%3])
                if(k == 5):
                    axs[1,2].legend(loc = (0.2,-0.5))

            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_method-SH_distrib.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_method-SH_distrib.eps"),
                        bbox_inches='tight', format='eps')


        #MF MSMTCSD
        if self.MF_MSMTCSD is not None:
            fig, axs = plt.subplots(nrows=2, ncols=nprops, sharex=False)
            fig.suptitle('MF + MSMT CSD')
            prop_idx = [0,4,5,6,10,11]
            tar = self.swapField(self.scaled_target_NNLS[:,:],self.MF_MSMTCSD_output_swapField)
            pred = self.MF_MSMTCSD_output

            for k in range(nprops*2):
                prop_gt_values = np.unique(tar[:,k])
                bins_edges = prop_gt_values - (prop_gt_values[1]-prop_gt_values[0])*0.5
                bins_edges = np.concatenate((bins_edges,
                                            np.array([prop_gt_values[-1]+(prop_gt_values[1]-prop_gt_values[0])*0.5])))
                hist = np.histogram(tar[:,k], bins = bins_edges)
                bin_middles = (hist[1][1:] + hist[1][0:-1])/2
                bin_size = hist[1][1] - hist[1][0]

                if(k//3 == 0):
                    axs[k//3,k%3].set_title(prop_names[k%3])

                if(k==3):
                    axs[k//3,k%3].set_xlabel('Parameter value')


                axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'k-*', label = 'Groundtruth')

                hist = np.histogram(pred[:,k], bins = bins_edges )
                bin_middles = (hist[1][1:] + hist[1][0:-1])/2
                bin_size = hist[1][1] - hist[1][0]

                axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'b-*', label = 'Prediction')
                axs[k//3,k%3].set_ylim(ylims[k%3])
                if(k == 5):
                    axs[1,2].legend(loc = (0.2,-0.5))

            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-MSMTCSD_method-MF_distrib.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-MSMTCSD_method-MF_distrib.eps"),
                        bbox_inches='tight', format='eps')

            # MF CSD
            if self.MF_CSD is not None:
                fig, axs = plt.subplots(nrows=2, ncols=nprops, sharex=False)
                fig.suptitle('MF + CSD')
                prop_idx = [0, 4, 5, 6, 10, 11]
                tar = self.swapField(self.scaled_target_NNLS[:, :], self.MF_CSD_output_swapField)
                pred = self.MF_CSD_output

                for k in range(nprops * 2):
                    prop_gt_values = np.unique(tar[:, k])
                    bins_edges = prop_gt_values - (prop_gt_values[1] - prop_gt_values[0]) * 0.5
                    bins_edges = np.concatenate((bins_edges,
                                                 np.array([prop_gt_values[-1] + (
                                                             prop_gt_values[1] - prop_gt_values[0]) * 0.5])))
                    hist = np.histogram(tar[:, k], bins=bins_edges)
                    bin_middles = (hist[1][1:] + hist[1][0:-1]) / 2
                    bin_size = hist[1][1] - hist[1][0]

                    if (k // 3 == 0):
                        axs[k // 3, k % 3].set_title(prop_names[k % 3])

                    if (k == 3):
                        axs[k // 3, k % 3].set_xlabel('Parameter value')

                    axs[k // 3, k % 3].plot(bin_middles, hist[0] / n / bin_size, 'k-*', label='Groundtruth')

                    hist = np.histogram(pred[:, k], bins=bins_edges)
                    bin_middles = (hist[1][1:] + hist[1][0:-1]) / 2
                    bin_size = hist[1][1] - hist[1][0]

                    axs[k // 3, k % 3].plot(bin_middles, hist[0] / n / bin_size, 'b-*', label='Prediction')
                    axs[k // 3, k % 3].set_ylim(ylims[k % 3])
                    if (k == 5):
                        axs[1, 2].legend(loc=(0.2, -0.5))

                plt.savefig(os.path.join(self.output_path,
                                         f"task-{self.task_name}_ses-{self.session_id}_orientation-CSD_method-MF_distrib.png"),
                            bbox_inches='tight')
                plt.savefig(os.path.join(self.output_path,
                                         f"task-{self.task_name}_ses-{self.session_id}_orientation-CSD_method-MF_distrib.eps"),
                            bbox_inches='tight', format='eps')

        #MF GT
        fig, axs = plt.subplots(nrows=2, ncols=nprops, sharex=False)
        fig.suptitle('MF + GROUNDTRUTH')
        tar = self.scaled_target_NNLS[:,:]
        pred = self.MF_GROUNDTRUTH_output

        for k in range(nprops*2):
            prop_gt_values = np.unique(tar[:,k])
            bins_edges = prop_gt_values - (prop_gt_values[1]-prop_gt_values[0])*0.5
            bins_edges = np.concatenate((bins_edges, 
                                        np.array([prop_gt_values[-1]+(prop_gt_values[1]-prop_gt_values[0])*0.5])))
            hist = np.histogram(tar[:,k], bins = bins_edges)
            bin_middles = (hist[1][1:] + hist[1][0:-1])/2
            bin_size = hist[1][1] - hist[1][0]
            
            if(k//3 == 0):
                axs[k//3,k%3].set_title(prop_names[k%3])
            
            if(k==3):
                axs[k//3,k%3].set_xlabel('Parameter value')
                

            axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'k-*', label = 'Groundtruth')
            
            hist = np.histogram(pred[:,k], bins = bins_edges )
            bin_middles = (hist[1][1:] + hist[1][0:-1])/2
            bin_size = hist[1][1] - hist[1][0]
            
            axs[k//3,k%3].plot(bin_middles,hist[0]/n/bin_size, 'b-*', label = 'Prediction')
            axs[k//3,k%3].set_ylim(ylims[k%3])
            if(k == 5):
                axs[1,2].legend(loc = (0.2,-0.5))

        plt.savefig(os.path.join(self.output_path,
                                 f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-MF_distrib.png"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path,
                                 f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-MF_distrib.eps"),
                    bbox_inches='tight', format='eps')



        #NoisyMF GT
        if self.NoisyMF_GROUNDTRUTH is not None:
            fig, axs = plt.subplots(nrows=2, ncols=nprops, sharex=False)
            fig.suptitle('NoisyMF + GROUNDTRUTH')
            tar = self.scaled_target_NNLS[:, :]

            pred = self.NoisyMF_GROUNDTRUTH_output

            for k in range(nprops * 2):
                prop_gt_values = np.unique(tar[:, k])
                bins_edges = prop_gt_values - (prop_gt_values[1] - prop_gt_values[0]) * 0.5
                bins_edges = np.concatenate((bins_edges,
                                             np.array(
                                                 [prop_gt_values[-1] + (prop_gt_values[1] - prop_gt_values[0]) * 0.5])))
                hist = np.histogram(tar[:, k], bins=bins_edges)
                bin_middles = (hist[1][1:] + hist[1][0:-1]) / 2
                bin_size = hist[1][1] - hist[1][0]

                if (k // 3 == 0):
                    axs[k // 3, k % 3].set_title(prop_names[k % 3])

                if (k == 3):
                    axs[k // 3, k % 3].set_xlabel('Parameter value')

                axs[k // 3, k % 3].plot(bin_middles, hist[0] / n / bin_size, 'k-*', label='Groundtruth')

                hist = np.histogram(pred[:, k], bins=bins_edges)
                bin_middles = (hist[1][1:] + hist[1][0:-1]) / 2
                bin_size = hist[1][1] - hist[1][0]

                axs[k // 3, k % 3].plot(bin_middles, hist[0] / n / bin_size, 'b-*', label='Prediction')
                axs[k // 3, k % 3].set_ylim(ylims[k % 3])
                if (k == 5):
                    axs[1, 2].legend(loc=(0.2, -0.5))

            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-NoisyMF_distrib.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-NoisyMF_distrib.eps"),
                        bbox_inches='tight', format='eps')

        return 1
    
    
    def __plot_GTvsPrediction(self):
        
        prop_names = ['$\\nu$', '$D_{ex}$', '$fvf$']
        nprops = len(prop_names)
        ylims = [(0.45,1.05),(0,3e-9), (0,0.85),
                 (0,0.55),(0,3e-9), (0,0.85),
                 ]
        
        # Hybrid method
        if self.metadata_Hybrid is not None:
            tar = self.swapField(self.scaled_target_NNLS[:,:],self.NNLS_output_swapField)
            pred = self.scaled_pred_hybrid[:,:]


            nrows = 2
            ncols = nprops
            fig,axs = plt.subplots(nrows,ncols)
            fig.suptitle('Hybrid MLP')
            #axs.set_title('Hybrid MLP')
            for j,p in enumerate(prop_names):
                if(j == 0):
                    axs[0,j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1,j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0,j].hexbin(tar[:,j],
                                pred[:,j],
                                gridsize = 16, cmap = 'Greys')
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])],
                              [np.min(tar[:,j]),np.max(tar[:,j])],   '--k')


                axs[0,j].set_title(prop_names[j])

                axs[1,j].set_xlabel('Groundtruth', loc = 'left')
                axs[1,j].hexbin(tar[:,j+nprops],pred[:,j+nprops],
                                gridsize = 16, cmap = 'Greys' )

                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],
                              [np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],   '--k')
                #Plot linear regressions of pred = f(target)
                reg = LinearRegression().fit(tar[:,j:j+1], pred[:,j:j+1])
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])],
                              [np.min(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0],
                               np.max(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0]],
                              '--r')

                reg = LinearRegression().fit(tar[:,j+nprops:j+nprops+1], pred[:,j+nprops:j+nprops+1])
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],
                              [np.min(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0],
                               np.max(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0]],
                              '--r')
                axs[0,j].set_ylim(ylims[j])
                axs[1,j].set_ylim(ylims[j+nprops])
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_method-Hybrid_gtvspred.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_method-Hybrid_gtvspred.eps"),
                        bbox_inches='tight', format='eps')

        # Fully learned method
        if self.metadata_Full is not None:
            prop_idx = [0,4,5,6,10,11]
            tar = self.scaled_target_SH[:,prop_idx]


            pred = self.scaled_pred_full[:,prop_idx]

            nrows = 2
            ncols = 3
            fig,axs = plt.subplots(nrows,ncols)
            fig.suptitle('FullyLearned MLP')
            #axs.set_title('Hybrid MLP')
            for j,p in enumerate(prop_names):
                if(j == 0):
                    axs[0,j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1,j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0,j].hexbin(tar[:,j],
                                pred[:,j],
                                gridsize = 16, cmap = 'Greys')
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])],
                              [np.min(tar[:,j]),np.max(tar[:,j])],   '--k')


                axs[0,j].set_title(prop_names[j])

                axs[1,j].set_xlabel('Groundtruth', loc = 'left')
                axs[1,j].hexbin(tar[:,j+nprops],pred[:,j+nprops],
                                gridsize = 16, cmap = 'Greys' )
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],
                              [np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],   '--k')

                #Plot linear regressions of pred = f(target)
                reg = LinearRegression().fit(tar[:,j:j+1], pred[:,j:j+1])
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])],
                              [np.min(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0],
                               np.max(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0]],
                              '--r')

                reg = LinearRegression().fit(tar[:,j+nprops:j+nprops+1], pred[:,j+nprops:j+nprops+1])
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],
                              [np.min(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0],
                               np.max(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0]],
                              '--r')
                axs[0,j].set_ylim(ylims[j])
                axs[1,j].set_ylim(ylims[j+nprops])
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_method-SH_gtvspred.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_method-SH_gtvspred.eps"),
                        bbox_inches='tight', format='eps')
        
        # Exhaustive MF (msmtcsd)
        if self.MF_MSMTCSD is not None:
            tar = self.swapField(self.scaled_target_NNLS[:,:],self.MF_MSMTCSD_output_swapField)
            pred = self.MF_MSMTCSD_output


            nrows = 2
            ncols = nprops
            fig,axs = plt.subplots(nrows,ncols)
            fig.suptitle('Exhaustive MF + MSMT CSD')
            #axs.set_title('Hybrid MLP')
            for j,p in enumerate(prop_names):
                if(j == 0):
                    axs[0,j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1,j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0,j].hexbin(tar[:,j],
                                pred[:,j],
                                gridsize = 16, cmap = 'Greys')
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])],
                              [np.min(tar[:,j]),np.max(tar[:,j])],   '--k')
                axs[0,j].set_title(prop_names[j])
                axs[1,j].set_xlabel('Groundtruth', loc = 'left')
                axs[1,j].hexbin(tar[:,j+nprops],pred[:,j+nprops],
                                gridsize = 16, cmap = 'Greys' )
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],
                              [np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],   '--k')

                #Plot linear regressions of pred = f(target)
                reg = LinearRegression().fit(tar[:,j:j+1], pred[:,j:j+1])
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])],
                              [np.min(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0],
                               np.max(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0]],
                              '--r')

                reg = LinearRegression().fit(tar[:,j+nprops:j+nprops+1], pred[:,j+nprops:j+nprops+1])
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],
                              [np.min(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0],
                               np.max(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0]],
                              '--r')
                axs[0,j].set_ylim(ylims[j])
                axs[1,j].set_ylim(ylims[j+nprops])
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-MSMTCSD_method-MF_gtvspred.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-MSMTCSD_method-MF_gtvspred.eps"),
                        bbox_inches='tight', format='eps')
        # Exhaustive MF (csd)
        if self.MF_CSD is not None:
            tar = self.swapField(self.scaled_target_NNLS[:, :], self.MF_CSD_output_swapField)
            pred = self.MF_CSD_output

            nrows = 2
            ncols = nprops
            fig, axs = plt.subplots(nrows, ncols)
            fig.suptitle('Exhaustive MF + CSD')
            # axs.set_title('Hybrid MLP')
            for j, p in enumerate(prop_names):
                if (j == 0):
                    axs[0, j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1, j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0, j].hexbin(tar[:, j],
                                 pred[:, j],
                                 gridsize=16, cmap='Greys')
                axs[0, j].plot([np.min(tar[:, j]), np.max(tar[:, j])],
                               [np.min(tar[:, j]), np.max(tar[:, j])], '--k')
                axs[0, j].set_title(prop_names[j])
                axs[1, j].set_xlabel('Groundtruth', loc='left')
                axs[1, j].hexbin(tar[:, j + nprops], pred[:, j + nprops],
                                 gridsize=16, cmap='Greys')
                axs[1, j].plot([np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])],
                               [np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])], '--k')

                # Plot linear regressions of pred = f(target)
                reg = LinearRegression().fit(tar[:, j:j + 1], pred[:, j:j + 1])
                axs[0, j].plot([np.min(tar[:, j]), np.max(tar[:, j])],
                               [np.min(tar[:, j]) * reg.coef_[0, 0] + reg.intercept_[0],
                                np.max(tar[:, j]) * reg.coef_[0, 0] + reg.intercept_[0]],
                               '--r')

                reg = LinearRegression().fit(tar[:, j + nprops:j + nprops + 1], pred[:, j + nprops:j + nprops + 1])
                axs[1, j].plot([np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])],
                               [np.min(tar[:, j + nprops]) * reg.coef_[0, 0] + reg.intercept_[0],
                                np.max(tar[:, j + nprops]) * reg.coef_[0, 0] + reg.intercept_[0]],
                               '--r')
                axs[0, j].set_ylim(ylims[j])
                axs[1, j].set_ylim(ylims[j + nprops])
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-CSD_method-MF_gtvspred.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-CSD_method-MF_gtvspred.eps"),
                        bbox_inches='tight', format='eps')

        # Exhaustive MF (groundtruth)
        tar = self.scaled_target_NNLS[:,:]
        pred = self.MF_GROUNDTRUTH_output
        
        nrows = 2
        ncols = nprops
        fig,axs = plt.subplots(nrows,ncols)
        fig.suptitle('Exhaustive MF + Groundtruth')
        for j,p in enumerate(prop_names):
            if(j == 0):
                axs[0,j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                axs[1,j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
            axs[0,j].hexbin(tar[:,j], 
                            pred[:,j], 
                            gridsize = 16, cmap = 'Greys')
            axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])], 
                          [np.min(tar[:,j]),np.max(tar[:,j])],   '--k')
            axs[0,j].set_title(prop_names[j])
            axs[1,j].set_xlabel('Groundtruth', loc = 'left')
            axs[1,j].hexbin(tar[:,j+nprops],pred[:,j+nprops], 
                            gridsize = 16, cmap = 'Greys' )
            axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])], 
                          [np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],   '--k')
            
            #Plot linear regressions of pred = f(target)
            reg = LinearRegression().fit(tar[:,j:j+1], pred[:,j:j+1])
            axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])], 
                          [np.min(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0],
                           np.max(tar[:,j])*reg.coef_[0,0] + reg.intercept_[0]],  
                          '--r')
            
            reg = LinearRegression().fit(tar[:,j+nprops:j+nprops+1], pred[:,j+nprops:j+nprops+1])
            axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])], 
                          [np.min(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0],
                           np.max(tar[:,j+nprops])*reg.coef_[0,0] + reg.intercept_[0]],  
                          '--r')
            axs[0,j].set_ylim(ylims[j])
            axs[1,j].set_ylim(ylims[j+nprops])
        plt.savefig(os.path.join(self.output_path,
                                 f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-MF_gtvspred.png"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path,
                                 f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-MF_gtvspred.eps"),
                    bbox_inches='tight', format='eps')

        if self.NoisyMF_GROUNDTRUTH is not None:
            # Exhaustive NoisyMF (groundtruth)
            tar = self.scaled_target_NNLS[:, :]
            pred = self.NoisyMF_GROUNDTRUTH_output

            nrows = 2
            ncols = nprops
            fig, axs = plt.subplots(nrows, ncols)
            fig.suptitle('Exhaustive NoisyMF + Groundtruth')
            for j, p in enumerate(prop_names):
                if (j == 0):
                    axs[0, j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1, j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0, j].hexbin(tar[:, j],
                                 pred[:, j],
                                 gridsize=16, cmap='Greys')
                axs[0, j].plot([np.min(tar[:, j]), np.max(tar[:, j])],
                               [np.min(tar[:, j]), np.max(tar[:, j])], '--k')
                axs[0, j].set_title(prop_names[j])
                axs[1, j].set_xlabel('Groundtruth', loc='left')
                axs[1, j].hexbin(tar[:, j + nprops], pred[:, j + nprops],
                                 gridsize=16, cmap='Greys')
                axs[1, j].plot([np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])],
                               [np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])], '--k')

                # Plot linear regressions of pred = f(target)
                reg = LinearRegression().fit(tar[:, j:j + 1], pred[:, j:j + 1])
                axs[0, j].plot([np.min(tar[:, j]), np.max(tar[:, j])],
                               [np.min(tar[:, j]) * reg.coef_[0, 0] + reg.intercept_[0],
                                np.max(tar[:, j]) * reg.coef_[0, 0] + reg.intercept_[0]],
                               '--r')

                reg = LinearRegression().fit(tar[:, j + nprops:j + nprops + 1], pred[:, j + nprops:j + nprops + 1])
                axs[1, j].plot([np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])],
                               [np.min(tar[:, j + nprops]) * reg.coef_[0, 0] + reg.intercept_[0],
                                np.max(tar[:, j + nprops]) * reg.coef_[0, 0] + reg.intercept_[0]],
                               '--r')
                axs[0, j].set_ylim(ylims[j])
                axs[1, j].set_ylim(ylims[j + nprops])
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-NoisyMF_gtvspred.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-NoisyMF_gtvspred.eps"),
                        bbox_inches='tight', format='eps')


        del pred
        del tar
        return 1
    
    def __plot_Residuals(self):
        
        prop_names = ['$\\nu$', '$D_{ex}$', '$fvf$']
        nprops = len(prop_names)

        ylims = [(-0.5,0.5),(-1e-9,1e-9), (-0.5,0.5)]
        
        # Hybrid method
        if self.metadata_Hybrid is not None:
            tar = self.swapField(self.scaled_target_NNLS[:,:],self.NNLS_output_swapField)
            pred = self.scaled_pred_hybrid[:,:]

            nrows = 2
            ncols = 3
            fig,axs = plt.subplots(nrows,ncols)
            fig.suptitle('Hybrid MLP')
            #axs.set_title('Hybrid MLP')
            for j,p in enumerate(prop_names):
                if(j == 0):
                    axs[0,j].set_ylabel('Residual\n $1^{st}$ fascicle')
                    axs[1,j].set_ylabel('Residual\n $2^{nd}$ fascicle')
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])], [0,0], '--k')
                axs[0,j].hexbin(tar[:,j],
                                tar[:,j] - pred[:,j],
                                gridsize = 16, cmap = 'Greys')
                axs[0,j].set_title(prop_names[j])
                axs[1,j].set_xlabel('Groundtruth', loc = 'left')
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])], [0,0], '--k')
                axs[1,j].hexbin(tar[:,j+nprops],
                                tar[:,j+nprops] - pred[:,j+nprops],
                                gridsize = 16, cmap = 'Greys' )
                axs[0,j].set_ylim(ylims[j])
                axs[1,j].set_ylim(ylims[j])

            plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_method-Hybrid_residuals.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_method-Hybrid_residuals.eps"),
                        bbox_inches='tight', format='eps')

        # Fully learned method
        if self.metadata_Full is not None:
            prop_idx = [0,4,5,6,10,11]
            tar = self.scaled_target_SH[:,prop_idx]
            pred = self.scaled_pred_full[:,prop_idx]

            nrows = 2
            ncols = 3
            fig,axs = plt.subplots(nrows,ncols)
            fig.suptitle('FullyLearned MLP')
            #axs.set_title('Hybrid MLP')
            for j,p in enumerate(prop_names):
                if(j == 0):
                    axs[0,j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1,j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])], [0,0], '--k')
                axs[0,j].hexbin(tar[:,j],
                                tar[:,j] - pred[:,j],
                                gridsize = 16, cmap = 'Greys')
                axs[0,j].set_title(prop_names[j])
                axs[1,j].set_xlabel('Groundtruth', loc = 'left')
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])], [0,0], '--k')
                axs[1,j].hexbin(tar[:,j+nprops],
                                tar[:,j+nprops] - pred[:,j+nprops],
                                gridsize = 16, cmap = 'Greys' )
                axs[0,j].set_ylim(ylims[j])
                axs[1,j].set_ylim(ylims[j])

            plt.savefig(
                os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_method-SH_residuals.png"),
                bbox_inches='tight')
            plt.savefig(
                os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_method-SH_residuals.eps"),
                bbox_inches='tight', format='eps')

        # Exhaustive MF (MSMTCSD)
        if self.MF_MSMTCSD is not None:
            tar = self.swapField(self.scaled_target_NNLS[:,:],self.MF_MSMTCSD_output_swapField)
            pred = self.MF_MSMTCSD_output

            nrows = 2
            ncols = nprops
            fig,axs = plt.subplots(nrows,ncols)
            fig.suptitle('Exhaustive MF + MSMT CSD')
            #axs.set_title('Hybrid MLP')
            for j,p in enumerate(prop_names):
                if(j == 0):
                    axs[0,j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1,j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0,j].hexbin(tar[:,j],
                                tar[:,j] - pred[:,j],
                                gridsize = 16, cmap = 'Greys')
                axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])],
                              [0,0],   '--k')
                axs[0,j].set_title(prop_names[j])
                axs[1,j].set_xlabel('Groundtruth', loc = 'left')
                axs[1,j].hexbin(tar[:,j+nprops],tar[:,j+nprops] - pred[:,j+nprops],
                                gridsize = 16, cmap = 'Greys' )
                axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])],
                              [0,0],   '--k')

                axs[0,j].set_ylim(ylims[j])
                axs[1,j].set_ylim(ylims[j])

            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-MSMTCSD_method-MF_residuals.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-MSMTCSD_method-MF_residuals.eps"),
                        bbox_inches='tight', format='eps')

        if self.MF_CSD is not None:
            tar = self.swapField(self.scaled_target_NNLS[:, :], self.MF_CSD_output_swapField)
            pred = self.MF_CSD_output

            nrows = 2
            ncols = nprops
            fig, axs = plt.subplots(nrows, ncols)
            fig.suptitle('Exhaustive MF + CSD')
            # axs.set_title('Hybrid MLP')
            for j, p in enumerate(prop_names):
                if (j == 0):
                    axs[0, j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1, j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0, j].hexbin(tar[:, j],
                                 tar[:, j] - pred[:, j],
                                 gridsize=16, cmap='Greys')
                axs[0, j].plot([np.min(tar[:, j]), np.max(tar[:, j])],
                               [0, 0], '--k')
                axs[0, j].set_title(prop_names[j])
                axs[1, j].set_xlabel('Groundtruth', loc='left')
                axs[1, j].hexbin(tar[:, j + nprops], tar[:, j + nprops] - pred[:, j + nprops],
                                 gridsize=16, cmap='Greys')
                axs[1, j].plot([np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])],
                               [0, 0], '--k')

                axs[0, j].set_ylim(ylims[j])
                axs[1, j].set_ylim(ylims[j])

            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-CSD_method-MF_residuals.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-CSD_method-MF_residuals.eps"),
                        bbox_inches='tight', format='eps')

        # Exhaustive MF (GROUNDTRUTH)
        tar = self.scaled_target_NNLS[:,:]

        pred = self.MF_GROUNDTRUTH_output
        
        nrows = 2
        ncols = nprops
        fig,axs = plt.subplots(nrows,ncols)
        fig.suptitle('Exhaustive MF + Groundtruth')
        #axs.set_title('Hybrid MLP')
        for j,p in enumerate(prop_names):
            if(j == 0):
                axs[0,j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                axs[1,j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
            axs[0,j].hexbin(tar[:,j], 
                            tar[:,j] - pred[:,j], 
                            gridsize = 16, cmap = 'Greys')
            axs[0,j].plot([np.min(tar[:,j]),np.max(tar[:,j])], 
                          [0,0],   '--k')
            axs[0,j].set_title(prop_names[j])
            axs[1,j].set_xlabel('Groundtruth', loc = 'left')
            axs[1,j].hexbin(tar[:,j+nprops],tar[:,j+nprops] - pred[:,j+nprops], 
                            gridsize = 16, cmap = 'Greys' )
            axs[1,j].plot([np.min(tar[:,j+nprops]),np.max(tar[:,j+nprops])], 
                          [0,0],   '--k')
            axs[0,j].set_ylim(ylims[j])
            axs[1,j].set_ylim(ylims[j])

        plt.savefig(os.path.join(self.output_path,
                                 f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-MF_residuals.png"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path,
                                 f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-MF_residuals.eps"),
                    bbox_inches='tight', format='eps')

        # Exhaustive NoisyMF (GROUNDTRUTH)
        if self.NoisyMF_GROUNDTRUTH is not None:

            tar = self.scaled_target_NNLS[:, :]

            pred = self.NoisyMF_GROUNDTRUTH_output

            nrows = 2
            ncols = nprops
            fig, axs = plt.subplots(nrows, ncols)
            fig.suptitle('Exhaustive NoisyMF + Groundtruth')
            # axs.set_title('Hybrid MLP')
            for j, p in enumerate(prop_names):
                if (j == 0):
                    axs[0, j].set_ylabel('Prediction\n $1^{st}$ fascicle')
                    axs[1, j].set_ylabel('Prediction\n $2^{nd}$ fascicle')
                axs[0, j].hexbin(tar[:, j],
                                 tar[:, j] - pred[:, j],
                                 gridsize=16, cmap='Greys')
                axs[0, j].plot([np.min(tar[:, j]), np.max(tar[:, j])],
                               [0, 0], '--k')
                axs[0, j].set_title(prop_names[j])
                axs[1, j].set_xlabel('Groundtruth', loc='left')
                axs[1, j].hexbin(tar[:, j + nprops], tar[:, j + nprops] - pred[:, j + nprops],
                                 gridsize=16, cmap='Greys')
                axs[1, j].plot([np.min(tar[:, j + nprops]), np.max(tar[:, j + nprops])],
                               [0, 0], '--k')
                axs[0, j].set_ylim(ylims[j])
                axs[1, j].set_ylim(ylims[j])

            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-NoisyMF_residuals.png"),
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.output_path,
                                     f"task-{self.task_name}_ses-{self.session_id}_orientation-GROUNDTRUTH_method-NoisyMF_residuals.eps"),
                        bbox_inches='tight', format='eps')
        del pred
        del tar

        return 1
    
    def __assess_nuCSF(self):#TODO FIx
        
        
        tar = self.scaled_target_NNLS[:,-2:-1]

        
        pred_hybrid = self.scaled_pred_hybrid[:,-2:-1]

        
        pred_full = self.scaled_pred_full[:,-2:-1]

        try:
            pred_msmt = self.MF_GROUNDTRUTH['nus_est']
            preds = [pred_hybrid, pred_full, pred_msmt]
            ncols = 3
        except KeyError:
            ncols = 2
            preds = [pred_hybrid, pred_full]
            
            
        methods_names = ['Hybrid', 'FullyLearned', 'MSMT CSD']
        fig,axs = plt.subplots(2,ncols)
        fig.suptitle('Assessment of $\\nu_{csf}$ predictions')
        for j in range(ncols):
            if(j == 0):
                axs[0,j].set_ylabel('Prediction')
                axs[1,j].set_ylabel('Residual')
            axs[0,j].plot([np.min(tar[:,0]),np.max(tar[:,0])], [np.min(tar[:,0]),np.max(tar[:,0])], '--k')
            axs[0,j].hexbin(tar[:,0],
                            preds[j][:,0],  
                            gridsize = 16, cmap = 'Greys')
            axs[0,j].set_title(methods_names[j])
            axs[1,j].set_xlabel('Groundtruth')
            axs[1,j].plot([np.min(tar[:,0]),np.max(tar[:,0])], [0,0], '--k')
            axs[1,j].hexbin(tar[:,0],
                            tar[:,0] -  preds[j][:,0], 
                            gridsize = 16, cmap = 'Greys' )

    def __plot_MAEbyAngularError(self):
        orientation = self.orientation_estimate

        if orientation == 'MSMTCSD':
            angles_rad_MF_est = np.arccos(self.abs_dots_MF_MSMTCSD) * 180/np.pi
            idx_sort_angles = np.zeros(self.abs_dots_MF_MSMTCSD.shape)
            labels = ['MF + MSMTCSD', 'MF + GT', 'Hybrid MLP']
        elif orientation == 'CSD':
            angles_rad_MF_est = np.arccos(self.abs_dots_MF_CSD) * 180/np.pi
            idx_sort_angles = np.zeros(self.abs_dots_MF_CSD.shape)
            labels = ['MF + CSD', 'MF + GT', 'Hybrid MLP']
        else:
            assert self.orientation_estimate in ["CSD", "MSMTCSD"], "orientation_estimate must be either CSD or MSMTCSD for this method"

        idx_sort_angles[:, 0] = np.argsort(angles_rad_MF_est[:, 0])
        idx_sort_angles[:, 1] = np.argsort(angles_rad_MF_est[:, 1])

        prop_names = ['$E[|\\nu - \hat{\\nu}|]$', 
                      '$E[|D_{ex} - \\hat{D_{ex}}|]$', 
                      '$E[|fvf - \\hat{fvf}|]$']
        colors = ['grey', 'black', 'green', 'blue']
         #,

        
        num_fasc = 2
        num_prop = len(prop_names)
        prop_idx_full = [0,4,5,6,10,11]
        
        fig31, ax31 = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))

        idx_prop_SH = [3,4,5,9,10,11]

        nbins = 20
        bins = np.zeros((nbins+1,num_fasc))
        bins[:,0] = np.linspace(np.min(angles_rad_MF_est[:,0]), np.max(angles_rad_MF_est[:,0]), nbins+1)
        bins[:,1] = np.linspace(np.min(angles_rad_MF_est[:,1]), np.max(angles_rad_MF_est[:,1]), nbins+1)
        
        bins_middles = (bins[1:,:] + bins[0:-1,:])/2
        if self.metadata_Hybrid is not None:
            scaled_target_NNLS_swapped = self.swapField(self.scaled_target_NNLS, self.NNLS_output_swapField)
        if self.MF_MSMTCSD is not None:
            scaled_target_MSMTCSD = self.swapField(self.scaled_target_NNLS, self.MF_MSMTCSD_output_swapField)
        if self.MF_CSD is not None:
            scaled_target_CSD = self.swapField(self.scaled_target_NNLS, self.MF_CSD_output_swapField)

        for i in range(num_fasc):
            
            idx_start = i*num_prop
            idx_end = (i+1)*num_prop

            AE_hybrid = np.abs(self.scaled_pred_hybrid[:,idx_start:idx_end] - scaled_target_NNLS_swapped[:,idx_start:idx_end])
            if orientation == 'MSMTCSD':
                AE_MF_est = np.abs(self.MF_MSMTCSD_output[:,idx_start:idx_end] - scaled_target_MSMTCSD[:,idx_start:idx_end])
            elif orientation == 'CSD':
                AE_MF_est = np.abs(self.MF_CSD_output[:, idx_start:idx_end] - scaled_target_CSD[:, idx_start:idx_end])

            for j,p in enumerate(prop_names):

                values_to_plot_MF_est = np.zeros(nbins)
                values_to_plot_NNLS = np.zeros(nbins)

                
                for l in range(nbins):
                    idx = np.logical_and(angles_rad_MF_est[:,i]>bins[l,i], angles_rad_MF_est[:,i]<=bins[l+1,i])
                    values_to_plot_MF_est[l] = np.mean(AE_MF_est[idx,j])
                    values_to_plot_NNLS[l] = np.mean(AE_hybrid[idx,j])


                ax31[j, i].plot(bins_middles[:,i],values_to_plot_MF_est,
                                color=colors[0], marker='x')
                
                ax31[j, i].plot(bins_middles[:,i],values_to_plot_NNLS,
                                color=colors[1], marker='x')
                
                #Hybrid
                # ax31[j, i].plot(nus_hybrid[i], mae_MF[:,j],
                #                 color=colors[1], marker='x')
                
                # ax31[j, i].plot(nus_hybrid[i], mae_hybrid[:,j],
                #                 color=colors[2], marker='x')
                # ax31[j, i].plot(nus_full[i], mae_full[:,j],
                #                 color=colors[3], marker='x')
                # ax31[j, i].invert_xaxis()

                if i == 0:
                    ax31[j, i].set_ylabel('%s' % (prop_names[j]))
                if j == 0:
                    ax31[j, i].set_title('fascicle %s' % (i + 1))
                if j == 2:
                    ax31[j, i].set_xlabel('Angular Error' .format(i + 1))

                ax31[j, i].yaxis.grid(True)

                if j == 1:
                    ax31[j, i].set_ylim(0, 9.0 * 10 ** -10)
                elif j == 2:
                    ax31[j, i].set_ylim(0, 0.4)
                else:
                    ax31[j, i].set_ylim(0, 0.35)
                

        fig31.legend(labels)
        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_maebyangularerror.png"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_maebyangularerror.eps"),
                    bbox_inches='tight', format='eps')
        
        return 1
        
    def __plot_AngularError(self, hide_MSMTCSD=False, hide_CSD=False):
        
        #Swapping orientations
        
        MF_MSMTCSD_est_ori =self.MF_MSMTCSD["est_orientations"]
        swapField = self.MF_MSMTCSD_output_swapField.astype(bool)
        MF_MSMTCSD_est_ori_swap = MF_MSMTCSD_est_ori.copy()
        (MF_MSMTCSD_est_ori_swap[swapField, 0], MF_MSMTCSD_est_ori_swap[swapField, 1]) = \
            (MF_MSMTCSD_est_ori[swapField, 1], MF_MSMTCSD_est_ori[swapField, 0])
        abs_dots_MF_MSMTCSD = np.abs(np.sum(MF_MSMTCSD_est_ori_swap*self.MF_GROUNDTRUTH['gt_orientations'], axis = 2))
        self.abs_dots_MF_MSMTCSD = abs_dots_MF_MSMTCSD
        angles_rad_MF_MSMTCSD = np.arccos(abs_dots_MF_MSMTCSD)

        MF_CSD_est_ori = self.MF_CSD["est_orientations"]
        swapField = self.MF_CSD_output_swapField.astype(bool)
        MF_CSD_est_ori_swap = MF_CSD_est_ori.copy()
        (MF_CSD_est_ori_swap[swapField, 0], MF_CSD_est_ori_swap[swapField, 1]) = \
            (MF_CSD_est_ori[swapField, 1], MF_CSD_est_ori[swapField, 0])
        abs_dots_MF_CSD = np.abs(np.sum(MF_CSD_est_ori_swap * self.MF_GROUNDTRUTH['gt_orientations'], axis=2))
        self.abs_dots_MF_CSD = abs_dots_MF_CSD
        angles_rad_MF_CSD = np.arccos(abs_dots_MF_CSD)

        if self.metadata_Full is not None:
            idx = [1,2,3,7,8,9]
            abs_dots_SH = np.zeros(abs_dots_MF_CSD.shape)
            pred_dir = self.scaling_fn_full(self.pred_full[:,idx], self.minis_full[idx],
                                self.maxis_full[idx],True)
            norms1 = np.linalg.norm(pred_dir[:,0:3], axis = 1)
            norms2 = np.linalg.norm(pred_dir[:,3:6], axis = 1)

            gt_dir = self.scaling_fn_full(self.target_SH[:,idx],  self.minis_full[idx],
                                self.maxis_full[idx],True)
            mul = pred_dir * gt_dir
            abs_dots_SH[:,0] = np.abs(np.sum(mul[:,0:3]/norms1[:,np.newaxis], axis = 1))
            abs_dots_SH[:,1] = np.abs(np.sum(mul[:,3:6]/norms2[:,np.newaxis], axis = 1))
            angles_rad_SH = np.arccos(abs_dots_SH)
            self.abs_dots_SH = abs_dots_SH
        else:
            abs_dots_SH = None
        
        bins = [k for k in range(60)]
        
        n = abs_dots_MF_CSD.shape[0]*2
        
        angles_rad_MF_CSD = angles_rad_MF_CSD.reshape(n)
        angles_rad_MF_MSMTCSD = angles_rad_MF_MSMTCSD.reshape(n)

        if self.metadata_Full is not None:
            angles_rad_SH = angles_rad_SH.reshape(n)
        
        hist_MF_CSD = np.histogram(angles_rad_MF_CSD*180/np.pi, bins= bins )
        bin_middles_MF_CSD = (hist_MF_CSD[1][1:] + hist_MF_CSD[1][0:-1])/2
        bin_size_MF_CSD = hist_MF_CSD[1][1] - hist_MF_CSD[1][0]

        hist_MF_MSMTCSD = np.histogram(angles_rad_MF_MSMTCSD * 180 / np.pi, bins=bins)
        bin_middles_MF_MSMTCSD = (hist_MF_MSMTCSD[1][1:] + hist_MF_MSMTCSD[1][0:-1]) / 2
        bin_size_MF_MSMTCSD = hist_MF_MSMTCSD[1][1] - hist_MF_MSMTCSD[1][0]

        if self.metadata_Full is not None:
            hist_SH = np.histogram(angles_rad_SH*180/np.pi, bins= bins )
            bin_middles_SH = (hist_SH[1][1:] + hist_SH[1][0:-1])/2
            bin_size_SH = hist_SH[1][1] - hist_SH[1][0]
        
        plt.figure()
        if not hide_MSMTCSD:
            plt.bar(bin_middles_MF_MSMTCSD, hist_MF_MSMTCSD[0], width = 4, color = 'r', alpha = 0.6, label = 'MSMT CSD')
        if not hide_CSD:
            plt.bar(bin_middles_MF_CSD, hist_MF_CSD[0], width=4, color='b', alpha=0.6, label='CSD')
        if self.metadata_Full is not None:
            plt.bar(bin_middles_SH, hist_SH[0], width = 4, color = 'k', alpha = 0.6, label = 'MLP FullyLearned')
        plt.xlabel('Angular Error')
        plt.ylabel('Number of samples')
        plt.legend()

        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_angularError.png"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_angularError.eps"),
                    bbox_inches='tight', format='eps')
        
        return abs_dots_MF_MSMTCSD, abs_dots_MF_CSD, abs_dots_SH
    
    def __plot_D2_by_nu(self, hide_MSMTCSD=False, hide_CSD=False):
        nus_hybrid = (np.unique(self.target_NNLS[:,0]),np.unique(self.target_NNLS[:,3]))    
        
        nus_full = (np.unique(self.target_SH[:,0]),np.unique(self.target_SH[:,6])) 
        
        
        fig31, ax31 = plt.subplots(nrows=2, ncols=2, figsize=(9, 12))
        # fig31.suptitle('Comparing the error of the 2 fascicles (NoSwap)')

        prop_names = ['$\\nu$', '$D_{ex}$', '$fvf$']
        colors_template = ['grey', 'black', 'green', 'blue']
        labels_template = ['MF + GROUNDTRUTH', 'MF + MSMT', 'Hybrid MLP', 'SH MLP']  # 'MF + GT',

        colors = ['grey']
        labels = ['MF + GROUNDTRUTH']
        id_dic = {"MF+GROUNDTRUTH": 0}
        i = 0
        if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
            labels.append("MF + MSMT")
            colors.append("black")
            i = i + 1
            id_dic["MF+MSMTCSD"] = i
        if self.MF_CSD is not None and not hide_CSD:
            labels.append("MF + CSD")
            colors.append("orange")
            i = i + 1
            id_dic["MF+CSD"] = i
        if self.metadata_Hybrid is not None:
            labels.append("Hybrid MLP")
            colors.append("green")
            i = i + 1
            id_dic["HybridMLP"] = i
        if self.metadata_Full is not None:
            labels.append("SH MLP")
            colors.append("blue")
            i = i + 1
            id_dic["SHMLP"] = i
        if self.NoisyMF_GROUNDTRUTH is not None:
            labels.append("NoisyMF + GROUNDTRUTH")
            colors.append("yellow")
            i = i + 1
            id_dic["NoisyMF+GROUNDTRUTH"] = i

        num_fasc = 2
        num_prop = len(prop_names)
        prop_idx_full = [0,4,5,6,10,11]

        if self.metadata_Hybrid is not None:
            scaled_target_NNLS_swapped = self.swapField(self.scaled_target_NNLS, self.NNLS_output_swapField)
        if self.MF_MSMTCSD is not None:
            scaled_target_MSMTCSD = self.swapField(self.scaled_target_NNLS, self.MF_MSMTCSD_output_swapField)
        if self.MF_CSD is not None:
            scaled_target_CSD = self.swapField(self.scaled_target_NNLS, self.MF_CSD_output_swapField)

        
        for i in range(0,num_fasc): 
            idx_hybrid = np.array([nus_hybrid[i][k] ==  self.target_NNLS[:,i*num_prop] for k in range(len(nus_hybrid[i]))])
            idx_full = np.array([nus_hybrid[i][k] ==  self.target_NNLS[:,i*num_prop] for k in range(len(nus_hybrid[i]))])
            
            idx_start = i*num_prop
            idx_end = (i+1)*num_prop
            if self.metadata_Hybrid is not None:
                D2_hybrid = self._D2ByNu(self.pred_hybrid[:,idx_start:idx_end],
                                         scaled_target_NNLS_swapped[:,idx_start:idx_end],
                                         idx_hybrid)

            if self.metadata_Full is not None:
                D2_full = self._D2ByNu(self.pred_full[:,prop_idx_full[idx_start:idx_end]],
                                         self.target_SH[:,prop_idx_full[idx_start:idx_end]],
                                         idx_full)

            if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
                D2_MF_MSMTCSD = self._D2ByNu(self.scaling_fn_hybrid(self.MF_MSMTCSD_output[:,idx_start:idx_end],
                                                            self.minis_hybrid[idx_start:idx_end],
                                                            self.maxis_hybrid[idx_start:idx_end],
                                                            inverse = False
                                                            ), #MF_MSMTCSD_output has not been subjected to minmax scaling
                                     scaled_target_MSMTCSD[:,idx_start:idx_end],
                                     idx_hybrid)

            if self.MF_CSD is not None and not hide_CSD:
                D2_MF_CSD = self._D2ByNu(self.scaling_fn_hybrid(self.MF_CSD_output[:,idx_start:idx_end],
                                                            self.minis_hybrid[idx_start:idx_end],
                                                            self.maxis_hybrid[idx_start:idx_end],
                                                            inverse = False
                                                            ), #MF_CSD_output has not been subjected to minmax scaling
                                     scaled_target_CSD[:,idx_start:idx_end],
                                     idx_hybrid)

            D2_MF_gt = self._D2ByNu(self.scaling_fn_hybrid(self.MF_GROUNDTRUTH_output[:,idx_start:idx_end], 
                                    self.minis_hybrid[idx_start:idx_end],
                                    self.maxis_hybrid[idx_start:idx_end],
                                    inverse = False
                                    ),
                                    self.target_NNLS[:,idx_start:idx_end],
                                    idx_hybrid)

            if self.NoisyMF_GROUNDTRUTH is not None:
                D2_NoisyMF_gt = self._D2ByNu(self.scaling_fn_hybrid(self.NoisyMF_GROUNDTRUTH_output[:, idx_start:idx_end],
                                                               self.minis_hybrid[idx_start:idx_end],
                                                               self.maxis_hybrid[idx_start:idx_end],
                                                               inverse=False
                                                               ),
                                        self.target_NNLS[:, idx_start:idx_end],
                                        idx_hybrid)
            
            for j,p in enumerate(prop_names[1:]):#The D2score is useless for nu for a structured set because the median is a perfect predictor when there is only one value in the target
                
                ax31[j, i].plot(nus_hybrid[i], D2_MF_gt[:,j],
                                color=colors[id_dic["MF+GROUNDTRUTH"]], marker='x')
                
                #Hybrid
                if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
                    ax31[j, i].plot(nus_hybrid[i], D2_MF_MSMTCSD[:,j],
                                    color=colors[id_dic["MF+MSMTCSD"]], marker='x')

                if self.MF_CSD is not None and not hide_CSD:
                    ax31[j, i].plot(nus_hybrid[i], D2_MF_CSD[:,j],
                                    color=colors[id_dic["MF+CSD"]], marker='x')

                if self.metadata_Hybrid is not None:
                    ax31[j, i].plot(nus_hybrid[i], D2_hybrid[:,j],
                                    color=colors[id_dic["HybridMLP"]], marker='x')

                if self.metadata_Full is not None:
                    ax31[j, i].plot(nus_full[i], D2_full[:,j],
                                    color=colors[id_dic["SHMLP"]], marker='x')

                if self.NoisyMF_GROUNDTRUTH is not None:
                    ax31[j, i].plot(nus_hybrid[i], D2_NoisyMF_gt[:,j],
                                    color=colors[id_dic["NoisyMF+GROUNDTRUTH"]], marker='x')
                ax31[j, i].invert_xaxis()

                if i == 0:
                    ax31[j, i].set_ylabel('D2 score ({0})'.format(prop_names[j+1]))
                if j == 0:
                    ax31[j, i].set_title('fasc %s' % (i + 1))
                if j == 2:
                    ax31[j, i].set_xlabel('nu %s' % (i + 1))

                ax31[j, i].yaxis.grid(True)
            ax31[1,0].set_xlabel('$\\nu_1$')
            ax31[1,1].set_xlabel('$\\nu_2$')

                # if j == 1:
                #     ax31[j, i].set_ylim(0, 9.0 * 10 ** -10)
                # elif j == 2:
                #     ax31[j, i].set_ylim(0, 0.4)
                # else:
                #     ax31[j, i].set_ylim(0, 0.35)
                
        

        fig31.legend(labels)

        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_d2bynu.png"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_d2bynu.eps"),
                    bbox_inches='tight', format='eps')
    
    def __plot_MAE_By_Nu(self, hide_MSMTCSD=False, hide_CSD=False):
        
        
        nus_hybrid = (np.unique(self.target_NNLS[:,0]),np.unique(self.target_NNLS[:,3]))    
        
        nus_full = (np.unique(self.target_SH[:,0]),np.unique(self.target_SH[:,6])) 
        
        
        fig31, ax31 = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
        # fig31.suptitle('Comparing the error of the 2 fascicles (NoSwap)')

        prop_names = ['$E[|\\nu - \hat{\\nu}|]$', 
                      '$E[|D_{ex} - \\hat{D_{ex}}|]$', 
                      '$E[|fvf - \\hat{fvf}|]$']
        colors_template = ['grey', 'black', 'green', 'blue']
        labels_template = ['MF + GROUNDTRUTH', 'MF + MSMT', 'Hybrid MLP', 'SH MLP']  # 'MF + GT',

        colors = ['grey']
        labels = ['MF + GROUNDTRUTH']
        id_dic = {"MF+GROUNDTRUTH":0}
        i=0
        if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
            labels.append("MF + MSMT")
            colors.append("black")
            i = i + 1
            id_dic["MF+MSMTCSD"] = i
        if self.MF_CSD is not None and not hide_CSD:
            labels.append("MF + CSD")
            colors.append("orange")
            i = i + 1
            id_dic["MF+CSD"] = i
        if self.metadata_Hybrid is not None:
            labels.append("Hybrid MLP")
            colors.append("green")
            i = i + 1
            id_dic["HybridMLP"] = i
        if self.metadata_Full is not None:
            labels.append("SH MLP")
            colors.append("blue")
            i = i + 1
            id_dic["SHMLP"] = i
        if self.NoisyMF_GROUNDTRUTH is not None:
            labels.append("NoisyMF + GROUNDTRUTH")
            colors.append("yellow")
            i = i + 1
            id_dic["NoisyMF+GROUNDTRUTH"] = i
        
        num_fasc = 2
        num_prop = len(prop_names)
        prop_idx_full = [0,4,5,6,10,11]

        if self.metadata_Hybrid is not None:
            scaled_target_NNLS_swapped = self.swapField(self.scaled_target_NNLS, self.NNLS_output_swapField)
        if self.MF_MSMTCSD is not None:
            scaled_target_MSMTCSD = self.swapField(self.scaled_target_NNLS, self.MF_MSMTCSD_output_swapField)
        if self.MF_CSD is not None:
            scaled_target_CSD = self.swapField(self.scaled_target_NNLS, self.MF_CSD_output_swapField)
        if self.metadata_Full is not None:
            scaled_target_SH_swapped = self.swapField(self.scaled_target_SH, self.SH_output_swapField, props_len=6)

        for i in range(num_fasc):
            idx_hybrid = np.array([nus_hybrid[i][k] == self.target_NNLS[:,i*num_prop] for k in range(len(nus_hybrid[i]))])
            idx_full = np.array([nus_hybrid[i][k] == self.target_NNLS[:,i*num_prop] for k in range(len(nus_hybrid[i]))])
            
            idx_start = i*num_prop
            idx_end = (i+1)*num_prop

            if self.metadata_Hybrid is not None:
                mae_hybrid = self._meanAbsErrByNu(self.scaled_pred_hybrid[:,idx_start:idx_end],
                                                  scaled_target_NNLS_swapped[:,idx_start:idx_end],
                                                  idx_hybrid)
            if self.metadata_Full is not None:
                mae_full = self._meanAbsErrByNu(self.scaled_pred_full[:,prop_idx_full[idx_start:idx_end]],
                                                scaled_target_SH_swapped[:,prop_idx_full[idx_start:idx_end]],
                                                idx_full)

            if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
                mae_MF_MSMTCSD = self._meanAbsErrByNu(self.MF_MSMTCSD_output[:,idx_start:idx_end], #MF_MSMTCSD_output has not been subjected to minmax scaling
                                              scaled_target_MSMTCSD[:,idx_start:idx_end],
                                              idx_hybrid)

            if self.MF_CSD is not None and not hide_CSD:
                mae_MF_CSD = self._meanAbsErrByNu(self.MF_CSD_output[:,idx_start:idx_end], #MF_MSMTCSD_output has not been subjected to minmax scaling
                                              scaled_target_CSD[:,idx_start:idx_end],
                                              idx_hybrid)
            
            mae_MF_gt = self._meanAbsErrByNu(self.MF_GROUNDTRUTH_output[:,idx_start:idx_end], 
                                             
                                             self.scaled_target_NNLS[:,idx_start:idx_end], 

                                             idx_hybrid)

            if self.NoisyMF_GROUNDTRUTH is not None:
                mae_NoisyMF_gt = self._meanAbsErrByNu(self.NoisyMF_GROUNDTRUTH_output[:, idx_start:idx_end],

                                                 self.scaled_target_NNLS[:,idx_start:idx_end],

                                                 idx_hybrid)
            
            for j,p in enumerate(prop_names):
                #MF !!

                ax31[j, i].plot(nus_hybrid[i], mae_MF_gt[:, j],
                                color=colors[id_dic["MF+GROUNDTRUTH"]], marker='x')

                # Hybrid
                if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
                    ax31[j, i].plot(nus_hybrid[i], mae_MF_MSMTCSD[:, j],
                                    color=colors[id_dic["MF+MSMTCSD"]], marker='x')

                if self.MF_CSD is not None and not hide_CSD:
                    ax31[j, i].plot(nus_hybrid[i], mae_MF_CSD[:, j],
                                    color=colors[id_dic["MF+CSD"]], marker='x')

                if self.metadata_Hybrid is not None:
                    ax31[j, i].plot(nus_hybrid[i], mae_hybrid[:, j],
                                    color=colors[id_dic["HybridMLP"]], marker='x')
                if self.metadata_Full is not None:
                    ax31[j, i].plot(nus_full[i], mae_full[:, j],
                                    color=colors[id_dic["SHMLP"]], marker='x')

                if self.NoisyMF_GROUNDTRUTH is not None:
                    ax31[j, i].plot(nus_hybrid[i], mae_NoisyMF_gt[:, j],
                                    color=colors[id_dic["NoisyMF+GROUNDTRUTH"]], marker='x')

                if i == 0:
                    ax31[j, i].set_ylabel('%s' % (prop_names[j]))
                if j == 0:
                    ax31[j, i].set_title('fascicle %s' % (i + 1))
                if j == 2:
                    ax31[j, i].set_xlabel('$\\nu_{0}$' .format(i + 1))

                ax31[j, i].yaxis.grid(True)

                if j == 1:
                    ax31[j, i].set_ylim(0, 9.0 * 10 ** -10)
                elif j == 2:
                    ax31[j, i].set_ylim(0, 0.4)
                else:
                    ax31[j, i].set_ylim(0, 0.35)
                
        

        fig31.legend(labels)

        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_maebynu.png"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_maebynu.eps"),
                    bbox_inches='tight', format='eps')
        return 1
        
    def __plot_MAE_By_Nucsf(self, hide_MSMTCSD=False, hide_CSD=False):
        
        
        nus = (np.unique(self.synth['nuscsf']), np.unique(self.synth['nuscsf']))   
        
        fig31, ax31 = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
        # fig31.suptitle('Comparing the error of the 2 fascicles (NoSwap)')

        prop_names = ['$\\nu$', '$D_{ex}$', '$fvf$']
        colors_template = ['grey', 'black', 'green', 'blue']
        labels_template = ['MF + GROUNDTRUTH', 'MF + MSMT',  'Hybrid MLP', 'SH MLP'] #'MF + GT',

        colors = ['grey']
        labels = ['MF + GROUNDTRUTH']
        id_dic = {"MF+GROUNDTRUTH":0}
        i=0
        if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
            labels.append("MF + MSMT")
            colors.append("black")
            i=i+1
            id_dic["MF+MSMTCSD"] = i
        if self.MF_CSD is not None and not hide_CSD:
            labels.append("MF + CSD")
            colors.append("orange")
            i=i+1
            id_dic["MF+CSD"] = i
        if self.metadata_Hybrid is not None:
            labels.append("Hybrid MLP")
            colors.append("green")
            i=i+1
            id_dic["HybridMLP"] = i
        if self.metadata_Full is not None:
            labels.append("SH MLP")
            colors.append("blue")
            i = i + 1
            id_dic["SHMLP"] = i
        if self.NoisyMF_GROUNDTRUTH is not None:
            labels.append("NoisyMF + GROUNDTRUTH")
            colors.append("yellow")
            i = i + 1
            id_dic["NoisyMF+GROUNDTRUTH"] = i

        
        num_fasc = 2
        num_prop = len(prop_names)
        prop_idx_full = [0,4,5,6,10,11]
        
        for i in range(num_fasc):
            idx = np.array([nus[i][k] ==  self.target_NNLS[:,i*num_prop] for k in range(len(nus[i]))])

            idx_start = i*num_prop
            idx_end = (i+1)*num_prop

            if self.metadata_Hybrid is not None:
                mae_hybrid = self._meanAbsErrByNu(self.scaling_fn_hybrid(self.pred_hybrid[:,idx_start:idx_end],
                                                                         self.minis_hybrid[idx_start:idx_end],
                                                                         self.maxis_hybrid[idx_start:idx_end],
                                                                         True),

                                                  self.scaling_fn_hybrid(self.swapField(self.target_NNLS[:,idx_start:idx_end],self.NNLS_output_swapField),
                                                                         self.minis_hybrid[idx_start:idx_end],
                                                                         self.maxis_hybrid[idx_start:idx_end],
                                                                         True),

                                                  idx)
            if self.metadata_Full is not None:
                mae_full = self._meanAbsErrByNu(self.scaling_fn_full(self.pred_full[:,prop_idx_full[idx_start:idx_end]],
                                                                    self.minis_full[prop_idx_full[idx_start:idx_end]],
                                                                    self.maxis_full[prop_idx_full[idx_start:idx_end]],
                                                                     True),

                                                self.scaling_fn_full(self.target_SH[:,prop_idx_full[idx_start:idx_end]],
                                                                     self.minis_full[prop_idx_full[idx_start:idx_end]],
                                                                     self.maxis_full[prop_idx_full[idx_start:idx_end]],
                                                                     True),

                                                idx)

            if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
                mae_MF_MSMTCSD = self._meanAbsErrByNu(self.MF_MSMTCSD_output[:,idx_start:idx_end], #MF_MSMTCSD_output has not been subjected to minmax scaling

                                              self.scaling_fn_hybrid(self.swapField(self.target_NNLS[:,idx_start:idx_end],self.MF_MSMTCSD_output_swapField),
                                                                    self.minis_hybrid[idx_start:idx_end].squeeze(),
                                                                    self.maxis_hybrid[idx_start:idx_end].squeeze(),
                                                                    True),
                                              idx)

            if self.MF_CSD is not None and not hide_CSD:
                mae_MF_CSD = self._meanAbsErrByNu(self.MF_CSD_output[:,idx_start:idx_end], #MF_MSMTCSD_output has not been subjected to minmax scaling

                                              self.scaling_fn_hybrid(self.swapField(self.target_NNLS[:,idx_start:idx_end],self.MF_CSD_output_swapField),
                                                                    self.minis_hybrid[idx_start:idx_end].squeeze(),
                                                                    self.maxis_hybrid[idx_start:idx_end].squeeze(),
                                                                    True),
                                              idx)



            mae_MF_gt = self._meanAbsErrByNu(self.MF_GROUNDTRUTH_output[:,idx_start:idx_end], 
                                             
                                             self.scaling_fn_hybrid(self.target_NNLS[:,idx_start:idx_end], 
                                                                   self.minis_hybrid[idx_start:idx_end].squeeze(),
                                                                   self.maxis_hybrid[idx_start:idx_end].squeeze(),
                                                                   True),
                                             idx)

            if self.NoisyMF_GROUNDTRUTH is not None:
                mae_NoisyMF_gt = self._meanAbsErrByNu(self.NoisyMF_GROUNDTRUTH_output[:, idx_start:idx_end],

                                                 self.scaling_fn_hybrid(self.target_NNLS[:, idx_start:idx_end],
                                                                        self.minis_hybrid[idx_start:idx_end].squeeze(),
                                                                        self.maxis_hybrid[idx_start:idx_end].squeeze(),
                                                                        True),
                                                 idx)
            
            for j,p in enumerate(prop_names):
                #MF !!

                
                ax31[j, i].plot(nus[i], mae_MF_gt[:,j],
                                color=colors[id_dic["MF+GROUNDTRUTH"]], marker='x')
                
                #Hybrid
                if self.MF_MSMTCSD is not None and not hide_MSMTCSD:
                    ax31[j, i].plot(nus[i], mae_MF_MSMTCSD[:,j],
                                    color=colors[id_dic["MF+MSMTCSD"]], marker='x')

                if self.MF_CSD is not None and not hide_CSD:
                    ax31[j, i].plot(nus[i], mae_MF_CSD[:,j],
                                    color=colors[id_dic["MF+CSD"]], marker='x')

                if self.metadata_Hybrid is not None:
                    ax31[j, i].plot(nus[i], mae_hybrid[:,j],
                                    color=colors[id_dic["HybridMLP"]], marker='x')
                if self.metadata_Full is not None:
                    ax31[j, i].plot(nus[i], mae_full[:,j],
                                    color=colors[id_dic["SHMLP"]], marker='x')

                if self.NoisyMF_GROUNDTRUTH is not None:
                    ax31[j, i].plot(nus[i], mae_NoisyMF_gt[:, j],
                                    color=colors[id_dic["NoisyMF+GROUNDTRUTH"]], marker='x')

                ax31[j, i].invert_xaxis()

                if i == 0:
                    ax31[j, i].set_ylabel('%s' % (prop_names[j]))
                if j == 0:
                    ax31[j, i].set_title('fasc %s' % (i + 1))
                if j == 2:
                    ax31[j, i].set_xlabel('nu %s' % (i + 1))

                ax31[j, i].yaxis.grid(True)

                if j == 1:
                    ax31[j, i].set_ylim(0, 9.0 * 10 ** -10)
                elif j == 2:
                    ax31[j, i].set_ylim(0, 0.4)
                else:
                    ax31[j, i].set_ylim(0, 0.35)
                
        

        fig31.legend(labels)
        
        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_maebynucsf.png"), bbox_inches = 'tight')
        plt.savefig(os.path.join(self.output_path, f"task-{self.task_name}_ses-{self.session_id}_orientation-{self.orientation_estimate}_maebynucsf.eps"),
                    bbox_inches='tight', format='eps')


    def __computeSwapFieldFullyLearned(self):

        if self.metadata_Full is None:
            return None

        tar1 = self.target_SH[:,0:4]
        tar2 = self.target_SH[:,6:10]