import time

from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import json
import os
import pickle
from scipy.stats import multivariate_normal

import fastmf.utils.mf_utils as mfu
import fastmf.utils.mf_estimator as mfe

from dipy.reconst.shm import real_sym_sh_basis, smooth_pinv
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.mcsd import response_from_mask_msmt
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from warnings import warn
import dipy.reconst.dti as dti
import dipy.direction.peaks as dp
from tqdm import tqdm

from fastmf.generation.synthetizer import SynthetizerFit


class Generator:
    def __init__(self, synthetizer, base_path, 
                 orientation_estimate_sh_max_order = 12, 
                 orientation_estimate = 'MSMTCSD', 
                 recompute_S0mean = False, compute_vf = False, compute_swap = False, random_seed = None):
        if isinstance(synthetizer, dict):
            self.synthetizer_dic = synthetizer
        elif isinstance(synthetizer, str):
            with open(synthetizer, 'rb') as handle:
                self.synthetizer_dic = pickle.load(handle)
        elif isinstance(synthetizer, SynthetizerFit):
            self.synthetizer_dic = synthetizer.data_dic
        else:
            raise ValueError("The synthetizer must be a dictionary, a path to a dictionary saved as numpy object or a SynthetizerFit object")
        
        assert orientation_estimate in ['CSD', 'MSMTCSD','GROUNDTRUTH' ], 'orientation_estimate must be CSD, MSMTCSD or GROUNDTRUTH but {0} was given'.format(orientation_estimate)
        self.orientation_estimate = orientation_estimate 
        
        self.task_name = self.synthetizer_dic['parameters']["task_name"]
        self.run_id = self.synthetizer_dic['parameters']["run_id"]
        self.base_path = base_path

        gam = mfu.get_gyromagnetic_ratio('H')
        G = self.synthetizer_dic['parameters']['scheme'][:, 3]
        Deltas = self.synthetizer_dic['parameters']['scheme'][:, 4]
        deltas = self.synthetizer_dic['parameters']['scheme'][:, 5]
        self.TE = np.mean(self.synthetizer_dic['parameters']['scheme'][:, 6])


        self.bvals = (gam * G * deltas) ** 2 * (Deltas - deltas / 3)  # in SI units s/m^2
        self.bvecs = self.synthetizer_dic['parameters']['scheme'][:, :3]

        # Preparation for an evantual usage of CSD
        self.gtab = gradient_table(self.bvals / 1e6, self.bvecs)
        self.odf_sphere = get_sphere('repulsion724')

        num_dwi = np.sum(self.bvals > 0)
        sh_max_vals = np.arange(2, orientation_estimate_sh_max_order + 1, 2)
        base_sizes = (sh_max_vals + 1) * (sh_max_vals + 2) // 2
        i_shmax = np.where(num_dwi >= base_sizes)[0][-1]
        sh_max_order = sh_max_vals[i_shmax]
        self.orientation_estimate_sh_max_order = sh_max_order

        self.ms_interpolator = mfu.init_PGSE_multishell_interp(
            self.synthetizer_dic['parameters']['MF_dict']['dictionary'],
            self.synthetizer_dic['parameters']['MF_dict']['sch_mat'],
            self.synthetizer_dic['parameters']['MF_dict']['orientation'])
        
        self.recompute_S0mean = recompute_S0mean #Whether to recompute S0mean for each voxel
                                                #for response function estimation in SSST CSD
        if(orientation_estimate in ['MSMTCSD', 'CSD']): #We need only to compute model and response function once ; the signal is then scaled down with S0mean (rather than scaling the response function up)
            self.__getCSDResponseFunctions(D = np.array([]), D_csf = np.array([]),D_gm = np.array([]))

        self.include_csf = self.synthetizer_dic['parameters']['include_csf']
        if(self.include_csf and (orientation_estimate == 'CSD')):
            warn('You are including csf in the voxels but using CSD ; you should use MSMT CSD.')
        self.compute_vf = compute_vf
        if(self.compute_vf and not(orientation_estimate == 'MSMTCSD')):
            raise ValueError('Volume fraction estimation (compute_vf == True) can only be done with MSMT CSD.')
        self.compute_swap = compute_swap

        self.random_seed = random_seed
    
    def __getCSDResponseFunctions(self, D = np.array([[]]), D_csf = np.array([[]]),D_gm = np.array([[]])):
        '''D contains the signals from which the white matter response function will be estimated
        D_csf contains the signals from which the csf response function will be estimated ; only used for MSMT CSD
        D_gm contains the signals from which the csf response function will be estimated ; only used for MSMT CSD
        If they all are arrays with at least one shape equals to zero, then the signals from the dictionary will be used
        S0_mean : only used for single tissue CSD ; the response function for single CSD is estimated by 
        fitting a tensor model to the WM signals, and is then computed from (avg(eig1), avg(eig2), avg(eig3), S0_mean)
        When one knows there is only WM in DWI_image_noisy, a good choice can be np.mean(DWI_image_noisy[bvals == 0])
        for MSMTCSD, the output is of type dipy.reconst.mcsd.MultiShellResponse and contains (among others) the Spherical Harmonics coefficients
        for single CSD, the output is a numpy array and contains eigenvalues and response for bval = 0
        '''
        MSMTCSD = self.orientation_estimate == 'MSMTCSD' #Variable for choosing the correct type of CSD
        if((np.any(D.shape)==0) and (np.any(D_csf.shape)==0)):
            D = self.synthetizer_dic['parameters']['MF_dict']['dictionary']
            D_csf = self.synthetizer_dic['parameters']['MF_dict']['sig_csf']
            if(len(D_csf.shape)==1):
                D_csf = D_csf[:,np.newaxis]


            #D_gm is not checked, since no gm in the dictionary is a common case 
            #The CSD functions all can work with D_gm = []
        
        if(MSMTCSD):            
            if(D_csf.shape[0] == 0):
                raise ValueError('D_csf contains no signals upon which to estimate the CSF response'+
                                 ' and MSMT CSD without CSF is not implemented.')
                
            if(D_gm.shape[0] == 0): #TODO : better checks and possibility to have only GM and WM ?
            #Currently supported : WM + CSF / WM + CSF + GM
                    D_full = np.concatenate((D,D_csf), axis = 1) 
                    n_gm = 0 #Number of gm signals
            else:
                D_full = np.concatenate((D,D_csf,D_gm), axis = 1) 
                n_gm = D_gm.shape[1] #number of gm signals
                
            
            
            n_axons = D.shape[1] #Number of WM signals
            n_csf = D_csf.shape[1] #Number of CSF signals
            
            
            #Change the size of D_full in the way expected by response_from_mask_msmt
            #dipy expects (Nx, Ny, N_slices, N_bvlasandbvecs)
            D_full_dipy = np.zeros([1, D_full.shape[1], 1, D_full.shape[0] ])
            D_full_dipy[0,:,0,:] = D_full.T 
            #D_full_dipy = D_full_dipy * np.mean(self.synthetizer_dic['M0s'])
            

            
            #Build the masks for each type of signal
            mask_wm = np.zeros([1,D_full.shape[1],1])
            mask_wm[0,0:n_axons,0] = 1
            mask_csf = np.zeros([1,D_full.shape[1],1])
            mask_csf[0,n_axons:n_axons+n_csf,0] = 1
            if(n_gm == 0):
                mask_gm = mask_csf #In this case, we will drop the gm response later on
            else:
                mask_gm = np.zeros([1,D_full.shape[1],1])
                mask_gm[0,n_axons+n_csf:,0] = 1
            
            #Estimate response functions from each signals type
            #I make this choice rather than use auto_response_msmt so the user has the choice of the method used to select the signals
            response_wm, response_gm, response_csf = response_from_mask_msmt(self.gtab, D_full_dipy,
                                                                             mask_wm,
                                                                             mask_gm, 
                                                                             mask_csf)
            if(n_gm == 0):#Typical case : no gm in dictionary ; response_gm was artificially computed as the same as csf, now we drop it
                response_gm = np.zeros(response_csf.shape)
            
            #Calculate spherical harmonics coefficients 
            #first one is CSF (isotropic), second is GM (isotropic), others are WM (anisotropic)
            ubvals = unique_bvals_tolerance(self.gtab.bvals) #Get lists of bvals values to build the SH coefficients
            response_function = multi_shell_fiber_response(sh_order=self.orientation_estimate_sh_max_order,
                                                bvals=ubvals,
                                                wm_rf=response_wm,
                                                gm_rf=response_gm,
                                                csf_rf=response_csf)
            #response_function.response = response_function.response
            self.msmt_response = response_function  
            self.msmt_model = MultiShellDeconvModel(self.gtab, 
                                                    response_function, 
                                                    reg_sphere = self.odf_sphere, 
                                                    iso = 2) #So we only need to create the model once
            if(not(self.recompute_S0mean)):
                M0_mean = np.mean(self.synthetizer_dic['M0s'])
                if D_full.shape[0] != self.bvals.shape[0]:
                    self.S0mean = np.mean(D_full[self.synthetizer_dic['parameters']['MF_dict']['bvals'] == 0, :]) * M0_mean
                else:
                    self.S0mean = np.mean(D_full[self.bvals == 0,:]) * M0_mean #2750 : hard coded average of M0 used to generate data
            
        else:
            if(not('avg_evals_SSST_CSD' in dir(self))): #If the tensor model has yet to be fit to the dictionary
                #Theoretically, I think the correct method is the one commented
                #However calculating an average over all bvalues seems to yield better results.
                # tol = 20
                # ubvals = unique_bvals_tolerance(self.gtab.bvals, tol) #List of possible values for bvals
                
                # bval_used = ubvals[1]*1e6 #Non zero bvalue used for fitting the tensor model
                # selected_bvals = np.abs(self.bvals - bval_used) < tol*1e6 / 1e6 
                # gtab_single_shell = gradient_table(self.bvals[selected_bvals], self.bvecs[selected_bvals,:])
                
                # tenmodel = dti.TensorModel(gtab_single_shell)
                # tenfit = tenmodel.fit(D[selected_bvals,:].T) #fit a tensor model to all wm voxels for a given non zero b-value
                # avg_evals = np.mean(tenfit.evals, axis = 0) #Average of eigenvalues of the tensor model over all wm voxels

                if self.bvals.shape[0] != D.shape[0]:
                    gtab_dic = gradient_table(self.synthetizer_dic['parameters']['MF_dict']['bvals'] / 1e6, self.synthetizer_dic['parameters']['MF_dict']['sch_mat'][:, :3])
                    tenmodel = dti.TensorModel(gtab_dic)
                else:
                    tenmodel = dti.TensorModel(self.gtab)
                tenfit = tenmodel.fit(D.T) #fit a tensor model to all wm voxels for a given non zero b-value
                avg_evals = np.mean(tenfit.evals, axis = 0) #Average of eigenvalues of the tensor model over all wm voxels
                self.avg_evals_SSST_CSD = avg_evals
                self.csd_response = [self.avg_evals_SSST_CSD,1] #The spherical harmonics coefficients will be computed when fitting the model
                self.csd_model = ConstrainedSphericalDeconvModel(self.gtab,
                                                      self.csd_response,
                                                      sh_order=self.orientation_estimate_sh_max_order)
            if(not(self.recompute_S0mean)):
                M0_mean = np.mean(self.synthetizer_dic['M0s'])
                if D.shape[0] != self.bvals.shape[0]:
                    self.S0mean = np.mean(D[self.synthetizer_dic['parameters']['MF_dict']['bvals'] == 0, :]) * M0_mean
                else:
                    self.S0mean = np.mean(D[self.bvals == 0,:]) * M0_mean #2750 : hard coded average of M0 used to generate data
        return 1 
    

    def __getCSDPeaks(self, y):
        """
        Get peak orientations using constrained spherical deconvolution (CSD), either single-shell single-tissue
        or multi-tissue depending on self.orientation_estimate
        compute_S0_mean is used for SSST : when it is true, S0_mean is re computed as np.mean(y[self.bvals == 0]), which is logical when there is only WM in the sources of y
                                           else, it is not recomputed and S0_mean = np.mean(D[self.bvals == 0, :])
        y is scaled down by np.mean(y[bvals==0]) #Is it a good / best choice ?
        """
        #Loading some variables in local variables for readability of code
        crossangle_min = self.synthetizer_dic["parameters"]["crossangle_min"]
        min_sep_angle = 0.75 * crossangle_min
        nu_min = self.synthetizer_dic["parameters"]["nu_min"]
        nu_max = self.synthetizer_dic["parameters"]["nu_max"]
        sh_max_order = self.orientation_estimate_sh_max_order #Maximal order of spherical harmonics used to compute CSD
        num_fasc = self.synthetizer_dic["parameters"]["num_fasc"] 
        nus_est = []
        #Type of csd :
        MSMTCSD = self.orientation_estimate == 'MSMTCSD'
        
        if(not(self.recompute_S0mean)):
            S0mean = self.S0mean #Average of M0s 
        else:
            S0mean = np.mean(y[self.bvals == 0]) #np.max ??
        
        if(MSMTCSD): #MSMT CSD

            
            peaks = dp.peaks_from_model(model=self.msmt_model, data=y/S0mean,
                                    relative_peak_threshold=0.2 * nu_min/nu_max,
                                    sh_order = sh_max_order,
                                    min_separation_angle=min_sep_angle,
                                    sphere=self.odf_sphere,
                                    npeaks=2)
            if(self.compute_vf):
                mcsd_fit = self.msmt_model.fit(y)
                nus_est = mcsd_fit.volume_fractions
                
            
        else: #Single shell single tissue CSD
            peaks = peaks_from_model(model=self.csd_model,
                                     data=y/S0mean,
                                     sphere=self.odf_sphere,
                                     relative_peak_threshold=0.2 * nu_min / nu_max,
                                     min_separation_angle=min_sep_angle,
                                     sh_order=sh_max_order,
                                     npeaks=num_fasc)

        return peaks.peak_dirs, peaks, nus_est


    def __estimateOrientations(self, sample, orientation_estimate=None):
        def __check_swap(est_dir1, est_dir2, fasc_dir_1, fasc_dir_2):
            corr11 = np.sum(np.multiply(est_dir1, fasc_dir_1))  # Scalar products
            corr21 = np.sum(np.multiply(est_dir2, fasc_dir_1))
            corr12 = np.sum(np.multiply(est_dir1, fasc_dir_2))
            corr22 = np.sum(np.multiply(est_dir2, fasc_dir_2))

            # We choose the most logical permutation of orest1 and orest2 by looking at the scalar products
            sum_1 = np.abs(corr11) + np.abs(corr22)
            sum_2 = np.abs(corr21) + np.abs(corr12)

            if (sum_1 > sum_2):  # The order is correct : just check sign
                swapped_samples = False
            else:  # The order needs to be changed
                swapped_samples = True

            return swapped_samples

        fasc_dir_1 = self.synthetizer_dic['orientations'][sample, 0, :]
        fasc_dir_2 = self.synthetizer_dic['orientations'][sample, 1, :]
        dir_est_failed = False #Used to track samples for which the estimation of orientations (CSD of MSMT CSD) failed to return 2 peaks
        est_dir1_swapped = np.array([])
        est_dir2_swapped = np.array([])
        
        nus_est = np.array([])

        if orientation_estimate is None:
            orientation_estimate = self.orientation_estimate
        
        #TODO : return the estimation of csf fraction when MSMT is used
        if(orientation_estimate in ["CSD", 'MSMTCSD']):
            #If one wants to modify the code in order to check the results of CSD, __getCSDPeaks also returns the dipy object and not just the directions
            peaks,peaks_data,nus_est = self.__getCSDPeaks(self.synthetizer_dic['DWI_noisy_store'][:, sample]) 
            num_pk_detected = np.sum(np.sum(np.abs(peaks), axis=1) > 0)
            if num_pk_detected < self.synthetizer_dic['parameters']['num_fasc']:
                dir_est_failed = True
                print("WARNING: CSD failed to detect {0} peaks for sample {1}".format(
                    self.synthetizer_dic['parameters']['num_fasc'] - num_pk_detected, sample))
                # There should always be at least 1 detected peak because the ODF
                # always has a max.
                # Pick second direction on a cone centered around first
                # direction with angle set to min separation angle above.
                # Using the same direction will lead to problems with NNLS down the
                # line with a poorly conditioned matrix (since the same submatrix
                # will be repeated)
                peaks[1] = peaks[0].copy()
                rot_ax = mfu.get_perp_vector(peaks[1])
                peaks[1] = mfu.rotate_vector(peaks[0], rot_ax, self.synthetizer_dic["parameters"]["crossangle_min"])
            est_dir1 = peaks[0]
            est_dir2 = peaks[1]
            
            if(self.compute_swap):
                corr11 = np.sum(np.multiply(est_dir1, fasc_dir_1)) #Scalar products 
                corr21 = np.sum(np.multiply(est_dir2, fasc_dir_1))
                corr12 = np.sum(np.multiply(est_dir1, fasc_dir_2))
                corr22 = np.sum(np.multiply(est_dir2, fasc_dir_2))
                
                #We choose the most logical permutation of orest1 and orest2 by looking at the scalar products
                sum_1 = np.abs(corr11) + np.abs(corr22)
                sum_2 = np.abs(corr21) + np.abs(corr12)
                
                if(sum_1>sum_2): #The order is correct : just check sign
                    if(corr11 >= 0): #Change orientation if needed : an atom rotated of 180° is identical to original
                        est_dir1_swapped = np.copy(est_dir1)
                    else:
                        est_dir1_swapped = np.copy(-est_dir1)   
                    if(corr22 >= 0):
                        est_dir2_swapped = np.copy(est_dir2)
                    else:
                        est_dir2_swapped = np.copy(-est_dir2)
                else: #The order needs to be changed
                    if(corr21 >= 0): #Change orientation if needed : an atom rotated of 180° is identical to original
                        est_dir1_swapped = np.copy(est_dir2)
                    else:
                        est_dir1_swapped = np.copy(-est_dir2)  
                    if(corr12 >= 0):
                        est_dir2_swapped = np.copy(est_dir1)
                    else:
                        est_dir2_swapped = np.copy(-est_dir1)
                est_dir1 = est_dir1_swapped
                est_dir2 = est_dir2_swapped
                
                
        elif orientation_estimate == "GROUNDTRUTH":
            est_dir1 = fasc_dir_1
            est_dir2 = fasc_dir_2
        else:
            raise NotImplementedError("orientation_estimate must be either CSD, GROUNDTRUTH or MSMTCSD")
            

        # Compute mean angular error
        swapped_samples = __check_swap(est_dir1, est_dir2, fasc_dir_1, fasc_dir_2)
        if swapped_samples:
            cos_12 = np.clip(np.abs(np.dot(est_dir1, fasc_dir_2)), 0, 1)
            cos_21 = np.clip(np.abs(np.dot(est_dir2, fasc_dir_1)), 0, 1)
            mean_ang_err = 0.5 * (np.arccos(cos_12) + np.arccos(cos_21)) * 180 / np.pi
        else:
            cos_11 = np.clip(np.abs(np.dot(est_dir1, fasc_dir_1)), 0, 1)
            cos_22 = np.clip(np.abs(np.dot(est_dir2, fasc_dir_2)), 0, 1)
            mean_ang_err = 0.5 * (np.arccos(cos_11) + np.arccos(cos_22)) * 180 / np.pi

        return est_dir1, est_dir2, mean_ang_err, nus_est, dir_est_failed, swapped_samples


    def __solveExhaustiveMF(self, MFtype, processes=1, orientation_estimate=None):

        def __computeMFminimization(sample):
            if self.include_csf:
                mf_dictionary = np.zeros((scheme.shape[0], num_fasc * num_atoms + 1), dtype=np.float64)
            else:
                mf_dictionary = np.zeros((scheme.shape[0], num_fasc * num_atoms), dtype=np.float64)
            est_dir1, est_dir2, mean_ang_err, nus_est, dir_est_failed, swapped_samples = self.__estimateOrientations(sample, orientation_estimate=orientation_estimate)
            failed_csd[sample, 0] = dir_est_failed
            mf_dictionary[:, :num_atoms] = mfu.interp_PGSE_from_multishell(
                self.synthetizer_dic['parameters']['scheme'],
                newdir=est_dir1,
                msinterp=self.ms_interpolator)
            mf_dictionary[:, num_atoms:num_atoms + num_atoms] = mfu.interp_PGSE_from_multishell(
                self.synthetizer_dic['parameters']['scheme'],
                newdir=est_dir2,
                msinterp=self.ms_interpolator)
            subdic_sizes = [num_atoms] * num_fasc
            if self.include_csf:
                mf_dictionary[:, num_fasc * num_atoms] = np.exp(-self.TE / T2_csf) * np.exp(-self.bvals * sig_csf)
                subdic_sizes.append(1)

            if MFtype == "MF":
                (w_nnz,  # non-negative weights (one per subdictionary)
                 ind_subdic,  # index of non-zero atom within each sub-dictionary
                 ind_totdic,  # index of non-zero atom in large dictionary
                 SoS,  # sum-of-squares residual
                 y_rec) = mfu.solve_exhaustive_posweights(mf_dictionary,
                                                          y_data[sample, :] / M0max,
                                                          np.atleast_1d(subdic_sizes))
            elif MFtype == "NoisyMF":
                (w_nnz,  # non-negative weights (one per subdictionary)
                 ind_subdic,  # index of non-zero atom within each sub-dictionary
                 ind_totdic,  # index of non-zero atom in large dictionary
                 SoS,  # sum-of-squares residual
                 y_rec) = mfe.solve_exhaustive_posweights_sigma(mf_dictionary,
                                                                y_data[sample, :] / M0max,
                                                                np.atleast_1d(subdic_sizes), verbose=True)
            # failed_csd[sample, 0] = dir_est_failed
            # est_orientations[sample, 0, :] = est_dir1
            # est_orientations[sample, 1, :] = est_dir2
            # mean_angular_errors[sample] = mean_ang_err

            # if (self.compute_vf):
            #    nus_ests[sample, :] = nus_est

            # sums_of_squares[sample] = SoS
            # weights_non_negative[sample, :] = w_nnz
            # indices_subdict[sample, :] = ind_subdic
            # indices_totdict[sample, :] = ind_totdic
            # y_reconstructed[sample, :] = y_rec

            #processed_samples += 1
            #if (processed_samples % 100 == 0):
            #    print(f'Progress {processed_samples}/{num_samples} ({processed_samples / num_samples * 100}%)')

            return (dir_est_failed, est_dir1, est_dir2, mean_ang_err, nus_est, SoS, w_nnz, ind_subdic, ind_totdic, y_rec, swapped_samples)


        if orientation_estimate is None:
            orientation_estimate = self.orientation_estimate
        type = self.synthetizer_dic['parameters']['type']
        task_name = self.synthetizer_dic['parameters']['task_name']
        run_id = self.synthetizer_dic['parameters']['run_id']

        num_samples = self.synthetizer_dic['parameters']['num_samples']
        num_fasc = self.synthetizer_dic['parameters']['num_fasc']
        dictionary = self.synthetizer_dic['parameters']['MF_dict']['dictionary']
        sig_csf = self.synthetizer_dic['parameters']['MF_dict']['sig_csf']
        T2_csf = self.synthetizer_dic['parameters']['MF_dict']['T2_csf']
        num_atoms = dictionary.shape[1]

        y_data = self.synthetizer_dic['DWI_noisy_store'].T
        M0s = self.synthetizer_dic['M0s']
        scheme = self.synthetizer_dic['parameters']['scheme']

        M0max = np.max(M0s)

        est_orientations = np.zeros((num_samples, num_fasc, 3))
        est_orientations_swapped = np.zeros((num_samples))

        if (self.compute_vf):
            nus_ests = np.zeros((num_samples, 3))  # Estimated volume fractions from dipy
        mean_angular_errors = np.zeros((num_samples))

        sums_of_squares = np.zeros((num_samples))
        if self.include_csf:
            weights_non_negative = np.zeros((num_samples, num_fasc + 1))
            indices_subdict = np.zeros((num_samples, num_fasc + 1), dtype=np.int32)
            indices_totdict = np.zeros((num_samples, num_fasc + 1), dtype=np.int32)
        else:
            weights_non_negative = np.zeros((num_samples, num_fasc))
            indices_subdict = np.zeros((num_samples, num_fasc), dtype=np.int32)
            indices_totdict = np.zeros((num_samples, num_fasc), dtype=np.int32)
        y_reconstructed = np.zeros((num_samples, scheme.shape[0]))

        failed_csd = np.zeros((num_samples, 1),
                              dtype='int16')  # Tracking of the samples for which the direction estimation failed to return 2 peaks

        print(f'Processing voxels (Exhaustive{MFtype}) ...')

        parallel = processes > 1
        if parallel:
            # create a process pool that uses all cpus
            with Pool(processes=processes) as pool:
                for result, sample in zip(pool.map(__computeMFminimization, range(num_samples)), range(num_samples)):
                    dir_est_failed, est_dir1, est_dir2, mean_ang_err, nus_est, SoS, w_nnz, ind_subdic, ind_totdic, y_rec, swapped_samples = result
                    failed_csd[sample, 0] = dir_est_failed
                    est_orientations[sample, 0, :] = est_dir1
                    est_orientations[sample, 1, :] = est_dir2
                    est_orientations_swapped[sample] = swapped_samples
                    mean_angular_errors[sample] = mean_ang_err

                    if (self.compute_vf):
                        nus_ests[sample, :] = nus_est

                    sums_of_squares[sample] = SoS
                    weights_non_negative[sample, :] = w_nnz
                    indices_subdict[sample, :] = ind_subdic
                    indices_totdict[sample, :] = ind_totdic
                    y_reconstructed[sample, :] = y_rec

        else:
            for sample in tqdm(range(num_samples)):
                dir_est_failed, est_dir1, est_dir2, mean_ang_err, nus_est, SoS, w_nnz, ind_subdic, ind_totdic, y_rec, swapped_samples = __computeMFminimization(sample)
                failed_csd[sample, 0] = dir_est_failed
                est_orientations[sample, 0, :] = est_dir1
                est_orientations[sample, 1, :] = est_dir2
                est_orientations_swapped[sample] = swapped_samples
                mean_angular_errors[sample] = mean_ang_err

                if (self.compute_vf):
                   nus_ests[sample, :] = nus_est

                sums_of_squares[sample] = SoS
                weights_non_negative[sample, :] = w_nnz
                indices_subdict[sample, :] = ind_subdic
                indices_totdict[sample, :] = ind_totdic
                y_reconstructed[sample, :] = y_rec

        MF_output = {
            "y_reconstructed": y_reconstructed,
            "sums_of_squares": sums_of_squares,
            "weights_non_negative": weights_non_negative,
            "indices_subdict": indices_subdict,
            "indices_totdict": indices_totdict,
            "est_orientations": est_orientations,
            "est_orientations_swapped": est_orientations_swapped,
            "gt_orientations": self.synthetizer_dic['orientations'],
            "mean_angular_errors": mean_angular_errors,
            "include_csf": self.include_csf,
            'failed_csd': failed_csd
        }

        output_path = os.path.join(self.base_path, "generator", f"type-{type}", f"{MFtype}")
        # Create folder if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = f"type-{type}_task-{task_name}_run-{run_id}_orientation-{orientation_estimate}_{MFtype}"
        json_metadata = {"task_name": task_name,
                         "run_id": run_id,
                         "type": type,
                         "orientation_estimate": orientation_estimate,
                         "num_samples": num_samples,
                         "include_csd": self.include_csf,

                         }

        with open(os.path.join(output_path, filename + ".json"), 'w') as outfile:
            json.dump(json_metadata, outfile, indent=4)

        with open(os.path.join(output_path, filename + ".pickle"), 'wb') as handle:
            pickle.dump(MF_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def computeExhaustiveMF(self, processes=1, orientation_estimate=None):
        self.__solveExhaustiveMF("MF",processes=processes, orientation_estimate=orientation_estimate)

    def computeExhaustiveNoisyMF(self, processes=1, orientation_estimate=None):
        self.__solveExhaustiveMF("NoisyMF",processes=processes, orientation_estimate=orientation_estimate)

    def computeNNLSWeights(self, sparse_storage=True, processes=1):

        def __computeNNLSminimization(sample):
            if self.include_csf:
                nnls_dictionary = np.zeros((scheme.shape[0], num_fasc * num_atoms + 1), dtype=np.float64)
            else:
                nnls_dictionary = np.zeros((scheme.shape[0], num_fasc * num_atoms), dtype=np.float64)

            est_dir1, est_dir2, mean_ang_err, nus_est, dir_est_failed, swapped_samples = self.__estimateOrientations(
                sample)

            if (self.orientation_estimate == 'GROUNDTRUTH'):
                est_dir1 = est_dir1 + 1e-6
                est_dir2 = est_dir2 + 1e-6

            # Compute NNLS weights
            nnls_dictionary[:, :num_atoms] = mfu.interp_PGSE_from_multishell(scheme, newdir=est_dir1,
                                                                             msinterp=self.ms_interpolator)
            nnls_dictionary[:, num_atoms:num_atoms + num_atoms] = mfu.interp_PGSE_from_multishell(scheme,
                                                                                                  newdir=est_dir2,
                                                                                                  msinterp=self.ms_interpolator)

            if self.include_csf:
                nnls_dictionary[:, -1] = np.exp(-self.TE / T2_csf) * np.exp(-self.bvals * sig_csf)

            (w_nnls, PP, _) = mfu.nnls_underdetermined(nnls_dictionary, y_data[sample, :])
            print("non zero w_nnls",np.count_nonzero(w_nnls))
            return (w_nnls, PP, est_dir1, est_dir2, mean_ang_err, nus_est, swapped_samples, dir_est_failed)

        type = self.synthetizer_dic['parameters']['type']
        task_name = self.synthetizer_dic['parameters']['task_name']
        run_id = self.synthetizer_dic['parameters']['run_id']

        num_samples = self.synthetizer_dic['parameters']['num_samples']
        num_fasc = self.synthetizer_dic['parameters']['num_fasc']

        dictionary = self.synthetizer_dic['parameters']['MF_dict']['dictionary']
        num_atoms = dictionary.shape[1]

        y_data = self.synthetizer_dic['DWI_noisy_store'].T
        scheme = self.synthetizer_dic['parameters']['scheme']

        sig_csf = self.synthetizer_dic['parameters']['MF_dict']['sig_csf']
        T2_csf = self.synthetizer_dic['parameters']['MF_dict']['T2_csf']

        # memory for storing NNLS weights
        if sparse_storage:
            sparsity = 0.01  # expected proportion of nnz atom weights per fascicle
            if self.include_csf:
                nnz_pred = int(np.ceil(sparsity * num_samples * (num_atoms * num_fasc + 1)))
            else:
                nnz_pred = int(np.ceil(sparsity * num_samples * num_atoms * num_fasc))
            nnz_cnt = 0  # non-zero entries (just for sparse case)
            # Store row and column indices of the dense weight matrix
            w_idx = np.zeros((nnz_pred, 2), dtype=np.int64)  # 2 is 2 !
            # Store weights themselves
            w_data = np.zeros(nnz_pred, dtype=np.float64)
        else:
            if self.include_csf:
                w_store = np.zeros((num_samples, num_fasc * num_atoms + 1), dtype=np.float)
            else:
                w_store = np.zeros((num_samples, num_fasc * num_atoms), dtype=np.float)
        nnz_hist = np.zeros(num_samples)  # always useful even in non sparse mode

        est_orientations = np.zeros((num_samples, num_fasc, 3))
        est_orientations_swapped = np.zeros((num_samples))
        mean_angular_errors = np.zeros((num_samples))
        failed_csd = np.zeros((num_samples,1), dtype = 'int16') #Tracking of samples for which the estimation of direction (SSST or MSMT) failed to return 2 peaks
        
        if(self.compute_vf):
            nus_ests = np.zeros([num_samples,3]) #To store volume fractions estimations : CSF,GM,WM
        else:
            nus_ests = np.array([])



        parallel = processes > 1
        if parallel:
            # create a process pool that uses all cpus
            with Pool(processes=processes) as pool:
                for result, sample in zip(pool.map(__computeNNLSminimization, range(num_samples)), range(num_samples)):

                    w_nnls, PP, est_dir1, est_dir2, mean_ang_err, nus_est, swapped_samples, dir_est_failed = result

                    nnz_hist[sample] = PP.size
                    est_orientations[sample, 0, :] = est_dir1
                    est_orientations[sample, 1, :] = est_dir2
                    est_orientations_swapped[sample] = swapped_samples
                    mean_angular_errors[sample] = mean_ang_err
                    failed_csd[sample, 0] = dir_est_failed

                    if (self.compute_vf):
                        nus_ests[sample, :] = nus_est  # Volume fractions estimation from dipy

                    if sparse_storage:
                        # Check size and double it if needed
                        if nnz_cnt + PP.size > w_data.shape[0]:
                            w_idx = np.concatenate((w_idx, np.zeros(w_idx.shape).astype('int64')), axis=0)
                            w_data = np.concatenate((w_data, np.zeros(w_data.shape)), axis=0)
                            print("Doubled size of index and weight arrays after sample %d "
                                  "(adding %d non-zero elements to %d, exceeding arrays' "
                                  " size of %d)"
                                  % (sample + 1, PP.size, nnz_cnt, w_data.shape[0]))
                        w_data[nnz_cnt:nnz_cnt + PP.size] = w_nnls[PP]

                        w_idx[nnz_cnt:nnz_cnt + PP.size, 0] = sample  # row indices
                        w_idx[nnz_cnt:nnz_cnt + PP.size, 1] = PP  # column indices

                        nnz_cnt += PP.size
                    else:
                        w_store[sample, :] = w_nnls
        else:
            print('Processing Voxels (NNLS) ... ')
            for sample in tqdm(range(num_samples)):

                w_nnls, PP, est_dir1, est_dir2, mean_ang_err, nus_est, swapped_samples, dir_est_failed = __computeNNLSminimization(sample)

                nnz_hist[sample] = PP.size
                est_orientations[sample, 0, :] = est_dir1
                est_orientations[sample, 1, :] = est_dir2
                est_orientations_swapped[sample] = swapped_samples
                mean_angular_errors[sample] = mean_ang_err
                failed_csd[sample, 0] = dir_est_failed

                if (self.compute_vf):
                    nus_ests[sample, :] = nus_est  # Volume fractions estimation from dipy


                if sparse_storage:
                    # Check size and double it if needed
                    if nnz_cnt + PP.size > w_data.shape[0]:
                        w_idx = np.concatenate((w_idx, np.zeros(w_idx.shape).astype('int64')), axis=0)
                        w_data = np.concatenate((w_data, np.zeros(w_data.shape)), axis=0)
                        print("Doubled size of index and weight arrays after sample %d "
                              "(adding %d non-zero elements to %d, exceeding arrays' "
                              " size of %d)"
                              % (sample + 1, PP.size, nnz_cnt, w_data.shape[0]))
                    w_data[nnz_cnt:nnz_cnt + PP.size] = w_nnls[PP]

                    w_idx[nnz_cnt:nnz_cnt + PP.size, 0] = sample  # row indices
                    w_idx[nnz_cnt:nnz_cnt + PP.size, 1] = PP  # column indices

                    nnz_cnt += PP.size
                else:
                    w_store[sample, :] = w_nnls


        if sparse_storage:
            # Discard unused memory
            w_idx = w_idx[:nnz_cnt, :]
            w_data = w_data[:nnz_cnt]

        output_path = os.path.join(self.base_path, "generator", f"type-{type}","NNLS")
        # Create folder if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = f"type-{type}_task-{task_name}_run-{run_id}_orientation-{self.orientation_estimate}_NNLS"
        json_metadata = {"task_name": task_name,
                         "run_id": run_id,
                         "type": type,
                         "orientation_estimate": self.orientation_estimate,
                         "compute_swap":self.compute_swap,
                         "num_samples": num_samples,
                         "sparse_storage": sparse_storage,
                         "include_csf": self.include_csf,
                         }

        with open(os.path.join(output_path, filename + ".json"), 'w') as outfile:
            json.dump(json_metadata, outfile, indent=4)



        if sparse_storage:
            NNLS_output = {
                "nnz_hist": nnz_hist,
                "w_idx": w_idx,
                "w_data": w_data,
                "est_orientations": est_orientations,
                "est_orientations_swapped": est_orientations_swapped,
                "gt_orientations": self.synthetizer_dic['orientations'],
                "mean_angular_errors": mean_angular_errors,
                "nus_ests": nus_ests.tolist(),
                "nus_csf": self.synthetizer_dic['nuscsf'],
                'failed_csd':failed_csd,
                
            }
            
            with open(os.path.join(output_path, filename + ".pickle"), 'wb') as handle:
                pickle.dump(NNLS_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            NNLS_weights = w_store
            np.save(os.path.join(output_path, filename), NNLS_weights)

    def computeSphericalHarmonics(self, M0_estimation=True, sh_max_order=12):

        num_samples = self.synthetizer_dic['parameters']['num_samples']
        y_data = self.synthetizer_dic['DWI_noisy_store'].T

        bO_indices = np.where(self.bvals <= 10)[0]

        # M0 scaling : TO DO : use the same heuristic for signal scaling in graph processing
        if M0_estimation:
            M0_est = np.zeros(num_samples)
            for sample in range(num_samples):
                # M0_est[i] = np.mean(y_data[b0_positions, i])
                norm_DW = np.max(y_data[sample, bO_indices])
                M0_est[sample] = max(y_data[sample, bO_indices])
                y_data[sample] = y_data[sample] / norm_DW

            del M0_est
        else:
            M0 = 500
            y_data = y_data / M0

        basis_size = int((sh_max_order + 1) * (sh_max_order + 2) / 2)

        shells = unique_bvals_tolerance(self.bvals, 
                                        0.02*np.min(self.bvals[self.bvals>30])) #TO DO : replace with non hard coded values
        shells = shells[shells>0]
        SH_ALL = np.zeros((y_data.shape[0], len(shells), basis_size))
        Y_reconstruct = np.zeros((y_data.shape[0], len(shells)))

        coef_matrix = []
        is_DWI_shell = []
        B_sh_full=[]

        print('Shells : ', shells)
        
        tol = 0.01 * shells[0]
        print('Processing voxels per shell (SphericalHarmonics)')
        for ks in tqdm(range(shells.shape[0])):
            s = shells[ks]
            is_shell = np.logical_and(s - tol < self.bvals, self.bvals < s + tol) #is_shell[k] is true if and only if bvecs[k] belongs to the current shell s
            # SH basis in native protocol space
            (r, theta_acq, phi_acq) = cart2sphere(self.bvecs[is_shell, 0], 
                                                  self.bvecs[is_shell, 1], 
                                                  self.bvecs[is_shell, 2])

            (B_sh, m_sh, n_sh) = real_sym_sh_basis(sh_max_order, theta_acq[:, None], phi_acq[:, None])

            assert B_sh.shape[1] == basis_size, "Check SH basis size"
            B_sh_full.append(B_sh)
            L = -n_sh * (n_sh + 1)  # Laplace Beltrami operator for each SH basis function, 1-D array
            smooth = 0.006  # Maxime Descoteaux's default value is 6e-3
            invB = smooth_pinv(B_sh, np.sqrt(smooth) * L)
            coef_matrix.append(invB)
            is_DWI_shell.append(is_shell)
            SH_ALL[:,ks,:] = (invB@y_data[:,is_shell].T).T

        type = self.synthetizer_dic['parameters']['type']
        task_name = self.synthetizer_dic['parameters']['task_name']
        run_id = self.synthetizer_dic['parameters']['run_id']

        output_path = os.path.join(self.base_path, "generator", f"type-{type}","SH")
        # Create folder if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = f"type-{type}_task-{task_name}_run-{run_id}_SH"
        json_metadata = {"task_name": task_name,
                         "run_id": run_id,
                         "type": type,
                         "sh_max_order": sh_max_order,
                         "M0_estimation": M0_estimation,
                         "num_samples": num_samples,
                         "basis_size": basis_size,
                         "shells": shells.tolist(),
                         }

        with open(os.path.join(output_path, filename + ".json"), 'w') as outfile:
            json.dump(json_metadata, outfile, indent=4)

        np.save(os.path.join(output_path,filename), SH_ALL)

