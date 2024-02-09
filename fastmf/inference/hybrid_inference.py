
import numpy as np
import os
import time
import nibabel as nib
import torch
import json
import pickle
from tqdm import tqdm

import fastmf.models.MLP_Split as MLP_Split
import fastmf.utils.mf_utils as mfu


def MinMaxScaler(x, minis, maxis, inverse=False):
    if (maxis.shape[0] > x.shape[1]):
        maxis = maxis[:-1]
        minis = minis[:-1]
    if (inverse):
        a = maxis - minis
        b = minis
    else:
        a = 1 / (maxis - minis)
        b = - minis * a

    return x * a + b
class Hybrid_Model():
    def __init__(self, metadata_Hybrid, model_state_dict_Hybrid, scaling_fn, dictionary_path, scaling_fn_path=None, device='cpu'):
        with open(metadata_Hybrid, 'r') as f:
            self.metadata_Hybrid = json.load(f)


        self.in_normalization_Hybrid = self.metadata_Hybrid["in_normalization"]
        self.target_normalization_Hybrid = self.metadata_Hybrid["target_normalization"]

        self.model_Hybrid = MLP_Split.Network(self.metadata_Hybrid['split_architecture'],
                                              self.metadata_Hybrid['final_architecture'])

        self.model_Hybrid.load_state_dict(torch.load(model_state_dict_Hybrid))
        self.model_Hybrid.eval()
        self.model_Hybrid.to(device)

        if (scaling_fn is None):
            self.scaling_fn_hybrid = lambda *args: args[0]
        elif (scaling_fn == 'MinMax'):
            scaler_path = scaling_fn_path #os.path.join(base_path, 'scaler', 'scaler-minmax_ses-{0}_SH.pickle'.format(session_id))
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)

            self.maxis_hybrid = scaler.data_max_
            self.minis_hybrid = scaler.data_min_

            self.scaling_fn_hybrid = lambda x, mi, ma, inverse: MinMaxScaler(x, mi, ma, inverse=inverse)

        self.device = device

        dictionary_structure = mfu.loadmat(dictionary_path)
        self.fasc_propnames = [s.strip() for s in dictionary_structure['fasc_propnames']]
        self.num_fasc = 2
        self.num_atoms = dictionary_structure['dictionary'].shape[1]
        self.include_csf = False
        self.sig_csf = dictionary_structure['sig_csf']
        self.T2_csf = dictionary_structure['T2_csf']
        self.sch_mat = dictionary_structure['sch_mat']

        self.ms_interpolator = mfu.init_PGSE_multishell_interp(
            dictionary_structure['dictionary'],
            dictionary_structure['sch_mat'],
            dictionary_structure['orientation'])

    def fit(self,
            data, mask, peaks,# requires 1 of 3
            pgse_scheme=None, bvals=None, bvecs=None,
            verbose=1, M0_estimation=True, weight_path=None):


        def __computeNNLSminimization(sample):
            if self.include_csf:
                nnls_dictionary = np.zeros((scheme.shape[0], self.num_fasc * self.num_atoms + 1), dtype=np.float64)
            else:
                nnls_dictionary = np.zeros((scheme.shape[0], self.num_fasc * self.num_atoms), dtype=np.float64)

            est_dir1 = peaks_1D[sample, 0, :]
            est_dir2 = peaks_1D[sample, 1, :]

            # Compute NNLS weights
            nnls_dictionary[:, :self.num_atoms] = mfu.interp_PGSE_from_multishell(scheme, newdir=est_dir1,
                                                                             msinterp=self.ms_interpolator)
            nnls_dictionary[:, self.num_atoms:self.num_atoms + self.num_atoms] = mfu.interp_PGSE_from_multishell(scheme,
                                                                                                  newdir=est_dir2,
                                                                                                  msinterp=self.ms_interpolator)

            if self.include_csf:
                nnls_dictionary[:, -1] = np.exp(-TE / self.T2_csf) * np.exp(-bvals * self.sig_csf)

            (w_nnls, PP, _) = mfu.nnls_underdetermined(nnls_dictionary, y_data[sample, :])

            return (w_nnls, PP)


        assert pgse_scheme is None or bvals is None, "Only one of pgse_scheme or bvals can be provided."
        assert pgse_scheme is not None or (bvals is not None and bvecs is not None), "If pgse_scheme is not provided, bvals and bvecs must be."

        if pgse_scheme is not None:
            if isinstance(pgse_scheme, str):
                pgse_scheme = np.loadtxt(pgse_scheme)
            elif isinstance(pgse_scheme, np.ndarray):
                pass
            else:
                raise ValueError("pgse_scheme must be a file path or a numpy array.")

            gam = mfu.get_gyromagnetic_ratio('H')
            G = pgse_scheme[:, 3]
            Deltas = pgse_scheme[:, 4]
            deltas = pgse_scheme[:, 5]

            bvals = (gam * G * deltas) ** 2 * (Deltas - deltas / 3)  # in SI units s/m^2
            bvecs = pgse_scheme[:, :3]
            scheme = pgse_scheme

        if bvals is not None and bvecs is not None:
            if isinstance(bvals,str):
                bvals = np.loadtxt(bvals)
            elif isinstance(bvals, np.ndarray):
                pass
            else:
                raise ValueError("bvals must be a file path or a numpy array.")

            if isinstance(bvecs,str):
                bvecs = np.loadtxt(bvecs).T
            elif isinstance(bvecs, np.ndarray):
                pass
            else:
                raise ValueError("bvecs must be a file path or a numpy array.")
            assert bvals.shape[0] == bvecs.shape[0], "bvals and bvecs must have the same number of samples."

            scheme = np.zeros((bvals.shape[0], 7))
            scheme[:, :3] = bvecs

            gam = mfu.get_gyromagnetic_ratio('H')
            G = np.mean(self.sch_mat[:, 3])
            Delta = np.mean(self.sch_mat[:, 4])
            delta = np.mean(self.sch_mat[:, 5])
            TE = np.mean(self.sch_mat[:, 6])

            #scheme[:, 3] = G
            scheme[:, 3] = np.clip((10**3)*np.sqrt(bvals/(Delta - delta / 3)) / (gam * delta), None, np.max(self.sch_mat[:,3])) # in SI units s/m^2
            scheme[:, 4] = Delta
            scheme[:, 5] = delta
            scheme[:, 6] = TE


        TE = np.mean(scheme[:, 6])

        VRB = verbose

        if VRB >= 3:
            print("bvecs shape : ", bvecs.shape)
            print("bvals shape : ", bvals.shape)
        # ------------------
        # Required arguments
        # ------------------
        # DWI Data
        nii_affine = None  # spatial affine transform for DWI data
        if isinstance(data, str):
            st_0 = time.time()
            if VRB >= 2:
                print("Loading data from file %s..." % data)
            nii_affine = nib.load(data).affine
            data_arr = nib.load(data).get_fdata()
            dur_0 = time.time() - st_0
            if VRB >= 2:
                print("Data loaded in %g s." % dur_0)
        else:
            data_arr = data  # no need to copy, won't be modified

        # ROI mask
        if isinstance(mask, str):
            if nii_affine is None:
                nii_affine = nib.load(mask).affine
            mask_arr = nib.load(mask).get_fdata()
        else:
            mask_arr = mask  # no need to copy, won't be modified

        img_shape = mask_arr.shape
        ROI = np.where(mask_arr > 0)  # (x,) (x,y) or (x,y,z)
        ROI_size = ROI[0].size

        if ROI_size == 0:
            raise ValueError("No voxel detected in mask. Please provide a non-empty mask.")

        if data_arr.shape[:-1] != img_shape:
            raise ValueError("Data and mask not compatible. Based on data, mask should have shape (%s), got (%s) instead." %
                             (" ".join("%d" % x for x in data_arr.shape[:-1]), " ".join("%d" % x for x in img_shape)))

        data_arr_1D = np.zeros((ROI_size, data_arr.shape[3]))
        peaks_1D = np.zeros((ROI_size, 2, 3))
        for i in range(ROI_size):
            vox = [axis[i] for axis in ROI]  # can be 1D, 2D, 3D, ...
            vox = tuple(vox)  # required for array indexing
            data_arr_1D[i, :] = data_arr[vox + (slice(None),)]
            peaks_1D[i, :, :] = peaks[vox + (slice(None), slice(None))]

        y_data = data_arr_1D
        num_samples = y_data.shape[0]

        if VRB >= 1:
            print("Mask succesfully applied on data_arr")

        if self.include_csf:
            w_store = np.zeros((num_samples, self.num_fasc * self.num_atoms + 1), dtype=np.float64)
        else:
            w_store = np.zeros((num_samples, self.num_fasc * self.num_atoms), dtype=np.float64)

        print('Processing voxels')

        if weight_path is None:
            for sample in tqdm(range(num_samples)):
                w_nnls, PP = __computeNNLSminimization(sample)
                w_store[sample, :] = w_nnls

            if self.in_normalization_Hybrid == "SumToOne":
                w_store = w_store / np.sum(w_store, axis=1)[:, np.newaxis]
            elif self.in_normalization_Hybrid == "Standard":
                w_store = (w_store - np.mean(w_store, axis=1)[:, np.newaxis])
                std_NNLS = w_store / np.std(w_store, axis=1)[:, np.newaxis]
                idx_pos_std = np.where(std_NNLS > 0)[0]
                w_store[idx_pos_std, :] = (w_store[idx_pos_std, :] / std_NNLS[idx_pos_std][:, np.newaxis])

                del idx_pos_std, std_NNLS
        else:
            w_store = np.load(weight_path)

        print("Applying MLP_Split network")
        pred_hybrid = self.model_Hybrid(torch.from_numpy(w_store.astype(
                    'float32'))).detach().numpy()

        scaled_pred_hybrid = self.scaling_fn_hybrid(pred_hybrid[:, :],
                                                self.minis_hybrid[:],
                                                self.maxis_hybrid[:],
                                                True)
        fitinfo = {'ROI_size': ROI_size,
                   'maxfasc': 2,
                   'affine': nii_affine,
                   'mask': mask_arr,
                   'fasc_propnames': self.fasc_propnames,
                   'num_params': 12}

        params = scaled_pred_hybrid

        return HybridModelFit(fitinfo, params, w_store, VRB)

class HybridModelFit():
    def __init__(self, fitinfo, model_params, w_store, verbose=0, M0_est=None):
        """
        """
        self.affine = fitinfo['affine']

        numfasc = fitinfo['maxfasc']
        mask = fitinfo['mask']
        ROI_size = model_params.shape[0]
        assert ROI_size == np.sum(mask > 0), ('Inconsistent mask and model '
                                              'parameter array')

        self.w_store = w_store

        # M0
        self.M0 = np.zeros(mask.shape)
        if M0_est is not None:
            self.M0[mask > 0] = M0_est
        parlist = []

        print("nu")
        # Create maps of fascicles' fractions nu and main orientations
        for k in range(numfasc):
            # volume fraction map
            print(str(k*(len(fitinfo['fasc_propnames'])) + k))
            nu_k = model_params[:, k*(len(fitinfo['fasc_propnames'])) + k]
            prop_map = np.zeros(mask.shape)
            prop_map[mask > 0] = nu_k
            par_name = 'frac_f%d' % k
            setattr(self, par_name, prop_map)
            parlist.append(par_name)

        # Fascicle-specific parameters and voxel-wise total parameters.
        # The latter refers to parameters averaged over all fascicles of
        # the voxel as
        # M_{tot} = sum_{k=1}^K nu_k M_k,
        # where K is the number of fascicles,
        # nu_k is the physical fraction of the voxel occupied by fascicle k
        # and M_k  is the microstructural property of fascicle k.
        # Note that M_{tot} may not always make sense physically.
        # For instance iff M is  the axon density then M_{tot} is the actual
        # physical fraction of space located inside axons in the voxel.
        # However if M represents the axonal radius then it
        # is much less clear what M_{tot} represents. Note also that
        # sum_{k=1}^K nu_k can be < 1 because of the presence of non-fascicle
        # compartments such as cerebro-spinal fluid (CSF) or extraaxonal
        # restricted (EAR).
        h=1
        print("metrics")
        for propname in fitinfo['fasc_propnames']:
            prop_tot_in_mask = np.zeros(ROI_size)  # map of voxel average
            for k in range(numfasc):
                # Create map of fascicle-specific properties
                # Get fraction of fascicle k
                nu_k = model_params[:, k*(len(fitinfo['fasc_propnames'])) + k]
                # get ID of fingerprint in subdictionary k
                #ID_k = model_params[:, 1 + numfasc + k].astype(int)
                # Leave property to zero if no weight assigned to fascicle!
                print(propname)
                print(str(h + k*len(fitinfo['fasc_propnames'])+k))
                prop_k_in_mask = model_params[:, h + k*len(fitinfo['fasc_propnames'])+k] * (nu_k > 0)
                # update average parameter
                prop_tot_in_mask += nu_k * prop_k_in_mask
                # Create map of fascicle-specific parameter and store it
                prop_map = np.zeros(mask.shape)
                prop_map[mask > 0] = prop_k_in_mask
                par_name = propname + '_f%d' % k  # start numbering at zero
                setattr(self, par_name, prop_map)
                parlist.append(par_name)
            # Create map of voxel-wise average property and store it
            prop_map = np.zeros(mask.shape)  # reuse prop_map to save memory
            prop_map[mask > 0] = prop_tot_in_mask
            par_tot_name = propname + '_tot'
            setattr(self, par_tot_name, prop_map)
            parlist.append(par_tot_name)
            h=h+1

        """  # Mean squared error
        self.MSE = np.zeros(mask.shape)
        self.MSE[mask > 0] = model_params[:, -2]
        parlist.append('MSE')

        # R squared (coefficient of determination, square of Pearson)
        self.R2 = np.zeros(mask.shape)
        self.R2[mask > 0] = model_params[:, -1]
        parlist.append('R2')"""

        # Store parameter names
        self.param_names = parlist

        # Display progress and user instructions
        if verbose >= 2:
            print("Microstructure Fingerprinting fit object constructed.")
            print("Assuming the fit object was named \'MF_fit\', "
                  "\'MF_fit.param_names\' gives you a list of all "
                  "the property maps as strings.\n"
                  "You can access those maps (NumPy arrays) "
                  "via \'MF_fit.property_name\',"
                  " where \'property_name\' is any of those strings."
                  "Here the list is the following:")
            for p in parlist:
                print('\t%s' % (p,))
            print("You can call \'MF_fit.write_nifti\' to write the "
                  "corresponding NIfTI files.",flush=True)

    def write_weights(self, output_name):

        np.save(output_name, self.w_store)

    def write_nifti(self, output_basename, affine=None):
        """Exports maps of fitted parameter as NIfTI files.
        Parameters
        ----------
        output_basename: str
            unix/like/path/to/output_file. In order to
            force the creation of compressed .nii.gz archives, provide a
            base name with the .nii.gz extension.
        affine: NumPy array
            Array with shape (4, 4), usually obtained as
            `affine = loaded_nifti_object.affine`. If not specified, an
            attempt will be made at finding an affine transform
            from the NIfTI files provided during the fitting.
        Returns
        -------
        fnames : list of all the files created.
        """
        if affine is None:
            affine = self.affine
        if affine is None:
            # no affine ever given to Fit object
            msg = ("Argument affine must be explicitely passed  because "
                   "no affine transform matrix was found during model "
                   "fitting. Expecting NumPy array with shape (4, 4).")
            raise ValueError(msg)

        # Special case for tarred archives .nii.gz
        niigz = '.nii.gz'
        if (len(output_basename) > len(niigz) and
                output_basename[-len(niigz):] == niigz):
            # case of .nii.gz file extension
            (path, fname) = os.path.split(output_basename[:-len(niigz)])
            ext = niigz
        else:
            # tail never contains a slash
            (path, tail) = os.path.split(output_basename)
            # ext is empty or starts with a period
            (fname, ext) = os.path.splitext(tail)
            if ext not in ['', '.nii']:
                raise ValueError("Unknown NIfTI extension %s in output %s" %
                                 (ext, output_basename))
            ext = '.nii'

        basename = os.path.join(path, fname)
        fnames = []
        for p in self.param_names:
            nii = nib.Nifti1Image(getattr(self, p), affine)
            nii_fname = '%s_%s%s' % (basename, p, ext)
            nib.save(nii, nii_fname)
            fnames.append(nii_fname)
        return fnames



