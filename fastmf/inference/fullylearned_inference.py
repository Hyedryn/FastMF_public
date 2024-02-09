
import numpy as np
import os
import time
import nibabel as nib
import torch
import json
import pickle
from tqdm import tqdm

from dipy.reconst.shm import real_sym_sh_basis, smooth_pinv
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import unique_bvals_tolerance

import fastmf.models.MLP_FullyLearned as MLP_Full
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
class FullyLearned_Model():
    def __init__(self, metadata_Full, model_state_dict_Full, scaling_fn, dictionary_path, scaling_fn_path=None, device='cpu'):
        with open(metadata_Full, 'r') as f:
            self.metadata_Full = json.load(f)
        self.in_normalization_Full = self.metadata_Full["in_normalization"]
        self.target_normalization_Full = self.metadata_Full["target_normalization"]

        print("Normalization: ", self.in_normalization_Full, self.target_normalization_Full)

        self.model_Full = MLP_Full.Network(self.metadata_Full['architecture'])

        self.model_Full.load_state_dict(torch.load(model_state_dict_Full))
        self.model_Full.eval()
        self.model_Full.to(device)

        if (scaling_fn is None):
            self.scaling_fn_full = lambda *args: args[0]
        elif (scaling_fn == 'MinMax'):
            scaler_path = scaling_fn_path #os.path.join(base_path, 'scaler', 'scaler-minmax_ses-{0}_SH.pickle'.format(session_id))
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)

            self.maxis_full = scaler.data_max_
            self.minis_full = scaler.data_min_

            self.scaling_fn_full = lambda x, mi, ma, inverse: MinMaxScaler(x, mi, ma, inverse=inverse)

        self.sh_max_order = 12
        self.device = device

        dictionary_structure = mfu.loadmat(dictionary_path)
        self.fasc_propnames = [s.strip() for s in dictionary_structure['fasc_propnames']]

    def fit(self,
            data, mask,  # requires 1 of 3
            pgse_scheme=None, bvals=None, bvecs=None,
            verbose=1, M0_estimation=True):

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
        for i in range(ROI_size):
            vox = [axis[i] for axis in ROI]  # can be 1D, 2D, 3D, ...
            vox = tuple(vox)  # required for array indexing
            data_arr_1D[i, :] = data_arr[vox + (slice(None),)]

        y_data = data_arr_1D
        num_samples = y_data.shape[0]

        bO_indices = np.where(bvals <= 10)[0]
        # M0 scaling : TO DO : use the same heuristic for signal scaling in graph processing
        if M0_estimation:
            M0_est = np.zeros(num_samples)
            for sample in range(num_samples):
                # M0_est[i] = np.mean(y_data[b0_positions, i])
                norm_DW = np.max(y_data[sample, bO_indices])
                M0_est[sample] = max(y_data[sample, bO_indices])
                y_data[sample] = y_data[sample] / norm_DW
        else:
            M0 = 500
            y_data = y_data / M0
        if VRB >= 1:
            print("Mask succesfully applied on data_arr")

        sh_max_order = self.sh_max_order
        basis_size = int((sh_max_order + 1) * (sh_max_order + 2) / 2)

        shells = unique_bvals_tolerance(bvals,
                                        0.1 * np.min(bvals[bvals > 50]))  # TO DO : replace with non hard coded values
        shells = shells[shells > 0]
        SH_ALL = np.zeros((y_data.shape[0], len(shells), basis_size))

        coef_matrix = []
        is_DWI_shell = []
        B_sh_full = []

        if VRB >= 1:
            print('Shells : ', shells)

        tol = 0.01 * shells[0]
        print('Processing voxels per shell (SphericalHarmonics)')
        for ks in tqdm(range(shells.shape[0])):
            s = shells[ks]
            is_shell = np.logical_and(s - tol < bvals, bvals < s + tol)  # is_shell[k] is true if and only if bvecs[k] belongs to the current shell s
            # SH basis in native protocol space
            (r, theta_acq, phi_acq) = cart2sphere(bvecs[is_shell, 0],
                                                  bvecs[is_shell, 1],
                                                  bvecs[is_shell, 2])

            (B_sh, m_sh, n_sh) = real_sym_sh_basis(sh_max_order, theta_acq[:, None], phi_acq[:, None])

            assert B_sh.shape[1] == basis_size, "Check SH basis size"
            B_sh_full.append(B_sh)
            L = -n_sh * (n_sh + 1)  # Laplace Beltrami operator for each SH basis function, 1-D array
            smooth = 0.006  # Maxime Descoteaux's default value is 6e-3
            invB = smooth_pinv(B_sh, np.sqrt(smooth) * L)
            coef_matrix.append(invB)
            is_DWI_shell.append(is_shell)
            SH_ALL[:, ks, :] = (invB @ y_data[:, is_shell].T).T

        input_SH = SH_ALL

        print("Applying SH network")
        pred_full = self.model_Full(torch.from_numpy(input_SH.reshape(input_SH.shape[0], input_SH.shape[1] * input_SH.shape[2]).astype(
                    'float32'))).detach().numpy()

        scaled_pred_full = self.scaling_fn_full(pred_full[:, :],
                                                self.minis_full[:],
                                                self.maxis_full[:],
                                                True)
        fitinfo = {'ROI_size': ROI_size,
                   'maxfasc': 2,
                   'affine': nii_affine,
                   'mask': mask_arr,
                   'fasc_propnames': self.fasc_propnames,
                   'num_params': 12}

        params = scaled_pred_full

        return FullyLearnedModelFit(fitinfo, params, VRB, M0_est)

class FullyLearnedModelFit():
    def __init__(self, fitinfo, model_params, verbose=0, M0_est=None):
        """
        """
        self.affine = fitinfo['affine']

        numfasc = fitinfo['maxfasc']
        mask = fitinfo['mask']
        ROI_size = model_params.shape[0]
        assert ROI_size == np.sum(mask > 0), ('Inconsistent mask and model '
                                              'parameter array')

        # M0
        self.M0 = np.zeros(mask.shape)
        if M0_est is not None:
            self.M0[mask > 0] = M0_est
        parlist = []

        print("nu")
        # Create maps of fascicles' fractions nu and main orientations
        for k in range(numfasc):
            # volume fraction map
            print(str(k*(len(fitinfo['fasc_propnames']) + 3) + k))
            nu_k = model_params[:, k*(len(fitinfo['fasc_propnames']) +3) + k]
            prop_map = np.zeros(mask.shape)
            prop_map[mask > 0] = nu_k
            par_name = 'frac_f%d' % k
            setattr(self, par_name, prop_map)
            parlist.append(par_name)

            # peak map
            s_peaks = 1 + (len(fitinfo['fasc_propnames'])+3+1) * (k)
            p_k = model_params[:, s_peaks:s_peaks+3]  # (ROI_size, 3)
            prop_map = np.zeros(mask.shape + (3,))
            prop_map[mask > 0] = p_k
            par_name = 'peak_f%d' % k
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
        h=3
        print("metrics")
        for propname in fitinfo['fasc_propnames']:
            prop_tot_in_mask = np.zeros(ROI_size)  # map of voxel average
            for k in range(numfasc):
                # Create map of fascicle-specific properties
                # Get fraction of fascicle k
                nu_k = model_params[:, k*(len(fitinfo['fasc_propnames']) + 3) + k]
                # get ID of fingerprint in subdictionary k
                #ID_k = model_params[:, 1 + numfasc + k].astype(int)
                # Leave property to zero if no weight assigned to fascicle!
                print(propname)
                print(str(1 + h + k*(len(fitinfo['fasc_propnames'])+3)+k))
                prop_k_in_mask = model_params[:, 1 + h + k*(len(fitinfo['fasc_propnames'])+3)+k] * (nu_k > 0)
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



