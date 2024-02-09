import os
import time
import tempfile
import numpy as np
import scipy
from itertools import product
import scipy.optimize
from tqdm import tqdm

# %% objective function
def L(Asmall, y, x):
    # y : measurement
    # x : w1,w2,sigma
    # Asmall : Sub dic

    u = np.sum(x[:Asmall.shape[1]] * Asmall, axis=1)
    u = u**2 + x[-1]**2
    return np.sum((y - np.sqrt(u))**2)

def solve_one_comb(Asmall, y):
    sigma_g = 1
    m, n = Asmall.shape
    x0 = np.zeros(n+1)
    x0[:-1] = np.ones(n) / n
    x0[-1] = sigma_g
    resu = scipy.optimize.minimize(lambda x: L(Asmall, y, x), x0=x0,
                                   method = 'Nelder-Mead',
                                   options = {'maxiter': 1000})

    return resu.x, resu.fun

def solve_exhaustive_posweights_sigma(A, y, dicsizes, printmsg=None, verbose=False, cMode=True):
    """
    Args:
      A: 2-D NumPy array.
      y: 1-D NumPy array of length A.shape[0].
      dicsizes: 1-D NumPy array containing strictly positive integers
        representing the size of each sub-dictionary in A.
        Their sum must equal A.shape[1].
      printmsg: str to be printed at the start of the execution of the
        function. Useful on computing clusters or in parallel computing.
        Default: None.

    Returns:
      w_nneg: 1-D NumPy array containing the K non-negative weights assigned
        to the one optimal column in each sub-dictionary. To get the full
        optimal w_opt do:
            w_opt = numpy.zeros(A.shape[1])
            w_opt[ind_atoms_totdic] = w_nneg.
      ind_atoms_subdic: 1-D numy array of size K containing the index of the
        column selected (having a non-zero weight) within each sub-dictionary
        Ak, i.e. ind_atoms_subdic[k] is in [0, dicsizes[k][.
      ind_atoms_totdic: 1-D NumPy array of size K containing the indices of
        all columns with non-zero weight in A, i.e. ind_atoms_totdic[k] is in
        [0, A.shape[1][.
      min_obj: floating-point scalar equal to ||Aw_opt-y||_2^2.
      y_recons: 1-D NumPy array equal to Aw_opt, i.e. the model prediction.
    """
    # Print message (can be useful on computing clusters or in // computing)
    if printmsg is not None:
        print(printmsg, end="")

    # --- Check inputs ---
    # A should be a 2D NumPy array
    assert isinstance(A, np.ndarray), "A should be a NumPy ndarray"
    assert A.ndim == 2, "A should be a 2D array"
    # A should not have zero columns
    assert not np.any(np.all(A == 0, axis=0)), "All-zero columns detected in A"
    # A should contain floating-point numbers
    if A.dtype is not np.float64:
        A = A.astype(np.float64)
    # y should be a NumPy float64 array
    assert isinstance(y, np.ndarray), "y should be a NumPy ndarray"
    if y.dtype is not np.float64:
        y = y.astype(np.float64)
    # Refuse empty data
    assert A.size > 0 and y.size > 0, "A and y should not be empty arrays"
    # A.shape[0] should match y
    msg = ("Number of rows in A (%d) should match number of elements in y (%d)"
           % (A.shape[0], y.size))
    assert A.shape[0] == y.size, msg

    # diclengths should be a NumPy int32 array with strictly positive entries
    assert isinstance(dicsizes, np.ndarray), ("dicsizes should be a "
                                              "NumPy ndarray")
    assert np.all(dicsizes > 0), "All entries of dicsizes should be > 0"
    if dicsizes.dtype is not np.int64:
        dicsizes = dicsizes.astype(np.int64)

    # Sum of subsizes should match total size of A
    msg = ("Number of columns of A (%d) does not equal sum of size of "
           "sub-matrices in diclengths array (%d)"
           % (A.shape[1], np.sum(dicsizes)))
    assert A.shape[1] == np.sum(dicsizes), msg


    # y is often read-only when passed by multiprocessing functions such as
    # multiprocessing.Pool.starmap/map, ipyparallel.Client.map/map_async, etc.
    # This made for Numba compilation errors in lsqnonneg_1var, lsqnonneg_2var
    if y.flags['WRITEABLE'] is False:
        y = y.copy()
        y.flags.writeable = True

    N_LSC = dicsizes.size  # number of large-scale compartments in voxel
    end_ind = np.cumsum(dicsizes)  # indices excluded in Python
    st_ind = np.zeros(dicsizes.size, dtype=np.int64)
    st_ind[1:] = end_ind[:-1]
    Nsubprob = np.prod(dicsizes)

    #  Compute all the combinations of atoms from each fascicle sub-dictionary
    # atom_indices = arrangements(dicsizes)  # too much memory used
    idx_range = tuple([np.arange(dicsizes[i])
                       for i in range(len(dicsizes))])

    # Prepare output
    w_nneg = np.zeros(N_LSC)
    ind_atoms_subdic = np.zeros(N_LSC, dtype=np.int64)
    y_sq = np.sum(y**2)
    min_obj = y_sq

    # Solve all subproblems. Note: do not create list of all index
    # combinations because way too memory expensive. Use itertools.product.
    if cMode:
        tmp = tempfile.mkdtemp()
        print("tmp: ", tmp)
        # Call a C function then extract results:
        np.savetxt(os.path.join(tmp, r'dicA.txt'), A, fmt='%f')
        np.savetxt(os.path.join(tmp, r'y.txt'), y, fmt='%f')

        time.sleep(1)

        # Delete
        csf = 1 in dicsizes
        # System call
        if csf and len(dicsizes)==3:
            cmd = f'cd {tmp}; neldermead 0.4 0.4 0.2 0.1 {dicsizes[0]} {len(y)}'
        elif len(dicsizes)==2:
            cmd = f'cd {tmp}; neldermead 0.5 0.5 0.1 {dicsizes[0]} {len(y)}'
        else:
            raise NotImplementedError(f"Fitting for dicsizes {dicsizes} not implemented.");
        #print(cmd)
        import subprocess
        result = subprocess.check_output(cmd, shell=True)

        # Load results
        w_nneg = np.loadtxt(os.path.join(tmp, r'w.txt'))
        sigma = w_nneg[-1]
        print("sigma: ", sigma)
        w_nneg = w_nneg[:-1]
        ind_atoms_subdic = np.loadtxt(os.path.join(tmp, r'ind.txt'), dtype=int)
        min_obj = np.loadtxt(os.path.join(tmp, r'obj.txt'))


        #w_nneg = w_nneg / np.sum(w_nneg)
    else:
        cnt = 0
        for idx in tqdm(product(*idx_range),desc="Solving subproblems",total=Nsubprob, mininterval=2, disable=not(verbose)):
            cnt += 1
            Asmall = A[:, st_ind + idx]
            print("Asmall.shape: ", Asmall.shape)
            print("stdind: ", st_ind)
            print("idx: ", idx)

            w, obj_fun = solve_one_comb(Asmall, y)

            if obj_fun < min_obj:
                w_nneg = w
                min_obj = obj_fun
                ind_atoms_subdic = np.atleast_1d(idx)

        assert Nsubprob == cnt, "Problem with number of subproblems solved"
    # absolute index within complete dictionary A
    ind_atoms_totdic = st_ind + ind_atoms_subdic
    # reconstructed data vector
    y_recons = np.dot(A[:, ind_atoms_totdic], w_nneg)
    return (w_nneg, ind_atoms_subdic, ind_atoms_totdic, min_obj, y_recons)




