import numpy as np
import sys
# from tqdm import tqdm
from scipy.io import readsav
# import matplotlib.pyplot as plt; plt.ion()
# from scipy.interpolate import BSpline, interp1d
# from scipy.optimize import nnls
import pickle
# from scipy.optimize import minimize
# from scipy.sparse.linalg import LinearOperator

from kascade.src import Bspline_functions, setup_funcs_synthetic_inversion, misc_functions, optimization_functions
from kascade.src import build_2D_kernel_edgeaware as build_2D_kernel
# import basis_funcs
# import make_batches_qr

# extracting the necessary fields from the datafiles
def extract_data_from_files(parkerdata, synthetic_data_By=None, synthetic_data_Bz=None):
    if(synthdata_By is not None):
        # and ensuring all quantities are in correct units
        S_theta_f = synthdata_By['S_TH_F_BV'].astype(np.float32)\
                    + synthdata_Bz['S_TH_F_BV'].astype(np.float32)                # saved in units of power spectral density (Ntheta x Nf)
        VSW_ = synthdata_Bz['VSW'].astype(np.float32)                             # saved in code units used, hence not multiplying by 1e3
        VA_ = synthdata_Bz['VA'].astype(np.float32)                               # saved in code units used, hence not multiplying by 1e3
        TH_BV_ = np.radians(synthdata_Bz['TH_deg'].astype(np.float32))            # saved in degrees
        F_ = synthdata_Bz['F_Hz'].astype(np.float32)                              # saved in Hz without the 2pi factor
        kperp = synthdata_Bz['kperp'].astype(np.float32)
        kpara = synthdata_Bz['kpara'].astype(np.float32)
    else:
        S_theta_f = parkerdata['S_TH_F_BV'].astype(np.float32)
        VSW_ = parkerdata['VSW'].astype(np.float32)                             # saved in code units used, hence not multiplying by 1e3
        VA_ = parkerdata['VA'].astype(np.float32)                               # saved in code units used, hence not multiplying by 1e3
        TH_BV_ = np.radians(parkerdata['TH_deg'].astype(np.float32))            # saved in degrees
        F_ = parkerdata['F_Hz'].astype(np.float32)                              # saved in Hz without the 2pi factor
        kperp = parkerdata['kperp'].astype(np.float32)
        kpara = parkerdata['kpara'].astype(np.float32)

    return S_theta_f, VSW_, VA_, TH_BV_, F_, kperp, kpara


if __name__=="__main__":
    #---------------------------- loading the datafiles ------------------------------#
    # datafile containing the synthetic Bz power spectra generated from generate_S_f_theta.py
    data_file_path_Bz = 'datafiles/synthetic_data_Bz.pkl'
    # datafile containing the synthetic By power spectra generated from generate_S_f_theta.py
    data_file_path_By = 'datafiles/synthetic_data_By.pkl'
    # datafile containing Parker data from Trevor, used to ensure we have similar freq and theta from synthetics
    trev_file_path = 'datafiles/e22_data_4_srijan.sav'
    trevdata = readsav(trev_file_path)

    with open(data_file_path_By, 'rb') as file:
        synthdata_By = pickle.load(file)

    with open(data_file_path_Bz, 'rb') as file:
        synthdata_Bz = pickle.load(file)

    #------------------------- obtaining the data from the loaded files ----------------#
    S_theta_f, VSW_, VA_, TH_BV_, F_, kperp, kpara = extract_data_from_files(trevdata, synthdata_By, synthdata_Bz)

    #-------- generating the frequency and wavenumber grid for synthetic case ----------#
    fmin, fmax, f0 = 0.01, 2, 0.248 # !!!!check if factor of 2pi is missing!!!!
    F_reduced = setup_funcs_synthetic_inversion.get_frequency_for_synthetic_inversion(fmin, fmax, f0, n1=85, n2=86, n3=85)

    # generating the interpolated data onto the chosen frequency grid
    S_theta_reduced_f = np.zeros((len(TH_BV_), len(F_reduced)), dtype=np.float32)

    for i in range(len(TH_BV_)):
        S_theta_reduced_f[i] = misc_functions.interpolating_Spec_in_freq(S_theta_f[i], F_, F_reduced)

    F_ = F_reduced * 1.0

    # reshaping to (Nf x Ntheta) and then flattening
    d = np.reshape(S_theta_reduced_f, (-1), 'C')

    # generating the wavenumber k-grids (in log space since that is where we perform the inversion)
    # making the k_grid dictionary
    log_k_grids = {}
    log_k_grids['log10_kperp'], log_k_grids['log10_kpara'] = np.log10(kperp), np.log10(kpara)
    kperpkperp, kparakpara = np.meshgrid(kperp, kpara, indexing='ij')

    #------------------------------ making the B-spline basis --------------------------#
    # making the control points
    NCTRL_KPERP, NCTRL_KPARA = 70, 70
    ctrl_kperp = np.linspace(log_k_grids['log10_kperp'].min(), log_k_grids['log10_kperp'].max(), NCTRL_KPERP)
    ctrl_kpara = np.linspace(log_k_grids['log10_kpara'].min(), log_k_grids['log10_kpara'].max(), NCTRL_KPARA)

    # making a list from the ctrl points in two dimensions
    ctrl_points = [ctrl_kperp, ctrl_kpara]

    # making in the basis functions
    basis_dict = {}
    basis_dict['basis_type'] = 'bspline'
    basis_dict['p'] = 3
    # Phi = basis_funcs.make_basis_functions(ctrl_points, log_k_grids, basis_dict)
    Phi, LTL = Bspline_functions.get_Bsplines(ctrl_points, 3, log_k_grids)
    Nbasis = Phi.shape[0]

    #------------------- STEP1: Non-negative least square inversion ---------------------#
    # building the complete 2D kernel for all data
    K, G = build_2D_kernel.build(kperp, kpara, VSW_, TH_BV_, VA_, F_, Phi, P2D=None)
    G = np.reshape(G, (-1, Nbasis), 'C')
    # d = np.reshape(K, (-1), 'C')

    # performing a linear inversion
    eps_d = (np.nanmin(d) / np.nanmax(d)) * np.nanmax(d)
    w = 1.0 / np.maximum(d, eps_d)

    G_og = G * 1.0
    d_og = d * 1.0

    G = G * w[:, None]
    d = d * w

    GTG = G.T @ G
    GTd = G.T @ d

    # using a regularization parameter \lambda (L-curve method)
    lambda_arr = np.logspace(-6, 4, 27)
    # D = np.eye(GTG.shape[0])
    D = LTL.toarray()

    # sys.exit()
    # performing the non-linear least square inversion
    Q2D_rec, P2D_rec = optimization_functions.nnls_inversion(GTG, GTd, D, Phi, kperpkperp, kparakpara,plot=True)