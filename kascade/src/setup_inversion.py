import numpy as np
from scipy.ndimage import gaussian_filter1d

import build_2D_kernel_edgeaware_bothbranch as build_2D_kernel

def generate_inversion_components(ksc, P2D=None, floor_frac=1e-12):
    # creating the dictionary attribute for performing the inversion
    ksc.inversion = {}

    # building the complete 2D kernel for all data
    K, G = build_2D_kernel.build(ksc.kperp, ksc.kpara, ksc.VSW, ksc.TH_BV, ksc.VA, ksc.F,
                                 ksc.basis['basis_functions'], P2D=P2D)

    if(ksc.d is None):
        ksc.d = np.reshape(K, (-1), 'C')

    # storing the unweighted G matrix
    G = np.reshape(G, (-1, ksc.basis['nbasis']), 'C')

    # performing a linear inversion
    eps_d = floor_frac * np.nanmax(ksc.d)
    w = 1.0 / np.maximum(ksc.d, eps_d)

    # storing the weights
    ksc.inversion['data_weights'] = w

    # scaling with the weights
    Gw = G * ksc.inversion['data_weights'][:, None]
    dw = ksc.d * ksc.inversion['data_weights']

    ksc.inversion['Gw'] = Gw
    ksc.inversion['dw'] = dw

    # storing the contracted matrices for inversion
    ksc.inversion['Gw_T_Gw'] = Gw.T @ Gw
    ksc.inversion['Gw_T_dw'] = Gw.T @ dw

    # using a regularization parameter \lambda (L-curve method)
    ksc.inversion['lambda_arr'] = np.logspace(-6, 4, 27)
    # D = np.eye(GTG.shape[0])
    ksc.inversion['D'] = ksc.basis['LTL'].toarray()

def generate_sigma_dex(ksc):
    # ----------------------------
    # Locally-estimated sigma_dex
    # ----------------------------
    log_d_2d = np.log10(np.maximum(ksc.P_theta_f, ksc.dfloor))
    smooth_log_d_2d = gaussian_filter1d(log_d_2d, sigma=2.0, axis=1, mode='nearest')
    local_sigma_2d = gaussian_filter1d(np.abs(log_d_2d - smooth_log_d_2d), sigma=4.0, axis=1, mode='nearest')
    local_sigma_2d = np.maximum(local_sigma_2d, 0.05)
    ksc.inversion['sigma_dex'] = np.reshape(local_sigma_2d, (-1), 'C')[ksc.inversion['good']].astype(np.float64)