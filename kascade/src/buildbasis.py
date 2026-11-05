from scipy.interpolate import BSpline
from scipy.sparse import kron
import json
import numpy as np
import matplotlib.pyplot as plt; plt.ion()

from kascade.src.buildclass import kascade

def add_basis_to_kascade(ksc_obj, config_path):
    # reading the config file defined by the user to store the basis parameters
    with open(config_path, 'r') as f:
        config = json.load(f)

    # creating a dictionary of basis parameters from the config file
    basis_params = config.get("basis_params", {})
    basis_dict = {key: value for key, value in basis_params.items()}

    ksc_obj.basis = {}
    ksc_obj.basis['basis_dict'] = basis_dict

    if(basis_dict['basis_type'] == 'bspline'):
        make_bspline_basis(ksc_obj)

def make_bspline_basis(ksc_obj):
    # making the control points
    ctrl_kperp = np.linspace(np.log10(ksc_obj.kperp).min(),
                             np.log10(ksc_obj.kperp).max(),
                             ksc_obj.basis['basis_dict']['nctrl_kperp'])
    ctrl_kpara = np.linspace(np.log10(ksc_obj.kpara).min(),
                             np.log10(ksc_obj.kpara).max(),
                             ksc_obj.basis['basis_dict']['nctrl_kpara'])

    # making a list from the ctrl points in two dimensions
    ksc_obj.basis['ctrl_points'] = {}
    ksc_obj.basis['ctrl_points']['kperp'] = ctrl_kperp
    ksc_obj.basis['ctrl_points']['kpara'] = ctrl_kpara

    # storing the basis functions and the LTL matrix in the ksc_obj
    ksc_obj.basis['basis_functions'], ksc_obj.basis['LTL'] = get_Bsplines(ksc_obj)
    ksc_obj.basis['nbasis'] = ksc_obj.basis['basis_functions'].shape[0]

def get_Bsplines(ksc_obj):
    log10_kperp = np.log10(ksc_obj.kperp)
    log10_kpara = np.log10(ksc_obj.kpara)
    p = ksc_obj.basis['basis_dict']['p']

    def get_Bsplines_1D(knots, r_grid):
        # Create an open (clamped) knot vector
        t = np.concatenate((
            np.repeat(knots[0], p + 1),
            knots[1:-1],
            np.repeat(knots[-1], p + 1)
        ))

        n_basis = len(t) - (p + 1)
        coeffs = np.eye(n_basis)

        spl = BSpline(t, coeffs, p, extrapolate=False)
        B_i_n = np.nan_to_num(spl(r_grid).T)
        Bder_i_n = np.nan_to_num(spl.derivative(2)(r_grid).T)
        return B_i_n, Bder_i_n

    B_i_kperp, Bder_i_kperp = get_Bsplines_1D(ksc_obj.basis['ctrl_points']['kperp'],
                                              log10_kperp)
    B_j_kpara, Bder_j_kpara = get_Bsplines_1D(ksc_obj.basis['ctrl_points']['kpara'],
                                              log10_kpara)

    # making the 2D B-splines grid
    B_ij_kperp_kpara = B_i_kperp[:,None,:,None] * B_j_kpara[None,:,None,:]

    # reshaping to make them of the shape (n_ctrl_kperp * n_ctrl_kpara, kperp, kpara)
    B_alpha_kperp_kpara = np.reshape(B_ij_kperp_kpara, (-1, len(log10_kperp), len(log10_kpara)), 'C')
    LTL = get_L_matrix(B_i_kperp, Bder_i_kperp, B_j_kpara, Bder_j_kpara, log10_kperp, log10_kpara)

    return B_alpha_kperp_kpara, LTL

def get_L_matrix(B_i_kperp, Bder_i_kperp, B_j_kpara, Bder_j_kpara, log10_kperp, log10_kpara):
    dkperp = np.mean(np.diff(log10_kperp))
    dkpara = np.mean(np.diff(log10_kpara))

    w_perp = np.full(B_i_kperp.shape[1], dkperp)
    w_para = np.full(B_j_kpara.shape[1], dkpara)

    # "mass" and "curvature" matrices in each dimension
    S_perp = B_i_kperp @ np.diag(w_perp) @ B_i_kperp.T
    S_perp_dd = Bder_i_kperp @ np.diag(w_perp) @ Bder_i_kperp.T

    S_para = B_j_kpara @ np.diag(w_para) @ B_j_kpara.T
    S_para_dd = Bder_j_kpara @ np.diag(w_para) @ Bder_j_kpara.T

    # L^T L equivalent
    RegMatrix = kron(S_para, S_perp_dd) + kron(S_para_dd, S_perp)
    return RegMatrix


if __name__ == "__main__":
    # file containing the data
    filepath = '/Users/sbdas/Documents/Research/SpaceWeather/Turbulence/kascade/datafiles/Jul162026_psd_arr_enc22_4srijan.sav'
    # Loading the JSON file with key mapping
    config_path = 'config.json'

    # creating an instance of the kascade class
    ksc = kascade(filepath, config_path)

    # now adding on the basis functions to the kascade class instance
    add_basis_to_kascade(ksc, config_path)

    # plotting four selected basis functions for sanity check
    # 1. Randomly pick 4 unique indices from the 484 available maps
    random_indices = np.random.choice(ksc.basis['basis_functions'].shape[0], size=4, replace=False)

    # 2. Set up a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 3. Flatten the axes array to easily loop through it alongside your indices
    for ax, idx in zip(axes.flat, random_indices):
        # Plot the 100x100 map using imshow
        im = ax.pcolormesh(ksc.basis['basis_functions'][idx], cmap="inferno", vmin=0, vmax=0.5)

        # Add a title indicating which map index was chosen
        ax.set_title(f"Basis Index: {idx}")
        ax.set_aspect('equal')  # Ensure the aspect ratio is equal for better visualization
        ax.set_xlabel("kpara index")
        ax.set_ylabel("kperp index")

        # Optional: Add a colorbar to each subplot
        fig.colorbar(im, ax=ax, shrink=0.7)
    
    plt.tight_layout()
