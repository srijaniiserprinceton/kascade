import numpy as np
import matplotlib.pyplot as plt; plt.ion()

def plot_Lcurve_knee(model_misfit, data_misfit, knee_idx, mu, ext='png', SAVE=False):
    fig = plt.figure()
    plt.plot(model_misfit, data_misfit, 'b')
    plt.plot(model_misfit, data_misfit, 'or')
    plt.plot(model_misfit[knee_idx], data_misfit[knee_idx], 'xk', markersize=14)
    plt.gca().text(0.95, 0.95, r'$\mu = $' + f'{mu:.2e}',
                   bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'),
                   transform=plt.gca().transAxes, ha='right', va='top')
    plt.grid(True)
    plt.xlabel('Model Misfit', fontsize=14, fontweight='bold')
    plt.ylabel('Data Misfit', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if(SAVE):
        plt.savefig(f'./Figures/kneeL_hybrid/kneeL_{tidx}.{ext}')
        plt.close(fig)

def plot_2D_spectra(m, ksc):
    # making the meshgrid to plot
    kperpkperp, kparakpara = np.meshgrid(ksc.kperp, ksc.kpara, indexing='ij')

    # constructing the 2D profile (first normalized and then un-normalized)
    Q2D_rec = np.sum(ksc.basis['basis_functions'] * m[:,None,None], axis=0)
    P2D_rec = Q2D_rec / ((kperpkperp**2) * kparakpara)
    # P2D = (kperpkperp**(-5/3)) * (kparakpara**(-2))

    # plotting the true and reconstructed profiles
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    # levels = np.linspace(np.log10(1e-8), np.log10(1e2), 15)
    vmin = -1
    vmax = np.nanmax(np.log10(P2D_rec))
    levels = np.linspace(vmin, vmax, 16)
    # im=ax.contourf(np.log10(kperpkperp/(2*np.pi)), np.log10(kparakpara/(2*np.pi)), np.log10(P2D_rec), levels=levels, cmap='nipy_spectral') 
    im=ax.contourf(np.log10(kperpkperp*ksc.rhoi), np.log10(kparakpara*ksc.di), np.log10(P2D_rec), levels=levels, cmap='nipy_spectral') 
    ax.set_title('Reconstructed P(k)')
    fig.colorbar(im, ax=ax)
    ax.set_aspect('equal')
    ax.set_ylabel(r'$k_{\parallel} \, d_i$')
    ax.set_xlabel(r'$k_{\perp} \, \rho_i$')
    ax.axvline(0.0, ls='dashed', color='k')
    ax.axhline(0.0, ls='dashed', color='k')

def compare_spectra(ksc):
    G = ksc.inversion['Gw'] / ksc.inversion['data_weights'][:, None]
    dmodel_log = np.reshape(G @ ksc.inversion['m_adam'], (-1, len(ksc.F)), 'C')
    dtrue = ksc.P_theta_f * 1.0

    fig, ax_ = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    for i, idx in enumerate(np.random.randint(0, len(dtrue), 9)):
        ax_[i//3, i%3].loglog(ksc.F, dmodel_log[idx], label='Forward-modeled S(f) [log-fit]', color='blue')
        ax_[i//3, i%3].loglog(ksc.F, dtrue[idx], '.', label='Trajectory-based S(f)', color='green')
        ax_[i//3, i%3].set_title(f"TH_BV = {np.degrees(ksc.TH_BV[idx]):.2f} degrees")
        ax_[i//3, i%3].set_xlabel("Frequency (Hz)")
        ax_[i//3, i%3].set_ylabel("S(f)")
        ax_[i//3, i%3].legend()
    plt.tight_layout()