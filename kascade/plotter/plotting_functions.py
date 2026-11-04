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
    
def plot_admissible_region(kperp, kpara, Vsw, theta, Va, f, title=""):
    mask, rad = admissible_mask(kperp, kpara, Vsw, theta, Va, f)
    KPERP, KPAR = np.meshgrid(kperp, kpara, indexing='ij')

    plt.figure(figsize=(6,5))
    plt.contourf(np.log10(KPERP), np.log10(KPAR), mask.astype(float), levels=[-0.1,0.5,1.1])
    plt.xlabel(r'$\log_{10} k_\perp$')
    plt.ylabel(r'$\log_{10} k_\parallel$')
    plt.title(title + f"\nθ={np.degrees(theta):.2f} deg, f={f:.3e}")
    plt.tight_layout()

    frac = mask.mean()
    print(f"{title}: admissible fraction = {frac:.4f}")
    return mask, rad
