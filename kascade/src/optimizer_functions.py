import numpy as np
from tqdm import tqdm
from scipy.linalg import solve
from scipy.optimize import minimize

from kascade.src import plotter, setup_inversion, misc_funcs

def geometric_knee(x, y):
    """
    Finding the optimal point using a geometric knee detection algorithm of a trade-off curve.
    x and y are the respective axes values in the trade-off curve. It is expected one of them
    is the model-misfit and the other is the data-misfit.

    Returns
    -------
    Returns the index corresponding to the knee of the 1D L-curve.

    """
    x, y = np.array(x), np.array(y)
    # Line: from first to last point
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    point_vecs = np.stack([x - x[0], y - y[0]], axis=1)
    proj_lens = point_vecs @ line_vec_norm
    proj_points = np.outer(proj_lens, line_vec_norm) + np.array([x[0], y[0]])
    distances = np.linalg.norm(point_vecs - (proj_points - np.array([x[0], y[0]])), axis=1)
    return np.argmax(distances)

def solve_nonnegative_quadratic_fista(H, g, x0=None, max_iter=2000, tol=1e-12,
                                      min_iter=2000, verbose=False):
    n = g.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.maximum(np.asarray(x0, dtype=np.float64), 0.0).copy()

    y = x.copy()
    t = 1.0

    L = np.linalg.eigvalsh(H).max()
    if L <= 0:
        L = 1.0

    f_prev = np.inf
    best_x = x.copy()
    best_f = np.inf

    for k in range(max_iter):
        grad = H @ y - g
        x_new = np.maximum(y - grad / L, 0.0)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)

        f = 0.5 * x_new @ (H @ x_new) - g @ x_new
        rel_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1.0)

        if f < best_f:
            best_f = f
            best_x = x_new.copy()

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"iter {k:04d}  obj={f:.6e}  rel_change={rel_change:.3e}")

        if k >= min_iter:
            if abs(f_prev - f) / max(abs(f_prev), 1.0) < tol and rel_change < tol:
                break

        x = x_new
        t = t_new
        f_prev = f

    return best_x

def drive_fistasolver(ksc, plot=False):
    m_solutions, data_misfit, model_misfit = {}, [], []
    m_prev = None
    for lam in tqdm(ksc.inversion['lambda_arr'][::-1]):                      # descending order
        H = ksc.inversion['Gw_T_Gw'] + lam * ksc.inversion['D']
        x0 = np.maximum(solve(H, ksc.inversion['Gw_T_dw'], assume_a='sym'), 0.0) if m_prev is None else m_prev
        m = solve_nonnegative_quadratic_fista(H, ksc.inversion['Gw_T_dw'], x0=x0, max_iter=10000, min_iter=2000, tol=1e-12)
        m_prev = m
        m_solutions[lam] = m

    data_misfit  = np.array([np.linalg.norm(ksc.inversion['Gw'] @ m_solutions[lam] - ksc.inversion['dw'])**2 for lam in ksc.inversion['lambda_arr']])
    model_misfit = np.array([m_solutions[lam] @ ksc.inversion['D'] @ m_solutions[lam] for lam in ksc.inversion['lambda_arr']])

    data_misfit = (data_misfit - np.nanmin(data_misfit)) / (np.nanmax(data_misfit) - np.nanmin(data_misfit))
    model_misfit = (model_misfit - np.nanmin(model_misfit)) / (np.nanmax(model_misfit) - np.nanmin(model_misfit))

    lam_knee_idx = geometric_knee(data_misfit, model_misfit)
    # plotting the L-curve
    plotter.plot_Lcurve_knee(model_misfit, data_misfit,
                    lam_knee_idx, ksc.inversion['lambda_arr'][lam_knee_idx])
    ksc.m_fista = m_solutions[ksc.inversion['lambda_arr'][lam_knee_idx]]

    # plotting the m_fista result based on plot flag
    if(plot):
        plotter.plot_2D_spectra(ksc.m_fista, ksc)


def get_lambda_for_nonlin(ksc):
    # ---------------------------------------------------------------
    # Data prep: drop non-positive/invalid points, log10
    # ---------------------------------------------------------------
    ksc.dfloor = ksc.dfloor_frac * np.nanmax(ksc.d[np.isfinite(ksc.d) & (ksc.d > 0)])
    ksc.inversion['good'] = np.isfinite(ksc.d) & (ksc.d > ksc.dfloor)
    print(f"dropping {np.sum(~ksc.inversion['good'])} / {ksc.d.size} non-positive/invalid points before log-fit")

    G = ksc.inversion['Gw'] / ksc.inversion['data_weights'][:, None]
    G_32 = G[ksc.inversion['good']].astype(np.float32)   # float32 copy, only for the fast subsampled sweep
    logd_32 = np.log10(ksc.d[ksc.inversion['good']]).astype(np.float32)
    D_32 = ksc.inversion['D'].astype(np.float32)
    m_fista_32 = np.maximum(ksc.m_fista.astype(np.float32), 0.0)

    # ---------------------------------------------------------------
    # L-curve sweep on a SUBSAMPLE of the data (fast, just to find lambda)
    # ---------------------------------------------------------------
    rng = np.random.default_rng(0)
    sub = rng.choice(len(logd_32), size=len(logd_32) // 32, replace=False)
    G_sub, logd_sub = G_32[sub], logd_32[sub]

    lambda_arr_log = ksc.inversion['lambda_arr']
    m_solutions_log = {}
    m_prev = None
    for lam in tqdm(lambda_arr_log[::-1]):
        m_start = m_prev if m_prev is not None else m_fista_32
        m_lam, _ = solve_log10_lbfgsb(G_sub, logd_sub, D_32, m_start, lam=lam, sigma_dex=1.0, maxiter=200)
        m_prev = m_lam
        m_solutions_log[lam] = m_lam

    data_misfit_log = np.array([
        np.sum((np.log10(np.maximum(G_sub @ m_solutions_log[lam], 1e-30)) - logd_sub)**2)
        for lam in lambda_arr_log
    ])
    model_misfit_log = np.array([m_solutions_log[lam] @ D_32 @ m_solutions_log[lam] for lam in lambda_arr_log])

    data_n = (data_misfit_log - data_misfit_log.min()) / (data_misfit_log.max() - data_misfit_log.min())
    model_n = (model_misfit_log - model_misfit_log.min()) / (model_misfit_log.max() - model_misfit_log.min())
    knee_idx_log = geometric_knee(data_n, model_n)
    plotter.plot_Lcurve_knee(model_n, data_n, knee_idx_log, lambda_arr_log[knee_idx_log])

    ksc.inversion['lam_log_best'] = lambda_arr_log[knee_idx_log]
    print(f"chosen lambda (log-space) = {ksc.inversion['lam_log_best']:.3e}")

def solve_log10_lbfgsb(G, logd_, D_, m_init, lam, sigma_dex=1.0, maxiter=200):
    def log10_obj_grad_scipy(m_, G, logd_, D_, lam, sigma_dex=1.0, floor=1e-30):
        pred = np.maximum(G @ m_, floor)
        r = (np.log10(pred) - logd_) / sigma_dex
        f_data = 0.5 * np.dot(r, r)
        Dm = D_ @ m_
        f_reg = 0.5 * lam * np.dot(m_, Dm)
        factor = r / (sigma_dex * np.log(10.0) * pred)
        g = G.T @ factor + lam * Dm
        return f_data + f_reg, g

    bounds = [(0, None)] * len(m_init)
    res = minimize(
        log10_obj_grad_scipy, m_init, args=(G, logd_, D_, lam, sigma_dex),
        method='L-BFGS-B', jac=True, bounds=bounds,
        options=dict(maxiter=maxiter, maxfun=maxiter * 3, ftol=1e-12, gtol=1e-10)
    )
    return res.x.astype(np.float32), res

def solve_log10_projected_adam_ckpt(G, logd_, D_, m_init, lam,
                                    sigma_dex_=1.0, max_iter=3000, lr=1e-1,
                                    beta1=0.9, beta2=0.999, eps=1e-8,
                                    clip_norm=None, tol=1e-10, ckpt_every=200,
                                    ckpt_name="log_fit_adam_final", verbose=True):
    m_ = np.maximum(np.asarray(m_init, dtype=np.float64).copy(), 0.0)
    mm = np.zeros_like(m_)
    vv = np.zeros_like(m_)
    f_prev = np.inf
    history = []

    for t in range(1, max_iter + 1):
        f, g = log10_obj_grad_fullbatch(m_, G, logd_, D_, lam, sigma_dex_=sigma_dex_)

        if clip_norm is not None:
            gnorm = np.linalg.norm(g)
            if gnorm > clip_norm:
                g = g * (clip_norm / gnorm)

        mm = beta1 * mm + (1 - beta1) * g
        vv = beta2 * vv + (1 - beta2) * (g * g)
        mhat, vhat = mm / (1 - beta1**t), vv / (1 - beta2**t)
        m_ = np.maximum(m_ - lr * mhat / (np.sqrt(vhat) + eps), 0.0)

        history.append(f)
        if t % 50 == 0 or t == 1:
            print(f"iter {t:04d}  obj={f:.6e}")
        if t % ckpt_every == 0:
            misc_funcs.save_ckpt(ckpt_name, m=m_, iter=t, obj=f)

        if abs(f_prev - f) / max(abs(f_prev), 1.0) < tol:
            print(f"converged at iter {t}")
            break
        f_prev = f

    misc_funcs.save_ckpt(ckpt_name, m=m_, iter=t, obj=f)
    return m_, history

def log10_obj_grad_fullbatch(m_, G, logd_, D_, lam, sigma_dex_=1.0, floor=1e-30):
    pred = np.maximum(G @ m_, floor)
    r = (np.log10(pred) - logd_) / sigma_dex_

    f_data = 0.5 * np.dot(r, r)
    Dm = D_ @ m_
    f_reg = 0.5 * lam * np.dot(m_, Dm)

    factor = r / (sigma_dex_ * np.log(10.0) * pred)
    g = G.T @ factor + lam * Dm
    return f_data + f_reg, g


def solve_log10_adam(ksc):
    # obtaining the sigma_dex first from data
    setup_inversion.generate_sigma_dex(ksc)

    # ---------------------------------------------------------------
    # Final full-data solve
    # ---------------------------------------------------------------
    G = ksc.inversion['Gw'] / ksc.inversion['data_weights'][:, None]

    G = G[ksc.inversion['good']].astype(np.float64)
    m_init = np.maximum(ksc.m_fista.astype(np.float64), 0.0)
    logd = np.log10(ksc.d[ksc.inversion['good']]).astype(np.float64)
    
    m_log_final, hist_final = solve_log10_projected_adam_ckpt(
        G, logd, ksc.inversion['D'], m_init, lam=ksc.inversion['lam_log_best'],
        sigma_dex_=ksc.inversion['sigma_dex'], max_iter=3000, lr=1e-1,
        clip_norm=1e4, ckpt_every=200, ckpt_name="log_fit_adam_final"
        )
    
    ksc.inversion['m_adam'] = m_log_final

    # plotting the final result
    plotter.plot_2D_spectra(ksc.inversion['m_adam'], ksc)
    plotter.compare_spectra(ksc)