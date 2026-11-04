import numpy as np
from tqdm import tqdm

LOG10 = np.log(10.0)


def _reconstruct_edges_from_centers(centers):
    """
    Reconstruct positive monotonic bin edges from positive monotonic centers
    using geometric means in the interior and symmetric extrapolation at ends.
    """
    centers = np.asarray(centers)
    if np.any(centers <= 0):
        raise ValueError("All centers must be > 0 to reconstruct log-space edges.")
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be a 1D array with length >= 2")

    edges = np.empty(centers.size + 1, dtype=centers.dtype)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0]**2 / edges[1]
    edges[-1] = centers[-1]**2 / edges[-2]
    return edges


def _make_log_midpoints_from_edges(edges, nsub, dtype=np.float32):
    """
    For each bin [edge_i, edge_{i+1}], build nsub midpoint quadrature points
    uniformly in log10-space.

    Returns
    -------
    subpts : (Nbin, nsub)
        Subcell quadrature points in linear k-space.
    dlog   : (Nbin,)
        Bin widths in log10-space.
    """
    edges = np.asarray(edges, dtype=np.float64)
    loge = np.log10(edges)

    left = loge[:-1]
    right = loge[1:]
    dlog = right - left

    frac = (np.arange(nsub, dtype=np.float64) + 0.5) / nsub
    log_sub = left[:, None] + dlog[:, None] * frac[None, :]
    subpts = np.power(10.0, log_sub)

    return subpts.astype(dtype), dlog.astype(dtype)


def _build_binint_weight_block(
    kperp, kpar, f, Vsw_theta, theta, VA,
    batch_theta=64, dtype=np.float32,
    nsub_perp=2, nsub_par=2
):
    """
    Build cell-integrated kernel weights Wcell with midpoint quadrature
    inside each (kperp, kpar) cell in log10-space.

    Returns
    -------
    Wcell : (Nθ, Nf, Nk, Nl)
        Cell-integrated forward weights for Q(k_perp,k_par), where
        Q = P2D * kperp^2 * kpar is treated as piecewise constant per cell.
    """
    # cast inputs
    kperp = np.asarray(kperp, dtype=dtype)
    kpar = np.asarray(kpar, dtype=dtype)
    f = np.asarray(f, dtype=dtype)
    Vsw_theta = np.asarray(Vsw_theta, dtype=dtype)
    theta = np.asarray(theta, dtype=dtype)
    VA = np.asarray(VA, dtype=dtype)

    Nk, Nl = kperp.size, kpar.size
    Nθ, Nf = theta.size, f.size

    # reconstruct edges from centers
    kperp_edges = _reconstruct_edges_from_centers(kperp.astype(np.float64))
    kpar_edges = _reconstruct_edges_from_centers(kpar.astype(np.float64))

    # subcell midpoint quadrature points in each bin
    kperp_sub, dlogk = _make_log_midpoints_from_edges(kperp_edges, nsub_perp, dtype=dtype)
    kpar_sub, dlogl = _make_log_midpoints_from_edges(kpar_edges, nsub_par, dtype=dtype)

    # weight per subcell in dlog10(kperp) dlog10(kpar)
    # current formulation integrates Q against kernel in log-space
    w_sub = (LOG10**2) * (dlogk[:, None] / nsub_perp) * (dlogl[None, :] / nsub_par)
    w_sub = w_sub.astype(dtype)   # (Nk, Nl)

    # precompute constants
    S = Vsw_theta * np.sin(theta)                # (Nθ,)
    C = Vsw_theta * np.cos(theta) + VA           # (Nθ,)
    TWO_PI_f = (2.0 * np.pi * f).astype(dtype)  # (Nf,)

    # output
    Wcell = np.zeros((Nθ, Nf, Nk, Nl), dtype=dtype)

    for t0 in tqdm(range(0, Nθ, batch_theta), desc="Building kernel"):
        t1 = min(t0 + batch_theta, Nθ)
        B = t1 - t0

        Sb = S[t0:t1][:, None, None, None]      # (B,1,1,1)
        Cb = C[t0:t1][:, None, None, None]      # (B,1,1,1)

        Wsum = np.zeros((B, Nf, Nk, Nl), dtype=dtype)

        # midpoint quadrature over each cell
        for ip in range(nsub_perp):
            kp = kperp_sub[:, ip][None, None, :, None]   # (1,1,Nk,1)

            for jp in range(nsub_par):
                kl = kpar_sub[:, jp][None, None, None, :]   # (1,1,1,Nl)

                # scalar cell weights for this subpoint
                num = 2.0 * w_sub[None, None, :, :]        # (1,1,Nk,Nl)

                a = (kp * Sb) ** 2

                # branch 1
                b1 = (TWO_PI_f[None, :, None, None] - kl * Cb) ** 2

                rad1 = a - b1

                mask1 = rad1 > 0

                denom1 = np.ones_like(rad1, dtype=dtype)
                np.sqrt(rad1, out=denom1, where=mask1)

                Wtmp1 = np.zeros_like(rad1, dtype=dtype)
                np.divide(num, denom1, out=Wtmp1, where=mask1)
                Wtmp1[~mask1] = 0.0

                # branch 2
                b2 = (TWO_PI_f[None, :, None, None] + kl * Cb) ** 2

                rad2 = a - b2

                mask2 = rad2 > 0

                denom2 = np.ones_like(rad2, dtype=dtype)
                np.sqrt(rad2, out=denom2, where=mask2)

                Wtmp2 = np.zeros_like(rad2, dtype=dtype)
                np.divide(num, denom2, out=Wtmp2, where=mask2)
                Wtmp2[~mask2] = 0.0


                Wsum += Wtmp1 + Wtmp2

        Wcell[t0:t1, :, :, :] = Wsum

    return Wcell


def build(
    kperp, kpar,                    # (Nk,), (Nl,)
    Vsw_theta, theta, VA,           # (Nθ,), (Nθ,), scalar or (Nθ,)
    f,                              # (Nf,)
    Phi,                            # (Nbasis, Nk, Nl)
    P2D=None,                       # if passed, interpreted on cell centers
    batch_theta=64,
    dtype=np.float32,
    nsub_perp=2,
    nsub_par=2
):
    """
    Bin-integrated replacement for the old center-sampled build().

    Returns
    -------
    K : (Nθ, Nf)
        Forward-model spectrum.
    G : (Nθ, Nf, Nbasis)
        Basis-projected kernel.
    """
    kperp = np.asarray(kperp, dtype=dtype)
    kpar = np.asarray(kpar, dtype=dtype)
    Phi = np.asarray(Phi, dtype=dtype)

    Nk, Nl = kperp.size, kpar.size
    Nbasis = Phi.shape[0]

    KPERP, KPAR = np.meshgrid(kperp, kpar, indexing='ij')

    # same Q-variable convention as your current code:
    # Q = P2D * kperp^2 * kpar
    if P2D is None:
        Q2D = (KPERP**(-5.0/3.0)) * (KPAR**(-2.0)) * (KPERP**2) * KPAR
    else:
        P2D = np.asarray(P2D, dtype=dtype)
        Q2D = P2D * (KPERP**2) * KPAR

    Wcell = _build_binint_weight_block(
        kperp=kperp,
        kpar=kpar,
        f=f,
        Vsw_theta=Vsw_theta,
        theta=theta,
        VA=VA,
        batch_theta=batch_theta,
        dtype=dtype,
        nsub_perp=nsub_perp,
        nsub_par=nsub_par
    )

    Nθ = np.asarray(theta).size
    Nf = np.asarray(f).size

    K = np.empty((Nθ, Nf), dtype=dtype)
    G = np.empty((Nθ, Nf, Nbasis), dtype=dtype)

    # same batched contractions as before
    for t0 in range(0, Nθ, batch_theta):
        t1 = min(t0 + batch_theta, Nθ)
        Wb = Wcell[t0:t1]   # (B,Nf,Nk,Nl)

        Z = np.einsum('bfkl,akl->bfa', Wb, Phi, optimize=True)
        KTEMP = np.einsum('bfkl,kl->bf', Wb, Q2D, optimize=True)

        K[t0:t1, :] = KTEMP
        G[t0:t1, :] = Z

    return K, G


def build_K2D(
    kperp, kpar,
    Vsw_theta, theta, VA,
    f,
    batch_theta=64,
    dtype=np.float32,
    nsub_perp=2,
    nsub_par=2
):
    """
    Bin-integrated replacement for the old center-sampled build_K2D().

    Returns
    -------
    A : (Nθ*Nf, Nk*Nl)
        Cellwise forward operator acting on Q = P2D * kperp^2 * kpar.
    """
    Wcell = _build_binint_weight_block(
        kperp=kperp,
        kpar=kpar,
        f=f,
        Vsw_theta=Vsw_theta,
        theta=theta,
        VA=VA,
        batch_theta=batch_theta,
        dtype=dtype,
        nsub_perp=nsub_perp,
        nsub_par=nsub_par
    )

    Nθ, Nf, Nk, Nl = Wcell.shape
    A = np.reshape(Wcell, (Nθ, Nf, Nk * Nl), order='C')
    A = np.reshape(A, (Nθ * Nf, Nk * Nl), order='C')
    return A