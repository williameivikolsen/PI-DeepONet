import numpy as np
import os

X = 10.0          # Slab length (cm)
J = 100           # Number of spatial cells
h = X / J         # Cell width (cm)

# Cell-centre coordinates  x_j = (j - 0.5) * h,  j = 1..J  (0-indexed: j=0..J-1)
x_centres = (np.arange(J) + 0.5) * h   # shape (J,)

# Gauss-Legendre quadrature over [-1, 1] with N=16 points
N_ANGLES = 16
mu_GL, w_GL = np.polynomial.legendre.leggauss(N_ANGLES)        # mu in [-1,1], weights sum to 2

# Split into positive (left→right) and negative (right→left) hemispheres
# The paper uses all N directions; positive μ: sweep L→R, negative μ: sweep R→L
# Ordering: first N/2 negative, last N/2 positive (standard GL ordering)
mu_all = mu_GL      # shape (N_ANGLES,)  already sorted ascending
w_all  = w_GL       # shape (N_ANGLES,)

def squared_exp_kernel(x: np.ndarray, length_scale: float, variance: float) -> np.ndarray:
    """
    Build the covariance matrix using the squared exponential kernel.

    C(xi, xj) = sigma^2 * exp( -(xi - xj)^2 / (2 * l^2) )

    Parameters
    ----------
    x            : 1-D array of spatial points, shape (M,)
    length_scale : l  — controls spatial smoothness
    variance     : sigma^2 — controls amplitude fluctuations

    Returns
    -------
    C : covariance matrix, shape (M, M)
    """
    diff = x[:, None] - x[None, :]          # (M, M) pairwise differences
    C = variance * np.exp(-diff**2 / (2.0 * length_scale**2))
    return C

def sample_grf(x: np.ndarray,
               mean: float,
               length_scale: float,
               variance: float,
               n_samples: int,
               rng: np.random.Generator) -> np.ndarray:
    """
    Draw n_samples realisations from a GRF defined on spatial points x.

    Uses the Cholesky decomposition  C = L L^T, then
    Q = M + L @ Z,  Z ~ N(0, I).

    Parameters
    ----------
    x            : spatial points where the source is evaluated, shape (N,)
    mean         : constant mean value of the GRF
    length_scale : l parameter of the squared exponential kernel
    variance     : sigma^2 parameter
    n_samples    : number of independent realisations to draw
    rng          : NumPy random Generator (for reproducibility)

    Returns
    -------
    Q_samples : array of shape (n_samples, M), each row is one source Q(x)
                NOTE: negative values are clipped to 0 (physical requirement)
    """
    N = len(x)
    C = squared_exp_kernel(x, length_scale, variance)

    # Add small nugget for numerical stability of Cholesky
    C += 1e-10 * np.eye(N)

    L = np.linalg.cholesky(C)                              # lower-triangular
    M = np.full(N, mean)                                   # shape (N,)

    # Draw standard normal vectors: shape (N, n_samples)
    Z = rng.standard_normal((N, n_samples))

    # Transform: Q = M + L @ Z  →  shape (n_samples, N)
    Q_samples = (M[:, None] + L @ Z).T

    # Physical constraint: source must be non-negative
    Q_samples = np.clip(Q_samples, 0.0, None)

    return Q_samples

def solve_sn_1d(Q_j: np.ndarray,
                Sigma_t: float,
                Sigma_s0: float,
                Sigma_s1: float,
                mu: np.ndarray,
                w: np.ndarray,
                h: float,
                max_iter: int = 10000,
                tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the steady-state 1D NTE for a given source distribution Q(x).

    Equation (1) discretised via:
      - Cell-averaged finite difference (Eq. 4-5)
      - Discrete ordinates S_N with Gauss-Legendre quadrature (Eq. 6)
      - Diamond difference relation  psi_{n,j} = (psi_{n,j+1/2} + psi_{n,j-1/2})/2  (Eq. 9)

    Boundary conditions (vacuum):
      psi(0, mu) = 0   for mu > 0  (left boundary, incoming from left)
      psi(X, mu) = 0   for mu < 0  (right boundary, incoming from right)

    Source iteration convergence criterion applied to scalar flux phi_0.

    Parameters
    ----------
    Q_j      : cell-average source values, shape (J,)
    Sigma_t  : total macroscopic cross section (cm^-1)
    Sigma_s0 : zeroth Legendre moment of scattering XS (cm^-1)
    Sigma_s1 : first Legendre moment of scattering XS (cm^-1), 0 = isotropic
    x_centres: cell-centre coordinates (used only for shape/reference)
    mu       : quadrature cosines, shape (N,)
    w        : quadrature weights,  shape (N,)
    h        : uniform cell width (cm)
    max_iter : maximum source-iteration sweeps
    tol      : relative convergence tolerance on phi_0

    Returns
    -------
    phi_0 : cell-average scalar flux  [Eq. 7], shape (J,)
    phi_1 : cell-average neutron current [Eq. 8], shape (J,)
    """
    J = len(Q_j)
    N = len(mu)

    # Initialise scalar flux and current to zero
    phi_0 = np.zeros(J)
    phi_1 = np.zeros(J)

    # Angular flux at cell centres, shape (N, J)
    psi_centre = np.zeros((N, J))

    for iteration in range(max_iter):
        phi_0_old = phi_0.copy()

        # Cell-average scattering + external source  [RHS of Eq. 6]
        # S_{n,j} = 0.5 * (Sigma_s0 * phi_0_j + 3 * mu_n * Sigma_s1 * phi_1_j) + Q_j / 2
        # shape (N, J)
        S = 0.5 * (
            Sigma_s0 * phi_0[None, :]
            + 3.0 * mu[:, None] * Sigma_s1 * phi_1[None, :]
            + Q_j[None, :]
        )
        # Angular sweep
        for n in range(N):
            mu_n = mu[n]
            if mu_n > 0.0:
                # Sweep left → right; incoming left-edge flux = 0 (vacuum BC)
                psi_in = 0.0                           # psi_{n, j-1/2} at j=0
                for j in range(J):
                    # From transport eq + diamond difference:
                    # (mu_n/h + Sigma_t/2) * psi_out = S[n,j] - (Sigma_t/2 - mu_n/h) * psi_in
                    A = mu_n / h + Sigma_t / 2.0
                    B = Sigma_t / 2.0 - mu_n / h
                    psi_out = (S[n, j] - B * psi_in) / A
                    # Enforce non-negativity (step-characteristic fix-up)
                    if psi_out < 0.0:
                        psi_out = 0.0
                    psi_centre[n, j] = 0.5 * (psi_out + psi_in)
                    psi_in = psi_out                   # march to next cell
            else:
                # Sweep right → left; incoming right-edge flux = 0 (vacuum BC)
                psi_in = 0.0                           # psi_{n, j+1/2} at j=J-1
                for j in range(J - 1, -1, -1):
                    # mu_n < 0; rewrite with |mu_n|:
                    # (|mu_n|/h + Sigma_t/2) * psi_out = S[n,j] - (Sigma_t/2 - |mu_n|/h)*psi_in
                    abs_mu = -mu_n
                    A = abs_mu / h + Sigma_t / 2.0
                    B = Sigma_t / 2.0 - abs_mu / h
                    psi_out = (S[n, j] - B * psi_in) / A
                    if psi_out < 0.0:
                        psi_out = 0.0
                    psi_centre[n, j] = 0.5 * (psi_out + psi_in)
                    psi_in = psi_out                   # march to next cell (leftward)

        # Update scalar flux [Eq. 7] and current [Eq. 8]
        phi_0 = np.einsum('n,nj->j', w, psi_centre)   # sum_n w_n * psi_{n,j}
        phi_1 = np.einsum('n,nj->j', w * mu, psi_centre)  # sum_n w_n * mu_n * psi_{n,j}

        # Convergence check on scalar flux
        norm_old = np.linalg.norm(phi_0_old)
        if norm_old > 0.0:
            rel_change = np.linalg.norm(phi_0 - phi_0_old) / norm_old
        else:
            rel_change = np.linalg.norm(phi_0)

        if rel_change < tol:
            break

    return phi_0, phi_1

def generate_dataset(Sigma_t: float,
                     Sigma_s0: float,
                     Sigma_s1: float,
                     n_samples: int,
                     grf_mean: float,
                     grf_length_scale: float,
                     grf_variance: float,
                     seed: int = 42) -> dict:
    """
    Generate a complete (source, flux) training dataset for one model.

    Returns a dict with keys:
      'Q'        : source distributions, shape (N_total, J)
      'phi_0'    : scalar flux,          shape (N_total, J)
      'mu_GL'    : quadrature cosines,   shape (N_angles,)
      'w_GL'     : quadrature weights,   shape (N_angles,)
      'x'        : cell centres,         shape (J,)
    """
    rng = np.random.default_rng(seed)

    print("Generating dataset.")

    # Sample GRF sources — same sources reused across all Sigma_s0 values
    Q_samples = sample_grf(
        x=x_centres,
        mean=grf_mean,
        length_scale=grf_length_scale,
        variance=grf_variance,
        n_samples=n_samples,
        rng=rng
    )   # shape (n_samples, J)

    all_Q     = []
    all_phi0  = []

    for i, Q_j in enumerate(Q_samples):
        if (i + 1) % 100 == 0:
            print(f"    Sample {i+1}/{n_samples} ...")

        phi_0, _ = solve_sn_1d(
            Q_j=Q_j,
            Sigma_t=Sigma_t,
            Sigma_s0=Sigma_s0,
            Sigma_s1=Sigma_s1,
            mu=mu_all,
            w=w_all,
            h=h
        )
        all_Q.append(Q_j)
        all_phi0.append(phi_0)

    dataset = {
        'Q':        np.array(all_Q),      # (N_total, J)
        'phi_0':    np.array(all_phi0),   # (N_total, J)
        'mu_GL':    mu_all.copy(),        # (N_angles,)
        'w_GL':     w_all.copy(),         # (N_angles,)
        'x':        x_centres.copy(),     # (J,)
    }

    print(f"  Done. Dataset shape: Q={dataset['Q'].shape}, phi_0={dataset['phi_0'].shape}")
    return dataset

if __name__ == "__main__":
    # Generate datasets

    os.makedirs("datasets", exist_ok=True)

    # ---- Common GRF training parameters (Section II.D) ----
    GRF_MEAN     = 5.0
    GRF_L        = 0.1
    GRF_VARIANCE = 1.0
    N_SAMPLES    = 500      # samples per model (200 for M_Para)

    # ------------------------------------------------------------------ #
    # Model 1: Isotropic Scattering  (M_Iso)                             #
    # Sigma_t=1.0, Sigma_a=0.5, Sigma_s0=0.5, Sigma_s1=0.0              #
    # ------------------------------------------------------------------ #
    ds_iso = generate_dataset(
        Sigma_t         = 1.0,
        Sigma_s0        = 0.5,
        Sigma_s1        = 0.0,
        n_samples       = N_SAMPLES,
        grf_mean        = GRF_MEAN,
        grf_length_scale= GRF_L,
        grf_variance    = GRF_VARIANCE,
        seed            = 42
    )
    np.savez("datasets/M_Iso_train.npz", **ds_iso)
    print("  Saved → datasets/M_Iso_train.npz")

    # ------------------------------------------------------------------ #
    # Model 2: Anisotropic Scattering  (M_Aniso)                         #
    # Sigma_t=1.0, Sigma_a=0.5, Sigma_s0=0.5, Sigma_s1=0.15             #
    # ------------------------------------------------------------------ #
    ds_aniso = generate_dataset(
        Sigma_t         = 1.0,
        Sigma_s0        = 0.5,
        Sigma_s1        = 0.15,
        n_samples       = N_SAMPLES,
        grf_mean        = GRF_MEAN,
        grf_length_scale= GRF_L,
        grf_variance    = GRF_VARIANCE,
        seed            = 42
    )
    np.savez("datasets/M_Aniso_train.npz", **ds_aniso)
    print("  Saved → datasets/M_Aniso_train.npz")

    # ------------------------------------------------------------------ #
    # Model 3: Pure Scattering  (M_Pure)                                 #
    # Sigma_t=1.0, Sigma_a=0.0, Sigma_s0=1.0, Sigma_s1=0.33             #
    # ------------------------------------------------------------------ #
    ds_pure = generate_dataset(
        Sigma_t         = 1.0,
        Sigma_s0        = 1.0,
        Sigma_s1        = 0.33,
        n_samples       = N_SAMPLES,
        grf_mean        = GRF_MEAN,
        grf_length_scale= GRF_L,
        grf_variance    = GRF_VARIANCE,
        seed            = 42
    )
    np.savez("datasets/M_Pure_train.npz", **ds_pure)
    print("  Saved → datasets/M_Pure_train.npz")
