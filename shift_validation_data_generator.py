import os
import numpy as np

from data_generator import (
    x_centres, h, mu_all, w_all,
    sample_grf, solve_sn_1d,
)

SIGMA_T, SIGMA_S0, SIGMA_S1 = 1.0, 0.5, 0.0
SHIFT_SEED = 8888
OUTPUT_DIR = "datasets"

# Shift-validation set.
#          l,    mean, variance
GROUPS = [(0.07,  3.0, 0.6),    # rough, low mean
          (0.06,  6.0, 2.5),    # rough, training-like mean
          (0.15,  8.0, 1.5),    # mildly smoother
          (0.40, 25.0, 1.5),    # smooth, high mean
          (0.70, 40.0, 1.0)]    # very smooth, nearly flat
N_PER_GROUP = 10


def solve_batch(Q_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the Sn problem for each source. Returns both the scalar flux and the
    cell-centre angular flux, in the same (N, N_angles, J) convention as the
    training and validation sets.
    """
    n = Q_samples.shape[0]
    phi_0_all = np.empty_like(Q_samples)                              # (n, J)
    psi_all   = np.empty((n, len(mu_all), Q_samples.shape[1]))        # (n, A, J)
    for i, Q_j in enumerate(Q_samples):
        phi_0, _, psi_centre = solve_sn_1d(
            Q_j=Q_j, Sigma_t=SIGMA_T, Sigma_s0=SIGMA_S0, Sigma_s1=SIGMA_S1,
            mu=mu_all, w=w_all, h=h,
        )
        phi_0_all[i] = phi_0
        psi_all[i]   = psi_centre
    return phi_0_all, psi_all


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SHIFT_SEED)

    Qs, phis, psis, group = [], [], [], []
    for g, (l, m, v) in enumerate(GROUPS):
        print(f"Shift-validation group {g}: l={l}, mean={m}, sigma2={v}")
        Q = sample_grf(x=x_centres, mean=m, length_scale=l,
                       variance=v, n_samples=N_PER_GROUP, rng=rng)
        phi, psi = solve_batch(Q)
        Qs.append(Q); phis.append(phi); psis.append(psi)
        group.append(np.full(N_PER_GROUP, g))

    Q     = np.concatenate(Qs)       # (50, J)
    phi_0 = np.concatenate(phis)     # (50, J)
    psi   = np.concatenate(psis)     # (50, A, J)

    # Same consistency guarantee as the training set: GL quadrature of psi
    # must reproduce phi_0.
    phi0_from_psi = np.einsum('n,inj->ij', w_all, psi)
    max_abs = np.max(np.abs(phi0_from_psi - phi_0))
    assert max_abs < 1e-10, f"phi_0 vs quadrature(psi) mismatch: {max_abs:.2e}"

    out_path = os.path.join(OUTPUT_DIR, "M_Iso_shiftval.npz")
    np.savez(out_path, Q=Q, phi_0=phi_0, psi=psi,
             mu_GL=mu_all, w_GL=w_all, x=x_centres,
             group=np.concatenate(group), group_params=np.array(GROUPS))
    print(f"Saved {out_path}  (Q {Q.shape}, phi_0 {phi_0.shape}, psi {psi.shape})")
    print(f"  Consistency check passed: max|phi_0 - quad(psi)| = {max_abs:.2e}")


if __name__ == "__main__":
    main()
