import os
import numpy as np

from data_generator import (
    x_centres, h, mu_all, w_all,
    sample_grf, solve_sn_1d,
)

SIGMA_T, SIGMA_S0, SIGMA_S1 = 1.0, 0.5, 0.0
VAL_SEED   = 7777
OUTPUT_DIR = "datasets"


def solve_batch(Q_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the Sn problem for each source. Returns both the scalar flux and
    the cell-centre angular flux so the validation set carries psi with the
    same (N, N_angles, J) convention as the training set.
    """
    n = Q_samples.shape[0]
    phi_0_all = np.empty_like(Q_samples)              # (n, J)
    psi_all   = np.empty((n, len(mu_all), Q_samples.shape[1]))  # (n, A, J)
    for i, Q_j in enumerate(Q_samples):
        phi_0, _, psi_centre = solve_sn_1d(
            Q_j=Q_j, Sigma_t=SIGMA_T, Sigma_s0=SIGMA_S0, Sigma_s1=SIGMA_S1,
            mu=mu_all, w=w_all, h=h,
        )
        phi_0_all[i] = phi_0
        psi_all[i]   = psi_centre
        if (i + 1) % 10 == 0:
            print(f"    sample {i+1}/{len(Q_samples)}")
    return phi_0_all, psi_all


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(VAL_SEED)

    print("Validation set: in-distribution half (l=0.10, mean=5.0, sigma2=1.0)")
    Q_in = sample_grf(x=x_centres, mean=5.0, length_scale=0.10,
                      variance=1.0, n_samples=25, rng=rng)
    phi_in, psi_in = solve_batch(Q_in)

    print("Validation set: mildly shifted half (l=0.09, mean=5.5, sigma2=1.1)")
    Q_shift = sample_grf(x=x_centres, mean=5.5, length_scale=0.09,
                         variance=1.1, n_samples=25, rng=rng)
    phi_shift, psi_shift = solve_batch(Q_shift)

    Q     = np.concatenate([Q_in,   Q_shift],   axis=0)   # (50, J)
    phi_0 = np.concatenate([phi_in, phi_shift], axis=0)   # (50, J)
    psi   = np.concatenate([psi_in, psi_shift], axis=0)   # (50, A, J)

    # Same consistency guarantee as the training set: GL quadrature of psi
    # must reproduce phi_0.
    phi0_from_psi = np.einsum('n,inj->ij', w_all, psi)
    max_abs = np.max(np.abs(phi0_from_psi - phi_0))
    assert max_abs < 1e-10, f"phi_0 vs quadrature(psi) mismatch: {max_abs:.2e}"

    out_path = os.path.join(OUTPUT_DIR, "M_Iso_val.npz")
    np.savez(out_path, Q=Q, phi_0=phi_0, psi=psi,
             mu_GL=mu_all, w_GL=w_all, x=x_centres)
    print(f"Saved {out_path}  (Q {Q.shape}, phi_0 {phi_0.shape}, psi {psi.shape})")
    print(f"  Consistency check passed: max|phi_0 - quad(psi)| = {max_abs:.2e}")


if __name__ == "__main__":
    main()