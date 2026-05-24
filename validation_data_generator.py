import os
import numpy as np

from data_generator import (
    x_centres, h, mu_all, w_all,
    sample_grf, solve_sn_1d,
)

SIGMA_T, SIGMA_S0, SIGMA_S1 = 1.0, 0.5, 0.0
VAL_SEED   = 7777
OUTPUT_DIR = "datasets"


def solve_batch(Q_samples: np.ndarray) -> np.ndarray:
    phi_0_all = np.empty_like(Q_samples)
    for i, Q_j in enumerate(Q_samples):
        phi_0, _ = solve_sn_1d(
            Q_j=Q_j, Sigma_t=SIGMA_T, Sigma_s0=SIGMA_S0, Sigma_s1=SIGMA_S1,
            mu=mu_all, w=w_all, h=h,
        )
        phi_0_all[i] = phi_0
        if (i + 1) % 10 == 0:
            print(f"    sample {i+1}/{len(Q_samples)}")
    return phi_0_all


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(VAL_SEED)

    print("Validation set: in-distribution half (l=0.10, mean=5.0, sigma2=1.0)")
    Q_in = sample_grf(x=x_centres, mean=5.0, length_scale=0.10,
                      variance=1.0, n_samples=25, rng=rng)
    phi_in = solve_batch(Q_in)

    print("Validation set: mildly shifted half (l=0.09, mean=5.5, sigma2=1.1)")
    Q_shift = sample_grf(x=x_centres, mean=5.5, length_scale=0.09,
                         variance=1.1, n_samples=25, rng=rng)
    phi_shift = solve_batch(Q_shift)

    Q     = np.concatenate([Q_in,   Q_shift],   axis=0)   # (50, J)
    phi_0 = np.concatenate([phi_in, phi_shift], axis=0)   # (50, J)

    out_path = os.path.join(OUTPUT_DIR, "M_Iso_val.npz")
    np.savez(out_path, Q=Q, phi_0=phi_0,
             mu_GL=mu_all, w_GL=w_all, x=x_centres)
    print(f"Saved {out_path}  (Q {Q.shape}, phi_0 {phi_0.shape})")


if __name__ == "__main__":
    main()