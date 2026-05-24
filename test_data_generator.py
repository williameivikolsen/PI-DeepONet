import os
import numpy as np

from data_generator import (
    x_centres, h, mu_all, w_all,
    sample_grf, solve_sn_1d,
)

SIGMA_T  = 1.0
SIGMA_S0 = 0.5
SIGMA_S1 = 0.0

N_GRF_SAMPLES = 200      # paper uses 200 samples per GRF scenario
TEST_SEED     = 2026
OUTPUT_DIR    = "datasets/test"

def solve_batch(Q_samples: np.ndarray, label: str) -> np.ndarray:
    """Run the Sn solver on each row of Q_samples; return phi_0 array."""
    n = Q_samples.shape[0]
    phi_0_all = np.empty_like(Q_samples)
    for i, Q_j in enumerate(Q_samples):
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"    [{label}] sample {i+1}/{n}")
        phi_0, _ = solve_sn_1d(
            Q_j=Q_j, Sigma_t=SIGMA_T, Sigma_s0=SIGMA_S0, Sigma_s1=SIGMA_S1,
            mu=mu_all, w=w_all, h=h,
        )
        phi_0_all[i] = phi_0
    return phi_0_all

def save_scenario(filename: str, Q: np.ndarray, phi_0: np.ndarray,
                  description: str, params: dict) -> None:
    """Save a single scenario as an .npz with consistent metadata."""
    path = os.path.join(OUTPUT_DIR, filename)
    np.savez(
        path,
        Q=Q,
        phi_0=phi_0,
        mu_GL=mu_all,
        w_GL=w_all,
        x=x_centres,
        description=description,
        **{f"param_{k}": v for k, v in params.items()},
    )
    print(f"    Saved → {path}  (Q shape {Q.shape}, phi_0 shape {phi_0.shape})")


def grf_scenario(label: str, filename: str, description: str,
                 length_scale: float, mean: float, variance: float,
                 rng: np.random.Generator) -> None:
    """Generate, solve, and save one GRF-based test scenario."""
    print(f"\n[{label}] {description}")
    print(f"    l={length_scale}, mean={mean}, sigma2={variance}, n={N_GRF_SAMPLES}")
    Q = sample_grf(
        x=x_centres, mean=mean, length_scale=length_scale, variance=variance,
        n_samples=N_GRF_SAMPLES, rng=rng,
    )
    phi_0 = solve_batch(Q, label)
    save_scenario(
        filename=filename, Q=Q, phi_0=phi_0, description=description,
        params={
            "length_scale": length_scale,
            "mean":         mean,
            "variance":     variance,
            "n_samples":    N_GRF_SAMPLES,
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(TEST_SEED)

    # -------------------------------------------------------------------
    # GRF-based scenarios
    # -------------------------------------------------------------------
    grf_scenarios = [
        # label,  filename,             description,                  l,    mean, var
        ("NS",    "M_Iso_test_NS.npz",  "No shift in GRF parameters", 0.10, 5.0,  1.0),
        ("SS_1",  "M_Iso_test_SS1.npz", "Small shift in GRF params",  0.11, 5.5,  1.1),
        ("SS_2",  "M_Iso_test_SS2.npz", "Small shift in GRF params",  0.09, 4.5,  0.9),
        ("SS_3",  "M_Iso_test_SS3.npz", "Small shift in GRF params",  0.09, 5.5,  1.1),
        ("LS_1",  "M_Iso_test_LS1.npz", "Large shift in GRF params",  0.08, 2.5,  0.5),
        ("LS_2",  "M_Iso_test_LS2.npz", "Large shift in GRF params",  0.20, 10.0, 2.0),
        ("LS_3",  "M_Iso_test_LS3.npz", "Large shift in GRF params",  0.30, 50.0, 2.0),
        ("LS_4",  "M_Iso_test_LS4.npz", "Large shift in GRF params",  0.50, 50.0, 2.0),
        ("LS_5",  "M_Iso_test_LS5.npz", "Large shift in GRF params",  0.80, 50.0, 2.0),
        ("LS_6",  "M_Iso_test_LS6.npz", "Large shift in GRF params",  1.00, 50.0, 2.0),
    ]
    for label, filename, desc, l, m, v in grf_scenarios:
        grf_scenario(label, filename, desc,
                     length_scale=l, mean=m, variance=v, rng=rng)

    # -------------------------------------------------------------------
    # Linear combination (single sample) — M_Iso coefficients: 1.22, 1.08, 1.68
    # Q_LC = a1*Q1 + a2*Q2 + a3*Q3 with
    #   Q1 ~ GRF(l=0.08, mean=2.5,  sigma2=0.5)
    #   Q2 ~ GRF(l=0.10, mean=5.0,  sigma2=1.0)
    #   Q3 ~ GRF(l=0.20, mean=10.0, sigma2=2.0)
    # -------------------------------------------------------------------
    print("\n[LC] Linear combination of three GRF sources")
    a1, a2, a3 = 1.22, 1.08, 1.68
    Q1 = sample_grf(x=x_centres, mean=2.5,  length_scale=0.08,
                    variance=0.5, n_samples=1, rng=rng)
    Q2 = sample_grf(x=x_centres, mean=5.0,  length_scale=0.10,
                    variance=1.0, n_samples=1, rng=rng)
    Q3 = sample_grf(x=x_centres, mean=10.0, length_scale=0.20,
                    variance=2.0, n_samples=1, rng=rng)
    Q_LC = a1 * Q1 + a2 * Q2 + a3 * Q3
    Q_LC = np.clip(Q_LC, 0.0, None)   # non-negativity for consistency
    phi_LC = solve_batch(Q_LC, "LC")
    save_scenario(
        filename="M_Iso_test_LC.npz", Q=Q_LC, phi_0=phi_LC,
        description="Linear combination of three GRF sources",
        params={"a1": a1, "a2": a2, "a3": a3,
                "Q1_l": 0.08, "Q1_mean": 2.5,  "Q1_var": 0.5,
                "Q2_l": 0.10, "Q2_mean": 5.0,  "Q2_var": 1.0,
                "Q3_l": 0.20, "Q3_mean": 10.0, "Q3_var": 2.0},
    )

    # -------------------------------------------------------------------
    # Nonlinear combination (single sample): Q_NLC = Q1 * Q2
    #   Q1 ~ GRF(l=0.08, mean=2.5,  sigma2=0.5)
    #   Q2 ~ GRF(l=0.20, mean=10.0, sigma2=2.0)
    # -------------------------------------------------------------------
    print("\n[NLC] Nonlinear (multiplicative) combination of two GRF sources")
    Q1_nlc = sample_grf(x=x_centres, mean=2.5,  length_scale=0.08,
                        variance=0.5, n_samples=1, rng=rng)
    Q2_nlc = sample_grf(x=x_centres, mean=10.0, length_scale=0.20,
                        variance=2.0, n_samples=1, rng=rng)
    Q_NLC = Q1_nlc * Q2_nlc
    Q_NLC = np.clip(Q_NLC, 0.0, None)
    phi_NLC = solve_batch(Q_NLC, "NLC")
    save_scenario(
        filename="M_Iso_test_NLC.npz", Q=Q_NLC, phi_0=phi_NLC,
        description="Nonlinear (product) combination of two GRF sources",
        params={"Q1_l": 0.08, "Q1_mean": 2.5,  "Q1_var": 0.5,
                "Q2_l": 0.20, "Q2_mean": 10.0, "Q2_var": 2.0},
    )

    # -------------------------------------------------------------------
    # Sinusoidal sources (single sample each): Q(x) = 12.5 + sin(2 pi f x)
    # -------------------------------------------------------------------
    for label, filename, freq in [
        ("SIN_LF", "M_Iso_test_SIN_LF.npz", 0.2),
        ("SIN_HF", "M_Iso_test_SIN_HF.npz", 0.8),
    ]:
        print(f"\n[{label}] Sinusoidal source, f = {freq}")
        Q_sin = (12.5 + np.sin(2.0 * np.pi * freq * x_centres))[None, :]
        Q_sin = np.clip(Q_sin, 0.0, None)   # always positive here; included for uniformity
        phi_sin = solve_batch(Q_sin, label)
        save_scenario(
            filename=filename, Q=Q_sin, phi_0=phi_sin,
            description=f"Sinusoidal source with frequency f={freq}",
            params={"mean": 12.5, "frequency": freq},
        )

    print("\nAll M_Iso test scenarios generated.")


if __name__ == "__main__":
    main()