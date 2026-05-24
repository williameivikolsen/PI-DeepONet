import os
import pickle
import time

import numpy as onp
import jax.numpy as jnp
from jax import vmap

import matplotlib.pyplot as plt
import seaborn as sns


TEST_DIR    = "datasets/test"
SIZES       = ["small", "medium", "large"]
MODEL_FILES = {
    "DeepONet":            "deeponet.pkl",
    "PI-DeepONet":         "pideeponet.pkl",
    "PI-DeepONet (no data)": "pideeponet_no_data.pkl",
}

SCENARIOS_GRF = [
    ("M_Iso_test_NS.npz",  "NS"),
    ("M_Iso_test_SS1.npz", "SS$_1$"),
    ("M_Iso_test_SS2.npz", "SS$_2$"),
    ("M_Iso_test_SS3.npz", "SS$_3$"),
    ("M_Iso_test_LS1.npz", "LS$_1$"),
    ("M_Iso_test_LS2.npz", "LS$_2$"),
    ("M_Iso_test_LS3.npz", "LS$_3$"),
    ("M_Iso_test_LS4.npz", "LS$_4$"),
    ("M_Iso_test_LS5.npz", "LS$_5$"),
    ("M_Iso_test_LS6.npz", "LS$_6$"),
]
SCENARIOS_SINGLE = [
    ("M_Iso_test_LC.npz",     "LC"),
    ("M_Iso_test_NLC.npz",    "NLC"),
    ("M_Iso_test_SIN_LF.npz", "SIN$_{LF}$"),
    ("M_Iso_test_SIN_HF.npz", "SIN$_{HF}$"),
]


# ---------------------------------------------------------------------------
# Model loading & prediction (factored from evaluate_model.py)
# ---------------------------------------------------------------------------
def load_model(path: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = dict(ckpt["config"])
    cfg["x_sensors"] = jnp.asarray(cfg["x_sensors"])

    is_PI = "N_angles" in cfg
    if is_PI:
        from model import PI_DeepONet
        model = PI_DeepONet(**cfg)
        kind = "PI"
    else:
        from nonPI_model import DeepONet
        model = DeepONet(**cfg)
        kind = "nonPI"

    model.params = ckpt["params"]
    return model, kind


def predict_phi0_nonPI(model, Q_batch, x):
    n, J = Q_batch.shape
    Q_jax = jnp.asarray(Q_batch)
    x_jax = jnp.asarray(x)
    out = onp.empty((n, J))
    for i in range(n):
        Q_rep = jnp.broadcast_to(Q_jax[i], (J, J))
        out[i] = onp.asarray(model.predict_s(model.params, Q_rep, x_jax))
    return out


def predict_phi0_PI(model, Q_batch, x):
    n, J = Q_batch.shape
    Q_jax = jnp.asarray(Q_batch)
    x_jax = jnp.asarray(x)

    def phi0_at(Q_i, x_j):
        psi_vec = vmap(
            lambda mu_k: model.operator_net(model.params, Q_i, x_j, mu_k)
        )(model.mu_GL)
        return jnp.dot(model.w_GL, psi_vec)

    out = onp.empty((n, J))
    for i in range(n):
        phi0 = vmap(lambda x_j: phi0_at(Q_jax[i], x_j))(x_jax)
        out[i] = onp.asarray(model.output_scale * phi0)
    return out


PREDICTORS = {"nonPI": predict_phi0_nonPI, "PI": predict_phi0_PI}


def r2(true, pred):
    ss_res = onp.sum((true - pred) ** 2)
    ss_tot = onp.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def are(true, pred):
    return float(onp.mean(onp.abs((true - pred) / true)) * 100.0)


# ---------------------------------------------------------------------------
# Evaluate one (size, model) combination across all scenarios
# ---------------------------------------------------------------------------
def evaluate_combination(ckpt_path):
    model, kind = load_model(ckpt_path)
    predictor = PREDICTORS[kind]

    results = {}
    for filename, label in SCENARIOS_GRF + SCENARIOS_SINGLE:
        ds = onp.load(os.path.join(TEST_DIR, filename), allow_pickle=True)
        Q, phi_0, x = ds["Q"], ds["phi_0"], ds["x"]
        phi_pred = predictor(model, Q, x)
        ares = onp.array([are(phi_0[i], phi_pred[i]) for i in range(len(Q))])
        r2s  = onp.array([r2 (phi_0[i], phi_pred[i]) for i in range(len(Q))])
        results[label] = {
            "ARE_mean": float(ares.mean()),
            "ARE_std":  float(ares.std()),
            "R2_mean":  float(r2s.mean()),
            "R2_std":   float(r2s.std()),
            "n":        len(Q),
        }
    return results


# ---------------------------------------------------------------------------
# Plotting in Sahadath style
# ---------------------------------------------------------------------------
def plot_ARE_per_size(all_results, size, out_dir):
    """
    For a given dataset size, plot ARE for all model types across scenarios.
    GRF scenarios get error bars; single-sample scenarios are points only.
    """
    grf_labels    = [lbl for _, lbl in SCENARIOS_GRF]
    single_labels = [lbl for _, lbl in SCENARIOS_SINGLE]
    all_labels    = grf_labels + single_labels

    x_pos = onp.arange(len(all_labels))
    width = 0.25  # bar/marker offset within group

    palette = sns.color_palette("deep", len(MODEL_FILES))

    fig, ax = plt.subplots(figsize=(11, 5))

    for k, (model_name, _) in enumerate(MODEL_FILES.items()):
        results = all_results[size].get(model_name)
        if results is None:
            continue   # checkpoint missing — skip silently
        means = onp.array([results[lbl]["ARE_mean"] for lbl in all_labels])
        stds  = onp.array([results[lbl]["ARE_std"]  for lbl in all_labels])
        # Hide std bars for single-sample scenarios (std is 0 by definition).
        stds[len(grf_labels):] = 0.0

        offset = (k - (len(MODEL_FILES) - 1) / 2) * width
        ax.errorbar(
            x_pos + offset, means, yerr=stds,
            fmt="o", capsize=3, color=palette[k], label=model_name,
            markersize=5, lw=1.3,
        )

    # Visual separator between GRF and single-sample groups.
    ax.axvline(len(grf_labels) - 0.5, color="gray", ls="--", lw=0.8, alpha=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_ylabel("Average Relative Error [\\%]")
    ax.set_title(f"M$_{{\\mathrm{{Iso}}}}$ test performance — training set: {size}")
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=True, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"ARE_{size}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_R2_per_size(all_results, size, out_dir):
    """Same layout as ARE plot but for R^2. Clamps display range so a few
    very-negative outliers don't crush the scale."""
    grf_labels    = [lbl for _, lbl in SCENARIOS_GRF]
    single_labels = [lbl for _, lbl in SCENARIOS_SINGLE]
    all_labels    = grf_labels + single_labels

    x_pos = onp.arange(len(all_labels))
    width = 0.25
    palette = sns.color_palette("deep", len(MODEL_FILES))

    fig, ax = plt.subplots(figsize=(11, 5))

    for k, (model_name, _) in enumerate(MODEL_FILES.items()):
        results = all_results[size].get(model_name)
        if results is None:
            continue
        means = onp.array([results[lbl]["R2_mean"] for lbl in all_labels])
        stds  = onp.array([results[lbl]["R2_std"]  for lbl in all_labels])
        stds[len(grf_labels):] = 0.0

        offset = (k - (len(MODEL_FILES) - 1) / 2) * width
        ax.errorbar(
            x_pos + offset, means, yerr=stds,
            fmt="o", capsize=3, color=palette[k], label=model_name,
            markersize=5, lw=1.3,
        )

    ax.axvline(len(grf_labels) - 0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_ylabel("$R^2$")
    ax.set_title(f"M$_{{\\mathrm{{Iso}}}}$ test performance — training set: {size}")
    # R^2 can go strongly negative for badly-fitted no-data models; clamp
    # the lower bound for readability but log a warning when truncated.
    ymin, ymax = ax.get_ylim()
    if ymin < -1.0:
        ax.set_ylim(-1.0, max(1.05, ymax))
    ax.axhline(0.0, color="gray", ls=":", lw=0.6, alpha=0.7)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=True, loc="lower left")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"R2_{size}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    # all_results[size][model_name] -> {label: {ARE_mean, ARE_std, ...}}
    all_results = {}
    for size in SIZES:
        print(f"\n=== Evaluating size: {size} ===")
        all_results[size] = {}
        for model_name, filename in MODEL_FILES.items():
            ckpt_path = os.path.join("trained_models", size, filename)
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {model_name}: {ckpt_path} not found")
                continue
            t0 = time.time()
            results = evaluate_combination(ckpt_path)
            print(f"  {model_name:<24s} done in {time.time()-t0:.1f}s")
            all_results[size][model_name] = results

    # Per-size plots
    print("\n=== Plotting ===")
    for size in SIZES:
        if not all_results[size]:
            continue
        plot_ARE_per_size(all_results, size, out_dir)
        plot_R2_per_size(all_results, size, out_dir)

    # Also print summary tables for completeness
    print("\n=== Summary (ARE %) ===")
    for size in SIZES:
        if not all_results[size]:
            continue
        print(f"\n--- {size} ---")
        labels = [lbl for _, lbl in SCENARIOS_GRF + SCENARIOS_SINGLE]
        header = f"{'Scenario':<10s} " + " ".join(
            f"{name:>22s}" for name in all_results[size].keys()
        )
        print(header)
        for lbl in labels:
            row = f"{lbl:<10s} "
            for name in all_results[size].keys():
                r = all_results[size][name][lbl]
                if r["ARE_std"] > 0:
                    row += f" {r['ARE_mean']:>9.3f} +/- {r['ARE_std']:>5.3f}"
                else:
                    row += f" {r['ARE_mean']:>21.3f}"
            print(row)


if __name__ == "__main__":
    main()