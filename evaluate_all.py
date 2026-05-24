import os
import time

import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import load_model, r2_score, are


TEST_DIR    = "datasets/test"
SIZES       = ["small", "medium", "large"]
MODEL_FILES = {
    "DeepONet":              "deeponet.pkl",
    "PI-DeepONet":           "pideeponet.pkl",
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


def evaluate_combination(ckpt_path):
    model, _kind, _ckpt = load_model(ckpt_path)

    results = {}
    for filename, label in SCENARIOS_GRF + SCENARIOS_SINGLE:
        ds = onp.load(os.path.join(TEST_DIR, filename), allow_pickle=True)
        Q, phi_0, x = ds["Q"], ds["phi_0"], ds["x"]
        phi_pred = onp.asarray(model.predict_phi0(model.params, Q, x))
        ares = onp.array([are(phi_0[i], phi_pred[i]) for i in range(len(Q))])
        r2s  = onp.array([r2_score(phi_0[i], phi_pred[i]) for i in range(len(Q))])
        results[label] = {
            "ARE_mean": float(ares.mean()),
            "ARE_std":  float(ares.std()),
            "R2_mean":  float(r2s.mean()),
            "R2_std":   float(r2s.std()),
            "n":        len(Q),
        }
    return results


def _plot_metric(all_results, size, out_dir, metric, ylabel, fname,
                 yscale=None, clamp_negative=False, legend_loc="upper left"):
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
        means = onp.array([results[lbl][f"{metric}_mean"] for lbl in all_labels])
        stds  = onp.array([results[lbl][f"{metric}_std"]  for lbl in all_labels])
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
    ax.set_ylabel(ylabel)
    ax.set_title(f"M$_{{\\mathrm{{Iso}}}}$ test performance — training set: {size}")
    if yscale is not None:
        ax.set_yscale(yscale)
    if clamp_negative:
        ymin, ymax = ax.get_ylim()
        if ymin < -1.0:
            ax.set_ylim(-1.0, max(1.05, ymax))
        ax.axhline(0.0, color="gray", ls=":", lw=0.6, alpha=0.7)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=True, loc=legend_loc)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{fname}_{size}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")
    plt.close(fig)


def main():
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

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

    print("\n=== Plotting ===")
    for size in SIZES:
        if not all_results[size]:
            continue
        _plot_metric(all_results, size, out_dir,
                     metric="ARE", ylabel="Average Relative Error [\\%]",
                     fname="ARE", yscale="log", legend_loc="upper left")
        _plot_metric(all_results, size, out_dir,
                     metric="R2", ylabel="$R^2$",
                     fname="R2", clamp_negative=True, legend_loc="lower left")

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
