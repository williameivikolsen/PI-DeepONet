import os

import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import load_model, r2_score, are

size = "large"

checkpoints = {
    "Benchmark":  "trained_models/" + size + "/deeponet.pkl",
    # "PI DeepONet 1": "trained_models/" + size + "/pideeponet_relu.pkl",
    "PI DeepONet 1": "trained_models/" + size + "/pideeponet_angular_scalar_data_relu.pkl",
    "PI DeepONet 2":  "trained_models/" + size + "/pideeponet_angular_relu.pkl",
    "Large DeepONet": "trained_models/" + size + "/large_deeponet.pkl",
}

test_dir = "datasets/test"

scenarios_grf = [
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
scenarios_single = [
    ("M_Iso_test_LC.npz",     "LC"),
    ("M_Iso_test_NLC.npz",    "NLC"),
    ("M_Iso_test_SIN_LF.npz", "SIN$_{LF}$"),
    ("M_Iso_test_SIN_HF.npz", "SIN$_{HF}$"),
]


def evaluate_combination(ckpt_path):
    model, _kind, _ckpt = load_model(ckpt_path)

    results = {}
    for filename, label in scenarios_grf + scenarios_single:
        ds = onp.load(os.path.join(test_dir, filename), allow_pickle=True)
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


results = {name: evaluate_combination(path) for name, path in checkpoints.items()}

labels  = [lbl for _, lbl in scenarios_grf + scenarios_single]
palette = sns.color_palette("deep", len(checkpoints))


def plot_metric(metric, ylabel):
    x_pos = onp.arange(len(labels))
    width = 0.8 / len(checkpoints)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Alternating background bands mark each scenario's column, so the
    # dodged points and their (horizontal, unrotated) label are visually
    # tied together without needing to trace a diagonal label back to them.
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="0.93", zorder=0)

    for k, (name, res) in enumerate(results.items()):
        print("---------")
        print(f"Model: {name}")
        print(f" NS ARE: {res['NS']['ARE_mean']:.4f}% +- {res['NS']['ARE_std']:.4f}%")
        print(f" NS R2:  {res['NS']['R2_mean']:.4f} +- {res['NS']['R2_std']:.4f}")
        means  = onp.array([res[lbl][f"{metric}_mean"] for lbl in labels])
        stds   = onp.array([res[lbl][f"{metric}_std"]  for lbl in labels])
        offset = (k - (len(checkpoints) - 1) / 2) * width
        ax.errorbar(x_pos + offset, means, yerr=stds, fmt="o", capsize=2,
                    color=palette[k], label=name, markersize=4, zorder=3)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6, zorder=1)
    fig.tight_layout()


plot_metric("ARE", "Average Relative Error [%]")
plot_metric("R2", "$R^2$")
plt.show()
