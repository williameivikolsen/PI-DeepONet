import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate_all import evaluate_combination, SCENARIOS_GRF, SCENARIOS_SINGLE

size = "small"

CHECKPOINTS = {
    "DeepONet":  "trained_models/" + size + "/deeponet.pkl",
    "PI DeepONet": "trained_models/" + size + "/pideeponet_relu.pkl",
    "PI DeepONet (angular)":  "trained_models/" + size + "/pideeponet_angular_relu.pkl",
}

results = {name: evaluate_combination(path) for name, path in CHECKPOINTS.items()}

labels  = [lbl for _, lbl in SCENARIOS_GRF + SCENARIOS_SINGLE]
palette = sns.color_palette("deep", len(CHECKPOINTS))


def plot_metric(metric, ylabel):
    x_pos = onp.arange(len(labels))
    width = 0.8 / len(CHECKPOINTS)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (name, res) in enumerate(results.items()):
        means  = onp.array([res[lbl][f"{metric}_mean"] for lbl in labels])
        stds   = onp.array([res[lbl][f"{metric}_std"]  for lbl in labels])
        offset = (k - (len(CHECKPOINTS) - 1) / 2) * width
        ax.errorbar(x_pos + offset, means, yerr=stds, fmt="o", capsize=2,
                    color=palette[k], label=name, markersize=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout()


plot_metric("ARE", "Average Relative Error [%]")
plot_metric("R2", "$R^2$")
plt.show()
