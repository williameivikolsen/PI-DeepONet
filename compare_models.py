import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate_all import evaluate_combination, SCENARIOS_GRF, SCENARIOS_SINGLE

size = "large"

CHECKPOINTS = {
    "Benchmark":  "trained_models/" + size + "/deeponet.pkl",
    # "PI DeepONet 1": "trained_models/" + size + "/pideeponet_relu.pkl",
    "PI DeepONet 1": "trained_models/" + size + "/pideeponet_angular_scalar_data_relu.pkl",
    "PI DeepONet 2":  "trained_models/" + size + "/pideeponet_angular_relu.pkl",
    "Large DeepONet": "trained_models/" + size + "/large_deeponet.pkl",
}

results = {name: evaluate_combination(path) for name, path in CHECKPOINTS.items()}

labels  = [lbl for _, lbl in SCENARIOS_GRF + SCENARIOS_SINGLE]
palette = sns.color_palette("deep", len(CHECKPOINTS))


def plot_metric(metric, ylabel):
    x_pos = onp.arange(len(labels))
    width = 0.8 / len(CHECKPOINTS)

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
        offset = (k - (len(CHECKPOINTS) - 1) / 2) * width
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
