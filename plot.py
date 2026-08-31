import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import load_model

# size = "large"
# file = "pideeponet_angular_relu"
# file = "large_deeponet"

# CHECKPOINT = f"trained_models/{size}/{file}.pkl"
CHECKPOINT = f"trained_models/lr_search/tanh/pideeponet_angular.pkl"

SAMPLE_IDX = 0

SCENARIOS = [
    ("NS",     "datasets/test/M_Iso_test_NS.npz"),
    ("LS1",    "datasets/test/M_Iso_test_LS1.npz"),
    ("LS3",    "datasets/test/M_Iso_test_LS3.npz"),
    ("LC",     "datasets/test/M_Iso_test_LC.npz"),
    ("NLC",    "datasets/test/M_Iso_test_NLC.npz"),
    ("SIN_HF", "datasets/test/M_Iso_test_SIN_HF.npz"),
]


model, kind, ckpt = load_model(CHECKPOINT)

palette  = sns.color_palette("deep", 3)
col_Q    = palette[0]
col_true = "#18A34F9C"
col_pred = "#B5183AD4"

fig, axes = plt.subplots(3, 2, figsize=(11, 10))

for ax, (label, path) in zip(axes.flat, SCENARIOS):
    ds_np = onp.load(path)
    ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files
             if ds_np[k].dtype.kind in "fiu"}

    Q_all     = onp.asarray(ds['Q'])
    phi_0_all = onp.asarray(ds['phi_0'])
    x         = onp.asarray(ds['x'])

    Q_i     = jnp.asarray(Q_all[SAMPLE_IDX])
    x_jax   = jnp.asarray(x)
    phi_0_i = phi_0_all[SAMPLE_IDX]

    phi_0_pred = onp.asarray(
        model.predict_phi0(model.params, Q_i[None, :], x_jax)[0]
    )

    # Source Q(x) on a secondary axis, kept behind the phi_0 curves.
    ax_Q = ax.twinx()
    ax_Q.plot(x, Q_all[SAMPLE_IDX], lw=1.2, color=col_Q, alpha=0.6)
    ax_Q.fill_between(x, 0, Q_all[SAMPLE_IDX], color=col_Q, alpha=0.10)
    ax_Q.set_ylabel("$Q(x)$", color=col_Q)
    ax_Q.tick_params(axis="y", labelcolor=col_Q)
    ax.set_zorder(ax_Q.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot(x, phi_0_i,    lw=2.0, color=col_true, label="true")
    ax.plot(x, phi_0_pred, lw=1.6, color=col_pred,
            linestyle="--", label="predicted")
    ax.set_title(label)
    ax.set_ylabel("$\\phi_0(x)$")

for ax in axes[-1, :]:
    ax.set_xlabel("x  [cm]")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
           bbox_to_anchor=(0.5, 1.02))

fig.tight_layout()
# plt.savefig(f"results/tanh_samples.pdf", bbox_inches="tight")
plt.show()