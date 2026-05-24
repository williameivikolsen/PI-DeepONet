import pickle
import numpy as onp
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import seaborn as sns

from nonPI_model import DeepONet


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
with open("trained_models/sahadath.pkl", "rb") as f:
    ckpt = pickle.load(f)

cfg = ckpt["config"]
cfg["x_sensors"] = jnp.asarray(cfg["x_sensors"])

model = DeepONet(**cfg)
model.params   = ckpt["params"]
model.loss_log = ckpt["loss_log"]


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------
def plot_loss_curve(model, log_every: int = 100, figsize=(6, 4.5)):
    loss_log = onp.asarray(model.loss_log)
    iters    = onp.arange(len(loss_log)) * log_every

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iters, loss_log, lw=2, color="black", label="Total")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (normalized)")
    ax.set_title("Training loss")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


fig1 = plot_loss_curve(model, log_every=ckpt["log_every"])


# ---------------------------------------------------------------------------
# Sample predictions
# ---------------------------------------------------------------------------
def predict_phi0(model, Q: jnp.ndarray, x_points: jnp.ndarray) -> jnp.ndarray:
    """
    Non-PI DeepONet predicts phi_0 directly (no GL quadrature step).
    predict_s already multiplies by output_scale internally, so callers
    get raw phi_0 in physical units.
    """
    # Q is a single source vector (J,); broadcast it across all x points.
    Q_batch = jnp.broadcast_to(Q, (x_points.shape[0], Q.shape[0]))
    return model.predict_s(model.params, Q_batch, x_points)


def plot_sample_predictions(model, ds, sample_index=0, figsize=(7, 9)):
    Q_all     = onp.asarray(ds['Q'])
    phi_0_all = onp.asarray(ds['phi_0'])
    x         = onp.asarray(ds['x'])

    x_jax = jnp.asarray(x)

    palette  = sns.color_palette("deep", 3)
    col_Q    = palette[0]
    col_true = palette[1]
    col_pred = palette[2]
    col_err  = sns.color_palette("flare", 4)[2]

    Q_i     = jnp.asarray(Q_all[sample_index])
    phi_0_i = phi_0_all[sample_index]

    phi_0_pred = onp.asarray(predict_phi0(model, Q_i, x_jax))

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # --- row 1: Q(x) -----------------------------------------------------
    ax = axes[0]
    ax.plot(x, Q_all[sample_index], lw=2, color=col_Q)
    ax.fill_between(x, 0, Q_all[sample_index], color=col_Q, alpha=0.15)
    ax.set_ylabel("$Q(x)$")
    ax.set_title(f"Sample {sample_index}: source $Q(x)$")

    # --- row 2: true vs predicted phi_0 ----------------------------------
    ax = axes[1]
    ax.plot(x, phi_0_i,    lw=2.0, color=col_true, label="true")
    ax.plot(x, phi_0_pred, lw=1.8, color=col_pred,
            linestyle="--", label="predicted")
    ax.set_ylabel("$\\phi_0(x)$")
    ax.set_title("Scalar flux")
    ax.legend(frameon=True, loc="best")

    # --- row 3: absolute error -------------------------------------------
    abs_err = onp.abs(phi_0_i - phi_0_pred)
    rel_l2  = float(onp.linalg.norm(phi_0_i - phi_0_pred)
                    / (onp.linalg.norm(phi_0_i) + 1e-12))
    ax = axes[2]
    ax.plot(x, abs_err, lw=2, color=col_err)
    ax.fill_between(x, 0, abs_err, color=col_err, alpha=0.20)
    ax.set_xlabel("x  [cm]")
    ax.set_ylabel("$|\\phi_0^{\\mathrm{true}} - \\phi_0^{\\mathrm{pred}}|$")
    ax.set_title(f"Absolute error  (rel $L_2$ = {rel_l2:.2e})")

    fig.tight_layout()
    return fig


ds_np = onp.load("datasets/test/M_Iso_test_NS.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files
         if ds_np[k].dtype.kind in "fiu"}   # float, int, unsigned int only

sample_index = 0

fig2 = plot_sample_predictions(model, ds, sample_index=sample_index)

plt.show()