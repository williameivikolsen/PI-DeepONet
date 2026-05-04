import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
from jax import vmap

# Helper functions for plotting etc

def plot_loss_curves(model,
                     log_every: int = 100,
                     figsize=(11, 4.5)):

    loss_log      = onp.asarray(model.loss_log)
    loss_data_log = onp.asarray(model.loss_data_log)
    loss_bcs_log  = onp.asarray(model.loss_bcs_log)
    loss_res_log  = onp.asarray(model.loss_res_log)

    iters = onp.arange(len(loss_log)) * log_every

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    # Total loss
    axes[0].plot(iters, loss_log, lw=2, color="black", label="Total")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss")
    axes[0].legend(frameon=True)

    # Components
    palette = sns.color_palette("deep", 3)
    axes[1].plot(iters, loss_data_log, lw=2, color=palette[0], label="$L_{\\mathrm{data}}$")
    axes[1].plot(iters, loss_bcs_log,  lw=2, color=palette[1], label="$L_{\\mathrm{BC}}$")
    axes[1].plot(iters, loss_res_log,  lw=2, color=palette[2], label="$L_{\\mathrm{phys}}$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss components")
    axes[1].legend(frameon=True)

    fig.tight_layout()

    return fig

def predict_phi0(model, Q: jnp.ndarray, x_points: jnp.ndarray) -> jnp.ndarray:
    def phi0_at(x_j):
        psi_vec = vmap(
            lambda mu_k: model.operator_net(model.params, Q, x_j, mu_k)
        )(model.mu_GL)
        return jnp.dot(model.w_GL, psi_vec)

    return vmap(phi0_at)(x_points)

def plot_sample_predictions(model,
                            ds: dict,
                            sample_indices=None,
                            figsize=None):

    if sample_indices is None:
        sample_indices = [0, 1, 2]
    sample_indices = list(sample_indices)
    n_rows = len(sample_indices)

    if figsize is None:
        figsize = (12, 2.6 * n_rows + 0.4)

    # Pull arrays as numpy for matplotlib.
    Q_all     = onp.asarray(ds['Q'])          # (N, J)
    phi_0_all = onp.asarray(ds['phi_0'])      # (N, J)
    x         = onp.asarray(ds['x'])          # (J,)
    J         = x.shape[0]

    # Use the dataset's sensor grid for predictions so that the true and
    # predicted curves are compared at identical x-points.
    x_jax = jnp.asarray(x)

    palette = sns.color_palette("deep", 3)
    col_Q     = palette[0]
    col_true  = palette[1]
    col_pred  = palette[2]
    col_err   = sns.color_palette("flare", 4)[2]

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize, squeeze=False)

    for row, idx in enumerate(sample_indices):
        Q_i     = jnp.asarray(Q_all[idx])
        phi_0_i = phi_0_all[idx]

        phi_0_pred = onp.asarray(predict_phi0(model, Q_i, x_jax))

        # --- column 1: Q(x) -------------------------------------------
        ax = axes[row, 0]
        ax.plot(x, Q_all[idx], lw=2, color=col_Q)
        ax.fill_between(x, 0, Q_all[idx], color=col_Q, alpha=0.15)
        ax.set_xlabel("x  [cm]")
        ax.set_ylabel("$Q(x)$")
        ax.set_title(f"Sample {idx}: source $Q(x)$")

        # --- column 2: true vs predicted phi_0 ------------------------
        ax = axes[row, 1]
        ax.plot(x, phi_0_i,    lw=2.0, color=col_true, label="true")
        ax.plot(x, phi_0_pred, lw=1.8, color=col_pred,
                linestyle="--", label="predicted")
        ax.set_xlabel("x  [cm]")
        ax.set_ylabel("$\\phi_0(x)$")
        ax.set_title(f"Sample {idx}: scalar flux")
        ax.legend(frameon=True, loc="best")

        # --- column 3: absolute error ---------------------------------
        abs_err = onp.abs(phi_0_i - phi_0_pred)
        # Relative L2 error printed in the title.
        rel_l2 = float(onp.linalg.norm(phi_0_i - phi_0_pred)
                       / (onp.linalg.norm(phi_0_i) + 1e-12))
        ax = axes[row, 2]
        ax.plot(x, abs_err, lw=2, color=col_err)
        ax.fill_between(x, 0, abs_err, color=col_err, alpha=0.20)
        ax.set_xlabel("x  [cm]")
        ax.set_ylabel("$|\\phi_0^{\\mathrm{true}} - \\phi_0^{\\mathrm{pred}}|$")
        ax.set_title(f"Sample {idx}: abs err  (rel $L_2$ = {rel_l2:.2e})")

    fig.tight_layout()

    return fig

