import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
from jax import vmap

# Helper functions for plotting etc

def plot_loss_curves(model,
                     log_every: int = 100,
                     figsize=(5, 7)):

    loss_log      = onp.asarray(model.loss_log)
    loss_data_log = onp.asarray(model.loss_data_log)
    loss_bcs_log  = onp.asarray(model.loss_bcs_log)
    loss_res_log  = onp.asarray(model.loss_res_log)

    iters = onp.arange(len(loss_log)) * log_every

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)


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
    # operator_net returns normalized psi (psi_tilde = psi/output_scale),
    # so the quadrature gives normalized phi_0. Multiply by output_scale
    # to return phi_0 in raw units for comparison against ds['phi_0'].
    def phi0_at(x_j):
        psi_vec = vmap(
            lambda mu_k: model.operator_net(model.params, Q, x_j, mu_k)
        )(model.mu_GL)
        return jnp.dot(model.w_GL, psi_vec)
    
    print("Output scale")
    print(model.output_scale)

    return model.output_scale * vmap(phi0_at)(x_points)

def plot_sample_predictions(model,
                            ds: dict,
                            sample_index: int = 0,
                            figsize=(7, 9)):
    # Pull arrays as numpy for matplotlib.
    Q_all     = onp.asarray(ds['Q'])          # (N, J)
    phi_0_all = onp.asarray(ds['phi_0'])      # (N, J)
    x         = onp.asarray(ds['x'])          # (J,)

    # Use the dataset's sensor grid for predictions so that the true and
    # predicted curves are compared at identical x-points.
    x_jax = jnp.asarray(x)

    palette = sns.color_palette("deep", 3)
    col_Q     = palette[0]
    col_true  = palette[1]
    col_pred  = palette[2]
    col_err   = sns.color_palette("flare", 4)[2]

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
    rel_l2 = float(onp.linalg.norm(phi_0_i - phi_0_pred)
                   / (onp.linalg.norm(phi_0_i) + 1e-12))
    ax = axes[2]
    ax.plot(x, abs_err, lw=2, color=col_err)
    ax.fill_between(x, 0, abs_err, color=col_err, alpha=0.20)
    ax.set_xlabel("x  [cm]")
    ax.set_ylabel("$|\\phi_0^{\\mathrm{true}} - \\phi_0^{\\mathrm{pred}}|$")
    ax.set_title(f"Absolute error  (rel $L_2$ = {rel_l2:.2e})")

    fig.tight_layout()
    return fig