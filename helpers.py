import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns


def load_model(path: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = dict(ckpt["config"])
    cfg["x_sensors"] = jnp.asarray(cfg["x_sensors"])

    is_PI = "N_angles" in cfg
    if is_PI:
        # Distinguish the vector-output angular model from the scalar PI
        # model. Prefer an explicit marker written at training time; fall
        # back to the trunk input dimension (vector model's trunk takes x
        # only -> input dim 1; scalar PI's trunk takes (x, mu) -> dim 2).
        model_type = cfg.pop("model_type", None)
        trunk_in = cfg["trunk_layers"][0] if "trunk_layers" in cfg else 2
        is_angular_vec = (model_type == "angular_vec") or \
                         (model_type is None and trunk_in == 1)

        if is_angular_vec:
            from model import PI_DeepONet_Angular
            model = PI_DeepONet_Angular(**cfg)
            kind = "PI_angular"
        else:
            from model import PI_DeepONet
            model = PI_DeepONet(**cfg)
            kind = "PI"
    else:
        from nonPI_model import DeepONet
        # model_type, if present, is not a DeepONet constructor arg.
        cfg.pop("model_type", None)
        model = DeepONet(**cfg)
        kind = "nonPI"

    model.params   = ckpt["params"]
    model.loss_log = ckpt["loss_log"]
    if is_PI:
        for key in ("loss_data_log", "loss_bcs_log", "loss_res_log"):
            if key in ckpt:
                setattr(model, key, ckpt[key])
    return model, kind, ckpt


def r2_score(true: onp.ndarray, pred: onp.ndarray) -> float:
    ss_res = onp.sum((true - pred) ** 2)
    ss_tot = onp.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def are(true: onp.ndarray, pred: onp.ndarray) -> float:
    return float(onp.mean(onp.abs((true - pred) / true)) * 100.0)


def plot_loss_curves(model, kind: str = "PI",
                     log_every: int = 100,
                     figsize=(5, 7)):
    loss_log = onp.asarray(model.loss_log)
    iters    = onp.arange(len(loss_log)) * log_every

    if kind == "nonPI":
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] / 1.5))
        ax.plot(iters, loss_log, lw=2, color="black", label="Total")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (normalized)")
        ax.set_title("Training loss")
        ax.legend(frameon=True)
        fig.tight_layout()
        return fig

    loss_data_log = onp.asarray(model.loss_data_log)
    loss_bcs_log  = onp.asarray(model.loss_bcs_log)
    loss_res_log  = onp.asarray(model.loss_res_log)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(iters, loss_log, lw=2, color="black", label="Total")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss")
    axes[0].legend(frameon=True)

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


def plot_sample_predictions(model,
                            ds: dict,
                            sample_index: int = 0,
                            figsize=(7, 9)):
    Q_all     = onp.asarray(ds['Q'])
    phi_0_all = onp.asarray(ds['phi_0'])
    x         = onp.asarray(ds['x'])

    Q_i     = jnp.asarray(Q_all[sample_index])
    x_jax   = jnp.asarray(x)
    phi_0_i = phi_0_all[sample_index]

    phi_0_pred = onp.asarray(
        model.predict_phi0(model.params, Q_i[None, :], x_jax)[0]
    )

    palette  = sns.color_palette("deep", 3)
    col_Q    = palette[0]
    col_true = palette[1]
    col_pred = palette[2]
    col_err  = sns.color_palette("flare", 4)[2]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.plot(x, Q_all[sample_index], lw=2, color=col_Q)
    ax.fill_between(x, 0, Q_all[sample_index], color=col_Q, alpha=0.15)
    ax.set_ylabel("$Q(x)$")
    ax.set_title(f"Sample {sample_index}: source $Q(x)$")

    ax = axes[1]
    ax.plot(x, phi_0_i,    lw=2.0, color=col_true, label="true")
    ax.plot(x, phi_0_pred, lw=1.8, color=col_pred,
            linestyle="--", label="predicted")
    ax.set_ylabel("$\\phi_0(x)$")
    ax.set_title("Scalar flux")
    ax.legend(frameon=True, loc="best")

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