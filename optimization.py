import optuna
import jax.numpy as jnp
import numpy as onp
from jax import random
from jax.nn import relu, tanh, gelu, softplus, elu

from model import (
    PI_DeepONet, DataGenerator, PI_DeepONet_Angular,
    build_psi_data_arrays, build_bcs_arrays, build_res_arrays,
    build_psi_val_batch,
)


SOFTPLUS_BETA = 5.0   # sharpness of the temperature-scaled softplus candidate

def sharpened_softplus(x, beta=SOFTPLUS_BETA):
    # (1/beta)*softplus(beta*x) -> exact ReLU as beta -> inf, while staying
    # C-infinity smooth for any finite beta (see plot_activations.py).
    return softplus(beta * x) / beta

ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "gelu": gelu,
    "softplus": softplus,
    "elu": elu,
    "softplus_beta5": sharpened_softplus,
}

LOSS_WEIGHTS = {
    "current_default": (0.25, 0.70, 0.05),
    "data_heavy":      (0.70, 0.25, 0.05),
    "balanced":        (0.50, 0.45, 0.05),
    "residual_heavy":  (0.10, 0.85, 0.05),
    "no_data":         (0.00, 0.90, 0.10),
    "bc_heavy":        (0.25, 0.55, 0.20),
}

# ---------------------------------------------------------------------------
# Load datasets once outside the objective
# ---------------------------------------------------------------------------
size = "small"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}

X_slab = 10.0
J      = int(ds['x'].shape[0])


def objective(trial):
    # Suggest hyperparameters
    branch_width  = trial.suggest_categorical("branch_width", [50, 100, 250])
    trunk_width   = trial.suggest_categorical("trunk_width", [100, 250, 500])
    n_per_sample  = trial.suggest_categorical("n_per_sample", [500, 750, 1000])
    activation_name = trial.suggest_categorical("activation", list(ACTIVATIONS))
    activation    = ACTIVATIONS[activation_name]
    loss_weights_name = trial.suggest_categorical("loss_weights", list(LOSS_WEIGHTS))
    lambda_data, lambda_res, lambda_bcs = LOSS_WEIGHTS[loss_weights_name]
    n_layers      = trial.suggest_int("n_layers", 2, 6)
    n_iter_trial  = 20000   # shorter than the full 100k for tractable search
    lr_transition_steps = trial.suggest_categorical(
        "lr_transition_steps", [n_iter_trial // 20, n_iter_trial // 10, n_iter_trial // 5]
    )

    # Vector-output angular model: trunk takes x only (input dim 1) and
    # emits A*p outputs, reshaped to (A, p) inside the model. p is the shared
    # latent width = branch's final width (the [p_latent] tail below).
    A_angles = 16
    p_latent = 100
    branch_layers = [J] + [branch_width] * n_layers + [p_latent]
    trunk_layers  = [1] + [trunk_width]  * n_layers + [A_angles * p_latent]

    # Build data
    data_in, data_out, phi_scale = build_psi_data_arrays(ds, normalize=True)
    bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=n_per_sample)
    res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=n_per_sample)

    data_dataset = DataGenerator(data_in, data_out, batch_size=1000,
                                 rng_key=random.PRNGKey(101))
    bcs_dataset  = DataGenerator(bcs_in,  bcs_out,  batch_size=1000,
                                 rng_key=random.PRNGKey(202), branch_table=bcs_Q)
    res_dataset  = DataGenerator(res_in,  res_out,  batch_size=1000,
                                 rng_key=random.PRNGKey(303), branch_table=res_Q)

    val_batch = build_psi_val_batch(val_ds, output_scale=phi_scale)

    # Build model
    model = PI_DeepONet(
        branch_layers, trunk_layers,
        N_angles=A_angles,
        Sigma_t=1.0, Sigma_s0=0.5, Sigma_s1=0.0,
        x_sensors=ds['x'], X=X_slab,
        lambda_data=lambda_data, lambda_res=lambda_res, lambda_bcs=lambda_bcs,
        output_scale=phi_scale,
        lr_transition_steps=lr_transition_steps,
        activation=activation
    )

    # Train
    model.train(
        data_dataset, bcs_dataset, res_dataset,
        nIter=n_iter_trial, log_every=500,
        val_batch=val_batch, val_every=500,
    )

    # Best validation ARE achieved across training (not final-iter ARE):
    # rewards configs that reach good generalization at SOME point.
    return float(model.best_val_ARE)


if __name__ == "__main__":
    study = optuna.create_study(
        storage=f"sqlite:///pi_deeponet_final.db",
        study_name=f"pi_deeponet_final",
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=30)
    print("\nBest params:", study.best_params)
    print(f"Best validation ARE: {study.best_value:.3f}%")