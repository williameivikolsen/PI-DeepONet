import os
import pickle

import optax
import optuna
import jax.numpy as jnp
import numpy as onp
from jax import random

from model import (
    PI_DeepONet_Angular,
    DataGenerator,
    build_psi_data_arrays, build_psi_val_batch,
    build_bcs_arrays, build_res_arrays,
)

LR_CANDIDATES = {
    "const_1e-2": lambda n: 1e-2,
    "const_3e-3": lambda n: 3e-3,
    "const_1e-3": lambda n: 1e-3,      # what angular_training.py currently uses
    "const_3e-4": lambda n: 3e-4,
    "const_1e-4": lambda n: 1e-4,
    "exp_1e-3_d0.9_report": lambda n: optax.exponential_decay(
        init_value=1e-3, transition_steps=max(n // 10, 1), decay_rate=0.9),
    "exp_1e-2_d0.9": lambda n: optax.exponential_decay(
        init_value=1e-2, transition_steps=max(n // 10, 1), decay_rate=0.9),
    "exp_1e-3_d0.9_fast": lambda n: optax.exponential_decay(
        init_value=1e-3, transition_steps=max(n // 20, 1), decay_rate=0.9),
    "exp_1e-3_d0.9_slow": lambda n: optax.exponential_decay(
        init_value=1e-3, transition_steps=max(n // 5, 1), decay_rate=0.9),
    "cosine_1e-3": lambda n: optax.cosine_decay_schedule(
        init_value=1e-3, decay_steps=n, alpha=0.0),
    "cosine_1e-2": lambda n: optax.cosine_decay_schedule(
        init_value=1e-2, decay_steps=n, alpha=0.01),
    "warmup_cosine_1e-2": lambda n: optax.warmup_cosine_decay_schedule(
        init_value=1e-5, peak_value=1e-2,
        warmup_steps=max(n // 20, 1), decay_steps=n, end_value=1e-5),
    "step_1e-3": lambda n: optax.piecewise_constant_schedule(
        init_value=1e-3,
        boundaries_and_scales={int(0.5 * n): 0.1, int(0.75 * n): 0.1}),
    "linear_1e-3": lambda n: optax.linear_schedule(
        init_value=1e-3, end_value=1e-5, transition_steps=n),
}

model_name   = "pideeponet_angular"
N_LAYERS     = 4
P_LATENT     = 100
BRANCH_WIDTH = 250
TRUNK_WIDTH  = 500
LAMBDA_DATA, LAMBDA_RES, LAMBDA_BCS = 0.7, 0.25, 0.05
N_PER_SAMPLE = 1000

size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}

X_slab = 10.0
J      = int(ds['x'].shape[0])
A      = int(ds['mu_GL'].shape[0])
SIGMA_T, SIGMA_S0, SIGMA_S1 = 1.0, 0.5, 0.0

E      = 2000
B      = 1000
D      = len(ds["Q"]) * len(ds["x"])
N_ITER = int(D * E / B)
LOG_EVERY = N_ITER // 100          # 100 validation points per trial

# Data loss supervises the full psi vector at the GL nodes; validation ARE is
# still measured on phi_0, as in angular_training.py.
data_in, data_out, phi_scale = build_psi_data_arrays(ds, normalize=True)
bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
val_batch = build_psi_val_batch(val_ds, output_scale=phi_scale)

branch_layers = [J] + N_LAYERS * [BRANCH_WIDTH] + [P_LATENT]
trunk_layers  = [1] + N_LAYERS * [TRUNK_WIDTH]  + [A * P_LATENT]

# Weights of the best trial so far. Seeded from an existing checkpoint so a
# resumed sweep does not overwrite a better run.
CKPT_PATH  = f"trained_models/lr_search/tanh/{model_name}.pkl"
_incumbent = {"val_ARE": float("inf")}
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH, "rb") as f:
        _incumbent["val_ARE"] = float(pickle.load(f).get("val_ARE", float("inf")))
    print(f"Existing checkpoint {CKPT_PATH}: val_ARE={_incumbent['val_ARE']:.3f}%")


def passes_guard(model, tol=1.0):
    """
    Same save-time check as angular_training.py: the restored params must
    reproduce best_val_ARE on both the training metric and the predict_phi0
    evaluation path, and the flux must be non-negative.
    """
    are_valpath = float(model.val_ARE(model.params, val_batch))
    phi_pred = onp.asarray(model.predict_phi0(model.params, val_ds['Q'], val_ds['x']))
    phi_true = onp.asarray(val_ds['phi_0'])
    are_predpath = float(onp.mean(onp.abs((phi_true - phi_pred) / phi_true)) * 100.0)

    ok = (abs(are_valpath - model.best_val_ARE) <= tol
          and abs(are_predpath - model.best_val_ARE) <= tol
          and phi_pred.min() >= 0.0)
    if not ok:
        print(f"  guard failed — not saving (best={model.best_val_ARE:.3f}%, "
              f"val path={are_valpath:.3f}%, predict path={are_predpath:.3f}%, "
              f"min phi_0={phi_pred.min():.2f})")
    return ok


def objective(trial):
    lr_name = trial.suggest_categorical("lr_config", list(LR_CANDIDATES))
    learning_rate = LR_CANDIDATES[lr_name](N_ITER)

    data_dataset = DataGenerator(data_in, data_out, batch_size=B,
                                 rng_key=random.PRNGKey(101))
    bcs_dataset  = DataGenerator(bcs_in,  bcs_out,  batch_size=B,
                                 rng_key=random.PRNGKey(202), branch_table=bcs_Q)
    res_dataset  = DataGenerator(res_in,  res_out,  batch_size=B,
                                 rng_key=random.PRNGKey(303), branch_table=res_Q)

    model = PI_DeepONet_Angular(
        branch_layers, trunk_layers,
        N_angles=A,
        Sigma_t=SIGMA_T, Sigma_s0=SIGMA_S0, Sigma_s1=SIGMA_S1,
        x_sensors=ds['x'], X=X_slab,
        lambda_data=LAMBDA_DATA, lambda_res=LAMBDA_RES, lambda_bcs=LAMBDA_BCS,
        lr_schedule=learning_rate,
        output_scale=phi_scale,
        Q_ref=float(jnp.mean(ds['Q'])),
        activation="tanh",
        seed=1234,
    )

    def report_to_optuna(it, loss, loss_data, loss_bcs, loss_res, val_ARE):
        if val_ARE is None:
            return
        trial.report(val_ARE if onp.isfinite(val_ARE) else float("inf"), it)
        if trial.should_prune():
            raise optuna.TrialPruned()

    model.train(
        data_dataset, bcs_dataset, res_dataset,
        nIter=N_ITER, log_every=LOG_EVERY,
        val_batch=val_batch, val_every=LOG_EVERY,
        callback=report_to_optuna,
    )

    val_ARE = float(model.best_val_ARE)

    # Keep the weights of the best trial only
    if val_ARE < _incumbent["val_ARE"] and passes_guard(model):
        _incumbent["val_ARE"] = val_ARE
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        with open(CKPT_PATH, "wb") as f:
            pickle.dump({
                "params": model.params,
                "config": {
                    "model_type":    "angular_vec",
                    "activation":    model.activation_name,
                    "branch_layers": branch_layers,
                    "trunk_layers":  trunk_layers,
                    "N_angles":      A,
                    "Sigma_t":       SIGMA_T,
                    "Sigma_s0":      SIGMA_S0,
                    "Sigma_s1":      SIGMA_S1,
                    "x_sensors":     onp.asarray(ds['x']),
                    "X":             X_slab,
                    "output_scale":  phi_scale,
                    "Q_ref":         model.Q_ref,
                },
                "loss_log":      model.loss_log,
                "loss_data_log": model.loss_data_log,
                "loss_bcs_log":  model.loss_bcs_log,
                "loss_res_log":  model.loss_res_log,
                "val_ARE_log":   model.val_ARE_log,
                "val_iter_log":  model.val_iter_log,
                "n_iter":        N_ITER,
                "log_every":     LOG_EVERY,
                "model_name":    model_name,
                "lr_config":     lr_name,
                "val_ARE":       val_ARE,
                "best_val_ARE":  val_ARE,
                "best_val_iter": model.best_val_iter,
            }, f)
        print(f"  new best: {lr_name} at {val_ARE:.3f}% -> saved {CKPT_PATH}")

    return val_ARE


if __name__ == "__main__":
    study = optuna.create_study(
        storage=f"sqlite:///{model_name}_lr_search.db",
        study_name=f"{model_name}_tanh_lr_search",
        direction="minimize",
        sampler=optuna.samplers.GridSampler({"lr_config": list(LR_CANDIDATES)}),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=N_ITER // 5,   # let warmup schedules get going first
            interval_steps=LOG_EVERY,
        ),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=len(LR_CANDIDATES))

    print("\n--- Learning-rate sweep results (best validation ARE) ---")
    finished = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    for t in sorted(finished, key=lambda t: t.value):
        print(f"  {t.params['lr_config']:<24s} {t.value:8.3f}%")
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    for t in pruned:
        print(f"  {t.params['lr_config']:<24s}   pruned")

    print("\nBest params:", study.best_params)
    print(f"Best validation ARE: {study.best_value:.3f}%")
    print(f"Best trial weights: {CKPT_PATH}")
