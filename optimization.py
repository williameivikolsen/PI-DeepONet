import sys

import optax
import optuna
import jax.numpy as jnp
import numpy as onp
from jax import random

from nonPI_model import (
    DeepONet, DataGenerator,
    build_data_arrays, build_val_batch,
)

# ---------------------------------------------------------------------------
LR_CANDIDATES = {
    "const_1e-2": lambda n: 1e-2,
    "const_3e-3": lambda n: 3e-3,
    "const_1e-3": lambda n: 1e-3,      # what nonPI_training.py currently uses
    "const_3e-4": lambda n: 3e-4,
    "const_1e-4": lambda n: 1e-4,
    "exp_1e-3_d0.9_report": lambda n: optax.exponential_decay(
        init_value=1e-3, transition_steps=max(n // 20, 1), decay_rate=0.9),
    "exp_1e-2_d0.9": lambda n: optax.exponential_decay(
        init_value=1e-2, transition_steps=max(n // 20, 1), decay_rate=0.9),
    "exp_1e-3_d0.9_fast": lambda n: optax.exponential_decay(
        init_value=1e-3, transition_steps=max(n // 40, 1), decay_rate=0.9),
    "exp_1e-3_d0.9_slow": lambda n: optax.exponential_decay(
        init_value=1e-3, transition_steps=max(n // 10, 1), decay_rate=0.9),
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

arch = {"branch": [250, 250, 250, 250], "trunk": [500, 500, 500, 500]}
model_name = sys.argv[1] if len(sys.argv) > 1 else "large_deeponet"
P_LATENT   = 100

# ---------------------------------------------------------------------------
# Load datasets once outside the objective
# ---------------------------------------------------------------------------
size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}

X_slab = 10.0
J      = int(ds['x'].shape[0])

E      = 2000
B      = 1000
D      = len(ds["Q"]) * len(ds["x"])
N_ITER = int(D * E / B)
LOG_EVERY = N_ITER // 100          # 100 validation points per trial

# Data arrays are identical for every trial, so build them once.
data_in, data_out, phi_scale = build_data_arrays(ds, normalize=True)
val_batch = build_val_batch(val_ds, output_scale=phi_scale)


def objective(trial):
    lr_name = trial.suggest_categorical("lr_config", list(LR_CANDIDATES))
    learning_rate = LR_CANDIDATES[lr_name](N_ITER)

    # Architecture is fixed across the sweep; only the learning rate varies.
    branch_layers = [J] + arch["branch"] + [P_LATENT]
    trunk_layers  = [1] + arch["trunk"]  + [P_LATENT]

    data_dataset = DataGenerator(data_in, data_out, batch_size=B,
                                 rng_key=random.PRNGKey(101))

    model = DeepONet(
        branch_layers, trunk_layers,
        Sigma_t=1.0, Sigma_s0=0.5, Sigma_s1=0.0,
        x_sensors=ds['x'], X=X_slab,
        lr_schedule=learning_rate,
        output_scale=phi_scale,
        Q_ref=float(jnp.mean(ds['Q'])),
        activation="relu",
        seed=1234,
    )

    def report_to_optuna(it, loss, val_ARE):
        if val_ARE is None:
            return
        trial.report(val_ARE if onp.isfinite(val_ARE) else float("inf"), it)
        if trial.should_prune():
            raise optuna.TrialPruned()

    model.train(
        data_dataset,
        nIter=N_ITER, log_every=LOG_EVERY,
        val_batch=val_batch, val_every=LOG_EVERY,
        callback=report_to_optuna,
    )

    return float(model.best_val_ARE)


if __name__ == "__main__":
    study = optuna.create_study(
        storage=f"sqlite:///{model_name}_lr_search.db",
        study_name=f"{model_name}_lr_search",
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
