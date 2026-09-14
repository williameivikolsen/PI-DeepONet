import os
import sys
import fcntl
import subprocess
import time

# Run plainly, this script is the LAUNCHER: it starts one worker per GPU in
# CARDS, waits for them, and prints the results. It never trains itself, so it
# keeps JAX on the CPU. A worker is this same script, started by the launcher
# with SWEEP_LRS set and CUDA_VISIBLE_DEVICES pinned to one card.
_ENV0  = dict(os.environ)
WORKER = "SWEEP_LRS" in os.environ
if not WORKER:
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
    # "const_1e-2": lambda n: 1e-2,
    # "const_3e-3": lambda n: 3e-3,
    # "const_1e-3": lambda n: 1e-3,
    "const_3e-4": lambda n: 3e-4,
    "const_1e-4": lambda n: 1e-4,
    # "const_3e-5": lambda n: 3e-5,
    # "const_1e-5": lambda n: 1e-5,
    # "exp_1e-3_d0.9_report": lambda n: optax.exponential_decay(
    #     init_value=1e-3, transition_steps=max(n // 10, 1), decay_rate=0.9),
    # "exp_1e-2_d0.9": lambda n: optax.exponential_decay(
    #     init_value=1e-2, transition_steps=max(n // 10, 1), decay_rate=0.9),
    "exp_1e-3_d0.9_fast": lambda n: optax.exponential_decay(
        init_value=1e-3, transition_steps=max(n // 20, 1), decay_rate=0.9),
    # "exp_1e-3_d0.9_slow": lambda n: optax.exponential_decay(
    #     init_value=1e-3, transition_steps=max(n // 5, 1), decay_rate=0.9),
    "cosine_1e-3": lambda n: optax.cosine_decay_schedule(
        init_value=1e-3, decay_steps=n, alpha=0.0),
    # "cosine_1e-2": lambda n: optax.cosine_decay_schedule(
    #     init_value=1e-2, decay_steps=n, alpha=0.01),
    # "warmup_cosine_1e-2": lambda n: optax.warmup_cosine_decay_schedule(
    #     init_value=1e-5, peak_value=1e-2,
    #     warmup_steps=max(n // 20, 1), decay_steps=n, end_value=1e-5),
    # "step_1e-3": lambda n: optax.piecewise_constant_schedule(
    #     init_value=1e-3,
    #     boundaries_and_scales={int(0.5 * n): 0.1, int(0.75 * n): 0.1}),
    # "linear_1e-3": lambda n: optax.linear_schedule(
    #     init_value=1e-3, end_value=1e-5, transition_steps=n),
}

model_name   = "pideeponet_angular"
N_LAYERS     = 4
P_LATENT     = 100
BRANCH_WIDTH = 250
TRUNK_WIDTH  = 500
LAMBDA_DATA, LAMBDA_RES, LAMBDA_BCS = 0.7, 0.25, 0.05
N_PER_SAMPLE = 1000
branch_activation = "relu"   # unbounded -> extrapolates in source amplitude
trunk_activation  = "tanh"
sweep_name        = "B5000"   # new study when the training setup changes; editing LR_CANDIDATES needs none
CARDS             = [1, 2, 3, 4, 5, 6, 7]   # GPUs the launcher may use, one worker per card

STUDY_NAME = f"{branch_activation}_{trunk_activation}_{sweep_name}"
# Workers write to one SQLite file at the same time; a generous busy timeout makes
# a write wait for the lock instead of failing the trial with "database is locked".
STORAGE    = optuna.storages.RDBStorage(
    "sqlite:///activation_studies.db", engine_kwargs={"connect_args": {"timeout": 60}})
LOG_DIR    = "logs"

size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}

X_slab = 10.0
J      = int(ds['x'].shape[0])
A      = int(ds['mu_GL'].shape[0])
SIGMA_T, SIGMA_S0, SIGMA_S1 = 1.0, 0.5, 0.0

B      = 5000
N_ITER = 100000
LOG_EVERY = N_ITER // 100          # 100 validation points per trial
print(f"Batch size {B}, {N_ITER} iterations per trial")

data_in, data_out = build_psi_data_arrays(ds)
Q_shift, Q_scale = 0.0, 1.0
# Q_shift, Q_scale = 0.0, float(jnp.sqrt(jnp.mean(ds['Q'] ** 2)))   # for a bounded branch activation
print(f"Branch input: (Q - {Q_shift:.6f}) / {Q_scale:.6f}")
bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
val_batch = build_psi_val_batch(val_ds)

branch_layers = [J] + N_LAYERS * [BRANCH_WIDTH] + [P_LATENT]
trunk_layers  = [1] + N_LAYERS * [TRUNK_WIDTH]  + [A * P_LATENT]

# Weights of the best trial so far. Several workers write here, so whether to
# keep a trial is decided against this file itself, under a lock (see objective).
CKPT_PATH  = f"trained_models/lr_search/{size}/{model_name}_{branch_activation}_{trunk_activation}_B{B}.pkl"
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH, "rb") as f:
        print(f"Existing checkpoint {CKPT_PATH}: "
              f"val_ARE={float(pickle.load(f).get('val_ARE', float('inf'))):.3f}%")


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


def objective(trial, lr_name):
    # The learning rate is assigned by the launcher, not sampled, and recorded as
    # a user attribute. As a categorical parameter it tied each study to one
    # frozen list of candidates: after any edit to LR_CANDIDATES, Optuna refused
    # new trials ("CategoricalDistribution does not support dynamic value space").
    trial.set_user_attr("lr_config", lr_name)
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
        x_sensors=ds['x'], X=X_slab, Q_shift=Q_shift, Q_scale=Q_scale,
        lambda_data=LAMBDA_DATA, lambda_res=LAMBDA_RES, lambda_bcs=LAMBDA_BCS,
        lr_schedule=learning_rate,
        branch_activation=branch_activation,
        trunk_activation=trunk_activation,
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

    # Keep the weights of the best trial only. Workers on other cards write the
    # same CKPT_PATH, so compare against the checkpoint on disk while holding an
    # exclusive lock, not against this process's memory: otherwise a worker that
    # finishes later could overwrite a better checkpoint from another card.
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    with open(CKPT_PATH + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        best_on_disk = float("inf")
        if os.path.exists(CKPT_PATH):
            with open(CKPT_PATH, "rb") as f:
                best_on_disk = float(pickle.load(f).get("val_ARE", float("inf")))
        if val_ARE < best_on_disk and passes_guard(model):
            with open(CKPT_PATH + ".tmp", "wb") as f:
                pickle.dump({
                    "params": model.params,
                    "config": {
                        "model_type":    "angular_vec",
                        "branch_activation": model.branch_activation_name,
                        "trunk_activation":  model.trunk_activation_name,
                        "branch_layers": branch_layers,
                        "trunk_layers":  trunk_layers,
                        "N_angles":      A,
                        "Sigma_t":       SIGMA_T,
                        "Sigma_s0":      SIGMA_S0,
                        "Sigma_s1":      SIGMA_S1,
                        "x_sensors":     onp.asarray(ds['x']),
                        "X":             X_slab,
                        "Q_shift":       Q_shift,
                        "Q_scale":       Q_scale,
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
            os.replace(CKPT_PATH + ".tmp", CKPT_PATH)   # atomic: never a half-written file
            print(f"  new best: {lr_name} at {val_ARE:.3f}% -> saved {CKPT_PATH}")

    return val_ARE


def lr_of(trial):
    # Learning rate of a stored trial: a user attribute since the parallel sweep,
    # a sampled parameter in trials from before it.
    return trial.user_attrs.get("lr_config", trial.params.get("lr_config"))


if __name__ == "__main__":
    TS = optuna.trial.TrialState

    if WORKER:
        # Run the learning rates the launcher assigned to this card, one after
        # another, all in the shared study. A crashing trial is recorded as FAIL
        # and the worker moves on.
        lrs = os.environ["SWEEP_LRS"].split(",")
        print(f"Worker on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}: {lrs}")
        for lr in lrs:
            study = optuna.load_study(
                study_name=STUDY_NAME, storage=STORAGE,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=3,
                    n_warmup_steps=N_ITER // 5,   # let warmup schedules get going first
                    interval_steps=LOG_EVERY,
                ),
            )
            study.optimize(lambda trial: objective(trial, lr), n_trials=1, catch=(Exception,))
        sys.exit(0)

    # Launcher. A learning rate counts as done once it has a COMPLETE or PRUNED
    # trial. FAILed ones are run again (GridSampler used to treat them as done and
    # skip them). RUNNING ones are left alone, in case another sweep is still
    # working on them.
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE,
                                direction="minimize", load_if_exists=True)
    done    = {lr_of(t) for t in study.trials if t.state in (TS.COMPLETE, TS.PRUNED)}
    running = {lr_of(t) for t in study.trials if t.state == TS.RUNNING} - done - {None}
    todo    = [lr for lr in LR_CANDIDATES if lr not in done and lr not in running]
    print(f"\nStudy {STUDY_NAME}: done {sorted(done & set(LR_CANDIDATES))}, to run {todo}"
          + (f", skipping {sorted(running)} (marked RUNNING)" if running else ""))

    os.makedirs(LOG_DIR, exist_ok=True)
    workers = []
    for i, card in enumerate(CARDS):
        lrs = todo[i::len(CARDS)]            # round-robin over the cards
        if not lrs:
            continue
        log_path = f"{LOG_DIR}/{STUDY_NAME}_gpu{card}.log"
        env = {**_ENV0, "CUDA_VISIBLE_DEVICES": str(card),
               "SWEEP_LRS": ",".join(lrs), "PYTHONUNBUFFERED": "1"}
        log = open(log_path, "a")           # append: a relaunch must not erase a failed trial's traceback
        log.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')}: {', '.join(lrs)} =====\n")
        log.flush()
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        workers.append((card, proc))
        print(f"  GPU {card}: {', '.join(lrs):<40s} pid {proc.pid}   log {log_path}")
    for card, proc in workers:
        proc.wait()
        print(f"  GPU {card} worker exited with code {proc.returncode}")

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    print("\n--- Learning-rate sweep results (best validation ARE) ---")
    finished = [t for t in study.trials if t.state == TS.COMPLETE]
    for t in sorted(finished, key=lambda t: t.value):
        print(f"  {str(lr_of(t)):<24s} {t.value:8.3f}%")
    for t in study.trials:
        if t.state == TS.PRUNED:
            print(f"  {str(lr_of(t)):<24s}   pruned")
    failed = ({lr_of(t) for t in study.trials if t.state == TS.FAIL}
              - {lr_of(t) for t in study.trials if t.state in (TS.COMPLETE, TS.PRUNED)})
    for lr in sorted(x for x in failed if x):
        print(f"  {lr:<24s}   FAILED - launch again to retry (see its log)")

    if finished:
        print(f"\nBest learning rate: {lr_of(study.best_trial)}")
        print(f"Best validation ARE: {study.best_value:.3f}%")
    print(f"Best trial weights: {CKPT_PATH}")
