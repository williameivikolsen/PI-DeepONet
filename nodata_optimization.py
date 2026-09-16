import os
import sys
import math
import fcntl
import subprocess
import time

_ENV0  = dict(os.environ)
WORKER = "SWEEP_LRS" in os.environ
if not WORKER:
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import pickle

import optax
import optuna
import jax.numpy as jnp
import numpy as onp
from jax import random
from jax.flatten_util import ravel_pytree

from model import (
    PI_DeepONet_Angular,
    DataGenerator,
    build_psi_data_arrays, build_psi_val_batch,
    build_bcs_arrays, build_res_arrays,
)

model_name   = "pideeponet_angular"
N_PER_SAMPLE = 1000
sweep_name   = "nodata"     # new study when the training setup changes
CARDS        = [0, 1, 2, 3, 4, 5, 6, 7]   # GPUs the launcher may use, one worker per card

# Peak learning rates, one per card, spanning two decades around 1.19e-4 (the best
# rate WITH data). Each is run with the warmup + cosine schedule below, so every
# trial ends annealed rather than bouncing around the minimum at a constant rate.
LR_PEAKS     = [2e-5, 4e-5, 8e-5, 1.2e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
WARMUP_STEPS = 2000         # Adam's gradient averages rebuild over ~1000 steps
END_FRACTION = 0.01         # final rate = peak * END_FRACTION

# Architecture of the best data-trained model. Only its SHAPE is reused: the
# weights are initialised fresh, or the model would inherit what that one learned
# from labelled data and this would not be a no-data experiment.
ARCH_CKPT = "trained_models/lr_search/large/pideeponet_angular_relu_tanh_arch_continued_annealed.pkl"
SEED      = 1234

STUDY_NAME = f"relu_tanh_{sweep_name}"
# Workers write to one SQLite file at the same time; a generous busy timeout makes
# a write wait for the lock instead of failing the trial with "database is locked".
STORAGE = optuna.storages.RDBStorage(
    "sqlite:///activation_studies.db", engine_kwargs={"connect_args": {"timeout": 60}})
LOG_DIR = "logs"

size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

with open(ARCH_CKPT, "rb") as f:
    _arch = pickle.load(f)
arch_cfg, arch_hp = _arch["config"], _arch["hyperparameters"]
branch_layers     = list(arch_cfg["branch_layers"])
trunk_layers      = list(arch_cfg["trunk_layers"])
branch_activation = arch_cfg["branch_activation"]
trunk_activation  = arch_cfg["trunk_activation"]
Q_shift, Q_scale  = arch_cfg["Q_shift"], arch_cfg["Q_scale"]
X_slab = float(arch_cfg["X"])
A      = int(arch_cfg["N_angles"])
SIGMA_T, SIGMA_S0, SIGMA_S1 = arch_cfg["Sigma_t"], arch_cfg["Sigma_s0"], arch_cfg["Sigma_s1"]
J      = int(ds['x'].shape[0])

# No supervised data term. The residual and boundary weights keep the ratio tuned
# by the architecture search; with lambda_data = 0 only their ratio matters, since
# Adam is invariant to the overall scale of the loss.
LAMBDA_DATA = 0.0
LAMBDA_RES  = arch_hp["res_over_data"]
LAMBDA_BCS  = arch_hp["bcs_over_data"]

# Selection set = validation set + shift-validation set, 50 sources each. These
# labels are used ONLY to rank, prune and checkpoint trials, never in the loss.
val_np   = onp.load("datasets/M_Iso_val.npz")
shift_np = onp.load("datasets/M_Iso_shiftval.npz")
assert onp.allclose(val_np["x"], shift_np["x"]), "validation sets must share the x grid"
val_ds   = {k: jnp.asarray(val_np[k])   for k in ("Q", "phi_0", "x")}
shift_ds = {k: jnp.asarray(shift_np[k]) for k in ("Q", "phi_0", "x")}
sel_ds   = {"Q":     jnp.concatenate([val_ds["Q"],     shift_ds["Q"]]),
            "phi_0": jnp.concatenate([val_ds["phi_0"], shift_ds["phi_0"]]),
            "x":     val_ds["x"]}

B      = 5000
N_ITER = 300000
LOG_EVERY = N_ITER // 100          # 100 selection-set evaluations per trial
print(f"Batch size {B}, {N_ITER} iterations per trial, lambda_data = {LAMBDA_DATA} (no data loss)")
print(f"Architecture from {ARCH_CKPT.split('/')[-1]}: branch {branch_layers}, trunk {trunk_layers}")

# train() always takes a data batch, so it gets the smallest possible one: the term
# is multiplied by lambda_data = 0 and contributes nothing to the gradient, and at
# one point per step it costs nothing either.
B_DATA = 1
data_in, data_out = build_psi_data_arrays(ds)
print(f"Branch input: (Q - {Q_shift:.6f}) / {Q_scale:.6f}")
bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
sel_batch   = build_psi_val_batch(sel_ds)
val_batch   = build_psi_val_batch(val_ds)
shift_batch = build_psi_val_batch(shift_ds)

# Weights of the best trial of this study. Several workers write here, so whether
# to keep a trial is decided against this file itself, under a lock (see objective).
CKPT_PATH = f"trained_models/lr_search/{size}/{model_name}_{STUDY_NAME}.pkl"
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH, "rb") as f:
        print(f"Existing checkpoint {CKPT_PATH}: "
              f"val_ARE={float(pickle.load(f).get('val_ARE', float('inf'))):.3f}%")


def passes_guard(model, tol=1.0):
    """
    Same save-time check as angular_optimization.py, on the selection set: the
    restored params must reproduce best_val_ARE on both the training metric and
    the predict_phi0 evaluation path, and the flux must be non-negative.
    """
    are_valpath = float(model.val_ARE(model.params, sel_batch))
    phi_pred = onp.asarray(model.predict_phi0(model.params, sel_ds['Q'], sel_ds['x']))
    phi_true = onp.asarray(sel_ds['phi_0'])
    are_predpath = float(onp.mean(onp.abs((phi_true - phi_pred) / phi_true)) * 100.0)

    ok = (abs(are_valpath - model.best_val_ARE) <= tol
          and abs(are_predpath - model.best_val_ARE) <= tol
          and phi_pred.min() >= 0.0)
    if not ok:
        print(f"  guard failed — not saving (best={model.best_val_ARE:.3f}%, "
              f"val path={are_valpath:.3f}%, predict path={are_predpath:.3f}%, "
              f"min phi_0={phi_pred.min():.2f})")
    return ok


def objective(trial, lr_peak):
    # The learning rate is assigned by the launcher, not sampled, and recorded as a
    # user attribute: as a categorical parameter it would freeze the study to one
    # fixed list of rates ("CategoricalDistribution does not support dynamic value
    # space" as soon as LR_PEAKS is edited).
    trial.set_user_attr("lr_peak", lr_peak)
    trial.set_user_attr("seed", SEED)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_peak, warmup_steps=WARMUP_STEPS,
        decay_steps=N_ITER, end_value=lr_peak * END_FRACTION)
    lr_config = f"warmup_cosine_{lr_peak:.1e}_to_{lr_peak * END_FRACTION:.1e}"
    print(f"\nTrial {trial.number}: {lr_config}, lambda data/res/bcs = "
          f"{LAMBDA_DATA} / {LAMBDA_RES:.4f} / {LAMBDA_BCS:.4f}", flush=True)

    data_dataset = DataGenerator(data_in, data_out, batch_size=B_DATA,
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
        lr_schedule=lr_schedule,
        branch_activation=branch_activation,
        trunk_activation=trunk_activation,
        seed=SEED,
    )
    trial.set_user_attr("n_params", int(ravel_pytree(model.params)[0].size))

    def report_to_optuna(it, loss, loss_data, loss_bcs, loss_res, val_ARE):
        if val_ARE is None:
            return
        trial.report(val_ARE if onp.isfinite(val_ARE) else float("inf"), it)
        if trial.should_prune():
            raise optuna.TrialPruned()

    model.train(
        data_dataset, bcs_dataset, res_dataset,
        nIter=N_ITER, log_every=LOG_EVERY,
        val_batch=sel_batch, val_every=LOG_EVERY,
        callback=report_to_optuna,
    )

    # Ranked by the median of the last 10 selection-set readings rather than the
    # single best one: the minimum of a noisy curve rewards a lucky reading. The
    # checkpoint still keeps the best parameters.
    score = float(onp.median(model.val_ARE_log[-10:]))
    if not math.isfinite(score):
        score = float("inf")
    best      = float(model.best_val_ARE)
    val_are   = float(model.val_ARE(model.params, val_batch))       # restored best params
    shift_are = float(model.val_ARE(model.params, shift_batch))
    trial.set_user_attr("best_sel_ARE", best)
    trial.set_user_attr("val_ARE", val_are)
    trial.set_user_attr("shift_ARE", shift_are)
    trial.set_user_attr("best_iter", int(model.best_val_iter))

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
        if best < best_on_disk and passes_guard(model):
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
                    "loss_data_log": model.loss_data_log,   # computed for logging only, weight 0
                    "loss_bcs_log":  model.loss_bcs_log,
                    "loss_res_log":  model.loss_res_log,
                    "val_ARE_log":   model.val_ARE_log,      # selection set (val + shift-val)
                    "val_iter_log":  model.val_iter_log,
                    "n_iter":        N_ITER,
                    "log_every":     LOG_EVERY,
                    "model_name":    model_name,
                    "lr_config":     lr_config,
                    "hyperparameters": {"lr": lr_peak, "warmup_steps": WARMUP_STEPS,
                                        "end_fraction": END_FRACTION, "seed": SEED,
                                        "lambda_data": LAMBDA_DATA,
                                        "res_over_data": LAMBDA_RES,
                                        "bcs_over_data": LAMBDA_BCS},
                    "architecture_from": ARCH_CKPT,
                    "study":         STUDY_NAME,
                    "trial":         trial.number,
                    "val_ARE":       best,                   # selection set; what checkpoints are compared on
                    "best_val_ARE":  best,
                    "best_val_iter": model.best_val_iter,
                    "valset_ARE":    val_are,
                    "shiftval_ARE":  shift_are,
                    "score":         score,
                }, f)
            os.replace(CKPT_PATH + ".tmp", CKPT_PATH)   # atomic: never a half-written file
            print(f"  new best: trial {trial.number} ({lr_config}) at {best:.3f}% -> saved {CKPT_PATH}")

    return score


def lr_of(trial):
    return trial.user_attrs.get("lr_peak")


if __name__ == "__main__":
    TS = optuna.trial.TrialState

    if WORKER:
        # Run the learning rates the launcher assigned to this card, one after
        # another, all in the shared study. A crashing trial is recorded as FAIL
        # and the worker moves on.
        lrs = [float(x) for x in os.environ["SWEEP_LRS"].split(",")]
        print(f"Worker on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}: {lrs}")
        for lr in lrs:
            study = optuna.load_study(
                study_name=STUDY_NAME, storage=STORAGE,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=N_ITER // 5,
                                                   interval_steps=LOG_EVERY),
            )
            study.optimize(lambda trial: objective(trial, lr), n_trials=1, catch=(Exception,))
        sys.exit(0)

    # ------------------------------- launcher -------------------------------
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE,
                                direction="minimize", load_if_exists=True)
    finished = [t for t in study.trials if t.state in (TS.COMPLETE, TS.PRUNED)]
    # A checkpoint without any finished trial behind it belongs to an earlier run
    # (for instance after the study was reset) and would silently outrank this one.
    if not finished and os.path.exists(CKPT_PATH):
        sys.exit(f"\n{CKPT_PATH} already exists, but study {STUDY_NAME} has no finished trials:\n"
                 f"those weights come from an earlier run and would silently outrank this one.\n"
                 f"Rename or remove the file (or change sweep_name), then relaunch.")

    # A rate counts as done once it has a COMPLETE or PRUNED trial; FAILed ones are
    # run again, RUNNING ones left alone in case another sweep is still on them.
    done    = {lr_of(t) for t in study.trials if t.state in (TS.COMPLETE, TS.PRUNED)}
    running = {lr_of(t) for t in study.trials if t.state == TS.RUNNING} - done - {None}
    todo    = [lr for lr in LR_PEAKS if lr not in done and lr not in running]
    print(f"\nStudy {STUDY_NAME}: done {sorted(done & set(LR_PEAKS))}, to run {todo}"
          + (f", skipping {sorted(running)} (marked RUNNING)" if running else ""))

    os.makedirs(LOG_DIR, exist_ok=True)
    workers = []
    for i, card in enumerate(CARDS):
        mine = todo[i::len(CARDS)]           # round-robin over the cards
        if not mine:
            continue
        label = ", ".join(f"{lr:.1e}" for lr in mine)
        log_path = f"{LOG_DIR}/{STUDY_NAME}_gpu{card}.log"
        env = {**_ENV0, "CUDA_VISIBLE_DEVICES": str(card), "PYTHONUNBUFFERED": "1",
               "SWEEP_LRS": ",".join(repr(lr) for lr in mine)}
        log = open(log_path, "a")           # append: a relaunch must not erase a failed trial's traceback
        log.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')}: {label} =====\n")
        log.flush()
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        workers.append((card, proc))
        print(f"  GPU {card}: peak lr {label:<28s} pid {proc.pid}   log {log_path}")
    for card, proc in workers:
        proc.wait()
        print(f"  GPU {card} worker exited with code {proc.returncode}")

    # ------------------------------- summary --------------------------------
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    count = {s: sum(t.state == s for t in study.trials) for s in (TS.COMPLETE, TS.PRUNED, TS.FAIL, TS.RUNNING)}
    print(f"\n--- {STUDY_NAME} (physics only, no data loss): {count[TS.COMPLETE]} complete, "
          f"{count[TS.PRUNED]} pruned, {count[TS.FAIL]} failed, {count[TS.RUNNING]} marked running ---")
    print(f"  {'peak lr':>9s} {'score':>7s} {'best':>7s} {'val':>7s} {'shift':>7s}")
    for t in sorted((t for t in study.trials if t.state == TS.COMPLETE), key=lambda t: t.value):
        u = t.user_attrs
        print(f"  {lr_of(t):9.1e} {t.value:7.3f} {u.get('best_sel_ARE', float('nan')):7.3f} "
              f"{u.get('val_ARE', float('nan')):7.3f} {u.get('shift_ARE', float('nan')):7.3f}")
    for t in study.trials:
        if t.state == TS.PRUNED:
            print(f"  {lr_of(t):9.1e}   pruned")
    failed = ({lr_of(t) for t in study.trials if t.state == TS.FAIL}
              - {lr_of(t) for t in study.trials if t.state in (TS.COMPLETE, TS.PRUNED)})
    for lr in sorted(x for x in failed if x):
        print(f"  {lr:9.1e}   FAILED - launch again to retry (see its log)")
    print("  score = median of the last 10 selection-set ARE readings (%); best / val / shift = ARE of the")
    print("  restored best parameters on the selection set / validation set / shift-validation set")
    print(f"\nBest weights of this study: {CKPT_PATH}")
