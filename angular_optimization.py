import os
import sys
import json
import math
import fcntl
import subprocess
import time

# Run plainly, this script is the LAUNCHER: it starts one worker per GPU in
# CARDS, waits for them, and prints the results. It never trains itself, so it
# keeps JAX on the CPU. A worker is this same script, started by the launcher
# with SWEEP_MODE set and CUDA_VISIBLE_DEVICES pinned to one card.
#
#   pinn/bin/python angular_optimization.py          architecture search (TPE)
#   pinn/bin/python angular_optimization.py reseed   re-run the best configurations with new seeds
_ENV0  = dict(os.environ)
WORKER = "SWEEP_MODE" in os.environ
MODE   = os.environ.get("SWEEP_MODE") or (sys.argv[1] if len(sys.argv) > 1 else "search")
if MODE not in ("search", "reseed"):
    sys.exit(f"Unknown mode {MODE!r}: run with no argument (search) or with 'reseed'.")
if not WORKER:
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import pickle
import warnings

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

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

model_name   = "pideeponet_angular"
P_LATENT     = 100
N_PER_SAMPLE = 1000
branch_activation = "relu"   # unbounded -> extrapolates in source amplitude
trunk_activation  = "tanh"
sweep_name        = "arch"   # new study when the training setup changes
CARDS             = [1, 2, 3, 4, 5, 6, 7]   # GPUs the launcher may use, one worker per card

# Search space. Every parameter is an int or a float, so a range can be edited
# later without starting a new study (Optuna freezes only the choices of
# CATEGORICAL parameters). Widths are searched as powers of two. Only the RATIOS
# of the loss weights matter, since Adam is invariant to the overall scale of the
# loss, so lambda_data is fixed at 1. The old default 0.7/0.25/0.05 is 1/0.36/0.07.
N_LAYERS      = (3, 6)
BRANCH_LOG2   = (7, 9)          # branch width 128 .. 512
TRUNK_LOG2    = (8, 10)         # trunk width  256 .. 1024
LR            = (5e-5, 5e-4)    # constant learning rate, log scale
RES_OVER_DATA = (0.1, 3.0)      # lambda_res / lambda_data, log scale
BCS_OVER_DATA = (0.01, 0.3)     # lambda_bcs / lambda_data, log scale

N_TRIALS  = 50                   # finished (COMPLETE or PRUNED) trials the search aims for
N_STARTUP = 14                   # random trials before TPE takes over: two waves on 7 cards
TOP_K, RESEED_SEEDS = 5, (1, 2)  # reseed mode: best TOP_K configurations, each with these seeds

STUDY_NAME  = f"{branch_activation}_{trunk_activation}_{sweep_name}"
RESEED_NAME = STUDY_NAME + "_reseed"
THIS_STUDY  = RESEED_NAME if MODE == "reseed" else STUDY_NAME
# Workers write to one SQLite file at the same time; a generous busy timeout makes
# a write wait for the lock instead of failing the trial with "database is locked".
STORAGE = optuna.storages.RDBStorage(
    "sqlite:///activation_studies.db", engine_kwargs={"connect_args": {"timeout": 60}})
LOG_DIR = "logs"

size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

# Selection set = validation set + shift-validation set, 50 sources each, so both
# weigh equally in the ARE that trials are ranked, pruned and checkpointed on. The
# validation set matches the training distribution; the shift-validation set
# (shift_validation_data_generator.py) holds rougher and smoother sources, from GRF
# parameters that are not any test scenario.
val_np   = onp.load("datasets/M_Iso_val.npz")
shift_np = onp.load("datasets/M_Iso_shiftval.npz")
assert onp.allclose(val_np["x"], shift_np["x"]), "validation sets must share the x grid"
val_ds   = {k: jnp.asarray(val_np[k])   for k in ("Q", "phi_0", "x")}
shift_ds = {k: jnp.asarray(shift_np[k]) for k in ("Q", "phi_0", "x")}
sel_ds   = {"Q":     jnp.concatenate([val_ds["Q"],     shift_ds["Q"]]),
            "phi_0": jnp.concatenate([val_ds["phi_0"], shift_ds["phi_0"]]),
            "x":     val_ds["x"]}

X_slab = 10.0
J      = int(ds['x'].shape[0])
A      = int(ds['mu_GL'].shape[0])
SIGMA_T, SIGMA_S0, SIGMA_S1 = 1.0, 0.5, 0.0

B      = 5000
N_ITER = 100000
LOG_EVERY = N_ITER // 100          # 100 selection-set evaluations per trial
print(f"Batch size {B}, {N_ITER} iterations per trial")

data_in, data_out = build_psi_data_arrays(ds)
Q_shift, Q_scale = 0.0, 1.0
# Q_shift, Q_scale = 0.0, float(jnp.sqrt(jnp.mean(ds['Q'] ** 2)))   # for a bounded branch activation
print(f"Branch input: (Q - {Q_shift:.6f}) / {Q_scale:.6f}")
bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=N_PER_SAMPLE)
sel_batch   = build_psi_val_batch(sel_ds)
val_batch   = build_psi_val_batch(val_ds)
shift_batch = build_psi_val_batch(shift_ds)

# Weights of the best trial of THIS study. Several workers write here, so whether
# to keep a trial is decided against this file itself, under a lock (see objective).
CKPT_PATH = f"trained_models/lr_search/{size}/{model_name}_{THIS_STUDY}.pkl"


def passes_guard(model, tol=1.0):
    """
    Same save-time check as angular_training.py, on the selection set: the
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


def objective(trial, seed=1234):
    n_layers  = trial.suggest_int("n_layers", *N_LAYERS)
    b_width   = 2 ** trial.suggest_int("branch_log2", *BRANCH_LOG2)
    t_width   = 2 ** trial.suggest_int("trunk_log2", *TRUNK_LOG2)
    lr        = trial.suggest_float("lr", *LR, log=True)
    res_ratio = trial.suggest_float("res_over_data", *RES_OVER_DATA, log=True)
    bcs_ratio = trial.suggest_float("bcs_over_data", *BCS_OVER_DATA, log=True)
    trial.set_user_attr("seed", seed)

    branch_layers = [J] + n_layers * [b_width] + [P_LATENT]
    trunk_layers  = [1] + n_layers * [t_width] + [A * P_LATENT]
    print(f"\nTrial {trial.number}: {n_layers} layers, branch {b_width}, trunk {t_width}, lr {lr:.2e}, "
          f"res/data {res_ratio:.3f}, bcs/data {bcs_ratio:.3f}, seed {seed}", flush=True)

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
        lambda_data=1.0, lambda_res=res_ratio, lambda_bcs=bcs_ratio,
        lr_schedule=lr,
        branch_activation=branch_activation,
        trunk_activation=trunk_activation,
        seed=seed,
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

    # Trials are RANKED by the median of their last 10 selection-set readings, not
    # by the single best one: two runs of one configuration have landed 0.15
    # points apart, and the minimum of a noisy curve rewards a lucky reading. The
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
                    "loss_data_log": model.loss_data_log,
                    "loss_bcs_log":  model.loss_bcs_log,
                    "loss_res_log":  model.loss_res_log,
                    "val_ARE_log":   model.val_ARE_log,      # selection set (val + shift-val)
                    "val_iter_log":  model.val_iter_log,
                    "n_iter":        N_ITER,
                    "log_every":     LOG_EVERY,
                    "model_name":    model_name,
                    "lr_config":     f"const_{lr:.1e}",
                    "hyperparameters": {**trial.params, "seed": seed},
                    "study":         THIS_STUDY,
                    "trial":         trial.number,
                    "val_ARE":       best,                   # selection set; what checkpoints are compared on
                    "best_val_ARE":  best,
                    "best_val_iter": model.best_val_iter,
                    "valset_ARE":    val_are,
                    "shiftval_ARE":  shift_are,
                    "score":         score,
                }, f)
            os.replace(CKPT_PATH + ".tmp", CKPT_PATH)   # atomic: never a half-written file
            print(f"  new best: trial {trial.number} at {best:.3f}% -> saved {CKPT_PATH}")

    return score


def launch(card, extra_env, label):
    log_path = f"{LOG_DIR}/{THIS_STUDY}_gpu{card}.log"
    env = {**_ENV0, "CUDA_VISIBLE_DEVICES": str(card), "SWEEP_MODE": MODE,
           "PYTHONUNBUFFERED": "1", **extra_env}
    log = open(log_path, "a")           # append: a relaunch must not erase a failed trial's traceback
    log.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')}: {label} =====\n")
    log.flush()
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    print(f"  GPU {card}: {label:<44s} pid {proc.pid}   log {log_path}")
    return card, proc


def describe(t):
    p = t.params
    return (f"{p['n_layers']} layers  branch {2 ** p['branch_log2']:<4d} trunk {2 ** p['trunk_log2']:<5d}"
            f"lr {p['lr']:.1e}  res/data {p['res_over_data']:5.2f}  bcs/data {p['bcs_over_data']:5.3f}")


if __name__ == "__main__":
    TS = optuna.trial.TrialState

    if WORKER and MODE == "search":
        # Keep taking trials from the shared study until it has N_TRIALS finished
        # ones. With constant_liar, TPE treats the other cards' running trials as
        # poor results, so the workers spread out instead of sampling one point.
        study = optuna.load_study(
            study_name=THIS_STUDY, storage=STORAGE,
            sampler=optuna.samplers.TPESampler(n_startup_trials=N_STARTUP,
                                               multivariate=True, constant_liar=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=N_ITER // 5,
                                               interval_steps=LOG_EVERY),
        )
        print(f"Search worker on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,),
                       callbacks=[optuna.study.MaxTrialsCallback(N_TRIALS, states=(TS.COMPLETE, TS.PRUNED))])
        sys.exit(0)

    if WORKER and MODE == "reseed":
        # Re-run the configurations the launcher assigned to this card, each with a
        # new seed and without pruning. PartialFixedSampler fixes every parameter.
        tasks = json.loads(os.environ["SWEEP_TASKS"])
        print(f"Reseed worker on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}: "
              f"{[(t['from_trial'], t['seed']) for t in tasks]}")
        for task in tasks:
            study = optuna.load_study(
                study_name=THIS_STUDY, storage=STORAGE, pruner=optuna.pruners.NopPruner(),
                sampler=optuna.samplers.PartialFixedSampler(task["params"], optuna.samplers.RandomSampler()))

            def run(trial, task=task):
                trial.set_user_attr("from_trial", task["from_trial"])
                return objective(trial, task["seed"])
            study.optimize(run, n_trials=1, catch=(Exception,))
        sys.exit(0)

    # ------------------------------- launcher -------------------------------
    study = optuna.create_study(study_name=THIS_STUDY, storage=STORAGE,
                                direction="minimize", load_if_exists=True)
    finished = [t for t in study.trials if t.state in (TS.COMPLETE, TS.PRUNED)]
    # A checkpoint without any finished trial behind it belongs to an earlier run
    # (for instance after the study was reset) and would silently outrank this one.
    if not finished and os.path.exists(CKPT_PATH):
        sys.exit(f"\n{CKPT_PATH} already exists, but study {THIS_STUDY} has no finished trials:\n"
                 f"those weights come from an earlier run and would silently outrank this one.\n"
                 f"Rename or remove the file (or change sweep_name), then relaunch.")

    os.makedirs(LOG_DIR, exist_ok=True)
    workers = []
    if MODE == "search":
        print(f"\nStudy {THIS_STUDY}: {len(finished)} of {N_TRIALS} trials finished")
        if len(finished) < N_TRIALS:
            for card in CARDS:
                workers.append(launch(card, {}, "architecture search"))
    else:
        main = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
        top  = sorted((t for t in main.trials if t.state == TS.COMPLETE), key=lambda t: t.value)[:TOP_K]
        done = {(t.user_attrs.get("from_trial"), t.user_attrs.get("seed")) for t in finished}
        tasks = [{"params": t.params, "seed": s, "from_trial": t.number}
                 for t in top for s in RESEED_SEEDS if (t.number, s) not in done]
        print(f"\nReseeding the best {len(top)} configurations of {STUDY_NAME}: {len(tasks)} runs to do")
        for i, card in enumerate(CARDS):
            mine = tasks[i::len(CARDS)]         # round-robin over the cards
            if mine:
                workers.append(launch(card, {"SWEEP_TASKS": json.dumps(mine)},
                                      ", ".join(f"trial {t['from_trial']} seed {t['seed']}" for t in mine)))
    for card, proc in workers:
        proc.wait()
        print(f"  GPU {card} worker exited with code {proc.returncode}")

    # ------------------------------- summary --------------------------------
    study = optuna.load_study(study_name=THIS_STUDY, storage=STORAGE)
    count = {s: sum(t.state == s for t in study.trials) for s in (TS.COMPLETE, TS.PRUNED, TS.FAIL, TS.RUNNING)}
    print(f"\n--- {THIS_STUDY}: {count[TS.COMPLETE]} complete, {count[TS.PRUNED]} pruned, "
          f"{count[TS.FAIL]} failed, {count[TS.RUNNING]} marked running ---")
    if MODE == "search":
        ranked = sorted((t for t in study.trials if t.state == TS.COMPLETE), key=lambda t: t.value)
        print(f"  {'trial':>5s}  {'configuration':70s} {'params':>9s} {'score':>6s} {'best':>6s} {'val':>6s} {'shift':>6s}")
        for t in ranked[:10]:
            u = t.user_attrs
            print(f"  {t.number:5d}  {describe(t):70s} {u.get('n_params', 0):9d} {t.value:6.3f} "
                  f"{u.get('best_sel_ARE', float('nan')):6.3f} {u.get('val_ARE', float('nan')):6.3f} "
                  f"{u.get('shift_ARE', float('nan')):6.3f}")
        print("  score = median of the last 10 selection-set ARE readings (%); best / val / shift = ARE of the")
        print("  restored best parameters on the selection set / validation set / shift-validation set")
        if ranked:
            print("\nNext, re-run the best configurations with new seeds:  "
                  "pinn/bin/python angular_optimization.py reseed")
    else:
        main = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
        top  = sorted((t for t in main.trials if t.state == TS.COMPLETE), key=lambda t: t.value)[:TOP_K]
        rows = []
        for t in top:
            runs = [t.value] + [r.value for r in study.trials
                                if r.state == TS.COMPLETE and r.user_attrs.get("from_trial") == t.number]
            rows.append((float(onp.mean(runs)), float(onp.std(runs)), runs, t))
        print("  score over seeds (the search run with seed 1234 plus the reseeds); lower is better")
        for mean, std, runs, t in sorted(rows, key=lambda r: r[0]):
            print(f"  trial {t.number:3d}  {describe(t)}  mean {mean:6.3f} ± {std:5.3f}  "
                  f"over {len(runs)} seeds  {[round(r, 3) for r in runs]}")
    print(f"\nBest weights of this study: {CKPT_PATH}")
