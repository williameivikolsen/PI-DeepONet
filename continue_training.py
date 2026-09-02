import os
import time

import numpy as onp
import jax
import jax.numpy as jnp
from jax import random
import optax

import pickle

from helpers import load_model
from model import (
    DataGenerator,
    build_psi_data_arrays,
    build_psi_val_batch,
    build_bcs_arrays,
    build_res_arrays,
)

print(jax.devices())

CHECKPOINT = "trained_models/lr_search/relu_gelu/pideeponet_angular.pkl"
OUT_PATH   = "trained_models/training_testing/large/pideeponet_angular_relu_gelu_continued.pkl"

# The checkpoint config does NOT record the learning rate, so load_model rebuilds
# the model with the constructor default (constant 1e-3). Set it explicitly to
# the rate that produced this checkpoint, or the continuation silently runs 10x
# too fast.
lr_config    = "const_1e-4"
lr_schedule  = 1e-4

E = 2000    # additional epochs
B = 1000    # batch size

size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

D = len(ds["Q"]) * len(ds["x"])
n_iter = int(D * E / B)
log_every = n_iter // 100

X_slab = 10.0
J = int(ds['x'].shape[0])
A = int(ds['mu_GL'].shape[0])

model, kind, ckpt = load_model(CHECKPOINT)
prev_n_iter = ckpt["n_iter"]
prev_best   = ckpt["best_val_ARE"]
print(f"\nLoaded {CHECKPOINT}")
print(f"  kind={kind}  branch/trunk = "
      f"{model.branch_activation_name}/{model.trunk_activation_name}")
print(f"  previous leg: {prev_n_iter} iters, best val ARE {prev_best:.4f}% "
      f"at iter {ckpt['best_val_iter']}")
print(f"  the loaded weights are that BEST iterate, not the final one")

# Adam moments are not stored in the checkpoint, so this is a warm restart, not
# a true resume: the optimiser state begins at zero and bias correction starts
# over. Expect a transient in the first few hundred iterations.
model.optimizer   = optax.adam(learning_rate=lr_schedule)
model.opt_state   = model.optimizer.init(model.params)
model.lr_schedule = lr_schedule
print(f"  optimiser rebuilt: adam at {lr_config} (Adam moments restart from zero)")

data_in, data_out = build_psi_data_arrays(ds)
bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=1000)
res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=1000)

val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}
val_batch = build_psi_val_batch(val_ds)
print(f"Loaded validation set: {val_ds['Q'].shape[0]} sources")

# Fresh generator keys, so the continuation draws new batches rather than
# replaying the exact sequence the model already trained on.
data_dataset = DataGenerator(data_in, data_out, batch_size=B,
                             rng_key=random.PRNGKey(111))
bcs_dataset  = DataGenerator(bcs_in,  bcs_out,  batch_size=B,
                             rng_key=random.PRNGKey(212), branch_table=bcs_Q)
res_dataset  = DataGenerator(res_in,  res_out,  batch_size=B,
                             rng_key=random.PRNGKey(313), branch_table=res_Q)

print(f"\n--- Continuing for {n_iter} iterations ({E} epochs) ---")
t0 = time.time()
model.train(data_dataset, bcs_dataset, res_dataset,
            nIter=n_iter, log_every=log_every,
            val_batch=val_batch, val_every=log_every)
dt = time.time() - t0
print(f"Training time: {dt:.1f} s  ({dt / n_iter * 1000:.1f} ms/iter)")

print(f"\n  previous best val ARE = {prev_best:.4f}%")
print(f"  this leg's best       = {model.best_val_ARE:.4f}%  "
      f"({'improved' if model.best_val_ARE < prev_best else 'NO improvement'})")

# ------------------------------------------------------------------------
# Save-time consistency guard.
# ------------------------------------------------------------------------
GUARD_TOL = 1.0  # percentage points

_are_valpath = float(model.val_ARE(model.params, val_batch))

# predict_phi0 path on the validation data.
_phi_pred = onp.asarray(model.predict_phi0(model.params,
                                           jnp.asarray(val_ds['Q']),
                                           jnp.asarray(val_ds['x'])))
_phi_true = onp.asarray(val_ds['phi_0'])
_are_predpath = float(onp.mean(onp.abs((_phi_true - _phi_pred) / _phi_true)) * 100.0)

print("\n=== save-time consistency guard ===")
print(f"  recorded best_val_ARE        = {model.best_val_ARE:.4f}%")
print(f"  val_ARE(restored params)     = {_are_valpath:.4f}%")
print(f"  predict_phi0 path ARE        = {_are_predpath:.4f}%")
print(f"  predict_phi0 pred range      = {_phi_pred.min():.2f} .. {_phi_pred.max():.2f}"
      f"   (true {_phi_true.min():.2f} .. {_phi_true.max():.2f})")

_fail = []
if abs(_are_valpath - model.best_val_ARE) > GUARD_TOL:
    _fail.append(
        f"val_ARE of restored params ({_are_valpath:.3f}%) != recorded "
        f"best_val_ARE ({model.best_val_ARE:.3f}%): best-params restoration failed."
    )
if abs(_are_predpath - model.best_val_ARE) > GUARD_TOL:
    _fail.append(
        f"predict_phi0 ARE ({_are_predpath:.3f}%) != recorded best_val_ARE "
        f"({model.best_val_ARE:.3f}%): the evaluation path disagrees with the "
        f"training metric — the checkpoint's headline number would be wrong."
    )
if _phi_pred.min() < 0.0:
    _fail.append(
        f"predict_phi0 produced negative phi_0 (min {_phi_pred.min():.2f}); "
        f"scalar flux must be non-negative."
    )

if _fail:
    print("  GUARD FAILED — checkpoint NOT saved:")
    for msg in _fail:
        print("    - " + msg)
    raise SystemExit(
        "Aborting save: restored params do not reproduce best_val_ARE under "
        "the evaluation path. See guard messages above."
    )
print("  guard passed: restored params reproduce best_val_ARE on both paths.\n")

# train() appends to the loss logs (load_model restored them) but resets the
# validation logs, so stitch those back together with the iteration numbers
# continuing from where the previous leg stopped.
val_ARE_log  = list(ckpt["val_ARE_log"])  + list(model.val_ARE_log)
val_iter_log = list(ckpt["val_iter_log"]) + [prev_n_iter + i for i in model.val_iter_log]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump({
        "params": model.params,
        "config": dict(ckpt["config"]),
        "lr_config":     lr_config,
        "loss_log":      model.loss_log,
        "loss_data_log": model.loss_data_log,
        "loss_bcs_log":  model.loss_bcs_log,
        "loss_res_log":  model.loss_res_log,
        "val_ARE_log":   val_ARE_log,
        "val_iter_log":  val_iter_log,
        "best_val_ARE":  model.best_val_ARE,
        "best_val_iter": prev_n_iter + model.best_val_iter,
        "n_iter":        prev_n_iter + n_iter,
        "log_every":     log_every,
        "continued_from":     CHECKPOINT,
        "continued_at_iter":  prev_n_iter,
        "prev_best_val_ARE":  prev_best,
    }, f)
print("Saved " + OUT_PATH)
