import os
import time

import numpy as onp
import jax
import jax.numpy as jnp
from jax import random

import pickle

from nonPI_model import (
    DeepONet,
    DataGenerator,
    build_data_arrays,
    build_val_batch,
)

print(jax.devices())

size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}
for k in ds:
    print(f"  {k:<10s} shape={tuple(ds[k].shape)}  dtype={ds[k].dtype}")

E = 2000 # Epochs
B = 1000 # Batch size
D = len(ds["Q"])*len(ds["x"]) # Number of points in dataset
n_iter = int(D*E/B)
log_every = n_iter//100

X_slab = 10.0
Sigma_t, Sigma_s0, Sigma_s1 = 1.0, 0.5, 0.0
J = int(ds['x'].shape[0])

data_in, data_out = build_data_arrays(ds)
# Branch input transform: (Q - Q_shift) / Q_scale. Constants come from the
# TRAINING SET as a whole — never from the sample being evaluated. Identity
# while the branch is relu: relu absorbs a scale factor exactly, and a shift
# would destroy the amplitude extrapolation the relu branch is there to give.
Q_shift, Q_scale = 0.0, 1.0
# Q_shift, Q_scale = 0.0, float(jnp.sqrt(jnp.mean(ds['Q'] ** 2)))   # for a bounded branch activation
print(f"Branch input: (Q - {Q_shift:.6f}) / {Q_scale:.6f}")

data_dataset = DataGenerator(data_in, data_out, batch_size=B,
                             rng_key=random.PRNGKey(101))

val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}
val_batch = build_val_batch(val_ds)
print(f"Loaded validation set: {val_ds['Q'].shape[0]} sources")

# branch_layers = [J, 250, 250, 250, 250, 100]
# trunk_layers  = [1, 500, 500, 500, 500, 100]
branch_layers = [100, 200, 200, 100]
trunk_layers = [1, 200, 200, 100]
model_name = "benchmark"

lr_config    = "const_3e-4"
lr_schedule  = 3e-4

model = DeepONet(
    branch_layers, trunk_layers,
    Sigma_t=Sigma_t, Sigma_s0=Sigma_s0, Sigma_s1=Sigma_s1,
    x_sensors=ds['x'], X=X_slab, Q_shift=Q_shift, Q_scale=Q_scale,
    lambda_data=1.0, lambda_res=1.0, lambda_bcs=1.0,
    lr_schedule=lr_schedule,
)
print(f"Learning rate: {lr_config}")
print(f"\nInstantiated DeepONet  (branch {branch_layers}, trunk {trunk_layers})")

print(f"\n--- Training for {n_iter} iterations ---")
t0 = time.time()
model.train(data_dataset,
            nIter=n_iter, log_every=log_every,
            val_batch=val_batch, val_every=log_every)
dt = time.time() - t0
print(f"Training time: {dt:.1f} s  ({dt / n_iter * 1000:.1f} ms/iter)")

# Save-time consistency guard
GUARD_TOL = 1.0  # percentage points

_are_valpath = float(model.val_ARE(model.params, val_batch))

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

os.makedirs("trained_models/training_testing/" + size, exist_ok=True)
out_path = (f"trained_models/training_testing/{size}/{model_name}_"
            f"{model.branch_activation_name}_{model.trunk_activation_name}.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "params": model.params,
        "config": {
            "branch_activation": model.branch_activation_name,
            "trunk_activation":  model.trunk_activation_name,
            "branch_layers": branch_layers,
            "trunk_layers":  trunk_layers,
            "Sigma_t":       Sigma_t,
            "Sigma_s0":      Sigma_s0,
            "Sigma_s1":      Sigma_s1,
            "x_sensors":     onp.asarray(ds['x']),
            "X":             X_slab,
            "Q_shift":       Q_shift,
            "Q_scale":       Q_scale,
        },
        "loss_log":      model.loss_log,
        "val_ARE_log":   model.val_ARE_log,
        "val_iter_log":  model.val_iter_log,
        "best_val_ARE":  model.best_val_ARE,
        "best_val_iter": model.best_val_iter,
        "n_iter": n_iter,
        "log_every": log_every,
        "lr_config": lr_config,
    }, f)
print("Saved " + out_path)
