import os
import time

import numpy as onp
import jax
import jax.numpy as jnp
from jax import random

import pickle

from model import (
    DataGenerator,
    build_data_arrays,
    build_val_batch,
    build_bcs_arrays,
    build_res_arrays,
)
from model_angular_scalar import PI_DeepONet_AngularScalar

print(jax.devices())

size = "small"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}
print(f"Loaded datasets/{size}/M_Iso_train.npz")
for k in ds:
    print(f"  {k:<10s} shape={tuple(ds[k].shape)}  dtype={ds[k].dtype}")

E = 2000  # Epochs
B = 1000  # Batch size
D = len(ds["Q"]) * len(ds["x"])   # N*J, matched to angular_training.py
n_iter = int(D * E / B)
log_every = n_iter // 100

X_slab = 10.0
Sigma_t, Sigma_s0, Sigma_s1 = 1.0, 0.5, 0.0
J = int(ds['x'].shape[0])
A = int(ds['mu_GL'].shape[0])

# --- Supervised phi_0 data arrays (scalar flux, normalized by phi_scale) ---
# Same vector-output trunk as PI_DeepONet_Angular (all A angles from one
# forward pass), but the data loss supervises scalar phi_0 via GL
# quadrature (PI_DeepONet_AngularScalar.loss_data), not the raw psi vector.
data_in, data_out, phi_scale = build_data_arrays(ds, normalize=True)
print(f"\nFlux normalization: phi_scale = {phi_scale:.6f}")
print(f"  Network learns psi/phi_scale (vector-output trunk); data loss")
print(f"  supervises phi_0 = GL-quadrature(psi) against true phi_0;")
print(f"  residual uses Q/phi_scale; predict_s un-normalizes.")
print(f"  phi_0-supervision points: {data_out.shape[0]}  (= N*J, each a scalar target)")

# --- Physics collocation sets: identical construction to angular_training.py ---
bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=1000)
res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=1000)

# --- Validation set (phi_0 form; ARE on phi_0, same as the other models) ---
val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}
val_batch = build_val_batch(val_ds, output_scale=phi_scale)
print(f"Loaded validation set: {val_ds['Q'].shape[0]} sources")

data_dataset = DataGenerator(data_in, data_out, batch_size=B,
                             rng_key=random.PRNGKey(101))
bcs_dataset  = DataGenerator(bcs_in,  bcs_out,  batch_size=B,
                             rng_key=random.PRNGKey(202), branch_table=bcs_Q)
res_dataset  = DataGenerator(res_in,  res_out,  batch_size=B,
                             rng_key=random.PRNGKey(303), branch_table=res_Q)

# Exact same architecture as angular_training.py's PI_DeepONet_Angular.
p_latent      = 100
n_layers      = 4
branch_layers = [J] + n_layers * [250] + [p_latent]
trunk_layers  = [1] + n_layers * [500] + [A * p_latent]

model = PI_DeepONet_AngularScalar(
    branch_layers, trunk_layers,
    N_angles=A,
    Sigma_t=Sigma_t, Sigma_s0=Sigma_s0, Sigma_s1=Sigma_s1,
    x_sensors=ds['x'], X=X_slab,
    lambda_data=0.7, lambda_res=0.25, lambda_bcs=0.05,
    lr_transition_steps=n_iter // 10,
    output_scale=phi_scale,
    Q_ref=float(jnp.mean(ds['Q'])),
    activation="relu"   # string name -> saved in config -> reconstructed on load
)
print(f"\nInstantiated PI_DeepONet_AngularScalar  (branch {branch_layers}, trunk {trunk_layers})")

print(f"\n--- Training for {n_iter} iterations ---")
t0 = time.time()
model.train(data_dataset, bcs_dataset, res_dataset,
            nIter=n_iter, log_every=log_every,
            val_batch=val_batch, val_every=log_every)
dt = time.time() - t0
print(f"Training time: {dt:.1f} s  ({dt / n_iter * 1000:.1f} ms/iter)")

# ------------------------------------------------------------------------
# Save-time consistency guard (same rationale as angular_training.py).
# ------------------------------------------------------------------------
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

os.makedirs("trained_models/" + size, exist_ok=True)
out_path = "trained_models/" + size + "/pideeponet_angular_scalar_data_" + model.activation_name + ".pkl"
with open(out_path, "wb") as f:
    pickle.dump({
        "params": model.params,
        "config": {
            "model_type":    "angular_vec_scalar_data",
            "activation":    model.activation_name,
            "branch_layers": branch_layers,
            "trunk_layers":  trunk_layers,
            "N_angles":      A,
            "Sigma_t":       Sigma_t,
            "Sigma_s0":      Sigma_s0,
            "Sigma_s1":      Sigma_s1,
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
        "best_val_ARE":  model.best_val_ARE,
        "best_val_iter": model.best_val_iter,
        "n_iter": n_iter,
        "log_every": log_every,
    }, f)
print("Saved " + out_path)
