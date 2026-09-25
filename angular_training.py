import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time

import numpy as onp
import jax
import jax.numpy as jnp
from jax import random
import optax

import pickle

from model import (
    PI_DeepONet_Angular,
    DataGenerator,
    build_psi_data_arrays,
    build_psi_val_batch,
    build_bcs_arrays,
    build_res_arrays,
)

print(jax.devices())

size = "large"

ds_np = onp.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}
print(f"Loaded datasets/{size}/M_Iso_train.npz")
for k in ds:
    print(f"  {k:<10s} shape={tuple(ds[k].shape)}  dtype={ds[k].dtype}")

B = 5000  # Batch size
n_iter = 300_000
log_every = n_iter // 100

X_slab = 10.0
Sigma_t, Sigma_s0, Sigma_s1 = 1.0, 0.5, 0.0
J = int(ds['x'].shape[0])
A = int(ds['mu_GL'].shape[0])

data_in, data_out = build_psi_data_arrays(ds)
Q_shift, Q_scale = 0.0, 1.0
print(f"Branch input: (Q - {Q_shift:.6f}) / {Q_scale:.6f}")
print(f"\npsi-supervision points: {data_out.shape[0]}  (= N*J, each carrying an A-vector target)")

bcs_in, bcs_out, bcs_Q = build_bcs_arrays(ds, X=X_slab, n_per_sample=1000)
res_in, res_out, res_Q = build_res_arrays(ds, X=X_slab, n_per_sample=1000)

val_np = onp.load("datasets/M_Iso_val.npz")
val_ds = {k: jnp.asarray(val_np[k]) for k in val_np.files}
val_batch = build_psi_val_batch(val_ds)
print(f"Loaded validation set: {val_ds['Q'].shape[0]} sources")

data_dataset = DataGenerator(data_in, data_out, batch_size=B,
                             rng_key=random.PRNGKey(101))
bcs_dataset  = DataGenerator(bcs_in,  bcs_out,  batch_size=B,
                             rng_key=random.PRNGKey(202), branch_table=bcs_Q)
res_dataset  = DataGenerator(res_in,  res_out,  batch_size=B,
                             rng_key=random.PRNGKey(303), branch_table=res_Q)

# Loss weighting: "none" (fixed lambdas only), "local_ntk" (a weight per point,
# Algorithm 1 of Wang, Wang & Perdikaris 2022) or "global_ntk" (a weight per
# loss term). ntk_alpha = 1 gives "NTK weights", 0.5 "moderate NTK weights".
# The NTK weights replace the fixed lambdas, as in the paper.
weighting      = "none"
ntk_alpha      = 1.0
ntk_chunk_size = 100    # points per NTK Jacobian chunk (bounds memory); None = whole batch

p_latent      = 100
n_layers      = 5
branch_layers = [J] + n_layers * [128] + [p_latent]
trunk_layers  = [1] + n_layers * [256] + [A * p_latent]

# Same warmup + cosine decay as nodata_optimization.py
lr_peak      = 1.2e-4
warmup_steps = 2000
end_fraction = 0.01
lr_schedule  = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=lr_peak, warmup_steps=warmup_steps,
    decay_steps=n_iter, end_value=lr_peak * end_fraction)
lr_config    = f"warmup_cosine_{lr_peak:.1e}_to_{lr_peak * end_fraction:.1e}"

seed = 123

model = PI_DeepONet_Angular(
    branch_layers, trunk_layers,
    N_angles=A,
    Sigma_t=Sigma_t, Sigma_s0=Sigma_s0, Sigma_s1=Sigma_s1,
    x_sensors=ds['x'], X=X_slab, Q_shift=Q_shift, Q_scale=Q_scale,
    lambda_data=0.7, lambda_res=0.25, lambda_bcs=0.05,
    # lambda_data=1.0, lambda_res=1.0, lambda_bcs=1.0, # NTK weighting
    weighting=weighting, ntk_alpha=ntk_alpha, ntk_chunk_size=ntk_chunk_size,
    lr_schedule=lr_schedule,
    branch_activation="relu",   # unbounded -> extrapolates in source amplitude
    trunk_activation="tanh",    # names are saved in config and reconstructed on load
    seed=seed,
)
print(f"Learning rate: {lr_config}")
print(f"\nInstantiated PI_DeepONet_Angular  (branch {branch_layers}, trunk {trunk_layers})")

print(f"\n--- Training for {n_iter} iterations ---")
t0 = time.time()
model.train(data_dataset, bcs_dataset, res_dataset,
            nIter=n_iter, log_every=log_every,
            val_batch=val_batch, val_every=log_every)
dt = time.time() - t0
print(f"Training time: {dt:.1f} s  ({dt / n_iter * 1000:.1f} ms/iter)")

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

os.makedirs("trained_models/training_testing/ntk", exist_ok=True)
out_path = (f"trained_models/training_testing/ntk/pideeponet_angular_"
            f"{model.branch_activation_name}_{model.trunk_activation_name}_seed{seed}.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "params": model.params,
        "config": {
            "model_type":    "angular_vec",
            "branch_activation": model.branch_activation_name,
            "trunk_activation":  model.trunk_activation_name,
            "branch_layers": branch_layers,
            "trunk_layers":  trunk_layers,
            "N_angles":      A,
            "Sigma_t":       Sigma_t,
            "Sigma_s0":      Sigma_s0,
            "Sigma_s1":      Sigma_s1,
            "x_sensors":     onp.asarray(ds['x']),
            "X":             X_slab,
            "Q_shift":       Q_shift,
            "Q_scale":       Q_scale,
            "seed":          seed,
            "weighting":     weighting,
            "ntk_alpha":     ntk_alpha,
        },
        "lr_config":     lr_config,
        "loss_log":      model.loss_log,
        "loss_data_log": model.loss_data_log,
        "loss_bcs_log":  model.loss_bcs_log,
        "loss_res_log":  model.loss_res_log,
        "lam_data_log":  model.lam_data_log,
        "lam_bcs_log":   model.lam_bcs_log,
        "lam_res_log":   model.lam_res_log,
        "val_ARE_log":   model.val_ARE_log,
        "val_iter_log":  model.val_iter_log,
        "best_val_ARE":  model.best_val_ARE,
        "best_val_iter": model.best_val_iter,
        "n_iter": n_iter,
        "log_every": log_every,
    }, f)
print("Saved " + out_path)