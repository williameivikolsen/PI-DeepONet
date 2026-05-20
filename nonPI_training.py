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
)

print(jax.devices())

E = 2000 # Epochs
B = 1000 # Batch size
D = 50000 # Number of points in dataset
n_iter = int(D*E/B)
log_every = 100

ds_np = onp.load("datasets/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}
print(f"Loaded datasets/M_Iso_train.npz")
for k in ds:
    print(f"  {k:<10s} shape={tuple(ds[k].shape)}  dtype={ds[k].dtype}")

X_slab = 10.0
Sigma_t, Sigma_s0, Sigma_s1 = 1.0, 0.5, 0.0
J = int(ds['x'].shape[0])

data_in, data_out = build_data_arrays(ds)

data_dataset = DataGenerator(data_in, data_out, batch_size=B,
                             rng_key=random.PRNGKey(101))

branch_layers = [J, 200, 200, 100]
trunk_layers  = [1, 200, 200, 100]

model = DeepONet(
    branch_layers, trunk_layers,
    Sigma_t=Sigma_t, Sigma_s0=Sigma_s0, Sigma_s1=Sigma_s1,
    x_sensors=ds['x'], X=X_slab,
    lambda_data=1.0, lambda_res=1.0, lambda_bcs=1.0,
    lr_transition_steps=n_iter//5
)
print(f"\nInstantiated PI_DeepONet  (branch {branch_layers}, trunk {trunk_layers})")

print(f"\n--- Training for {n_iter} iterations ---")
t0 = time.time()
model.train(data_dataset,
            nIter=n_iter, log_every=log_every)
dt = time.time() - t0
print(f"Training time: {dt:.1f} s  ({dt / n_iter * 1000:.1f} ms/iter)")

os.makedirs("trained_models", exist_ok=True)
with open("trained_models/sahadath.pkl", "wb") as f:
    pickle.dump({
        "params": model.params,
        "config": {
            "branch_layers": branch_layers,
            "trunk_layers":  trunk_layers,
            "Sigma_t":       Sigma_t,
            "Sigma_s0":      Sigma_s0,
            "Sigma_s1":      Sigma_s1,
            "x_sensors":     onp.asarray(ds['x']),
            "X":             X_slab,
        },
        "loss_log":      model.loss_log,
        "n_iter": n_iter,
        "log_every": log_every,
    }, f)
print("Saved trained_models/sahadath.pkl")