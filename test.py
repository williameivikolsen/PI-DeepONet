import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from nonPI_model import DeepONet
import jax.numpy as jnp
from nonPI_model import build_data_arrays

with open("./trained_models/sahadath.pkl", "rb") as f:
    ckpt = pickle.load(f)
cfg = ckpt["config"]
cfg["x_sensors"] = jnp.asarray(cfg["x_sensors"])

model = DeepONet(**cfg)
model.params        = ckpt["params"]
model.loss_log      = ckpt["loss_log"]

# Reload the same training dataset and re-normalize it the same way
ds_np = np.load("datasets/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

data_in, data_out, phi_scale = build_data_arrays(ds, normalize=True)

# Sanity check: phi_scale should match what was saved in the checkpoint
assert abs(phi_scale - cfg["output_scale"]) < 1e-5, \
    f"phi_scale mismatch: dataset={phi_scale}, checkpoint={cfg['output_scale']}"

# Full-dataset loss (in normalized space, comparable to training-time loss)
full_loss = float(model.loss(model.params, (data_in, data_out)))

# Full-dataset ARE on raw fluxes (scale-invariant, but compute on raw for clarity)
Q_full, x_full = data_in
pred_norm = jnp.concatenate([
    model.predict_s(model.params, Q_full[i:i+10000], x_full[i:i+10000]) / model.output_scale
    for i in range(0, Q_full.shape[0], 10000)
])
are = float(jnp.mean(jnp.abs((data_out - pred_norm) / data_out)) * 100)

print(f"Full-dataset loss (normalized): {full_loss:.3e}")
print(f"Full-dataset ARE:               {are:.3f}%")
print(f"Last logged batch loss:         {model.loss_log[-1]:.3e}")