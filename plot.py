import pickle
import numpy as onp
import jax.numpy as jnp
from model import PI_DeepONet
from helpers import plot_loss_curves, plot_sample_predictions
import matplotlib.pyplot as plt

with open("trained_models/data_only_4_angles.pkl", "rb") as f:
    ckpt = pickle.load(f)

cfg = ckpt["config"]
cfg["x_sensors"] = jnp.asarray(cfg["x_sensors"])

model = PI_DeepONet(**cfg)
model.params        = ckpt["params"]
model.loss_log      = ckpt["loss_log"]
model.loss_data_log = ckpt["loss_data_log"]
model.loss_bcs_log  = ckpt["loss_bcs_log"]
model.loss_res_log  = ckpt["loss_res_log"]

fig1 = plot_loss_curves(
    model,
    log_every=ckpt["log_every"],
)

# Plot sample predictions

ds_np = onp.load("datasets/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}

sample_indices = [0, 42, 123, 314]

fig2 = plot_sample_predictions(
    model,
    ds,
    sample_indices=sample_indices,
)

plt.show()