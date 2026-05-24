import pickle
import numpy as onp
import jax.numpy as jnp
from model import PI_DeepONet
from helpers import plot_loss_curves, plot_sample_predictions
import matplotlib.pyplot as plt

with open("trained_models/optimal_model.pkl", "rb") as f:
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

ds_np = onp.load("datasets/test/M_Iso_test_NS.npz")
ds = {k: jnp.asarray(ds_np[k]) for k in ds_np.files if ds_np[k].dtype.kind in "fiu"}
sample_index = 0

fig2 = plot_sample_predictions(
    model,
    ds,
    sample_index=sample_index,
)

plt.show()